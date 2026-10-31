"""Durable, fail-closed full-population M0 + nine daily updates and exact-window audits."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .batches import selection_key
from .configuration import DEFAULT_CONFIG, load_proposal
from .data_common import json_save, sha256
from .gates import choose_foundation, early_learning, native_learning
from .protocol import day_date, version_schedule
from .runtime import source_binding

REPO = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
PACKAGE = "experiments.recflow_full_population.code."
FOUNDATION_SECONDS = 15*3600


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def pid_ticks(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(") ",1)[1].split()[19]


def checkpoints(directory):
    result=[]
    for path in sorted(Path(directory).glob("step_*.pt")):
        metadata=read(path.with_suffix(".json"))
        if metadata["status"]!="committed":
            raise ValueError("Uncommitted checkpoint")
        result.append((path,metadata))
    return result


class Chain:
    def __init__(self, output, data, readiness):
        self.output, self.data, self.readiness = output.resolve(), data.resolve(), readiness.resolve()
        self.started=time.monotonic()
        self.foundation_start=None
        self.child=None
        self.serial=0
        self.versions=[]
        self.env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                      PYTHONPATH=str(REPO/"src")+os.pathsep+str(REPO), PYTHONUNBUFFERED="1",
                      EVOKV_ATTENTION_BACKEND="auto")

    def bind(self):
        result=source_binding(self.data,"vocabulary_max")
        paths=list((BASE/"code").glob("*.py"))+[DEFAULT_CONFIG, self.readiness]
        paths += list((self.data/"selections").glob("*.npy"))+list((self.data/"vocabulary_max").glob("*.npy"))
        paths += [self.data/name for name in ["population.npz","preparation_audit.json"]]
        for path in paths:
            result[str(path.resolve())]=sha256(path)
        return result

    def verify(self):
        if shutil.disk_usage(self.output).free < 32*1024**3:
            raise RuntimeError("Fewer than 32 GiB remain; preserve existing artifacts and stop before next stage")

    def run(self, name, module, arguments, *, gpu=True, timeout=4*3600):
        self.verify()
        self.serial+=1
        command=[sys.executable]
        if gpu:
            command += ["-m","torch.distributed.run","--standalone","--nproc_per_node=4"]
        command += ["-m",PACKAGE+module]+[str(value) for value in arguments]
        label=f"{self.serial:02d}_{name}"
        log=self.output/"logs"/f"{label}.log"
        begun=time.monotonic()
        record=dict(stage=name,command=command,started_at=now(),log=str(log),status="running")
        with log.open("x") as handle:
            self.child=subprocess.Popen(command,cwd=REPO,env=self.env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            record.update(pid=self.child.pid,pid_start_ticks=pid_ticks(self.child.pid))
            json_save(self.output/"active_process.json",record)
            json_save(self.output/"stages"/f"{label}.json",record)
            print(json.dumps(record),flush=True)
            while True:
                try:
                    code=self.child.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    json_save(self.output/"progress.json",dict(status="running",stage=name,pid=self.child.pid,
                        updated_at=now(),elapsed_seconds=time.monotonic()-self.started,completed_models=len(self.versions),
                        foundation_elapsed_seconds=self.foundation_elapsed()))
                    if time.monotonic()-begun>timeout:
                        raise TimeoutError(f"Stage deadline exceeded: {name}")
        record.update(status="completed" if code==0 else "failed",returncode=code,
                      finished_at=now(),wall_seconds=time.monotonic()-begun)
        json_save(self.output/"stages"/f"{label}.json",record)
        json_save(self.output/"active_process.json",record)
        self.child=None
        print(json.dumps(record),flush=True)
        if code:
            raise RuntimeError(f"Stage {name} failed with exit {code}; no automatic retry")

    def foundation_elapsed(self):
        return time.monotonic()-self.foundation_start if self.foundation_start is not None else 0.

    def train_args(self, directory, phase, days, epochs, checkpoint_every, *, parent=None, resume=None, stop=None):
        args=["--data",self.data,"--config",DEFAULT_CONFIG,"--output",directory,
              "--phase",phase,"--scale","max","--days",*days,"--epochs",epochs,
              "--global-batch",64,"--micro-batch",16,"--workers",4,"--listwise-weight",.1,
              "--dtype","bfloat16","--communication","rows","--checkpoint-every",checkpoint_every,"--log-every",100,
              "--wall-hours",2 if phase=="update" else 15,
              "--budget-spent-seconds",0 if phase=="update" else self.foundation_elapsed()]
        if parent:
            args += ["--parent",parent]
        if resume:
            args += ["--resume",resume]
        if stop is not None:
            args += ["--stop-after-steps",stop]
        return args

    def evaluate(self, name, *, checkpoint=None, parent=None, day=None, initialize=False):
        directory=self.output/"evaluations"/name
        role=1 if day is None else 2
        args=["--data",self.data,"--config",DEFAULT_CONFIG,"--output",directory,"--role",role,
              "--days",*([24,25,26,27] if day is None else [day]),"--batch-size",16,"--workers",2]
        args += ["--initialize"] if initialize else ["--checkpoint",checkpoint]
        if parent:
            args += ["--parent",parent]
        self.run(name,"evaluate",args,timeout=3*3600)
        self.run(name+"_audit","audit_scored_requests",["--evaluation",directory,"--data",self.data,
                    "--output",directory/"independent_audit.json"],gpu=False,timeout=3600)
        if read(directory/"independent_audit.json")["status"]!="passed":
            raise ValueError("Independent complete-candidate audit failed")
        return directory,read(directory/"summary.json")

    def append_model(self, version, checkpoint, parent, evaluation, summary, train_days):
        side=read(checkpoint.with_suffix(".json"))
        if side.get("status", "committed") != "committed" or not side["metadata"]["quality_use_permitted"]:
            raise ValueError("Release checkpoint is incomplete or not permitted for quality")
        actual_parent = side["metadata"].get("arguments", {}).get("parent")
        if parent and (not actual_parent or Path(actual_parent).resolve() != parent.resolve()):
            raise ValueError("Actual training parent changed")
        if summary.get("checkpoint") and Path(summary["checkpoint"]).resolve() != checkpoint.resolve():
            raise ValueError("Evaluated checkpoint differs from release")
        self.versions.append(dict(version=version,checkpoint=str(checkpoint),checkpoint_sha256=side["checkpoint_sha256"],
            parent_checkpoint=str(parent) if parent else None,
            parent_sha256=read(parent.with_suffix(".json"))["checkpoint_sha256"] if parent else None,
            training_days=train_days,evaluation_days=summary["days"],evaluation=str(evaluation),metrics=summary["metrics"],
            paired_current_minus=summary.get("paired_current_minus",{}),requests=summary["requests"],users=summary["users"],
            positive_requests=summary["positive_requests"],positive_users=summary["positive_users"]))
        json_save(self.output/"model_chain_inventory.json",dict(status="in_progress",versions=self.versions))
        rows=[]
        for item in self.versions:
            metrics=item["metrics"]
            rebuild=metrics["current"]["ndcg@10"]
            reuse=metrics.get("reuse",{}).get("ndcg@10")
            interval=item["paired_current_minus"].get("reuse",{}).get("ci95",[None,None])
            day=item["evaluation_days"][0]
            rows.append(dict(version=item["version"],train_start_day=item["training_days"][0],
                train_end_day=item["training_days"][-1],eval_day=day,eval_date=day_date(day),
                requests=item["requests"],users=item["users"],positive_requests=item["positive_requests"],
                positive_users=item["positive_users"],rebuild_ndcg10=rebuild,reuse_ndcg10=reuse,
                rebuild_minus_reuse=(rebuild-reuse) if reuse is not None else None,
                relative_loss_pct=100*(rebuild-reuse)/rebuild if reuse is not None and rebuild else None,
                ci95_low=interval[0],ci95_high=interval[1],
                rebuild_recall10=metrics["current"]["recall@10"],reuse_recall10=metrics.get("reuse",{}).get("recall@10")))
        temporary=self.output/"version_metrics.csv.partial"
        with temporary.open("w") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary,self.output/"version_metrics.csv")

    def execute(self):
        if self.output.exists():
            raise ValueError("Use a new run directory; existing results are immutable")
        readiness=read(self.readiness)
        if readiness["status"]!="passed":
            raise ValueError("Missing CPU/actual-Max/four-GPU readiness")
        if shutil.disk_usage(self.output.parent).free < 300*1024**3:
            raise RuntimeError("Insufficient headroom for retaining all checkpoints and score arrays")
        for name in ["logs","stages","training","evaluations","gates","source_snapshot"]:
            (self.output/name).mkdir(parents=True,exist_ok=False)
        self.binding=self.bind()
        for path in (BASE/"code").glob("*.py"):
            shutil.copy2(path,self.output/"source_snapshot"/path.name)
        shutil.copy2(DEFAULT_CONFIG,self.output/"source_snapshot/full.json")
        self.manifest=read(self.data/"selection_manifest.json")["datasets"]
        json_save(self.output/"plan.json",dict(status="frozen_before_new_quality",created_at=now(),
            coordinator_pid=os.getpid(),coordinator_start_ticks=pid_ticks(os.getpid()),
            sources_and_inputs=self.binding,proposal=load_proposal(),version_schedule=version_schedule(),
            total_versions=10,total_edges=9,physical_gpus=[0,1,2,3],training_population=30000,history_cap=1024,
            early_check_step=4096,validation_requests=16384,foundation_budget_seconds=FOUNDATION_SECONDS,
            alignment_max_epochs=2,selection="highest heldout native NDCG10 among completed alignment epochs; earliest tie",
            learning_gates=["early paired heldout BCE improvement over identical initialization", "selected native NDCG10 above both random and foundation-only popularity"],
            no_cache_gap_selection=True,retain_all_signs=True,estimated_total_hours=[22,29.5],
            evaluation_semantics="same fixed training users; next complete day; causal matched-prefix snapshot",
            prediction_artifacts="all complete candidate FP32 scores and independently audited request metrics"))
        self.foundation_start=time.monotonic()
        initial,_=self.evaluate("untrained_validation",initialize=True)
        warm=self.output/"training/warmup"
        warm_days=list(range(1,24))
        self.run("warmup_early","train",self.train_args(warm,"warmup",warm_days,1,16384,stop=4096),timeout=15*3600)
        early=warm/"step_004096.pt"
        if not early.exists():
            raise ValueError("Warmup did not reach the fixed early learning check")
        early_eval,_=self.evaluate("early_validation",checkpoint=early)
        gate=early_learning(initial,early_eval)
        json_save(self.output/"gates/early_learning.json",gate)
        if not gate["passed"]:
            raise ValueError("Early validation has not established learning; preserve results and halt the chain")
        self.run("warmup_full","train",self.train_args(warm,"warmup",warm_days,1,16384,resume=early),timeout=15*3600)
        warm_complete=read(warm/"completion.json")
        if warm_complete["status"]!="completed" or warm_complete["next_epoch"]!=1:
            raise ValueError("Warmup did not complete one full epoch inside foundation budget")
        warm_final=warm/read(warm/"latest.json")["checkpoint"]
        align=self.output/"training/alignment"
        epoch_steps=math.ceil(self.manifest[selection_key(0,[24,25,26,27])]["requests"]/64)
        self.run("alignment_epoch1","train",self.train_args(align,"alignment",[24,25,26,27],2,epoch_steps,
            parent=warm_final,stop=epoch_steps),timeout=15*3600)
        first=align/f"step_{epoch_steps:06d}.pt"
        if not first.exists() or read(first.with_suffix(".json"))["metadata"]["next_epoch"]!=1:
            raise ValueError("No complete first alignment epoch; never promote a partial model")
        val1,summary1=self.evaluate("alignment_epoch1_validation",checkpoint=first)
        candidates=[dict(epoch=1,checkpoint=str(first),evaluation=str(val1),summary=summary1)]
        progress=read(align/"progress.json")
        validation_seconds=sum(read(p)["wall_seconds"] for p in (self.output/"stages").glob("*alignment_epoch1_validation*.json"))
        projected=epoch_steps*progress["seconds_per_step"]*1.15+validation_seconds*1.25+300
        remaining=FOUNDATION_SECONDS-self.foundation_elapsed()
        admission=dict(foundation_remaining_seconds=remaining,projected_epoch2_with_validation_seconds=projected,
                       execute_second_epoch=remaining>=projected,uses_cache_gap=False)
        json_save(self.output/"gates/second_epoch_budget.json",admission)
        if admission["execute_second_epoch"]:
            self.run("alignment_epoch2","train",self.train_args(align,"alignment",[24,25,26,27],2,epoch_steps,resume=first),timeout=15*3600)
            final=align/f"step_{2*epoch_steps:06d}.pt"
            if final.exists() and read(final.with_suffix(".json"))["metadata"]["next_epoch"]==2:
                val2,summary2=self.evaluate("alignment_epoch2_validation",checkpoint=final)
                candidates.append(dict(epoch=2,checkpoint=str(final),evaluation=str(val2),summary=summary2))
            else:
                json_save(self.output/"gates/incomplete_second_epoch.json",dict(status="retained_not_promoted",completion=read(align/"completion.json")))
        selected=choose_foundation(candidates)
        gate=native_learning(selected["summary"])
        gate.update(selected_epoch=selected["epoch"],selected_checkpoint=selected["checkpoint"],
                    candidates=candidates,foundation_elapsed_seconds=self.foundation_elapsed())
        json_save(self.output/"gates/foundation_learning.json",gate)
        if self.foundation_elapsed()>FOUNDATION_SECONDS:
            raise ValueError("Foundation exceeded the 15-hour budget; do not dispatch successors")
        if not gate["passed"]:
            raise ValueError("Foundation native ranking did not pass the two fixed baseline checks; preserve all results")
        parent=Path(selected["checkpoint"])
        evaluation,summary=self.evaluate("m0_d28",checkpoint=parent,day=28)
        self.append_model(0,parent,None,evaluation,summary,list(range(1,28)))
        for version,day in enumerate(range(28,37),1):
            directory=self.output/"training"/f"m{version}_d{day}"
            self.run(f"m{version}_training_d{day}","train",self.train_args(directory,"update",[day],1,100000,parent=parent),timeout=2*3600+300)
            completion=read(directory/"completion.json")
            if completion["status"]!="completed" or completion["next_epoch"]!=1:
                raise ValueError("Daily update failed to complete a full epoch; no partial successor")
            checkpoint=directory/read(directory/"latest.json")["checkpoint"]
            evaluation,summary=self.evaluate(f"m{version}_d{day+1}",checkpoint=checkpoint,parent=parent,day=day+1)
            self.append_model(version,checkpoint,parent,evaluation,summary,[day])
            parent=checkpoint
        self.verify()
        json_save(self.output/"model_chain_inventory.json",dict(status="completed",versions=self.versions))
        json_save(self.output/"completion.json",dict(status="completed",finished_at=now(),versions=10,edges=9,
            all_evaluations_independently_audited=True,all_positive_and_negative_results_retained=True,
            elapsed_seconds=time.monotonic()-self.started,quality_claim="observed-positive complete-candidate request ranking",
            scope_limit="matched-prefix snapshots; no persistent serving/FIFO experiment"))

    def stop_child(self):
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid,signal.SIGTERM)
            try:
                self.child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid,signal.SIGKILL)
                self.child.wait(timeout=10)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--data",type=Path,default=BASE/"data")
    parser.add_argument("--readiness",type=Path,required=True)
    args=parser.parse_args()
    lock=(BASE/"runs/active.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    chain=Chain(args.output,args.data,args.readiness)
    def interrupted(signum,frame):
        raise InterruptedError(f"Coordinator received signal {signum}")
    for sig in [signal.SIGTERM,signal.SIGINT]:
        signal.signal(sig,interrupted)
    try:
        chain.execute()
    except BaseException as error:
        chain.stop_child()
        # Never alter a previous initialized run after duplicate-dispatch refusal.
        if hasattr(chain,"binding"):
            json_save(chain.output/"failure.json",dict(status="stopped",at=now(),error=str(error),
                traceback=traceback.format_exc(),completed_models=len(chain.versions),all_artifacts_preserved=True))
        raise


if __name__=="__main__":
    main()
