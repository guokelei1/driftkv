"""Bounded durable independent branches, with immutable inputs and complete scoring."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.recflow_full_population.code.data_common import json_save, sha256
from .common import BASE, CONFIG, DATA, ORIGINAL, REPO, branch_binding, load_spec, read

PACKAGE = "experiments.recflow_release_design.code."
GIB = 1024**3


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "stages", "branches", "source_snapshot"):
        (output / name).mkdir()
    proposal = BASE / "preparation_2026_10_29/plan.json"
    original_plan = read(ORIGINAL / "plan.json")
    specs = []
    for spec in read(proposal)["initial_development_batch"]:
        path = output / "branches" / spec["branch"] / "spec.json"
        path.parent.mkdir()
        json_save(path, spec)
        load_spec(path)
        specs.append(str(path))
    binding = dict(original_plan["sources_and_inputs"], **branch_binding())
    for path in [proposal, ORIGINAL / "completion.json", *map(Path, specs)]:
        binding[str(path)] = sha256(path)
    for path in (BASE / "code").glob("*.py"):
        shutil.copy2(path, output / "source_snapshot" / path.name)
    checkpoint = Path(read(specs[0])["first_pass_checkpoint_path"])
    example_eval = Path(read(specs[0])["original_evaluation_summary"]).parent
    eval_bytes = sum(p.stat().st_size for p in example_eval.rglob("*") if p.is_file())
    disk = shutil.disk_usage(output)
    budget = 18*GIB*len(specs)
    if disk.free < budget + 48*GIB:
        raise RuntimeError("Insufficient disk for all bounded branches and 48 GiB headroom")
    storage = dict(checked_at=now(), free_bytes=disk.free, total_bytes=disk.total,
                   checkpoint_bytes=checkpoint.stat().st_size, example_evaluation_bytes=eval_bytes,
                   planned_branches=len(specs), reservation_bytes_per_branch=18*GIB,
                   projected_free_after_reservation=disk.free-budget, deletion_performed=False,
                   decision="Enough space; retain raw inputs, processed data, original checkpoints and all outcomes",
                   raw_recflow="approximately 45 GiB; runtime reads processed arrays, but raw inputs remain useful for reconstruction",
                   processed_recflow="approximately 35 GiB shared by old and new runs through symlinks; not disposable")
    json_save(output / "storage_assessment.json", storage)
    plan = dict(created_at=now(), status="prepared", branch_specs=specs, binding=binding,
                physical_gpus=[0,1,2,3], global_batch=64, micro_batch=16, extra_epochs=1, total_update_epochs=2,
                shuffle_seed=18, preserve_all_signs=True, select_by_gap=False,
                scope="Four independent second-pass release-edge development branches; no continuous successor chain",
                quality="Full next-day original training cohort, NDCG@10 primary, Recall@10 secondary",
                storage=storage,
                original_baseline_metrics=str(ORIGINAL / "version_metrics.csv"), estimated_hours=[4,5])
    json_save(output / "plan.json", plan)
    print(json.dumps(dict(status="prepared", output=str(output), branches=len(specs), free_gib=disk.free/GIB)), flush=True)


class Runner:
    def __init__(self, output):
        self.output = output
        self.plan = read(output / "plan.json")
        self.child, self.serial, self.results = None, 0, []
        self.started = time.monotonic()
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                        PYTHONPATH=str(REPO/"src")+os.pathsep+str(REPO), PYTHONUNBUFFERED="1", EVOKV_ATTENTION_BACKEND="auto")

    def verify(self, gpu):
        if shutil.disk_usage(self.output).free < 48*GIB:
            raise RuntimeError("Less than 48 GiB disk headroom; stopping before another stage")
        if gpu:
            raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
            available = [int(x) for x in raw.split()]
            if len(available)<4 or min(available[:4])<32*1024:
                raise RuntimeError("Four GPUs no longer have the measured 32 GiB memory headroom")

    def stage(self, name, module, args, *, gpu, timeout):
        self.verify(gpu)
        self.serial += 1
        label = f"{self.serial:02d}_{name}"
        command = [sys.executable]
        if gpu:
            command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4"]
        command += ["-m", module] + list(map(str,args))
        log = self.output / "logs" / f"{label}.log"
        record = dict(stage=name, command=command, started_at=now(), log=str(log), status="running")
        begun = time.monotonic()
        with log.open("x") as handle:
            self.child = subprocess.Popen(command, cwd=REPO, env=self.env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            record.update(pid=self.child.pid, pid_start_ticks=Path(f"/proc/{self.child.pid}/stat").read_text().rsplit(") ",1)[1].split()[19])
            json_save(self.output / "active_process.json", record)
            json_save(self.output / "stages" / f"{label}.json", record)
            print(json.dumps(record), flush=True)
            while self.child.poll() is None:
                try:
                    self.child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    json_save(self.output / "progress.json", dict(status="running", stage=name, completed_branches=len(self.results),
                              updated_at=now(), elapsed_seconds=time.monotonic()-self.started, pid=self.child.pid))
                    if time.monotonic()-begun > timeout:
                        raise TimeoutError(name)
            code = self.child.returncode
        record.update(status="completed" if code==0 else "failed", returncode=code, finished_at=now(), wall_seconds=time.monotonic()-begun)
        json_save(self.output / "active_process.json", record)
        json_save(self.output / "stages" / f"{label}.json", record)
        self.child = None
        if code:
            raise RuntimeError(f"{name} failed; retained logs, no automatic retry")

    def evaluate(self, spec_path, checkpoint, directory, *, diagnostic=False):
        spec = read(spec_path)
        args = ["--branch-spec",spec_path,"--checkpoint",checkpoint,"--parent",spec["previous_published_checkpoint"],
                "--data",DATA,"--config",CONFIG,"--output",directory,"--days",spec["eval_day"],"--role",2,
                "--batch-size",16,"--workers",2]
        if diagnostic:
            args += ["--baseline-parity", "--diagnostic-batches", 1]
        self.stage(directory.name, PACKAGE+"evaluate",args,gpu=True,timeout=3*3600)
        if diagnostic:
            self.stage("baseline_parity_check", PACKAGE+"compare", ["--branch-spec",spec_path,"--evaluation",directory,
                       "--output",directory/"parity_check.json","--parity"],gpu=False,timeout=600)
        else:
            self.stage(spec["branch"]+"_audit", "experiments.recflow_full_population.code.audit_scored_requests",
                       ["--evaluation",directory,"--data",DATA,"--output",directory/"independent_audit.json"],gpu=False,timeout=3600)
            self.stage(spec["branch"]+"_comparison", PACKAGE+"compare", ["--branch-spec",spec_path,"--evaluation",directory,
                       "--output",directory/"original_vs_extra_epoch.json"],gpu=False,timeout=1800)

    def execute(self):
        first = Path(self.plan["branch_specs"][0])
        self.evaluate(first, read(first)["first_pass_checkpoint_path"], self.output / "diagnostic_original64", diagnostic=True)
        for item in self.plan["branch_specs"]:
            spec_path, spec = Path(item), read(item)
            directory = spec_path.parent / "training"
            args = ["--branch-spec",spec_path,"--data",DATA,"--config",CONFIG,"--output",directory,
                    "--phase","update","--scale","max","--days",spec["train_day"],"--epochs",1,
                    "--parent",spec["first_pass_checkpoint_path"],"--global-batch",64,"--micro-batch",16,"--workers",4,
                    "--listwise-weight",.1,"--dtype","bfloat16","--communication","rows","--seed",18,
                    "--checkpoint-every",100000,"--log-every",100,"--wall-hours",2]
            self.stage(spec["branch"]+"_train",PACKAGE+"extend",args,gpu=True,timeout=2.5*3600)
            completion = read(directory / "completion.json")
            if completion["status"]!="completed" or completion["next_epoch"]!=1 or completion["next_step"]!=0:
                raise ValueError("Second pass did not finish")
            checkpoint = directory / read(directory / "latest.json")["checkpoint"]
            evaluation = spec_path.parent / "evaluation"
            self.evaluate(spec_path, checkpoint, evaluation)
            result = read(evaluation / "original_vs_extra_epoch.json")
            result.update(checkpoint=str(checkpoint), checkpoint_sha256=read(checkpoint.with_suffix(".json"))["checkpoint_sha256"])
            self.results.append(result)
            json_save(self.output / "branch_inventory.json", dict(status="in_progress", branches=self.results))
            rows = []
            for r in self.results:
                rows.append(dict(branch=r["branch"], original_rebuild=r["old_metrics"]["current"]["ndcg@10"],
                      original_reuse=r["old_metrics"]["reuse"]["ndcg@10"], rebuild=r["new_metrics"]["current"]["ndcg@10"],
                      reuse=r["new_metrics"]["reuse"]["ndcg@10"], relative_loss_pct=r["new_relative_loss_pct"],
                      ci95_low=r["new_gap_ci95"][0],ci95_high=r["new_gap_ci95"][1],
                      native_change=r["new_minus_old_rebuild"]["difference"],
                      native_change_ci95_low=r["new_minus_old_rebuild"]["ci95"][0],
                      native_change_ci95_high=r["new_minus_old_rebuild"]["ci95"][1]))
            with (self.output / "branch_metrics.csv").open("w") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        self.verify(False)
        completion = dict(status="completed", completed_branches=len(self.results), all_signs_retained=True,
                          all_windows_independently_audited=True, elapsed_seconds=time.monotonic()-self.started, finished_at=now())
        json_save(self.output / "branch_inventory.json", dict(status="completed", branches=self.results))
        json_save(self.output / "completion.json", completion)
        json_save(self.output / "progress.json", completion)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--prepare",action="store_true")
    args=p.parse_args()
    output=args.output.resolve()
    if args.prepare:
        prepare(output)
        return
    if (output/"completion.json").exists() or (output/"failure.json").exists() or (output/"active_process.json").exists():
        raise ValueError("Existing attempt must be preserved; no implicit restart")
    with (output/"coordinator.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runner=Runner(output)
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        try:
            runner.execute()
        except BaseException:
            if runner.child is not None and runner.child.poll() is None:
                os.killpg(runner.child.pid,signal.SIGTERM)
                try:runner.child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(runner.child.pid,signal.SIGKILL);runner.child.wait()
            json_save(output/"failure.json",dict(status="failed",at=now(),traceback=traceback.format_exc(),all_outputs_retained=True))
            raise


if __name__ == "__main__":
    main()
