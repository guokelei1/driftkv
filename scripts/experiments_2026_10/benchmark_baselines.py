#!/usr/bin/env python3
"""Real-model offline-evaluation speed probe; no population quality run."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]

import pyarrow.parquet as pq
import torch

from experiments_2026_10.baselines import choose_repairs,dataset_path,evaluate,event_shapes,prepare_inputs
from experiments_2026_10.fast_baselines import NativeGraphs,evaluate_batched
from design_one.prepare_history import load_histories_semijoin
from evaluate_yambda500m_foundation_raw import load_model
from hstu_kvcache.read_correction_v4.nonlinear import NonlinearTokenReadCorrection
from hstu_kvcache.read_correction_v5.query_only.core import QueryFeatureCorrection
from unified_reuse_2026_09.common import sha256,write_json
from read_correction_2026_09.cost import CostModel


def load_policy(folder,method,tag,device):
    cls=QueryFeatureCorrection if method=="query_only" else NonlinearTokenReadCorrection
    values=torch.load(folder/"calibration.pt",map_location="cpu",weights_only=True)
    modules=[]
    for layer in values:
        module=cls(**layer["config"]).to(device)
        module.load_state_dict(layer["state_dict"])
        modules.append(module.eval().requires_grad_(False))
    return dict(tag=tag,method=method,modules=modules,calibration_flops=0,
        weight_source=str(folder),weights_sha256=sha256(folder/"calibration.pt"))


def difference(actual,wanted):
    key=lambda r:(r["method"],r["uid"],r["query_timestamp"],r["raw_item_id"])
    a={key(r):r["score"] for r in actual}
    b={key(r):r["score"] for r in wanted}
    assert len(a)==len(actual) and a.keys()==b.keys()
    maximum=0.
    for k in a:
        maximum=max(maximum,abs(a[k]-b[k]))
        torch.testing.assert_close(torch.tensor(a[k]),torch.tensor(b[k]),atol=3e-5,rtol=3e-5)
    return dict(scores=len(a),max_absolute_logit_error=maximum,atol=3e-5,rtol=3e-5)


def run(args):
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    source=ROOT/"experiments/01_quality_cost/runs/yambda/large/baselines_v0_v1/seed_17/attempt_1"
    canary=ROOT/"experiments/01_quality_cost/runs/yambda/large/baseline_canary/seed_17/attempt_1"
    binding=json.loads((source/"configuration.json").read_text())
    cfg=binding["settings"]
    request_path=source/"requests.parquet"
    if args.full_test_sample:
        cfg=json.loads((ROOT/"experiments/01_quality_cost/configs/baselines_yambda_large_v0_v1_201.json").read_text())
        binding=prepare_inputs(cfg,out)
        request_path=out/"requests.parquet"
        candidates=binding["roles"]["quality"]
        selected=sorted(candidates,key=lambda uid:hashlib.sha256(f"baseline_speed_sample_20261021/17/{uid}".encode()).digest())[:args.users]
    else:
        selected=binding["roles"]["cost"][:args.users]
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False
    os.environ["EVOKV_ATTENTION_BACKEND"]="triton"
    device=torch.device("cuda:0")
    torch.cuda.set_device(device)
    monitor_file=(out/"gpu_util.csv").open("w")
    monitor=subprocess.Popen(["nvidia-smi","--id=0","--query-gpu=timestamp,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits","-lms","1000"],stdout=monitor_file,stderr=subprocess.DEVNULL)
    try:
        began=time.perf_counter()
        path=dataset_path("large")
        ds=json.loads(path.read_text())
        histories=load_histories_semijoin(selected,dataset_path=path,known_vocab_size=ds["foundation_items"],oov_buckets=256,
            start_timestamp=231*86400,end_timestamp=245*86400,max_history=1024,threads=8)
        by_user=defaultdict(lambda:defaultdict(list))
        for row in pq.read_table(request_path,filters=[("uid","in",selected)]).to_pylist():
            by_user[row["uid"]][row["query_timestamp"]].append(row)
        input_seconds=time.perf_counter()-began
        began=time.perf_counter()
        checkpoints=ROOT/"results/unified_training_2026_09/large/seed17/checkpoints"
        parent,raw=load_model(checkpoints/"v0/checkpoint_100.pt",device)
        del raw
        current,raw=load_model(checkpoints/"v1/checkpoint_100.pt",device)
        del raw
        parent.eval().requires_grad_(False);current.eval().requires_grad_(False)
        torch.cuda.synchronize(device)
        model_seconds=time.perf_counter()-began
        saved=json.loads((source/"partial_budgets.json").read_text())
        policies=saved["policies"]
        if args.current_points:
            cm=CostModel(current.cfg.hidden_size,len(current.blocks),current.cfg.num_heads,cfg["attention_backend"])
            policies,_=choose_repairs(event_shapes(histories,by_user,selected,cfg),cm,
                saved["layer_profile_gains"],saved["profile_flops"],cfg)
        for budget in (128,256):
            policies.append(load_policy(source/f"query_only_fit{budget}","query_only",f"query_only_fit{budget}",device))
        policies.append(load_policy(canary/"query_only_fit16","query_only","query_only_shape3",device))
        for budget in (4,8,16):
            policies.append(load_policy(canary/f"history_conditioned_fit{budget}","history_conditioned",f"history_shape{budget}",device))
        config=dict(status="speed_probe",machine="g201",physical_gpu=0,users=selected,
            scale="large",edge="v0_to_v1",settings=cfg,input_seconds=input_seconds,model_load_seconds=model_seconds,
            method_points=[{k:v for k,v in p.items() if k!="modules"} for p in policies],
            scope="runtime/numerical equivalence only, no AUC or population run; completed small and full fits provide actual parameter geometry",
            command=sys.argv,source_sha256=sha256(Path(__file__)))
        if args.full_test_sample:
            config["population_binding"]=binding
            config["sampling"]="independent speed-only SHA256 namespace; no model outcomes or label balancing"
        write_json(out/"configuration.json",config)
        def progress(phase):
            def report(value):
                print(json.dumps(dict(phase=phase,**value)),flush=True)
                write_json(out/"progress.json",dict(phase=phase,**value))
            return report
        if args.batch_sweep:
            # Paired users and fixed selector inputs; no AUC-based tuning.
            variants=[(32,16),(16,16),(32,4),(32,64),(64,16),(128,16)]
            engines={pair:NativeGraphs() for pair in variants}
            reference=None
            records=[]
            for repeat in range(2):
                order=variants if repeat==0 else variants[::-1]
                for band,query_batch in order:
                    selected_cfg=dict(cfg,append_band_size=band,query_batch=query_batch,repair_query_batch=16)
                    engine=engines[band,query_batch]
                    if repeat==0:
                        evaluate_batched(parent,current,histories,by_user,selected[:8],policies,selected_cfg,device,
                            graphs=engine,stop_after_last_query=True)
                    actual,record=evaluate_batched(parent,current,histories,by_user,selected,policies,selected_cfg,device,
                        graphs=engine,stop_after_last_query=True)
                    if reference is None:reference=actual
                    check=difference(actual,reference)
                    value=dict(repeat=repeat,append_band_size=band,query_batch=query_batch,
                        repair_query_batch=16,users=len(selected),seconds=record["seconds"],
                        graph_prepare_seconds=record["graphs"]["prepare_seconds"],equivalence=check)
                    records.append(value)
                    print(json.dumps(dict(phase="batch_sweep",**value)),flush=True)
                    write_json(out/"partial_results.json",dict(records=records))
            aggregates=[]
            for band,query_batch in variants:
                times=[r["seconds"] for r in records if (r["append_band_size"],r["query_batch"])==(band,query_batch)]
                aggregates.append(dict(append_band_size=band,query_batch=query_batch,
                    seconds=times,mean_seconds=sum(times)/len(times)))
            write_json(out/"summary.json",dict(status="complete",scope="batch tuning only; no quality selection",
                records=records,aggregates=aggregates,best=min(aggregates,key=lambda r:r["mean_seconds"]),
                tested_users=len(selected),policy_points=len(policies),machine="g201",physical_gpu=0,
                population_run_started=False,selector_inputs="fixed first <=16 same-timestamp candidates independent of score tile",
                training_batches="unchanged; optimizer settings and completed weights are not part of this speed sweep"))
            return
        results={}
        verify=selected[:args.verify_users]
        started=time.perf_counter()
        reference,_=evaluate(parent,current,histories,by_user,verify,policies,cfg,device,out)
        torch.cuda.synchronize(device)
        results["serial_reference"]=dict(users=len(verify),seconds=time.perf_counter()-started,
            feedback_events=len(reference)//len(policies))
        print(json.dumps(dict(phase="serial_reference",**results["serial_reference"])),flush=True)
        for name,enabled,prune in (("batched",False,False),("batched_graph",True,False),("batched_graph_pruned",True,True)):
            engine=NativeGraphs(enabled=enabled)
            actual,stats=evaluate_batched(parent,current,histories,by_user,verify,policies,cfg,device,
                graphs=engine,stop_after_last_query=prune,progress=progress(name))
            stats["equivalence"]=difference(actual,reference)
            results[name]=stats
            print(json.dumps(dict(phase=name,**stats)),flush=True)
            if name=="batched_graph_pruned":
                results["warm_sample"]={}
                for offset in range(0,len(selected),args.block_users):
                    group=selected[offset:offset+args.block_users]
                    began=time.perf_counter()
                    _,record=evaluate_batched(parent,current,histories,by_user,group,policies,cfg,device,
                        graphs=engine,stop_after_last_query=True,progress=progress("warm_sample"))
                    torch.cuda.synchronize(device)
                    record.update(users=len(group),uids=group,seconds=time.perf_counter()-began)
                    results["warm_sample"][str(offset)]=record
                    write_json(out/"partial_results.json",results)
            del engine
            gc.collect();torch.cuda.empty_cache()
        summary=dict(status="complete",results=results,input_seconds=input_seconds,model_load_seconds=model_seconds,
            tested_users=len(selected),policy_points=len(policies),machine="g201",physical_gpu=0,
            torch_version=torch.__version__,gpu=torch.cuda.get_device_name(device),created_utc=datetime.now(timezone.utc).isoformat(),
            scope=config["scope"],population_run_started=False)
        write_json(out/"summary.json",summary)
    finally:
        monitor.terminate();monitor.wait();monitor_file.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--users",type=int,default=64)
    p.add_argument("--verify-users",type=int,default=8)
    p.add_argument("--block-users",type=int,default=16)
    p.add_argument("--full-test-sample",action="store_true")
    p.add_argument("--batch-sweep",action="store_true")
    p.add_argument("--current-points",action="store_true")
    run(p.parse_args())
