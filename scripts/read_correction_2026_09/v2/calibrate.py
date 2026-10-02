#!/usr/bin/env python3
"""Independent v2 Q distillation and explicit full-history residual calibration."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pyarrow as pa
import torch

from evaluate_yambda500m_foundation_raw import load_histories, load_model
from hstu_kvcache.read_correction import build_correction as build_v1_correction
from read_correction_2026_09.calibrate import capture
from read_correction_2026_09.cost import CostModel
from read_correction_2026_09.v2.query_only.fit import fit_q
from read_correction_2026_09.v2.common import (
    OUTPUT, PANEL_ROOT, RESERVATIONS, edge_name, plan, sha256, sources, write_json,
)
from selective_recompute_2026_09.calibrate import snapshot


def mixed_candidates(uid, history_items, known, count=16, seed=17, recent_budget=None):
    """Recent unique known IDs, then uniform fill; default recent quota is half."""
    recent, seen = [], set()
    for value in reversed(history_items):
        item = int(value)
        if 0 < item < known and item not in seen:
            recent.append(item)
            seen.add(item)
            if len(recent) == (count//2 if recent_budget is None else recent_budget):
                break
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(uid)]))
    result = list(recent)
    while len(result) < count:
        item = int(rng.integers(1, known))
        if item not in seen:
            result.append(item)
            seen.add(item)
    return np.asarray(result, dtype=np.int64)


def load_data(args, config):
    """Shared bounded snapshot preparation; usable by a separate H v2 fitter."""
    binding_path = args.panel_root/args.scale/edge_name(args.edge)/"binding.json"
    binding = json.loads(binding_path.read_text())
    reservation_path = args.reservations/args.scale/edge_name(args.edge)/"calibration_users.json"
    reservation = json.loads(reservation_path.read_text())
    if reservation["source_binding"]["sha256"] != sha256(binding_path):
        raise RuntimeError("calibration reservation panel binding changed")
    uids = reservation["uids"][:max(args.budgets)]
    if len(uids)<max(args.budgets):
        raise RuntimeError("requested budget exceeds the expanded independent reservation")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    free,total = torch.cuda.mem_get_info(device)
    if free/total<config["initial_free_fraction"]:
        raise RuntimeError("GPU lacks initial free-memory reserve")
    torch.cuda.set_per_process_memory_fraction(config["memory_fraction"],device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(config["torch_threads"])
    pa.set_cpu_count(config["history_threads"])
    torch.backends.cuda.matmul.allow_tf32=False
    os.environ["EVOKV_ATTENTION_BACKEND"]=config["attention_backend"]
    started=time.perf_counter()
    parent,pp=load_model(ROOT/binding["sources"]["parent"]["path"],device)
    current,cp=load_model(ROOT/binding["sources"]["current"]["path"],device)
    if pp["config"]!=cp["config"]:
        raise RuntimeError("parent/current architecture differs")
    current.requires_grad_(False);parent.requires_grad_(False)
    dataset_path=ROOT/binding["sources"]["dataset"]["path"]
    dataset=json.loads(dataset_path.read_text())
    known=int(cp.get("known_vocab_size",dataset["foundation_items"]))
    cutover=int(binding["cutover"])
    history=load_histories(uids,dataset_path=dataset_path,known_vocab_size=known,
        oov_buckets=int(cp["config"]["num_items"])-known,start_timestamp=cutover,end_timestamp=cutover+1,
        max_history=config["history_length"],threads=config["history_threads"])
    histories={uid:snapshot(history,uid,cutover,config["history_length"]) for uid in uids}
    del history,pp
    width=current.blocks[0].attn.num_heads*current.blocks[0].attn.head_dim
    cache_bytes=16*len(current.blocks)*width*sum(len(histories[uid][0]) for uid in uids)
    if cache_bytes>128*(1<<30):
        raise RuntimeError("CPU calibration caches exceed 128 GiB per process")
    batch=config["capture_batches"][args.scale]
    rows=capture(parent,current,histories,uids,cutover=cutover,known=known,
        queries=config["calibration_queries_per_user"],device=device,batch_size=batch,
        history_length=config["history_length"],attention_backend=config["attention_backend"])
    for uid in uids:
        rows[uid]["candidates"]=torch.from_numpy(mixed_candidates(uid,histories[uid][1],known,
            config["calibration_queries_per_user"],config["seed"]))
    del parent,histories
    gc.collect()
    metadata={"panel_binding_sha256":sha256(binding_path),"calibration_users_sha256":sha256(reservation_path),
        "cutover":cutover,"model_config":cp["config"],"cache_bytes":cache_bytes,
        "shared_snapshot_users":len(uids),
        "cache_prepare_seconds":time.perf_counter()-started,
        "checkpoint_hashes":{key:binding["sources"][key]["sha256"] for key in ("parent","current")},
        "candidate_rule":"half most-recent unique known items from strictly pre-release snapshot, half uniform known catalog; missing recent slots filled uniformly; SeedSequence([17,uid])",
        "teacher":"Current Full on aligned strictly-pre-release history; no feedback labels; mapped-item tie order shared with inherited cache"}
    del cp
    return current,rows,uids,device,metadata


def run(args):
    config=plan()
    methods=[args.method] if args.method else ["query_only","history_conditioned"]
    if args.budgets and not args.method:
        raise ValueError("--budgets requires --method; otherwise use method-specific budget flags")
    budgets={method:sorted(set(args.budgets or getattr(args,"query_budgets" if method=="query_only" else "history_budgets")
                              or config["budgets"][method])) for method in methods}
    if args.epochs is not None:
        for method in methods:
            config["query_epochs" if method=="query_only" else "history_epochs"]=args.epochs
    source_hashes=sources()
    reservation_path=args.reservations/args.scale/edge_name(args.edge)/"calibration_users.json"
    reserved=json.loads(reservation_path.read_text())["uids"]
    binding_path=args.panel_root/args.scale/edge_name(args.edge)/"binding.json"
    base_root=ROOT/"results/read_correction_2026_09/history_conditioned/development/v1"/args.scale/edge_name(args.edge)
    pending=[]
    for method in methods:
      for budget in budgets[method]:
        path=args.output_root/method/args.scale/edge_name(args.edge)/f"calibration_c{budget}.json"
        if path.exists():
            previous=json.loads(path.read_text())
            expected_kind="query_only" if method=="query_only" else "history_conditioned_v2"
            if (previous.get("status")!="complete" or previous.get("kind")!=expected_kind
                    or previous.get("settings")!=config or previous.get("uids")!=reserved[:budget]
                    or previous.get("execution_source_hashes")!=source_hashes
                    or previous.get("panel_binding_sha256")!=sha256(binding_path)
                    or previous.get("calibration_users_sha256")!=sha256(reservation_path)
                    or previous.get("weights_sha256")!=sha256(path.with_suffix(".pt"))):
                raise RuntimeError(f"existing calibration inputs/weights differ: {path}")
            if method=="history_conditioned" and previous["base_calibration"]["weights_sha256"]!=sha256(base_root/f"calibration_c{budget}.pt"):
                raise RuntimeError("inherited H base changed")
            print(json.dumps({"status":"calibration_already_complete","output":str(path)}),flush=True)
        else:
            pending.append((method,budget,path))
    if not pending:
        return
    load_args=argparse.Namespace(**vars(args))
    load_args.budgets=[max(budget+(16 if method=="history_conditioned" else 0) for method,budget,_ in pending)]
    current,rows,all_uids,device,metadata=load_data(load_args,config)
    for method,budget,path in pending:
        selected=all_uids[:budget]
        base_record=None
        if method=="query_only":
            kind="query_only"
            modules,fit=fit_q(current,rows,selected,config=config,device=device,scale=args.scale,
                batch_size=config["capture_batches"][args.scale],train_batch=config["calibration_batches"]["query_only"][args.scale])
        else:
            from read_correction_2026_09.v2.history_conditioned.fit import fit as fit_history
            kind="history_conditioned_v2"
            base_path=base_root/f"calibration_c{budget}.pt"
            base_record=json.loads(base_path.with_suffix(".json").read_text())
            if base_record["uids"]!=selected or sha256(base_path)!=base_record["weights_sha256"]:
                raise RuntimeError("H base weights or independent training UIDs changed")
            base_artifact=torch.load(base_path,map_location="cpu",weights_only=False)
            base_modules=[build_v1_correction("history_conditioned",item["config"],item["state_dict"]).to(device).eval()
                          for item in base_artifact["modules"]]
            validation_uids=all_uids[budget:budget+16]
            if len(validation_uids)!=16:
                raise RuntimeError("H needs 16 additional users independent of its retained base fit")
            modules,fit=fit_history(current,rows,selected,base_modules,config,device,args.scale,
                config["calibration_batches"]["history_conditioned"][args.scale],validation_uids=validation_uids)
            cost=CostModel.for_scale(args.scale,config["attention_backend"])
            torch_cost=CostModel.for_scale(args.scale,"torch")
            ledger=fit["cost"]
            ledger["parent_and_teacher_cache_flops"]=2*sum((cost if rows[uid]["parent"].seq_len==config["history_length"] else torch_cost)
                .full_cache(rows[uid]["parent"].seq_len) for uid in selected+validation_uids)
            ledger["inherited_v1_calibration_flops"]=int(base_record["cost"]["calibration_flops"])
            ledger["calibration_flops"]=sum(value for key,value in ledger.items()
                if key!="calibration_flops" and isinstance(value,(int,float)))
            del base_artifact,base_modules
        torch.cuda.synchronize(device)
        record={"status":"complete","kind":kind,"method":method,"revision":"v2",
            "scale":args.scale,"edge":edge_name(args.edge),"users":budget,"uids":selected,
            "queries_per_user":config["calibration_queries_per_user"],"settings":config,
            "evaluation_role":"development_exploration","fit":fit,"cost":fit["cost"],
            "history_length_histogram":dict(Counter(rows[uid]["parent"].seq_len for uid in selected)),
            "execution_source_hashes":source_hashes,"peak_allocated_gib":torch.cuda.max_memory_allocated(device)/(1<<30),
            "peak_reserved_gib":torch.cuda.max_memory_reserved(device)/(1<<30),
            "cpu_peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1<<20),**metadata}
        if base_record is not None:
            record["base_calibration"]={"weights_sha256":base_record["weights_sha256"],
                "path":str(base_root/f"calibration_c{budget}.pt"),"users":budget,
                "validation_note":"new heavy-branch validation is the next 16 reservation UIDs, unseen by both retained base and new branch"}
        payload={"kind":kind,"modules":[{"config":module.get_config(),"state_dict":
            {key:value.detach().cpu() for key,value in module.state_dict().items()}} for module in modules],"metadata":record}
        path.parent.mkdir(parents=True,exist_ok=True)
        weights=path.with_suffix(".pt")
        temporary=weights.with_suffix(".pt.partial")
        torch.save(payload,temporary);temporary.replace(weights)
        record["weights_sha256"]=sha256(weights)
        write_json(path,record)
        print(json.dumps({"status":"calibration_complete","output":str(path),
                          "selected_epoch":fit.get("selected_epoch"),"fit_seconds":fit.get("fit_seconds")}),flush=True)
        del modules,payload


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale",choices=("medium","large","max"),required=True)
    parser.add_argument("--edge",type=int,choices=range(1,6),required=True)
    parser.add_argument("--gpu",type=int,required=True)
    parser.add_argument("--method",choices=("query_only","history_conditioned"))
    parser.add_argument("--budgets",type=int,nargs="+")
    parser.add_argument("--query-budgets",type=int,nargs="+")
    parser.add_argument("--history-budgets",type=int,nargs="+")
    parser.add_argument("--epochs",type=int)
    parser.add_argument("--panel-root",type=Path,default=PANEL_ROOT)
    parser.add_argument("--reservations",type=Path,default=RESERVATIONS)
    parser.add_argument("--output-root",type=Path,default=OUTPUT)
    run(parser.parse_args())
