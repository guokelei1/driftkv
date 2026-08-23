#!/usr/bin/env python3
"""Measure grouped GPU transition runtime for sealed 8L policy assignments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

import eval_p9_cutover_profiler_raw as p9prof
import eval_p9_materialized_lineage_canary as transition
import eval_scale_8l_hs_raw as hs
import train_p7_theta0 as p7
from hstu_kvcache.models import transition_work

ROOT=Path(__file__).resolve().parents[1]
POPULATION=ROOT/"data/manifests/scale_8l_population_v1"
SCHEDULER=ROOT/"results/scale_8l_v1/scheduler/scheduler_result.json"
SEAL=ROOT/"results/scale_8l_v1/scheduler/assignment_seal.json"
OUTPUT=ROOT/"results/scale_8l_v1/policy_runtime"
ACTIONS=("noop","layer0_recent128","layer0_middle","layer0_full","hybrid_tail128","exact_all")


@torch.no_grad()
def execute_groups(current,parent,reader,rows_by_action,device,batch_size=4):
    runtime=0.0; work={}; states=0
    for action,rows in rows_by_action.items():
        grouped={}
        for row in rows: grouped.setdefault(int(row["effective_prefix_length"]),[]).append(row)
        for length in sorted(grouped):
            values=grouped[length]
            for begin in range(0,len(values),batch_size):
                micro=values[begin:begin+batch_size]
                items,behaviors,deltas,_=p9prof.state_tensors(reader,micro,device)
                parent_cache=parent.compute_kv(items,behaviors,deltas)
                torch.cuda.synchronize(device); tick=time.perf_counter()
                migrated=transition.migrate(action,current,parent_cache,(items,behaviors,deltas))
                torch.cuda.synchronize(device); runtime += time.perf_counter()-tick
                measured=transition_work(action,parent_cache,items,behaviors,deltas)
                for key,value in measured.__dict__.items(): work[key]=work.get(key,0)+int(value)
                states += len(micro); del parent_cache,migrated
    return {"states":states,"grouped_transition_runtime_seconds":runtime,"logical_work":work}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--release",choices=("r1_edge1","r1_edge2","r2"),required=True)
    parser.add_argument("--device",choices=tuple(f"cuda:{i}" for i in range(4)),required=True)
    parser.add_argument("--state-limit",type=int); parser.add_argument("--output",type=Path); args=parser.parse_args()
    output=(args.output or OUTPUT/args.release/"m0_f_seed17").resolve()
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    seal=json.loads(SEAL.read_text())
    if seal["status"]!="sealed_before_quality" or p7.sha256_file(SCHEDULER)!=seal["scheduler_result_sha256"]: raise RuntimeError("assignments not sealed")
    scheduler=json.loads(SCHEDULER.read_text()); cell=next(x for x in scheduler["cells"] if x["release"]==args.release)
    edge="edge2" if args.release=="r1_edge2" else "edge1"; frame=pq.read_table(POPULATION/edge/"states.parquet").to_pandas()
    cutover=21168000 if edge=="edge2" else 19958400
    frame["cutover"]=cutover
    if args.state_limit is not None:
        # Canary selection is deterministic and independent of action benefit.
        frame=frame.sort_values("uid").head(args.state_limit).reset_index(drop=True)
    rows={int(row["uid"]):row for row in frame.to_dict("records")}; assignments=pq.read_table(ROOT/cell["primary_assignments"]).to_pandas()
    current_path=hs.checkpoint_path(args.release); current,child=hs.load_model(current_path,torch.device(args.device)); parent,_=hs.load_model(ROOT/child["parent_checkpoint"],torch.device(args.device))
    reader=p9prof.RawStateReader(); results=[]
    # The same 1% probe population is shared by all three budgets. Measure it once:
    # every non-No-op action is executed, and the resulting Exact state is reusable.
    assignments=assignments[assignments.uid.astype(int).isin(set(frame.uid.astype(int)))]
    primary=assignments[np.isclose(assignments.budget_fraction,.05)]
    probe_uids=primary[primary.calibration_sample].uid.astype(int).tolist()
    probe_rows=[rows[uid] for uid in probe_uids]
    probe=execute_groups(current,parent,reader,{action:probe_rows for action in ACTIONS[1:]},torch.device(args.device))
    for budget in (.05,.10,.25):
        selected=assignments[np.isclose(assignments.budget_fraction,budget)]
        groups={action:[] for action in ACTIONS}
        for row in selected.itertuples():
            if not bool(row.calibration_sample): groups[str(row.action)].append(rows[int(row.uid)])
        migration=execute_groups(current,parent,reader,{a:v for a,v in groups.items() if v},torch.device(args.device))
        results.append({"budget_fraction":budget,"probe":probe,"unsampled_migration":migration,
            "charged_grouped_transition_runtime_seconds":probe["grouped_transition_runtime_seconds"]+migration["grouped_transition_runtime_seconds"],
            "action_counts_unsampled":{a:len(v) for a,v in groups.items()}})
    output.mkdir(parents=True); payload={"status":"scale_8l_mixed_policy_grouped_runtime_measured","release":args.release,
        "states":len(frame),"assignment_seal_sha256":p7.sha256_file(SEAL),"results":results,
        "runtime_excludes_checkpoint_load_parent_cache_materialization_and_storage_IO":True,
        "logical_IO_reported_in_transition_work":True,"qualification_or_theta3_read":False}
    (output/"result.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
