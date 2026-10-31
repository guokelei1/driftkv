#!/usr/bin/env python3
"""CPU-only full-input coverage/cost preflight and statistics runtime probe."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
from collections import Counter
import gc
import json
import time
import numpy as np
from experiments.figure9max.common import (
    HERE, configuration, digest, edge_config, ledger, load_inputs, policies, write_json,
)
from hstu_kvcache.methods.kv_map.cost_max import deployment_flops, fitting_flops
from hstu_kvcache.evaluation.cost import CostModel

def offline_shapes(inputs,cfg):
    stats=Counter()
    at=cfg["release_day"]*86400
    for uid in inputs["quality"]:
        rounds=inputs["by_user"].get(uid)
        if not rounds:
            continue
        stats["feedback_users"]+=1
        times=inputs["histories"].rows[uid][0]
        cursor=int(np.searchsorted(times,at,side="left"))
        n=min(cursor,1024)
        stats["short_initial_users" if n<1024 else "full_initial_users"]+=1
        for timestamp,rows in sorted(rounds.items()):
            stop=int(np.searchsorted(times,timestamp,side="left"))
            while cursor<stop:
                width=1<<(min(128,stop-cursor).bit_length()-1)
                stats["append_bands"]+=1
                stats["append_short" if n<1024 else "append_full"]+=1
                stats["append_tokens"]+=width
                cursor+=width
                n=min(1024,n+width)
            stats["query_batches"]+=(len(rows)+15)//16
            stats["feedback_requests"]+=len(rows)
    return dict(stats)

def planned_fit_cost(inputs,cfg,rows):
    # Same declared analytical terms as the numerical fitting executor.
    cm=CostModel(320,16,10,"triton")
    observations=capture=0
    result,details={},{}
    for count,uid in enumerate(inputs["fit"],1):
        n=min(1024,int(np.searchsorted(inputs["histories"].rows[uid][0],cfg["release_day"]*86400,side="left")))
        observations+=n
        capture+=2*(cm if n==1024 else CostModel(320,16,10,"torch")).full_cache(n)
        for p in rows:
            if p["method"]=="kv_map" and p["fit_users"]==count:
                audit=fitting_flops(observations,p["width"],cfg["kv_training"]["epochs"],
                    cfg["kv_training"]["batch_rows"],capture)
                result[p["tag"]]=audit["total_flops"]
                details[p["tag"]]=audit
    return result,details

def prepare_edge(edge):
    cfg=configuration()
    rows=policies(cfg)
    ecfg=edge_config(cfg,edge)
    inputs=load_inputs(cfg,edge,include_fit=True)
    assert len(inputs["quality"])==cfg["expected_quality_users"]
    assert inputs["requests"]==cfg["expected_requests"][edge-1]
    workload=offline_shapes(inputs,ecfg)
    fitting,detail=planned_fit_cost(inputs,ecfg,rows)
    cost=ledger(inputs,ecfg,rows,fitting)
    record=dict(edge=edge,window=[ecfg["release_day"],ecfg["end_day"]],
        window_name="E14_partial" if edge==5 else "E14",population=cfg["expected_quality_users"],requests=inputs["requests"],
        input_fingerprint=inputs["fingerprint"],input_load_seconds=inputs["load_seconds"],
        offline_workload=workload,costs=cost,kv_map_fit_audit=detail)
    print(json.dumps(dict(stage="input_coverage_passed",edge=edge,requests=inputs["requests"],seconds=inputs["load_seconds"])),flush=True)
    return record

def main():
    cfg=configuration()
    rows=policies(cfg)
    records=[]
    started=time.perf_counter()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=5) as pool:
        for record in pool.map(prepare_edge,cfg["edges"]):
            records.append(record)
            write_json(HERE/"planned_costs.json",dict(status="running",edges=records))
    # Timing only: synthetic scalar arrays at the true full-edge dimensions.
    # They are never reported as experiment quality and are never persisted.
    from hstu_kvcache.evaluation.statistics import weighted_auc
    rng=np.random.default_rng(17)
    requests=cfg["expected_requests"][0]
    labels=rng.integers(2,size=requests).astype(np.float64)
    users=np.sort(rng.integers(cfg["expected_quality_users"]//2,size=requests))
    score=rng.normal(size=requests)
    weights=np.stack([np.bincount(rng.integers(cfg["expected_quality_users"],size=cfg["expected_quality_users"]),minlength=cfg["expected_quality_users"]) for _ in range(100)]).astype(np.uint16)
    begin=time.perf_counter()
    weighted_auc(score,labels,users,weights)
    statistic_seconds=time.perf_counter()-begin
    stats=dict(scope="Synthetic arrays for CPU timing only, not quality evidence",requests=requests,
        measured_bootstrap_draws=100,measured_seconds=statistic_seconds,
        projected_1000_draws_29_points_5_edges_seconds=statistic_seconds*10*29*5)
    # Time a realistic per-rank rolling numeric checkpoint, then remove it.
    from experiments.figure9max.common import Rolling
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(dir=HERE) as tmp:
        store=Rolling(tmp,"synthetic-statistics-speed-only",0)
        count=requests//4
        store.append(1,labels[:count],rng.normal(size=(count,29)).astype(np.float32))
        store.save(1,29,True)
        checkpoint=dict(seconds=store.save_seconds,bytes=(store.folder/"checkpoint.npz").stat().st_size,
            scope="Synthetic final-size rank scalar checkpoint; temporary arrays removed")
    write_json(HERE/"planned_costs.json",dict(status="complete",config_sha256=digest(HERE/"config.json"),calibration_arithmetic_version=3,edges=records,statistics_timing=stats,
        checkpoint_timing=checkpoint,preparation_seconds=time.perf_counter()-started,
        purpose="CPU-only full-input coverage and prospective cost calculation; no full-population GPU quality run launched"))
    print(json.dumps(dict(stage="preflight_complete",statistics=stats,checkpoint=checkpoint)),flush=True)

if __name__=="__main__":
    import sys
    main()
