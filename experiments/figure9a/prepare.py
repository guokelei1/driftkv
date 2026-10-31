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
from experiments.figure9a.common import (
    HERE, configuration, digest, edge_config, ledger, load_inputs, policies, write_json,
)
from hstu_kvcache.methods.kv_map.cost_large import deployment_flops, fitting_flops
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
    cm=CostModel(320,10,10,"triton")
    observations=capture=0
    result,details={},{}
    for count,uid in enumerate(inputs["fit"],1):
        n=min(1024,int(np.searchsorted(inputs["histories"].rows[uid][0],cfg["release_day"]*86400,side="left")))
        observations+=n
        capture+=2*(cm if n==1024 else CostModel(320,10,10,"torch")).full_cache(n)
        for p in rows:
            if p["method"]=="kv_map" and p["fit_users"]==count:
                audit=fitting_flops(observations,p["width"],cfg["kv_training"]["epochs"],
                    cfg["kv_training"]["batch_rows"],capture)
                result[p["tag"]]=audit["total_flops"]
                details[p["tag"]]=audit
    return result,details

def kv_only():
    """Reprice only KV maps using the already verified full event ledger."""
    cfg=configuration()
    rows=policies(cfg)
    record=json.loads((HERE/"planned_costs.json").read_text())
    legacy=json.loads((HERE/"legacy_cost_audit.json").read_text())
    for key in ("release_days","end_days","dataset","role_source","history_length","append_band_size","query_batch"):
        assert cfg[key]==legacy["configuration"][key],f"event protocol changed: {key}"
    for edge in record["edges"]:
        ecfg=edge_config(cfg,edge["edge"])
        inputs=load_inputs(cfg,edge["edge"],limit=0,include_fit=True)
        fitting,detail=planned_fit_cost(inputs,ecfg,rows)
        source_rows=legacy["edges"][edge["edge"]-1]["map_rows"]
        cost=edge["costs"]
        for field in ("repair_flops","calibration_flops","old_boundary_bytes","selection_sort_comparisons"):
            cost[field]={k:v for k,v in cost[field].items() if not k.startswith("kv_map_")}
        for p in rows:
            if p["method"]!="kv_map":
                continue
            tag=p["tag"]
            cost["repair_flops"][tag]=deployment_flops(source_rows,p["width"])
            cost["calibration_flops"][tag]=fitting[tag]
            cost["old_boundary_bytes"][tag]=0
            cost["selection_sort_comparisons"][tag]=0
        edge["kv_map_fit_audit"]=detail
        edge["kv_map_deployment_rows"]=source_rows
        edge["prerelease_fitting_input_fingerprint"]=inputs["fingerprint"]
        percents={tag:100*(cost["repair_flops"][tag]+value)/cost["once_rebuild_flops"] for tag,value in fitting.items()}
        assert all(5<=v<=30 for v in percents.values()),percents
        print(json.dumps(dict(edge=edge["edge"],cost_flops_percent=percents)),flush=True)
        del inputs
        gc.collect()
    record.update(status="complete",config_sha256=digest(HERE/"config.json"),calibration_arithmetic_version=3,
        cost_axis="Complete added analytical FLOPs / once-per-user Rebuild FLOPs * 100; no wall time inputs",
        revision="Retained identical full-population event ledger; freshly derived prerelease fit row counts; actual 8-epoch MLP forward/backward/AdamW and standalone deployment mapping.")
    write_json(HERE/"planned_costs.json",record)

def main():
    cfg=configuration()
    rows=policies(cfg)
    records=[]
    started=time.perf_counter()
    for edge in cfg["edges"]:
        ecfg=edge_config(cfg,edge)
        inputs=load_inputs(cfg,edge,include_fit=True)
        assert len(inputs["quality"])==75807
        assert inputs["requests"]==[345630,318504,323362,335971,347575][edge-1]
        workload=offline_shapes(inputs,ecfg)
        fitting,detail=planned_fit_cost(inputs,ecfg,rows)
        cost=ledger(inputs,ecfg,rows,fitting)
        record=dict(edge=edge,window=[ecfg["release_day"],ecfg["end_day"]],
            window_name="E14_partial" if edge==5 else "E14",population=75807,requests=inputs["requests"],
            input_fingerprint=inputs["fingerprint"],input_load_seconds=inputs["load_seconds"],
            offline_workload=workload,costs=cost,kv_map_fit_audit=detail)
        records.append(record)
        write_json(HERE/"planned_costs.json",dict(status="running",edges=records))
        print(json.dumps(dict(stage="input_coverage_passed",edge=edge,requests=inputs["requests"],
            seconds=inputs["load_seconds"],workload=workload)),flush=True)
        del inputs
        gc.collect()
    # Timing only: synthetic scalar arrays at the true full-edge dimensions.
    # They are never reported as experiment quality and are never persisted.
    from hstu_kvcache.evaluation.statistics import weighted_auc
    rng=np.random.default_rng(17)
    requests=345630
    labels=rng.integers(2,size=requests).astype(np.float64)
    users=np.sort(rng.integers(34190,size=requests))
    score=rng.normal(size=requests)
    weights=np.stack([np.bincount(rng.integers(75807,size=75807),minlength=75807) for _ in range(100)]).astype(np.uint16)
    begin=time.perf_counter()
    weighted_auc(score,labels,users,weights)
    statistic_seconds=time.perf_counter()-begin
    stats=dict(scope="Synthetic arrays for CPU timing only, not quality evidence",requests=requests,
        measured_bootstrap_draws=100,measured_seconds=statistic_seconds,
        projected_1000_draws_29_points_5_edges_seconds=statistic_seconds*10*29*5)
    # Time a realistic per-rank rolling numeric checkpoint, then remove it.
    from experiments.figure9a.common import Rolling
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
    kv_only() if "--kv-only" in sys.argv else main()
