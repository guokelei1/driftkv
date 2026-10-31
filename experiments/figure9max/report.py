#!/usr/bin/env python3
"""Rebuild the compact readiness report from retained timing/input evidence."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import json
import numpy as np
from scipy.optimize import nnls
from experiments.figure9max.common import HERE, configuration, digest, policies, write_json

def main():
    cfg=configuration()
    planned=json.loads((HERE/"planned_costs.json").read_text())
    assert planned["status"]=="complete"
    assert planned["config_sha256"]==digest(HERE/"config.json")
    final=HERE/"runs/benchmark"
    benchmark_cfg=json.loads((final/"configuration.json").read_text())
    # Only the full-population assertion was corrected after the probe:
    # the old manifest counted fit/validation feedback as well as quality.
    # This field is never read by benchmark numerical execution.
    differences=[k for k in cfg if cfg[k]!=benchmark_cfg.get(k)]
    assert set(differences)<= {"expected_requests"}, differences
    summaries=[]
    correctness=[]
    for rank in range(4):
        folder=final/f"gpu{rank}"
        summary=json.loads((folder/"benchmark.json").read_text())
        check=json.loads((folder/"correctness.json").read_text())
        assert summary["status"]=="complete" and check["status"]=="passed"
        assert summary["config_sha256"]==digest(final/"configuration.json")
        expected=planned["edges"][summary["edge"]-1]["costs"]["calibration_flops"]
        assert summary["calibration_flops"]==expected
        summaries.append(summary);correctness.append(check)
    cases=[s["cases"][0] for s in summaries]
    # Positive work coefficients; this is a scheduling estimate, not a claim
    # of hardware FLOP/s. Include both event-shape and request-based views.
    estimates=[]
    for work_key in ("query_groups","requests"):
        x=np.array([[c["counts"]["users"],c["counts"]["append_groups"],c["counts"][work_key]] for c in cases],dtype=float)
        y=np.array([c["steady_seconds"] for c in cases])
        scales=np.maximum(x.mean(0),1)
        coef,_=nnls(x/scales,y)
        coef=coef/scales
        projected=[]
        for edge in planned["edges"]:
            work=edge["offline_workload"]
            feature=[edge["population"],work["append_bands"],work["query_batches" if work_key=="query_groups" else "feedback_requests"]]
            projected.append(float(np.dot(feature,coef)/4))
        estimates.append(dict(features=["eligible_users","append_bands",work_key],coefficients=coef.tolist(),
            fitted_sample_seconds=(x@coef).tolist(),gpu_seconds_by_edge=projected,total_gpu_seconds=sum(projected)))
    center=float(np.mean([e["total_gpu_seconds"] for e in estimates]))
    statistics=planned["statistics_timing"]["projected_1000_draws_29_points_5_edges_seconds"]
    # Independent edge statistics overlap the following edge's GPU evaluation.
    setup=sum(e["input_load_seconds"] for e in planned["edges"])+sum(s["fit_seconds"] for s in summaries)/4*5
    total=center+setup+statistics/5
    comparison=[]
    previous=HERE/"nonexistent_legacy_summary.json"
    if previous.exists():
        for worker in json.loads(previous.read_text())["workers"]:
            comparison.append(dict(edge=worker["edge"],cases=worker["cases"]))
    kv_points=[]
    for p in policies(cfg):
        if p["method"]!="kv_map":
            continue
        ratios=[]
        phases=[]
        for edge in planned["edges"]:
            cost=edge["costs"];tag=p["tag"];den=cost["once_rebuild_flops"]
            audit=edge["kv_map_fit_audit"][tag]
            deployed=cost["repair_flops"][tag]
            ratios.append(100*(audit["total_flops"]+deployed)/den)
            teacher=audit["parts"]["teacher_source_preparation"]
            phases.append(dict(edge=edge["edge"],teacher_preparation_flops=teacher,
                fitting_flops=audit["total_flops"]-teacher,deployment_flops=deployed,
                complete_added_flops=audit["total_flops"]+deployed,rebuild_flops=den,
                steps=audit["steps"],training_rows=audit["training_rows"]))
        kv_points.append(dict(**p,mean_cost_pct=float(np.mean(ratios)),cost_pct_by_edge=ratios,
            min_cost_pct=min(ratios),max_cost_pct=max(ratios),flop_breakdown_by_edge=phases))
    report=dict(status="ready_authorized",formal_started=False,configuration="config.json",
        points_per_edge=len(policies(cfg)),edges=5,total_method_edge_points=5*len(policies(cfg)),
        total_population_edge_users=cfg["expected_quality_users"]*5,total_feedback_events=sum(e["requests"] for e in planned["edges"]),
        selected_user_batch=1,parallelism="4 independent A40 UID shards per edge; CPU UID statistics overlap next edge",
        benchmark_users_per_gpu=cases[0]["counts"]["users"],benchmark_total_users=sum(c["counts"]["users"] for c in cases),
        benchmark_cases=cases,maximum_score_error=max(c["max_score_error"] for c in correctness),
        benchmark_configuration_differences=differences,configuration_difference_note="Quality-only coverage assertion corrected; numerical settings and benchmark inputs unchanged. See feedback_coverage.json.",
        cost_basis="Complete added analytical FLOPs / once-per-user Rebuild FLOPs; wall times below are scheduling-only",
        kv_map_points=kv_points,kv_training=cfg["kv_training"],
        real_trajectory_numeric_resume="passed on all four GPUs",checkpoint_generations=2,checkpoint_seconds=1800,
        checkpoint_timing=planned["checkpoint_timing"],runtime_models=estimates,
        projected_gpu_hours=center/3600,projected_setup_minutes=setup/60,
        projected_statistics_all_edges_minutes=statistics/60,projected_total_hours=total/3600,
        scheduling_range_hours=[float(total/3600*.8),float(total/3600*1.4)],
        assumptions="Four A40 remain available; frozen full workload; projected from real-model probes on edges 1/2/3/5. Host statistics timing uses synthetic scalar arrays at actual dimensions. UID/workload skew, shape mix and input contention make this a scheduling estimate, not a measured full-run duration.",
        legacy_batch_comparison=comparison,
        retention="Only core scripts/configuration/input fingerprints/costs/benchmark summaries; no full-population GPU quality run or trace outputs.")
    write_json(HERE/"speed_report.json",report)
    print(json.dumps({k:report[k] for k in ("status","projected_total_hours","scheduling_range_hours","maximum_score_error")},indent=2))

if __name__=="__main__":
    main()
