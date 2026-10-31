#!/usr/bin/env python3
"""Bridge persistent and request-local selective repair on frozen partial users.

This is a separate rule-sensitivity study, not a replacement of E3's main
once-per-release protocol. The same immutable fitting snapshots support the
8/16/32/64-user selector comparison; held-out AUC never selects intervals.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from experiments.ch3_motivation.code import common, selective
from experiments.ch3_motivation.code.run_selective import compare_scores, sources as selective_sources
from hstu_kvcache.evaluation.baselines import event_shapes
from hstu_kvcache.evaluation.statistics import paired_user_bootstrap
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.common.artifacts import sha256


def sources():
    return {**selective_sources(), "experiments/ch3_motivation/code/run_rule_sensitivity.py": sha256(Path(__file__))}


def request_local_costs(policies, request_shapes, shapes, cm, cfg):
    """Declare both possible per-request Full comparators, without once costs.

    The shared reference maintains Reuse for every rule, including Full's raw
    history view. A standalone per-request Full can omit those native writes;
    its Full-Reuse difference is also shown, with undefined ratios explicit.
    """
    requests = Counter({tuple(row["shape"]): row["count"] for row in request_shapes})
    local = dict(shapes, requests=requests)
    full_repair = selective.repair_cost(dict(method="rebuild"), local, cm, cfg,
                                       schedule="request_local")[0]
    native_appends = sum(count*selective._cost_for_length(cm,n,cfg).band_append(
        n,width,window_size=cfg["history_length"])
        for (n,width,_),count in shapes["appends"].items())
    replacement_gap = full_repair-native_appends
    rows = []
    for policy in policies:
        added, sorts = selective.repair_cost(policy,local,cm,cfg,schedule="request_local")
        # Rebuild's standalone implementation needs only raw history, not the
        # common native K/V writes generated for the selective policies.
        standalone = added-native_appends if policy["method"]=="rebuild" else added
        rows.append(dict(method=policy["tag"], calibration_flops=int(policy.get("calibration_flops",0)),
            actual_repair_and_selection_flops=added-int(policy.get("calibration_flops",0)),
            shared_reference_added_flops=added,
            percent_of_per_request_full_repair=100*added/full_repair if full_repair else None,
            standalone_added_over_reuse_flops=standalone,
            percent_of_standalone_full_minus_reuse=100*standalone/replacement_gap if replacement_gap>0 else None,
            sorting_comparisons_estimate=sorts))
    return dict(per_request_full_repair_flops=full_repair,
        common_native_append_flops=native_appends,
        standalone_full_minus_reuse_flops=replacement_gap,
        standalone_denominator_positive=replacement_gap>0,
        request_shapes=request_shapes,results=rows,
        primary_axis="shared reference's added repair FLOPs / per-request Full repair FLOPs",
        standalone_axis="per-request Full replaces native K/V writes; signed Full-Reuse difference shown separately",
        scope="actual E14 requests and complete chronological append tail; once-per-release Rebuild is not a denominator")


def merge_reference_parts(parts):
    observed = defaultdict(Counter)
    requests = Counter()
    for part in parts:
        for tag, counts in part["observed"].items():
            observed[tag].update(counts)
        for row in part["request_shapes"]:
            requests[tuple(row["shape"])]+=row["count"]
    return dict(observed={tag:dict(counts) for tag,counts in observed.items()},
                request_shapes=[dict(shape=list(shape),count=count) for shape,count in sorted(requests.items())])


def selector_comparison(current, samples, histories, fit_users, validation_users, cfg, device):
    """Fitting-prefix selection with an independent validation-MSE comparison."""
    validation, _ = selective.profile_intervals(current,samples,histories,validation_users,cfg,device)
    val_mse = {tuple(row["interval"]):row["mse"] for row in validation["intervals"]}
    fits = {}
    for count in (8,16,32,64):
        profile,_=selective.profile_intervals(current,samples,histories,fit_users[:count],cfg,device)
        fits[str(count)]=profile
    records=[]
    for count,profile in fits.items():
        for width,interval in profile["selected"].items():
            records.append(dict(fit_users=int(count),layers=int(width),interval=interval,
                validation_mse=val_mse[tuple(interval)],
                validation_best_mse=min(value for (start,end),value in val_mse.items() if end-start+1==int(width)),
                same_interval_as_64=interval==fits["64"]["selected"][width],
                standalone_calibration_flops=profile["standalone_calibration_flops_by_width"][width]))
    return dict(fit_profiles=fits,validation_profile=validation,comparison=records,
        fit_users=[int(uid) for uid in fit_users[:64]],validation_users=[int(uid) for uid in validation_users],
        selection_scope="same nested fitting prefixes; validation teacher-logit MSE only; no evaluation feedback outcomes")


def policies_for_bridge(cm, comparison, cfg, profile_users, *, include_token_rules=True):
    profile=comparison["fit_profiles"][str(profile_users)]
    policies=[dict(tag="reuse",method="reuse",calibration_flops=0),
              dict(tag="rebuild",method="rebuild",calibration_flops=0)]
    for width in cfg["partial_points"]["layer"]:
        policies.append(dict(tag=f"layer_{width}layers",method="layer",layers=int(width),
            interval=tuple(profile["selected"][str(width)]),
            calibration_flops=profile["standalone_calibration_flops_by_width"][str(width)]))
    if include_token_rules:
        policies.extend(dict(tag=f"{method}_20pct",method=method,fraction=.2,calibration_flops=0)
                        for method in ("tail","deviation"))
    return policies


def evaluate_blocks(parent,current,histories,by_user,users,policies,cfg,device,output,identity):
    folder=output/"score_blocks"
    folder.mkdir(exist_ok=True)
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    scores,parts=[],[]
    for offset in range(0,len(users),256):
        selected=users[offset:offset+256]
        path=folder/f"part_{offset:06d}.parquet"
        marker=folder/f"part_{offset:06d}.json"
        if marker.exists():
            saved=json.loads(marker.read_text())
            if saved["fingerprint"]!=fingerprint or saved["users"]!=selected or saved["scores_sha256"]!=sha256(path):
                raise ValueError("rule sensitivity block identity changed")
            rows=pq.read_table(path).to_pylist()
            execution=saved["execution"]
        else:
            rows,execution=selective.evaluate_reference(parent,current,histories,by_user,selected,
                policies,cfg,device,semantics="request_local")
            schema=pa.schema([("uid",pa.uint64()),("query_timestamp",pa.int64()),("raw_item_id",pa.uint64()),
                ("item_idx",pa.int64()),("label",pa.int64()),("method",pa.string()),("score",pa.float64())])
            temporary=path.with_suffix(".parquet.writing")
            pq.write_table(pa.Table.from_pylist(rows,schema=schema),temporary)
            temporary.replace(path)
            common.write_json(marker,dict(fingerprint=fingerprint,users=selected,scores_sha256=sha256(path),execution=execution))
        scores.extend(rows)
        parts.append(execution)
        print(json.dumps(dict(stage="request_local_bridge",users=offset+len(selected),total=len(users),scores=len(scores))),flush=True)
    return scores,merge_reference_parts(parts)


def summarize(scores,users,policies,execution,costs,cfg,identity,output):
    common.quality_role(users)
    from sklearn.metrics import roc_auc_score
    groups=defaultdict(list)
    for row in scores:
        groups[row["method"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row:(row["uid"],row["query_timestamp"],row["raw_item_id"]))
    intervals,confidence=paired_user_bootstrap(groups,users,replicates=1000,seed=27121+cfg["edge"])
    aucs={tag:float(roc_auc_score([row["label"] for row in rows],[row["score"] for row in rows])) for tag,rows in groups.items()}
    gap=aucs["rebuild"]-aucs["reuse"]
    fees={row["method"]:row for row in costs["results"]}
    results=[]
    for policy in policies:
        tag=policy["tag"]
        results.append(dict(method=tag,auc=aucs[tag],auc_gain_over_reuse=aucs[tag]-aucs["reuse"],
            recovery_percent=100*(aucs[tag]-aucs["reuse"])/gap if gap else None,
            **intervals[tag],**{key:value for key,value in fees[tag].items() if key!="method"},
            setting={key:policy[key] for key in ("interval","layers","fraction") if key in policy}))
    value=dict(experiment="E3 rule-lifetime sensitivity",state="complete",edge=cfg["edge"],
        quality_users=len(users),feedback_users=len({row["uid"] for row in groups["reuse"]}),feedback_events=len(groups["reuse"]),
        results=results,reference_gap=gap,confidence=confidence,execution=execution,costs=costs,identity=identity,
        evaluation_window=cfg.get("evaluation_window",dict(name="E14",complete_source_horizon=True)),
        semantics="each request repairs common rolling Reuse state; all repaired views are discarded; Full is per-request",
        interpretation="separate bridge to historical rule semantics; does not replace persistent E3 main results")
    common.write_json(output/"summary.json",value)
    fields=["method","auc","auc_gain_over_reuse","recovery_percent","shared_reference_added_flops",
        "percent_of_per_request_full_repair","standalone_added_over_reuse_flops","percent_of_standalone_full_minus_reuse"]
    with (output/"metrics_by_point.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(results)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge",type=int,choices=range(1,6),required=True)
    parser.add_argument("--panel",choices=("development","main","confirmation"),default="development")
    parser.add_argument("--input-root",type=Path,default=ROOT/"experiments/ch3_motivation/inputs")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--profile-users",type=int,choices=(8,16,32,64),default=64)
    parser.add_argument("--diagnostic-users",type=int,default=8)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--resume",action="store_true")
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--run",action="store_true")
    args=parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    population_path=args.input_root/"population.json"
    population=json.loads(population_path.read_text())
    users=population[args.panel]
    if args.check:
        if not 8<=args.diagnostic_users<=16:
            parser.error("check uses 8--16 numerical/performance users")
        users=users[:args.diagnostic_users]
    common.quality_role(users,diagnostic=args.check)
    folder=args.input_root/f"edge_{args.edge}"
    binding,histories,by_user=common.load_edge(folder)
    for name in ("history","requests"):
        if sha256(folder/binding[name]["path"])!=binding[name]["sha256"]:
            raise ValueError(f"frozen {name} pack changed")
    if binding["population_sha256"]!=sha256(population_path):
        raise ValueError("population identity changed")
    cfg=binding["settings"]
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    code=sources()
    device=torch.device("cuda:0")
    parent,current,models=common.load_models(args.edge,device)
    identity=dict(edge=args.edge,panel=args.panel,users=users,profile_users=args.profile_users,
        settings=cfg,models=models,sources=code,population_sha256=sha256(population_path),
        input_binding_sha256=sha256(folder/"inputs.json"),semantics="request_local")
    path=output/"identity.json"
    if path.exists() and json.loads(path.read_text())!=identity:
        raise ValueError("existing attempt identity changed; use a new output directory")
    if any((output/"score_blocks").glob("part_*.json")) and not args.resume:
        raise ValueError("existing score blocks require explicit --resume with unchanged identity")
    common.write_json(path,identity)
    started=time.perf_counter()
    selection_path=output/"selector_sensitivity.json"
    if args.resume and selection_path.exists():
        comparison=json.loads(selection_path.read_text())
    else:
        fit=population["fit"][:64]
        validation=population["validation"]
        samples=common.capture_samples(parent,current,histories,fit+validation,cfg,device,boundaries=True)
        comparison=selector_comparison(current,samples,histories,fit,validation,cfg,device)
        del samples
        common.write_json(selection_path,comparison)
    cm=CostModel(current.cfg.hidden_size,len(current.blocks),current.cfg.num_heads,cfg["attention_backend"])
    policies=policies_for_bridge(cm,comparison,cfg,args.profile_users)
    if args.check:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter()
        full_layer=dict(tag="layer_full_endpoint",method="layer",interval=(0,len(current.blocks)-1),calibration_flops=0)
        scores,execution=selective.evaluate_reference(parent,current,histories,by_user,users,
            policies+[full_layer],cfg,device,semantics="request_local")
        torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-start
        expected=[row for row in scores if row["method"]=="rebuild"]
        actual=[dict(row,method="rebuild") for row in scores if row["method"]=="layer_full_endpoint"]
        full_check=compare_scores(expected,actual)
        persistent,_=selective.evaluate_reference(parent,current,histories,by_user,users,
            policies[:1],cfg,device,semantics="persistent")
        reuse_check=compare_scores(persistent,[row for row in scores if row["method"]=="reuse"])
        result=dict(role="numerical_performance_diagnostic",quality_metrics_computed=False,users=len(users),
            replay_seconds=elapsed,seconds_per_user=elapsed/len(users),
            projected_512_replay_seconds=elapsed/len(users)*512,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),full_endpoint_check=full_check,reuse_check=reuse_check,
            execution=execution,identity=identity,total_seconds_excluding_model_input_loading=time.perf_counter()-started)
        common.write_json(output/"check.json",result)
        print(json.dumps({key:result[key] for key in ("users","replay_seconds","projected_512_replay_seconds","quality_metrics_computed")}),flush=True)
    else:
        scores,execution=evaluate_blocks(parent,current,histories,by_user,users,policies,cfg,device,output,
            dict(identity,selector_sensitivity_sha256=sha256(selection_path)))
        costs=request_local_costs(policies,execution["request_shapes"],event_shapes(histories,by_user,users,cfg),cm,cfg)
        summarize(scores,users,policies,execution,costs,cfg,identity,output)
    if sources()!=code:
        raise ValueError("sources changed during execution")
    common.write_json(output/"status.json",dict(state="complete",mode="check" if args.check else "run",
        users=len(users),seconds_excluding_model_input_loading=time.perf_counter()-started))


if __name__=="__main__":
    main()
