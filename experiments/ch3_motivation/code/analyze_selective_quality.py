#!/usr/bin/env python3
"""Audit and summarize both complete, freshly measured E3 2k panels on CPU."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from experiments.ch3_motivation.code import common, run_selective, run_selective_population
from experiments.ch3_motivation.code.audit_scenario_qualification import audit_blocks
from experiments.ch3_motivation.code.scenario_inputs import load_scenario_population
from hstu_kvcache.evaluation.statistics import weighted_auc
from hstu_kvcache.common.artifacts import sha256


def _key(row):
    return tuple(row[name] for name in ("method","uid","query_timestamp","raw_item_id","item_idx","label"))


def _groups(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    for group in groups.values():
        group.sort(key=lambda row:_key(row)[1:])
    return groups


def compare_fresh_anchors(rows,anchors,users,*,tolerance=1e-4):
    """Numerical comparison only; no small-population quality metric is made."""
    selected=set(users)
    measured={_key(row):row["score"] for row in rows if row["method"] in ("reuse","rebuild")}
    reference={_key(row):row["score"] for row in anchors if row["uid"] in selected}
    count=sum(row["method"] in ("reuse","rebuild") for row in rows)
    reference_count=sum(row["uid"] in selected for row in anchors)
    if len(measured)!=count or len(reference)!=reference_count or measured.keys()!=reference.keys():
        raise ValueError("fresh E3/qualification anchor event coverage differs")
    error=max((abs(value-reference[key]) for key,value in measured.items()),default=0.)
    if not np.isfinite(error) or error>tolerance:
        raise ValueError("fresh E3/qualification anchor scores exceed numerical tolerance")
    return dict(scores=count,maximum_absolute_difference=error,tolerance=tolerance,passed=True,
                historical_scores_reused=False,reference="fresh three-anchor qualification in this study")


def aggregate_quality(edge_rows,users,*,replicates=1000,seed=20261028):
    """One UID-cluster draw is shared by all methods and all release edges."""
    common.quality_role(users)
    if not edge_rows:
        raise ValueError("no complete release rows supplied")
    population=np.asarray(sorted(users),dtype=np.int64)
    rng=np.random.default_rng(seed)
    weights=np.asarray([np.bincount(rng.integers(len(users),size=len(users)),minlength=len(users))
                        for _ in range(replicates)],dtype=np.uint16)
    methods=list(_groups(edge_rows[0]))
    if not {"reuse","rebuild"}<=set(methods):
        raise ValueError("quality aggregation requires both new anchors")
    points={tag:[] for tag in methods}
    draws={tag:[] for tag in methods}
    for rows in edge_rows:
        groups=_groups(rows)
        if set(groups)!=set(methods):
            raise ValueError("release method coverage differs")
        base=groups["reuse"]
        keys=[_key(row)[1:] for row in base]
        uids=np.asarray([row["uid"] for row in base],dtype=np.int64)
        index=np.searchsorted(population,uids)
        if np.any(index>=len(population)) or not np.array_equal(population[index],uids):
            raise ValueError("score UID is outside the frozen panel")
        labels=np.asarray([row["label"] for row in base],dtype=np.float64)
        for tag,group in groups.items():
            if [_key(row)[1:] for row in group]!=keys:
                raise ValueError("paired method feedback coverage differs")
            values=np.asarray([row["score"] for row in group],dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError("nonfinite quality score")
            points[tag].append(float(roc_auc_score(labels,values)))
            draws[tag].append(weighted_auc(values,labels,index,weights))
    mean_points={tag:float(np.mean(value)) for tag,value in points.items()}
    mean_draws={tag:np.mean(value,axis=0) for tag,value in draws.items()}
    gap=mean_points["rebuild"]-mean_points["reuse"]
    gap_draws=mean_draws["rebuild"]-mean_draws["reuse"]
    valid=np.isfinite(gap_draws)&(gap_draws>0)
    def interval(value):
        finite=np.asarray(value)[np.isfinite(value)]
        return np.percentile(finite,[2.5,97.5]).tolist() if len(finite) else None
    results=[]
    for tag in methods:
        gain=mean_points[tag]-mean_points["reuse"]
        gain_draws=mean_draws[tag]-mean_draws["reuse"]
        results.append(dict(method=tag,mean_auc=mean_points[tag],mean_auc_ci95=interval(mean_draws[tag]),
            mean_auc_gain_over_reuse=gain,mean_auc_gain_over_reuse_ci95=interval(gain_draws),
            mean_auc_minus_rebuild=mean_points[tag]-mean_points["rebuild"],
            mean_auc_minus_rebuild_ci95=interval(mean_draws[tag]-mean_draws["rebuild"]),
            recovery_percent=100*gain/gap if gap else None,
            recovery_percent_ci95=interval(100*gain_draws/gap_draws) if valid.all() else None,
            per_edge_auc=points[tag]))
    return dict(quality_users=len(users),edges=len(edge_rows),mean_reference_gap=gap,
        mean_reference_gap_ci95=interval(gap_draws),results=results,
        confidence=dict(replicates=replicates,seed=seed,shared_uid_multiplicity_across_releases=True,
            all_selected_users_including_no_feedback=True,
            nonpositive_or_invalid_reference_gap_replicates=int((~valid).sum()),
            method="paired UID-cluster percentile bootstrap; same draw for all releases and methods",
            scope="conditional on the frozen fitting users and weights; not training-seed variability"),
        quality_aggregation="equal release mean of pooled raw-feedback AUCs; recovery is ratio of mean gains, never mean release recovery",
        recovery_rule="unclipped; no bounded recovery interval if any aggregate bootstrap reference gap is nonpositive")


def aggregate_costs(edge_costs,methods):
    """Add complete FLOPs before dividing, independently in each event scenario."""
    result={}
    for name in ("actual_event_ledger","candidate_reads_230"):
        ledgers=[value[name] for value in edge_costs]
        denominator=sum(int(value["once_rebuild_flops"]) for value in ledgers)
        if denominator<=0:
            raise ValueError("nonpositive summed Rebuild cost")
        groups=[]
        for ledger in ledgers:
            group={row["method"]:row for row in ledger["results"]}
            if set(group)!=set(methods) or len(group)!=len(ledger["results"]):
                raise ValueError("cost ledger method coverage differs")
            groups.append(group)
        rows=[]
        for tag in methods:
            added=sum(int(value[tag]["added_flops"]) for value in groups)
            row=dict(method=tag,added_flops=added,percent_of_once_rebuild=100*added/denominator)
            for field in ("calibration_flops","selection_and_repair_flops","extra_read_flops"):
                if all(field in value[tag] for value in groups):
                    row[field]=sum(int(value[tag][field]) for value in groups)
            rows.append(row)
        common_fields=set.intersection(*(set(value) for value in ledgers))
        result[name]=dict(once_rebuild_flops=denominator,results=rows,
            **{key:sum(int(value[key]) for value in ledgers) for key in common_fields if key.startswith("common_")},
            aggregation="sum method FLOPs / sum once-Rebuild FLOPs; complete independent edge ledger retained")
    return result


def _audit_edge(folder,manifest_path,edge,code,anchor_cache):
    binding,users,population,provenance=load_scenario_population(manifest_path,edge,minimum_users=2000,require_positive=False)
    if len(users)!=2000:
        raise ValueError("E3 main/confirmation requires the fixed 2000-user panel")
    identity=json.loads((folder/"identity.json").read_text())
    saved=json.loads((folder/"summary.json").read_text())
    status=json.loads((folder/"status.json").read_text())
    if (status["state"]!="complete" or status["mode"]!="run" or saved["identity"]!=identity
            or identity["source_sha256"]!=code or identity["settings"]!=binding["settings"]
            or identity["input_binding_sha256"]!=provenance["input_binding_sha256"]
            or identity["checkpoint_sha256"]!=provenance["anchor_models"]
            or identity["evaluation_users_sha256"]!=run_selective._hash_json(users)
            or identity["calibration_users_sha256"]!=run_selective._hash_json(population["fit"][:binding["settings"]["profile_users"]])):
        raise ValueError("E3 result/input/source identity differs")
    policies=json.loads((folder/"policies.json").read_text())["policies"]
    methods=[policy["tag"] for policy in policies]
    if (len(methods)!=24 or len(set(methods))!=24 or len(saved["results"])!=24
            or set(methods)!={row["method"] for row in saved["results"]}
            or saved["confidence"]["replicates"]!=1000):
        raise ValueError("E3 complete 24-point coverage differs")
    profile=folder/"layer_profile.json"
    fitted=json.loads(profile.read_text())
    if fitted["fitting_uids"]!=population["fit"][:binding["settings"]["profile_users"]]:
        raise ValueError("E3 interval selection used another calibration population")
    for policy in policies:
        result=next(value for value in saved["results"] if value["method"]==policy["tag"])
        setting={key:policy[key] for key in ("interval","layers","fraction") if key in policy}
        if result["setting"]!=setting or result["calibration_flops"]!=policy["calibration_flops"]:
            raise ValueError("E3 summary policy/calibration setting differs")
        if policy["method"]=="layer" and (policy["interval"]!=fitted["selected"][str(policy["layers"])]
                or policy["calibration_flops"]!=fitted["standalone_calibration_flops_by_width"][str(policy["layers"])]):
            raise ValueError("E3 Layer setting differs from the exact interval profile")
    selected=set(users)
    requests=pq.read_table(provenance["requests_path"]).to_pylist()
    events=[row for row in requests if row["uid"] in selected]
    rows,block_audit=audit_blocks(folder/"quality_shards",users,
        dict(identity,layer_profile_sha256=sha256(profile)),methods,events,
        block_size=binding["settings"]["evaluation_checkpoint_users"])
    costs=json.loads((folder/"costs.json").read_text())
    if costs!=saved["costs"] or saved["quality_users"]!=len(users) or saved["feedback_events"]!=len(events):
        raise ValueError("E3 summary population/cost differs")
    actual_cost={value["method"]:value for value in costs["actual_event_ledger"]["results"]}
    normalized_cost={value["method"]:value for value in costs["candidate_reads_230"]["results"]}
    for result in saved["results"]:
        tag=result["method"]
        if (any(result[key]!=actual_cost[tag][key] for key in ("added_flops","calibration_flops","selection_and_repair_flops","percent_of_once_rebuild"))
                or result["normalized_230_percent_of_once_rebuild"]!=normalized_cost[tag]["percent_of_once_rebuild"]):
            raise ValueError("E3 point summary differs from its complete cost ledgers")
    for tag,group in _groups(rows).items():
        result=next(value for value in saved["results"] if value["method"]==tag)
        auc=float(roc_auc_score([row["label"] for row in group],[row["score"] for row in group]))
        if abs(auc-result["auc"])>1e-14:
            raise ValueError("E3 summary AUC differs from raw scores")
    if edge not in anchor_cache:
        qualifier=Path(manifest_path).parents[1]
        qidentity=json.loads((qualifier/"identity.json").read_text())
        union=sorted({uid for value in qidentity["panels"].values() for uid in value})
        union_set=set(union)
        qevents=[row for row in requests if row["uid"] in union_set]
        anchor_cache[edge],_=audit_blocks(qualifier/"union_current_anchors/quality_shards",union,
            dict(qidentity,union=union,branch="current_reuse_and_once_rebuild"),["reuse","rebuild"],qevents)
    numerical=compare_fresh_anchors(rows,anchor_cache[edge],users)
    return users,rows,costs,dict(edge=edge,quality_users=len(users),feedback_users=saved["users_with_feedback"],
        feedback_events=len(events),evaluation_window=saved["evaluation_window"],results=saved["results"],
        summary_path=str((folder/"summary.json").resolve()),summary_sha256=sha256(folder/"summary.json"),
        costs_path=str((folder/"costs.json").resolve()),costs_sha256=sha256(folder/"costs.json"),
        profile_sha256=sha256(profile),identity_sha256=sha256(folder/"identity.json"),
        block_audit=block_audit,fresh_anchor_comparison=numerical)


def _write_csv(path,rows,fields):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(run_root,qualification_root,output,*,attempt="attempt_1"):
    run_root,qualification_root,output=Path(run_root),Path(qualification_root),Path(output)
    folders={(panel,edge):run_root/f"edge_{edge}"/panel/attempt for panel in ("main","confirmation") for edge in range(1,6)}
    missing=[str(folder/name) for folder in folders.values() for name in ("summary.json","status.json") if not (folder/name).exists()]
    if missing:
        raise ValueError("all ten E3 runs must finish before quality analysis; missing "+", ".join(missing))
    if any(json.loads((folder/"status.json").read_text())["state"]!="complete" for folder in folders.values()):
        raise ValueError("all ten E3 runs must have complete status before quality analysis")
    code=run_selective_population.sources()
    anchor_cache={}
    panels={}
    edge_csv=[]
    aggregate_csv=[]
    for panel in ("main","confirmation"):
        all_rows,all_costs,edges=[],[],[]
        common_users=None
        for edge in range(1,6):
            users,rows,costs,record=_audit_edge(folders[panel,edge],qualification_root/f"edge_{edge}"/panel/"scenario_manifest.json",edge,code,anchor_cache)
            if common_users is not None and users!=common_users:
                raise ValueError("cross-release panel UID membership/order changed")
            common_users=users
            all_rows.append(rows);all_costs.append(costs);edges.append(record)
            for result in record["results"]:
                row=dict(panel=panel,edge=edge,evaluation_window=record["evaluation_window"]["name"],
                    method=result["method"],setting=json.dumps(result["setting"],sort_keys=True),auc=result["auc"],
                    auc_gain_over_reuse=result["auc_gain_over_reuse"],recovery_percent=result["recovery_percent"],
                    added_flops=result["added_flops"],percent_of_once_rebuild=result["percent_of_once_rebuild"],
                    normalized_230_percent_of_once_rebuild=result["normalized_230_percent_of_once_rebuild"])
                for field in ("auc_ci95","auc_gain_over_reuse_ci95","recovery_percent_ci95"):
                    row[field+"_low"],row[field+"_high"]=result[field] or [None,None]
                edge_csv.append(row)
        quality=aggregate_quality(all_rows,common_users,replicates=1000,seed=20261028+(panel=="confirmation"))
        methods=[row["method"] for row in quality["results"]]
        costs=aggregate_costs(all_costs,methods)
        panels[panel]=dict(**quality,costs=costs,edge_records=edges,
            quality_uids_sha256=hashlib.sha256(json.dumps(common_users).encode()).hexdigest())
        for result in quality["results"]:
            row=dict(panel=panel,method=result["method"],mean_auc=result["mean_auc"],
                mean_auc_gain_over_reuse=result["mean_auc_gain_over_reuse"],mean_auc_minus_rebuild=result["mean_auc_minus_rebuild"],
                recovery_percent=result["recovery_percent"])
            for field in ("mean_auc_ci95","mean_auc_gain_over_reuse_ci95","mean_auc_minus_rebuild_ci95","recovery_percent_ci95"):
                row[field+"_low"],row[field+"_high"]=result[field] or [None,None]
            for name,prefix in (("actual_event_ledger","actual"),("candidate_reads_230","normalized_230")):
                cost=next(value for value in costs[name]["results"] if value["method"]==result["method"])
                row[prefix+"_added_flops"]=cost["added_flops"]
                row[prefix+"_percent_of_once_rebuild"]=cost["percent_of_once_rebuild"]
            aggregate_csv.append(row)
        del all_rows
    if run_selective_population.sources()!=code:
        raise ValueError("E3 numerical sources changed during analysis")
    output.mkdir(parents=True,exist_ok=True)
    result=dict(state="complete",experiment="E3 fresh main and independent confirmation",panels=panels,
        audit=dict(quality_runs=10,points=240,blocks=sum(edge["block_audit"]["blocks"] for value in panels.values() for edge in value["edge_records"]),
            all_hashes_uids_events_sources_passed=True,anchor_tolerance=1e-4),
        source_sha256=dict(code,**{str(Path(__file__).resolve().relative_to(ROOT)):sha256(Path(__file__))}),
        selection="fixed main and confirmation roles; all five release edges retained irrespective of method outcomes",
        scope="measured partial populations; E14_partial release remains labelled; persistent repair only; request-local bridge is separate")
    common.write_json(output/"summary.json",result)
    _write_csv(output/"metrics_by_edge.csv",edge_csv,list(edge_csv[0]))
    _write_csv(output/"metrics_by_panel.csv",aggregate_csv,list(aggregate_csv[0]))
    print(json.dumps(dict(summary=str(output/"summary.json"),**result["audit"])),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,default=ROOT/"experiments/ch3_motivation/runs/e3")
    parser.add_argument("--qualification-root",type=Path,default=ROOT/"experiments/ch3_motivation/runs/scenario_qualification/attempt_1")
    parser.add_argument("--attempt",default="attempt_1")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    analyze(args.run_root,args.qualification_root,args.output,attempt=args.attempt)
