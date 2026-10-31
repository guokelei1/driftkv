#!/usr/bin/env python3
"""New three-anchor measurements for Chapter 3 scenario qualification.

Every frozen population is retained. A later illustrative choice based on
these anchors is outcome-conditioned diagnostic evidence, not an unbiased
population estimate. No selective or learned-method scores are read here.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]

import numpy as np
import torch

from experiments.ch3_motivation.code import common, selective
from experiments.ch3_motivation.code.run_selective import compare_scores, sources as selective_sources
from hstu_kvcache.evaluation import fast_baselines as fast
from hstu_kvcache.evaluation.statistics import weighted_auc
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.common.artifacts import sha256


def sources():
    return {**selective_sources(),"experiments/ch3_motivation/code/qualify_scenario.py":sha256(Path(__file__))}


def frozen_panels(population):
    return dict(main=population["main"],confirmation=population["confirmation"],
        sensitivity_20261023=population["population_sensitivity"]["20261023"],
        sensitivity_20261024=population["population_sensitivity"]["20261024"])


@torch.no_grad()
def old_native_reference(parent,histories,by_user,users,cfg,device):
    """Old weights produce initial K/V, every append, and every future score."""
    output=[]
    for uid in users:
        trajectory=NativeTrajectory(parent,cfg["edge"]-1,histories.rows[uid],cfg["release_day"]*common.DAY,
            max_length=cfg["history_length"],append_band_size=cfg["append_band_size"])
        for timestamp,requests in sorted(by_user.get(uid,{}).items()):
            trajectory.advance(timestamp)
            for offset in range(0,len(requests),cfg["query_batch"]):
                rows=requests[offset:offset+cfg["query_batch"]]
                ids=torch.tensor([[row["item_idx"] for row in rows]],device=device)
                delta=torch.tensor([timestamp-trajectory.last_timestamp],device=device,dtype=torch.float32)
                values=score(parent,trajectory.cache,ids,delta)[0][0].cpu().tolist()
                output.extend(dict(row,method="old_full",score=value) for row,value in zip(rows,values,strict=True))
        trajectory.advance(cfg["end_day"]*common.DAY)
    return output


def bootstrap_anchors(scores,users,*,replicates=1000,seed=28121):
    """One common UID multiplicity draw supplies all three AUC differences."""
    common.quality_role(users)
    from sklearn.metrics import roc_auc_score
    groups=defaultdict(list)
    selected=set(users)
    for row in scores:
        if row["uid"] in selected:
            groups[row["method"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row:(row["uid"],row["query_timestamp"],row["raw_item_id"]))
    if set(groups)!={"old_full","reuse","rebuild"}:
        raise ValueError("three freshly measured anchors are required")
    key=lambda row:(row["uid"],row["query_timestamp"],row["raw_item_id"],row["label"])
    keys=[key(row) for row in groups["reuse"]]
    for tag,rows in groups.items():
        if keys!=[key(row) for row in rows]:
            raise ValueError(f"anchor event coverage differs: {tag}")
    labels=np.asarray([row["label"] for row in groups["reuse"]],dtype=np.float64)
    population=np.asarray(sorted(users),dtype=np.int64)
    row_uids=np.asarray([row["uid"] for row in groups["reuse"]],dtype=np.int64)
    index=np.searchsorted(population,row_uids)
    rng=np.random.default_rng(seed)
    weights=np.empty((replicates,len(users)),dtype=np.uint16)
    for draw in range(replicates):
        weights[draw]=np.bincount(rng.integers(len(users),size=len(users)),minlength=len(users))
    aucs={}
    draws={}
    for tag,rows in groups.items():
        values=np.asarray([row["score"] for row in rows],dtype=np.float64)
        aucs[tag]=float(roc_auc_score(labels,values))
        draws[tag]=weighted_auc(values,labels,index,weights)
    def interval(values):
        values=np.asarray(values)
        valid=values[np.isfinite(values)]
        return np.percentile(valid,[2.5,97.5]).tolist() if len(valid) else None
    update=draws["rebuild"]-draws["old_full"]
    cache=draws["rebuild"]-draws["reuse"]
    update_gain=aucs["rebuild"]-aucs["old_full"]
    cache_gap=aucs["rebuild"]-aucs["reuse"]
    update_ci,cache_ci=interval(update),interval(cache)
    both_point=update_gain>0 and cache_gap>0
    both_ci=bool(update_ci and cache_ci and update_ci[0]>0 and cache_ci[0]>0)
    return dict(quality_users=len(users),feedback_users=len(set(row_uids.tolist())),feedback_events=len(labels),
        quality_uids=[int(uid) for uid in users],
        quality_uids_sha256=hashlib.sha256(json.dumps([int(uid) for uid in users]).encode()).hexdigest(),
        auc=aucs,auc_ci95={tag:interval(value) for tag,value in draws.items()},
        new_rebuild_minus_old_full=update_gain,new_rebuild_minus_old_full_ci95=update_ci,
        new_rebuild_minus_reuse=cache_gap,new_rebuild_minus_reuse_ci95=cache_ci,
        current_reuse_minus_old_full=aucs["reuse"]-aucs["old_full"],
        current_reuse_minus_old_full_ci95=interval(draws["reuse"]-draws["old_full"]),
        release_gain_erased_percent=100*cache_gap/update_gain if update_gain>0 else None,
        release_gain_erased_ci95=interval(100*cache/update) if np.all(np.isfinite(update)&(update>0)) else None,
        point_qualifies=both_point,uid95_qualifies=both_ci,
        qualification=dict(new_rebuild_minus_old_full=update_gain,new_rebuild_minus_reuse=cache_gap,
                           point_qualifies=both_point,uid95_qualifies=both_ci),
        confidence=dict(method="paired UID-cluster percentile bootstrap, same draws for all three anchors",
            replicates=replicates,seed=seed,all_selected_users_including_no_feedback=True,
            nonpositive_update_gain_replicates=int(np.sum(~np.isfinite(update)|(update<=0))),
            nonpositive_cache_gap_replicates=int(np.sum(~np.isfinite(cache)|(cache<=0)))),
        scope="predeclared partial population comparison; any example selected from these results is outcome-conditioned diagnostic, not an unbiased total-population estimate")


def scenario_manifest(panel,users,summary,summary_path,folder,binding,population_path,by_user):
    return dict(schema_version=1,edge=binding["settings"]["edge"],uids=[int(uid) for uid in users],
        roles=dict(quality=[int(uid) for uid in users],e6=[int(uid) for uid in users if by_user.get(uid)]),
        qualification=dict(summary_path=str(summary_path.resolve()),summary_sha256=sha256(summary_path),
            **{key:summary[key] for key in ("new_rebuild_minus_old_full","new_rebuild_minus_reuse","point_qualifies","uid95_qualifies")},
            scope=summary["scope"]),
        history=dict(path=str((folder/binding["history"]["path"]).resolve()),sha256=binding["history"]["sha256"]),
        requests=dict(path=str((folder/binding["requests"]["path"]).resolve()),sha256=binding["requests"]["sha256"]),
        input_binding=dict(path=str((folder/"inputs.json").resolve()),sha256=sha256(folder/"inputs.json")),
        source_population=dict(path=str(population_path.resolve()),sha256=sha256(population_path)),
        selection=dict(panel_name=panel,description="frozen panel membership; no per-UID model-gain or method-quality selection"),
        e6_minimum_feedback_users=2000,e6_has_enough_feedback_users=sum(bool(by_user.get(uid)) for uid in users)>=2000)


def diagnose(parent,current,histories,by_user,users,cfg,device,output,identity):
    if not 8<=len(users)<=16:
        raise ValueError("numerical diagnostic requires 8--16 users")
    old_policy=[dict(tag="old_full",method="reuse",calibration_flops=0)]
    current_policy=[dict(tag=name,method=name,calibration_flops=0) for name in ("reuse","rebuild")]
    expected_old=old_native_reference(parent,histories,by_user,users,cfg,device)
    expected_current,_=selective.evaluate_reference(parent,current,histories,by_user,users,current_policy,cfg,device)
    comparisons=[]
    for graphs in (False,True):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter()
        actual_old,old_exec=fast.evaluate_batched(parent,parent,histories,by_user,users,old_policy,cfg,device,
            graphs=fast.NativeGraphs(enabled=graphs),stop_after_last_query=False)
        actual_current,current_exec=fast.evaluate_batched(parent,current,histories,by_user,users,current_policy,cfg,device,
            graphs=fast.NativeGraphs(enabled=graphs),stop_after_last_query=False)
        torch.cuda.synchronize(device)
        comparisons.append(dict(cuda_graphs=graphs,old_native=compare_scores(expected_old,actual_old),
            current_anchors=compare_scores(expected_current,actual_current),seconds=time.perf_counter()-start,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),old_execution=old_exec,current_execution=current_exec))
    result=dict(role="numerical_performance_diagnostic",users=len(users),quality_metrics_computed=False,
        comparisons=comparisons,identity=identity,
        old_full_scope="old checkpoint at release, old native append, old native score throughout the future window")
    common.write_json(output/"check.json",result)
    return result


def measure_edge(edge,population,population_path,input_root,output_root,device,*,check=False,diagnostic_users=8):
    folder=input_root/f"edge_{edge}"
    binding,histories,by_user=common.load_edge(folder)
    cfg=binding["settings"]
    for name in ("history","requests"):
        if sha256(folder/binding[name]["path"])!=binding[name]["sha256"]:
            raise ValueError(f"frozen {name} pack changed")
    if binding["population_sha256"]!=sha256(population_path):
        raise ValueError("population differs from frozen input binding")
    parent,current,models=common.load_models(edge,device)
    code=sources()
    panels=frozen_panels(population)
    union=sorted({int(uid) for users in panels.values() for uid in users})
    output=output_root/f"edge_{edge}"
    output.mkdir(parents=True,exist_ok=True)
    identity=dict(experiment="three_anchor_scenario_qualification",edge=edge,models=models,
        population_sha256=sha256(population_path),panels={name:list(users) for name,users in panels.items()},
        input_binding_sha256=sha256(folder/"inputs.json"),sources=code,settings=cfg,
        old_full="parent checkpoint produces every initial row, native behavior write, and score",
        current_rebuild="once before first history consumer; subsequent Current native writes",
        current_reuse="same old initial cache; subsequent Current native writes")
    path=output/"identity.json"
    if path.exists() and json.loads(path.read_text())!=identity:
        raise ValueError("qualification attempt identity changed; use a new output directory")
    common.write_json(path,identity)
    if check:
        result=diagnose(parent,current,histories,by_user,population["development"][:diagnostic_users],
            cfg,device,output,identity)
        if sources()!=code:
            raise ValueError("numerical sources changed during diagnostic")
        return result
    for users in panels.values():
        common.quality_role(users)
    started=time.perf_counter()
    old_output=output/"union_old_full"
    current_output=output/"union_current_anchors"
    old_output.mkdir(exist_ok=True)
    current_output.mkdir(exist_ok=True)
    def progress(branch,value):
        record=dict(stage="scenario_anchors",edge=edge,branch=branch,**value)
        print(json.dumps(record),flush=True)
        common.write_json(output/"progress.json",record)
    old_scores,old_execution=fast.evaluate_checkpointed(parent,parent,histories,by_user,union,
        [dict(tag="old_full",method="reuse",calibration_flops=0)],cfg,device,old_output,
        identity=dict(identity,union=union,branch="old_weights_for_initial_append_and_score"),
        progress=lambda value:progress("old_full",value))
    current_scores,current_execution=fast.evaluate_checkpointed(parent,current,histories,by_user,union,
        [dict(tag=name,method=name,calibration_flops=0) for name in ("reuse","rebuild")],cfg,device,current_output,
        identity=dict(identity,union=union,branch="current_reuse_and_once_rebuild"),
        progress=lambda value:progress("current_anchors",value))
    fresh=old_scores+current_scores
    # E6 needs feedback users, not merely 2k selected resident users. This
    # candidate is selected by UID hash and feedback presence, never model gains.
    feedback=[uid for uid in union if by_user.get(uid)]
    feedback.sort(key=lambda uid:hashlib.sha256(f"scenario-feedback-2000/{edge}/{uid}".encode()).digest())
    if len(feedback)>=2000:
        panels["feedback_union_2000"]=feedback[:2000]
    summaries={}
    for name,users in panels.items():
        destination=output/name
        destination.mkdir(exist_ok=True)
        summary=bootstrap_anchors(fresh,users,replicates=1000,seed=28121+edge)
        summary.update(edge=edge,panel=name,identity=identity,evaluation_window=cfg.get("evaluation_window",dict(name="E14",complete_source_horizon=True)))
        summary_path=destination/"summary.json"
        common.write_json(summary_path,summary)
        manifest=scenario_manifest(name,users,summary,summary_path,folder,binding,population_path,by_user)
        if name=="feedback_union_2000":
            manifest["selection"]["description"]="first 2000 UID hashes within four frozen panels' union having real future feedback; no per-UID model-gain selection"
        common.write_json(destination/"scenario_manifest.json",manifest)
        summaries[name]={key:summary[key] for key in ("quality_users","feedback_users","feedback_events",
            "auc","new_rebuild_minus_old_full","new_rebuild_minus_old_full_ci95","new_rebuild_minus_reuse",
            "new_rebuild_minus_reuse_ci95","point_qualifies","uid95_qualifies")}
        print(json.dumps(dict(stage="scenario_qualified",edge=edge,panel=name,**summaries[name])),flush=True)
    if sources()!=code:
        raise ValueError("numerical sources changed during qualification")
    result=dict(state="complete",edge=edge,union_users=len(union),union_feedback_users=len(feedback),
        summaries=summaries,execution=dict(old_full=old_execution,current_anchors=current_execution),
        seconds_excluding_input_model_loading=time.perf_counter()-started,
        identity=identity,method_quality_scores_read=False,
        scope="all four frozen population results retained; no scenario automatically selected")
    common.write_json(output/"summary.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge",type=int,choices=range(1,6),default=1)
    parser.add_argument("--all-edges",action="store_true")
    parser.add_argument("--input-root",type=Path,default=ROOT/"experiments/ch3_motivation/inputs")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--diagnostic-users",type=int,default=8)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if args.check and args.all_edges:
        parser.error("run one edge per numerical diagnostic")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    device=torch.device("cuda:0")
    population_path=args.input_root/"population.json"
    population=json.loads(population_path.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for edge in (range(1,6) if args.all_edges else [args.edge]):
        result=measure_edge(edge,population,population_path,args.input_root,args.output,device,
            check=args.check,diagnostic_users=args.diagnostic_users)
        results.append(result)
        if args.check:
            print(json.dumps(dict(passed=True,users=result["users"],quality_metrics_computed=False,
                comparisons=[dict(cuda_graphs=row["cuda_graphs"],old_max_error=row["old_native"]["maximum_absolute_difference"],
                    current_max_error=row["current_anchors"]["maximum_absolute_difference"],seconds=row["seconds"])
                    for row in result["comparisons"]])),flush=True)
        del result
        torch.cuda.empty_cache()
    if args.run:
        common.write_json(args.output/"summary.json",dict(state="complete",edges=results,
            scope="new three-anchor scenario screening; all frozen panels retained; no correction method quality evaluated"))


if __name__=="__main__":
    main()
