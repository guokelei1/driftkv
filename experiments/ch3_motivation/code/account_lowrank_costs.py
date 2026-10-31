#!/usr/bin/env python3
"""Complete FG/full-H deployment FLOPs on the actual fixed E6 population.

This CPU sidecar does not run a model forward or infer a deployable low-rank
algorithm from hindsight rank interventions. Research decomposition/repeated
rank-score work remains a separate, explicitly unestimated overhead.
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
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
import numpy as np
import pyarrow.parquet as pq
import torch
from hstu_kvcache.models import HSTU,HSTUConfig
from experiments.ch3_motivation.code.common import write_json
from experiments.ch3_motivation.code import plan_probe_quality as quality
from experiments.ch3_motivation.code.fg_trajectory_control import load_fg_control
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.probe_ledger import full_probe_ledger
from experiments.ch3_motivation.code.evaluate_probes import correction_cost, apply_projected_costs
from experiments.ch3_motivation.code.calibration_cost_correction import checkpoint_correction, apply_to_costs
from experiments.ch3_motivation.code.scenario_inputs import load_scenario_population
from hstu_kvcache.evaluation.history import restore_pack
from hstu_kvcache.evaluation.baselines import event_shapes
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.common.artifacts import sha256

SCENARIOS=ROOT/"experiments/ch3_motivation/runs/scenario_qualification/attempt_1"
CALIBRATION_ACCOUNTING="shared_row_normalization_corrected_v1"


def _uid_hash(users):
    return hashlib.sha256(json.dumps(users).encode()).hexdigest()


def _merge(*snapshots):
    result={}
    for snapshot in snapshots:
        for name,digest in snapshot.items():
            path=Path(name);path=str((path if path.is_absolute() else ROOT/path).resolve())
            if path in result and result[path]!=digest:raise ValueError(f"E6 cost frozen SHA conflict: {path}")
            result[path]=digest
    return result


def _check(files):
    for path,digest in files.items():
        if sha256(Path(path))!=digest:raise ValueError(f"E6 cost dependency changed: {path}")


def sources():
    paths=[Path(__file__),*[Path(__file__).with_name(name) for name in (
        "plan_probe_quality.py","probe_ledger.py","evaluate_probes.py","probe_cost.py","probe_projected_cost.py",
        "probe_projected_fast.py","probe_fast.py","probes.py","probe_variants.py","fg_trajectory_control.py",
        "scenario_inputs.py","common.py","selective.py","calibration_cost_correction.py")]]
    paths.extend((ROOT/"src/hstu_kvcache/models").glob("*.py"))
    paths.extend(ROOT/name for name in ("src/hstu_kvcache/adaptation/reader.py","src/hstu_kvcache/evaluation/history.py",
        "src/hstu_kvcache/evaluation/baselines.py","src/hstu_kvcache/evaluation/cost.py",
        "src/hstu_kvcache/methods/baselines/protocol/cost.py","src/hstu_kvcache/common/artifacts.py"))
    return _merge({str(path):sha256(path) for path in paths},quality.sources())


def load_geometry(path,expected_sha=None):
    """Map the real final checkpoint header; allocate model tensors on meta."""
    path=Path(path).resolve();digest=sha256(path)
    if expected_sha is not None and digest!=expected_sha:raise ValueError("E6 backbone SHA differs")
    payload=torch.load(path,map_location="cpu",mmap=True,weights_only=False)
    status=str(payload.get("status",""))
    if not status.startswith("formal_") or not status.endswith("_checkpoint") or payload.get("progress")!=1.:
        raise ValueError("E6 geometry requires a completed formal backbone checkpoint")
    config=HSTUConfig(**payload["config"])
    with torch.device("meta"):
        model=HSTU(config).eval().requires_grad_(False)
    info=dict(path=str(path),sha256=digest,status=status,progress=payload["progress"],version=payload["version"],
        config=payload["config"],parameter_device="meta",weight_tensors_loaded_into_model=False)
    del payload
    return model,info


def load_population(path,edge):
    binding,users,population,provenance=load_scenario_population(path,edge,role="e6",minimum_users=2000,require_positive=False)
    if len(users)!=2000:raise ValueError("E6 cost requires its fixed exact2000 feedback users")
    histories=restore_pack(Path(provenance["history_path"]))
    by_user=defaultdict(lambda:defaultdict(list))
    at,end=binding["settings"]["release_day"]*86400,binding["settings"]["end_day"]*86400
    for row in pq.read_table(provenance["requests_path"],columns=["uid","query_timestamp"],filters=[("uid","in",users)]).to_pylist():
        uid,timestamp=int(row["uid"]),int(row["query_timestamp"])
        if not at<=timestamp<end:raise ValueError("E6 feedback lies outside its frozen E14 window")
        by_user[uid][timestamp].append(None)  # Shapes require counts, no labels/items/scores.
    if any(not by_user.get(uid) for uid in users):raise ValueError("E6 cost population includes a user without feedback")
    return binding,users,population,provenance,histories,by_user


def timeline_counts(histories,by_user,users,cfg):
    at,end=cfg["release_day"]*86400,cfg["end_day"]*86400
    full=quality_prefix=0
    for uid in users:
        times=histories.rows[uid][0];before=int(np.searchsorted(times,at,side="left"))
        full+=int(np.searchsorted(times,end,side="left"))-before
        quality_prefix+=int(np.searchsorted(times,max(by_user[uid]),side="left"))-before
    return dict(users=len(users),feedback_events=sum(len(v) for rounds in by_user.values() for v in rounds.values()),
        complete_behavior_appended_rows=full,before_last_feedback_behavior_rows=quality_prefix,
        post_last_feedback_behavior_rows=full-quality_prefix,
        scope="input-only strict timestamp counts; includes all E14 behavior after the last scored feedback")


def method_cost(probe,fit,model,histories,by_user,users,cfg):
    calibration=fit["cumulative_calibration_flops_estimate"]
    if not isinstance(calibration,int) or calibration<0:raise ValueError("actual complete calibration cost is missing")
    c=probe.cfg
    if (c.num_layers!=len(model.blocks) or c.num_heads!=model.cfg.num_heads or
            c.head_dim!=model.blocks[0].attn.head_dim):raise ValueError("probe and actual backbone geometry differ")
    execution=dict(full_probe_ledger(histories,by_user,users,cfg,stage=probe.stage,padded=True),padded_execution=True)
    shapes=event_shapes(histories,by_user,users,cfg)
    cm=CostModel(model.cfg.hidden_size,len(model.blocks),model.cfg.num_heads,cfg["attention_backend"])
    raw=correction_cost(probe,execution,calibration,cm,shapes,cfg,len(users),model=model)
    cost=apply_projected_costs(raw,probe,execution,histories,by_user,users,cfg,model)
    return cost,execution


def prepare_evidence(scenario_root=SCENARIOS):
    """No selected weights, model geometry, FLOP totals or AUC are invented."""
    rows={};inputs={}
    for edge in range(1,6):
        path=Path(scenario_root)/f"edge_{edge}/feedback_union_2000/scenario_manifest.json"
        binding,users,_,provenance,histories,by_user=load_population(path,edge)
        cfg=binding["settings"]
        rows[str(edge)]=dict(population_file=str(path.resolve()),population_sha256=sha256(path),uids_sha256=_uid_hash(users),
            timeline=timeline_counts(histories,by_user,users,cfg),
            ledgers={stage:full_probe_ledger(histories,by_user,users,cfg,stage=stage,padded=True) for stage in ("fg","fgh")})
        for name in ("history","requests","input_binding","calibration_input_binding","source_population","qualification_summary"):
            inputs[provenance[name+"_path"]]=provenance[name+"_sha256"]
        inputs[str(path.resolve())]=sha256(path)
    _check(_merge(inputs))
    return dict(format="e6_complete_cost_input_evidence_v1",status="input_ledgers_prepared",formal_cost_computed=False,
        gpu_used=False,quality_auc_computed=False,cost_bindings={},edges=rows,input_sha256=_merge(inputs),sources_sha256=sources())


def generate(plan_path,*,output=None):
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text());code=sources()
    result=dict(format="e6_complete_cost_v1",status="waiting_formal_weights",complete_serving_cost_ready=False,
        e6_plan=str(plan_path),e6_plan_sha256=sha256(plan_path),formal_plan_sha256=plan.get("formal_plan_sha256"),
        cost_bindings={},gpu_used=False,quality_auc_computed=False,sources_sha256=code,
        calibration_accounting=dict(format="e6_raw_and_corrected_calibration_v1",
            primary_accounts="actual_event/analytic230 use corrected calibration",
            raw_accounts="raw_actual_event/raw_analytic230 preserve the original checkpoint ledger",
            correction_scope="deduct duplicated shared row normalization from calibration only; deployment operations and both Rebuild denominators unchanged"),
        actual_event_scope="defined FG/H deployment execution on real E6 users/events; not the total FLOPs of run_lowrank_release research panels/guards/repeated rank scores",
        research_overhead=dict(status="not_accounted_as_deployment",svd_flops=None,oracle_guard_flops=None,
            hindsight_rank_read_intervention_flops=None,
            scope="research decomposition/QR/SVD, guards, snapshot transfer and repeated rank scoring are separate; no full research-pipeline total or deployable rank-specific writer FLOPs are claimed"))
    bindings=plan.get("weight_bindings",{})
    if plan.get("status")!="prepared_for_root_review" or not bindings:return result
    if set(bindings)!={f"edge_{e}_seed_{s}" for e in range(1,6) for s in (17,23,29)}:
        raise ValueError("E6 complete costs require the actual15 weight bindings")
    missing=[binding[field] for binding in bindings.values() for field in ("trained_fg_checkpoint","full_h_checkpoint")
        if not Path(binding[field]).is_file()]
    if missing:
        result["missing_dependencies"]=missing
        return result
    formal_path=Path(plan["formal_plan"]);formal=json.loads(formal_path.read_text())
    if sha256(formal_path)!=plan["formal_plan_sha256"]:raise ValueError("E6 source formal plan changed")
    frozen=_merge(plan["frozen_sha256"],code,{str(plan_path):sha256(plan_path)})
    _check(frozen)
    populations,geometries={},{}
    for key,binding in sorted(bindings.items()):
        edge,seed=binding["edge"],binding["seed"]
        if binding["id"]!=key or key!=f"edge_{edge}_seed_{seed}":raise ValueError("E6 cost binding id differs")
        if edge not in populations:populations[edge]=load_population(binding["population_file"],edge)
        raw,users,population,provenance,histories,by_user=populations[edge]
        if (users!=binding["uids"] or _uid_hash(users)!=binding["uids_sha256"] or binding["users"]!=2000 or
                sha256(Path(binding["population_file"]))!=binding["population_sha256"] or
                provenance!=binding["population_provenance"] or provenance["anchor_models"]!=binding["models"]):
            raise ValueError("E6 cost uses different population/input/anchor bindings")
        job=next(j for j in formal["main_jobs"] if j["edge"]==edge and j["seed"]==seed)
        fit_binding=formal["main_input_bindings"][str(edge)]
        for method,field in (("fg","trained_fg_checkpoint"),("fgh","full_h_checkpoint")):
            if sha256(Path(binding[field]))!=binding[field+"_sha256"]:raise ValueError("E6 actual FG/H weight changed")
            quality.audit_weight(Path(binding[field]),job,fit_binding,method)
        full,fit=load_variant(binding["full_h_checkpoint"])
        fg,fg_fit,pair=load_fg_control(binding["trained_fg_checkpoint"],full,fit,source_population=population,
            evaluation_uids=users,edge=edge,models=binding["models"],calibration_binding_sha256=provenance["calibration_input_binding_sha256"])
        if pair!=binding["paired_control"]:raise ValueError("E6 paired-control provenance differs")
        current=None;model_records=[]
        for name,digest in binding["models"].items():
            path=Path(name);path=(path if path.is_absolute() else ROOT/path).resolve();cache_key=(str(path),digest)
            if cache_key not in geometries:geometries[cache_key]=load_geometry(path,digest)
            model,info=geometries[cache_key];model_records.append(info)
            if info["version"]==f"v{edge}":current=model
            frozen=_merge(frozen,{str(path):digest})
        if current is None or {row["version"] for row in model_records}!={f"v{edge-1}",f"v{edge}"}:
            raise ValueError("E6 actual model headers do not identify the adjacent pair")
        actual,analytic,raw_actual,raw_analytic,corrections,ledgers={},{},{},{},{},{}
        for method,probe,metadata in (("fg",fg,fg_fit),("fgh",full,fit)):
            if metadata["cumulative_calibration_flops_estimate"]!=binding["cumulative_calibration_flops_estimate"][method]:
                raise ValueError("E6 recorded calibration cost differs from its actual checkpoint")
            cost,ledger=method_cost(probe,metadata,current,histories,by_user,users,raw["settings"])
            field="trained_fg_checkpoint" if method=="fg" else "full_h_checkpoint"
            correction=checkpoint_correction(binding[field],expected_sha256=binding[field+"_sha256"])
            if (correction["edge"]!=edge or correction["seed"]!=seed or correction["method_label"]!=method):
                raise ValueError("E6 calibration correction uses a different FG/H binding")
            exported=apply_to_costs(cost,correction)
            original,corrected=exported["raw"],exported["corrected"]
            raw_analytic[method]=original["candidate_reads_230"]
            raw_actual[method]={k:v for k,v in original.items() if k!="candidate_reads_230"}
            analytic[method]=corrected["candidate_reads_230"]
            actual[method]={k:v for k,v in corrected.items() if k!="candidate_reads_230"}
            for account in (actual[method],analytic[method]):
                account.update(calibration_accounting=CALIBRATION_ACCOUNTING,
                    calibration_correction_record=f"calibration_cost_corrections.{method}")
            corrections[method]=exported["calibration_correction"]
            ledgers[method]=ledger
        value=dict(id=key,edge=edge,seed=seed,formal_plan_sha256=plan["formal_plan_sha256"],
            full_h_checkpoint_sha256=binding["full_h_checkpoint_sha256"],trained_fg_checkpoint_sha256=binding["trained_fg_checkpoint_sha256"],
            population_sha256=binding["population_sha256"],uids_sha256=binding["uids_sha256"],users=2000,
            population_provenance=provenance,model_geometry=model_records,
            execution_settings=dict(deployment={name:raw["settings"][name] for name in
                ("history_length","append_band_size","query_batch","attention_backend","release_day","end_day")},
                probes={method:dict(query_chunk=probe.cfg.query_chunk,row_chunk=probe.cfg.row_chunk,
                    depth=probe.cfg.depth,width=probe.cfg.hidden_width,stage=probe.stage)
                    for method,probe in (("fg",fg),("fgh",full))},backend="projected_padded_graph_with_short_window_reference",
                analytic230="230 release-source candidate reads/user, cfg.query_batch grouping, actual full behavior evolution retained; distinct from research230-instance panels",
                research_driver_execution_included=False),
            timeline=timeline_counts(histories,by_user,users,raw["settings"]),actual_event=actual,analytic230=analytic,
            raw_actual_event=raw_actual,raw_analytic230=raw_analytic,calibration_cost_corrections=corrections,
            calibration_accounting=result["calibration_accounting"],
            complete_execution_ledgers=ledgers,research_overhead=result["research_overhead"],
            scope="complete E6 feedback-population native writer/read/release/correction calibration accounting; last-feedback quality shortcut is never used; no E5 resident-population substitution")
        result["cost_bindings"][key]=value
        if output is not None:write_json(Path(output)/"bindings"/f"{key}.json",value)
    _check(frozen)
    if sources()!=code:raise ValueError("E6 cost sources changed during accounting")
    result.update(status="complete",complete_serving_cost_ready=True,frozen_sha256=frozen)
    return result


def main(args):
    if args.prepare_cohorts:
        result=prepare_evidence(args.scenario_root);write_json(args.output/"input_evidence.json",result)
    else:
        if args.plan is None:raise ValueError("provide the actual E6 formal plan")
        result=generate(args.plan,output=args.output);write_json(args.output/"summary.json",result)
        if result["cost_bindings"]:
            rows=[dict(binding_id=key,edge=b["edge"],seed=b["seed"],method=method,ledger=kind,
                calibration_accounting=cost["calibration_accounting"],
                raw_calibration_flops=b["raw_"+kind][method]["calibration_flops"],
                calibration_normalization_deduction_flops=b["calibration_cost_corrections"][method]["shared_row_normalization_deduction_flops"],
                calibration_flops=cost["calibration_flops"],
                raw_complete_added_flops=b["raw_"+kind][method]["complete_added_flops"],
                complete_added_flops=cost["complete_added_flops"],once_rebuild_flops=cost["once_rebuild_flops"],
                raw_percent_of_once_rebuild=b["raw_"+kind][method]["percent_of_once_rebuild"],
                percent_of_once_rebuild=cost["percent_of_once_rebuild"]) for key,b in result["cost_bindings"].items()
                for kind in ("actual_event","analytic230") for method,cost in b[kind].items()]
            with (args.output/"costs.csv").open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status=result["status"],bindings=len(result["cost_bindings"]),gpu_used=False)))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prepare-cohorts",action="store_true")
    parser.add_argument("--scenario-root",type=Path,default=SCENARIOS)
    main(parser.parse_args())
