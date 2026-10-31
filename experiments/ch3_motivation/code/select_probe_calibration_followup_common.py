#!/usr/bin/env python3
"""Select a final three-seed candidate from the unchanged 475-weight task."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

from experiments.ch3_motivation.code import (
    select_probe_calibration_common as previous, validate_probe_calibration_followup_common as followup,
)
from experiments.ch3_motivation.code.common import ROOT, write_json
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.select_probe_common import _check_metrics, fingerprint
from hstu_kvcache.common.artifacts import sha256

FORMAT="ch3_final_calibration_common_selection_v1"
EDGES=followup.EDGES


def _check_point(point,spec,training,contexts,pair_failed):
    if any(point.get(key)!=value for key,value in spec.items()):
        raise ValueError("PhaseB validation row changed its planned specification")
    if point["status"] not in ("validated","failed","pending"):
        raise ValueError("PhaseB CPU identity-only rows cannot enter final selection")
    if point["status"]!="validated":
        if point.get("selection_eligible",False):raise ValueError("failed/pending PhaseB row claims eligibility")
        return None
    if set(point["contexts"])!=set(followup.RULES):raise ValueError("PhaseB common point lacks a context")
    for rule in followup.RULES:
        cell=point["contexts"][rule]
        if cell["status"]!="validated" or cell["common_validation_sha256"]!=fingerprint(contexts[rule]):
            raise ValueError("PhaseB point uses a different common fingerprint")
        _check_metrics(cell["metrics"],contexts[rule]["validation_uids"])
    mean=math.fsum(point["contexts"][rule]["metrics"]["loss"] for rule in followup.RULES)/3
    if not math.isclose(mean,point["mean_context_loss"],rel_tol=1e-7,abs_tol=1e-12):
        raise ValueError("PhaseB objective is not an equal three-context mean")
    path=Path(spec["checkpoint"])
    if sha256(path)!=point["checkpoint_sha256"]:raise ValueError("PhaseB checkpoint changed")
    probe,metadata=load_variant(path,"cpu")
    followup.check_weight(probe,metadata,spec,training,contexts["cutover"]["models"])
    if (metadata["identity"]!=point["training_identity"] or asdict(probe.cfg)!=point["config"] or
            probe.stage!=point["stage"] or getattr(probe,"variant_name","head")!=point["variant"] or
            metadata["best_step"]!=point["best_step"] or metadata["completed_steps"]!=point["completed_steps"] or
            point["step_zero"]!=(metadata["best_step"]==0) or
            point["fitting_users"]!=len(metadata["identity"]["fit_uids"]) or
            metadata["cumulative_calibration_flops_estimate"]!=point["cumulative_calibration_flops_estimate"] or
            point["sampling_view_key"]!=training["sampling_provenance"]["view_key"] or
            point["sampling_provenance"]!=training["sampling_provenance"] or
            point["verified_sampling_identity"]!=training["verified_sampling_identity"]):
        raise ValueError("PhaseB actual typed checkpoint and common row differ")
    matched_failure=pair_failed and spec["recipe"] in ("reference","balanced_h_init")
    eligible=not metadata.get("diagnostic_only",False) and metadata.get("quality_use_permitted",True) is not False and not matched_failure
    if point["selection_eligible"]!=eligible or (point.get("matched_initialization_status")=="failed")!=matched_failure:
        raise ValueError("PhaseB eligibility differs from its actual typed/matched-control evidence")
    cost=metadata["cumulative_calibration_flops_estimate"]
    if not math.isfinite(cost) or cost<=0:raise ValueError("PhaseB standalone cumulative calibration cost is invalid")
    return dict(metadata["identity"],input_root=training["sampling_provenance"]["sourcebinding"]["input_root"])


def matched_diagnostics(groups,paired_by_edge):
    primary=[group for group in groups if group["method_label"]=="fgh" and group["seed"]==17]
    reference=next((group for group in primary if group["recipe"]=="calibration_followup/reference"),None)
    rows=[]
    for group in primary:
        row=dict(recipe=group["recipe"],fit_users=group["fit_users"],width=group["width"],
            h_initialization=(group.get("settings") or {}).get("h_initialization"),
            complete_five_edges=group["complete_five_edges"],selection_eligible=group["selection_eligible"],
            mean_common_loss=group["mean_common_loss"],common_loss_by_edge=group["common_loss_by_edge"],
            step_zero_edges=group["step_zero_edges"],failed_edges=group["failed_edges"],pending_edges=group["pending_edges"],
            complete_calibration_flops_estimate_sum=group["complete_calibration_flops_estimate_sum"],
            mean_common_loss_minus_reference=None,common_loss_minus_reference_by_edge=None)
        if reference and reference["complete_five_edges"] and group["complete_five_edges"]:
            row.update(mean_common_loss_minus_reference=group["mean_common_loss"]-reference["mean_common_loss"],
                common_loss_minus_reference_by_edge={str(edge):group["common_loss_by_edge"][str(edge)]-
                    reference["common_loss_by_edge"][str(edge)] for edge in EDGES})
        rows.append(row)
    return dict(recipes=rows,paired_initialization_by_edge={str(edge):audit for edge,audit in paired_by_edge.items()},
        scope="seven matched input-fixed PhaseB controls; three-context pre-release losses only; all original475 candidates remain in final selection")


def choose(groups,ready,phase_b_completed,failed,rule,old_recommendation):
    result=previous.choose(groups,ready and phase_b_completed,failed,rule)
    result.update(phase_b_completed=phase_b_completed,final_candidate_for_three_seeds=bool(result["recommendation"]),
        expected_weights=475,expected_context_cells=1425,previous370_recommendation=old_recommendation,
        selection_scope="combined475 final selection remains pending; no recommendation or training launch" if not result["recommendation"] else
            "PhaseB completed: combined475 final candidate for correction seeds17/23/29; root review/freeze and independent population confirmation remain required; no training launched")
    if result["recommendation"] is not None:
        result["recommendation"]["caveat"]="final common-validation candidate on correction seed17 only; preserve its actual driver/law/init; step0 and failures retained; three-seed fitting and independent population confirmation remain pending; no population AUC used"
    return result


def rollup(sampling_plan,common_plan,followup_plan,old_paths,sampling_paths,followup_paths):
    common_plan,followup_plan=Path(common_plan),Path(followup_plan)
    base=previous.rollup(sampling_plan,common_plan,old_paths,sampling_paths)
    groups=base["groups"];points=base["points"]
    if not followup_plan.exists() or not common_plan.exists():
        return dict(format=FORMAT,expected_weights=475,expected_context_cells=1425,groups=groups,points=points,
            pending_plans=[str(path.resolve()) for path in (followup_plan,common_plan) if not path.exists()],
            selection=choose(groups,False,False,[],{},base["selection"]["recommendation"]))
    specs={};trainings={};frozen={};prepared_by_edge={}
    for edge in EDGES:
        plan,common,prepared,bindings=followup.load_plan(followup_plan,common_plan,edge)
        frozen.update(bindings);prepared_by_edge[edge]=prepared
        for spec,training in prepared:
            checkpoint=spec["checkpoint"]
            if checkpoint in specs:raise ValueError("duplicate planned PhaseB checkpoint")
            specs[checkpoint],trainings[checkpoint]=spec,training
    if not specs:
        return dict(format=FORMAT,expected_weights=475,expected_context_cells=1425,groups=groups,points=points,
            pending_followup_status=plan["status"],observed_weights=base.get("observed_weights",0),
            selection=choose(groups,False,False,[],common["selection"],base["selection"]["recommendation"]))
    if len(specs)!=105:raise ValueError("final common comparison requires the planned105 PhaseB weights")
    summaries={};contexts={};observed={};paired={};summary_hashes=dict(base["summary_sha256"])
    for path in map(Path,followup_paths):
        summary=json.loads(path.read_text());edge=summary["edge"]
        if (edge not in EDGES or edge in summaries or summary["format"]!=followup.FORMAT or
                summary["followup_plan_sha256"]!=sha256(followup_plan) or summary["common_plan_sha256"]!=sha256(common_plan)):
            raise ValueError("PhaseB common edge/format/plan changed")
        config_path=path.with_name("configuration.json");config=json.loads(config_path.read_text())
        if (sha256(config_path)!=summary["configuration_sha256"] or config["format"]!=followup.FORMAT or
                config["mode"]!="common_validation" or config["edge"]!=edge or
                config["followup_plan_sha256"]!=sha256(followup_plan) or config["common_plan_sha256"]!=sha256(common_plan) or
                config["contexts"]!=summary["contexts"]):
            raise ValueError("PhaseB common configuration differs from its summary")
        followup.phase_a.verify_hashes(config["frozen_sha256"])
        contexts[edge]=previous.old._check_contexts(dict(summary,format=previous.old.FORMAT),common,edge)
        if str(edge) in base.get("common_contexts_by_edge",{}) and contexts[edge]!=base["common_contexts_by_edge"][str(edge)]:
            raise ValueError("PhaseB and previous370 use different common fingerprints/targets/scales")
        expected={key for key,spec in specs.items() if spec["edge"]==edge}
        actual=[str(Path(point["checkpoint"]).resolve()) for point in summary["points"]]
        if len(set(actual))!=len(actual) or not set(actual)<=expected:raise ValueError("duplicate/unplanned PhaseB common weight")
        if summary["status"]!="in_progress" and set(actual)!=expected:raise ValueError("final PhaseB summary omits planned weights")
        paired[edge]=followup.check_paired_predecessors(prepared_by_edge[edge],copy.deepcopy(summary["points"]))
        if ("paired_initialization" in summary or summary["status"]!="in_progress") and summary.get("paired_initialization")!=paired[edge]:
            raise ValueError("PhaseB matched initialization proof differs from actual tensors")
        for point in summary["points"]:
            checkpoint=str(Path(point["checkpoint"]).resolve())
            settings=_check_point(point,specs[checkpoint],trainings[checkpoint],contexts[edge],
                summary["status"]!="in_progress" and paired[edge]["status"]=="failed")
            observed[checkpoint]=dict(point=point,settings=settings)
        counts={status:sum(point["status"]==status for point in summary["points"]) for status in ("validated","pending","failed")}
        terminal="waiting_weights" if counts["pending"] else "complete_with_failures" if counts["failed"] or paired[edge]["status"]=="failed" else "complete"
        if summary["status"]!="in_progress" and (summary["status"]!=terminal or any(summary.get(key)!=value for key,value in counts.items())):
            raise ValueError("PhaseB common status/counts differ from actual rows and matched proof")
        summaries[edge]=summary;summary_hashes[str(path.resolve())]=sha256(path)
    rows=[]
    for checkpoint,spec in specs.items():
        item=observed.get(checkpoint);point=item["point"] if item else dict(spec,status="missing",contexts={})
        row=dict(spec,recipe="calibration_followup/"+spec["recipe"],validation=point)
        if point["status"]=="validated":
            settings=item["settings"]
            row.update(settings=settings,normalized_identity=previous.normalized_sampling_identity(settings),
                source_binding=previous.old._input_binding(Path(settings["input_root"]),spec["edge"]),
                sampling_view_key=trainings[checkpoint]["sampling_provenance"]["view_key"])
        rows.append(row)
    new_groups=previous._group_sampling(rows)
    for group in new_groups:
        for edge,provenance in group["sampling_provenance_by_edge"].items():
            identity=next(row["settings"] for row in rows if row["recipe"]==group["recipe"] and row["method_label"]==group["method_label"]
                and str(row["edge"])==edge and row["validation"]["status"]=="validated")
            provenance.update({key:identity[key] for key in followup.INITIALIZATION_FIELDS})
    combined=points+rows;groups=groups+new_groups
    if len(combined)!=475 or len({row["checkpoint"] for row in combined})!=475:
        raise ValueError("final comparison must retain475 distinct planned weights")
    completed=set(summaries)==set(EDGES) and len(observed)==105 and all(
        summary["status"] in ("complete","complete_with_failures") for summary in summaries.values())
    ready=base["selection"]["all_five_edges_complete"] and completed
    failed=[row["checkpoint"] for row in combined if row["validation"]["status"]=="failed"]
    selection=choose(groups,ready,completed,failed,common["selection"],base["selection"]["recommendation"])
    selection.update(sampling_phase_a=base["selection"]["sampling_phase_a"],
        missing_phase_b_edges=sorted(set(EDGES)-set(summaries)),
        needs_matched_initialization_review=any(audit["status"]=="failed" for audit in paired.values()))
    if selection["needs_matched_initialization_review"]:selection["needs_failure_review"]=True
    followup.phase_a.verify_hashes(frozen)
    return dict(format=FORMAT,expected_weights=475,expected_context_cells=1425,observed_weights=base["observed_weights"]+len(observed),
        previous_expected_weights=370,phase_b_expected_weights=105,points=combined,groups=groups,selection=selection,
        matched_diagnostics=matched_diagnostics(new_groups,paired),summary_sha256=summary_hashes,
        common_contexts_by_edge={**base.get("common_contexts_by_edge",{}),**{str(edge):value for edge,value in contexts.items()}},
        followup_plan_path=str(followup_plan.resolve()),followup_plan_sha256=sha256(followup_plan),
        common_plan_path=str(common_plan.resolve()),common_plan_sha256=sha256(common_plan),
        selector_sha256=sha256(Path(__file__)),previous_selector_sha256=sha256(Path(previous.__file__)),
        scope="all475 on the same old fit64-ID/shared-three-context-RMS task; final common candidate only; no population AUC or GPU launch")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-plan",type=Path,default=ROOT/"experiments/ch3_motivation/configs/probe_calibration_sampling.json")
    parser.add_argument("--common-plan",type=Path,default=ROOT/"experiments/ch3_motivation/configs/probe_common_temporal_validation.json")
    parser.add_argument("--followup-plan",type=Path,required=True)
    parser.add_argument("--old-common-root",type=Path,default=ROOT/"experiments/ch3_motivation/runs/e5/common_temporal_validation/attempt_1")
    parser.add_argument("--sampling-common-root",type=Path,default=ROOT/"experiments/ch3_motivation/runs/e5/calibration_sampling_common/attempt_1")
    parser.add_argument("--followup-common-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    paths=lambda root:[path for edge in EDGES if (path:=root/f"edge_{edge}/summary.json").exists()]
    result=rollup(args.sampling_plan,args.common_plan,args.followup_plan,paths(args.old_common_root),
        paths(args.sampling_common_root),paths(args.followup_common_root))
    write_json(args.output/"summary.json",result);write_json(args.output/"selection.json",result["selection"])
    print(json.dumps(dict(status=result["selection"]["status"],observed_weights=result.get("observed_weights",0),gpu_launched=False)))
