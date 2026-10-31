#!/usr/bin/env python3
"""Combine the audited old280 and sampling90 search on identical common contexts."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

from experiments.ch3_motivation.code import (
    select_probe_temporal_common as old, validate_probe_calibration_common as sampling,
)
from experiments.ch3_motivation.code.common import ROOT, write_json
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.select_probe_common import (
    EDGES, FORMAL_SEEDS, _check_metrics, fingerprint,
)
from hstu_kvcache.common.artifacts import sha256

FORMAT="ch3_combined_sampling_common_selection_v1"
SAMPLING_EDGE_FIELDS=old.EDGE_FIELDS|{"sampling_manifest","sampling_manifest_sha256"}


def normalized_sampling_identity(identity):
    """Discard only verified edge-local bindings; keep actual recipe/init settings."""
    return {key:value for key,value in identity.items() if key not in SAMPLING_EDGE_FIELDS}


def _check_point(point,spec,training,contexts):
    if any(point.get(key)!=value for key,value in spec.items()):
        raise ValueError("sampling common point changed its planned specification")
    if point["status"] not in ("validated","failed","pending"):
        raise ValueError("CPU identity checks cannot enter common selection")
    if point["status"]!="validated":
        if point.get("selection_eligible",False):
            raise ValueError("pending/failed sampling weight claims selection eligibility")
        return None
    if set(point["contexts"])!=set(old.RULES):
        raise ValueError("sampling point lacks a common context")
    for rule in old.RULES:
        cell=point["contexts"][rule]
        if cell["status"]!="validated" or cell["common_validation_sha256"]!=fingerprint(contexts[rule]):
            raise ValueError("sampling weight uses a different common fingerprint")
        _check_metrics(cell["metrics"],contexts[rule]["validation_uids"])
    mean=math.fsum(point["contexts"][rule]["metrics"]["loss"] for rule in old.RULES)/3
    if not math.isclose(mean,point["mean_context_loss"],rel_tol=1e-7,abs_tol=1e-12):
        raise ValueError("sampling context objective is not an equal three-context mean")
    path=Path(spec["checkpoint"])
    if sha256(path)!=point["checkpoint_sha256"]:
        raise ValueError("sampling checkpoint changed after common validation")
    probe,metadata=load_variant(path,"cpu")
    sampling.check_sampling_weight(probe,metadata,spec,training,contexts["cutover"]["models"])
    if (metadata["identity"]!=point["training_identity"] or asdict(probe.cfg)!=point["config"]
            or probe.stage!=point["stage"] or getattr(probe,"variant_name","head")!=point["variant"]
            or metadata["best_step"]!=point["best_step"] or metadata["completed_steps"]!=point["completed_steps"]
            or point["step_zero"]!=(metadata["best_step"]==0)
            or metadata["cumulative_calibration_flops_estimate"]!=point["cumulative_calibration_flops_estimate"]):
        raise ValueError("sampling typed checkpoint and validation row differ")
    eligible=not metadata.get("diagnostic_only",False) and metadata.get("quality_use_permitted",True) is not False
    if point.get("selection_eligible",False)!=eligible:
        raise ValueError("sampling point eligibility differs from its actual checkpoint")
    cost=metadata["cumulative_calibration_flops_estimate"]
    if not math.isfinite(cost) or cost<=0:
        raise ValueError("sampling independent cumulative calibration cost is invalid")
    settings=dict(metadata["identity"],input_root=training["sampling_provenance"]["sourcebinding"]["input_root"])
    return settings


def _group_sampling(rows):
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row["recipe"],row["method_label"],row["seed"])].append(row)
    groups=[]
    for (recipe,method,seed),values in sorted(grouped.items()):
        values.sort(key=lambda row:row["edge"])
        valid=[row for row in values if row["validation"]["status"]=="validated"]
        if valid and any(row["normalized_identity"]!=valid[0]["normalized_identity"] for row in valid):
            raise ValueError("one sampling recipe changes non-edge fitting/init settings")
        complete=[row["edge"] for row in valid]==EDGES
        first=valid[0] if valid else values[0]
        point=first["validation"]
        losses={str(row["edge"]):row["validation"]["mean_context_loss"] for row in valid}
        groups.append(dict(recipe=recipe,method_label=method,seed=seed,
            variant=point.get("variant"),width=point.get("config",{}).get("hidden_width"),
            depth=point.get("config",{}).get("depth"),fit_users=point.get("fitting_users"),
            complete_five_edges=complete,validated_edges=[row["edge"] for row in valid],
            failed_edges=[row["edge"] for row in values if row["validation"]["status"]=="failed"],
            pending_edges=[row["edge"] for row in values if row["validation"]["status"] in ("pending","missing")],
            step_zero_edges=[row["edge"] for row in values if row["validation"].get("step_zero")],
            selection_eligible=complete and all(row["validation"].get("selection_eligible",False) for row in valid),
            common_loss_by_edge=losses,mean_common_loss=math.fsum(losses.values())/5 if complete else None,
            mean_loss_by_context={rule:math.fsum(row["validation"]["contexts"][rule]["metrics"]["loss"] for row in valid)/5
                                  for rule in old.RULES} if complete else None,
            complete_calibration_flops_estimate_sum=sum(row["validation"].get("cumulative_calibration_flops_estimate",0) or 0 for row in values),
            settings=first.get("settings"),checkpoints_by_edge={str(row["edge"]):dict(checkpoint=row["checkpoint"],
                checkpoint_sha256=row["validation"].get("checkpoint_sha256"),status=row["validation"]["status"],
                best_step=row["validation"].get("best_step"),completed_steps=row["validation"].get("completed_steps")) for row in values},
            input_bindings_by_edge={str(row["edge"]):row["source_binding"] for row in valid},
            gap_provenance_by_edge={str(row["edge"]):{key:row["settings"].get(key) for key in (
                "fitting_driver","fitting_driver_sha256","query_delta_rule","gap_manifest","gap_manifest_sha256",
                "time_rule","source_cutoff","candidate_order","query_delta_layout")} for row in valid},
            sampling_provenance_by_edge={str(row["edge"]):dict(
                **{key:row["settings"][key] for key in ("sampling_manifest","sampling_manifest_sha256","gap_manifest",
                    "gap_manifest_sha256","source_cutoff","candidate_order","time_rule","query_delta_layout","query_delta_rule",
                    "candidate_catalog_source_users","time_law_source_users","gradient_fitting_users","teacher_fitting_users")},
                view_key=row["sampling_view_key"]) for row in valid}))
    return groups


def choose(groups,ready,failed,rule):
    selection=dict(status="not_ready",all_five_edges_complete=ready,selection_seed=17,primary_family="full_item",
        primary_stage="fgh",future_correction_seeds=FORMAL_SEEDS,development_seed_robustness_established=False,
        needs_failure_review=bool(failed),failed_checkpoints=failed,recommendation=None,rule=rule,
        selection_scope="combined370 search-stage recommendation only; PhaseB calibration and final3seed freeze remain pending")
    candidates=[group for group in groups if group["variant"]=="full_item" and group["method_label"]=="fgh"
                and group["seed"]==17 and group["selection_eligible"]]
    if ready and candidates:
        minimum=min(group["mean_common_loss"] for group in candidates)
        threshold=minimum*1.01
        tied=[group for group in candidates if group["mean_common_loss"]<=threshold]
        chosen=min(tied,key=lambda group:(group["fit_users"],group["width"],group["complete_calibration_flops_estimate_sum"],
                                         group["mean_common_loss"],group["recipe"]))
        settings=chosen["settings"]
        selection.update(status="recommendation_for_review",minimum_mean_common_loss=minimum,fixed_one_percent_threshold=threshold,
            within_one_percent=[group["recipe"] for group in tied],recommendation=dict(chosen,edges=EDGES,correction_seeds=FORMAL_SEEDS,
                fitting_driver=settings["fitting_driver"],fitting_driver_sha256=settings["fitting_driver_sha256"],
                query_delta_rule=settings["query_delta_rule"],backbone_retraining=False,
                caveat="selected on correction seed17 only; step0 remains visible and does not establish a nonzero h benefit; no population AUC used"))
    elif ready:
        selection["status"]="no_complete_eligible_primary_recipe"
    return selection


def choose_sampling_inputs(groups,ready,failed):
    """Choose the PhaseB input view only among the six actual law256 recipes."""
    recipes=[group for group in groups if group["method_label"]=="fgh" and group["seed"]==17]
    if recipes and len(recipes)!=6:
        raise ValueError("sampling PhaseA requires exactly six planned primary input recipes")
    candidates=[group for group in recipes if group["selection_eligible"]]
    for group in candidates:
        settings=group["settings"]
        if (group["variant"]!="full_item" or group["fit_users"]!=256 or
                settings["fitting_driver"]!="run_probe_calibration.py" or
                settings["candidate_catalog_source_users"]!=256 or settings["time_law_source_users"]!=256):
            raise ValueError("PhaseB input recommendation must retain the actual new law256 recipe")
    ranked=sorted(candidates,key=lambda group:(group["mean_common_loss"],
        group["complete_calibration_flops_estimate_sum"],group["recipe"]))
    ranks={group["recipe"]:rank for rank,group in enumerate(ranked,1)}
    result=dict(status="not_ready",all_five_edges_complete=ready,selection_seed=17,
        expected_input_recipes=6,complete_eligible_input_recipes=len(candidates),
        needs_failure_review=bool(failed),failed_checkpoints=failed,recommendation=None,
        recipes=[dict(group,rank=ranks.get(group["recipe"])) for group in sorted(recipes,key=lambda group:group["recipe"])],
        ranked_recipes=[dict(group,rank=ranks[group["recipe"]]) for group in ranked],
        ranking_rule="lowest equal-three-context then equal-five-edge mean common loss; exact ties by independent calibration FLOPs then recipe name",
        selection_scope="new six input recipes only; retain catalog256/time-law256; PhaseB budgets/capacity and final3seed selection remain pending",
        global_recommendation_role="combined370 external common benchmark only; never substitute an old law64 winner for this new law256 input setting")
    if ready and len(candidates)==6:
        chosen=ranked[0]
        settings=chosen["settings"]
        result.update(status="recommendation_for_review",recommendation=dict(chosen,edges=EDGES,
            correction_seeds=FORMAL_SEEDS,fitting_driver=settings["fitting_driver"],
            fitting_driver_sha256=settings["fitting_driver_sha256"],query_delta_rule=settings["query_delta_rule"],
            backbone_retraining=False,
            caveat="PhaseA input-view choice on correction seed17 only; step0 and all six outcomes retained; not a final budget/capacity choice; no population AUC used"))
    elif ready:
        result["status"]="incomplete_eligible_input_recipes"
    return result


def rollup(fitting_plan_path,common_plan_path,old_summary_paths,new_summary_paths):
    fitting_plan_path,common_plan_path=Path(fitting_plan_path),Path(common_plan_path)
    absent=[str(path.resolve()) for path in (fitting_plan_path,common_plan_path) if not path.exists()]
    if absent:
        return dict(format=FORMAT,expected_weights=370,expected_context_cells=1110,points=[],groups=[],pending_plans=absent,
            selection=dict(status="not_ready",recommendation=None,all_five_edges_complete=False,
                sampling_phase_a=choose_sampling_inputs([],False,[])))
    # This is the unchanged strict old280 selector, including every typed
    # legacy identity and all its original context/metric/source checks.
    previous=old.rollup(common_plan_path,old_summary_paths)
    specs,trainings,frozen={}, {}, {}
    for edge in EDGES:
        fitting,common,prepared,bindings=sampling.load_plan(fitting_plan_path,common_plan_path,edge)
        frozen.update(bindings)
        for spec,training in prepared:
            checkpoint=spec["checkpoint"]
            if checkpoint in specs:
                raise ValueError("duplicate planned sampling checkpoint")
            specs[checkpoint],trainings[checkpoint]=spec,training
    if len(specs)!=90:
        raise ValueError("combined comparison requires all90 sampling weights")
    summaries,contexts,points,summary_hashes={}, {}, {}, dict(previous["summary_sha256"])
    for path in map(Path,new_summary_paths):
        summary=json.loads(path.read_text())
        edge=summary["edge"]
        if (edge not in EDGES or edge in summaries or summary["format"]!=sampling.FORMAT
                or summary["fitting_plan_sha256"]!=sha256(fitting_plan_path)
                or summary["common_plan_sha256"]!=sha256(common_plan_path)):
            raise ValueError("unexpected sampling common edge/format/plan")
        configuration=path.with_name("configuration.json")
        config=json.loads(configuration.read_text())
        if (sha256(configuration)!=summary["configuration_sha256"] or config["mode"]!="common_validation"
                or config["format"]!=sampling.FORMAT or config["edge"]!=edge or config["contexts"]!=summary["contexts"]
                or config["fitting_plan_sha256"]!=sha256(fitting_plan_path) or config["common_plan_sha256"]!=sha256(common_plan_path)):
            raise ValueError("sampling common configuration differs from its summary")
        sampling.verify_hashes(config["frozen_sha256"])
        # Only the outer envelope differs. The context identity keeps the old
        # format and is checked with the original strict common-context gate.
        contexts[edge]=old._check_contexts(dict(summary,format=old.FORMAT),common,edge)
        if edge in previous["common_contexts_by_edge"] and contexts[edge]!=previous["common_contexts_by_edge"][edge]:
            raise ValueError("old280 and sampling90 use different common fingerprints/targets/scales")
        summaries[edge]=summary
        summary_hashes[str(path.resolve())]=sha256(path)
        observed=set()
        for point in summary["points"]:
            checkpoint=str(Path(point["checkpoint"]).resolve())
            if checkpoint in observed or checkpoint not in specs or specs[checkpoint]["edge"]!=edge:
                raise ValueError("duplicate or unplanned sampling validation checkpoint")
            observed.add(checkpoint)
            settings=_check_point(point,specs[checkpoint],trainings[checkpoint],contexts[edge])
            points[checkpoint]=dict(point=point,settings=settings)
        expected={path for path,spec in specs.items() if spec["edge"]==edge}
        if summary["status"] in ("complete","complete_with_failures","waiting_weights") and observed!=expected:
            raise ValueError("final sampling common summary omits a planned weight")
        counts={status:sum(point["status"]==status for point in summary["points"]) for status in ("validated","pending","failed")}
        terminal="waiting_weights" if counts["pending"] else "complete_with_failures" if counts["failed"] else "complete"
        if summary["status"]!="in_progress" and (summary["status"]!=terminal or any(summary.get(key)!=value for key,value in counts.items())):
            raise ValueError("sampling common status/counts differ from its rows")
    rows=[]
    for checkpoint,spec in specs.items():
        observed=points.get(checkpoint)
        point=observed["point"] if observed else dict(spec,status="missing",contexts={})
        row=dict(spec,validation=point)
        if point["status"]=="validated":
            settings=observed["settings"]
            row.update(settings=settings,normalized_identity=normalized_sampling_identity(settings),
                source_binding=old._input_binding(Path(settings["input_root"]),spec["edge"]),
                sampling_view_key=trainings[checkpoint]["sampling_provenance"]["view_key"])
        rows.append(row)
    sampling_groups=_group_sampling(rows)
    groups=previous["groups"]+sampling_groups
    combined=previous["points"]+rows
    if len(combined)!=370 or len({row["checkpoint"] for row in combined})!=370:
        raise ValueError("old/new checkpoint inventory must be370 distinct planned weights")
    sampling_ready=set(summaries)==set(EDGES) and len(points)==90 and all(
        summary["status"] in ("complete","complete_with_failures") for summary in summaries.values())
    ready=previous["selection"]["all_five_edges_complete"] and sampling_ready
    failed=[row["checkpoint"] for row in combined if row["validation"]["status"]=="failed"]
    selection=choose(groups,ready,failed,common["selection"])
    selection["sampling_phase_a"]=choose_sampling_inputs(sampling_groups,sampling_ready,
        [row["checkpoint"] for row in rows if row["validation"]["status"]=="failed"])
    selection.update(missing_old_edges=previous["selection"]["missing_edges"],missing_sampling_edges=sorted(set(EDGES)-set(summaries)))
    sampling.verify_hashes(frozen)
    return dict(format=FORMAT,fitting_plan=str(fitting_plan_path.resolve()),fitting_plan_sha256=sha256(fitting_plan_path),
        common_plan=str(common_plan_path.resolve()),common_plan_sha256=sha256(common_plan_path),
        selector_sha256=sha256(Path(__file__)),legacy_selector_sha256=sha256(Path(old.__file__)),
        expected_weights=370,observed_weights=previous["observed_weights"]+len(points),expected_context_cells=1110,
        old_expected_weights=280,sampling_expected_weights=90,groups=groups,points=combined,selection=selection,
        summary_sha256=summary_hashes,common_contexts_by_edge={str(edge):value for edge,value in
            {**previous["common_contexts_by_edge"],**contexts}.items()},
        scope="same fit64-ID/shared-three-context-scale problem only; fixed five-edge objective; no population quality, final recipe freeze or GPU launch")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitting-plan",type=Path,default=ROOT/"experiments/ch3_motivation/configs/probe_calibration_sampling.json")
    parser.add_argument("--common-plan",type=Path,default=ROOT/"experiments/ch3_motivation/configs/probe_common_temporal_validation.json")
    parser.add_argument("--old-common-root",type=Path,default=ROOT/"experiments/ch3_motivation/runs/e5/common_temporal_validation/attempt_1")
    parser.add_argument("--sampling-common-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    paths=lambda root:[path for edge in EDGES if (path:=root/f"edge_{edge}/summary.json").exists()]
    result=rollup(args.fitting_plan,args.common_plan,paths(args.old_common_root),paths(args.sampling_common_root))
    write_json(args.output/"summary.json",result)
    write_json(args.output/"selection.json",result["selection"])
    print(json.dumps(dict(status=result["selection"]["status"],observed_weights=result.get("observed_weights",0),gpu_launched=False)))
