#!/usr/bin/env python3
"""Select a global E5 recipe from the frozen three-context common validation.

Every planned weight stays visible. Selection averages the three contexts
within each edge, then the five edges equally; it never reads population AUC.
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
import json
import math
from pathlib import Path
import sys

import torch
from experiments.ch3_motivation.code import calibration_gaps
from experiments.ch3_motivation.code.plan_probe_formal import _input_binding
from experiments.ch3_motivation.code.prepare_temporal_common_plan import RULES
from experiments.ch3_motivation.code.run_probe_temporal import FIXED as TEMPORAL_FIXED
from experiments.ch3_motivation.code.select_probe_common import (
    ROOT, EDGES, FORMAL_SEEDS, _check_metrics, fingerprint, sha256, write_json,
)
from experiments.ch3_motivation.code.validate_probe_temporal_common import (
    FORMAT, _sources as validation_sources,
)

EDGE_FIELDS = {"edge", "models", "input_binding_sha256", "calibration_candidate_sha256",
               "gap_manifest", "gap_manifest_sha256", "source_cutoff"}
LEGACY_DEFAULTS = dict(depth=1, fit_users=64, queries=16, steps=256, learning_rate=.0001,
    read_weight=1., score_weight=1., patience=128, validation_every=32,
    stages="f,fg,fgh", controls=False, candidate_source="feedback", recent_fraction=.5,
    candidate_seed=17, row_chunk=1024, query_chunk=16, weight_decay=.0001, gradient_clip=5.)


def normalized_recipe_identity(identity):
    """Only edge-local bindings disappear AFTER each exact binding was checked."""
    return {key: value for key, value in identity.items() if key not in EDGE_FIELDS}


def _jobs(plan):
    jobs = {}
    for path, digest in plan["source_fitting_plans_sha256"].items():
        path = Path(path)
        if sha256(path) != digest:
            raise ValueError("source fitting plan changed")
        source = json.loads(path.read_text())
        if path.name == "probe_temporal_calibration.json":
            sweep = "temporal_calibration"
        else:
            sweep = {"f_row_input_sweep.json": "f_row_input_sweep",
                "probe_ladder_development.json": "ladder_development",
                "probe_ladder_stability.json": "ladder_stability",
                "probe_ladder_feedback_pool.json": "ladder_feedback_pool"}[path.name]
        for job in source["jobs"]:
            key = (f"{sweep}/{job['name']}", job["edge"], job.get("seed", 17))
            if key in jobs:
                raise ValueError("duplicate source recipe job")
            # The legacy sweep defaults to f-only; ladder plans explicitly
            # request their extra stages. Match the executed sweep rather than
            # the standalone run_probe_variants CLI's three-stage default.
            if sweep != "temporal_calibration":
                job = {"stages": "f", "controls": False, **job}
            jobs[key] = job
    return jobs


def _check_contexts(summary, plan, edge):
    if (summary["format"] != FORMAT or summary["edge"] != edge or
            summary["quality_auc_computed"] is not False or
            summary["fitting_performed"] is not False or summary["post_release_labels_loaded"] is not False or
            summary["independent_fitting_users"] != 64 or summary["scale_user_context_observations"] != 192 or
            set(summary["contexts"]) != set(RULES)):
        raise ValueError("not the fixed pre-release three-context comparison")
    binding = plan["edge_bindings"][str(edge)]
    contexts = summary["contexts"]
    reference = contexts["cutover"]
    for rule in RULES:
        common = contexts[rule]
        if (common["format"] != FORMAT or common["edge"] != edge or common["query_delta_rule"] != rule or
                common["population_sha256"] != binding["population_sha256"] or
                common["input_binding_sha256"] != binding["input_binding_sha256"] or
                common["candidate_file_sha256"] != binding["candidate_file_sha256"] or
                common["gap_manifest_sha256"] != binding["gap_manifest_sha256"] or
                common["queries"] != 16 or common["recent_fraction"] != .5 or common["candidate_seed"] != 17):
            raise ValueError("common candidate/input/time identity differs from the frozen plan")
        gaps, prov = calibration_gaps.load_gap_manifest(binding["gap_manifest"], binding["input_root"], edge, rule)
        expected_deltas = {str(uid): gap.tolist() for uid, gap in gaps.items()}
        if (common["gap_provenance"] != prov or common["query_deltas"] != expected_deltas or
                common["query_deltas_sha256"] != fingerprint(expected_deltas) or
                common["scale_fit_uids"] != prov["fit_uids"] or common["validation_uids"] != prov["validation_uids"] or
                common["candidate_ids"] != prov["candidate_ids"] or prov["fallback_used"]):
            raise ValueError("common delta/user/candidate assignment differs from the fitting-only manifest")
        if (any(common[key] != prov[key] for key in ("time_rule", "source_cutoff", "candidate_order")) or
                Path(common["gap_manifest"]).resolve() != Path(binding["gap_manifest"]).resolve() or
                common["numerical_sources"] != {str(p.relative_to(ROOT)):sha256(p) for p in validation_sources()}):
            raise ValueError("common temporal/numerical provenance differs")
        scales = common["loss_scales"]
        if (scales != reference["loss_scales"] or scales["users"] != 192 or
                scales["unique_fitting_users"] != 64 or scales["context_count"] != 3 or
                scales["user_context_observations"] != 192 or scales["contexts"] != RULES or
                scales["source"] != "fitting_users_only" or
                scales["version"] != "fit_pool_equal_user_length_normalized_v1" or
                scales["validation_uses_fitting_scales"] is not True):
            raise ValueError("common loss scales are not shared fit64 x three contexts")
        for key in ("models", "candidate_ids", "candidate_order", "time_rule", "source_cutoff", "numerical_sources"):
            if common[key] != reference[key]:
                raise ValueError("one edge changes its common inputs across contexts")
        if (common["objective"]["read_weight"] != 1. or common["objective"]["score_weight"] != 1.):
            raise ValueError("common validation must use read+score with equal weights")
        native = summary["native_reuse"][rule]
        if native["common_validation_sha256"] != fingerprint(common):
            raise ValueError("native baseline uses different common inputs")
        _check_metrics(native["metrics"], common["validation_uids"])
    return contexts


def _check_training(point, spec, job, common):
    path = Path(spec["checkpoint"])
    if sha256(path) != point["checkpoint_sha256"]:
        raise ValueError("validated checkpoint SHA changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata, config = payload["metadata"], payload["config"]
    identity = metadata["identity"]
    if (identity != point["training_identity"] or config != point["config"] or
            payload["stage"] != point["stage"] or payload.get("variant", "head") != point["variant"] or
            metadata["best_step"] != point["best_step"] or metadata["completed_steps"] != point["completed_steps"] or
            metadata["cumulative_calibration_flops_estimate"] != point["cumulative_calibration_flops_estimate"]):
        raise ValueError("typed checkpoint and common-validation row differ")
    driver = identity.get("fitting_driver", "run_probe_variants.py")
    if driver != spec["fitting_driver"]:
        raise ValueError("source fitting driver differs")
    driver_path = ROOT / "experiments/ch3_motivation/code" / driver
    driver_sha = identity.get("fitting_driver_sha256") if driver == "run_probe_temporal.py" else (
        identity.get("numerical_sources", {}).get(str(driver_path.relative_to(ROOT))))
    if driver_sha != sha256(driver_path):
        raise ValueError("source fitting driver SHA differs")
    root = Path(job.get("input_root", ROOT / "experiments/ch3_motivation/inputs"))
    source_binding = _input_binding(root, spec["edge"])
    population = json.loads((root / "population.json").read_text())
    defaults = TEMPORAL_FIXED if driver == "run_probe_temporal.py" else LEGACY_DEFAULTS
    expected = dict(defaults, **job)
    count = int(expected["fit_users"])
    if (identity["edge"] != spec["edge"] or identity["seed"] != spec["seed"] or
            metadata.get("method_label", payload["stage"]) != spec["method_label"] or
            payload["stage"] != {"f_extended": "f", "fg_extended": "fg"}.get(spec["method_label"], spec["method_label"]) or
            identity["models"] != common["models"] or
            identity["population_sha256"] != source_binding["population_sha256"] or
            identity["input_binding_sha256"] != source_binding["input_binding_sha256"] or
            identity["fit_uids"] != population["fit"][:count] or identity["validation_users"] != population["validation"] or
            identity["validation_users"] != common["validation_uids"] or
            config["hidden_width"] != expected["width"] or config["depth"] != expected["depth"] or
            config.get("include_item") is not True or config.get("normalize") is not True or
            point["variant"] != expected["variant"]):
        raise ValueError("source recipe architecture/roles/models/input binding differs")
    for key in ("learning_rate", "steps", "queries", "candidate_seed", "read_weight", "score_weight",
                "patience", "validation_every", "candidate_source", "recent_fraction", "row_chunk",
                "query_chunk", "weight_decay", "gradient_clip", "stages", "controls"):
        # Temporal FIXED puts stages/controls in its per-job identity.
        value = expected.get(key, "f,fg,fgh" if key == "stages" else None)
        if identity.get(key) != value:
            raise ValueError(f"source recipe {key} differs")
    candidate_sha = source_binding["candidate_file_sha256"] if identity["candidate_source"] == "feedback" else None
    if identity.get("calibration_candidate_sha256") != candidate_sha:
        raise ValueError("source fitting candidate binding differs")
    rule = identity.get("query_delta_rule", "cutover")
    if driver == "run_probe_temporal.py":
        if (rule != spec["query_delta_rule"] or rule != job["gap_rule"] or
                identity["gap_manifest_sha256"] != spec["training_gap_manifest_sha256"] or
                Path(identity["gap_manifest"]).resolve() != Path(job["gap_manifest"]).resolve() or
                sha256(job["gap_manifest"]) != identity["gap_manifest_sha256"] or
                any(identity[key] != common[key] for key in ("time_rule", "source_cutoff", "candidate_order"))):
            raise ValueError("source temporal law/order/manifest differs")
    elif rule != "cutover":
        raise ValueError("release-gap fitting driver cannot represent a temporal rule")
    if (metadata.get("diagnostic_only") or metadata.get("quality_use_permitted") is False or
            metadata.get("fitting_uses_future_feedback") or
            metadata.get("fit_pool_loss_scales", {}).get("version") != "fit_pool_equal_user_length_normalized_v1"):
        raise ValueError("diagnostic/noncausal weights cannot enter formal selection")
    cost = metadata["cumulative_calibration_flops_estimate"]
    if not math.isfinite(cost) or cost <= 0:
        raise ValueError("independent cumulative calibration FLOPs are missing or invalid")
    settings = dict(identity, input_root=str(root.resolve()), fitting_driver=driver,
                    fitting_driver_sha256=driver_sha, query_delta_rule=rule)
    return settings, source_binding


def rollup(plan_path, summary_paths):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    if (plan["format"] != "ch3_temporal_common_plan_v1" or plan["rules"] != RULES or
            len(plan["weights"]) != 280 or len({p["checkpoint"] for p in plan["weights"]}) != 280):
        raise ValueError("expected all original240 and new40 planned weights")
    for name, digest in plan["executor_sources_sha256"].items():
        if sha256(name) != digest:
            raise ValueError("common-validation executor source changed")
    if sha256(Path(__file__).with_name("prepare_temporal_common_plan.py")) != plan["planner_sha256"]:
        raise ValueError("common-validation plan generator changed")
    jobs = _jobs(plan)
    specs = {str(Path(p["checkpoint"]).resolve()): p for p in plan["weights"]}
    summaries, contexts, points, source_hashes = {}, {}, {}, {}
    for path in map(Path, summary_paths):
        summary = json.loads(path.read_text())
        edge = summary["edge"]
        if edge not in EDGES or edge in summaries or summary["plan_sha256"] != sha256(plan_path):
            raise ValueError("unexpected/duplicate edge or common plan SHA")
        contexts[edge] = _check_contexts(summary, plan, edge)
        summaries[edge] = summary
        source_hashes[str(path.resolve())] = sha256(path)
        observed = set()
        for point in summary["points"]:
            checkpoint = str(Path(point["checkpoint"]).resolve())
            if checkpoint in observed or checkpoint not in specs or specs[checkpoint]["edge"] != edge:
                raise ValueError("duplicate or unplanned validation checkpoint")
            observed.add(checkpoint)
            spec = specs[checkpoint]
            if any(point.get(key) != value for key, value in spec.items()):
                raise ValueError("common point changed its source recipe specification")
            if point["status"] == "validated":
                if set(point["contexts"]) != set(RULES):
                    raise ValueError("validated point lacks a common context")
                for rule in RULES:
                    cell = point["contexts"][rule]
                    if (cell["status"] != "validated" or
                            cell["common_validation_sha256"] != fingerprint(contexts[edge][rule])):
                        raise ValueError("weight uses a different common context")
                    _check_metrics(cell["metrics"], contexts[edge][rule]["validation_uids"])
                mean = math.fsum(point["contexts"][r]["metrics"]["loss"] for r in RULES) / 3
                if not math.isclose(point["mean_context_loss"], mean, rel_tol=1e-7, abs_tol=1e-12):
                    raise ValueError("context means are not equally weighted")
            elif point["status"] not in ("failed", "pending"):
                raise ValueError("unexpected point status")
            points[checkpoint] = point
        expected = {p for p, spec in specs.items() if spec["edge"] == edge}
        if summary["status"] in ("complete", "complete_with_failures", "waiting_weights") and observed != expected:
            raise ValueError("final common summary omits planned checkpoints")
        counts = {s: sum(p["status"] == s for p in summary["points"]) for s in ("validated", "pending", "failed")}
        if any(summary.get(s) != n for s, n in counts.items()) and summary["status"] != "in_progress":
            raise ValueError("common summary status/counts differ")
        final_status = "waiting_weights" if counts["pending"] else (
            "complete_with_failures" if counts["failed"] else "complete")
        if summary["status"] != "in_progress" and summary["status"] != final_status:
            raise ValueError("common summary success/failure status differs from its rows")
    # Edge models, manifests and candidate IDs differ, but not the population,
    # reserved roles, objective, contexts, backend or numerical definitions.
    if contexts:
        first = next(iter(contexts.values()))["cutover"]
        for values in contexts.values():
            common = values["cutover"]
            for key in ("population_sha256", "scale_fit_uids", "validation_uids", "queries", "recent_fraction",
                        "candidate_seed", "time_rule", "candidate_order", "objective", "backend", "numerical_sources"):
                if common[key] != first[key]:
                    raise ValueError("common protocol changes across edges")
    grouped, all_rows = defaultdict(list), []
    for checkpoint, spec in specs.items():
        point = points.get(checkpoint, dict(spec, status="missing", contexts={}))
        job = jobs[(spec["recipe"], spec["edge"], spec["seed"])]
        row = dict(spec, validation=point)
        if point["status"] == "validated":
            settings, source_binding = _check_training(point, spec, job, contexts[spec["edge"]]["cutover"])
            row.update(settings=settings, source_binding=source_binding,
                       normalized_identity=normalized_recipe_identity(settings))
        all_rows.append(row)
        grouped[(spec["recipe"], spec["method_label"], spec["seed"])].append(row)
    groups = []
    for (recipe, method, seed), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["edge"])
        valid = [r for r in rows if r["validation"]["status"] == "validated"]
        if valid and any(r["normalized_identity"] != valid[0]["normalized_identity"] for r in valid):
            raise ValueError("one recipe changes non-edge fitting settings across edges")
        complete = [r["edge"] for r in valid] == EDGES
        first = valid[0] if valid else rows[0]
        point = first["validation"]
        losses = {str(r["edge"]): r["validation"]["mean_context_loss"] for r in valid}
        group = dict(recipe=recipe, method_label=method, seed=seed,
            variant=point.get("variant"), width=point.get("config", {}).get("hidden_width"),
            depth=point.get("config", {}).get("depth"), fit_users=point.get("fitting_users"),
            complete_five_edges=complete, validated_edges=[r["edge"] for r in valid],
            failed_edges=[r["edge"] for r in rows if r["validation"]["status"] == "failed"],
            pending_edges=[r["edge"] for r in rows if r["validation"]["status"] in ("pending", "missing")],
            step_zero_edges=[r["edge"] for r in rows if r["validation"].get("step_zero")],
            selection_eligible=complete and all(r["validation"].get("selection_eligible", False) for r in valid),
            common_loss_by_edge=losses, mean_common_loss=math.fsum(losses.values()) / 5 if complete else None,
            mean_loss_by_context={rule: math.fsum(r["validation"]["contexts"][rule]["metrics"]["loss"]
                for r in valid) / 5 for rule in RULES} if complete else None,
            complete_calibration_flops_estimate_sum=sum(r["validation"].get("cumulative_calibration_flops_estimate", 0) or 0 for r in rows),
            settings=first.get("settings"), checkpoints_by_edge={str(r["edge"]): dict(
                checkpoint=r["checkpoint"], checkpoint_sha256=r["validation"].get("checkpoint_sha256"),
                status=r["validation"]["status"], best_step=r["validation"].get("best_step"),
                completed_steps=r["validation"].get("completed_steps")) for r in rows},
            input_bindings_by_edge={str(r["edge"]): r["source_binding"] for r in valid},
            gap_provenance_by_edge={str(r["edge"]): {key: r["settings"].get(key) for key in
                ("fitting_driver", "fitting_driver_sha256", "query_delta_rule", "gap_manifest",
                 "gap_manifest_sha256", "time_rule", "source_cutoff", "candidate_order", "query_delta_layout")}
                for r in valid})
        groups.append(group)
    ready = (set(summaries) == set(EDGES) and len(points) == 280 and
        all(s["status"] in ("complete", "complete_with_failures") for s in summaries.values()) and
        all(p["status"] in ("validated", "failed") for p in points.values()))
    failed = [p["checkpoint"] for p in points.values() if p["status"] == "failed"]
    selection = dict(status="not_ready", all_five_edges_complete=ready, selection_seed=17,
        primary_family="full_item", primary_stage="fgh", future_correction_seeds=FORMAL_SEEDS,
        development_seed_robustness_established=False,
        needs_failure_review=bool(failed), failed_checkpoints=failed, recommendation=None,
        rule=plan["selection"], missing_edges=sorted(set(EDGES) - set(summaries)))
    selection["selection_scope"] = "current280 search-stage suggestion only; expanded calibration search is separate; no final3seed freeze or launch"
    candidates = [g for g in groups if g["variant"] == "full_item" and g["method_label"] == "fgh" and
                  g["seed"] == 17 and g["selection_eligible"]]
    if ready and candidates:
        minimum = min(g["mean_common_loss"] for g in candidates)
        threshold = minimum * 1.01
        tied = [g for g in candidates if g["mean_common_loss"] <= threshold]
        chosen = min(tied, key=lambda g: (g["fit_users"], g["width"],
            g["complete_calibration_flops_estimate_sum"], g["mean_common_loss"], g["recipe"]))
        settings = chosen["settings"]
        selection.update(status="recommendation_for_review", minimum_mean_common_loss=minimum,
            fixed_one_percent_threshold=threshold, within_one_percent=[g["recipe"] for g in tied],
            recommendation=dict(chosen, edges=EDGES, correction_seeds=FORMAL_SEEDS,
                fitting_driver=settings["fitting_driver"], fitting_driver_sha256=settings["fitting_driver_sha256"],
                query_delta_rule=settings["query_delta_rule"], backbone_retraining=False,
                caveat="selected on17 only; h step0 is retained and does not establish nonzero interaction; no AUC used"))
    elif ready:
        selection["status"] = "no_complete_eligible_primary_recipe"
    return dict(format="ch3_temporal_common_selection_v1", plan=str(plan_path.resolve()),
        plan_sha256=sha256(plan_path), selector_sha256=sha256(Path(__file__)), summary_sha256=source_hashes,
        expected_weights=280, observed_weights=len(points), expected_context_cells=840,
        groups=groups, points=all_rows, selection=selection, common_contexts_by_edge=contexts,
        scope="new fit64-ID/shared-three-context-scale validation only; old fit256 common losses excluded; no AUC or GPU launch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--common-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = rollup(args.plan, [p for e in EDGES if (p := args.common_root / f"edge_{e}/summary.json").exists()])
    write_json(args.output / "summary.json", result)
    write_json(args.output / "selection.json", result["selection"])
    print(json.dumps(dict(status=result["selection"]["status"], observed_weights=result["observed_weights"],
                         gpu_launched=False)))
