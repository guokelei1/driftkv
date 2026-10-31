#!/usr/bin/env python3
"""Summarize common pre-release validation and recommend one frozen E5 recipe.

Selection uses development seed17, full_item/fgh, all five edges equally, and
the common validator's fixed problem. Within 1% of the minimum mean loss,
prefer fewer fit users, smaller width, then fewer complete calibration FLOPs.
This is a CPU-only analysis; it never fits weights or reads evaluation AUC.
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
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EDGES = [1, 2, 3, 4, 5]
SELECTION_SEED = 17
FORMAL_SEEDS = [17, 23, 29]
FORMAT = "ch3_common_pre_release_validation_v1"
SETTING_KEYS = ("learning_rate", "steps", "read_weight", "score_weight", "patience",
    "validation_every", "stages", "controls", "row_chunk", "query_chunk", "weight_decay",
    "gradient_clip", "candidate_source", "recent_fraction", "candidate_seed", "queries",
    "sample_candidates", "include_item", "normalize")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _check_metrics(metrics, validation_uids):
    rows = metrics["per_user"]
    if metrics["users"] != 16 or [row["uid"] for row in rows] != validation_uids:
        raise ValueError("common validation per-UID coverage differs")
    for key in ("loss", "read_mse", "score_mse", "raw_score_mse"):
        values = [float(row[key]) for row in rows]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("common validation contains an invalid loss")
        if not math.isclose(float(metrics[key]), math.fsum(values) / 16, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("common validation equal-user mean differs")
    if not math.isclose(metrics["loss"], metrics["read_mse"] + metrics["score_mse"],
                        rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("common validation objective is not read+score with equal weights")


def _check_common(summary):
    common = summary["common_validation"]
    token = fingerprint(common)
    if common["format"] != FORMAT or summary["common_validation_sha256"] != token:
        raise ValueError("common validation identity fingerprint differs")
    if (common["queries"] != 16 or common["recent_fraction"] != .5 or
            common["candidate_seed"] != 17 or len(common["validation_uids"]) != 16 or
            len(common["scale_fit_uids"]) != 64 or
            common["loss_scales"]["users"] != 64 or
            common["loss_scales"]["source"] != "fitting_users_only" or
            common["loss_scales"]["version"] != "fit_pool_equal_user_length_normalized_v1" or
            not common["loss_scales"]["validation_uses_fitting_scales"] or
            common["objective"]["read_weight"] != 1. or common["objective"]["score_weight"] != 1.):
        raise ValueError("common validation protocol is not the fixed Q16/50%/fit64 task")
    if (len(common["fitting_candidate_pool_uids"]) != 256 or
            common["fitting_candidate_pool_uids"][:64] != common["scale_fit_uids"]):
        raise ValueError("common validation does not use the fixed fit256 candidate view")
    uids = common["scale_fit_uids"] + common["validation_uids"]
    if len(set(uids)) != 80 or set(common["candidate_ids"]) != {str(uid) for uid in uids}:
        raise ValueError("common validation candidate user roles differ")
    if fingerprint(common["candidate_ids"]) != common["candidate_ids_sha256"]:
        raise ValueError("common validation candidate identity differs")
    if any(len(values) != 16 or len(set(values)) != 16 for values in common["candidate_ids"].values()):
        raise ValueError("common validation requires 16 distinct candidates per user")
    if (summary["quality_auc_computed"] is not False or summary["fitting_performed"] is not False or
            summary["post_release_labels_loaded"] is not False):
        raise ValueError("selection requires pre-release validation-only evidence")
    if summary["native_reuse"]["common_validation_sha256"] != token:
        raise ValueError("native baseline uses a different validation problem")
    _check_metrics(summary["native_reuse"]["metrics"], common["validation_uids"])
    return common


def _settings(entry, point, job):
    identity = point.get("training_identity") or {}
    result = dict(variant=entry["variant"], width=entry["width"], depth=entry["depth"],
                  fit_users=entry["fit_users"],
                  input_root=str(Path(job.get("input_root",
                      ROOT / "experiments/ch3_motivation/inputs")).resolve()))
    result.update({key: identity[key] for key in SETTING_KEYS if key in identity})
    # The exact source job remains available if a failed checkpoint could not
    # expose its training identity. Never manufacture a successful validation.
    for key in SETTING_KEYS:
        if key not in result and key in job:
            result[key] = job[key]
    return result


def rollup(inventory_path, summary_paths, *, plan_directory=None, expected_points=None):
    """Return all rows plus a suggestion; missing edges cannot produce a choice."""
    inventory_path = Path(inventory_path)
    inventory = json.loads(inventory_path.read_text())
    if expected_points is not None and len(inventory) != expected_points:
        raise ValueError("checkpoint inventory count differs from the frozen plan")
    if not inventory:
        raise ValueError("empty checkpoint inventory")
    by_checkpoint = {str(Path(row["checkpoint"]).resolve()): row for row in inventory}
    if len(by_checkpoint) != len(inventory):
        raise ValueError("duplicate checkpoint in common validation inventory")
    if any(row["edge"] not in EDGES for row in inventory):
        raise ValueError("inventory contains an unexpected edge")
    plan_directory = Path(plan_directory or ROOT / "experiments/ch3_motivation/configs")
    plans = {sha256(path): path for path in plan_directory.glob("*.json")}
    sources, source_plans = {}, {}
    for entry in inventory:
        path = Path(entry["source_summary"]).resolve()
        key = str(path)
        if key not in sources:
            if sha256(path) != entry["source_summary_sha256"]:
                raise ValueError("source sweep summary changed")
            source = json.loads(path.read_text())
            if source["status"] != "complete":
                raise ValueError("source fitting sweep is incomplete")
            sources[key] = source
            plan_hash = source["identity"]["plan_sha256"]
            if plan_hash not in plans:
                raise ValueError("source sweep plan is absent or changed")
            plan_path = plans[plan_hash]
            source_plans[key] = (plan_path, json.loads(plan_path.read_text()))
        elif sha256(path) != entry["source_summary_sha256"]:
            raise ValueError("inconsistent source sweep binding in inventory")
    for path, source in sources.items():
        entries = [row for row in inventory if str(Path(row["source_summary"]).resolve()) == path]
        canonical = lambda row: {key: value for key, value in row.items()
                                 if key not in ("source_summary", "source_summary_sha256")}
        if {fingerprint(canonical(row)) for row in entries} != {fingerprint(row) for row in source["points"]}:
            raise ValueError("inventory does not cover its complete source sweep")

    summaries, common_by_edge, common_points, artifacts = {}, {}, {}, {}
    for path in map(Path, summary_paths):
        summary = json.loads(path.read_text())
        common = _check_common(summary)
        edge = int(common["edge"])
        if edge not in EDGES or edge in summaries:
            raise ValueError("unexpected or duplicate common validation edge")
        summaries[edge], common_by_edge[edge] = summary, common
        artifacts[str(path.resolve())] = sha256(path)
        observed = set()
        for point in summary["points"]:
            checkpoint = str(Path(point["checkpoint"]).resolve())
            if checkpoint in observed or checkpoint not in by_checkpoint:
                raise ValueError("duplicate or unplanned common validation checkpoint")
            observed.add(checkpoint)
            entry = by_checkpoint[checkpoint]
            if entry["edge"] != edge or point["common_validation_sha256"] != summary["common_validation_sha256"]:
                raise ValueError("checkpoint edge or common validation identity differs")
            if point.get("checkpoint_sha256") not in (None, entry["checkpoint_sha256"]):
                raise ValueError("checkpoint hash differs between inventory and validation")
            if point["status"] == "validated":
                if sha256(checkpoint) != entry["checkpoint_sha256"]:
                    raise ValueError("validated checkpoint changed")
                for key, expected in (("variant", entry["variant"]), ("method_label", entry["method_label"]),
                        ("seed", entry["seed"]), ("best_step", entry["best_step"]),
                        ("fitting_users", entry["fit_users"])):
                    if point[key] != expected:
                        raise ValueError(f"common validation checkpoint {key} differs from inventory")
                if point["config"]["hidden_width"] != entry["width"] or point["config"]["depth"] != entry["depth"]:
                    raise ValueError("common validation checkpoint capacity differs from inventory")
                _check_metrics(point["metrics"], common["validation_uids"])
            elif point["status"] != "failed":
                raise ValueError("unexpected common validation point status")
            common_points[checkpoint] = point
        expected = {path for path, entry in by_checkpoint.items() if entry["edge"] == edge}
        if summary["status"] in ("complete", "complete_with_failures") and observed != expected:
            raise ValueError("completed common validation summary has missing checkpoints")
        if summary["status"] in ("complete", "complete_with_failures"):
            failed = sum(point["status"] == "failed" for point in summary["points"])
            if (summary["failed"] != failed or summary["validated"] != len(observed) - failed or
                    (summary["status"] == "complete") != (failed == 0)):
                raise ValueError("completed common validation status/counts differ")
    # Across edges the teacher models and candidates may change. UID roles,
    # fixed loss protocol and numerical definitions must remain unchanged.
    protocol_keys = ("population_sha256", "scale_fit_uids", "fitting_candidate_pool_uids", "validation_uids", "queries",
                     "recent_fraction", "candidate_seed", "backend", "numerical_sources")
    if common_by_edge:
        first = next(iter(common_by_edge.values()))
        for common in common_by_edge.values():
            if any(common[key] != first[key] for key in protocol_keys):
                raise ValueError("common validation protocol or reserved population differs across edges")
        model_hashes = {}
        for common in common_by_edge.values():
            for model, digest in common["models"].items():
                if model in model_hashes and model_hashes[model] != digest:
                    raise ValueError("shared foundation checkpoint identity differs across edges")
                model_hashes[model] = digest

    rows, grouped = [], defaultdict(list)
    for checkpoint, entry in by_checkpoint.items():
        source_path = str(Path(entry["source_summary"]).resolve())
        source_folder = Path(source_path).parent
        sweep, attempt = source_folder.parent.name, source_folder.name
        recipe = f"{sweep}/{attempt}/{entry['name']}"
        plan_path, plan = source_plans[source_path]
        jobs = [job for job in plan["jobs"] if job["name"] == entry["name"] and
                job["edge"] == entry["edge"] and job.get("seed", 17) == entry["seed"]]
        if len(jobs) != 1:
            raise ValueError("inventory point does not identify one source recipe job")
        point = common_points.get(checkpoint, dict(status="missing", checkpoint=checkpoint))
        row = dict(recipe=recipe, source_sweep=sweep, name=entry["name"], edge=entry["edge"],
            seed=entry["seed"], variant=entry["variant"], width=entry["width"], depth=entry["depth"],
            fit_users=entry["fit_users"], method_label=entry["method_label"],
            best_step=entry["best_step"], completed_steps=entry["completed_steps"],
            step_zero=entry["best_step"] == 0,
            calibration_flops_estimate=entry["calibration_flops_estimate"],
            source_summary=source_path, source_summary_sha256=entry["source_summary_sha256"],
            source_plan=str(plan_path.resolve()), source_plan_sha256=sha256(plan_path),
            settings=_settings(entry, point, jobs[0]), common_validation=point)
        rows.append(row)
        grouped[(source_path, entry["name"], entry["method_label"], entry["seed"])].append(row)
    groups = []
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda row: row["edge"])
        edges = [row["edge"] for row in values]
        if len(set(edges)) != len(edges):
            raise ValueError("duplicate edge within a recipe/stage/seed")
        validated = [row for row in values if row["common_validation"]["status"] == "validated"]
        settings = [row["settings"] for row in validated]
        if settings and any(setting != settings[0] for setting in settings):
            raise ValueError("one recipe changes its fitting configuration across edges")
        valid_edges = [row["edge"] for row in validated]
        complete = valid_edges == EDGES
        first = values[0]
        group = {key: first[key] for key in ("recipe", "source_sweep", "name", "seed", "variant",
                                             "width", "depth", "fit_users", "method_label")}
        group.update(edges=edges, validated_edges=valid_edges, complete_five_edges=complete,
            source_summary=first["source_summary"], source_summary_sha256=first["source_summary_sha256"],
            source_plan=first["source_plan"], source_plan_sha256=first["source_plan_sha256"],
            failed_edges=[row["edge"] for row in values if row["common_validation"]["status"] == "failed"],
            missing_edges=[edge for edge in EDGES if edge not in valid_edges and
                not any(row["edge"] == edge and row["common_validation"]["status"] == "failed" for row in values)],
            step_zero_edges=[row["edge"] for row in values if row["step_zero"]],
            selection_eligible=complete and all(row["common_validation"].get("selection_eligible", False)
                                               for row in validated),
            common_loss_by_edge={str(row["edge"]): row["common_validation"]["metrics"]["loss"] for row in validated},
            mean_common_loss=math.fsum(row["common_validation"]["metrics"]["loss"] for row in validated) / 5
                             if complete else None,
            mean_read_mse=math.fsum(row["common_validation"]["metrics"]["read_mse"] for row in validated) / 5
                          if complete else None,
            mean_score_mse=math.fsum(row["common_validation"]["metrics"]["score_mse"] for row in validated) / 5
                           if complete else None,
            mean_raw_score_mse=math.fsum(row["common_validation"]["metrics"]["raw_score_mse"] for row in validated) / 5
                               if complete else None,
            complete_calibration_flops_estimate_sum=sum(row["calibration_flops_estimate"] for row in values),
            completed_steps_sum=sum(row["completed_steps"] for row in values),
            checkpoints_by_edge={str(row["edge"]): dict(checkpoint=row["common_validation"]["checkpoint"],
                status=row["common_validation"]["status"], best_step=row["best_step"],
                completed_steps=row["completed_steps"]) for row in values},
            settings=settings[0] if settings else first["settings"])
        groups.append(group)
    ready = (set(summaries) == set(EDGES) and all(summary["status"] in
             ("complete", "complete_with_failures") for summary in summaries.values()) and
             len(common_points) == len(inventory))
    candidates = [group for group in groups if group["variant"] == "full_item" and
        group["method_label"] == "fgh" and group["seed"] == SELECTION_SEED and group["selection_eligible"]]
    selection = dict(status="not_ready", all_five_edges_complete=ready, selection_seed=SELECTION_SEED,
        primary_family="full_item", primary_stage="fgh", future_correction_seeds=FORMAL_SEEDS,
        development_seed_robustness_established=False, training_authorized=False,
        rule="equal-edge minimum common loss; fixed 1% set, then fit_users, width, complete calibration FLOPs",
        zero_minimum_rule="if the minimum is zero, only zero mean loss is within 1%",
        missing_edges=sorted(set(EDGES) - set(summaries)), recommendation=None)
    if ready and candidates:
        minimum = min(group["mean_common_loss"] for group in candidates)
        threshold = minimum * 1.01
        tied = [group for group in candidates if group["mean_common_loss"] <= threshold]
        chosen = min(tied, key=lambda group: (group["fit_users"], group["width"],
            group["complete_calibration_flops_estimate_sum"], group["mean_common_loss"], group["recipe"]))
        selection.update(status="recommendation_for_review", minimum_mean_common_loss=minimum,
            fixed_one_percent_threshold=threshold, within_one_percent=[group["recipe"] for group in tied],
            recommendation=dict(chosen, edges=EDGES, correction_seeds=FORMAL_SEEDS,
                backbone_retraining=False,
                h_stage_selected_step_zero_edges=chosen["step_zero_edges"],
                caveat="selection uses only development seed17; seeds23/29 and population quality are not yet evidence; selected h step-zero edges do not establish a nonzero interaction benefit"))
    elif ready:
        selection["status"] = "no_complete_eligible_primary_recipe"
    comparisons = []
    by_recipe = {(group["recipe"], group["seed"], group["method_label"]): group for group in groups}
    for recipe, seed in sorted({(group["recipe"], group["seed"]) for group in groups}):
        for before, after in (("f", "fg"), ("fg", "fgh"), ("f", "f_extended"), ("fg", "fg_extended")):
            a, b = by_recipe.get((recipe, seed, before)), by_recipe.get((recipe, seed, after))
            if a is None or b is None:
                continue
            complete = a["complete_five_edges"] and b["complete_five_edges"]
            delta = {str(edge): b["common_loss_by_edge"][str(edge)] - a["common_loss_by_edge"][str(edge)]
                     for edge in EDGES} if complete else None
            comparisons.append(dict(recipe=recipe, seed=seed, before=before, after=after,
                complete_five_edges=complete, common_loss_delta_by_edge=delta,
                mean_common_loss_delta=math.fsum(delta.values()) / 5 if complete else None,
                interpretation="negative is lower validation loss; no quality/AUC claim"))
    return dict(format="ch3_common_validation_selection_v1", inventory_sha256=sha256(inventory_path),
        inventory_points=len(inventory), observed_points=len(common_points), summary_sha256=artifacts,
        analysis_source_sha256=sha256(Path(__file__)), groups=groups, points=rows,
        stage_comparisons=comparisons, selection=selection,
        native_reuse_by_edge={str(edge): summary["native_reuse"]["metrics"] for edge, summary in summaries.items()},
        common_identity_by_edge=common_by_edge,
        cost_scope="sum inventory independent-method complete calibration estimates, including source/teacher prep, all executed fitting/validation and prior nested stages; backward is estimated; shared search reuse never discounts this cost",
        scope="pre-release common validation only; all recipes, stages, seeds, failures and step-zero outcomes retained; no evaluation AUC or training")


def main(args):
    paths = args.summary
    if not paths:
        paths = [path for edge in EDGES if (path := args.common_root / f"edge_{edge}/summary.json").exists()]
    result = rollup(args.inventory, paths, plan_directory=args.plan_directory, expected_points=args.expected_points)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "summary.json", result)
    fields = ("recipe", "method_label", "seed", "variant", "fit_users", "width", "depth",
        "complete_five_edges", "validated_edges", "failed_edges", "missing_edges", "step_zero_edges",
        "mean_common_loss", "mean_read_mse", "mean_score_mse", "mean_raw_score_mse",
        "complete_calibration_flops_estimate_sum", "completed_steps_sum")
    with (args.output / "groups.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(group[key]) if isinstance(group[key], list) else group[key]
                         for key in fields} for group in result["groups"])
    write_json(args.output / "selection.json", result["selection"])
    print(json.dumps(dict(status=result["selection"]["status"], points=result["inventory_points"],
        complete_groups=sum(group["complete_five_edges"] for group in result["groups"])), allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path,
        default=ROOT / "experiments/ch3_motivation/runs/e5/common_validation_queue/attempt_1/checkpoint_inventory.json")
    parser.add_argument("--common-root", type=Path,
        default=ROOT / "experiments/ch3_motivation/runs/e5/common_validation/attempt_1")
    parser.add_argument("--summary", type=Path, nargs="*")
    parser.add_argument("--plan-directory", type=Path, default=ROOT / "experiments/ch3_motivation/configs")
    parser.add_argument("--expected-points", type=int, default=240)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
