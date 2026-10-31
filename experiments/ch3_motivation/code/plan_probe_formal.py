#!/usr/bin/env python3
"""Generate reviewable E5 fitting plans after common-validation selection.

This only writes plans. Main ladders preserve the selected fitting recipe;
the separate f population curve holds candidates/validation/update budget
fixed while estimating normalization on its own 64/128/256 gradient users.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import json
from pathlib import Path
import sys

from experiments.ch3_motivation.code.select_probe_common import ROOT, EDGES, FORMAL_SEEDS, sha256, write_json

JOB_SETTINGS = ("learning_rate", "steps", "read_weight", "score_weight", "patience",
                "validation_every", "candidate_source", "recent_fraction", "queries")


def _input_binding(root, edge):
    root = Path(root)
    population = root / "population.json"
    directory = root / f"edge_{edge}"
    inputs = directory / "inputs.json"
    candidates = directory / "calibration_feedback_candidates.json"
    binding = json.loads(inputs.read_text())
    candidate_data = json.loads(candidates.read_text())
    population_sha = sha256(population)
    if binding["population_sha256"] != population_sha or candidate_data["population_sha256"] != population_sha:
        raise ValueError("formal plan input/candidate roles changed")
    return dict(input_root=str(root.resolve()), population_sha256=population_sha,
        input_binding_sha256=sha256(inputs), candidate_file=str(candidates.resolve()),
        candidate_file_sha256=sha256(candidates),
        candidate_catalog_source_users=len(candidate_data["fitting_pool_uids"])
            if "fitting_pool_uids" in candidate_data else None)


def generate(selection_path, scope_path, *, curve_input_root=None):
    selection_path, scope_path = Path(selection_path), Path(scope_path)
    scope = json.loads(scope_path.read_text())
    if (scope["training_seeds"] != FORMAL_SEEDS or scope["edges"] != EDGES or
            scope["e5"]["main"]["seeds"] != FORMAL_SEEDS or
            scope["e5"]["fitting_population_curve"]["users"] != [64, 128, 256]):
        raise ValueError("formal scope differs from the predeclared five-edge/three-seed design")
    result = dict(status="waiting_selection", gpu_launch_performed=False, source_scope=str(scope_path.resolve()),
        source_scope_sha256=sha256(scope_path), planner_source_sha256=sha256(Path(__file__)),
        selector_source_sha256=sha256(Path(__file__).with_name("select_probe_common.py")),
        edges=EDGES, correction_seeds=FORMAL_SEEDS, backbone_retraining=False,
        main_quality=scope["e5"]["main"], confirmation_quality=scope["e5"]["confirmation"],
        e6_uses_main_fgh_without_refitting=True, main_jobs=[], curve_jobs=[],
        population_curve_policy=dict(gradient_fit_users=[64, 128, 256], teacher_fit_users=[64, 128, 256],
            candidate_catalog_source_users=256, validation_users=16,
            candidates="one fixed fit256 pre-release candidate file per edge; same query count/fraction/seed",
            training_statistics="each N estimates input statistics and fit-pool loss RMS from its own N fitting users, using the frozen fit_probe algorithm",
            interpretation="natural fitting-data-size curve, including changes in normalization estimates; not a change of gradient users with every statistic held fixed",
            common_validation_scale_source_users=64, original_validation_losses_cross_N_comparable=False,
            optimizer_budget="same step cap for all N, at least256; patience0 executes the full cap and still restores the best validated checkpoint",
            fixed_training_RMS="optional separate sensitivity, not required for this curve",
            coverage_reporting="derive unique gradient users seen as min(completed_steps,N) from the frozen shuffle-without-replacement sampler; also report best_step and its coverage; verify completed_steps equals the common cap"),
        controls=dict(f_extended="independent copy of best f, additional2x base stage budget",
            fg_extended="independent copy of best fg, additional1x base stage budget",
            branch_rule="extended controls never replace the f/fg predecessor used by the main ladder",
            actual_steps="retain actual early stopping and charge all executed calibration work"),
        cost_scope=scope["common"]["cost_model_selection_scope"],
        prerequisites="root reviews selected settings and schedules concrete jobs; this generator never starts GPU work")
    if not selection_path.exists():
        return result
    selection = json.loads(selection_path.read_text())
    result.update(source_selection=str(selection_path.resolve()), source_selection_sha256=sha256(selection_path))
    if selection["status"] != "recommendation_for_review" or not selection["all_five_edges_complete"]:
        return result
    chosen = selection["recommendation"]
    if (selection["selection_seed"] != 17 or chosen["variant"] != "full_item" or
            chosen["method_label"] != "fgh" or chosen["edges"] != EDGES or
            chosen["correction_seeds"] != FORMAL_SEEDS):
        raise ValueError("selection differs from the predeclared primary family/stage/seeds")
    settings = chosen["settings"]
    if settings.get("include_item", True) is not True or settings.get("normalize", True) is not True:
        raise ValueError("selected variant requires a driver setting not used by the frozen fitting runner")
    fixed = dict(variant="full_item", width=chosen["width"], depth=chosen["depth"],
        fit_users=chosen["fit_users"], input_root=settings["input_root"],
        **{key: settings[key] for key in JOB_SETTINGS})
    curve_root = Path(curve_input_root or ROOT / "experiments/ch3_motivation/inputs_feedback_pool/fit256")
    binding_main, binding_curve = {}, {}
    for edge in EDGES:
        binding_main[str(edge)] = _input_binding(fixed["input_root"], edge)
        binding_curve[str(edge)] = _input_binding(curve_root, edge)
        if (binding_curve[str(edge)]["candidate_catalog_source_users"] != 256 or
                binding_main[str(edge)]["population_sha256"] != binding_curve[str(edge)]["population_sha256"]):
            raise ValueError("curve candidates require the fixed fit256 catalog and original reserved population")
    curve_steps = max(256, int(fixed["steps"]))
    for edge in EDGES:
        for seed in FORMAL_SEEDS:
            result["main_jobs"].append(dict(fixed, name="selected_main", edge=edge, seed=seed,
                stages="f,fg,fgh", controls=True))
            for count in (64, 128, 256):
                result["curve_jobs"].append(dict(fixed, name=f"selected_f_curve_fit{count}", edge=edge,
                    seed=seed, fit_users=count, input_root=str(curve_root.resolve()),
                    stages="f", controls=False, patience=0, steps=curve_steps))
    result.update(status="prepared_for_root_review", selected_recipe=chosen["recipe"],
        selected_development_seed=17, selected_h_step_zero_edges=chosen["step_zero_edges"],
        numerical_constants={key: settings[key] for key in ("row_chunk", "query_chunk", "weight_decay", "gradient_clip")
                             if key in settings},
        main_input_bindings=binding_main, curve_input_bindings=binding_curve,
        main_training_recipe="exact selected input view, step cap, learning rate, losses and early stopping; only add the predeclared seeds and sibling controls",
        curve_training_recipe="same selected width/depth/learning rate/loss weights/query rule; one fit256 candidate view, own-N statistics, identical update cap and patience0 across all three N",
        curve_steps_per_job=curve_steps, reuse_main_f_for_curve=False,
        job_counts=dict(main_ladders=15, main_stage_weights_with_controls=75, curve_f_weights=45,
            main_quality_points=75, curve_quality_points=45, confirmation_quality_points=15,
            confirmation_additional_fitting=0),
        evaluation_reuse="confirmation uses the seed17 primary ladder weights on its separate fixed2000 users; E6 uses the matched per-edge/per-seed main fgh weights",
        population_accounting=[dict(gradient_fit_users=n, teacher_fit_users=n,
            candidate_catalog_source_users=256, private_validation_users=16,
            training_loss_RMS_source_users=n, common_validation_loss_RMS_source_users=64) for n in (64, 128, 256)],
        existing_seed17_fgh_sources=chosen["checkpoints_by_edge"],
        existing_weight_reuse="may reuse exactly matching frozen seed17 branches after identity/hash audit; never silently reuse a main f point for the different curve recipe",
        seed_claim="configuration selected on17 only; 23/29 are frozen repetitions, not another recipe-selection opportunity")
    return result


def main(args):
    result = generate(args.selection, args.scope, curve_input_root=args.curve_input_root)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "plan.json", result)
    if result["status"] == "prepared_for_root_review":
        write_json(args.output / "main_fitting.json", dict(status=result["status"], jobs=result["main_jobs"],
            scope="fitting only; root schedules execution; no automatic quality launch"))
        write_json(args.output / "f_population_curve.json", dict(status=result["status"], jobs=result["curve_jobs"],
            population_accounting=result["population_accounting"], policy=result["population_curve_policy"]))
    print(json.dumps(dict(status=result["status"], gpu_launch_performed=False,
                         main_jobs=len(result["main_jobs"]), curve_jobs=len(result["curve_jobs"]))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path,
        default=ROOT / "experiments/ch3_motivation/runs/e5/common_selection/attempt_1/selection.json")
    parser.add_argument("--scope", type=Path, default=ROOT / "experiments/ch3_motivation/configs/formal_quality_scope.json")
    parser.add_argument("--curve-input-root", type=Path,
        default=ROOT / "experiments/ch3_motivation/inputs_feedback_pool/fit256")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
