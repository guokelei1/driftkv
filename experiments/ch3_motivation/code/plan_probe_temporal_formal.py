#!/usr/bin/env python3
"""Prepare formal jobs from the new temporal-common recommendation; no launch.

Release-gap choices use the existing frozen planner. Temporal choices produce
the actual supported fixed fit64 main recipe, while their separate fixed-law,
fit256-catalog population curve stays explicitly pending implementation.
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

from experiments.ch3_motivation.code import calibration_gaps, plan_probe_formal
from experiments.ch3_motivation.code.run_probe_temporal import FIXED
from experiments.ch3_motivation.code.select_probe_common import ROOT, EDGES, FORMAL_SEEDS, sha256, write_json


def generate(selection_path, scope_path, *, curve_input_root=None):
    result = plan_probe_formal.generate(selection_path, scope_path, curve_input_root=curve_input_root)
    result.update(planner_source_sha256=sha256(Path(__file__)),
        selector_source_sha256=sha256(Path(__file__).with_name("select_probe_temporal_common.py")),
        legacy_planner_source_sha256=sha256(Path(plan_probe_formal.__file__)),
        selection_scope="current280 search-stage suggestion only; expanded calibration search remains separate; no final3seed freeze or launch")
    if result["status"] != "prepared_for_root_review":
        return result
    selection = json.loads(Path(selection_path).read_text())
    chosen = selection["recommendation"]
    settings = chosen["settings"]
    driver, rule = chosen["fitting_driver"], chosen["query_delta_rule"]
    if settings["fitting_driver"] != driver or settings["query_delta_rule"] != rule:
        raise ValueError("recommended driver/rule disagrees with its settings")
    source = ROOT / "experiments/ch3_motivation/code" / driver
    if chosen["fitting_driver_sha256"] != sha256(source):
        raise ValueError("selected fitting driver source changed")
    if driver == "run_probe_variants.py":
        if rule != "cutover":
            raise ValueError("legacy fitting driver only supports release-gap calibration")
        for job in result["main_jobs"] + result["curve_jobs"]:
            job.update(fitting_driver=driver, fitting_driver_sha256=sha256(source), query_delta_rule="cutover")
        result["selected_query_delta_rule"] = "cutover"
        return result
    if driver != "run_probe_temporal.py" or rule not in ("gap_mix50", "gap_empirical100"):
        raise ValueError("unsupported actual fitting driver/rule")
    for key, expected in FIXED.items():
        actual = chosen[{"width": "width", "depth": "depth", "fit_users": "fit_users"}.get(key, key)] if key in (
            "variant", "width", "depth", "fit_users") else settings.get(key)
        if actual != expected:
            raise ValueError(f"selected temporal {key} is not supported by the fixed temporal fitting driver")
    bindings, jobs = {}, []
    for edge in EDGES:
        provenance = chosen["gap_provenance_by_edge"][str(edge)]
        binding = result["main_input_bindings"][str(edge)]
        manifest = Path(provenance["gap_manifest"])
        views, actual = calibration_gaps.load_gap_manifest(manifest, binding["input_root"], edge, rule)
        if (sha256(manifest) != provenance["gap_manifest_sha256"] or
                provenance["fitting_driver"] != driver or provenance["fitting_driver_sha256"] != sha256(source) or
                provenance["query_delta_rule"] != rule or actual["fallback_used"] or
                any(provenance[key] != actual[key] for key in ("time_rule", "source_cutoff", "candidate_order")) or
                len(actual["fit_uids"]) != 64 or len(actual["validation_uids"]) != 16 or len(views) != 80):
            raise ValueError("selected per-edge temporal binding changed")
        bindings[str(edge)] = {**binding, **provenance, "gap_manifest": str(manifest.resolve()),
            "time_law_source_fit_users": 64, "candidate_catalog_source_users": 64}
        for seed in FORMAL_SEEDS:
            jobs.append(dict(FIXED, input_root=settings["input_root"], name="selected_main",
                edge=edge, seed=seed, stages="f,fg,fgh", controls=True, fitting_driver=driver,
                fitting_driver_sha256=sha256(source), query_delta_rule=rule, gap_rule=rule,
                gap_manifest=str(manifest.resolve()), gap_manifest_sha256=sha256(manifest),
                time_rule=actual["time_rule"], source_cutoff=actual["source_cutoff"],
                candidate_order=actual["candidate_order"], query_delta_layout=provenance["query_delta_layout"]))
    # The existing temporal CLI intentionally fits64 only. Its immutable main
    # manifest cannot supply new128/256 target users or a different catalog.
    result.update(status="prepared_for_root_review", main_status="prepared_for_root_review",
        main_jobs=jobs, main_input_bindings=bindings, curve_jobs=[], curve_input_bindings={},
        selected_fitting_driver=driver, selected_fitting_driver_sha256=sha256(source),
        selected_query_delta_rule=rule, source_temporal_common_selection_sha256=sha256(selection_path),
        main_training_recipe="exact selected temporal rule, fit64 candidate IDs and per-edge manifest; correction seeds17/23/29 and independent sibling controls",
        curve_plan_status="pending_temporal_curve_driver",
        curve_pending=dict(gradient_fit_users=[64,128,256], target_fit_users=256,
            candidate_catalog_source_users=256, time_law_source_fit_users=64, validation_users=16,
            manifest="new separate curve manifest: fit256+val16 targets on fixed catalog256, assigned by the already frozen fit64 law; main manifest remains unchanged",
            identity_fields=["time_law_source_fit_uids", "target_fit_uids", "candidate_catalog_source_users",
                "parent_fit64_gap_manifest_sha256", "validation_uids", "query_delta_rule", "candidate_order"],
            fitting="new concrete driver for prefixN teacher/input/RMS/gradient users, identical rule/LR/Q16/cap, patience0; not implemented here",
            executable_jobs_generated=False),
        temporal_dispatch=dict(driver=driver,
            accepted_cli_fields=["input_root", "output", "edge", "seed", "gap_manifest", "gap_rule", "controls", "device", "gpu"],
            metadata_fields="verify FIXED settings against the driver; do not pass old-driver architecture/optimizer flags to this fixed temporal CLI"),
        job_counts=dict(main_ladders=15, main_stage_weights_with_controls=75, curve_f_weights=0,
            planned_curve_f_weights_after_implementation=45, main_quality_points=75,
            confirmation_quality_points=15, confirmation_additional_fitting=0),
        quality_plan_gate="main15 and its confirmation are independently reviewable; curve explicitly waits for preparation; no automatic launch")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--scope", type=Path,
        default=ROOT / "experiments/ch3_motivation/configs/formal_quality_scope.json")
    parser.add_argument("--curve-input-root", type=Path,
        default=ROOT / "experiments/ch3_motivation/inputs_feedback_pool/fit256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.selection, args.scope, curve_input_root=args.curve_input_root)
    write_json(args.output / "plan.json", result)
    if result["main_jobs"]:
        write_json(args.output / "main_fitting.json", dict(status=result.get("main_status", result["status"]),
            jobs=result["main_jobs"], scope="plan only; use each job's real fitting_driver"))
    print(json.dumps(dict(status=result["status"], main_jobs=len(result["main_jobs"]),
                         curve_jobs=len(result["curve_jobs"]), gpu_launched=False)))
