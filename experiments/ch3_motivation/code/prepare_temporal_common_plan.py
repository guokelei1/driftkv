#!/usr/bin/env python3
"""Freeze the new fit64-ID, three-context common validation before scoring."""
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

from experiments.ch3_motivation.code.select_probe_common import ROOT, sha256, write_json

RULES = ["cutover", "gap_mix50", "gap_empirical100"]
OLD_SOURCES = [("f_row_input_sweep", "f_row_input_sweep.json"),
    ("ladder_development", "probe_ladder_development.json"),
    ("ladder_stability", "probe_ladder_stability.json"),
    ("ladder_feedback_pool", "probe_ladder_feedback_pool.json")]


def _labels(stages, controls):
    result = []
    for stage in stages.split(","):
        result.append(stage)
        if controls and stage in ("f", "fg"):
            result.append(stage + "_extended")
    return result


def prepare(*, config_root=None, runs_root=None, input_root=None, gap_root=None):
    config_root = Path(config_root or ROOT / "experiments/ch3_motivation/configs")
    runs_root = Path(runs_root or ROOT / "experiments/ch3_motivation/runs/e5")
    input_root = Path(input_root or ROOT / "experiments/ch3_motivation/inputs_feedback_pool/fit64")
    gap_root = Path(gap_root or runs_root / "calibration_gap_views/attempt_1")
    points, sources = [], {}
    for sweep, filename in OLD_SOURCES:
        path = config_root / filename
        source = json.loads(path.read_text())
        sources[str(path.resolve())] = sha256(path)
        for job in source["jobs"]:
            seed, edge = job.get("seed", 17), job["edge"]
            for label in _labels(job.get("stages", "f"), job.get("controls", False)):
                checkpoint = runs_root / sweep / "attempt_1" / job["name"] / f"edge_{edge}" / f"seed_{seed}" / f"{label}.pt"
                points.append(dict(origin="release_gap_240", recipe=f"{sweep}/{job['name']}",
                    edge=edge, seed=seed, method_label=label, checkpoint=str(checkpoint.resolve()),
                    fitting_driver="run_probe_variants.py"))
    old_count = len(points)
    temporal_path = config_root / "probe_temporal_calibration.json"
    temporal = json.loads(temporal_path.read_text())
    sources[str(temporal_path.resolve())] = sha256(temporal_path)
    for job in temporal["jobs"]:
        for label in _labels("f,fg,fgh", job["controls"]):
            points.append(dict(origin="temporal_40", recipe=f"temporal_calibration/{job['name']}",
                edge=job["edge"], seed=job["seed"], method_label=label,
                checkpoint=str((Path(job["output"]) / f"{label}.pt").resolve()),
                fitting_driver="run_probe_temporal.py", query_delta_rule=job["gap_rule"],
                training_gap_manifest_sha256=sha256(Path(job["gap_manifest"]))))
    if old_count != 240 or len(points) - old_count != 40 or len({p["checkpoint"] for p in points}) != 280:
        raise ValueError("expected all original240 and new40 distinct checkpoints")
    bindings = {}
    population_sha = sha256(input_root / "population.json")
    for edge in (1, 2, 3, 4, 5):
        directory = input_root / f"edge_{edge}"
        manifest = gap_root / f"edge_{edge}/gap_manifest.json"
        value = json.loads(manifest.read_text())
        candidate = directory / "calibration_feedback_candidates.json"
        if (value["edge"] != edge or value["input_binding"]["population_sha256"] != population_sha or
                value["input_binding"]["candidate_sha256"] != sha256(candidate) or
                Path(value["input_binding"]["input_root"]).resolve() != input_root.resolve() or
                value["gap_law"]["fallback_used"]):
            raise ValueError("temporal common context must use the strict original fit64 gap/candidate binding")
        bindings[str(edge)] = dict(input_root=str(input_root.resolve()),
            input_binding_sha256=sha256(directory / "inputs.json"), population_sha256=population_sha,
            candidate_file=str(candidate.resolve()), candidate_file_sha256=sha256(candidate),
            gap_manifest=str(manifest.resolve()), gap_manifest_sha256=sha256(manifest))
    here = Path(__file__).resolve()
    executor_paths = [here.with_name(name) for name in ("validate_probe_temporal_common.py",
        "validate_probe_common.py", "calibration_gaps.py", "common.py", "probes.py",
        "probe_variants.py", "prepare_candidates.py")]
    executor_paths.append(ROOT / "src/hstu_kvcache/adaptation/reader.py")
    return dict(format="ch3_temporal_common_plan_v1", status="frozen_before_new_common_scores",
        planner_sha256=sha256(Path(__file__)), source_fitting_plans_sha256=sources,
        executor_sources_sha256={str(p): sha256(p) for p in executor_paths}, edge_bindings=bindings,
        rules=RULES, expected_old_weights=240, expected_temporal_weights=40, expected_weights=280,
        expected_context_weight_cells=840, weights=points,
        protocol=dict(gradient_fitting_performed=False, population_auc_used=False, queries=16,
            observed_feedback_fraction=.5, candidate_seed=17, validation_users=16,
            candidate_catalog_source_fit_users=64, time_law_source_reserved_fit_users=64,
            loss_scale_unique_fit_users=64, loss_scale_user_context_observations=192,
            scale_policy="one per-edge RMS from the same64 fitting users crossed with all3 contexts equally; share it across contexts and every weight; validation never sets scales",
            teacher_cache="same current-model rebuild of the strict pre-release prefix",
            input_semantics="synthetic query-gap features on fixed cutover caches, not historical chronological query replay"),
        selection=dict(status="predeclared_before_common_scores", primary_family="full_item", primary_stage="fgh",
            development_seed=17, future_correction_seeds=[17, 23, 29],
            primary_objective="mean over all3 context losses within each edge, then equal mean over all5 edges; one global recipe for all edges",
            complete_requirement="all5 edges and all3 contexts for a recipe; pending/failed cells are never averaged away",
            near_minimum="fixed set whose mean loss is <=1.01 times the global minimum; if minimum0 only0 qualifies",
            tie_order=["fewer fit_users", "smaller width", "smaller sum of independent-method cumulative calibration FLOPs", "smaller mean loss", "recipe name"],
            step_zero="retain and permit comparison; selected fgh step0 does not establish a nonzero interaction benefit",
            failed="retain every planned weight and failure; a failed recipe cannot be selected",
            forbidden="no per-edge recipe selection, no dev/main/confirmation AUC, no old fit256-common loss pooled into the new objective"),
        prior_comparison="the previous240 common validation used fit256 candidate IDs and cutover-only scales; retain it separately. Even the new cutover context is a new fit64-ID/shared3context-scale problem.",
        formal_planner_extension=dict(original_frozen_files_modified=False, automatic_gpu_launch=False,
            dispatch="release-gap recipe keeps run_probe_variants.py; temporal recipe must explicitly dispatch run_probe_temporal.py",
            temporal_identity_fields=["fitting_driver", "fitting_driver_sha256", "query_delta_rule",
                "gap_manifest", "gap_manifest_sha256", "time_rule", "source_cutoff", "candidate_order"],
            main="same selected rule and per-edge manifest bindings for seeds17/23/29, same native f/fg/fgh plus sibling controls; strict identity/hash audit before any seed17 reuse",
            curve="current temporal driver/manifests support fit64 only. Prepare a separate curve manifest for all fit256+val16 target UIDs on the fixed fit256 candidate catalog, assigning the same selected rule using the already fixed fit64 law. N64/128/256 changes only the gradient/teacher/input-statistics UID prefix and its natural fitting RMS. Record gradientN, catalog256 and time-law64 separately; never silently run a release-gap curve and label it temporal.",
            structure="retain main_jobs/curve_jobs and main_input_bindings/curve_input_bindings; every temporal job records its real fitting_driver, query_delta_rule and exact per-edge gap-manifest path/SHA. Keep main and curve manifests separate.",
            multiedge_identity="within an edge require exact gap-manifest path/SHA/cutoff for all stages/seeds; across edges bind these natural edge-specific fields through the formal plan while retaining driverSHA, rule, time_rule, candidate_order and layout",
            confirmation="reuse exact seed17 stage weights on its disjoint fixed population; no new fitting",
            safety="original planner and selector stay frozen; this plan is a review artifact, not GPU launch authorization"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = prepare()
    write_json(args.output, value)
    print(json.dumps(dict(status=value["status"], weights=value["expected_weights"], cells=840,
                         output=str(args.output), gpu_launched=False)))
