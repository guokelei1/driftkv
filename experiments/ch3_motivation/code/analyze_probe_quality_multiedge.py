#!/usr/bin/env python3
"""Five-edge E5 AUC differences with one paired UID draw per fixed panel.

Main and confirmation retain separate populations and predeclared seed counts.
Only a complete edge/seed/method grid produces the primary aggregate. Recovery
is a ratio of equal-edge mean AUC differences; seed point ranges and conditional
UID intervals describe different sources of variation.
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

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import torch

from experiments.ch3_motivation.code.analyze_probe_quality import (
    ShardLoader, _context, _same_events, _interval,
)
from experiments.ch3_motivation.code import plan_probe_quality as quality
from experiments.ch3_motivation.code.probe_cost_export import (
    collect_costs, write_cost_csvs, sources as cost_export_sources,
)
from experiments.ch3_motivation.code.common import write_json
from hstu_kvcache.evaluation.statistics import weighted_auc
from hstu_kvcache.common.artifacts import sha256

DEFAULT_SCOPE = ROOT/"experiments/ch3_motivation/configs/formal_quality_scope.json"
DEFAULT_FORMAL_PLAN = ROOT/"experiments/ch3_motivation/runs/e5/formal_plan/attempt_1/plan.json"
CONTRASTS = dict(g=("fg", "f"), h=("fgh", "fg"),
    f_extended_minus_f=("f_extended", "f"), fg_extended_minus_fg=("fg_extended", "fg"),
    fg_minus_f_extended=("fg", "f_extended"), fgh_minus_fg_extended=("fgh", "fg_extended"),
    fgh_minus_f_extended=("fgh", "f_extended"))
CONFIG_FIELDS = ("variant", "fit_users", "width", "depth")
JOB_SETTINGS = ("learning_rate", "steps", "read_weight", "score_weight", "patience",
    "validation_every", "candidate_source", "recent_fraction", "queries", "stages", "controls")
EDGE_INPUT_FIELDS = ("sampling_manifest", "sampling_manifest_sha256", "gap_manifest",
    "gap_manifest_sha256", "source_cutoff")
TIME_INPUT_FIELDS = (*EDGE_INPUT_FIELDS, "time_rule", "candidate_order", "query_delta_layout",
    "candidate_catalog_source_users", "time_law_source_users", "gradient_fitting_users",
    "teacher_fitting_users")
INITIALIZATION_FIELDS = ("h_initialization", "h_initialization_spec", "initialization_helper_sha256")


def _training_record(summary, payload):
    """Read the actual tagged/legacy-head fitting metadata without building a GPU model."""
    metadata, config = payload["metadata"], payload["config"]
    identity, evaluation = metadata["identity"], summary["identity"]
    variant = payload.get("variant", "head")
    if variant not in ("head", "full_item", "layerwise_kv"):
        raise ValueError("unknown typed checkpoint variant")
    if (("variant" in payload or "format" in payload) and payload.get("format") != "ch3_probe_variant_v1"):
        raise ValueError("unknown typed checkpoint format")
    if variant == "head" and "variant" not in payload and config["item_dim"] != config["head_dim"]:
        raise ValueError("untagged checkpoint is not a head-local probe")
    label = metadata.get("method_label", metadata["stage"])
    stage = {"f_extended": "f", "fg_extended": "fg"}.get(label, label)
    if (label != summary["method_label"] or payload["stage"] != stage or metadata["stage"] != stage or
        variant != summary["variant"] or metadata.get("variant", variant) != variant or
        int(metadata["seed"]) != summary["seed"] or int(identity["seed"]) != summary["seed"] or
        int(identity["edge"]) != summary["edge"] or identity["models"] != evaluation["models"] or
        config["hidden_width"] != summary["width"] or config["depth"] != summary["depth"] or
        metadata["training_users"] != summary["fit_users"]):
        raise ValueError("checkpoint training seed/stage/architecture/edge differs from quality summary")
    if metadata.get("diagnostic_only") or metadata.get("quality_use_permitted") is False or metadata.get("fitting_uses_future_feedback"):
        raise ValueError("diagnostic or future-feedback-fitted weights cannot define formal E5 quality")
    fitting, validation = identity["fit_uids"], identity["validation_users"]
    if (len(fitting) != summary["fit_users"] or len(set(fitting)) != len(fitting) or
        not validation or len(set(validation)) != len(validation) or set(fitting) & set(validation) or
        set(evaluation["users"]) & set(fitting+validation)):
        raise ValueError("checkpoint fitting/validation/evaluation roles differ or overlap")
    calibration = (evaluation.get("scenario") or {}).get("calibration_input_binding_sha256", evaluation["input_binding_sha256"])
    if identity["input_binding_sha256"] != calibration:
        raise ValueError("checkpoint calibration source differs from preserved evaluation input binding")
    normalization = metadata.get("fit_pool_loss_scales", {})
    if (normalization.get("version") != "fit_pool_equal_user_length_normalized_v1" or
            normalization.get("source") != "fitting_users_only"):
        raise ValueError("formal E5 requires the corrected fixed fitting-pool loss normalization")
    return dict(identity=identity, config=config, variant=variant,
        objective=metadata["objective"], normalization_version=normalization.get("version"),
        normalization_source=normalization.get("source"),
        checkpoint_sha256=evaluation["checkpoint_sha256"], best_step=metadata["best_step"],
        completed_steps=metadata["completed_steps"])


def _recipe(training, *, across_edges):
    excluded = {"seed"}
    if across_edges:
        excluded.update(("edge", "models", "input_binding_sha256", "calibration_candidate_sha256"))
        # These vary naturally by release, but may be discarded only after the
        # prepared main job and the actual checkpoint/input files agree.
        if training.get("formal_main_calibration_bound"):
            excluded.update(EDGE_INPUT_FIELDS)
    return dict(identity={key: value for key, value in training["identity"].items() if key not in excluded},
        config=training["config"], variant=training["variant"], objective=training["objective"],
        normalization_version=training["normalization_version"], normalization_source=training["normalization_source"])


def _check_planned_job(item, job, binding, population, constants, *, audit_checkpoint=False):
    """Check metadata in memory; formal main additionally audits the real files.

    The population-curve analyzer also uses the metadata-only interface before
    its own manifest audit, so no input-normalization permission is set here.
    """
    summary, training = item["summary"], item["training"]
    identity = training["identity"]
    if any(summary[key] != job[key] for key in ("edge", "seed", "variant", "fit_users", "width", "depth")):
        raise ValueError("quality checkpoint architecture/seed differs from prepared main job")
    if any(identity.get(key) != job[key] for key in JOB_SETTINGS):
        raise ValueError("quality checkpoint recipe differs from prepared main job")
    if any(identity.get(key) != value for key, value in constants.items()):
        raise ValueError("quality checkpoint numerical constants differ from prepared plan")
    if (identity["input_binding_sha256"] != binding["input_binding_sha256"] or
        identity["population_sha256"] != binding["population_sha256"] or
        identity.get("calibration_candidate_sha256") != (binding["candidate_file_sha256"] if job["candidate_source"] == "feedback" else None) or
        identity["fit_uids"] != population["fit"][:job["fit_users"]] or
        identity["validation_users"] != population["validation"] or
        Path(job["input_root"]).resolve() != Path(binding["input_root"]).resolve()):
        raise ValueError("quality checkpoint input/candidate/role binding differs from prepared main job")
    if identity.get("query_delta_rule", "cutover") != job.get("query_delta_rule", "cutover"):
        raise ValueError("quality checkpoint temporal rule differs from prepared job")
    driver = identity.get("fitting_driver", "run_probe_variants.py")
    if "fitting_driver" in job or "fitting_driver" in identity:
        driver_sha = identity.get("fitting_driver_sha256", identity.get("numerical_sources", {}).get(
            f"experiments/ch3_motivation/code/{driver}"))
        if (driver != job.get("fitting_driver") or driver_sha != job.get("fitting_driver_sha256")):
            raise ValueError("quality checkpoint fitting driver/SHA differs from prepared job")
    for key in (*TIME_INPUT_FIELDS, *INITIALIZATION_FIELDS):
        if key not in identity and key not in job:
            continue
        actual, planned = identity.get(key), job.get(key)
        if key in ("sampling_manifest", "gap_manifest") and actual and planned:
            actual, planned = str(Path(actual).resolve()), str(Path(planned).resolve())
        if actual != planned:
            raise ValueError(f"quality checkpoint {key} differs from prepared job")
    if driver in quality.SAMPLING_DRIVERS:
        if any(key not in identity or key not in job for key in TIME_INPUT_FIELDS):
            raise ValueError("quality sampling job lacks its explicit input/actual-N identity")
        if (binding.get("sampling_manifest_sha256") != identity["sampling_manifest_sha256"] or
            not binding.get("sampling_manifest") or
            Path(binding["sampling_manifest"]).resolve() != Path(identity["sampling_manifest"]).resolve()):
            raise ValueError("quality sampling manifest differs from prepared input binding")
        if driver == "run_probe_calibration_init.py" and any(
                key not in identity or key not in job for key in INITIALIZATION_FIELDS):
            raise ValueError("quality initialization job lacks its explicit initialization identity")
    elif any(key in identity for key in ("sampling_manifest", "sampling_manifest_sha256")):
        raise ValueError("quality sampling identity has no recognized explicit fitting driver")
    if audit_checkpoint:
        checkpoint = Path(summary["identity"]["checkpoint"])
        audit = quality.audit_weight(checkpoint, job, binding, summary["method_label"])
        if (audit["checkpoint_sha256"] != summary["identity"]["checkpoint_sha256"] or
                audit["checkpoint_sha256"] != training["checkpoint_sha256"]):
            raise ValueError("audited fitting checkpoint differs from quality score weight binding")
        return audit


def _formal_plan_audit(loaded, plan, loader, scope_path):
    if plan is None or plan.get("status") != "prepared_for_root_review":
        return False
    loader.verify(plan["source_scope"], sha256(scope_path))
    if plan["source_scope_sha256"] != sha256(scope_path):
        raise ValueError("prepared formal plan scope binding changed")
    jobs = {(job["edge"], job["seed"]): job for job in plan["main_jobs"]}
    if len(jobs) != 15 or len(plan["main_jobs"]) != 15 or set(jobs) != {
            (edge, seed) for edge in range(1, 6) for seed in (17, 23, 29)}:
        raise ValueError("prepared formal plan does not contain all fifteen main ladders")
    if any(job["stages"] != "f,fg,fgh" or job["controls"] is not True for job in jobs.values()):
        raise ValueError("prepared main jobs require all three nested stages and both extended controls")
    populations = {}
    for edge in range(1, 6):
        binding = plan["main_input_bindings"][str(edge)]
        root = Path(binding["input_root"])
        loader.verify(root/"population.json", binding["population_sha256"])
        loader.verify(root/f"edge_{edge}/inputs.json", binding["input_binding_sha256"])
        loader.verify(binding["candidate_file"], binding["candidate_file_sha256"])
        populations[edge] = loader.json(root/"population.json")
    for item in loaded:
        s = item["summary"]
        if ((s["identity"]["panel"] == "main" and s["seed"] in (17, 23, 29)) or
                (s["identity"]["panel"] == "confirmation" and s["seed"] == 17)):
            if s["identity"]["users"] != populations[s["edge"]][s["identity"]["panel"]]:
                raise ValueError("quality UID order differs from the prepared main/confirmation population")
            job = jobs[s["edge"], s["seed"]]
            audit = _check_planned_job(item, job, plan["main_input_bindings"][str(s["edge"])],
                populations[s["edge"]], plan.get("numerical_constants", {}), audit_checkpoint=True)
            for key in ("sampling_manifest", "gap_manifest"):
                if job.get(key):
                    loader.verify(Path(job[key]), job[f"{key}_sha256"])
            item["training"].update(formal_main_calibration_bound=True, planned_weight_audit=audit)
    return True


def _scope(scope):
    """The formal experiment's existing scope, not a new population contract."""
    if scope["edges"] != [1, 2, 3, 4, 5]:
        raise ValueError("formal E5 aggregation retains all five adjacent edges")
    expected = {}
    for panel in ("main", "confirmation"):
        value = scope["e5"][panel]
        expected[panel] = dict(users=int(value["users"]), seeds=list(value["seeds"]),
            methods=list(value.get("methods", value["stages"])))
    if (expected["main"] != dict(users=2000, seeds=[17, 23, 29],
            methods=["f", "fg", "fgh", "f_extended", "fg_extended"]) or
        expected["confirmation"] != dict(users=2000, seeds=[17], methods=["f", "fg", "fgh"])):
        raise ValueError("E5 formal main/confirmation scope differs from the frozen experiment")
    return expected


def _ratio(point, draws, gap, gap_draws):
    valid = np.isfinite(gap_draws) & (gap_draws > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = 100*draws/gap_draws
    return dict(point=float(100*point/gap) if gap != 0 else None,
        conditional_uid_ci95=_interval(values) if gap > 0 and valid.all() else None,
        nonpositive_or_invalid_gap_replicates=int((~valid).sum()),
        positive_reference_gap_point=bool(gap > 0))


def _method_metric(values, reuse, rebuild):
    gain, gap = values-reuse, rebuild-reuse
    ratio = _ratio(gain[0], gain[1:], gap[0], gap[1:])
    return dict(mean_auc=float(values[0]), conditional_uid_auc_ci95=_interval(values[1:]),
        mean_auc_gain_over_reuse=float(gain[0]), conditional_uid_gain_ci95=_interval(gain[1:]),
        mean_auc_minus_rebuild=float(values[0]-rebuild[0]),
        conditional_uid_minus_rebuild_ci95=_interval(values[1:]-rebuild[1:]),
        recovery_percent=ratio["point"], conditional_uid_recovery_ci95=ratio["conditional_uid_ci95"],
        valid_auc_replicates=int(np.isfinite(values[1:]).sum()))


def _contrast_metric(delta, gap):
    ratio = _ratio(delta[0], delta[1:], gap[0], gap[1:])
    return dict(mean_auc_difference=float(delta[0]), conditional_uid_auc_difference_ci95=_interval(delta[1:]),
        recovery_percentage_point_difference=ratio["point"],
        conditional_uid_recovery_difference_ci95=ratio["conditional_uid_ci95"],
        valid_auc_replicates=int(np.isfinite(delta[1:]).sum()))


def _seed_status(seeds, expected):
    seeds, expected = sorted(seeds), sorted(expected)
    complete = seeds == expected
    return dict(observed_seeds=seeds, observed_seed_count=len(seeds), expected_seeds=expected,
        expected_seed_count=len(expected), planned_seeds_complete=complete,
        missing_expected_seeds=sorted(set(expected)-set(seeds)),
        three_seed_complete=complete and len(expected) == 3,
        training_seed_robustness_evaluated=complete and len(expected) == 3,
        status=("complete_three_seed_grid" if len(expected) == 3 else "complete_predeclared_single_seed_grid")
            if complete else "incomplete_planned_seed_grid",
        interpretation=("three planned correction-training seeds; a complete grid is required; point range is distinct from conditional UID uncertainty"
            if len(expected) == 3 else "population confirmation at predeclared seed17; no three-seed robustness estimate"))


def _panel(loaded, panel, expected, edges, *, replicates, seed, formal_plan_ready):
    users, configuration, contexts, anchors, points = None, None, {}, {}, {}
    edge_recipes, common_recipe = {}, None
    extras = []
    for item in loaded:
        summary, identity = item["summary"], item["summary"]["identity"]
        if identity["panel"] != panel:
            continue
        edge, correction_seed, method = int(summary["edge"]), int(summary["seed"]), summary["method_label"]
        if edge not in edges or correction_seed not in expected["seeds"] or method not in expected["methods"]:
            extras.append(dict(edge=edge, seed=correction_seed, method=method, source_summary=item["path"],
                reason="outside this panel's frozen edge/seed/method grid; retained, not pooled"))
            continue
        if len(identity["users"]) != expected["users"]:
            raise ValueError("formal E5 panel must contain its fixed 2000 eligible users")
        if users is not None and identity["users"] != users:
            raise ValueError("panel UID membership/order differs across edges, methods, or seeds")
        users = identity["users"]
        config = {field: summary[field] for field in CONFIG_FIELDS}
        if configuration is not None and config != configuration:
            raise ValueError("provide one predeclared common fitting configuration; quality cannot select among configurations")
        configuration = config
        training = item["training"]
        edge_recipe = _recipe(training, across_edges=False)
        recipe = _recipe(training, across_edges=True)
        if edge in edge_recipes and edge_recipe != edge_recipes[edge]:
            raise ValueError("same-edge stages/seeds do not share the same training identity except seed")
        if common_recipe is not None and recipe != common_recipe:
            raise ValueError("cross-edge training recipes differ beyond natural edge/model/input/candidate bindings")
        edge_recipes[edge], common_recipe = edge_recipe, recipe
        context = _context(identity)
        if edge in contexts and context != contexts[edge]:
            raise ValueError("same-edge E5 population/models/input/source/backend context differs")
        if contexts and any(context[key] != next(iter(contexts.values()))[key] for key in ("sources", "backend")):
            raise ValueError("cross-edge E5 numerical sources/backend differ")
        contexts[edge] = context
        if edge in anchors:
            for tag in ("reuse", "rebuild"):
                old, new = anchors[edge][tag], item["anchors"][tag]
                if not _same_events(old, new) or [row["score"] for row in old] != [row["score"] for row in new]:
                    raise ValueError("matched E5 anchor events/scores differ across methods or seeds")
        anchors[edge] = item["anchors"]
        key = edge, correction_seed, method
        if key in points:
            raise ValueError("duplicate edge/seed/method; select one predeclared fitting run")
        points[key] = item
    wanted = [(edge, correction_seed, method) for edge in edges
        for correction_seed in expected["seeds"] for method in expected["methods"]]
    missing = [dict(edge=edge, seed=correction_seed, method=method)
        for edge, correction_seed, method in wanted if (edge, correction_seed, method) not in points]
    ready = not missing and formal_plan_ready
    result = dict(status="complete" if ready else "not_ready", panel=panel,
        configuration=configuration, quality_users=len(users) if users is not None else None,
        expected=dict(edges=edges, **expected), supplied_points=len(points), expected_points=len(wanted),
        missing=missing, out_of_scope=extras,
        observed_seed_status=_seed_status({key[1] for key in points}, expected["seeds"]),
        complete_grid=not missing, formal_plan_prepared=formal_plan_ready, primary_aggregate_computed=ready,
        training_recipe_sha256=hashlib.sha256(json.dumps(common_recipe, sort_keys=True).encode()).hexdigest()
            if common_recipe is not None else None,
        quality_uids_sha256=hashlib.sha256(json.dumps(users).encode()).hexdigest() if users is not None else None)
    result["observed_seed_status"].update(planned_seeds_complete=not missing,
        three_seed_complete=not missing and len(expected["seeds"]) == 3,
        training_seed_robustness_evaluated=ready and len(expected["seeds"]) == 3,
        status="not_ready_incomplete_edge_seed_method_grid" if missing else
            ("complete_three_seed_grid" if len(expected["seeds"]) == 3 else "complete_predeclared_single_seed_grid"))
    if not ready:
        return result, users
    population = np.asarray(sorted(users), dtype=np.int64)
    rng = np.random.default_rng(seed)
    weights = np.empty((replicates+1, len(population)), dtype=np.uint16)
    weights[0] = 1
    for draw in range(1, replicates+1):
        weights[draw] = np.bincount(rng.integers(len(population), size=len(population)), minlength=len(population))
    baseline, measured = defaultdict(list), defaultdict(list)
    edge_records = []
    for edge in edges:
        base = anchors[edge]["reuse"]
        labels = np.asarray([row["label"] for row in base], dtype=np.float64)
        uid_values = np.asarray([row["uid"] for row in base], dtype=np.int64)
        index = np.searchsorted(population, uid_values)
        if np.any(index >= len(population)) or not np.array_equal(population[index], uid_values):
            raise ValueError("E5 score UID outside the fixed panel")
        def auc(rows):
            return weighted_auc(np.asarray([row["score"] for row in rows]), labels, index, weights)
        edge_baselines = {tag: auc(rows) for tag, rows in anchors[edge].items()}
        if any(not np.isfinite(values[0]) for values in edge_baselines.values()):
            raise ValueError("each release point requires both observed feedback labels")
        for tag, values in edge_baselines.items():
            baseline[tag].append(values)
        edge_record = dict(edge=edge, feedback_events=len(base),
            users_with_feedback=len(set(uid_values.tolist())),
            reuse_auc=float(edge_baselines["reuse"][0]), rebuild_auc=float(edge_baselines["rebuild"][0]),
            evaluation_window=points[edge, expected["seeds"][0], expected["methods"][0]]["summary"].get("evaluation_window"),
            methods=[])
        for correction_seed in expected["seeds"]:
            for method in expected["methods"]:
                item = points[edge, correction_seed, method]
                if not _same_events(item["rows"], base):
                    raise ValueError("cross-method paired event/label alignment differs")
                values = auc(item["rows"])
                summary = item["summary"]
                if (abs(values[0]-summary["auc"]) > 1e-12 or
                    abs(edge_baselines["reuse"][0]-summary["reuse_auc"]) > 1e-12 or
                    abs(edge_baselines["rebuild"][0]-summary["rebuild_auc"]) > 1e-12):
                    raise ValueError("E5 summary AUC differs from verified feedback scores")
                measured[correction_seed, method].append(values)
                edge_record["methods"].append(dict(seed=correction_seed, method=method,
                    auc=float(values[0]), source_summary=item["path"],
                    checkpoint_sha256=item["training"]["checkpoint_sha256"],
                    best_step=item["training"]["best_step"], completed_steps=item["training"]["completed_steps"]))
        edge_records.append(edge_record)
    baseline = {tag: np.mean(values, axis=0) for tag, values in baseline.items()}
    measured = {key: np.mean(values, axis=0) for key, values in measured.items()}
    reuse, rebuild = baseline["reuse"], baseline["rebuild"]
    gap = rebuild-reuse
    ratio_valid = np.isfinite(gap[1:]) & (gap[1:] > 0)
    per_seed, all_contrasts = [], defaultdict(list)
    for correction_seed in expected["seeds"]:
        methods = {method: _method_metric(measured[correction_seed, method], reuse, rebuild)
            for method in expected["methods"]}
        increments = {}
        for name, (after, before) in CONTRASTS.items():
            if after not in expected["methods"] or before not in expected["methods"]:
                continue
            delta = measured[correction_seed, after]-measured[correction_seed, before]
            increments[name] = dict(after=after, before=before, **_contrast_metric(delta, gap))
            all_contrasts[name].append(delta)
        per_seed.append(dict(seed=correction_seed, methods=methods, increments=increments))
    across = dict(methods={}, increments={}, seed_status=_seed_status(expected["seeds"], expected["seeds"]))
    for method in expected["methods"]:
        matrix = np.stack([measured[correction_seed, method] for correction_seed in expected["seeds"]])
        metrics = _method_metric(matrix.mean(0), reuse, rebuild)
        point_recovery = 100*(matrix[:, 0]-reuse[0])/gap[0] if gap[0] != 0 else None
        metrics["training_seed_point_range"] = dict(auc=[float(matrix[:, 0].min()), float(matrix[:, 0].max())],
            auc_gain_over_reuse=[float((matrix[:, 0]-reuse[0]).min()), float((matrix[:, 0]-reuse[0]).max())],
            recovery_percent=[float(point_recovery.min()), float(point_recovery.max())] if point_recovery is not None else None)
        across["methods"][method] = metrics
    for name, values in all_contrasts.items():
        matrix = np.stack(values)
        metrics = _contrast_metric(matrix.mean(0), gap)
        point_recovery = 100*matrix[:, 0]/gap[0] if gap[0] != 0 else None
        metrics.update(after=CONTRASTS[name][0], before=CONTRASTS[name][1],
            training_seed_point_range=dict(auc_difference=[float(matrix[:, 0].min()), float(matrix[:, 0].max())],
                recovery_percentage_point_difference=[float(point_recovery.min()), float(point_recovery.max())]
                    if point_recovery is not None else None))
        across["increments"][name] = metrics
    result.update(per_seed=per_seed, across_seed=across, edge_records=edge_records,
        anchors=dict(equal_edge_mean_reuse_auc=float(reuse[0]), equal_edge_mean_rebuild_auc=float(rebuild[0]),
            mean_reference_gap=float(gap[0]), conditional_uid_reference_gap_ci95=_interval(gap[1:]),
            nonpositive_or_invalid_gap_replicates=int((~ratio_valid).sum()),
            positive_reference_gap_point=bool(gap[0] > 0)),
        confidence=dict(replicates=replicates, seed=seed,
            method="paired UID-cluster percentile bootstrap; one draw shared across all five edges, methods and seeds",
            shared_uid_multiplicity_across_edges=True, shared_uid_multiplicity_across_training_seeds=True,
            all_eligible_uids_including_no_feedback=True,
            scope="conditional on the frozen fitted weights; conditional UID intervals do not estimate training-seed variability",
            aggregation="equal-edge mean AUC differences first, then recovery ratio; never mean edge recovery percentages",
            recovery_rule="retain signed point ratios; no bounded recovery interval if point reference gap or any bootstrap reference gap is nonpositive/invalid"))
    return result, users


def aggregate_loaded(loaded, scope, *, replicates=1000, seed=20261031, formal_plan_ready=True):
    """Numerical core; callers supply artifacts already audited by ShardLoader."""
    if replicates < 1:
        raise ValueError("positive paired UID bootstrap replicate count required")
    expected = _scope(scope)
    unknown = [item["path"] for item in loaded if item["summary"]["identity"]["panel"] not in expected]
    if unknown:
        raise ValueError("formal aggregation requires explicit main/confirmation panel labels")
    panels, populations = {}, {}
    for index, panel in enumerate(("main", "confirmation")):
        panels[panel], populations[panel] = _panel(loaded, panel, expected[panel], scope["edges"],
            replicates=replicates, seed=seed+index, formal_plan_ready=formal_plan_ready)
    if all(populations.values()) and set(populations["main"]) & set(populations["confirmation"]):
        raise ValueError("main and independent confirmation UID populations overlap")
    if all(value["configuration"] is not None for value in panels.values()) and panels["main"]["configuration"] != panels["confirmation"]["configuration"]:
        raise ValueError("independent confirmation must evaluate the same predeclared fitting configuration")
    paired = defaultdict(dict)
    for item in loaded:
        s = item["summary"]
        if s["seed"] == 17 and s["method_label"] in expected["confirmation"]["methods"]:
            paired[s["edge"], s["method_label"]][s["identity"]["panel"]] = item["training"]["checkpoint_sha256"]
    if any(len(value) == 2 and value["main"] != value["confirmation"] for value in paired.values()):
        raise ValueError("main and confirmation seed17 must use exactly the same per-edge/method checkpoint SHA")
    return dict(status="complete" if all(value["primary_aggregate_computed"] for value in panels.values()) else "not_ready",
        experiment="E5 five-edge paired quality and direct increments", panels=panels,
        population_policy="main and independent confirmation analyzed separately; no population or seed pooling",
        selection="all supplied expected edges/controls retained irrespective of outcomes; no quality-based configuration selection")


def analyze(summary_paths, *, scope_path=DEFAULT_SCOPE, formal_plan_path=DEFAULT_FORMAL_PLAN,
            replicates=1000, seed=20261031):
    summary_paths = list(summary_paths)
    loader = ShardLoader()
    scope_path = Path(scope_path)
    scope = loader.json(scope_path)
    pending = [str(Path(path).resolve()) for path in summary_paths if not Path(path).exists()]
    loaded = [loader.load(path) for path in summary_paths if Path(path).exists()]
    metadata_cache = {}
    for item in loaded:
        summary = item["summary"]
        checkpoint = summary["identity"]["checkpoint"]
        if checkpoint not in metadata_cache:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            metadata_cache[checkpoint] = {key: payload[key] for key in payload if key != "state_dict"}
            del payload
        item["training"] = _training_record(summary, metadata_cache[checkpoint])
    plan_path = Path(formal_plan_path)
    plan = loader.json(plan_path) if plan_path.exists() else None
    plan_ready = _formal_plan_audit(loaded, plan, loader, scope_path)
    result = aggregate_loaded(loaded, scope, replicates=replicates, seed=seed, formal_plan_ready=plan_ready)
    result["costs"] = collect_costs(loaded,result,family="ladder")
    result.update(pending_summary_paths=pending,
        formal_scope=dict(path=str(scope_path.resolve()), sha256=loader.verify(scope_path)),
        formal_plan=dict(path=str(plan_path.resolve()), sha256=loader.verify(plan_path) if plan is not None else None,
            status=plan.get("status") if plan else "missing", prepared_and_bound=plan_ready),
        verification=dict(artifact_sha256=loader.hashes, analysis_sources={**quality.sources(), **cost_export_sources(), **{str(path.relative_to(ROOT)): sha256(path)
            for path in (Path(__file__), ROOT/"experiments/ch3_motivation/code/analyze_probe_quality.py",
                ROOT/"src/hstu_kvcache/evaluation/statistics.py")}}))
    return result


def run(args):
    result = analyze(args.summary, scope_path=args.scope, formal_plan_path=args.formal_plan,
        replicates=args.replicates, seed=args.bootstrap_seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_cost_csvs(result["costs"],args.output)
    write_json(args.output/"summary.json", result)
    rows = []
    for panel, report in result["panels"].items():
        if not report["primary_aggregate_computed"]:
            continue
        for name, value in report["across_seed"]["increments"].items():
            interval = value["conditional_uid_auc_difference_ci95"] or [None, None]
            rows.append(dict(panel=panel, contrast=name, after=value["after"], before=value["before"],
                training_seeds=",".join(map(str, report["expected"]["seeds"])),
                mean_auc_difference=value["mean_auc_difference"], conditional_uid_ci95_low=interval[0],
                conditional_uid_ci95_high=interval[1],
                recovery_percentage_point_difference=value["recovery_percentage_point_difference"],
                three_seed_complete=report["across_seed"]["seed_status"]["three_seed_complete"]))
    if rows:
        with (args.output/"paired_increments.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(status=result["status"], output=str(args.output/"summary.json"),
        panels={name: value["status"] for name, value in result["panels"].items()})), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", nargs="*", type=Path, default=[])
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--formal-plan", type=Path, default=DEFAULT_FORMAL_PLAN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261031)
    run(parser.parse_args())
