#!/usr/bin/env python3
"""The fixed E5 f calibration-population curve on the main2000 panel.

A complete5-edge x3-seed x3-size grid and the prepared curve plan are required.
All points share UID bootstrap draws; gradient/teacher/input/RMS users vary
naturally with N. Candidate/time-law source pools are bound by the formal plan.
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
from experiments.ch3_motivation.code.analyze_probe_quality_multiedge import (
    DEFAULT_SCOPE, DEFAULT_FORMAL_PLAN, _training_record, _check_planned_job, _recipe, _method_metric,
    _contrast_metric, _seed_status,
)
from experiments.ch3_motivation.code.common import write_json
from experiments.ch3_motivation.code.probe_cost_export import (
    collect_costs, write_cost_csvs, sources as cost_export_sources,
)
from experiments.ch3_motivation.code.sampling_identity import (
    SAMPLING_DRIVERS, validate_sampling_capture, validate_sampling_identity,
)
from hstu_kvcache.evaluation.statistics import weighted_auc
from hstu_kvcache.common.artifacts import sha256

EDGES, SEEDS, COUNTS = (1, 2, 3, 4, 5), (17, 23, 29), (64, 128, 256)
CONTRASTS = {"f128_minus_f64": (128, 64), "f256_minus_f128": (256, 128), "f256_minus_f64": (256, 64)}


def curve_training_record(summary, payload):
    """Reuse actual typed checkpoint validation, then require the curve's full cap."""
    training = _training_record(summary, payload)
    metadata, identity = payload["metadata"], training["identity"]
    n, cap = int(summary["fit_users"]), int(identity["steps"])
    scales = metadata["fit_pool_loss_scales"]
    if (summary["method_label"] != "f" or n not in COUNTS or identity.get("stages") != "f" or
            identity.get("controls") is not False or identity.get("patience") != 0 or cap < 256):
        raise ValueError("curve requires separate f-only checkpoints, patience0 and a common cap at least256")
    if (training["completed_steps"] != cap or not 0 <= training["best_step"] <= cap or
            scales.get("users") != n or scales.get("validation_uses_fitting_scales") is not True or
            metadata.get("validation_users") != 16 or metadata.get("teacher_preparation_users") != n+16):
        raise ValueError("curve training completion, own-N fitting RMS or teacher/validation coverage differs")
    if any(identity.get(key, n) != n for key in ("gradient_fitting_users", "teacher_fitting_users")):
        raise ValueError("curve gradient/teacher membership count differs from N")
    if identity.get("fitting_driver") in SAMPLING_DRIVERS:
        validate_sampling_capture(identity, metadata)
        if any(identity.get(key) != n for key in ("fit_users", "gradient_fitting_users", "teacher_fitting_users")):
            raise ValueError("sampling-v1 curve must explicitly record its actual N teacher/gradient capture")
    training.update(normalization_users=n, validation_users=16, teacher_preparation_users=n+16,
        gradient_users_seen=min(n, training["completed_steps"]),
        best_checkpoint_gradient_users_seen=min(n, training["best_step"]),
        normalization_estimate=dict(read_scales=scales["read_scales"], score_scale=scales["score_scale"]))
    return training


def _curve_recipe(training, *, across_edges):
    # Only N-dependent membership is removed. Numerical recipes, candidate
    # bindings and temporal identities still pass the shared strict checker.
    adjusted = dict(training, identity={key: value for key, value in training["identity"].items()
        if key not in ("fit_uids", "fit_users", "gradient_fitting_users", "teacher_fitting_users")})
    if across_edges and training.get("formal_curve_calibration_bound"):
        adjusted["identity"] = {key:value for key,value in adjusted["identity"].items()
            if key not in ("gap_manifest", "gap_manifest_sha256", "sampling_manifest",
                "sampling_manifest_sha256", "source_cutoff")}
    return _recipe(adjusted, across_edges=across_edges)


def _scope(scope):
    curve, main = scope["e5"]["fitting_population_curve"], scope["e5"]["main"]
    if (scope["edges"] != list(EDGES) or curve["users"] != list(COUNTS) or
            curve["seeds"] != list(SEEDS) or curve["method"] != "f" or main["users"] != 2000):
        raise ValueError("population curve differs from frozen5-edge/3-seed/N64,128,256/main2000 scope")


def _sampling_binding(identity, job, binding, population, loader, policy, cache, *, config=None):
    """Use the actual new fixed-source sampler, including its cutover view."""
    if (policy.get("time_law_source_users") != 256 or
            policy.get("candidate_catalog_source_users") != 256):
        return False
    if not all(key in binding for key in ("sampling_manifest", "sampling_manifest_sha256")):
        return False
    path = Path(binding["sampling_manifest"])
    if not path.exists():
        return False
    digest = loader.verify(path, binding["sampling_manifest_sha256"])
    if binding["sampling_manifest_sha256"] != digest:
        raise ValueError("sampling curve requires an explicit nonnull plan manifest SHA256")
    driver = identity.get("fitting_driver")
    driver_sha = identity.get("fitting_driver_sha256")
    if (driver not in SAMPLING_DRIVERS or job.get("fitting_driver") != driver or
            not isinstance(driver_sha, str) or len(driver_sha) != 64 or
            any(char not in "0123456789abcdef" for char in driver_sha) or
            job.get("fitting_driver_sha256") != driver_sha or
            identity.get("numerical_sources", {}).get(f"experiments/ch3_motivation/code/{driver}") != driver_sha):
        raise ValueError("sampling curve fitting driver/SHA must be explicitly bound by job and checkpoint")
    for metadata in (identity, job):
        if (metadata.get("sampling_manifest_sha256") != digest or
                metadata.get("gap_manifest_sha256") != digest or
                Path(metadata.get("sampling_manifest", "")).resolve() != path.resolve() or
                Path(metadata.get("gap_manifest", "")).resolve() != path.resolve()):
            raise ValueError("sampling curve job/checkpoint/manifest binding differs")
    from experiments.ch3_motivation.code.calibration_sampling import load_sampling_manifest
    key = (str(path.resolve()), digest, job["edge"], job["fit_users"], job["recent_fraction"], job["query_delta_rule"])
    if key not in cache:
        _, provenance = load_sampling_manifest(path,binding["input_root"],job["edge"],
            job["recent_fraction"],job["query_delta_rule"],fit_users=job["fit_users"])
        cache[key] = provenance
    provenance = cache[key]
    validate_sampling_identity(identity, sampling_cache=cache, config=config)
    if driver == "run_probe_calibration_init.py" and any(identity.get(name) != job.get(name)
            for name in ("h_initialization", "h_initialization_spec", "initialization_helper_sha256")):
        raise ValueError("sampling curve initialization identity differs from its prepared job")
    if (provenance["manifest_sha"] != digest or
            provenance["gap_rule"] != job["query_delta_rule"] or
            provenance["feedback_fraction"] != job["recent_fraction"] or
            provenance["candidate_count"] != identity["queries"] or
            provenance["candidate_seed"] != identity["candidate_seed"] or
            provenance["fit_uids"] != population["fit"][:job["fit_users"]] or
            provenance["validation_uids"] != population["validation"] or
            provenance["catalog_source_users"] != policy["candidate_catalog_source_users"] or
            provenance["time_law_source_users"] != policy["time_law_source_users"] or
            identity.get("candidate_catalog_source_users") != provenance["catalog_source_users"] or
            identity.get("time_law_source_users") != provenance["time_law_source_users"] or
            identity.get("query_delta_layout") != "[B,16]" or
            identity.get("candidate_source") != "feedback" or
            any(identity.get(name) != job["fit_users"] for name in
                ("fit_users", "gradient_fitting_users", "teacher_fitting_users")) or
            any(identity.get(name) != provenance[name] for name in ("time_rule", "source_cutoff", "candidate_order"))):
        raise ValueError("sampling curve source roles/time/candidate provenance differs")
    return True


def _temporal_binding(identity, job, binding, population, loader, policy, sampling_cache=None, *, config=None):
    rule = job.get("query_delta_rule", "cutover")
    if identity.get("query_delta_rule", "cutover") != rule:
        raise ValueError("curve calibration query delta rule differs from the prepared job")
    driver = identity.get("fitting_driver", "run_probe_variants.py")
    driver_sha = identity.get("fitting_driver_sha256", identity.get("numerical_sources", {}).get(
        f"experiments/ch3_motivation/code/{driver}"))
    if (driver != job.get("fitting_driver", driver) or
            driver_sha != job.get("fitting_driver_sha256", driver_sha)):
        raise ValueError("curve fitting driver identity differs from the prepared job")
    if identity.get("fitting_driver") in SAMPLING_DRIVERS:
        return _sampling_binding(identity,job,binding,population,loader,policy,
            {} if sampling_cache is None else sampling_cache,config=config)
    if any(key in identity for key in ("sampling_manifest", "sampling_manifest_sha256")):
        raise ValueError("sampling curve has no recognized explicit fitting driver")
    if rule == "cutover":
        if identity.get("gap_manifest_sha256") is not None or job.get("gap_manifest_sha256") is not None:
            raise ValueError("release-gap curve unexpectedly carries a temporal gap manifest")
        return True
    if rule not in ("gap_mix50", "gap_empirical100"):
        raise ValueError("unknown planned curve temporal rule")
    law_users = policy.get("time_law_source_users")
    if law_users not in (64, 256):
        return False
    # The original fit64 manifest cannot be relabelled as a fit256 target view.
    # No temporal aggregate is ready until the dedicated curve artifact exists.
    if not all(key in binding for key in ("gap_manifest", "gap_manifest_sha256")):
        return False
    path = Path(binding["gap_manifest"])
    if not path.exists():
        return False
    loader.verify(path, binding["gap_manifest_sha256"])
    if (identity.get("gap_manifest_sha256") != binding["gap_manifest_sha256"] or
            job.get("gap_manifest_sha256") != binding["gap_manifest_sha256"] or
            Path(identity.get("gap_manifest", "")).resolve() != path.resolve() or
            Path(job.get("gap_manifest", "")).resolve() != path.resolve()):
        raise ValueError("curve temporal identity/job/manifest bindings differ")
    manifest = loader.json(path)
    needed = ("time_law_source_fit_uids", "target_fit_uids", "candidate_catalog_source_users",
              "parent_fit64_gap_manifest", "parent_fit64_gap_manifest_sha256", "validation_uids")
    if not all(key in manifest for key in needed):
        return False
    if (manifest["time_law_source_fit_uids"] != population["fit"][:law_users] or
            manifest["target_fit_uids"] != population["fit"][:256] or
            manifest["candidate_catalog_source_users"] != 256 or
            manifest["validation_uids"] != population["validation"]):
        raise ValueError("temporal curve declared-law/catalog/target/validation roles differ")
    # This is the retained law64 branch. The new fixed-law256 sampling view
    # will be bound through its actual manifest schema when prepared; it must
    # never pass by relabelling this old parent64 artifact.
    if law_users != 64:
        return False
    parent_path = Path(manifest["parent_fit64_gap_manifest"])
    loader.verify(parent_path, manifest["parent_fit64_gap_manifest_sha256"])
    parent = loader.json(parent_path)
    if parent["fit_uids"] != population["fit"][:64] or parent["validation_uids"] != population["validation"]:
        raise ValueError("curve time law does not come from the original frozen fit64 manifest")
    if (manifest.get("edge") != job["edge"] or parent["edge"] != job["edge"] or
            manifest.get("source_cutoff") != parent["source_cutoff"] or
            manifest.get("time_rule") != parent["time_rule"] or
            manifest.get("candidate_count") != identity["queries"] or
            manifest.get("candidate_seed") != identity["candidate_seed"] or
            rule not in manifest.get("rules", ())):
        raise ValueError("curve temporal rule/query/cutoff differs from its fixed parent law")
    input_binding = manifest.get("input_binding", {})
    if not input_binding:
        return False
    if (input_binding.get("population_sha256") != binding["population_sha256"] or
            input_binding.get("sha256") != binding["input_binding_sha256"] or
            input_binding.get("candidate_sha256") != binding["candidate_file_sha256"]):
        raise ValueError("curve temporal target input/candidate binding differs")
    if any(identity.get(key) != manifest.get(key) for key in ("time_rule", "source_cutoff", "candidate_order")):
        raise ValueError("curve temporal time/candidate-order provenance differs")
    return True


def audit_curve_plan(loaded, plan, loader, scope_path):
    wanted = {(edge, seed, n) for edge in EDGES for seed in SEEDS for n in COUNTS}
    result = dict(ready=False, expected_jobs=45, supplied_jobs=0, missing_jobs=[], temporal_bindings_ready=True)
    if plan is None or plan.get("status") != "prepared_for_root_review":
        result.update(status=plan.get("status") if plan else "missing", missing_jobs=[list(k) for k in sorted(wanted)])
        return result
    loader.verify(plan["source_scope"], sha256(scope_path))
    if plan["source_scope_sha256"] != sha256(scope_path):
        raise ValueError("prepared curve plan scope binding changed")
    policy = plan.get("population_curve_policy", {})
    result["calibration_source_policy"] = policy
    if policy.get("candidate_catalog_source_users") != 256:
        raise ValueError("formal curve policy must declare its fixed256-user candidate catalog")
    jobs = {}
    for job in plan["curve_jobs"]:
        key = int(job["edge"]), int(job["seed"]), int(job["fit_users"])
        if key not in wanted or key in jobs:
            raise ValueError("duplicate or unexpected formal curve job")
        if (job["stages"] != "f" or job["controls"] is not False or
                job["patience"] != 0 or job["steps"] < 256):
            raise ValueError("planned curve jobs require independent f-only full-cap fitting")
        jobs[key] = job
    if len({job["steps"] for job in jobs.values()}) > 1:
        raise ValueError("curve jobs do not share the same optimization cap")
    bindings, populations, main_populations = {}, {}, {}
    for edge in EDGES:
        binding = plan.get("curve_input_bindings", {}).get(str(edge))
        if binding is None:
            continue
        root = Path(binding["input_root"])
        loader.verify(root/"population.json", binding["population_sha256"])
        loader.verify(root/f"edge_{edge}/inputs.json", binding["input_binding_sha256"])
        loader.verify(binding["candidate_file"], binding["candidate_file_sha256"])
        population, candidates = loader.json(root/"population.json"), loader.json(binding["candidate_file"])
        if (binding.get("candidate_catalog_source_users") != 256 or
                candidates.get("edge") != edge or candidates.get("population_sha256") != binding["population_sha256"] or
                candidates.get("fitting_pool_uids") != population["fit"][:256] or
                candidates.get("validation_uids") != population["validation"] or
                len(population["validation"]) != 16):
            raise ValueError("curve requires one fixed fit256 candidate catalog and same16 validation users")
        main_binding = plan.get("main_input_bindings", {}).get(str(edge))
        if main_binding is not None:
            main_population = Path(main_binding["input_root"])/"population.json"
            loader.verify(main_population, main_binding["population_sha256"])
            main_populations[edge] = loader.json(main_population)
            if population["main"] != main_populations[edge]["main"]:
                raise ValueError("curve population differs from the formal main UID population")
        bindings[edge], populations[edge] = binding, population
    sampling_cache = {}
    for item in loaded:
        s = item["summary"]
        key = s["edge"], s["seed"], s["fit_users"]
        if key not in jobs or s["edge"] not in bindings:
            continue
        job, binding, population = jobs[key], bindings[s["edge"]], populations[s["edge"]]
        _check_planned_job(item, job, binding, population, plan.get("numerical_constants", {}))
        if s["identity"]["users"] != population["main"]:
            raise ValueError("curve quality must use the same fixed main2000 UID order")
        if not _temporal_binding(item["training"]["identity"], job, binding, population, loader, policy, sampling_cache,
                config=item["training"]["config"]):
            result["temporal_bindings_ready"] = False
        else:
            item["training"]["formal_curve_calibration_bound"] = True
    missing = sorted(wanted-set(jobs))
    result.update(status="complete" if not missing and len(bindings) == 5 and len(main_populations) == 5 and result["temporal_bindings_ready"] else "not_ready",
        supplied_jobs=len(jobs), missing_jobs=[list(k) for k in missing],
        missing_input_edges=sorted(set(EDGES)-set(bindings)),
        missing_main_population_edges=sorted(set(EDGES)-set(main_populations)),
        ready=not missing and len(bindings) == 5 and len(main_populations) == 5 and result["temporal_bindings_ready"])
    return result


def aggregate_loaded(loaded, scope, *, replicates=1000, seed=20261101, formal_plan_ready=True,
                     calibration_source_policy=None):
    """Numerical core for already audited score rows and checkpoint metadata."""
    _scope(scope)
    if replicates < 1:
        raise ValueError("positive UID bootstrap replicate count required")
    users, config, common_recipe = None, None, None
    points, contexts, anchors, edge_recipes = {}, {}, {}, {}
    for item in loaded:
        s, identity, training = item["summary"], item["summary"]["identity"], item["training"]
        edge, correction_seed, n = s["edge"], s["seed"], s["fit_users"]
        if identity["panel"] != "main" or s["method_label"] != "f" or edge not in EDGES or correction_seed not in SEEDS or n not in COUNTS:
            raise ValueError("provide only the predeclared main-panel f population curve")
        if len(identity["users"]) != 2000 or (users is not None and identity["users"] != users):
            raise ValueError("curve requires the same fixed2000 UID membership/order across edges/N/seeds")
        users = identity["users"]
        architecture = {key: s[key] for key in ("variant", "width", "depth")}
        if config is not None and architecture != config:
            raise ValueError("curve architecture differs across N/edges/seeds")
        config = architecture
        edge_recipe, recipe = _curve_recipe(training, across_edges=False), _curve_recipe(training, across_edges=True)
        if edge in edge_recipes and edge_recipe != edge_recipes[edge]:
            raise ValueError("same-edge curve recipes differ beyond gradient fitting membership")
        if formal_plan_ready and common_recipe is not None and recipe != common_recipe:
            raise ValueError("cross-edge curve recipes differ beyond bound natural edge variations")
        edge_recipes[edge], common_recipe = edge_recipe, recipe
        context = _context(identity)
        if edge in contexts and context != contexts[edge]:
            raise ValueError("curve quality context differs across N/seeds within an edge")
        if contexts and any(context[key] != next(iter(contexts.values()))[key] for key in ("sources", "backend")):
            raise ValueError("curve quality numerical sources/backend differ across edges")
        contexts[edge] = context
        if edge in anchors:
            for tag in ("reuse", "rebuild"):
                before, after = anchors[edge][tag], item["anchors"][tag]
                if not _same_events(before, after) or [row["score"] for row in before] != [row["score"] for row in after]:
                    raise ValueError("curve paired anchor events/scores differ")
        anchors[edge] = item["anchors"]
        key = edge, correction_seed, n
        if key in points:
            raise ValueError("duplicate curve edge/seed/N cell")
        points[key] = item
    wanted = {(edge, correction_seed, n) for edge in EDGES for correction_seed in SEEDS for n in COUNTS}
    missing = [dict(edge=edge, seed=correction_seed, fit_users=n) for edge, correction_seed, n in sorted(wanted-set(points))]
    policy = calibration_source_policy or {}
    temporal = any(item["training"]["identity"].get("query_delta_rule", "cutover") != "cutover" for item in loaded)
    sampling = any(item["training"]["identity"].get("fitting_driver") in SAMPLING_DRIVERS for item in loaded)
    source_policy_ready = (policy.get("candidate_catalog_source_users") == 256 and
        policy.get("time_law_source_users") == 256) if sampling else (
        not temporal or policy.get("time_law_source_users") in (64, 256))
    ready = not missing and formal_plan_ready and source_policy_ready
    result = dict(status="complete" if ready else "not_ready", experiment="E5 f fitting-population curve",
        panel="main", quality_users=len(users) if users is not None else None, configuration=config,
        expected=dict(edges=list(EDGES), seeds=list(SEEDS), fitting_users=list(COUNTS), points=45),
        supplied_points=len(points), missing=missing, complete_grid=not missing,
        formal_curve_plan_bound=formal_plan_ready, primary_aggregate_computed=ready,
        observed_seed_status=_seed_status({key[1] for key in points}, SEEDS),
        quality_uids_sha256=hashlib.sha256(json.dumps(users).encode()).hexdigest() if users is not None else None,
        calibration_source_policy=policy, temporal_source_policy_ready=source_policy_ready,
        policy="same main2000 UID draws across all edges/N/seeds; own-N input statistics/RMS; candidate/time-law sources fixed by the explicit formal curve policy; no AUC selection",
        curve_weight_policy="separate stages=f/controls=False full-cap checkpoints; main ladder weights cannot substitute")
    result["observed_seed_status"].update(three_seed_complete=not missing,
        planned_seeds_complete=not missing, training_seed_robustness_evaluated=ready,
        status="not_ready_incomplete_edge_seed_size_grid" if missing else "complete_three_seed_grid")
    if not ready:
        return result
    population = np.asarray(sorted(users), dtype=np.int64)
    rng = np.random.default_rng(seed)
    weights = np.ones((replicates+1, len(population)), dtype=np.uint16)
    for draw in range(1, replicates+1):
        weights[draw] = np.bincount(rng.integers(len(population), size=len(population)), minlength=len(population))
    measured, baseline, edge_records = defaultdict(list), defaultdict(list), []
    for edge in EDGES:
        base = anchors[edge]["reuse"]
        labels = np.asarray([row["label"] for row in base], dtype=float)
        uids = np.asarray([row["uid"] for row in base], dtype=np.int64)
        indices = np.searchsorted(population, uids)
        if np.any(indices >= len(population)) or not np.array_equal(population[indices], uids):
            raise ValueError("curve feedback UID outside the fixed panel")
        def auc(rows):
            if not _same_events(rows, base):
                raise ValueError("curve feedback keys/items/labels differ")
            return weighted_auc(np.asarray([row["score"] for row in rows]), labels, indices, weights)
        edge_baseline = {tag: auc(rows) for tag, rows in anchors[edge].items()}
        if any(not np.isfinite(values[0]) for values in edge_baseline.values()):
            raise ValueError("each edge requires both feedback labels")
        for tag, values in edge_baseline.items():
            baseline[tag].append(values)
        record = dict(edge=edge, feedback_events=len(base), users_with_feedback=len(set(uids.tolist())),
            reuse_auc=float(edge_baseline["reuse"][0]), rebuild_auc=float(edge_baseline["rebuild"][0]),
            evaluation_window=points[edge, SEEDS[0], COUNTS[0]]["summary"].get("evaluation_window"), points=[])
        for correction_seed in SEEDS:
            for n in COUNTS:
                item = points[edge, correction_seed, n]
                values, s, t = auc(item["rows"]), item["summary"], item["training"]
                if any(abs(a-b) > 1e-12 for a,b in [(values[0],s["auc"]),
                    (edge_baseline["reuse"][0],s["reuse_auc"]),(edge_baseline["rebuild"][0],s["rebuild_auc"])]):
                    raise ValueError("curve summary AUC differs from verified feedback scores")
                measured[correction_seed,n].append(values)
                record["points"].append(dict(seed=correction_seed,fit_users=n,auc=float(values[0]),
                    source_summary=item["path"],checkpoint_sha256=t["checkpoint_sha256"],
                    best_step=t["best_step"],completed_steps=t["completed_steps"],
                    gradient_users_seen=t["gradient_users_seen"],best_checkpoint_gradient_users_seen=t["best_checkpoint_gradient_users_seen"],
                    normalization_users=t["normalization_users"],teacher_preparation_users=t["teacher_preparation_users"],
                    validation_users=t["validation_users"],candidate_catalog_source_users=256,
                    time_law_source_users=policy.get("time_law_source_users"),
                    query_delta_rule=t["identity"].get("query_delta_rule", "cutover"),
                    time_law_applied=t["identity"].get("query_delta_rule", "cutover") != "cutover",
                    normalization_estimate=t["normalization_estimate"]))
        edge_records.append(record)
    baseline = {key:np.mean(value,axis=0) for key,value in baseline.items()}
    measured = {key:np.mean(value,axis=0) for key,value in measured.items()}
    reuse,rebuild = baseline["reuse"],baseline["rebuild"]
    gap = rebuild-reuse
    per_seed = []
    for correction_seed in SEEDS:
        per_seed.append(dict(seed=correction_seed,methods={str(n):_method_metric(measured[correction_seed,n],reuse,rebuild) for n in COUNTS},
            increments={name:dict(after=after,before=before,**_contrast_metric(measured[correction_seed,after]-measured[correction_seed,before],gap)) for name,(after,before) in CONTRASTS.items()}))
    across = dict(methods={},increments={},seed_status=_seed_status(SEEDS,SEEDS))
    for n in COUNTS:
        matrix = np.stack([measured[correction_seed,n] for correction_seed in SEEDS])
        metric = _method_metric(matrix.mean(0),reuse,rebuild)
        recovery = 100*(matrix[:,0]-reuse[0])/gap[0] if gap[0] != 0 else None
        metric["training_seed_point_range"] = dict(auc=[float(matrix[:,0].min()),float(matrix[:,0].max())],
            auc_gain_over_reuse=[float((matrix[:,0]-reuse[0]).min()),float((matrix[:,0]-reuse[0]).max())],
            recovery_percent=[float(recovery.min()),float(recovery.max())] if recovery is not None else None)
        across["methods"][str(n)] = metric
    for name,(after,before) in CONTRASTS.items():
        matrix = np.stack([measured[correction_seed,after]-measured[correction_seed,before] for correction_seed in SEEDS])
        recovery = 100*matrix[:,0]/gap[0] if gap[0] != 0 else None
        across["increments"][name] = dict(after=after,before=before,**_contrast_metric(matrix.mean(0),gap),
            training_seed_point_range=dict(auc_difference=[float(matrix[:,0].min()),float(matrix[:,0].max())],
                recovery_percentage_point_difference=[float(recovery.min()),float(recovery.max())] if recovery is not None else None))
    result.update(per_seed=per_seed,across_seed=across,edge_records=edge_records,
        anchors=dict(equal_edge_mean_reuse_auc=float(reuse[0]),equal_edge_mean_rebuild_auc=float(rebuild[0]),
            mean_reference_gap=float(gap[0]),conditional_uid_reference_gap_ci95=_interval(gap[1:]),
            nonpositive_or_invalid_gap_replicates=int((~(np.isfinite(gap[1:])&(gap[1:]>0))).sum())),
        confidence=dict(replicates=replicates,seed=seed,shared_uid_draw_across_edges_sizes_seeds=True,
            all_eligible_uids_including_no_feedback=True,
            aggregation="equal-edge mean AUC differences first, then recovery ratio; seed points averaged before conditional-UID CI",
            scope="conditional on frozen weights; training-seed point ranges are distinct from UID bootstrap intervals"))
    return result


def analyze(summary_paths, *, scope_path=DEFAULT_SCOPE, formal_plan_path=DEFAULT_FORMAL_PLAN,replicates=1000,seed=20261101):
    loader = ShardLoader()
    scope_path = Path(scope_path)
    scope = loader.json(scope_path)
    loaded,pending,cache = [],[],{}
    for path in map(Path,summary_paths):
        if not path.exists() or loader.json(path).get("status") != "complete":
            pending.append(str(path.resolve()))
            continue
        item = loader.load(path)
        checkpoint = item["summary"]["identity"]["checkpoint"]
        if checkpoint not in cache:
            payload = torch.load(checkpoint,map_location="cpu",weights_only=False)
            cache[checkpoint] = {key:value for key,value in payload.items() if key != "state_dict"}
            del payload
        item["training"] = curve_training_record(item["summary"],cache[checkpoint])
        loaded.append(item)
    plan_path = Path(formal_plan_path)
    plan = loader.json(plan_path) if plan_path.exists() else None
    audit = audit_curve_plan(loaded,plan,loader,scope_path)
    result = aggregate_loaded(loaded,scope,replicates=replicates,seed=seed,formal_plan_ready=audit["ready"],
        calibration_source_policy=audit.get("calibration_source_policy"))
    result["costs"] = collect_costs(loaded,result,family="f_population_curve")
    result.update(pending_summary_paths=pending,formal_plan_audit=audit,
        formal_scope=dict(path=str(scope_path.resolve()),sha256=loader.verify(scope_path)),
        formal_plan=dict(path=str(plan_path.resolve()),sha256=loader.verify(plan_path) if plan is not None else None),
        verification=dict(artifact_sha256=loader.hashes,analysis_sources={**cost_export_sources(), **{str(p.relative_to(ROOT)):sha256(p) for p in
            [Path(__file__),ROOT/"experiments/ch3_motivation/code/sampling_identity.py",ROOT/"experiments/ch3_motivation/code/analyze_probe_quality.py",ROOT/"experiments/ch3_motivation/code/analyze_probe_quality_multiedge.py",ROOT/"src/hstu_kvcache/evaluation/statistics.py"]}}))
    return result


def main(args):
    result = analyze(args.summary,scope_path=args.scope,formal_plan_path=args.formal_plan,replicates=args.replicates,seed=args.bootstrap_seed)
    args.output.mkdir(parents=True,exist_ok=True)
    write_cost_csvs(result["costs"],args.output)
    write_json(args.output/"summary.json",result)
    if result["primary_aggregate_computed"]:
        records = [dict(contrast=name,**value) for name,value in result["across_seed"]["increments"].items()]
        with (args.output/"paired_increments.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(dict(status=result["status"],supplied_points=result["supplied_points"],expected_points=45,output=str(args.output/"summary.json"))))
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",nargs="*",type=Path,default=[])
    parser.add_argument("--scope",type=Path,default=DEFAULT_SCOPE)
    parser.add_argument("--formal-plan",type=Path,default=DEFAULT_FORMAL_PLAN)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--replicates",type=int,default=1000)
    parser.add_argument("--bootstrap-seed",type=int,default=20261101)
    main(parser.parse_args())
