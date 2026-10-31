#!/usr/bin/env python3
"""Bind the actual E5 FG/H weights to the fixed E6 experiment; never launch it.

Fifteen release jobs include the seed17 K/V sensitivities. Four separate
512-user jobs compare the release and timestamp protocols on edges2/5.
The full-cost CPU sidecar uses these same weight/population bindings.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow.parquet as pq

from experiments.ch3_motivation.code import plan_probe_quality as quality
from experiments.ch3_motivation.code.common import write_json
from experiments.ch3_motivation.code.fg_trajectory_control import load_fg_control
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.run_lowrank import (
    source_paths, validate_fit_roles, scenario_qualification_record,
)
from experiments.ch3_motivation.code.run_lowrank_release import sources as release_sources
from experiments.ch3_motivation.code.scenario_inputs import load_scenario_population
from hstu_kvcache.evaluation.history import restore_pack
from hstu_kvcache.common.artifacts import sha256

EDGES = (1, 2, 3, 4, 5)
SEEDS = (17, 23, 29)
RANKS = (0, 1, 2, 4, 6, 8, 16)
ALL_VARIANTS = ("both_norm", "both_raw", "key_norm", "value_norm")
TIMESTAMP_DRIVER = "run_lowrank_timestamp_guarded.py"
SCOPE = ROOT/"experiments/ch3_motivation/configs/formal_quality_scope.json"
SCENARIOS = ROOT/"experiments/ch3_motivation/runs/scenario_qualification/attempt_1"


def uid_hash(users):
    return hashlib.sha256(json.dumps(users).encode()).hexdigest()


def merge_frozen(*snapshots):
    """Resolve aliases and reject conflicting snapshots before any merge."""
    result = {}
    for snapshot in snapshots:
        for name, digest in snapshot.items():
            path = Path(name)
            path = str((path if path.is_absolute() else ROOT/path).resolve())
            if path in result and result[path] != digest:
                raise ValueError(f"conflicting E6 frozen snapshots: {path}")
            result[path] = digest
    return result


def _check_files(bindings):
    for name, digest in bindings.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT/path
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"E6 frozen dependency changed: {path}")


def sources():
    paths = {Path(__file__).resolve(), *(p.resolve() for p in source_paths())}
    paths.update((ROOT/name).resolve() for name in release_sources())
    paths.update((ROOT/name).resolve() for name in quality.sources())
    timestamp = Path(__file__).with_name(TIMESTAMP_DRIVER)
    if timestamp.exists():
        paths.add(timestamp.resolve())
    return {str(path): sha256(path) for path in sorted(paths)}


def _scope(path):
    value = json.loads(Path(path).read_text())
    e6 = value["e6"]
    if (value["edges"] != list(EDGES) or value["training_seeds"] != list(SEEDS) or
            e6["main"]["users"] != 2000 or e6["main"]["seeds"] != list(SEEDS) or
            not e6["main"]["feedback_users_required"] or e6["main"]["variants"] != ["both_norm"] or
            e6["main"]["ranks"] != list(RANKS) or e6["main"]["candidate_instances"] != 230 or
            e6["kv_sensitivity"]["users"] != 2000 or e6["kv_sensitivity"]["seeds"] != [17] or
            e6["kv_sensitivity"]["variants"] != list(ALL_VARIANTS[1:]) or
            e6["protocol_sensitivity"]["representative_edges"] != [2, 5] or
            e6["protocol_sensitivity"]["seeds"] != [17] or
            e6["protocol_sensitivity"]["users_at_least"] != 512):
        raise ValueError("E6 scope differs from the fixed five-edge/three-seed experiment")
    return value


def _grid(formal, main_root):
    jobs = formal["main_jobs"]
    if (len(jobs) != 15 or {(j["edge"], j["seed"]) for j in jobs} !=
            {(e, s) for e in EDGES for s in SEEDS}):
        raise ValueError("E6 needs all15 actual main-fitting cells")
    common = None
    for job in jobs:
        if job["stages"] != "f,fg,fgh" or job["controls"] is not True or job["variant"] != "full_item":
            raise ValueError("E6 must use the actual formal primary FG/H ladder with controls")
        recipe = {key: job[key] for key in (*quality.FIELDS, "width", "depth", "fit_users", "variant")}
        recipe.update({key: job.get(key) for key in ("fitting_driver", "fitting_driver_sha256",
            "query_delta_rule", "recent_fraction", "h_initialization")})
        if common is None:
            common = recipe
        if recipe != common:
            raise ValueError("E6 primary weights must retain one frozen E5 recipe across all15cells")
        if "output" in job:
            expected = Path(main_root)/job["name"]/f"edge_{job['edge']}"/f"seed_{job['seed']}"
            if Path(job["output"]).resolve() != expected.resolve():
                raise ValueError("formal fitting output escapes the supplied main-fitting root")
    return sorted(jobs, key=lambda j: (j["edge"], j["seed"]))


def workload_counts(histories, rows, users, settings):
    """Input-only causal workload proxy, not FLOPs or a runtime measurement."""
    per_user = defaultdict(lambda: defaultdict(list))
    selected = set(users)
    at = int(settings["release_day"])*86400
    end = int(settings["end_day"])*86400
    window = int(settings.get("history_length", 1024))
    for row in rows:
        uid, timestamp = int(row["uid"]), int(row["query_timestamp"])
        if uid in selected:
            if not at <= timestamp < end:
                raise ValueError("E6 workload request lies outside the frozen window")
            per_user[uid][timestamp].append(int(row["item_idx"]))
    total = Counter(users=len(users))
    for uid in users:
        times = histories.rows[uid][0]
        queries = per_user[uid]
        if not queries:
            raise ValueError("fixed E6 population includes a user without feedback")
        initial = min(int(np.searchsorted(times, at, side="left")), window)
        before = int(np.searchsorted(times, at, side="left"))
        after = int(np.searchsorted(times, max(queries), side="left"))
        count = sum(len(items) for items in queries.values())
        panels = (count+229)//230
        # The alternate protocol assigns each distinct E14 target to one
        # fixed release panel, then decomposes the panels touched at each t.
        targets = dict.fromkeys(item for timestamp in sorted(queries) for item in sorted(queries[timestamp]))
        panel_of = {item: index//230 for index, item in enumerate(targets)}
        timestamp_panels = {timestamp: len({panel_of[item] for item in items}) for timestamp, items in queries.items()}
        total.update(initial_rows=initial, behavior_appends_before_last_read=after-before,
            feedback_instances=count, timestamps=len(queries), release_panels=panels,
            timestamp_panels=sum(timestamp_panels.values()),
            release_matrix_row_candidate_work=initial*230*panels,
            timestamp_matrix_row_candidate_work=sum(min(int(np.searchsorted(times,t,side="left")),window)*230*n
                for t, n in timestamp_panels.items()))
    return dict(total)


def workload_score(counts, variants, protocol):
    # Matrix-width proxies: both + both + key-half + value-half = three
    # full-KV decompositions. Causal trajectories are shared across variants.
    factor = sum(1. if variant.startswith("both") else .5 for variant in variants)
    matrix = counts["release_matrix_row_candidate_work"] if protocol == "release" else counts["timestamp_matrix_row_candidate_work"]
    return (matrix*factor + 1024*(counts["initial_rows"]+2*counts["behavior_appends_before_last_read"])
            + 230*1024*counts["timestamps"])


def assign_gpus(jobs):
    """Greedy longest-workload-first assignment; costs use only fixed inputs."""
    loads, orders = [0., 0.], [[], []]
    for job in sorted(jobs, key=lambda j: (-j["workload_proxy"], j["id"])):
        gpu = min(range(2), key=lambda g: (loads[g], g))
        job["gpu"] = gpu
        job["argv"] += ["--gpu", str(gpu)]
        job["command"] = shlex.join(job["argv"])
        orders[gpu].append(job["id"])
        loads[gpu] += job["workload_proxy"]
    return dict(policy="LPT input-only proxy; each GPU executes its assigned jobs sequentially; no concurrent jobs on one card",
        proxy_is_measured_runtime=False, proxy_units="history/candidate matrix and causal replay work; no timing guarantee",
        gpu_orders={str(g): orders[g] for g in range(2)}, gpu_proxy_totals={str(g): loads[g] for g in range(2)})


def _command(binding, *, job_id, protocol, users, variants, output_root, diagnostic=False):
    driver = "run_lowrank_release.py" if protocol == "release" else TIMESTAMP_DRIVER
    output = Path(output_root)/job_id/"attempt_1"
    argv = [str(ROOT/".venv/bin/python"), str(Path(__file__).with_name(driver).resolve()),
        "--edge", str(binding["edge"]), "--input-root", binding["input_root"],
        "--population-file", binding["population_file"], "--probe", binding["full_h_checkpoint"],
        "--fg-probe", binding["trained_fg_checkpoint"], "--fg-method-label", "fg", "--output", str(output.resolve()),
        "--candidates", "230", "--candidate-chunk", "230", "--row-chunk", "1024", "--checkpoint-users", "1" if diagnostic else "4",
        "--variants", ",".join(variants), "--method", "randomized", "--oversample", "16", "--power-iterations", "2",
        "--batch-rank-scores", "--execution", "projected"]
    if protocol == "release":
        argv += ["--batch-heads", "25", "--verify-exact-heads", "2"]
        if diagnostic:
            argv += ["--verify-every-user", "--compare-rank-execution"]
    else:
        argv += ["--decomposition-batch-heads", "25", "--verify-svd-heads", "2"]
    if diagnostic:
        argv += ["--diagnostic-users", "8"]
    elif len(users) == 512:
        argv += ["--sensitivity-users", "512"]
    return dict(id=job_id, edge=binding["edge"], seed=binding["seed"], protocol=protocol,
        users=len(users), uids_sha256=uid_hash(users), variants=list(variants), ranks=list(RANKS),
        weight_binding_id=binding["id"], full_h_checkpoint=binding["full_h_checkpoint"],
        trained_fg_checkpoint=binding["trained_fg_checkpoint"], population_file=binding["population_file"],
        input_root=binding["input_root"], output=str(output.resolve()), argv=argv,
        quality_auc_permitted=not diagnostic, bootstrap=dict(replicates=1000, seed=17021, unit="paired UID"),
        numerical_prerequisite="selected_release8" if protocol == "release" else "selected_timestamp8")


def audit_readiness(path, job, binding, frozen_sources):
    """Accept only the selected pair/population's bounded numerical artifact."""
    path = Path(path).resolve()
    result = json.loads(path.read_text())
    identity = json.loads((path.parent/"configuration.json").read_text())
    if (result.get("status") != "complete" or result.get("quality_auc_computed") is not False or
            result["users"] != 8 or identity["edge"] != job["edge"] or
            identity["users"] != binding["uids"][:8] or
            identity["variants"] != list(ALL_VARIANTS) or identity["ranks"] != list(RANKS) or
            identity["models"] != binding["models"] or
            identity["population"]["manifest_sha256"] != binding["population_sha256"]):
        raise ValueError("selected E6 readiness uses different users/models/variants or reports quality")
    full_sha = identity.get("full_h_checkpoint_sha256", identity.get("checkpoint_sha256"))
    if (full_sha != binding["full_h_checkpoint_sha256"] or
            identity["trained_fg_control"]["checkpoint_sha256"] != binding["trained_fg_checkpoint_sha256"]):
        raise ValueError("selected E6 readiness uses another FG/H pair")
    recorded_sources = identity.get("sources", identity.get("source_sha256", {}))
    required_sources = release_sources() if job["protocol"] == "release" else {
        str(path.relative_to(ROOT)): sha256(path) for path in source_paths()}
    if job["protocol"] == "timestamp":
        driver_path = Path(__file__).with_name(TIMESTAMP_DRIVER).resolve()
        required_sources[str(driver_path.relative_to(ROOT))] = sha256(driver_path)
    if any(recorded_sources.get(name) != digest for name, digest in required_sources.items()):
        raise ValueError("selected E6 readiness lacks the actual numerical dependency identities")
    for name, digest in recorded_sources.items():
        absolute = str((ROOT/name).resolve())
        if frozen_sources.get(absolute) != digest:
            raise ValueError("selected E6 readiness numerical sources differ")
    decomp = identity["decomposition"]
    if any(decomp.get(key) != value for key, value in
            (("method", "randomized"), ("oversample", 16), ("power_iterations", 2), ("batch_heads", 25))):
        raise ValueError("selected E6 readiness decomposition differs")
    execution = identity["execution"]
    if (execution["candidate_chunk"] != 230 or execution["row_chunk"] != 1024 or
            identity.get("candidates", identity.get("candidate_size")) != 230 or
            execution.get("mode", execution.get("interaction_and_writer")) != "projected" or
            not execution.get("batch_rank_scores", identity.get("batch_rank_scores", False))):
        raise ValueError("selected E6 readiness candidate/execution settings differ")
    expected_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if result["identity_hash"] != expected_hash:
        raise ValueError("selected E6 readiness configuration/result identity differs")
    if result["full_rank_max_error"] > 1e-5 or not np.isfinite(result["full_rank_max_error"]):
        raise ValueError("selected E6 readiness fails full-rank logit equality")
    guard = result["energy_guard"]
    if guard["matrices"] <= 0 or guard["passed_without_exact"]+guard["exact_fallback_matrices"] != guard["matrices"]:
        raise ValueError("selected E6 readiness lacks all-matrix numerical screening")
    if job["protocol"] == "timestamp" and (guard["expected_matrices"] != guard["matrices"] or
            guard["all_matrix_scope"] is not True):
        raise ValueError("selected timestamp E6 readiness has incomplete matrix coverage")
    if job["protocol"] == "release":
        checks = result["rank_execution_comparison"]
        if (checks["checks"] <= 0 or checks["logit_max_error"] is None or
                not np.isfinite(checks["logit_max_error"]) or checks["logit_max_error"] > 1e-5):
            raise ValueError("selected E6 readiness fails serial/batched rank equality")
    if not 0 < result["peak_reserved_bytes"] < 80*1024**3:
        raise ValueError("selected E6 readiness lacks measured80GiB-card memory headroom")
    return dict(status="passed", path=str(path), sha256=sha256(path),
        configuration_sha256=sha256(path.parent/"configuration.json"), seconds=result["seconds"],
        peak_reserved_bytes=result["peak_reserved_bytes"], full_rank_max_error=result["full_rank_max_error"],
        memory_scope="measured selected8 only; longer users can require more memory; old width128 guard is not reused")


def generate(formal_path, main_root, *, output_root, scope_path=SCOPE, scenario_root=SCENARIOS,
             release_readiness=None, timestamp_readiness=None):
    formal_path, main_root = Path(formal_path).resolve(), Path(main_root).resolve()
    _scope(scope_path)
    code = sources()
    result = dict(format="e6_formal_handoff_v1", status="waiting_formal_plan", gpu_launch_performed=False,
        formal_plan=str(formal_path), main_fitting_root=str(main_root), scope=str(Path(scope_path).resolve()),
        scope_sha256=sha256(scope_path), sources_sha256=code, jobs=[], weight_bindings={}, readiness_jobs=[],
        ready_to_launch_quality=False, complete_serving_cost_ready=False,
        cost_scope="root CPU sidecar on these exact FG/H pairs and complete2000-user E14 event ledgers; not E5 main2000 costs; SVD/intervention work is research overhead",
        population_policy="preserve all five predeclared feedback_union_2000 cohorts; no H-gain or per-edge positivity filtering",
        inference_scope="hindsight read-rank interventions on full-H snapshots; FG owns a separate causal trajectory; no lowrank-writer deployment claim")
    if not formal_path.exists():
        result["missing_dependencies"] = [str(formal_path)]
        return result
    formal = json.loads(formal_path.read_text())
    result["formal_plan_sha256"] = sha256(formal_path)
    if formal["status"] != "prepared_for_root_review":
        result.update(status="waiting_selection", upstream_status=formal["status"])
        return result
    formal_frozen = merge_frozen(formal["frozen_sha256"])
    _check_files(formal_frozen)
    jobs = _grid(formal, main_root)
    missing = [str(quality.checkpoint_path(main_root, job, method).resolve()) for job in jobs
        for method in ("fg", "fgh") if not quality.checkpoint_path(main_root, job, method).is_file()]
    manifests = {edge: Path(scenario_root)/f"edge_{edge}"/"feedback_union_2000/scenario_manifest.json" for edge in EDGES}
    missing.extend(str(path.resolve()) for path in manifests.values() if not path.is_file())
    if missing:
        result.update(status="waiting_weights_or_population", missing_dependencies=missing)
        return result
    timestamp_driver = Path(__file__).with_name(TIMESTAMP_DRIVER)
    if not timestamp_driver.exists():
        result.update(status="waiting_timestamp_guard_entry", missing_dependencies=[str(timestamp_driver)])
        return result
    populations, workloads = {}, {}
    for edge, path in manifests.items():
        binding, users, population, provenance = load_scenario_population(path, edge, role="e6",
            minimum_users=2000, require_qualification=True, require_positive=False)
        if len(users) != 2000:
            raise ValueError("E6 formal handoff requires the unchanged exact2000 feedback cohort")
        populations[edge] = binding, users, population, provenance
        histories = restore_pack(provenance["history_path"])
        rows = pq.read_table(provenance["requests_path"], columns=["uid", "query_timestamp", "item_idx"], filters=[("uid", "in", users)]).to_pylist()
        workloads[edge] = {n: workload_counts(histories, rows, users[:n], binding["settings"]) for n in (512, 2000)}
    for job in jobs:
        edge, seed = job["edge"], job["seed"]
        binding, users, population, provenance = populations[edge]
        fit_binding = formal["main_input_bindings"][str(edge)]
        if (fit_binding["input_binding_sha256"] != provenance["calibration_input_binding_sha256"] or
                fit_binding["population_sha256"] != provenance["source_population_sha256"]):
            raise ValueError("E6 population calibration source differs from actual formal fitting inputs")
        fg_path, full_path = (quality.checkpoint_path(main_root, job, method).resolve() for method in ("fg", "fgh"))
        audited = {method: quality.audit_weight(path, job, fit_binding, method)
            for method, path in (("fg", fg_path), ("fgh", full_path))}
        full, metadata = load_variant(full_path, "cpu")
        models = provenance["anchor_models"]
        fitting = validate_fit_roles(metadata, population, users, edge, models, provenance["calibration_input_binding_sha256"])
        fg, fg_metadata, paired = load_fg_control(fg_path, full, metadata, source_population=population,
            evaluation_uids=users, edge=edge, models=models,
            calibration_binding_sha256=provenance["calibration_input_binding_sha256"])
        calibration_costs = {method: actual.get("cumulative_calibration_flops_estimate")
            for method, actual in (("fg", fg_metadata), ("fgh", metadata))}
        if any(value is None or not np.isfinite(value) or value < 0 for value in calibration_costs.values()):
            raise ValueError("E6 actual FG/H weights lack finite standalone cumulative calibration FLOPs")
        key = f"edge_{edge}_seed_{seed}"
        result["weight_bindings"][key] = dict(id=key, edge=edge, seed=seed, full_h_checkpoint=str(full_path),
            full_h_checkpoint_sha256=sha256(full_path), trained_fg_checkpoint=str(fg_path), trained_fg_checkpoint_sha256=sha256(fg_path),
            input_root=str(Path(provenance["input_binding_path"]).parent.parent), models=models,
            population_file=str(manifests[edge].resolve()), population_sha256=provenance["manifest_sha256"],
            uids=users, uids_sha256=uid_hash(users), users=2000, population_provenance=provenance,
            fitting=fitting, paired_control=paired, weight_audit=audited,
            cumulative_calibration_flops_estimate=calibration_costs,
            h_best_step=metadata["best_step"], qualification=scenario_qualification_record(provenance),
            calibration_source_input_binding_sha256=provenance["calibration_input_binding_sha256"])
        del fg, full
    planned = []
    for key, binding in result["weight_bindings"].items():
        variants = ALL_VARIANTS if binding["seed"] == 17 else ("both_norm",)
        job = _command(binding, job_id=f"main_{key}", protocol="release", users=binding["uids"],
            variants=variants, output_root=output_root)
        job["workload_counts"] = workloads[binding["edge"]][2000]
        planned.append(job)
    for edge in (2, 5):
        binding = result["weight_bindings"][f"edge_{edge}_seed_17"]
        for protocol in ("release", "timestamp"):
            job = _command(binding, job_id=f"protocol_{protocol}_edge_{edge}_seed_17", protocol=protocol,
                users=binding["uids"][:512], variants=("both_norm",), output_root=output_root)
            job["workload_counts"] = workloads[edge][512]
            planned.append(job)
    for job in planned:
        job["workload_proxy"] = workload_score(job["workload_counts"], job["variants"], job["protocol"])
    result["scheduling"] = assign_gpus(planned)
    selected = result["weight_bindings"]["edge_2_seed_17"]
    for protocol in ("release", "timestamp"):
        diagnostic = _command(selected, job_id=f"selected_{protocol}8", protocol=protocol,
            users=selected["uids"][:8], variants=ALL_VARIANTS, output_root=output_root, diagnostic=True)
        diagnostic["argv"] += ["--gpu", "0"]
        diagnostic["command"] = shlex.join(diagnostic["argv"])
        diagnostic["numerical_prerequisite"] = None
        result["readiness_jobs"].append(diagnostic)
    readiness = {}
    for job, path in zip(result["readiness_jobs"], (release_readiness, timestamp_readiness), strict=True):
        readiness[job["id"]] = audit_readiness(path, job, selected, code) if path else dict(status="not_run", quality_auc_permitted=False)
    result.update(status="prepared_for_root_review", jobs=planned, readiness=readiness,
        ready_to_launch_quality=all(row["status"] == "passed" for row in readiness.values()),
        job_counts=dict(formal_release=15, protocol512=4, total_quality_commands=19, selected8_diagnostics=2,
            main_both_norm_points=105, secondary_kv_points=105),
        workload_by_edge={str(edge): {str(n): row for n, row in values.items()} for edge, values in workloads.items()},
        statistics="1000 paired-UID draws within each edge; across-edge reporting must share UID multiplicities and ratio equal-edge mean AUC differences; negative/zero H gains retained, no signed-ratio recovery claim",
        cost_sidecar=dict(format="e6_complete_cost_v1", binding_key="weight_bindings keys", required_fields=["formal_plan_sha256",
            "full_h_checkpoint_sha256", "trained_fg_checkpoint_sha256", "population_sha256", "uids_sha256", "actual_event", "analytic230"],
            research_overhead="separate randomized/exact SVD, oracle guards and hindsight rank read interventions; no deployable lowrank-write cost assertion"))
    snapshots = [code, formal_frozen,
        {str(formal_path): sha256(formal_path), str(Path(scope_path).resolve()): sha256(scope_path)}]
    for binding in result["weight_bindings"].values():
        snapshot = {binding["full_h_checkpoint"]: binding["full_h_checkpoint_sha256"],
            binding["trained_fg_checkpoint"]: binding["trained_fg_checkpoint_sha256"],
            binding["population_file"]: binding["population_sha256"]}
        provenance = binding["population_provenance"]
        for name in ("history", "requests", "source_population", "input_binding", "calibration_input_binding", "qualification_summary"):
            reference = {provenance[name+"_path"]: provenance[name+"_sha256"]}
            snapshot = merge_frozen(snapshot, reference)
        snapshots.append(snapshot)
    frozen = merge_frozen(*snapshots)
    _check_files(frozen)
    if sources() != code:
        raise ValueError("E6 handoff sources changed while auditing real weights")
    result["frozen_sha256"] = frozen
    return result


def main(args):
    result = generate(args.formal_plan, args.main_fitting_root, output_root=args.result_root,
        scope_path=args.scope, scenario_root=args.scenario_root,
        release_readiness=args.release_readiness, timestamp_readiness=args.timestamp_readiness)
    write_json(args.output/"plan.json", result)
    (args.output/"README.md").write_text("E6 formal handoff\n\nStatus: "+result["status"]+". No GPU was launched.\n\n"
        "The exact E5 primary FG/H pair is checked with the quality audit and paired trajectory validator. "
        "All five fixed feedback cohorts remain, including adverse anchors, zero H and partial edge5. "
        "Selected weights need separate8-user release/timestamp numerical and memory checks before formal execution. "
        "The19 quality commands share seed17 raw/H work across four K/V variants; each GPU runs sequentially according to the input-only workload plan. "
        "Complete serving costs require the separate CPU sidecar on the identical bindings; SVD/oracle work remains research overhead.\n")
    print(json.dumps(dict(status=result["status"], jobs=len(result["jobs"]),
        ready_to_launch_quality=result["ready_to_launch_quality"], gpu_launched=False)))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-plan", type=Path, required=True)
    parser.add_argument("--main-fitting-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--scope", type=Path, default=SCOPE)
    parser.add_argument("--scenario-root", type=Path, default=SCENARIOS)
    parser.add_argument("--release-readiness", type=Path)
    parser.add_argument("--timestamp-readiness", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
