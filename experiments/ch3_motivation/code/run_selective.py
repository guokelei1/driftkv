#!/usr/bin/env python3
"""Recompute Chapter 3 selective curves on newly frozen partial populations.

Run ``--check`` first for an 8--16-user numerical/performance diagnostic. It
reports no AUC. ``--run`` is a separate explicit population run, with fresh
Reuse/Rebuild anchors and bounded-loss score blocks.
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
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import torch

from experiments.ch3_motivation.code import common, selective
from hstu_kvcache.evaluation.baselines import event_shapes, layer_boundary_accounting
from hstu_kvcache.evaluation import fast_baselines as fast
from hstu_kvcache.evaluation.statistics import paired_user_bootstrap
from hstu_kvcache.evaluation.cost import CostModel, eager_read
from hstu_kvcache.common.artifacts import sha256


def sources():
    paths = ["experiments/ch3_motivation/code/run_selective.py", "experiments/ch3_motivation/code/selective.py",
        "experiments/ch3_motivation/code/common.py", "src/hstu_kvcache/evaluation/baselines.py",
        "src/hstu_kvcache/evaluation/fast_baselines.py", "src/hstu_kvcache/evaluation/statistics.py",
        "src/hstu_kvcache/methods/baselines/protocol/cost.py", "src/hstu_kvcache/evaluation/cost.py",
        "src/hstu_kvcache/methods/cachetide/kernels/reader.py",
        "src/hstu_kvcache/methods/baselines/layer_recompute/core.py",
        "src/hstu_kvcache/methods/baselines/sparse_recompute.py",
        "src/hstu_kvcache/methods/baselines/tail_recompute/core.py",
        "src/hstu_kvcache/methods/baselines/deviation_recompute/core.py",
        "src/hstu_kvcache/methods/baselines/query_recompute/core.py",
        "src/hstu_kvcache/evaluation/trajectory.py", "src/hstu_kvcache/models/state_transition.py",
        "src/hstu_kvcache/models/hstu.py", "src/hstu_kvcache/models/block.py",
        "src/hstu_kvcache/adaptation/reader.py"]
    return {path: sha256(ROOT / path) for path in paths}


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _panel(population, name, seed):
    if name == "sensitivity":
        return population["population_sensitivity"][str(seed)]
    return population[name]


def _score_key(row):
    return row["method"], row["uid"], row["query_timestamp"], row["raw_item_id"], row["label"]


def compare_scores(reference, actual, tolerance=1e-4):
    reference = {_score_key(row): row["score"] for row in reference}
    by_method = defaultdict(list)
    if len(actual) != len(reference) or {_score_key(row) for row in actual} != set(reference):
        raise AssertionError("optimized/reference event coverage differs")
    for row in actual:
        by_method[row["method"]].append(abs(row["score"] - reference[_score_key(row)]))
    maximum = max((max(values) for values in by_method.values()), default=0.)
    result = dict(scores=len(actual), maximum_absolute_difference=maximum, tolerance=tolerance,
        by_method={tag: max(values) for tag, values in by_method.items()}, passed=maximum <= tolerance)
    if not result["passed"]:
        raise AssertionError(f"optimized/reference maximum error {maximum} exceeds {tolerance}")
    return result


def _measure(device, operation):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = operation()
    torch.cuda.synchronize(device)
    return result, dict(seconds=time.perf_counter() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def check(parent, current, histories, by_user, users, policies, cfg, device, output, identity):
    """No AUC is computed by any branch of this numerical diagnostic."""
    if not 8 <= len(users) <= 16:
        raise ValueError("the selective diagnostic uses 8--16 users")
    common.quality_role(users, diagnostic=True)
    (reference, metadata), reference_resources = _measure(device, lambda:
        selective.evaluate_reference(parent, current, histories, by_user, users, policies,
                                     cfg, device, semantics="persistent"))
    checks = []
    for enabled in (False, True):
        engine = fast.NativeGraphs(enabled=enabled)
        (actual, execution), resources = _measure(device, lambda:
            fast.evaluate_batched(parent, current, histories, by_user, users, policies,
                cfg, device, graphs=engine, stop_after_last_query=False))
        checks.append(dict(cuda_graphs=enabled, numerical=compare_scores(reference, actual),
                           resources=resources, execution=execution))
        del engine, actual
    result = dict(role="numerical_performance_diagnostic", users=len(users),
        user_ids=[int(uid) for uid in users], quality_metrics_computed=False,
        semantics="once before first history consumer; repaired state persists independently",
        identity=identity, reference=dict(metadata=metadata, resources=reference_resources),
        comparisons=checks, passed=all(item["numerical"]["passed"] for item in checks),
        timing_scope="shared-machine progress and numerical readiness; not controlled paper timing")
    common.write_json(output / "check.json", result)
    return result


def cost_ledgers(histories, by_user, users, policies, shapes, cm, cfg):
    """Keep actual-event work and the 230-read analytic scenario explicit."""
    native_appends = sum(count * selective._cost_for_length(cm, n, cfg).band_append(
        n, width, window_size=cfg["history_length"])
        for (n, width, _full), count in shapes["appends"].items())
    actual_reads = sum(count * eager_read(cm, n, queries=q)
                       for (n, q), count in shapes["queries"].items())
    old_initial = sum(count * selective._cost_for_length(cm, n, cfg).full_cache(n)
                      for n, count in shapes["initial"].items())
    denominator = selective.repair_cost(dict(method="rebuild"), shapes, cm, cfg)[0]
    source_lengths = Counter()
    first_230 = Counter(shapes["first"])
    analytic_reads = 0
    for uid in users:
        events = histories.rows[uid]
        stop = int(np.searchsorted(events[0], cfg["release_day"] * common.DAY, side="left"))
        n = min(stop, cfg["history_length"])
        source_lengths[n] += 1
        has_append = stop < int(np.searchsorted(events[0], cfg["end_day"] * common.DAY, side="left"))
        if not by_user.get(uid) and not has_append:
            # The hypothetical reads create a consumer for an otherwise idle user.
            first_230[n, "query", min(cfg["query_batch"], cfg["candidate_reads_per_user"])] += 1
        reads = cfg["candidate_reads_per_user"]
        for offset in range(0, reads, cfg["query_batch"]):
            analytic_reads += eager_read(cm, n, queries=min(cfg["query_batch"], reads - offset))
    scenario_shapes = dict(shapes, first=first_230)
    denominator_230 = selective.repair_cost(dict(method="rebuild"), scenario_shapes, cm, cfg)[0]
    actual, normalized = [], []
    for policy in policies:
        added, sorts = selective.repair_cost(policy, shapes, cm, cfg)
        scenario_added, scenario_sorts = selective.repair_cost(policy, scenario_shapes, cm, cfg)
        actual.append(dict(method=policy["tag"], added_flops=added,
            calibration_flops=int(policy.get("calibration_flops", 0)),
            selection_and_repair_flops=added-int(policy.get("calibration_flops", 0)),
            percent_of_once_rebuild=100 * added / denominator,
            sorting_comparisons_estimate=sorts, extra_read_flops=0))
        normalized.append(dict(method=policy["tag"], added_flops=scenario_added,
            percent_of_once_rebuild=100 * scenario_added / denominator_230,
            sorting_comparisons_estimate=scenario_sorts, extra_read_flops=0))
    return dict(
        actual_event_ledger=dict(users=len(users), once_rebuild_flops=denominator,
            common_old_cache_initialization_flops=old_initial, common_native_append_flops=native_appends,
            common_native_read_flops=actual_reads, results=actual,
            first_consumer_shapes=[dict(shape=list(shape), count=count) for shape, count in shapes["first"].items()],
            append_shapes=[dict(shape=list(shape), count=count) for shape, count in shapes["appends"].items()],
            query_shapes=[dict(shape=list(shape), count=count) for shape, count in shapes["queries"].items()],
            scope="all population events, including no-feedback users and events after final scored request"),
        candidate_reads_230=dict(users=len(users), candidate_reads_per_user=cfg["candidate_reads_per_user"],
            once_rebuild_flops=denominator_230, common_native_read_flops=analytic_reads,
            common_native_append_flops=native_appends, results=normalized,
            source_length_histogram=[dict(rows=n, users=count) for n, count in source_lengths.items()],
            first_consumer_shapes=[dict(shape=list(shape), count=count) for shape, count in first_230.items()],
            scope="analytic read-volume normalization; native reads use release source lengths; actual appends retained; "
                  "otherwise idle users acquire a first query; no selective policy adds per-read arithmetic",
            reference_20_appends_applied=False),
        arithmetic="multiply-add=2; actual executed dense/tiled shapes; shared research execution does not discount standalone method cost")


def summarize(scores, users, policies, costs, cfg, output, identity, execution):
    common.quality_role(users)
    from sklearn.metrics import roc_auc_score
    groups = defaultdict(list)
    for row in scores:
        groups[row["method"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: (row["uid"], row["query_timestamp"], row["raw_item_id"]))
    aucs = {tag: float(roc_auc_score([row["label"] for row in rows], [row["score"] for row in rows]))
            for tag, rows in groups.items()}
    if set(aucs) != {policy["tag"] for policy in policies}:
        raise ValueError("population score coverage is missing a policy")
    intervals, confidence = paired_user_bootstrap(groups, users,
        replicates=cfg["confidence_replicates"], seed=17021+cfg["edge"])
    gap = aucs["rebuild"] - aucs["reuse"]
    cost_by_tag = {row["method"]: row for row in costs["actual_event_ledger"]["results"]}
    normalized = {row["method"]: row for row in costs["candidate_reads_230"]["results"]}
    rows = []
    for policy in policies:
        tag = policy["tag"]
        gain = aucs[tag] - aucs["reuse"]
        rows.append(dict(method=tag, setting={key: policy[key] for key in ("interval", "layers", "fraction") if key in policy},
            auc=aucs[tag], auc_gain_over_reuse=gain,
            recovery_percent=100 * gain / gap if gap else None,
            **intervals[tag], **{key: value for key, value in cost_by_tag[tag].items() if key != "method"},
            normalized_230_percent_of_once_rebuild=normalized[tag]["percent_of_once_rebuild"]))
    result = dict(role="measured_partial_population", experiment="E3", model_scale="Yambda-Large",
        edge=f"V{cfg['edge']-1}->V{cfg['edge']}", quality_users=len(users),
        users_with_feedback=len({row["uid"] for row in groups["reuse"]}), feedback_events=len(groups["reuse"]),
        evaluation_window=cfg.get("evaluation_window", dict(name="E14", complete_source_horizon=True)),
        reference_gap=gap, results=rows, confidence=confidence, identity=identity,
        costs=costs, execution=execution,
        quality_aggregation="single AUC from every raw feedback score; no per-user or block AUC averaging",
        state_schedule="independent adjacent edge; once-first-consumer repair; causal own-state appends and FIFO",
        timing_scope="execution progress; not a controlled paper timing result",
        historical_scores_reused=False)
    common.write_json(output / "summary.json", result)
    fields = ["method", "layers", "interval_start", "interval_end", "fraction", "auc", "auc_ci95_low", "auc_ci95_high",
        "auc_gain_over_reuse", "gain_ci95_low", "gain_ci95_high", "recovery_percent", "recovery_ci95_low", "recovery_ci95_high",
        "added_flops", "calibration_flops", "selection_and_repair_flops", "percent_of_once_rebuild",
        "normalized_230_percent_of_once_rebuild", "quality_users", "feedback_users", "feedback_events"]
    with (output / "metrics_by_point.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            setting = row["setting"]
            record = {key: row[key] for key in ("method", "auc", "auc_gain_over_reuse", "recovery_percent", "added_flops",
                "calibration_flops", "selection_and_repair_flops", "percent_of_once_rebuild", "normalized_230_percent_of_once_rebuild")}
            record.update(layers=setting.get("layers"), fraction=setting.get("fraction"),
                interval_start=setting.get("interval", [None, None])[0], interval_end=setting.get("interval", [None, None])[1],
                quality_users=len(users), feedback_users=result["users_with_feedback"], feedback_events=result["feedback_events"])
            for name, prefix in (("auc_ci95", "auc_ci95"), ("auc_gain_over_reuse_ci95", "gain_ci95"),
                                 ("recovery_percent_ci95", "recovery_ci95")):
                ci = row[name] or [None, None]
                record[prefix+"_low"], record[prefix+"_high"] = ci
            writer.writerow(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--panel", choices=("development", "main", "confirmation", "sensitivity"), default="development")
    parser.add_argument("--population-seed", type=int, default=20261023)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-users", type=int)
    parser.add_argument("--gpu", help="physical CUDA_VISIBLE_DEVICES value, set before CUDA initialization")
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run and args.diagnostic_users is not None:
        parser.error("--diagnostic-users is for --check; population runs use the frozen panel")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    directory = args.inputs or ROOT / f"experiments/ch3_motivation/inputs/edge_{args.edge}"
    population_path = directory.parent / "population.json"
    population = json.loads(population_path.read_text())
    binding, histories, by_user = common.load_edge(directory)
    cfg = dict(binding["settings"])
    if cfg["edge"] != args.edge:
        raise ValueError("edge argument differs from input binding")
    for name in ("history", "requests"):
        if sha256(directory / binding[name]["path"]) != binding[name]["sha256"]:
            raise ValueError(f"frozen {name} pack changed")
    users = _panel(population, args.panel, args.population_seed)
    if args.check:
        count = args.diagnostic_users if args.diagnostic_users is not None else 8
        if not 8 <= count <= 16:
            parser.error("--check uses 8--16 diagnostic users")
        users = users[:count]
    else:
        common.quality_role(users)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / "quality_shards").glob("part_*.json")) and not args.resume:
        raise ValueError("score blocks already exist; use --resume with identical code/inputs or a new output directory")
    code_hashes = sources()
    device = torch.device("cuda:0")
    parent, current, model_hashes = common.load_models(args.edge, device)
    identity = dict(experiment="ch3_selective_fresh", panel=args.panel,
        population_seed=args.population_seed if args.panel == "sensitivity" else population["seed"],
        population_sha256=sha256(population_path), input_binding_sha256=sha256(directory / "inputs.json"),
        input_hashes={name: binding[name]["sha256"] for name in ("history", "requests")},
        checkpoint_sha256=model_hashes, source_sha256=code_hashes, settings=cfg,
        evaluation_users_sha256=_hash_json(users), calibration_users_sha256=_hash_json(population["fit"][:cfg["profile_users"]]))
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("output code/inputs/settings changed; use a new attempt rather than mixing or resuming score blocks")
    common.write_json(identity_path, identity)
    started = time.perf_counter()
    common.write_json(output / "status.json", dict(state="profiling", command=sys.argv, pid=os.getpid(), identity_sha256=_hash_json(identity)))
    profile_path = output / "layer_profile.json"
    if args.resume and profile_path.exists():
        profiles = json.loads(profile_path.read_text())
        profile_resources = dict(resumed=True)
    else:
        def do_profile():
            samples = common.capture_samples(parent, current, histories, population["fit"][:cfg["profile_users"]],
                                             cfg, device, boundaries=True)
            return selective.profile_intervals(current, samples, histories, population["fit"][:cfg["profile_users"]], cfg, device)
        (profiles, _), profile_resources = _measure(device, do_profile)
        common.write_json(profile_path, profiles)
    cm = CostModel(current.cfg.hidden_size, len(current.blocks), current.cfg.num_heads, cfg["attention_backend"])
    shapes = event_shapes(histories, by_user, users, cfg)
    policies, denominator = selective.make_policies(cm, shapes, profiles, cfg)
    common.write_json(output / "policies.json", dict(policies=policies, once_rebuild_flops=denominator,
        layer_profile_resources=profile_resources, layer_boundary_state=layer_boundary_accounting(policies, shapes, cm)))
    if args.check:
        result = check(parent, current, histories, by_user, users, policies, cfg, device, output, identity)
        result["layer_profile_resources"] = profile_resources
        result["seconds_total"] = time.perf_counter() - started
        if sources() != code_hashes:
            raise ValueError("execution sources changed during diagnostic; rerun after code is frozen")
        common.write_json(output / "check.json", result)
        print(json.dumps(dict(passed=result["passed"], users=len(users), quality_metrics_computed=False,
            seconds=result["seconds_total"], check=str(output / "check.json"))), flush=True)
    else:
        costs = cost_ledgers(histories, by_user, users, policies, shapes, cm, cfg)
        common.write_json(output / "costs.json", costs)
        def progress(value):
            print(json.dumps(dict(stage="selective_quality", **value)), flush=True)
            common.write_json(output / "status.json", dict(state="quality", **value))
        scores, execution = fast.evaluate_checkpointed(parent, current, histories, by_user,
            users, policies, cfg, device, output, identity=dict(identity, layer_profile_sha256=sha256(profile_path)), progress=progress)
        if sources() != code_hashes:
            raise ValueError("execution sources changed during quality run; retain shards for audit but do not report mixed-source results")
        result = summarize(scores, users, policies, costs, cfg, output, identity, execution)
        print(json.dumps(dict(users=len(users), feedback_events=result["feedback_events"],
            summary=str(output / "summary.json"))), flush=True)
    common.write_json(output / "status.json", dict(state="complete", mode="check" if args.check else "run",
        users=len(users), seconds=time.perf_counter() - started, source_sha256=code_hashes))


if __name__ == "__main__":
    main()
