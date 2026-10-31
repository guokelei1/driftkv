"""Audit one real chain edge, with optional historical native score references."""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from experiments.recflow_full_population.code.audit_scored_requests import independent_metrics
from experiments.recflow_full_population.code.batches import RequestBatches
from experiments.recflow_full_population.code.data_common import json_save
from experiments.recflow_full_population.code.protocol import day_bounds_ms, day_date, paired_uid_interval
from experiments.recflow_release_design.code.interval_protocol import SampledRequestBatches, ancestry


INTEGER_FIELDS = {"uid", "rid", "day", "candidates", "exposed", "positives", "history_length"}
METRICS = [f"{metric}@{k}" for metric in ("ndcg", "recall") for k in (5, 10, 20)]


def read(path):
    return json.loads(Path(path).read_text())


def key(row):
    return tuple(int(row[name]) for name in ("uid", "day", "rid"))


def metric_rows(directory, selected_keys=None):
    result = {}
    paths = sorted(Path(directory).glob("rank_*.csv"))
    if not paths:
        raise ValueError(f"No distributed metric files: {directory}")
    for path in paths:
        with path.open() as handle:
            for raw in csv.DictReader(handle):
                identity = key(raw)
                if selected_keys is not None and identity not in selected_keys:
                    continue
                if identity in result:
                    raise ValueError("Repeated request in distributed metric files")
                result[identity] = {name: (int(value) if name in INTEGER_FIELDS else float(value))
                                    if value else None for name, value in raw.items()}
    return result, paths


def audit(run, plan, summary):
    data, day = Path(plan["data"]), int(plan["eval_day"])
    if summary.get("status") != "completed":
        raise ValueError("Scoring is not complete")
    chosen = np.load(run / "chosen_uids.npy")
    if chosen.ndim != 1 or len(np.unique(chosen)) != len(chosen):
        raise ValueError("Chosen UID list is not a distinct one-dimensional population")
    full = RequestBatches(data, [day], role=2)
    dataset = SampledRequestBatches(data, day, run / "selection.npy")
    expected_selection = full.records[np.isin(full.records["uid"], chosen)]
    if not np.array_equal(dataset.records, expected_selection):
        raise ValueError("Selection omits or changes a chosen user's day requests")
    expected = {key(row): row for row in dataset.records}
    if len(expected) != len(dataset):
        raise ValueError("Repeated request in selected population")
    lo, hi = day_bounds_ms(day)
    if np.any((dataset.records["ts"] < lo) | (dataset.records["ts"] >= hi)):
        raise ValueError("Selected request timestamp lies outside the evaluation day")
    lineage = ancestry(plan["checkpoint"], plan["cache_source"])
    if day <= lineage[-1]["train_end_day"]:
        raise ValueError("Evaluation does not follow target training")
    rows, csv_paths = metric_rows(run)
    if len(csv_paths) != summary["world_size"] or rows.keys() != expected.keys():
        raise ValueError("Metric files do not cover the complete selected population exactly once")
    stage = data / "stage_v2" / f"d{day:02d}"
    features = np.load(stage / "features.npy", mmap_mode="r")
    truth = np.load(stage / "labels.npy", mmap_mode="r")
    seen, candidate_total, positive_total, positive_requests = set(), 0, 0, 0
    methods = ("current", "reuse")
    sums = {method: {metric: 0. for metric in METRICS} for method in methods}
    maximum_metric_error = 0.
    score_paths = sorted(run.glob("rank_*_scores/batch_*.npz"))
    if not score_paths:
        raise ValueError("No raw score files")
    for path in score_paths:
        with np.load(path) as stored:
            arrays = {name: stored[name] for name in stored.files}
        ids, labels, counts = (arrays[name] for name in ("raw_video_ids", "labels", "counts"))
        actual = {method: independent_metrics(ids, labels, counts, arrays[method]) for method in methods}
        for j, raw_count in enumerate(counts):
            identity = tuple(int(arrays[name][j]) for name in ("uid", "day", "rid"))
            if identity in seen or identity not in expected:
                raise ValueError("Raw scores contain a repeated or unselected request")
            seen.add(identity)
            req, row, count = expected[identity], rows[identity], int(raw_count)
            start, stop = int(req["start"]), int(req["stop"])
            if (count != stop - start or row["candidates"] != count
                    or not np.array_equal(ids[j, :count], features[start:stop, 0])
                    or not np.array_equal(labels[j, :count], truth[start:stop])):
                raise ValueError("Saved candidate IDs or labels differ from the full original request")
            if len(np.unique(ids[j, :count])) != count:
                raise ValueError("Repeated candidate in a request")
            positives, exposed = int((labels[j, :count] == 1).sum()), int((labels[j, :count] >= 0).sum())
            if (row["positives"] != positives or row["exposed"] != exposed
                    or row["history_length"] != min(1024, int(req["history_length"]))):
                raise ValueError("Recorded feedback or history length differs from actual request")
            candidate_total += count
            positive_total += positives
            positive_requests += positives > 0
            for method in methods:
                for metric in METRICS:
                    saved = row[f"{method}_{metric}"]
                    if positives:
                        if saved is None:
                            raise ValueError("Positive request lacks a defined metric")
                        error = abs(float(actual[method][metric][j]) - saved)
                        maximum_metric_error = max(maximum_metric_error, error)
                        if error > 1e-12:
                            raise ValueError("Per-request metric differs from independent recomputation")
                        sums[method][metric] += float(actual[method][metric][j])
                    elif saved is not None:
                        raise ValueError("Zero-positive request has a defined NDCG or Recall")
    if seen != expected.keys() or summary["requests"] != len(expected):
        raise ValueError("Raw scores do not cover every selected request exactly once")
    if positive_requests != summary["positive_requests"]:
        raise ValueError("Summary positive-request denominator differs")
    maximum_summary_error = 0.
    for method in methods:
        for metric in METRICS:
            actual = sums[method][metric] / positive_requests if positive_requests else None
            saved = summary["metrics"][method][metric]
            if actual is None or saved is None:
                if actual is not None or saved is not None:
                    raise ValueError("Summary has an incorrect undefined metric")
            else:
                error = abs(actual - saved)
                maximum_summary_error = max(maximum_summary_error, error)
                if error > 1e-12:
                    raise ValueError("Summary differs from complete positive-request macro mean")
    if plan.get("native_reference"):
        controls = [read(run / f"native_reference_rank_{rank}.json") for rank in range(summary["world_size"])]
        if any(row.get("status") != "passed" for row in controls):
            raise ValueError("An original-batch native score comparison did not pass")
        native_control = {"status": "passed", "requests": sum(r["requests"] for r in controls),
                          "maximum_abs_score_error": max(r["max_abs_score_error"] for r in controls)}
    else:
        native_control = {"status": "not_available", "requests": 0,
                          "maximum_abs_score_error": None,
                          "reason": "No historical native score reference was declared for this target"}
    result = {"status": "passed", "requests": len(expected), "positive_requests": positive_requests,
              "zero_positive_requests": len(expected) - positive_requests,
              "chosen_users": len(chosen), "active_users": len(np.unique(dataset.records["uid"])),
              "positive_users": len({r["uid"] for r in rows.values() if r["positives"] > 0}),
              "candidate_instances": candidate_total, "observed_positive_instances": positive_total,
              "raw_score_files": len(score_paths), "metric_files": len(csv_paths),
              "maximum_independent_per_request_metric_error": maximum_metric_error,
              "maximum_independent_summary_error": maximum_summary_error,
              "all_chosen_users_complete_day_requests": True,
              "candidate_membership": "Every candidate ID and label matches the full logged request",
              "actual_training_path": lineage, "native_score_control": native_control}
    return result, rows, expected, chosen


def compare_native(plan, rows, lineage):
    if not plan.get("native_reference"):
        return {"status": "not_available", "requests": 0,
                "reference": None, "historical_reuse_available": False,
                "all_metrics_exactly_equal": None,
                "reason": "No historical native scores; current metrics were audited from actual raw scores only"}, None
    reference = Path(plan["native_reference"])
    old_summary = read(reference / "summary.json")
    original_args = old_summary.get("arguments", {})
    original_checkpoint = old_summary.get("checkpoint") or original_args.get("checkpoint")
    original_parent = old_summary.get("parent_checkpoint") or original_args.get("parent")
    has_reuse = "reuse" in old_summary.get("metrics", {})
    if (not original_checkpoint
            or Path(original_checkpoint).resolve() != Path(plan["checkpoint"]).resolve()
            or old_summary["days"] != [plan["eval_day"]]):
        raise ValueError("Original native reference is not the same target and evaluation day")
    if has_reuse and (not original_parent
                      or Path(original_parent).resolve() != Path(lineage[-2]["checkpoint"]).resolve()):
        raise ValueError("Historical reuse reference does not use the target's actual immediate parent")
    original, _ = metric_rows(reference, rows.keys())
    if original.keys() != rows.keys():
        raise ValueError("Original reference lacks a selected request")
    errors = {metric: 0. for metric in METRICS}
    changed = {metric: 0 for metric in METRICS}
    for identity, new in rows.items():
        old = original[identity]
        if any(new[name] != old[name] for name in INTEGER_FIELDS):
            raise ValueError("Original request membership, label counts, or history differs")
        for metric in METRICS:
            if has_reuse and f"reuse_{metric}" not in old:
                raise ValueError("Historical summary advertises reuse but its per-request metrics are absent")
            left, right = new[f"current_{metric}"], old[f"current_{metric}"]
            if left is None or right is None:
                if left is not None or right is not None:
                    raise ValueError("Original and new native metrics use different valid requests")
                continue
            error = abs(left - right)
            errors[metric] = max(errors[metric], error)
            changed[metric] += error != 0.
    parity = {"status": "passed" if max(errors.values()) <= 1e-6 else "failed",
              "requests": len(rows), "maximum_abs_per_request_error": errors,
              "requests_with_nonzero_error": changed,
              "all_metrics_exactly_equal": max(errors.values()) == 0.,
              "material_mismatch_tolerance": 1e-6,
              "historical_reuse_available": has_reuse,
              "reference": str(reference)}
    return parity, original


def describe_slice(identities, rows, original, chosen_count):
    subset = [rows[identity] for identity in identities]
    valid = [identity for identity in identities if rows[identity]["positives"] > 0]
    result = {"chosen_users": chosen_count, "requests": len(subset), "positive_requests": len(valid),
              "zero_positive_requests": len(subset) - len(valid),
              "active_users": len({row["uid"] for row in subset}),
              "positive_users": len({rows[identity]["uid"] for identity in valid})}
    if not valid:
        return dict(result, metrics=None, comparisons=None)
    has_reference = original is not None
    has_historical_reuse = has_reference and all(
        f"reuse_{metric}" in original[identity] for identity in valid for metric in METRICS)
    uids = [rows[identity]["uid"] for identity in valid]
    metrics = {}
    comparisons = {}
    for metric in METRICS:
        rebuild = np.array([rows[identity][f"current_{metric}"] for identity in valid])
        reuse = np.array([rows[identity][f"reuse_{metric}"] for identity in valid])
        old_rebuild = np.array([original[identity][f"current_{metric}"] for identity in valid]) if has_reference else None
        old_reuse = np.array([original[identity][f"reuse_{metric}"] for identity in valid]) if has_historical_reuse else None
        metrics[metric] = {"rebuild": float(rebuild.mean()), "ancestor_reuse": float(reuse.mean()),
                           "original_rebuild": float(old_rebuild.mean()) if has_reference else None,
                           "immediate_parent_reuse": float(old_reuse.mean()) if has_historical_reuse else None}
        if metric not in ("ndcg@10", "recall@10"):
            continue
        ancestor_gap = paired_uid_interval(uids, rebuild, reuse)
        ancestor_gap["relative_loss_pct"] = float(100 * (rebuild - reuse).mean() / rebuild.mean()) if rebuild.mean() else None
        immediate_gap, change = None, None
        if has_historical_reuse:
            immediate_gap = paired_uid_interval(uids, old_rebuild, old_reuse)
            immediate_gap["relative_loss_pct"] = float(100 * (old_rebuild - old_reuse).mean() / old_rebuild.mean()) if old_rebuild.mean() else None
            change = paired_uid_interval(uids, rebuild - reuse, old_rebuild - old_reuse)
        comparisons[metric] = {"ancestor_cache_loss": ancestor_gap,
                               "immediate_cache_loss": immediate_gap,
                               "change_in_cache_loss": change}
    return dict(result, metrics=metrics, comparisons=comparisons)


def analyze(run):
    started = time.monotonic()
    plan, summary = read(run / "plan.json"), read(run / "summary.json")
    audit_started = time.monotonic()
    checked, rows, expected, chosen = audit(run, plan, summary)
    checked["wall_seconds"] = time.monotonic() - audit_started
    parity_started = time.monotonic()
    parity, original = compare_native(plan, rows, checked["actual_training_path"])
    checked["sample_native_metric_comparison"] = parity
    checked["native_comparison_wall_seconds"] = time.monotonic() - parity_started
    if parity["status"] == "failed":
        checked["status"] = "failed"
        json_save(run / "audit.json", checked)
        raise ValueError("Material unexplained sampled Rebuild metric mismatch against original scores")
    json_save(run / "audit.json", checked)
    day = int(plan["eval_day"])
    lo, hi = day_bounds_ms(day)
    middle = lo + (hi - lo) // 2
    identities = list(expected)
    overall = describe_slice(identities, rows, original, len(chosen))
    slices = {}
    for name, selected_users in (("chosen_uid_positions_even", chosen[::2]),
                                  ("chosen_uid_positions_odd", chosen[1::2])):
        users = set(map(int, selected_users))
        local = [identity for identity in identities if identity[0] in users]
        slices[name] = describe_slice(local, rows, original, len(users))
    for name, start, stop in (("first_12_hours", lo, middle), ("second_12_hours", middle, hi)):
        local = [identity for identity in identities if start <= int(expected[identity]["ts"]) < stop]
        slices[name] = describe_slice(local, rows, original, len(chosen))
        slices[name]["timestamp_interval_ms_half_open"] = [start, stop]
    elapsed = {"gpu_evaluator_wall_seconds": summary["wall_seconds"],
               "cpu_candidate_audit_wall_seconds": checked["wall_seconds"],
               "cpu_native_comparison_wall_seconds": checked["native_comparison_wall_seconds"],
               "cpu_analyzer_wall_seconds": time.monotonic() - started}
    elapsed["scoring_plus_analysis_wall_seconds"] = elapsed["gpu_evaluator_wall_seconds"] + elapsed["cpu_analyzer_wall_seconds"]
    analysis = {"status": "completed", "checkpoint": plan["checkpoint"], "cache_source": plan["cache_source"],
                "immediate_optimizer_parent": checked["actual_training_path"][-1]["actual_parent"],
                "actual_training_path": checked["actual_training_path"],
                "evaluation_day": day, "evaluation_date": day_date(day),
                "evaluation_interval_ms_half_open": [lo, hi],
                "primary_metric": "positive-request macro observed-positive NDCG@10",
                "secondary_metric": "positive-request macro Recall@10",
                "positive_gap_definition": "Rebuild minus Reuse; positive means reuse loses ranking quality",
                "gap_change_definition": "When historical reuse exists: ancestor cache loss minus immediate-parent cache loss, paired on identical requests; otherwise absent",
                "overall": overall, "predetermined_slices": slices,
                "native_metric_comparison": parity, "elapsed": elapsed,
                "scope": "Development sample of original training users; matched-prefix snapshot, not persistent serving",
                "uncertainty": "Nominal user-cluster bootstrap intervals, not independent experiments or multiplicity-adjusted claims",
                "all_signs_retained": True}
    json_save(run / "analysis.json", analysis)
    completion = {"status": "completed", "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  "audit": str(run / "audit.json"), "analysis": str(run / "analysis.json"),
                  "requests": checked["requests"], "positive_requests": checked["positive_requests"],
                  "chosen_users": checked["chosen_users"], "active_users": checked["active_users"],
                  "positive_users": checked["positive_users"], "elapsed": elapsed,
                  "primary_result": overall["comparisons"]["ndcg@10"] if overall["comparisons"] else None,
                  "historical_native_reference_status": parity["status"],
                  "next_experiment_dispatched": False}
    json_save(run / "completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if (run / "completion.json").exists():
        raise ValueError("Analysis already completed; retain its existing results")
    print(json.dumps(analyze(run)), flush=True)


if __name__ == "__main__":
    main()
