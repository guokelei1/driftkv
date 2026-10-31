#!/usr/bin/env python3
"""Descriptive cold-resumption/length boundaries from completed frozen scores.

Groups use only event/request time and cache length, never score outcomes. Every
selected user/release is retained, including windows with no history consumer.
No replay, fitting, scoring or new bootstrap is performed.
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
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
import numpy as np
import pyarrow.parquet as pq
from hstu_kvcache.evaluation.paired import FixedOrderAUC
from hstu_kvcache.evaluation.release_context import END, RELEASES
from hstu_kvcache.evaluation.panels import load_history, load_panel
from hstu_kvcache.common.artifacts import sha256, write_json


def length_bucket(length):
    if length is None:
        return "idle_no_consumer"
    if length == 0:
        return "0"
    if length <= 256:
        return "1_to_256"
    if length <= 512:
        return "257_to_512"
    return "513_to_1024"


def group_windows(times, first_requests, *, releases=RELEASES, end=END, cap=1024):
    """At window t, count consecutive fully idle windows immediately before it."""
    result, idle_run = {}, 0
    for index, (version, at) in enumerate(releases):
        until = releases[index + 1][1] if index + 1 < len(releases) else end
        begin, stop = np.searchsorted(times, [at, until], side="left")
        behavior_time = None if stop == begin else int(times[begin])
        request_time = first_requests.get(version)
        available = [value for value in (behavior_time, request_time) if value is not None]
        first = min(available) if available else None
        result[version] = dict(first_consumer_timestamp=first,
            first_consumer="none" if first is None else
                ("recommendation" if request_time == first else "behavior_before_recommendation"),
            first_behavior_timestamp=behavior_time, first_request_timestamp=request_time,
            behavior_events=int(stop - begin), skipped_complete_windows=idle_run,
            cold_group="idle_no_consumer" if first is None else f"skip_{idle_run}",
            first_read_history_length=None if first is None else min(cap, int(np.searchsorted(times, first, side="left"))))
        result[version]["length_group"] = length_bucket(result[version]["first_read_history_length"])
        idle_run = idle_run + 1 if first is None else 0
    return result


def quality(labels, scores, mask):
    if not mask.any() or np.unique(labels[mask]).size < 2:
        return dict(auc=None, reference_gain_auc=None, method_gain_auc=None,
                    method_minus_rebuild_auc=None, recovery_percent=None,
                    undefined_reason="No requests or only one label class")
    auc = {name: float(FixedOrderAUC(labels[mask], value[mask],
        np.zeros(int(mask.sum()), dtype=np.int64))(np.ones(1, dtype=np.int64)))
        for name, value in scores.items()}
    reference, gain = auc["rebuild"] - auc["reuse"], auc["method"] - auc["reuse"]
    return dict(auc=auc, reference_gain_auc=reference, method_gain_auc=gain,
        method_minus_rebuild_auc=auc["method"] - auc["rebuild"],
        recovery_percent=100 * gain / reference if reference > 1e-12 else None,
        undefined_reason=None if reference > 1e-12 else "Observed Rebuild gain is nonpositive")


def summarize_dimension(dimension, snapshots, labels, scores, uids, versions, all_versions):
    by_group = {}
    groups = sorted({row[dimension] for row in snapshots})
    for group in groups:
        units = [row for row in snapshots if row[dimension] == group]
        by_release = {}
        for version in all_versions:
            assigned = {row["uid"] for row in units if row["version"] == version}
            mask = (versions == version) & np.isin(uids, list(assigned))
            selected = [row for row in units if row["version"] == version]
            values = quality(labels, scores, mask)
            by_release[version] = dict(user_release_snapshots=len(selected), distinct_users=len(assigned),
                scored_users=int(len(np.unique(uids[mask]))), candidate_requests=int(mask.sum()),
                positive_requests=int((labels[mask] == 1).sum()),
                behavior_events=sum(row["behavior_events"] for row in selected), **values)
        finite = [record for record in by_release.values() if record["auc"] is not None]
        # Never silently average only represented releases.
        if len(finite) == len(all_versions):
            means = {name: float(np.mean([record["auc"][name] for record in finite])) for name in scores}
            reference, gain = means["rebuild"] - means["reuse"], means["method"] - means["reuse"]
            overall = dict(equal_release_mean_auc=means, reference_gain_auc=reference,
                method_gain_auc=gain, method_minus_rebuild_auc=means["method"] - means["rebuild"],
                recovery_percent=100 * gain / reference if reference > 1e-12 else None,
                undefined_reason=None if reference > 1e-12 else "Equal-release reference gain nonpositive")
        else:
            overall = dict(recovery_percent=None, undefined_reason="One or more releases lack a defined AUC; per-release points only")
        births = Counter()
        old_rows = total_rows = 0
        for row in units:
            source = row["source_births_at_first_request"] or {}
            births.update({str(key): int(value) for key, value in source.items()})
            total_rows += sum(source.values())
            old_rows += sum(value for key, value in source.items() if int(key) < int(row["version"][1:]) - 1)
        by_group[group] = dict(user_release_snapshots=len(units), distinct_users=len({row["uid"] for row in units}),
            first_consumer_counts=dict(Counter(row["first_consumer"] for row in units)),
            source_birth_rows_at_first_request=dict(births),
            birth_rows_older_than_adjacent_version=old_rows, total_birth_rows_at_first_request=total_rows,
            by_release=by_release, overall=overall,
            statistical_scope="Descriptive point estimates only; no group confidence intervals or noninferiority claim")
    return by_group


def run(args):
    configuration = json.loads((args.results / "configuration.json").read_text())
    summary = json.loads((args.results / "summary.json").read_text())
    if summary["status"] != "complete" or len(summary["selected_uids"]) != 4096:
        raise ValueError("cold boundaries require the completed full4096 frozen result")
    panel = Path(configuration["panel"])
    binding, groups, _ = load_panel(panel)
    selected = summary["selected_uids"]
    if set(selected) != set(groups["evaluation"]):
        raise ValueError("grouping changed the complete frozen population")
    units = {int(row["uid"]): row for row in summary["units"]}
    columns = ["uid", "version", "query_timestamp", "label", "reuse_logit",
               "first_request_rebuild_logit", "partial_logit", "history_length"]
    table = pq.read_table(args.results / "scores.parquet", columns=columns)
    data = {name: table[name].to_numpy() for name in columns}
    first_request, first_request_length = {}, {}
    for uid, version, timestamp, length in zip(data["uid"], data["version"], data["query_timestamp"], data["history_length"], strict=True):
        key = int(uid), str(version)
        if key not in first_request or timestamp < first_request[key]:
            first_request[key], first_request_length[key] = int(timestamp), int(length)
    histories = load_history(panel, requested_uids=selected)
    cap = binding["model_config"]["max_seq_len"]
    snapshots = []
    for uid in selected:
        times = histories.rows[uid][0]
        requests = {version: first_request[uid, version] for version, _ in RELEASES if (uid, version) in first_request}
        temporal = group_windows(times, requests, cap=cap)
        for version, row in temporal.items():
            record = units[uid]["by_release"][version]
            actual_prepare = record["cost_counters"].get("compiled_prepare_calls", 0)
            if bool(actual_prepare) != (row["first_consumer_timestamp"] is not None):
                raise ValueError("time-derived consumption differs from the retained code-preparation counter")
            if (uid, version) in first_request:
                exact = min(cap, int(np.searchsorted(times, first_request[uid, version], side="left")))
                if exact != first_request_length[uid, version]:
                    raise ValueError("recorded first recommendation history length differs from causal timeline")
            snapshots.append(dict(uid=uid, version=version, **row,
                current_code_prepares=actual_prepare,
                source_births_at_publish=record["source_births_at_publish"],
                source_births_at_first_request=record["source_births_at_first_request"]))
    scores = dict(reuse=data["reuse_logit"], rebuild=data["first_request_rebuild_logit"], method=data["partial_logit"])
    all_versions = [version for version, _ in RELEASES]
    result = dict(status="complete_cpu_descriptive_analysis", users=len(selected),
        user_release_snapshots=len(snapshots), candidate_requests=table.num_rows,
        rules=dict(cold="Consecutive prior complete release windows with neither any behavior nor recommendation; reset after any history consumer",
            lengths="First history consumption, including behavior before the first recommendation; fixed 0,1–256,257–512,513–1024 buckets",
            boundaries="Chosen using time/length before inspecting grouped scores; all groups and idle windows retained"),
        cold=summarize_dimension("cold_group", snapshots, data["label"], scores, data["uid"], data["version"], all_versions),
        history_lengths=summarize_dimension("length_group", snapshots, data["label"], scores, data["uid"], data["version"], all_versions),
        snapshots=snapshots, scores_sha256=sha256(args.results / "scores.parquet"),
        panel_sha256=sha256(panel / "binding.json"), source_sha256=sha256(Path(__file__)),
        scope="Timing-defined descriptive boundaries on the existing frozen4096 result; group composition can vary by release; no new inference or confidence intervals",
        limitation="Native producer age is descriptive provenance, not the number of missed code preparations; undefined/nonpositive reference gain is not assigned a recovery percentage")
    output = args.results / "cold_boundaries.json"
    write_json(output, result)
    print(json.dumps(dict(status="complete", output=str(output),
        cold={group: dict(snapshots=record["user_release_snapshots"], users=record["distinct_users"], overall=record["overall"])
              for group, record in result["cold"].items()},
        history_lengths={group: dict(snapshots=record["user_release_snapshots"], overall=record["overall"])
                         for group, record in result["history_lengths"].items()})), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    run(p.parse_args())
