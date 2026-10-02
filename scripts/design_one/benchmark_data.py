#!/usr/bin/env python3
"""Prepare the frozen Medium V4->V5 Design 1 requests from retained scores."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from selective_recompute_2026_09.prepare import paired_population, metrics, sorted_requests
from unified_reuse_2026_09.common import sha256, write_json

CONFIG = ROOT / "configs/design/medium_v0_v5_development.json"
PANEL = ROOT / "results/design_one_2026_10/benchmark_panel"
EDGE = "v4_to_v5"


def checked(source):
    path = ROOT / source["path"]
    if sha256(path) != source["sha256"]:
        raise RuntimeError(f"frozen input changed: {path}")
    return path


def policy_inputs(config, reference, evaluation_uids):
    """Reuse the selected historical fits with their original panel binding."""
    entries = []
    for tag, method, count in (("Q", "query_only", 256), ("H", "history_conditioned", 128)):
        ref = next(row for row in config["stages"]["design1"]["reference_results"]
                   if row["method"] == method and row["calibration_users"] == count)
        summary_path = checked(ref["summary"])
        old = json.loads(summary_path.read_text())["inputs"]
        if tag == "Q":
            policy = old["policies"]["query_pure_cross_phi_joint8_c256"]
            binding, variant = policy["calibration"], policy["variant"]
        else:
            binding, variant = old["calibration"], "map_all"
        record_path = checked(binding)
        record = json.loads(record_path.read_text())
        weights = record_path.with_suffix(".pt")
        if record["status"] != "complete" or sha256(weights) != binding["weights_sha256"]:
            raise RuntimeError(f"historical calibration weights changed: {tag}")
        if (record["weights_sha256"] != binding["weights_sha256"]
                or record["panel_binding_sha256"] != reference["sha256"]):
            raise RuntimeError(f"historical calibration binding changed: {tag}")
        fitting = record["uids"] + record["validation_uids"]
        if (len(record["uids"]) != count or len(fitting) != len(set(fitting))
                or set(fitting).intersection(evaluation_uids)):
            raise RuntimeError(f"calibration/validation/evaluation overlap: {tag}")
        entries.append({"tag": tag, "method": method, "variant": variant,
            "budget": count, "calibration_dir": str(record_path.parent),
            "calibration": binding, "historical_summary": ref["summary"],
            "original_panel_binding": reference,
            "calibration_flops": record["cost"]["calibration_flops"]})
    return entries


def select_uids(groups, max_users=0, shard_index=0, num_shards=1):
    """Canaries are prefixes of the already fixed 128-user pilot."""
    if max_users < 0 or max_users > len(groups["design1_pilot"]):
        raise ValueError("--max-users must be 0 (complete panel) or 1..128 (fixed pilot)")
    if not 0 <= shard_index < num_shards:
        raise ValueError("invalid UID shard")
    chosen = groups["design1_pilot"][:max_users] if max_users else groups["design1_evaluation"]
    return sorted(chosen)[shard_index::num_shards]


def load_panel(panel_dir=PANEL):
    binding_path = panel_dir / "binding.json"
    binding = json.loads(binding_path.read_text())
    checked(binding["configuration"])
    groups = json.loads(checked(binding["users_file"]).read_text())
    table = pq.read_table(checked(binding["files"]["evaluation_requests"]))
    by_user = defaultdict(list)
    for row in table.to_pylist():
        by_user[int(row["uid"])].append(row)
    if set(by_user) != set(groups["design1_evaluation"]):
        raise RuntimeError("prepared benchmark differs from frozen evaluation UIDs")
    return binding, groups, by_user


def check_history(history, by_user, uids, cutover, max_length=1024):
    """Check the exact causal prefix and append/eviction counts before scoring."""
    requests, same_timestamp_events = 0, 0
    for uid in uids:
        times = history.rows[uid][0]
        if len(times) < 1 or np.any(times[1:] < times[:-1]):
            raise RuntimeError(f"empty or unordered causal history: {uid}")
        before = int(np.searchsorted(times, cutover, side="left"))
        if not 0 < before <= max_length:
            raise RuntimeError(f"invalid release prefix: {uid}")
        for row in by_user[uid]:
            timestamp = int(row["query_timestamp"])
            stop = int(np.searchsorted(times, timestamp, side="left"))
            appends = stop - before
            expected = {"history_length": min(stop, max_length),
                "cache_length": min(stop, max_length),
                "append_count_since_cutover": appends,
                "rolling_evictions": max(0, stop - max_length)}
            if timestamp < cutover or appends < 0 or any(int(row[k]) != v for k, v in expected.items()):
                raise RuntimeError(f"causal history/count mismatch: {row['request_id']}")
            if stop and int(times[stop - 1]) >= timestamp:
                raise RuntimeError("query contains a contemporaneous/future history event")
            same_timestamp_events += int(np.searchsorted(times, timestamp, side="right")) - stop
            requests += 1
    return {"users": len(uids), "requests": requests,
        "same_timestamp_events_excluded": same_timestamp_events,
        "rule": "history timestamps strictly before query; queries precede all same-timestamp appends"}


def prepare(config_path, output, check_history_users=0):
    config = json.loads(config_path.read_text())
    stage = config["stages"]["design1"]
    users_path = checked(config["users"])
    groups = json.loads(users_path.read_text())
    evaluation = set(groups[stage["evaluation_group"]])
    if len(evaluation) != stage["full_development_users"] or not set(groups["design1_pilot"]).issubset(evaluation):
        raise RuntimeError("fixed evaluation/pilot population differs")
    if evaluation.intersection(groups["design1_fit"] + groups["design1_validation"]):
        raise RuntimeError("Design 1 fitting users overlap evaluation")
    original = next(row["binding"] for row in config["users"]["evaluation_sources"] if row["edge"] == EDGE)
    original_panel = json.loads(checked(original).read_text())
    population, reuse, sources = paired_population("medium", 5)
    for key, source in original_panel["sources"].items():
        if key in sources and source != sources[key]:
            raise RuntimeError(f"population source differs from retained reference: {key}")
    window = next(row for row in config["profile"]["continuous_windows"] if row["edge"] == EDGE)
    for source in window["request_sources"]:
        checked(source)
    cutover = int(original_panel["cutover"])
    if list(reuse["days"]) != stage["days_half_open"] or cutover != window["cutover_seconds"]:
        raise RuntimeError("release window differs from fixed design")
    table = sorted_requests(population.filter(pc.is_in(population["uid"], value_set=pa.array(sorted(evaluation)))))
    if len(table) != stage["requests"] or set(table["uid"].to_pylist()) != evaluation:
        raise RuntimeError("missing or extra Design 1 users/requests")
    for key in ("full_logit", "reuse_logit"):
        if not np.isfinite(table[key].to_numpy()).all():
            raise RuntimeError(f"nonfinite retained {key}")
    times = table["query_timestamp"].to_numpy()
    if times.min() < cutover or times.max() >= stage["days_half_open"][1] * 86400:
        raise RuntimeError("request lies outside the frozen release window")
    old = sorted_requests(pq.read_table(checked(original_panel["files"]["evaluation_requests"])))
    subset = sorted_requests(table.filter(pc.is_in(table["uid"], value_set=pa.array(groups["motivation_reference"]))))
    if not subset.equals(old):
        raise RuntimeError("retained 3000-user reference changed during expansion")
    policies = policy_inputs(config, original, evaluation)
    output.mkdir(parents=True, exist_ok=True)
    requests_path = output / "evaluation_requests.parquet"
    if (output / "binding.json").exists():
        raise FileExistsError(f"prepared panel exists; choose a fresh output: {output}")
    pq.write_table(table, requests_path, compression="zstd")
    binding = {"status": "prepared", "scale": "medium", "edge": EDGE,
        "evaluation_role": "development_exploration", "days": reuse["days"], "cutover": cutover,
        "configuration": {"path": str(config_path.resolve()), "sha256": sha256(config_path)},
        "users_file": {"path": str(users_path), "sha256": sha256(users_path)},
        "sources": sources, "original_reference_panel": original, "policies": policies,
        "files": {"evaluation_requests": {"path": str(requests_path.resolve()),
            "sha256": sha256(requests_path), "rows": len(table)}},
        "metrics": metrics(table), "reference_rows_verified": len(old),
        "sharding": "sorted selected UIDs[shard_index::num_shards]; every selected user's requests retained",
        "causality": "Retained causal Full/Reuse source seals verified; actual event-prefix counts checked by each evaluator before scoring."}
    if check_history_users:
        from evaluate_yambda500m_foundation_raw import load_histories
        chosen = select_uids(groups, check_history_users)
        dataset_path = checked(config["data"]["dataset"])
        dataset = json.loads(dataset_path.read_text())
        history = load_histories(chosen, dataset_path=dataset_path,
            known_vocab_size=dataset["foundation_items"], start_timestamp=cutover,
            end_timestamp=reuse["days"][1] * 86400, max_history=1024, threads=4)
        by_user = defaultdict(list)
        for row in table.to_pylist():
            by_user[row["uid"]].append(row)
        binding["cpu_causal_check"] = check_history(history, by_user, chosen, cutover)
    write_json(output / "binding.json", binding)
    print(json.dumps({"output": str(output), **binding["metrics"]}), flush=True)
    return binding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=PANEL)
    parser.add_argument("--check-history-users", type=int, default=0)
    args = parser.parse_args()
    prepare(args.config, args.output.resolve(), args.check_history_users)
