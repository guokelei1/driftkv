#!/usr/bin/env python3
"""Profile real contiguous-layer replay on reserved release-cutover histories.

This is label-free calibration: deterministic catalog queries are compared with
the current model's own Full logits. Evaluation users and feedback labels are
never used to choose intervals. Launch orchestration owns formal authorization.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from evaluate_yambda500m_foundation_raw import load_histories, load_model
from hstu_kvcache.baselines import layer_recompute as lr
from selective_recompute_2026_09.common import (
    OUTPUT, PLAN, edge_name, plan, sha256, signature, sources, write_json,
)
from selective_recompute_2026_09.cost import CostModel


def snapshot(history, uid: int, cutover: int, max_length: int):
    timestamps, items, behaviors = history.rows[uid]
    stop = int(np.searchsorted(timestamps, cutover, side="left"))
    values = sorted(zip(timestamps[:stop], items[:stop], behaviors[:stop], strict=True),
                    key=lambda row: (int(row[0]), int(row[1]), int(row[2])))[-max_length:]
    if not values:
        raise RuntimeError(f"calibration user {uid} has no strictly pre-release history")
    return tuple(np.asarray([row[column] for row in values]) for column in range(3))


@torch.inference_mode()
def profile_batch(parent, current, histories, uids, *, cutover, known_items, query_count, device):
    """One equal-length cohort; return independent interval squared errors."""
    raw = [histories[uid] for uid in uids]
    timestamps = torch.as_tensor(np.stack([row[0] for row in raw]), device=device)
    items = torch.as_tensor(np.stack([row[1] for row in raw]), dtype=torch.long, device=device)
    behaviors = torch.as_tensor(np.stack([row[2] for row in raw]), dtype=torch.long, device=device)
    deltas = torch.zeros_like(timestamps, dtype=torch.float32)
    deltas[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
    query_deltas = (cutover - timestamps[:, -1]).float()
    candidates = np.stack([
        np.random.default_rng(np.random.SeedSequence([17, int(uid)])).choice(
            known_items - 1, query_count, replace=False) + 1
        for uid in uids
    ])
    candidate_ids = torch.as_tensor(candidates, dtype=torch.long, device=device)
    state = lr.capture_state(parent, items, behaviors, deltas)
    teacher_cache = current.compute_kv(items, behaviors, deltas)
    teacher = current.score_cc_reuse(teacher_cache, candidate_ids, query_deltas)
    del teacher_cache
    errors = {}
    for interval in lr.enumerate_intervals(len(current.blocks), include_reuse=False):
        repaired = lr.recompute_interval(current, state, items, behaviors, deltas, interval)
        logits = current.score_cc_reuse(repaired.cache, candidate_ids, query_deltas)
        errors[interval] = float((logits.float() - teacher.float()).square().double().sum().item())
        del repaired, logits
    return errors


def run(args):
    config = plan()
    output_root = args.output_root or OUTPUT
    if args.limit_users is not None:
        if args.limit_users < 1:
            raise ValueError("--limit-users must be positive")
        if args.output_root is None or not output_root.resolve().is_relative_to((OUTPUT / "probes").resolve()):
            raise ValueError("limited calibration requires explicit --output-root under results/selective_recompute_2026_09/probes")
    panel = args.panel_root / args.scale / edge_name(args.edge)
    binding_path = panel / "binding.json"
    bound = json.loads(binding_path.read_text())
    users_record = bound["users_file"]
    users_path = ROOT / users_record["path"]
    requests_record = bound["files"]["calibration_requests"]
    request_path = ROOT / requests_record["path"]
    if sha256(users_path) != users_record["sha256"] or sha256(request_path) != requests_record["sha256"]:
        raise RuntimeError("frozen calibration panel changed")
    reserved = json.loads(users_path.read_text())["calibration"]
    request_uids = set(pq.read_table(request_path, columns=["uid"])["uid"].to_pylist())
    if set(reserved) != request_uids:
        raise RuntimeError("calibration users and their frozen request panel disagree")
    selected = reserved[:args.limit_users] if args.limit_users else reserved
    if len(reserved) != config["calibration_users_per_edge"] or not selected:
        raise RuntimeError("calibration reservation does not match the experiment plan")
    execution_sources = sources()
    source_signature = signature({
        "execution_sources": execution_sources,
        "panel_binding_sha256": sha256(binding_path),
        "uids": selected,
        "protocol": "cutover_snapshot_uniform_known_catalog_queries_seed17_mse_v1",
    })
    output = output_root / "layer" / args.scale / edge_name(args.edge) / "calibration.json"
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get("source_signature") != source_signature:
            raise RuntimeError(f"existing calibration has different sources or users: {output}")
        print(json.dumps({"status": "already_complete", "output": str(output)}), flush=True)
        return previous

    os.environ["EVOKV_ATTENTION_BACKEND"] = config["attention_backend"]
    torch.set_num_threads(config["torch_threads"])
    pa.set_cpu_count(config["history_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    free, total = torch.cuda.mem_get_info(device)
    if free < config["initial_free_fraction"] * total:
        raise RuntimeError(f"GPU {args.gpu} lacks the planned free-memory margin")
    torch.cuda.set_per_process_memory_fraction(config["memory_fraction"], device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    parent, parent_payload = load_model(ROOT / bound["sources"]["parent"]["path"], device)
    current, current_payload = load_model(ROOT / bound["sources"]["current"]["path"], device)
    if parent_payload["config"] != current_payload["config"]:
        raise RuntimeError("calibration checkpoints have different architectures")
    dataset_path = ROOT / bound["sources"]["dataset"]["path"]
    dataset = json.loads(dataset_path.read_text())
    known = int(current_payload.get("known_vocab_size", dataset["foundation_items"]))
    max_length = int(current_payload["config"]["max_seq_len"])
    if max_length != config["history_length"]:
        raise RuntimeError("checkpoint history length differs from the plan")
    oov = int(current_payload["config"]["num_items"]) - known
    model_config = current_payload["config"]
    del parent_payload, current_payload
    torch.cuda.synchronize(device)
    model_seconds = time.perf_counter() - start
    cutover = int(bound["cutover"])
    loaded = load_histories(
        selected, dataset_path=dataset_path, known_vocab_size=known, oov_buckets=oov,
        start_timestamp=cutover, end_timestamp=cutover + 1, max_history=max_length,
        threads=config["history_threads"],
    )
    histories = {uid: snapshot(loaded, uid, cutover, max_length) for uid in selected}
    del loaded
    history_seconds = time.perf_counter() - start - model_seconds
    by_length = defaultdict(list)
    for uid in selected:
        by_length[len(histories[uid][0])].append(uid)
    errors = defaultdict(float)
    batch_size = {"medium": 8, "large": 4, "max": 2}[args.scale]
    reductions = []
    completed = 0
    query_count = int(config["profiling_candidates_per_user"])
    profile_start = time.perf_counter()
    for length, group in sorted(by_length.items()):
        offset = 0
        while offset < len(group):
            uids = group[offset:offset + batch_size]
            try:
                values = profile_batch(parent, current, histories, uids, cutover=cutover,
                                       known_items=known, query_count=query_count, device=device)
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                reductions.append({"completed_users": completed, "from": batch_size,
                                   "to": max(1, batch_size // 2)})
                batch_size = max(1, batch_size // 2)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            for interval, error in values.items():
                errors[interval] += error
            offset += len(uids)
            completed += len(uids)
            print(json.dumps({"status": "calibrating", "scale": args.scale,
                              "edge": edge_name(args.edge), "users": completed,
                              "total_users": len(selected), "history_length": length,
                              "elapsed_seconds": time.perf_counter() - start}), flush=True)
    torch.cuda.synchronize(device)
    compute_seconds = time.perf_counter() - profile_start
    observations = len(selected) * query_count
    intervals = [{"interval": list(interval), "layers": interval[1] - interval[0] + 1,
                  "squared_error_sum": error, "logit_mse": error / observations}
                 for interval, error in sorted(errors.items())]
    selected_intervals = {}
    for count in range(1, len(current.blocks) + 1):
        winner = min((row for row in intervals if row["layers"] == count),
                     key=lambda row: (row["logit_mse"], *row["interval"]))
        selected_intervals[str(count)] = winner["interval"]
    histories_and_queries = [(len(histories[uid][0]), query_count) for uid in selected]
    result = {
        "status": "complete", "scale": args.scale, "edge": edge_name(args.edge),
        "probe_only": args.limit_users is not None, "users": len(selected), "uids": selected,
        "queries_per_user": query_count, "cutover": cutover,
        "selection": "minimum mean squared Current-Full logit error for each interval length; lexicographic ties",
        "candidates": "uniform without replacement from mapped known IDs [1, known_vocab_size), SeedSequence([17, uid])",
        "teacher": "Current Full at strictly pre-cutover history; no feedback labels",
        "history_tie_order": "timestamp_mapped_item_behavior", "first_history_delta": 0,
        "selected_intervals": selected_intervals, "intervals": intervals,
        "history_query_histogram": [
            {"history_length": n, "queries": q, "users": count}
            for (n, q), count in sorted(Counter(histories_and_queries).items())
        ],
        "cost": CostModel.for_scale(args.scale, config["attention_backend"]).layer_profile(histories_and_queries),
        "elapsed_seconds": time.perf_counter() - start,
        "model_load_seconds": model_seconds, "history_load_seconds": history_seconds,
        "compute_seconds": compute_seconds, "batch_size_final": batch_size,
        "batch_reductions": reductions,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1 << 30),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1 << 30),
        "source_signature": source_signature, "execution_source_hashes": execution_sources,
        "plan_sha256": sha256(PLAN), "panel_binding_sha256": sha256(binding_path),
        "calibration_requests_sha256": requests_record["sha256"],
        "model_config": model_config,
        "checkpoint_hashes": {name: bound["sources"][name]["sha256"] for name in ("parent", "current")},
    }
    if not all(np.isfinite(row["logit_mse"]) for row in intervals):
        raise RuntimeError("nonfinite calibration logits; no intervals admitted")
    write_json(output, result)
    print(json.dumps({"status": "complete", "output": str(output), "users": len(selected),
                      "compute_seconds": compute_seconds}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large", "max"), required=True)
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--gpu", type=int, choices=range(4), required=True)
    parser.add_argument("--panel-root", type=Path, default=OUTPUT / "panels")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--limit-users", type=int)
    run(parser.parse_args())
