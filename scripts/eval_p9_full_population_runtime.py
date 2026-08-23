#!/usr/bin/env python3
"""Measure migration-only runtime over one frozen full cutover population."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

import analyze_p9_transition_costs as p96
import eval_p8_release_raw as p8raw
import eval_p9_cutover_profiler_raw as profiler
import eval_p9_materialized_lineage_canary as rolling
import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_10_full_population_runtime_contract_v1.yaml"
MANIFEST = ROOT / "data/manifests/p9_full_population_v1"
OUTPUT_ROOT = ROOT / "results/p9/full_population_runtime"


def validate() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p9_7_full_population_cost_sha256": ROOT / "results/p9/p9_7_full_population_costs_v1.json",
        "p9_8_adjudication_sha256": ROOT / "results/p9/p9_8_cutover_profiler_v1.json",
        "p9_8_opportunity_sha256": ROOT / "results/p9/p9_8_cutover_opportunity_v1.json",
        "p9_9_raw_seal_sha256": ROOT / "results/p9/p9_9_heldout_rolling_quality_raw_seal_v1.json",
        "p9_9_adjudication_sha256": ROOT / "results/p9/p9_9_heldout_rolling_quality_v1.json",
        "state_transition_source_sha256": ROOT / "src/hstu_kvcache/models/state_transition.py",
        "population_materialization_sha256": ROOT / "data/manifests/p9_full_population_v1/materialization_summary.json",
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P9.10 input hash mismatch: {key}")
    return contract


def condition(contract: dict, name: str) -> dict:
    matches = [row for row in contract["scope"]["benchmark_conditions"] if row["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"unknown benchmark condition {name}")
    return matches[0]


def transfer_proxy(cache, device, repetitions=5) -> dict:
    cpu_k = torch.empty(cache.k.shape, dtype=cache.k.dtype, device="cpu", pin_memory=True)
    cpu_v = torch.empty(cache.v.shape, dtype=cache.v.dtype, device="cpu", pin_memory=True)
    h2d_k = torch.empty_like(cache.k)
    h2d_v = torch.empty_like(cache.v)
    d2h, h2d = [], []
    byte_count = cache.k.numel() * cache.k.element_size() + cache.v.numel() * cache.v.element_size()
    for _ in range(repetitions):
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        cpu_k.copy_(cache.k, non_blocking=True)
        cpu_v.copy_(cache.v, non_blocking=True)
        torch.cuda.synchronize(device)
        d2h.append(time.perf_counter() - begin)
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        h2d_k.copy_(cpu_k, non_blocking=True)
        h2d_v.copy_(cpu_v, non_blocking=True)
        torch.cuda.synchronize(device)
        h2d.append(time.perf_counter() - begin)
    return {
        "bytes": byte_count,
        "D2H_seconds_points": d2h,
        "H2D_seconds_points": h2d,
        "D2H_GBps_median": byte_count / np.median(d2h) / 1e9,
        "H2D_GBps_median": byte_count / np.median(h2d) / 1e9,
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), required=True)
    parser.add_argument("--state-limit", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = validate()
    spec = condition(contract, args.condition)
    output = (args.output or OUTPUT_ROOT / args.condition).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    device = torch.device(args.device)
    repetitions = args.repetitions or int(contract["batching"]["full_population_repetitions"])
    states = pq.read_table(MANIFEST / spec["edge"] / "states.parquet").to_pylist()
    for row in states:
        row["cutover"] = 21168000 if spec["edge"] == "edge2" else 19958400
    if args.state_limit is not None:
        states.sort(key=lambda row: hashlib.sha256(str(row["uid"]).encode()).digest())
        states = states[: args.state_limit]
    states.sort(key=lambda row: int(row["raw_prefix_end_exclusive"]))
    checkpoint = p8raw.TRAIN_ROOT / spec["release"] / f"{spec['model']}_seed{spec['seed']}" / "selected.pt"
    current, child = p8raw.load_model(checkpoint, device)
    parent, _ = p8raw.load_model(ROOT / child["parent_checkpoint"], device)
    reader = profiler.RawStateReader()
    groups: dict[int, list[dict]] = {}
    for row in states:
        groups.setdefault(int(row["effective_prefix_length"]), []).append(row)
    actions = list(contract["scope"]["actions"])
    action_seconds = {action: [0.0] * repetitions for action in actions}
    action_batches = {action: 0 for action in actions}
    raw_reconstruction_seconds = raw_h2d_seconds = parent_build_seconds = 0.0
    batch_sizes = []
    transfer = None
    warmed = False
    logical = {action: {key: 0 for key in p96.logical_cost(1, action)} for action in actions}
    started = time.perf_counter()
    for length in sorted(groups):
        group = groups[length]
        for begin_index in range(0, len(group), int(contract["batching"]["batch_size"])):
            micro = group[begin_index : begin_index + int(contract["batching"]["batch_size"])]
            batch_sizes.append(len(micro))
            begin = time.perf_counter()
            items, behaviors, deltas, _ = profiler.state_tensors(reader, micro, torch.device("cpu"))
            raw_reconstruction_seconds += time.perf_counter() - begin
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            tensors = tuple(value.to(device) for value in (items, behaviors, deltas))
            torch.cuda.synchronize(device)
            raw_h2d_seconds += time.perf_counter() - begin
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            parent_cache = parent.compute_kv(*tensors)
            torch.cuda.synchronize(device)
            parent_build_seconds += time.perf_counter() - begin
            if transfer is None and length == 512 and len(micro) == int(contract["batching"]["batch_size"]):
                transfer = transfer_proxy(parent_cache, device)
            if not warmed:
                for action in actions:
                    value = rolling.migrate(action, current, parent_cache, tensors)
                    del value
                torch.cuda.synchronize(device)
                warmed = True
            for action in actions:
                per_state = p96.logical_cost(length, action)
                for key, value in per_state.items():
                    logical[action][key] += int(value) * len(micro)
                for repetition in range(repetitions):
                    torch.cuda.synchronize(device)
                    begin = time.perf_counter()
                    migrated = rolling.migrate(action, current, parent_cache, tensors)
                    torch.cuda.synchronize(device)
                    action_seconds[action][repetition] += time.perf_counter() - begin
                    del migrated
                action_batches[action] += 1
            del parent_cache, tensors
    if transfer is None:
        raise RuntimeError("P9.10 population lacks a full 512-token benchmark batch")
    action_results = []
    for action in actions:
        points = action_seconds[action]
        action_results.append({
            "action": action,
            "rollout_seconds_points": points,
            "rollout_seconds_median": float(np.median(points)),
            "states_per_second_median": float(len(states) / np.median(points)) if np.median(points) > 0 else None,
            "per_state_ms_median": float(1000 * np.median(points) / len(states)),
            "batch_measurements_per_repetition": action_batches[action],
            "logical_totals": logical[action],
        })
    payload = {
        "status": "P9_10_full_population_migration_runtime_measured",
        "condition": spec, "scope": "canary" if args.state_limit is not None else "full",
        "states": len(states), "unique_prefix_lengths": len(groups),
        "batch_size_distribution": {
            "batches": len(batch_sizes), "mean": float(np.mean(batch_sizes)),
            "p50": float(np.median(batch_sizes)), "p10": float(np.quantile(batch_sizes, 0.1)),
            "full_batch_fraction": float(np.mean(np.asarray(batch_sizes) == int(contract["batching"]["batch_size"]))),
        },
        "repetitions": repetitions, "actions": action_results,
        "raw_reconstruction_seconds": raw_reconstruction_seconds,
        "raw_history_H2D_seconds": raw_h2d_seconds,
        "parent_cache_reference_build_seconds_excluded": parent_build_seconds,
        "pinned_transfer_proxy": transfer,
        "total_benchmark_wall_seconds": time.perf_counter() - started,
        "contract_sha256": p7.sha256_file(CONTRACT),
        "evaluator_sha256": p7.sha256_file(Path(__file__)),
        "checkpoint_sha256": p7.sha256_file(checkpoint),
        "scheduler_authorized": False,
        "storage_KV_IO_measured": False,
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"], "condition": spec["name"], "scope": payload["scope"],
        "states": len(states), "wall_seconds": payload["total_benchmark_wall_seconds"],
        "actions": [{"action": row["action"], "per_state_ms": row["per_state_ms_median"]} for row in action_results],
    }, indent=2))


if __name__ == "__main__":
    main()
