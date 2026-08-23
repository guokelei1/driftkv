#!/usr/bin/env python3
"""Evaluate dependency-closed P9.4 state-transition actions on sealed P8 F cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

import eval_p7_h_raw as p7eval
import eval_p8_release_raw as p8raw
import eval_p9_tomography_raw as common
import train_p7_theta0 as p7
from hstu_kvcache.data.p7_training import load_p7_requests
from hstu_kvcache.models import (
    HSTUKVCache,
    hybrid_tail_refresh,
    project_exact_layer0_segment,
    transition_work,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_4_executor_contract_v1.yaml"
OUTPUT_ROOT = ROOT / "results/p9/executor_raw"
ACTIONS = (
    "noop", "layer0_recent128", "layer0_middle", "layer0_full",
    "hybrid_tail32", "hybrid_tail128", "exact_all",
)


def validate_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p9_3_result": ROOT / "configs/contracts/p9_3_2d_tomography_result_v1.yaml",
        "p9_3_raw_seal": ROOT / "results/p9/p9_3_2d_tomography_raw_seal_v1.json",
        "p9_3_target_free": ROOT / "results/p9/p9_3_2d_tomography_v1.json",
        "p9_3_quality": ROOT / "results/p9/p9_3_2d_quality_companions_v1.json",
    }
    for name, path in paths.items():
        if p7.sha256_file(path) != contract["input_hashes"][name]:
            raise RuntimeError(f"P9.4 input hash mismatch: {name}")
    if tuple(row["name"] for row in contract["actions"]) != ACTIONS:
        raise RuntimeError("P9.4 action order changed")
    return contract


def schema() -> pa.Schema:
    return pa.schema([
        ("request_id", pa.string()), ("uid", pa.int64()), ("query_timestamp", pa.int64()),
        ("release", pa.string()), ("model", pa.string()), ("seed", pa.int32()),
        ("action", pa.string()),
        ("prefix_tokens_at_cutover", pa.int32()), ("suffix_tokens_after_cutover", pa.int32()),
        ("full_logit", pa.float32()), ("reuse_logit", pa.float32()), ("action_logit", pa.float32()),
        ("projection_tokens", pa.int32()), ("recomputed_token_layers", pa.int32()),
        ("attention_pair_work", pa.int64()), ("old_kv_read_bytes", pa.int64()),
        ("new_kv_write_bytes", pa.int64()), ("raw_history_read_bytes", pa.int64()),
        ("prototype_transition_runtime_ms", pa.float64()),
    ])


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def transition(action: str, current, parent_cache: HSTUKVCache, tensors: dict[str, torch.Tensor], prefix: int) -> HSTUKVCache:
    items = tensors["items"][:, :prefix]
    behaviors = tensors["behaviors"][:, :prefix]
    deltas = tensors["deltas"][:, :prefix]
    if action == "noop":
        return parent_cache
    if action.startswith("layer0_"):
        segment = action.removeprefix("layer0_")
        segment = "recent_128" if segment == "recent128" else segment
        return project_exact_layer0_segment(
            current, parent_cache, items, behaviors, deltas, segment
        )
    if action.startswith("hybrid_tail"):
        return hybrid_tail_refresh(
            current, parent_cache, items, behaviors, deltas, int(action.removeprefix("hybrid_tail"))
        )
    if action == "exact_all":
        return current.compute_kv(items, behaviors, deltas)
    raise ValueError(f"unknown P9.4 action: {action}")


def evaluate(release: str, model_name: str, seed: int, device: torch.device, limit: int | None, output: Path) -> dict:
    validate_contract()
    split, cutover = p8raw.RELEASE_EDGE[release]
    checkpoint = p8raw.TRAIN_ROOT / release / f"{model_name}_seed{seed}" / "selected.pt"
    current, child = p8raw.load_model(checkpoint, device)
    if not child["admitted"]:
        raise RuntimeError("P9.4 refuses non-admitted P8 edge")
    parent_path = ROOT / child["parent_checkpoint"]
    parent, _ = p8raw.load_model(parent_path, device)
    requests = load_p7_requests(common.MANIFEST, common.RAW_LISTENS, split, "F", manifest_kind="fidelity")
    selected = common.selected_requests(requests, cutover, limit)
    baseline = common.p8_baselines(release, model_name, seed)
    base = p7eval.load_base("F", device)
    rows = []
    max_full_delta = 0.0
    max_reuse_delta = 0.0
    max_exact_delta = 0.0
    max_noop_delta = 0.0
    max_layer0_region_delta = 0.0
    runtime_by_action = {action: 0.0 for action in ACTIONS}
    start_time = time.monotonic()
    for micro in p8raw.shape_groups(selected, cutover):
        tensors = p7eval.collate(micro, device, history_tokens=512)
        pairs = []
        for request in micro:
            assert request.history_timestamps is not None
            prefix = int(np.searchsorted(request.history_timestamps, cutover, side="left"))
            pairs.append((prefix, len(request.history_timestamps) - prefix))
        if len(set(pairs)) != 1:
            raise RuntimeError("P9.4 grouped incompatible prefix/suffix shapes")
        prefix, suffix = pairs[0]
        items = tensors["items"][:, :prefix]
        behaviors = tensors["behaviors"][:, :prefix]
        deltas = tensors["deltas"][:, :prefix]
        parent_prefix = parent.compute_kv(items, behaviors, deltas)
        current_prefix = current.compute_kv(items, behaviors, deltas)
        reuse_score = common.score_cache(current, common.append_suffix(current, parent_prefix, tensors, prefix, suffix), tensors)
        full = p7eval.score_path(current, tensors, device, workload="F", chunk_size=1)
        base_scores = base(tensors["features"].float()).float()
        for index, request in enumerate(micro):
            expected = baseline[request.request_id]
            max_full_delta = max(max_full_delta, abs(float(base_scores[index, 0] + full[index, 0]) - expected[0]))
            max_reuse_delta = max(max_reuse_delta, abs(float(base_scores[index, 0] + reuse_score[index, 0]) - expected[1]))
        for action in ACTIONS:
            synchronize(device)
            action_start = time.perf_counter()
            migrated = transition(action, current, parent_prefix, tensors, prefix)
            synchronize(device)
            elapsed_ms = 1000.0 * (time.perf_counter() - action_start)
            runtime_by_action[action] += elapsed_ms
            if action.startswith("layer0_"):
                selected_region = (
                    slice(max(0, prefix - 128), prefix) if action == "layer0_recent128"
                    else slice(prefix // 4, max(prefix // 4 + 1, (3 * prefix + 1) // 4)) if action == "layer0_middle"
                    else slice(0, prefix)
                )
                max_layer0_region_delta = max(
                    max_layer0_region_delta,
                    float((migrated.k[0, :, selected_region] - current_prefix.k[0, :, selected_region]).abs().max()),
                    float((migrated.v[0, :, selected_region] - current_prefix.v[0, :, selected_region]).abs().max()),
                )
            state = common.append_suffix(current, migrated, tensors, prefix, suffix)
            action_score = common.score_cache(current, state, tensors)
            if action == "noop":
                max_noop_delta = max(max_noop_delta, float((action_score - reuse_score).abs().max()))
            if action == "exact_all":
                max_exact_delta = max(max_exact_delta, float((action_score - full).abs().max()))
            work = transition_work(action, parent_prefix, items, behaviors, deltas)
            per_request_runtime = elapsed_ms / len(micro)
            for index, request in enumerate(micro):
                rows.append({
                    "request_id": request.request_id, "uid": request.uid, "query_timestamp": request.query_timestamp,
                    "release": release, "model": model_name, "seed": seed, "action": action,
                    "prefix_tokens_at_cutover": prefix, "suffix_tokens_after_cutover": suffix,
                    "full_logit": float(base_scores[index, 0] + full[index, 0]),
                    "reuse_logit": float(base_scores[index, 0] + reuse_score[index, 0]),
                    "action_logit": float(base_scores[index, 0] + action_score[index, 0]),
                    "projection_tokens": work.projection_tokens,
                    "recomputed_token_layers": work.recomputed_token_layers,
                    "attention_pair_work": work.attention_pair_work,
                    "old_kv_read_bytes": work.old_kv_read_bytes // len(micro),
                    "new_kv_write_bytes": work.new_kv_write_bytes // len(micro),
                    "raw_history_read_bytes": work.raw_history_read_bytes // len(micro),
                    "prototype_transition_runtime_ms": per_request_runtime,
                })
    invariants = {
        "max_full_baseline_abs_delta": max_full_delta,
        "max_reuse_baseline_abs_delta": max_reuse_delta,
        "max_exact_all_residual_delta": max_exact_delta,
        "max_noop_reuse_delta": max_noop_delta,
        "max_layer0_projected_region_KV_delta": max_layer0_region_delta,
    }
    if max(invariants.values()) > 1e-5:
        raise RuntimeError(f"P9.4 executor invariant failed: {invariants}")
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "F_fidelity_executor.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema()), raw_path, compression="zstd")
    payload = {
        "status": "P9_4_dependency_closed_executor_raw_written",
        "release": release, "model": model_name, "seed": seed,
        "requests": len(selected), "actions": list(ACTIONS),
        "contract_hash": p7.sha256_file(CONTRACT),
        "checkpoint_hash": p7.sha256_file(checkpoint), "parent_checkpoint_hash": p7.sha256_file(parent_path),
        "source_p8_raw_hash": p7.sha256_file(common.P8_RAW / release / f"{model_name}_seed{seed}" / "F_fidelity.parquet"),
        "invariants": invariants, "prototype_runtime_ms_by_action": runtime_by_action,
        "elapsed_seconds": time.monotonic() - start_time,
        "raw_path": str(raw_path.relative_to(ROOT)), "raw_sha256": p7.sha256_file(raw_path),
        "runtime_caveat": "prototype runtime includes PyTorch cache materialization; logical byte counters are reported separately",
    }
    (output / "raw_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=("r0", "r1_edge1", "r1_edge2", "r2"), required=True)
    parser.add_argument("--model", choices=("m0_f", "m1"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 37, 71), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--request-limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    common.allowed_device(args.device)
    suffix = "full" if args.request_limit is None else f"canary{args.request_limit}"
    output = (args.output or OUTPUT_ROOT / suffix / args.release / f"{args.model}_seed{args.seed}").resolve()
    payload = evaluate(args.release, args.model, args.seed, torch.device(args.device), args.request_limit, output)
    print(json.dumps({key: payload[key] for key in ("status", "release", "model", "seed", "requests", "elapsed_seconds", "invariants")}, indent=2))


if __name__ == "__main__":
    main()
