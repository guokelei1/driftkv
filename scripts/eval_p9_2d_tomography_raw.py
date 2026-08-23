#!/usr/bin/env python3
"""P9.3 diagnostic layer-by-position exact-KV splice.

This localizes interaction structure. It is not a dependency-closed migration
executor and must never be entered directly into a system cost frontier.
"""

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
import eval_p9_tomography_raw as coarse
import train_p7_theta0 as p7
from hstu_kvcache.data.p7_training import load_p7_requests
from hstu_kvcache.models import HSTUKVCache

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_3_2d_tomography_contract_v1.yaml"
OUTPUT_ROOT = ROOT / "results/p9/tomography_2d_raw"


def action_names_2d(layers: int) -> tuple[str, ...]:
    return tuple(f"layer_{layer}__{segment}" for layer in range(layers) for segment in coarse.SEGMENTS)


def parse_action(action: str) -> tuple[int, str]:
    layer_name, segment = action.split("__", 1)
    if not layer_name.startswith("layer_") or segment not in coarse.SEGMENTS:
        raise ValueError(f"invalid 2-D action: {action}")
    return int(layer_name.removeprefix("layer_")), segment


def diagnostic_cache_2d(parent: HSTUKVCache, exact: HSTUKVCache, action: str) -> HSTUKVCache:
    if parent.k.shape != exact.k.shape or parent.v.shape != exact.v.shape or parent.seq_len != exact.seq_len:
        raise ValueError("parent and exact cache shapes/lengths differ")
    layer, segment = parse_action(action)
    if not 0 <= layer < parent.k.shape[0]:
        raise ValueError(f"layer outside cache: {layer}")
    selected = coarse.segment_slice(segment, parent.seq_len)
    k, v = parent.k.clone(), parent.v.clone()
    k[layer, :, selected, :].copy_(exact.k[layer, :, selected, :])
    v[layer, :, selected, :].copy_(exact.v[layer, :, selected, :])
    return HSTUKVCache(k=k, v=v, seq_len=parent.seq_len)


def validate_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p8_evidence_seal": ROOT / "results/p9/p8_evidence_seal_v1.json",
        "p9_2_raw_seal": ROOT / "results/p9/p9_2_tomography_raw_seal_v1.json",
        "p9_2_quality_companions": ROOT / "results/p9/p9_2_quality_companions_v1.json",
        "p9_1_risk_concentration": ROOT / "results/p9/p9_1_risk_concentration_v1.json",
        "p9_2_closure_result": ROOT / "configs/contracts/p9_2_closure_result_v1.yaml",
    }
    for name, path in paths.items():
        if p7.sha256_file(path) != contract["input_hashes"][name]:
            raise RuntimeError(f"P9.3 input hash mismatch: {name}")
    return contract


def result_schema() -> pa.Schema:
    return coarse.result_schema()


def evaluate(release: str, model_name: str, seed: int, device: torch.device, limit: int | None, output: Path) -> dict:
    contract = validate_contract()
    allowed = {(row["release"], row["model"]) for row in contract["scope"]["semantic_cells"]}
    if (release, model_name) not in allowed:
        raise ValueError(f"cell is outside frozen P9.3 semantic scope: {(release, model_name)}")
    split, cutover = p8raw.RELEASE_EDGE[release]
    checkpoint = p8raw.TRAIN_ROOT / release / f"{model_name}_seed{seed}" / "selected.pt"
    current, child = p8raw.load_model(checkpoint, device)
    if not child["admitted"]:
        raise RuntimeError("P9.3 refuses non-admitted P8 edge")
    parent_path = ROOT / child["parent_checkpoint"]
    parent, _ = p8raw.load_model(parent_path, device)
    requests = load_p7_requests(coarse.MANIFEST, coarse.RAW_LISTENS, split, "F", manifest_kind="fidelity")
    selected = coarse.selected_requests(requests, cutover, limit)
    baseline = coarse.p8_baselines(release, model_name, seed)
    base = p7eval.load_base("F", device)
    actions = action_names_2d(current.cfg.num_layers)
    rows = []
    max_full_delta = 0.0
    max_reuse_delta = 0.0
    start_time = time.monotonic()
    for micro in p8raw.shape_groups(selected, cutover):
        tensors = p7eval.collate(micro, device, history_tokens=512)
        pairs = []
        for request in micro:
            assert request.history_timestamps is not None
            prefix = int(np.searchsorted(request.history_timestamps, cutover, side="left"))
            pairs.append((prefix, len(request.history_timestamps) - prefix))
        if len(set(pairs)) != 1:
            raise RuntimeError("P9.3 grouped incompatible prefix/suffix shapes")
        prefix, suffix = pairs[0]
        parent_prefix = parent.compute_kv(tensors["items"][:, :prefix], tensors["behaviors"][:, :prefix], tensors["deltas"][:, :prefix])
        current_prefix = current.compute_kv(tensors["items"][:, :prefix], tensors["behaviors"][:, :prefix], tensors["deltas"][:, :prefix])
        reuse = coarse.score_cache(current, coarse.append_suffix(current, parent_prefix, tensors, prefix, suffix), tensors)
        exact = coarse.score_cache(current, coarse.append_suffix(current, current_prefix, tensors, prefix, suffix), tensors)
        full = p7eval.score_path(current, tensors, device, workload="F", chunk_size=1)
        base_scores = base(tensors["features"].float()).float()
        max_full_delta = max(max_full_delta, float((full - exact).abs().max()))
        for index, request in enumerate(micro):
            expected = baseline.get(request.request_id)
            if expected is None:
                raise RuntimeError(f"request missing from sealed P8 raw: {request.request_id}")
            max_full_delta = max(max_full_delta, abs(float(base_scores[index, 0] + full[index, 0]) - expected[0]))
            max_reuse_delta = max(max_reuse_delta, abs(float(base_scores[index, 0] + reuse[index, 0]) - expected[1]))
        for action in actions:
            mixed = diagnostic_cache_2d(parent_prefix, current_prefix, action)
            score = coarse.score_cache(current, coarse.append_suffix(current, mixed, tensors, prefix, suffix), tensors)
            for index, request in enumerate(micro):
                rows.append({
                    "request_id": request.request_id, "uid": request.uid, "query_timestamp": request.query_timestamp,
                    "release": release, "model": model_name, "seed": seed,
                    "action": action, "action_kind": "layer_segment",
                    "prefix_tokens_at_cutover": prefix, "suffix_tokens_after_cutover": suffix,
                    "full_logit": float(base_scores[index, 0] + full[index, 0]),
                    "reuse_logit": float(base_scores[index, 0] + reuse[index, 0]),
                    "diagnostic_logit": float(base_scores[index, 0] + score[index, 0]),
                })
    if max_full_delta > 1e-5 or max_reuse_delta > 1e-5:
        raise RuntimeError(f"P9.3 baseline mismatch full={max_full_delta} reuse={max_reuse_delta}")
    payload = {
        "status": "P9_3_diagnostic_2d_tomography_raw_written",
        "diagnostic_not_executable_action": True,
        "release": release, "model": model_name, "seed": seed, "requests": len(selected),
        "actions": list(actions),
        "contract_hash": p7.sha256_file(CONTRACT),
        "source_p8_raw_hash": p7.sha256_file(coarse.P8_RAW / release / f"{model_name}_seed{seed}" / "F_fidelity.parquet"),
        "checkpoint_hash": p7.sha256_file(checkpoint), "parent_checkpoint_hash": p7.sha256_file(parent_path),
        "max_full_baseline_abs_delta": max_full_delta, "max_reuse_baseline_abs_delta": max_reuse_delta,
        "elapsed_seconds": time.monotonic() - start_time,
    }
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "F_fidelity_2d_tomography.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=result_schema()), raw_path, compression="zstd")
    payload["raw_path"] = str(raw_path.relative_to(ROOT))
    payload["raw_sha256"] = p7.sha256_file(raw_path)
    (output / "raw_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=("r0", "r1_edge1", "r2"), required=True)
    parser.add_argument("--model", choices=("m0_f", "m1"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 37, 71), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--request-limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    coarse.allowed_device(args.device)
    if args.request_limit is not None and args.request_limit < 1:
        raise ValueError("request limit must be positive")
    suffix = "full" if args.request_limit is None else f"canary{args.request_limit}"
    output = (args.output or OUTPUT_ROOT / suffix / args.release / f"{args.model}_seed{args.seed}").resolve()
    payload = evaluate(args.release, args.model, args.seed, torch.device(args.device), args.request_limit, output)
    print(json.dumps({key: payload[key] for key in ("status", "release", "model", "seed", "requests", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
