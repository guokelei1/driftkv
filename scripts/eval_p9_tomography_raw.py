#!/usr/bin/env python3
"""P9.2 diagnostic layer/position recovery scan on sealed P8 F fidelity data.

An intervention copies exact current-model K/V into a parent-prefix cache, then
uses the normal current-model suffix/query path.  This is deliberately a
diagnostic splice, not a deployable migration action: downstream layer K/V can
still depend on stale lower-layer hidden states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

import eval_p7_h_raw as p7eval
import eval_p8_release_raw as p8raw
import train_p7_theta0 as p7
from hstu_kvcache.data.p7_training import P7Request, load_p7_requests
from hstu_kvcache.models import HSTUKVCache

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_tomography_contract_v1.yaml"
EVIDENCE = ROOT / "results/p9/p8_evidence_seal_v1.json"
MANIFEST = ROOT / "data/manifests/p8_release_v1"
RAW_LISTENS = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
P8_RAW = ROOT / "results/p8/staleness_raw"
OUTPUT_ROOT = ROOT / "results/p9/tomography_raw"
SEGMENTS = ("oldest_half", "middle", "recent_128", "recent_32", "recent_8", "recent_1")


def allowed_device(value: str) -> int:
    if not value.startswith("cuda:"):
        raise argparse.ArgumentTypeError("P9 tomography requires cuda:0 or cuda:1")
    try:
        index = int(value.split(":", 1)[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("device must be cuda:0 or cuda:1") from error
    if index not in (0, 1):
        raise argparse.ArgumentTypeError("GPU allowlist is cuda:0,cuda:1")
    return index


def action_names(layers: int) -> tuple[str, ...]:
    return tuple([*(f"layer_{layer}" for layer in range(layers)), *SEGMENTS])


def segment_slice(name: str, length: int) -> slice:
    if length < 1:
        raise ValueError("segment has no parent-prefix tokens")
    if name == "oldest_half":
        return slice(0, max(1, (length + 1) // 2))
    if name == "middle":
        return slice(length // 4, max(length // 4 + 1, (3 * length + 1) // 4))
    if name.startswith("recent_"):
        width = int(name.removeprefix("recent_"))
        return slice(max(0, length - width), length)
    raise ValueError(f"unknown history segment: {name}")


def diagnostic_cache(parent: HSTUKVCache, exact: HSTUKVCache, action: str) -> HSTUKVCache:
    """Copy a selected exact cache region into an otherwise stale cache."""
    if parent.k.shape != exact.k.shape or parent.v.shape != exact.v.shape:
        raise ValueError("parent and exact cache shapes differ")
    if parent.seq_len != exact.seq_len:
        raise ValueError("parent and exact cache lengths differ")
    k, v = parent.k.clone(), parent.v.clone()
    if action.startswith("layer_"):
        layer = int(action.removeprefix("layer_"))
        if not 0 <= layer < k.shape[0]:
            raise ValueError(f"layer action outside cache: {action}")
        k[layer].copy_(exact.k[layer])
        v[layer].copy_(exact.v[layer])
    else:
        selected = segment_slice(action, parent.seq_len)
        k[:, :, selected, :].copy_(exact.k[:, :, selected, :])
        v[:, :, selected, :].copy_(exact.v[:, :, selected, :])
    return HSTUKVCache(k=k, v=v, seq_len=parent.seq_len)


def selected_requests(requests: list[P7Request], cutover: int, limit: int | None) -> list[P7Request]:
    eligible = []
    for request in requests:
        assert request.history_timestamps is not None
        prefix = int(np.searchsorted(request.history_timestamps, cutover, side="left"))
        if prefix:
            eligible.append(request)
    if limit is None:
        return eligible
    # Hash-only selection is deterministic and has no label/target dependency.
    eligible.sort(key=lambda row: hashlib.sha256(row.request_id.encode()).digest())
    return eligible[:limit]


def p8_baselines(release: str, model: str, seed: int) -> dict[str, tuple[float, float]]:
    path = P8_RAW / release / f"{model}_seed{seed}" / "F_fidelity.parquet"
    table = pq.read_table(path, columns=["request_id", "current_full512_logit", "reuse_parent_kv_logit"])
    rows = table.to_pydict()
    return {
        str(request_id): (float(full), float(reuse))
        for request_id, full, reuse in zip(
            rows["request_id"], rows["current_full512_logit"], rows["reuse_parent_kv_logit"], strict=True
        )
    }


def append_suffix(current, cache: HSTUKVCache, tensors: dict[str, torch.Tensor], prefix: int, suffix: int) -> HSTUKVCache:
    if not suffix:
        return cache
    _, updated = current.forward_with_cache(
        cache,
        tensors["items"][:, prefix : prefix + suffix],
        tensors["behaviors"][:, prefix : prefix + suffix],
        tensors["deltas"][:, prefix : prefix + suffix],
    )
    return updated


@torch.no_grad()
def score_cache(current, cache: HSTUKVCache, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return current.score_cc_reuse(
        cache, tensors["candidates"], tensors["query_deltas"],
        prefix_lengths=tensors["lengths"], query_type_ids=tensors["query_types"],
    ).float()


def result_schema() -> pa.Schema:
    return pa.schema([
        ("request_id", pa.string()), ("uid", pa.int64()), ("query_timestamp", pa.int64()),
        ("release", pa.string()), ("model", pa.string()), ("seed", pa.int32()),
        ("action", pa.string()), ("action_kind", pa.string()),
        ("prefix_tokens_at_cutover", pa.int32()), ("suffix_tokens_after_cutover", pa.int32()),
        ("full_logit", pa.float32()), ("reuse_logit", pa.float32()), ("diagnostic_logit", pa.float32()),
    ])


def evaluate(release: str, model_name: str, seed: int, device: torch.device, limit: int | None, output: Path) -> dict:
    split, cutover = p8raw.RELEASE_EDGE[release]
    checkpoint = p8raw.TRAIN_ROOT / release / f"{model_name}_seed{seed}" / "selected.pt"
    current, child = p8raw.load_model(checkpoint, device)
    if not child["admitted"]:
        raise RuntimeError("P9 refuses non-admitted P8 edge")
    parent_path = ROOT / child["parent_checkpoint"]
    parent, _ = p8raw.load_model(parent_path, device)
    requests = load_p7_requests(MANIFEST, RAW_LISTENS, split, "F", manifest_kind="fidelity")
    selected = selected_requests(requests, cutover, limit)
    baseline = p8_baselines(release, model_name, seed)
    base = p7eval.load_base("F", device)
    rows = []
    max_full_delta = 0.0
    max_reuse_delta = 0.0
    start_time = time.monotonic()
    groups = p8raw.shape_groups(selected, cutover)
    for micro in groups:
        tensors = p7eval.collate(micro, device, history_tokens=512)
        pairs = []
        for request in micro:
            assert request.history_timestamps is not None
            prefix = int(np.searchsorted(request.history_timestamps, cutover, side="left"))
            pairs.append((prefix, len(request.history_timestamps) - prefix))
        if len(set(pairs)) != 1:
            raise RuntimeError("P9 grouped incompatible prefix/suffix shapes")
        prefix, suffix = pairs[0]
        parent_prefix = parent.compute_kv(tensors["items"][:, :prefix], tensors["behaviors"][:, :prefix], tensors["deltas"][:, :prefix])
        current_prefix = current.compute_kv(tensors["items"][:, :prefix], tensors["behaviors"][:, :prefix], tensors["deltas"][:, :prefix])
        reuse = score_cache(current, append_suffix(current, parent_prefix, tensors, prefix, suffix), tensors)
        exact_from_prefix = score_cache(current, append_suffix(current, current_prefix, tensors, prefix, suffix), tensors)
        full = p7eval.score_path(current, tensors, device, workload="F", chunk_size=1)
        base_scores = base(tensors["features"].float()).float()
        max_full_delta = max(max_full_delta, float((full - exact_from_prefix).abs().max()))
        for index, request in enumerate(micro):
            expected = baseline.get(request.request_id)
            if expected is None:
                raise RuntimeError(f"request missing from sealed P8 raw: {request.request_id}")
            max_full_delta = max(max_full_delta, abs(float(base_scores[index, 0] + full[index, 0]) - expected[0]))
            max_reuse_delta = max(max_reuse_delta, abs(float(base_scores[index, 0] + reuse[index, 0]) - expected[1]))
        for action in action_names(current_prefix.k.shape[0]):
            mixed = diagnostic_cache(parent_prefix, current_prefix, action)
            score = score_cache(current, append_suffix(current, mixed, tensors, prefix, suffix), tensors)
            for index, request in enumerate(micro):
                rows.append({
                    "request_id": request.request_id, "uid": request.uid, "query_timestamp": request.query_timestamp,
                    "release": release, "model": model_name, "seed": seed,
                    "action": action, "action_kind": "layer" if action.startswith("layer_") else "segment",
                    "prefix_tokens_at_cutover": prefix, "suffix_tokens_after_cutover": suffix,
                    "full_logit": float(base_scores[index, 0] + full[index, 0]),
                    "reuse_logit": float(base_scores[index, 0] + reuse[index, 0]),
                    "diagnostic_logit": float(base_scores[index, 0] + score[index, 0]),
                })
    payload = {
        "status": "P9_2_diagnostic_tomography_raw_written",
        "diagnostic_not_executable_action": True,
        "release": release, "model": model_name, "seed": seed, "requests": len(selected),
        "actions": list(action_names(current.cfg.num_layers)), "source_p8_raw_hash": p7.sha256_file(P8_RAW / release / f"{model_name}_seed{seed}" / "F_fidelity.parquet"),
        "checkpoint_hash": p7.sha256_file(checkpoint), "parent_checkpoint_hash": p7.sha256_file(parent_path),
        "max_full_baseline_abs_delta": max_full_delta, "max_reuse_baseline_abs_delta": max_reuse_delta,
        "elapsed_seconds": time.monotonic() - start_time,
    }
    if max_full_delta > 1e-5 or max_reuse_delta > 1e-5:
        raise RuntimeError(f"P9 baseline mismatch full={max_full_delta} reuse={max_reuse_delta}")
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "F_fidelity_tomography.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=result_schema()), raw_path, compression="zstd")
    payload["raw_path"] = str(raw_path.relative_to(ROOT))
    payload["raw_sha256"] = p7.sha256_file(raw_path)
    (output / "raw_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=("r0", "r1_edge1", "r1_edge2", "r2"), required=True)
    parser.add_argument("--model", choices=("m0_f", "m1"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 37, 71), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--request-limit", type=int, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    allowed_device(args.device)
    if args.request_limit is not None and args.request_limit < 1:
        raise ValueError("request limit must be positive")
    if not EVIDENCE.exists():
        raise FileNotFoundError("P9 evidence is not sealed")
    contract = yaml.safe_load(CONTRACT.read_text())
    evidence = json.loads(EVIDENCE.read_text())
    if evidence["contract_hash"] != p7.sha256_file(CONTRACT):
        raise RuntimeError("P9 contract changed after evidence seal")
    suffix = "full" if args.request_limit is None else f"canary{args.request_limit}"
    output = (args.output or OUTPUT_ROOT / suffix / args.release / f"{args.model}_seed{args.seed}").resolve()
    payload = evaluate(args.release, args.model, args.seed, torch.device(args.device), args.request_limit, output)
    print(json.dumps({key: payload[key] for key in ("status", "release", "model", "seed", "requests", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
