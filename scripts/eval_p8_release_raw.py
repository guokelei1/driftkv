#!/usr/bin/env python3
"""Write raw Previous/Recent/Full/Reuse/Base scores for one P8 edge."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import eval_p7_h_raw as p7eval
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import train_p7_theta0 as p7train

from hstu_kvcache.data.p7_training import P7Request, load_p7_requests
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/p8_release_v1"
RAW_LISTENS = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
BASE_ROOT = ROOT / "results/p7/base_fit/frozen_base_bundle_v1"
TRAIN_ROOT = ROOT / "results/p8/release_training"
OUTPUT_ROOT = ROOT / "results/p8/staleness_raw"
CONTRACT = ROOT / "configs/contracts/f_release_chain_contract_v1.yaml"
MODELS = {"m0_f": ("F",), "m1": ("N", "R", "F")}
RELEASE_EDGE = {
    "r0": ("edge1_evaluation", 231 * 86_400),
    "r1_edge1": ("edge1_evaluation", 231 * 86_400),
    "r1_edge2": ("edge2_evaluation", 245 * 86_400),
    "r2": ("edge1_evaluation", 231 * 86_400),
}
VIEWS = {
    "N": ("quality", "fidelity"),
    "R": ("quality_rankable", "fidelity_all_eligible"),
    "F": ("quality", "fidelity"),
}
BATCH = {"N": 4, "R": 2, "F": 16}
CHUNK = {"N": 25, "R": 16, "F": 1}


def load_model(path: Path, device: torch.device) -> tuple[HSTU, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = HSTU(HSTUConfig(**payload["config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload


def common_schema() -> pa.Schema:
    return pa.schema([
        ("request_id", pa.string()), ("uid", pa.int64()), ("query_timestamp", pa.int64()),
        ("workload", pa.string()), ("view", pa.string()), ("model_name", pa.string()),
        ("seed", pa.int32()), ("release", pa.string()), ("candidate_position", pa.int32()),
        ("candidate_id", pa.int64()), ("base_logit", pa.float32()),
        ("previous_full_logit", pa.float32()), ("current_recent32_logit", pa.float32()),
        ("current_full512_logit", pa.float32()), ("reuse_parent_kv_logit", pa.float32()),
        ("prefix_tokens_at_cutover", pa.int32()), ("suffix_tokens_after_cutover", pa.int32()),
    ])


def quality_schema() -> pa.Schema:
    return pa.schema(list(common_schema()) + [
        pa.field("target_index", pa.int32()), pa.field("label", pa.int8()),
        pa.field("is_target", pa.bool_()), pa.field("is_organic", pa.int8()),
        pa.field("prior_30m_same_item", pa.bool_()), pa.field("latest_item", pa.bool_()),
        pa.field("feedback_history_stratum_v2", pa.string()),
    ])


def cache_for_suffix(
    parent: HSTU,
    current: HSTU,
    tensors: dict[str, torch.Tensor],
    prefix_len: int,
    suffix_len: int,
) -> tuple[HSTUKVCache, float]:
    prefix_items = tensors["items"][:, :prefix_len]
    prefix_behaviors = tensors["behaviors"][:, :prefix_len]
    prefix_deltas = tensors["deltas"][:, :prefix_len]
    parent_cache = parent.compute_kv(prefix_items, prefix_behaviors, prefix_deltas)
    current_cache = current.compute_kv(prefix_items, prefix_behaviors, prefix_deltas)
    cache_delta = max(
        float((parent_cache.k - current_cache.k).abs().max()),
        float((parent_cache.v - current_cache.v).abs().max()),
    )
    if suffix_len:
        _, reused = current.forward_with_cache(
            parent_cache,
            tensors["items"][:, prefix_len : prefix_len + suffix_len],
            tensors["behaviors"][:, prefix_len : prefix_len + suffix_len],
            tensors["deltas"][:, prefix_len : prefix_len + suffix_len],
        )
        return reused, cache_delta
    return parent_cache, cache_delta


@torch.no_grad()
def score_reuse(
    parent: HSTU,
    current: HSTU,
    requests: list[P7Request],
    device: torch.device,
    cutover: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], float]:
    tensors = p7eval.collate(requests, device, history_tokens=512)
    pairs = []
    for row in requests:
        assert row.history_timestamps is not None
        timestamps = row.history_timestamps[-len(row.history_items) :]
        prefix = int(np.searchsorted(timestamps, cutover, side="left"))
        pairs.append((prefix, len(timestamps) - prefix))
    if len(set(pairs)) != 1:
        raise ValueError("reuse microbatch must have one prefix/suffix shape")
    prefix_len, suffix_len = pairs[0]
    if prefix_len < 1:
        raise ValueError("reuse request has no retained parent token")
    cache, cache_delta = cache_for_suffix(parent, current, tensors, prefix_len, suffix_len)
    chunks = []
    for start in range(0, tensors["candidates"].shape[1], CHUNK[requests[0].workload]):
        end = min(start + CHUNK[requests[0].workload], tensors["candidates"].shape[1])
        chunks.append(current.score_cc_reuse(
            cache, tensors["candidates"][:, start:end], tensors["query_deltas"],
            prefix_lengths=tensors["lengths"], query_type_ids=tensors["query_types"],
        ))
    return torch.cat(chunks, dim=1).float(), tensors, cache_delta


def shape_groups(requests: list[P7Request], cutover: int) -> list[list[P7Request]]:
    groups: dict[tuple[int, int], list[P7Request]] = defaultdict(list)
    for row in requests:
        assert row.history_timestamps is not None
        timestamps = row.history_timestamps[-len(row.history_items) :]
        prefix = int(np.searchsorted(timestamps, cutover, side="left"))
        if not prefix:
            continue
        key = (prefix, len(timestamps) - prefix)
        groups[key].append(row)
    return [values[start : start + BATCH[values[0].workload]] for values in groups.values() for start in range(0, len(values), BATCH[values[0].workload])]


def evaluate(model_name: str, seed: int, release: str, device: torch.device, output: Path) -> None:
    split, cutover = RELEASE_EDGE[release]
    checkpoint = TRAIN_ROOT / release / f"{model_name}_seed{seed}" / "selected.pt"
    current, child = load_model(checkpoint, device)
    if not child["admitted"]:
        raise RuntimeError("refusing primary staleness evaluation for rejected release")
    parent_path = ROOT / child["parent_checkpoint"]
    parent, _ = load_model(parent_path, device)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    cache_path_max_delta = 0.0
    for workload in MODELS[model_name]:
        base = p7eval.load_base(workload, device)
        for view in VIEWS[workload]:
            requests = load_p7_requests(MANIFEST, RAW_LISTENS, split, workload, manifest_kind=view)
            rows = []
            evaluated = 0
            for micro in shape_groups(requests, cutover):
                reuse, tensors, cache_delta = score_reuse(parent, current, micro, device, cutover)
                cache_path_max_delta = max(cache_path_max_delta, cache_delta)
                previous = p7eval.score_path(parent, tensors, device, workload=workload, chunk_size=CHUNK[workload])
                full = p7eval.score_path(current, tensors, device, workload=workload, chunk_size=CHUNK[workload])
                recent_tensors = p7eval.collate(micro, device, history_tokens=32)
                recent = p7eval.score_path(current, recent_tensors, device, workload=workload, chunk_size=CHUNK[workload])
                base_scores = base(tensors["features"].float()).float()
                for index, request in enumerate(micro):
                    prefix = int(np.searchsorted(request.history_timestamps, cutover, side="left"))
                    suffix = len(request.history_timestamps) - prefix
                    for position, candidate in enumerate(request.candidate_ids):
                        row = {
                            "request_id": request.request_id, "uid": request.uid,
                            "query_timestamp": request.query_timestamp, "workload": workload,
                            "view": view, "model_name": model_name, "seed": seed, "release": release,
                            "candidate_position": position, "candidate_id": int(candidate),
                            "base_logit": float(base_scores[index, position]),
                            "previous_full_logit": float(base_scores[index, position] + previous[index, position]),
                            "current_recent32_logit": float(base_scores[index, position] + recent[index, position]),
                            "current_full512_logit": float(base_scores[index, position] + full[index, position]),
                            "reuse_parent_kv_logit": float(base_scores[index, position] + reuse[index, position]),
                            "prefix_tokens_at_cutover": prefix, "suffix_tokens_after_cutover": suffix,
                        }
                        if "quality" in view:
                            row.update({
                                "target_index": request.target_index, "label": request.label,
                                "is_target": position == request.target_index if request.target_index is not None else None,
                                "is_organic": request.is_organic,
                                "prior_30m_same_item": request.prior_30m_same_item,
                                "latest_item": request.latest_item,
                                "feedback_history_stratum_v2": request.target_stratum,
                            })
                        rows.append(row)
                    evaluated += 1
            path = output / f"{workload}_{view}.parquet"
            schema = quality_schema() if "quality" in view else common_schema()
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
            artifacts.append({
                "workload": workload, "view": view, "path": str(path.relative_to(ROOT)),
                "sha256": p7train.sha256_file(path), "requests": evaluated, "candidate_rows": len(rows),
                "excluded_no_parent_token": len(requests) - evaluated, "schema": schema.names,
            })
    payload = {
        "status": "raw_paths_written_metrics_not_computed", "contract_hash": p7train.sha256_file(CONTRACT),
        "release": release, "model_name": model_name, "seed": seed,
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_hash": p7train.sha256_file(checkpoint),
        "parent_checkpoint": str(parent_path.relative_to(ROOT)), "parent_hash": p7train.sha256_file(parent_path),
        "cache_path_max_abs_delta": cache_path_max_delta, "artifacts": artifacts,
        "metrics_computed": False,
    }
    (output / "raw_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("status", "release", "model_name", "seed", "cache_path_max_abs_delta")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 37, 71), required=True)
    parser.add_argument("--release", choices=sorted(RELEASE_EDGE), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or OUTPUT_ROOT / args.release / f"{args.model}_seed{args.seed}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    evaluate(args.model, args.seed, args.release, torch.device(args.device), output)


if __name__ == "__main__":
    main()
