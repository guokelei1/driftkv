from __future__ import annotations

import gc
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ..models import HSTUKVCache
from .kuairand_projected_persistent import (
    _broadcast,
    _distributed,
    _initialize_model,
    _load_checkpoint,
    _manifest_path,
    _read_manifest,
    _seed,
    load_persistent_config,
)
from .kuairand_projected_scale import (
    _lookup,
    _projected_query_embedding,
)
from .kuairand_query_multiversion import _edge_config
from .kuairand_query_transition import (
    _atomic_json,
    build_workload,
    file_sha256,
    load_config,
)

PROTOCOL = "evokv_kuairand_large_hotkv_adjacent_timing_v0"
METHODS = ("reuse", "recompute")


def load_timing_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    baseline = document.get("baseline")
    timing = document.get("timing")
    profiles = document.get("profiles")
    execution = document.get("execution")
    outputs = document.get("outputs")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not all(
            isinstance(value, dict)
            for value in (baseline, timing, profiles, execution, outputs)
        )
        or timing.get("primary_scope") != "hot_hbm_hstu_core_only"
        or timing.get("methods") != list(METHODS)
        or timing.get("cache_residency") != "hbm_before_timing"
        or timing.get("input_residency") != "hbm_before_timing"
        or timing.get("cuda_synchronized") is not True
        or timing.get("rank_aggregation") != "elementwise_max"
        or timing.get("excluded")
        != [
            "data_loading",
            "model_loading",
            "checkpoint_loading",
            "source_cache_generation",
            "cpu_to_gpu_transfer",
            "embedding_lookup",
            "candidate_scoring",
            "metric_computation",
            "result_serialization",
        ]
        or int(execution.get("world_size", 0)) != 2
        or execution.get("devices") != [0, 1]
        or not isinstance(outputs.get("root"), str)
        or set(profiles) != {"canary", "full"}
    ):
        raise ValueError("KuaiRand hot-K/V timing config differs")
    for artifact_name in ("registry", "large_config"):
        artifact = baseline.get(artifact_name)
        artifact_path = (
            Path(artifact.get("path", "")) if isinstance(artifact, dict) else Path()
        )
        if (
            not isinstance(artifact, dict)
            or not artifact_path.is_file()
            or file_sha256(artifact_path) != artifact.get("sha256")
        ):
            raise ValueError("KuaiRand hot-K/V baseline binding differs")
    for artifact in document.get("implementation", []):
        artifact_path = Path(artifact.get("path", ""))
        if (
            not artifact_path.is_file()
            or file_sha256(artifact_path) != artifact.get("sha256")
        ):
            raise ValueError("KuaiRand hot-K/V implementation binding differs")
    for name, profile in profiles.items():
        versions = profile.get("target_versions")
        users = int(profile.get("users_per_edge", 0))
        if (
            not isinstance(versions, list)
            or versions != sorted(set(int(value) for value in versions))
            or not versions
            or min(versions) < 2
            or max(versions) > 8
            or users < 16
            or users % 16
            or int(profile.get("history_length", 0)) != 511
            or int(profile.get("local_batch_size", 0)) != 8
            or int(profile.get("warmup_iterations", -1)) < 1
            or int(profile.get("measured_repeats", 0)) < 1
            or not isinstance(profile.get("selection_seed"), int)
            or profile.get("name") != name
        ):
            raise ValueError("KuaiRand hot-K/V timing profile differs")
    return document


def _edge_document(
    baseline: dict[str, Any], base_config: dict[str, Any], target_version: int
) -> dict[str, Any]:
    transition = baseline["transitions"][target_version - 1]
    document = _edge_config(base_config, transition, 1.0)
    document["data"]["evaluation_targets_per_user"] = int(
        baseline["evaluation"]["targets_per_user"]
    )
    document["data"]["user_limit"] = baseline["data"].get("user_limit")
    document["evaluation"]["candidate_count"] = int(
        baseline["evaluation"]["candidate_count"]
    )
    return document


def _selection_digest(records: list[tuple[int, int]]) -> str:
    payload = json.dumps(records, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _selected_keys(
    workload: dict[str, Any],
    target_version: int,
    history_length: int,
    users_per_edge: int,
    seed: int,
) -> tuple[list[Any], dict[str, Any]]:
    by_user: dict[int, tuple[int, str, Any]] = {}
    for key in workload["evaluation_keys"]:
        record = workload["evaluation"][key]
        if len(record["history"]) != history_length:
            continue
        user = int(record["user_id"])
        ordinal = int(record["query_ordinal"])
        candidate = (ordinal, repr(key), key)
        current = by_user.get(user)
        if current is None or candidate[:2] < current[:2]:
            by_user[user] = candidate
    ranked = sorted(
        by_user,
        key=lambda user: hashlib.sha256(
            f"{seed}:{target_version}:{user}".encode()
        ).digest(),
    )
    if len(ranked) < users_per_edge:
        raise RuntimeError("KuaiRand hot-K/V eligible user count differs")
    selected_users = ranked[:users_per_edge]
    selected_keys = [by_user[user][2] for user in selected_users]
    records = [
        (user, int(workload["evaluation"][key]["query_ordinal"]))
        for user, key in zip(selected_users, selected_keys, strict=True)
    ]
    return selected_keys, {
        "eligible_unique_users": len(ranked),
        "selected_unique_users": len(selected_users),
        "selected_user_query_sha256": _selection_digest(records),
        "minimum_user_id": min(selected_users),
        "maximum_user_id": max(selected_users),
    }


def _make_local_batches(
    workload: dict[str, Any],
    selected_keys: list[Any],
    local_batch_size: int,
    rank: int,
    world_size: int,
) -> list[dict[str, Any]]:
    global_batch_size = local_batch_size * world_size
    if len(selected_keys) % global_batch_size:
        raise ValueError("KuaiRand hot-K/V selected batch boundary differs")
    batches = []
    for batch_index, start in enumerate(
        range(0, len(selected_keys), global_batch_size)
    ):
        chunk = selected_keys[start : start + global_batch_size]
        local_keys = chunk[
            rank * local_batch_size : (rank + 1) * local_batch_size
        ]
        items = torch.stack(
            [
                torch.as_tensor(
                    workload["evaluation"][key]["history"], dtype=torch.long
                )
                for key in local_keys
            ]
        )
        batches.append(
            {
                "batch_index": batch_index,
                "items": items,
                "users": [
                    int(workload["evaluation"][key]["user_id"])
                    for key in local_keys
                ],
            }
        )
    return batches


@torch.no_grad()
def _capture_source_caches(
    dense,
    embedding,
    batches: list[dict[str, Any]],
    workload: dict[str, Any],
    device: torch.device,
) -> list[HSTUKVCache]:
    dense.eval()
    embedding.eval()
    author_by_item = torch.as_tensor(
        np.asarray(workload["author_by_item"]).copy(),
        dtype=torch.long,
        device=device,
    )
    captured = []
    for batch in batches:
        items = batch["items"].to(device)
        prefix_items = items[:, :-1]
        prefix_lengths = torch.full(
            (len(items),), prefix_items.shape[1], dtype=torch.long, device=device
        )
        prefix_vectors = _lookup(
            embedding, prefix_items, prefix_lengths, author_by_item
        )
        cache = dense.core.compute_kv_from_item_embeddings(
            prefix_vectors,
            torch.ones_like(prefix_items),
            torch.zeros_like(prefix_items, dtype=torch.float32),
            prefix_lengths,
        )
        captured.append(
            HSTUKVCache(
                k=cache.k.detach().cpu(),
                v=cache.v.detach().cpu(),
                seq_len=cache.seq_len,
            )
        )
        del items, prefix_items, prefix_vectors, cache
        torch.cuda.empty_cache()
    del author_by_item
    return captured


@torch.no_grad()
def _timed_forward(
    dense,
    method: str,
    old_cache: HSTUKVCache,
    suffix: torch.Tensor,
    full_embedded: torch.Tensor,
    full_lengths: torch.Tensor,
) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if method == "reuse":
        hidden, cache = dense.core.forward_with_cache_embedded(old_cache, suffix)
    elif method == "recompute":
        hidden, cache = dense.core.forward_embedded(
            full_embedded,
            return_kv=True,
            lengths=full_lengths,
        )
    else:
        raise ValueError("KuaiRand hot-K/V timing method differs")
    end.record()
    end.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    if cache is None:
        raise RuntimeError("KuaiRand hot-K/V output cache is missing")
    checksum = float(
        hidden[:, -1].float().sum().item()
        + cache.k[:, :, -1, :1].float().sum().item()
    )
    del hidden, cache, start, end
    return elapsed_ms, checksum


def _sample_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": len(values),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p25_ms": float(np.percentile(array, 25)),
        "p75_ms": float(np.percentile(array, 75)),
        "p95_ms": float(np.percentile(array, 95)),
        "minimum_ms": float(array.min()),
        "maximum_ms": float(array.max()),
        "stdev_ms": float(array.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def _combine_rank_samples(
    gathered: list[dict[str, Any]],
    users: int,
    global_batch_size: int,
    batches: int,
    repeats: int,
) -> dict[str, Any]:
    methods = {}
    for method in METHODS:
        keys = sorted(gathered[0]["samples"][method])
        if len(keys) != batches * repeats or any(
            sorted(rank_payload["samples"][method]) != keys
            for rank_payload in gathered
        ):
            raise RuntimeError("KuaiRand hot-K/V rank sample keys differ")
        maximum_rank = [
            max(float(rank_payload["samples"][method][key]) for rank_payload in gathered)
            for key in keys
        ]
        pass_ms = [
            sum(
                max(
                    float(rank_payload["samples"][method][f"{batch}:{repeat}"])
                    for rank_payload in gathered
                )
                for batch in range(batches)
            )
            for repeat in range(repeats)
        ]
        pass_summary = _sample_summary(pass_ms)
        median_pass_ms = float(pass_summary["median_ms"])
        methods[method] = {
            "max_rank_batch_cuda_ms": _sample_summary(maximum_rank),
            "serial_full_pass_cuda_ms": pass_summary,
            "median_ms_per_user": median_pass_ms / users,
            "median_users_per_second": users * 1000.0 / median_pass_ms,
            "global_batch_size": global_batch_size,
        }
    reuse_ms = methods["reuse"]["serial_full_pass_cuda_ms"]["median_ms"]
    recompute_ms = methods["recompute"]["serial_full_pass_cuda_ms"]["median_ms"]
    return {
        "methods": methods,
        "comparison": {
            "recompute_over_reuse_ratio": recompute_ms / reuse_ms,
            "reuse_over_recompute_ratio": reuse_ms / recompute_ms,
            "absolute_median_pass_difference_ms": recompute_ms - reuse_ms,
            "reuse_time_saved_percent": 100.0 * (recompute_ms - reuse_ms) / recompute_ms,
        },
    }


@torch.no_grad()
def _benchmark_edge(
    dense,
    embedding,
    tracker,
    geometry: dict[str, Any],
    baseline: dict[str, Any],
    baseline_sha256: str,
    base_config: dict[str, Any],
    timing_config: dict[str, Any],
    timing_config_sha256: str,
    profile: dict[str, Any],
    target_version: int,
    output_path: Path,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, Any] | None:
    if rank == 0 and output_path.is_file():
        cached = json.loads(output_path.read_text())
        if (
            cached.get("status") != "complete"
            or cached.get("config", {}).get("sha256") != timing_config_sha256
            or cached.get("profile") != profile["name"]
            or cached.get("target_version") != target_version
            or cached.get("source_version") != target_version - 1
        ):
            raise RuntimeError("KuaiRand hot-K/V cached edge differs")
    else:
        cached = None
    cached = _broadcast(cached, rank)
    if cached is not None:
        return cached if rank == 0 else None

    edge_document = _edge_document(baseline, base_config, target_version)
    workload = build_workload(edge_document)
    selected_keys, selection = _selected_keys(
        workload,
        target_version,
        int(profile["history_length"]),
        int(profile["users_per_edge"]),
        int(profile["selection_seed"]),
    )
    batches = _make_local_batches(
        workload,
        selected_keys,
        int(profile["local_batch_size"]),
        rank,
        world_size,
    )
    checkpoint_root = Path(baseline["outputs"]["checkpoint_root"])
    source_version = target_version - 1
    source_manifest = _load_checkpoint(
        checkpoint_root,
        source_version,
        dense,
        embedding,
        tracker,
        baseline,
        baseline_sha256,
        rank,
    )
    source_caches = _capture_source_caches(
        dense, embedding, batches, workload, device
    )
    target_manifest = _load_checkpoint(
        checkpoint_root,
        target_version,
        dense,
        embedding,
        tracker,
        baseline,
        baseline_sha256,
        rank,
    )
    dense.eval()
    embedding.eval()
    torch.cuda.reset_peak_memory_stats(device)
    author_by_item = torch.as_tensor(
        np.asarray(workload["author_by_item"]).copy(),
        dtype=torch.long,
        device=device,
    )
    samples: dict[str, dict[str, float]] = {method: {} for method in METHODS}
    checksums: dict[str, list[float]] = {method: [] for method in METHODS}
    for batch, source_cache in zip(batches, source_caches, strict=True):
        batch_index = int(batch["batch_index"])
        items = batch["items"].to(device)
        lengths = torch.full(
            (len(items),), items.shape[1], dtype=torch.long, device=device
        )
        vectors = _lookup(embedding, items, lengths, author_by_item)
        history_embedded = dense.core.combine_input_features(
            vectors,
            torch.ones_like(items),
            torch.zeros_like(items, dtype=torch.float32),
        )
        query = _projected_query_embedding(
            dense, embedding, len(items), author_by_item, device
        )
        full_embedded = torch.cat((history_embedded, query), dim=1)
        full_lengths = lengths + 1
        suffix = torch.cat((history_embedded[:, -1:], query), dim=1)
        old_cache = HSTUKVCache(
            k=source_cache.k.to(device),
            v=source_cache.v.to(device),
            seq_len=source_cache.seq_len,
        )
        torch.cuda.synchronize(device)
        if batch_index == 0:
            for _ in range(int(profile["warmup_iterations"])):
                for method in METHODS:
                    _, checksum = _timed_forward(
                        dense,
                        method,
                        old_cache,
                        suffix,
                        full_embedded,
                        full_lengths,
                    )
                    checksums[method].append(checksum)
            torch.cuda.synchronize(device)
        for repeat in range(int(profile["measured_repeats"])):
            ordered_methods = (
                METHODS
                if (batch_index + repeat) % 2 == 0
                else tuple(reversed(METHODS))
            )
            for method in ordered_methods:
                elapsed_ms, checksum = _timed_forward(
                    dense,
                    method,
                    old_cache,
                    suffix,
                    full_embedded,
                    full_lengths,
                )
                samples[method][f"{batch_index}:{repeat}"] = elapsed_ms
                checksums[method].append(checksum)
        del (
            items,
            lengths,
            vectors,
            history_embedded,
            query,
            full_embedded,
            full_lengths,
            suffix,
            old_cache,
        )
        torch.cuda.empty_cache()
    if not all(
        values and all(np.isfinite(value) for value in values)
        for values in checksums.values()
    ):
        raise RuntimeError("KuaiRand hot-K/V output checksum differs")
    rank_payload = {
        "rank": rank,
        "samples": samples,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
    }
    if dist.is_initialized():
        gathered: list[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, rank_payload)
    else:
        gathered = [rank_payload]
    if rank == 0:
        combined = _combine_rank_samples(
            gathered,
            int(profile["users_per_edge"]),
            int(profile["local_batch_size"]) * world_size,
            len(batches),
            int(profile["measured_repeats"]),
        )
        result = {
            "protocol": PROTOCOL,
            "status": "complete",
            "scientific_result": False,
            "formal_result": False,
            "config": {
                "path": timing_config["config_path"],
                "sha256": timing_config_sha256,
            },
            "profile": profile["name"],
            "source_version": source_version,
            "target_version": target_version,
            "transition": baseline["transitions"][target_version - 1],
            "selection": selection,
            "request_shape": {
                "history_tokens": int(profile["history_length"]),
                "cached_prefix_tokens": int(profile["history_length"]) - 1,
                "reuse_suffix_tokens": 2,
                "recompute_input_tokens": int(profile["history_length"]) + 1,
                "users": int(profile["users_per_edge"]),
                "global_batch_size": int(profile["local_batch_size"]) * world_size,
                "batches_per_pass": len(batches),
            },
            "timing_scope": timing_config["timing"],
            "checkpoints": {
                "source_manifest": {
                    "path": str(_manifest_path(checkpoint_root, source_version)),
                    "sha256": file_sha256(
                        _manifest_path(checkpoint_root, source_version)
                    ),
                    "checkpoint_bytes": int(source_manifest["checkpoint_bytes"]),
                },
                "target_manifest": {
                    "path": str(_manifest_path(checkpoint_root, target_version)),
                    "sha256": file_sha256(
                        _manifest_path(checkpoint_root, target_version)
                    ),
                    "checkpoint_bytes": int(target_manifest["checkpoint_bytes"]),
                },
            },
            "geometry": geometry,
            "rank_runtime": gathered,
            **combined,
        }
        _atomic_json(output_path, result)
        print(
            f"phase=kuairand_hotkv_timing source={source_version} "
            f"target={target_version} reuse_ms="
            f"{result['methods']['reuse']['serial_full_pass_cuda_ms']['median_ms']:.3f} "
            f"recompute_ms="
            f"{result['methods']['recompute']['serial_full_pass_cuda_ms']['median_ms']:.3f} "
            f"ratio={result['comparison']['recompute_over_reuse_ratio']:.3f}",
            flush=True,
        )
    else:
        result = None
    del workload, selected_keys, batches, source_caches, author_by_item
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _valid_edge(
    path: Path,
    timing_config_sha256: str,
    profile: dict[str, Any],
    target_version: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    document = json.loads(path.read_text())
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "complete"
        or document.get("config", {}).get("sha256") != timing_config_sha256
        or document.get("profile") != profile["name"]
        or document.get("source_version") != target_version - 1
        or document.get("target_version") != target_version
        or document.get("request_shape", {}).get("users")
        != int(profile["users_per_edge"])
    ):
        raise ValueError("KuaiRand hot-K/V edge result differs")
    return document


def _summarize(
    config_path: Path,
    config_sha256: str,
    timing_config: dict[str, Any],
    profile: dict[str, Any],
    edges: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    reuse = [
        float(edge["methods"]["reuse"]["serial_full_pass_cuda_ms"]["median_ms"])
        for edge in edges
    ]
    recompute = [
        float(
            edge["methods"]["recompute"]["serial_full_pass_cuda_ms"]["median_ms"]
        )
        for edge in edges
    ]
    ratios = [
        float(edge["comparison"]["recompute_over_reuse_ratio"])
        for edge in edges
    ]
    users = int(profile["users_per_edge"])
    return {
        "protocol": PROTOCOL,
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": config_sha256},
        "profile": profile["name"],
        "timing_scope": timing_config["timing"],
        "edges": [
            {
                "source_version": edge["source_version"],
                "target_version": edge["target_version"],
                "path": edge["result_path"],
                "sha256": edge["result_sha256"],
                "reuse_median_pass_ms": edge["methods"]["reuse"][
                    "serial_full_pass_cuda_ms"
                ]["median_ms"],
                "recompute_median_pass_ms": edge["methods"]["recompute"][
                    "serial_full_pass_cuda_ms"
                ]["median_ms"],
                "recompute_over_reuse_ratio": edge["comparison"][
                    "recompute_over_reuse_ratio"
                ],
                "reuse_time_saved_percent": edge["comparison"][
                    "reuse_time_saved_percent"
                ],
            }
            for edge in edges
        ],
        "aggregate": {
            "edges": len(edges),
            "users_per_edge": users,
            "requests_across_edges": users * len(edges),
            "median_edge_reuse_pass_ms": statistics.median(reuse),
            "median_edge_recompute_pass_ms": statistics.median(recompute),
            "sum_edge_reuse_pass_ms": sum(reuse),
            "sum_edge_recompute_pass_ms": sum(recompute),
            "sum_recompute_over_reuse_ratio": sum(recompute) / sum(reuse),
            "mean_edge_recompute_over_reuse_ratio": statistics.fmean(ratios),
            "median_edge_recompute_over_reuse_ratio": statistics.median(ratios),
            "minimum_edge_recompute_over_reuse_ratio": min(ratios),
            "maximum_edge_recompute_over_reuse_ratio": max(ratios),
            "aggregate_reuse_time_saved_percent": 100.0
            * (sum(recompute) - sum(reuse))
            / sum(recompute),
        },
        "elapsed_seconds_including_untimed_setup": elapsed_seconds,
    }


def preflight_timing(
    config_path: str | Path, profile_name: str, build_selections: bool = True
) -> dict[str, Any]:
    path = Path(config_path)
    timing_config = load_timing_config(path)
    profile = timing_config["profiles"][profile_name]
    baseline_path = Path(timing_config["baseline"]["large_config"]["path"])
    baseline = load_persistent_config(baseline_path)
    baseline_sha256 = file_sha256(baseline_path)
    checkpoint_root = Path(baseline["outputs"]["checkpoint_root"])
    manifests = []
    for version in sorted(
        {
            value
            for target in profile["target_versions"]
            for value in (int(target) - 1, int(target))
        }
    ):
        manifest = _read_manifest(
            checkpoint_root,
            version,
            baseline,
            baseline_sha256,
            False,
        )
        manifests.append(
            {
                "version": version,
                "path": str(_manifest_path(checkpoint_root, version)),
                "sha256": file_sha256(_manifest_path(checkpoint_root, version)),
                "checkpoint_bytes": int(manifest["checkpoint_bytes"]),
            }
        )
    selections = []
    if build_selections:
        base_config = load_config(baseline["parent"]["base_config"]["path"])
        for target_version in profile["target_versions"]:
            edge_document = _edge_document(
                baseline, base_config, int(target_version)
            )
            workload = build_workload(edge_document)
            keys, selection = _selected_keys(
                workload,
                int(target_version),
                int(profile["history_length"]),
                int(profile["users_per_edge"]),
                int(profile["selection_seed"]),
            )
            selections.append(
                {
                    "target_version": int(target_version),
                    "evaluation_records": len(workload["evaluation_keys"]),
                    "selected_records": len(keys),
                    **selection,
                }
            )
            del workload, keys
            gc.collect()
    return {
        "protocol": PROTOCOL,
        "status": "ready",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(path), "sha256": file_sha256(path)},
        "profile": profile,
        "baseline_geometry": {
            "global_model_parameter_bytes": baseline["checkpoint"][
                "expected_global_parameter_bytes"
            ],
            "world_size": baseline["execution"]["world_size"],
        },
        "manifests": manifests,
        "selections": selections,
    }


def run_timing(config_path: str | Path, profile_name: str) -> dict[str, Any]:
    started = time.monotonic()
    path = Path(config_path)
    timing_config = load_timing_config(path)
    timing_config["config_path"] = str(path)
    timing_config_sha256 = file_sha256(path)
    profile = timing_config["profiles"][profile_name]
    output_root = Path(timing_config["outputs"]["root"]) / profile_name
    result_path = output_root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if (
            result.get("status") != "complete"
            or result.get("config", {}).get("sha256") != timing_config_sha256
            or result.get("profile") != profile_name
        ):
            raise ValueError("KuaiRand hot-K/V cached summary differs")
        return result
    edge_paths = {
        int(target): output_root
        / "edges"
        / f"theta_{int(target)}_from_theta_{int(target) - 1}.json"
        for target in profile["target_versions"]
    }
    cached_edges = [
        _valid_edge(edge_paths[int(target)], timing_config_sha256, profile, int(target))
        for target in profile["target_versions"]
    ]
    if all(edge is not None for edge in cached_edges):
        edges = []
        for target, edge in zip(profile["target_versions"], cached_edges, strict=True):
            assert edge is not None
            edge_path = edge_paths[int(target)]
            edges.append(
                edge
                | {
                    "result_path": str(edge_path),
                    "result_sha256": file_sha256(edge_path),
                }
            )
        result = _summarize(
            path,
            timing_config_sha256,
            timing_config,
            profile,
            edges,
            time.monotonic() - started,
        )
        _atomic_json(result_path, result)
        return result

    baseline_path = Path(timing_config["baseline"]["large_config"]["path"])
    baseline = load_persistent_config(baseline_path)
    baseline_sha256 = file_sha256(baseline_path)
    rank, world_size, device = _distributed(baseline)
    _seed(int(baseline["training"]["seed"]))
    base_config = load_config(baseline["parent"]["base_config"]["path"])
    first_manifest = json.loads(
        _manifest_path(Path(baseline["outputs"]["checkpoint_root"]), 1).read_text()
    )
    semantic_rows = int(first_manifest["geometry"]["semantic_embedding_rows"])
    dense, embedding, tracker, geometry = _initialize_model(
        baseline,
        base_config,
        semantic_rows,
        rank,
        world_size,
        device,
    )
    if int(geometry["global_model_parameter_bytes"]) != int(
        baseline["checkpoint"]["expected_global_parameter_bytes"]
    ):
        raise RuntimeError("KuaiRand hot-K/V model geometry differs")
    edges = []
    for target_version in profile["target_versions"]:
        target_version = int(target_version)
        edge_path = edge_paths[target_version]
        edge_path.parent.mkdir(parents=True, exist_ok=True)
        edge = _benchmark_edge(
            dense,
            embedding,
            tracker,
            geometry,
            baseline,
            baseline_sha256,
            base_config,
            timing_config,
            timing_config_sha256,
            profile,
            target_version,
            edge_path,
            rank,
            world_size,
            device,
        )
        if rank == 0:
            assert edge is not None
            edges.append(
                edge
                | {
                    "result_path": str(edge_path),
                    "result_sha256": file_sha256(edge_path),
                }
            )
    if rank == 0:
        result = _summarize(
            path,
            timing_config_sha256,
            timing_config,
            profile,
            edges,
            time.monotonic() - started,
        )
        _atomic_json(result_path, result)
    else:
        result = None
    if dist.is_initialized():
        dist.barrier()
        result = _broadcast(result, rank)
        dist.destroy_process_group()
    return result
