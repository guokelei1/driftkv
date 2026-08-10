from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..models import HSTUKVCache
from .kuairand_projected_persistent import (
    _load_checkpoint,
    load_persistent_config,
)
from .kuairand_projected_scale import (
    _distributed,
    _evaluation_batches,
    _initialize_model,
    _lookup,
    _seed,
)
from .kuairand_query_multiversion import _edge_config
from .kuairand_query_transition import (
    _atomic_json,
    _candidate_metrics_tie_aware,
    build_workload,
    file_sha256,
    load_config,
)

PROTOCOL = "evokv_kuairand_candidate_robustness_v0"
METRICS = (
    "candidate_cross_entropy",
    "mrr",
    "ndcg_at_5",
    "hit_rate_at_5",
    "pairwise_win_rate",
    "mean_margin",
    "hardest_margin",
    "positive_probability",
)
LOWER_IS_BETTER = {"candidate_cross_entropy"}


def _score_candidates(
    hidden: torch.Tensor,
    normalized_candidates: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return torch.einsum(
        "nh,nch->nc",
        F.normalize(hidden, dim=-1),
        normalized_candidates,
    ) / temperature


def _bootstrap(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    output = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        count = min(256, samples - start)
        selected = generator.integers(0, len(values), size=(count, len(values)))
        output[start : start + count] = values[selected].mean(axis=1)
    lower, upper = np.quantile(output, [0.025, 0.975])
    return float(lower), float(upper)


def _candidate_values(scores: torch.Tensor, count: int) -> dict[str, torch.Tensor]:
    selected = scores[:, :count]
    standard = _candidate_metrics_tie_aware(
        selected, torch.zeros(len(selected), dtype=torch.long, device=selected.device)
    )
    positive = selected[:, 0]
    negatives = selected[:, 1:]
    pairwise = (
        (positive.unsqueeze(1) > negatives).float()
        + 0.5 * (positive.unsqueeze(1) == negatives).float()
    ).mean(dim=1)
    return {
        "candidate_cross_entropy": standard["candidate_cross_entropy"],
        "mrr": standard["mrr"],
        "ndcg_at_5": standard["ndcg_at_5"],
        "hit_rate_at_5": standard["hit_rate_at_5"],
        "pairwise_win_rate": pairwise,
        "mean_margin": positive - negatives.mean(dim=1),
        "hardest_margin": positive - negatives.max(dim=1).values,
        "positive_probability": torch.softmax(selected, dim=1)[:, 0],
    }


@torch.no_grad()
def _capture(
    dense,
    embedding,
    batches: list[dict[str, Any]],
    workload: dict[str, Any],
    device: torch.device,
) -> list[HSTUKVCache]:
    dense.eval()
    embedding.eval()
    author_by_item = torch.as_tensor(
        np.asarray(workload["author_by_item"]).copy(), dtype=torch.long, device=device
    )
    output = []
    for batch in batches:
        items = batch["items"].to(device)
        lengths = torch.full(
            (len(items),), items.shape[1], dtype=torch.long, device=device
        )
        vectors = _lookup(embedding, items, lengths, author_by_item)
        cache = dense.core.compute_kv_from_item_embeddings(
            vectors,
            torch.ones_like(items),
            torch.zeros_like(items, dtype=torch.float32),
            lengths,
        )
        output.append(
            HSTUKVCache(
                k=cache.k.detach().cpu(),
                v=cache.v.detach().cpu(),
                seq_len=cache.seq_len,
            )
        )
    return output


@torch.no_grad()
def _evaluate_target(
    target_version: int,
    dense,
    embedding,
    batches: list[dict[str, Any]],
    captures: dict[int, list[HSTUKVCache]],
    workload: dict[str, Any],
    base_config: dict[str, Any],
    counts: tuple[int, ...],
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]] | None, dict[str, float]]:
    dense.eval()
    embedding.eval()
    author_by_item = torch.as_tensor(
        np.asarray(workload["author_by_item"]).copy(), dtype=torch.long, device=device
    )
    temperature = float(base_config["training"]["temperature"])
    records = []
    maximum_hidden_error = 0.0
    maximum_score_error = 0.0
    for batch_index, batch in enumerate(batches):
        items = batch["items"].to(device)
        candidates = batch["candidates"].to(device)
        lengths = torch.full(
            (len(items),), items.shape[1], dtype=torch.long, device=device
        )
        vectors = _lookup(embedding, items, lengths, author_by_item)
        behaviors = torch.ones_like(items)
        deltas = torch.zeros_like(items, dtype=torch.float32)
        current_cache = dense.core.compute_kv_from_item_embeddings(
            vectors, behaviors, deltas, lengths
        )
        query = torch.zeros(len(items), 1, dense.cfg.hidden_size, device=device)
        incremental, _ = dense.core.forward_with_cache_embedded(current_cache, query)
        prefix_embedded = dense.core.combine_input_features(vectors, behaviors, deltas)
        full_hidden, _ = dense.core.forward_embedded(
            torch.cat((prefix_embedded, query), dim=1), lengths=lengths + 1
        )
        candidate_lengths = torch.full(
            (len(candidates),),
            candidates.shape[1],
            dtype=torch.long,
            device=device,
        )
        candidate_vectors = _lookup(
            embedding, candidates, candidate_lengths, author_by_item
        )
        normalized_candidates = F.normalize(candidate_vectors, dim=-1)

        recompute_scores = _score_candidates(
            full_hidden[:, -1], normalized_candidates, temperature
        )
        incremental_scores = _score_candidates(
            incremental[:, -1], normalized_candidates, temperature
        )
        maximum_hidden_error = max(
            maximum_hidden_error,
            float((incremental[:, -1] - full_hidden[:, -1]).abs().max().item()),
        )
        maximum_score_error = max(
            maximum_score_error,
            float((incremental_scores - recompute_scores).abs().max().item()),
        )
        recompute_values = {
            count: _candidate_values(recompute_scores, count) for count in counts
        }
        for source_version, source_captures in captures.items():
            source_cache = source_captures[batch_index]
            old_cache = HSTUKVCache(
                k=source_cache.k.to(device),
                v=source_cache.v.to(device),
                seq_len=source_cache.seq_len,
            )
            reuse_hidden, _ = dense.core.forward_with_cache_embedded(old_cache, query)
            reuse_scores = _score_candidates(
                reuse_hidden[:, -1], normalized_candidates, temperature
            )
            reuse_values = {
                count: _candidate_values(reuse_scores, count) for count in counts
            }
            for row, key in enumerate(batch["keys"]):
                if not batch["valid"][row]:
                    continue
                user_id = int(workload["evaluation"][key]["user_id"])
                for count in counts:
                    records.append(
                        {
                            "source_version": source_version,
                            "target_version": target_version,
                            "candidate_count": count,
                            "user_id": user_id,
                            "recompute": [
                                float(recompute_values[count][metric][row].item())
                                for metric in METRICS
                            ],
                            "reuse": [
                                float(reuse_values[count][metric][row].item())
                                for metric in METRICS
                            ],
                        }
                    )
    sanity_tensor = torch.tensor(
        [maximum_hidden_error, maximum_score_error], dtype=torch.float64, device=device
    )
    dist.all_reduce(sanity_tensor, op=dist.ReduceOp.MAX)
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, records)
    sanity = {
        "maximum_same_model_incremental_hidden_absolute_error": float(
            sanity_tensor[0].item()
        ),
        "maximum_same_model_incremental_score_absolute_error": float(
            sanity_tensor[1].item()
        ),
    }
    if rank != 0:
        return None, sanity
    return [record for shard in gathered for record in shard], sanity


def _summarize(records: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                int(record["target_version"]),
                int(record["source_version"]),
                int(record["candidate_count"]),
                int(record["user_id"]),
            )
        ].append(record)
    users: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for (target, source, count, user_id), values in grouped.items():
        users[(target, source, count)].append(
            {
                "user_id": user_id,
                "recompute": np.asarray(
                    [value["recompute"] for value in values], dtype=np.float64
                ).mean(axis=0),
                "reuse": np.asarray(
                    [value["reuse"] for value in values], dtype=np.float64
                ).mean(axis=0),
            }
        )
    rows = []
    for ordinal, ((target, source, count), values) in enumerate(sorted(users.items())):
        recompute = np.stack([value["recompute"] for value in values])
        reuse = np.stack([value["reuse"] for value in values])
        comparisons = {}
        endpoints = {"recompute": {}, "reuse": {}}
        for metric_index, metric in enumerate(METRICS):
            current_values = recompute[:, metric_index]
            stale_values = reuse[:, metric_index]
            advantage = (
                stale_values - current_values
                if metric in LOWER_IS_BETTER
                else current_values - stale_values
            )
            lower, upper = _bootstrap(
                advantage,
                samples,
                seed + ordinal * 100003 + metric_index * 1000003,
            )
            baseline = float(stale_values.mean())
            relative = None
            if metric not in ("mean_margin", "hardest_margin") and baseline:
                relative = 100.0 * float(advantage.mean()) / baseline
            endpoints["recompute"][metric] = float(current_values.mean())
            endpoints["reuse"][metric] = baseline
            comparisons[metric] = {
                "absolute": float(advantage.mean()),
                "relative_percent": relative,
                "user_bootstrap_95": {
                    "lower": lower,
                    "upper": upper,
                    "samples": samples,
                },
                "positive_direction_with_ci": lower > 0,
                "positive_user_fraction": float((advantage > 0).mean()),
            }
        rows.append(
            {
                "target_version": target,
                "source_version": source,
                "cache_age": target - source,
                "candidate_count": count,
                "negative_count": count - 1,
                "users": len(values),
                "endpoints": endpoints,
                "recompute_over_reuse": comparisons,
            }
        )
    return rows


def _candidate_hash(workload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in workload["evaluation_keys"]:
        digest.update(np.asarray(workload["candidate_maps"][key], dtype="<i8").tobytes())
    return digest.hexdigest()


def run_candidate_robustness(
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    path = Path(config_path)
    output = Path(output_path)
    document = load_persistent_config(path)
    rank, world_size, device = _distributed(document)
    _seed(int(document["training"]["seed"]))
    if output.is_file():
        result = json.loads(output.read_text()) if rank == 0 else None
        payload = [result]
        dist.broadcast_object_list(payload, src=0)
        dist.destroy_process_group()
        return payload[0]
    base_config = load_config(document["parent"]["base_config"]["path"])
    targets = (7, 8)
    counts = (100, 500, 1000)
    workloads = {}
    batches = {}
    for target in targets:
        edge_document = _edge_config(base_config, document["transitions"][target - 1], 1.0)
        edge_document["data"]["evaluation_targets_per_user"] = int(
            document["evaluation"]["targets_per_user"]
        )
        edge_document["data"]["user_limit"] = document["data"].get("user_limit")
        edge_document["evaluation"]["candidate_count"] = counts[-1]
        workloads[target] = build_workload(edge_document)
        batches[target] = _evaluation_batches(
            workloads[target],
            int(document["evaluation"]["local_batch_size"]),
            rank,
            world_size,
        )
    embedding_rows = int(workloads[targets[0]]["metadata"]["embedding_rows"])
    dense, embedding, tracker, geometry = _initialize_model(
        document, base_config, embedding_rows, rank, world_size, device
    )
    checkpoint_root = Path(document["outputs"]["checkpoint_root"])
    captures: dict[int, dict[int, list[HSTUKVCache]]] = {
        target: {} for target in targets
    }
    for source in range(1, targets[-1]):
        _load_checkpoint(
            checkpoint_root,
            source,
            dense,
            embedding,
            tracker,
            document,
            file_sha256(path),
            rank,
        )
        for target in targets:
            if source < target:
                captures[target][source] = _capture(
                    dense, embedding, batches[target], workloads[target], device
                )
        if rank == 0:
            print(f"phase=kuairand_candidate_robustness_capture source={source}", flush=True)
    all_records = []
    sanity = {}
    for target in targets:
        _load_checkpoint(
            checkpoint_root,
            target,
            dense,
            embedding,
            tracker,
            document,
            file_sha256(path),
            rank,
        )
        target_records, target_sanity = _evaluate_target(
            target,
            dense,
            embedding,
            batches[target],
            captures[target],
            workloads[target],
            base_config,
            counts,
            rank,
            world_size,
            device,
        )
        sanity[str(target)] = target_sanity
        if rank == 0:
            assert target_records is not None
            all_records.extend(target_records)
            print(f"phase=kuairand_candidate_robustness_target target={target}", flush=True)
        del captures[target]
        torch.cuda.empty_cache()
    if rank == 0:
        result = {
            "protocol": PROTOCOL,
            "status": "complete",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(path), "sha256": file_sha256(path)},
            "geometry": geometry,
            "targets": list(targets),
            "candidate_counts": list(counts),
            "nested_candidate_semantics": "one positive followed by prefixes of one deterministic 999-negative frequency-matched set",
            "workloads": {
                str(target): {
                    "metadata": workloads[target]["metadata"],
                    "candidate_ids_sha256": _candidate_hash(workloads[target]),
                }
                for target in targets
            },
            "sanity": sanity,
            "rows": _summarize(all_records, 2000, 44119),
        }
        _atomic_json(output, result)
    else:
        result = None
    payload = [result]
    dist.broadcast_object_list(payload, src=0)
    dist.destroy_process_group()
    return payload[0]
