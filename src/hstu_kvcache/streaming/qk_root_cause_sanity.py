from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..data.qk_stream_chain import QKStreamChainCorpus, load_corpus
from ..models import HSTUKVCache
from .qk_stream_runner import _atomic_json, _load_model, _runtime
from .qk_stream_version import (
    distributed_full_catalog_metrics,
    eligible_training_records,
    evaluation_suffix,
    file_sha256,
    fp16_storage_fp32_consumption,
    local_role_records,
    prefix_inputs,
    record_window,
)

PROTOCOL = "evokv_root_cause_qk_sanity_v0"
METRICS = (
    "cross_entropy",
    "ndcg_at_10",
    "mrr",
    "hit_rate_at_10",
    "hit_rate_at_50",
    "hit_rate_at_200",
)


def _validate_document(document: dict[str, object]) -> None:
    data = document.get("data")
    interventions = document.get("interventions")
    quality = document.get("quality")
    execution = document.get("execution")
    outputs = document.get("outputs")
    source = document.get("source_checkpoint")
    current = document.get("current_checkpoint")
    methods = () if not isinstance(interventions, dict) else interventions.get("methods")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not all(
            isinstance(value, dict)
            for value in (data, interventions, quality, execution, outputs, source, current)
        )
        or data.get("role") != "stream_train"
        or data.get("edge") != 2
        or data.get("optimizer_participants_only") is not True
        or int(data.get("record_limit_per_rank", 0)) < 1
        or not isinstance(methods, list)
        or methods
        != [
            "fresh_full_a",
            "fresh_full_b",
            "stale_theta1",
            "zero_prefix",
            "no_prefix",
            "wrong_user_fresh",
            "shuffled_prefix",
            "recent_4",
            "recent_16",
            "recent_64",
        ]
        or interventions.get("recent_lengths") != [4, 16, 64]
        or int(quality.get("target_chunk", 0)) < 1
        or int(quality.get("full_catalog_item_chunk", 0)) < 1
        or int(quality.get("bootstrap_samples", 0)) < 1
        or execution.get("world_size") != 2
        or execution.get("cuda_visible_devices") != "0,1"
        or source.get("version") != 1
        or current.get("version") != 2
    ):
        raise ValueError("QK root-cause sanity config differs")


def _selected_records(
    corpus: QKStreamChainCorpus,
    document: dict[str, object],
    rank: int,
    world_size: int,
    device: torch.device,
) -> list[int]:
    data = document["data"]
    participants = set(eligible_training_records(corpus, 2).tolist())
    records = local_role_records(corpus, str(data["role"]), rank, world_size)
    selected = []
    for raw_record in records:
        record = int(raw_record)
        if record not in participants:
            continue
        _, current, following = record_window(corpus, record, 2)
        if current < int(data["minimum_prefix_length"]):
            continue
        offset = int(corpus.arrays["record_offsets"][record])
        positives = int(
            np.count_nonzero(corpus.arrays["label"][offset + current + 1 : offset + following + 1])
        )
        if bool(data["require_evaluation_positive"]) and not positives:
            continue
        selected.append(record)
        if len(selected) == int(data["record_limit_per_rank"]):
            break
    if len(selected) != int(data["record_limit_per_rank"]):
        raise RuntimeError("QK root-cause selected record coverage differs")
    counts = torch.tensor([len(selected)], dtype=torch.int64, device=device)
    minimum = counts.clone()
    maximum = counts.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if minimum.item() != maximum.item():
        raise RuntimeError("QK root-cause rank record counts differ")
    return selected


def _cpu_fp16_cache(cache: HSTUKVCache) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k.to(dtype=torch.float16, device="cpu"),
        v=cache.v.to(dtype=torch.float16, device="cpu"),
        seq_len=cache.seq_len,
    )


def _device_fp32_cache(cache: HSTUKVCache, device: torch.device) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k.to(device=device, dtype=torch.float32),
        v=cache.v.to(device=device, dtype=torch.float32),
        seq_len=cache.seq_len,
    )


@torch.no_grad()
def _snapshot_source_caches(
    corpus: QKStreamChainCorpus,
    records: list[int],
    source_dense,
    source_embedding,
    device: torch.device,
    rank: int,
) -> list[HSTUKVCache]:
    caches = []
    for ordinal, record in enumerate(records):
        items, behaviors, deltas, length = prefix_inputs(corpus, record, 2)
        lengths = torch.tensor([length], dtype=torch.int64, device=device)
        vectors = source_embedding(items.unsqueeze(0).to(device), lengths)
        cache = source_dense.core.compute_kv_from_item_embeddings(
            vectors,
            behaviors.unsqueeze(0).to(device),
            deltas.unsqueeze(0).to(device),
            lengths,
        )
        caches.append(_cpu_fp16_cache(cache))
        del vectors, cache
        print(
            f"phase=qk_root_cause_source rank={rank} record={ordinal + 1}/{len(records)}",
            flush=True,
        )
    return caches


def _donor_record(
    corpus: QKStreamChainCorpus,
    role_records: np.ndarray,
    record: int,
    required_length: int,
) -> int:
    boundaries = corpus.arrays["edge_last_ordinals"]
    eligible = role_records[
        (role_records != record) & (boundaries[role_records, 2] >= required_length)
    ]
    if not len(eligible):
        raise RuntimeError("QK root-cause wrong-user donor is absent")
    lengths = boundaries[eligible, 2].astype(np.int64)
    order = np.lexsort((eligible, lengths - required_length))
    return int(eligible[order[0]])


@torch.no_grad()
def _current_cache(
    corpus: QKStreamChainCorpus,
    record: int,
    current_dense,
    current_embedding,
    device: torch.device,
    *,
    start: int = 0,
    stop: int | None = None,
    permutation: torch.Tensor | None = None,
) -> HSTUKVCache:
    items, behaviors, deltas, length = prefix_inputs(corpus, record, 2)
    stop = length if stop is None else stop
    items = items[start:stop]
    behaviors = behaviors[start:stop]
    deltas = deltas[start:stop]
    if permutation is not None:
        items = items.index_select(0, permutation)
        behaviors = behaviors.index_select(0, permutation)
    lengths = torch.tensor([len(items)], dtype=torch.int64, device=device)
    vectors = current_embedding(items.unsqueeze(0).to(device), lengths)
    cache = current_dense.core.compute_kv_from_item_embeddings(
        vectors,
        behaviors.unsqueeze(0).to(device),
        deltas.unsqueeze(0).to(device),
        lengths,
    )
    result = fp16_storage_fp32_consumption(cache)
    del vectors, cache
    return result


def _empty_cache(reference: HSTUKVCache) -> HSTUKVCache:
    return HSTUKVCache(
        k=reference.k[:, :, :0].clone(),
        v=reference.v[:, :, :0].clone(),
        seq_len=0,
    )


def _zero_cache(reference: HSTUKVCache) -> HSTUKVCache:
    return HSTUKVCache(
        k=torch.zeros_like(reference.k),
        v=torch.zeros_like(reference.v),
        seq_len=reference.seq_len,
    )


def _relative_tensor_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((value - reference).double())
    denominator = torch.linalg.vector_norm(reference.double()).clamp_min(1e-12)
    return float((numerator / denominator).item())


def _metric_sums(nll: torch.Tensor, ranks: torch.Tensor) -> dict[str, float]:
    ranks = ranks.double()
    return {
        "cross_entropy": float(nll.double().sum().item()),
        "ndcg_at_10": float(
            torch.where(
                ranks <= 10,
                torch.reciprocal(torch.log2(ranks + 1.0)),
                torch.zeros_like(ranks),
            )
            .sum()
            .item()
        ),
        "mrr": float(torch.reciprocal(ranks).sum().item()),
        "hit_rate_at_10": float((ranks <= 10).double().sum().item()),
        "hit_rate_at_50": float((ranks <= 50).double().sum().item()),
        "hit_rate_at_200": float((ranks <= 200).double().sum().item()),
    }


def _bootstrap_interval(
    targets: np.ndarray,
    gaps: np.ndarray,
    samples: int,
    seed: int,
) -> list[float]:
    generator = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        count = min(256, samples - start)
        selected = generator.integers(0, len(targets), size=(count, len(targets)))
        values[start : start + count] = gaps[selected].sum(axis=1) / targets[selected].sum(axis=1)
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _aggregate(
    records: list[dict[str, object]],
    methods: list[str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    records = sorted(records, key=lambda value: int(value["record"]))
    targets = np.asarray([value["targets"] for value in records], dtype=np.float64)
    denominator = float(targets.sum())
    endpoints = {}
    for method in methods:
        endpoints[method] = {
            metric: float(
                sum(value["metric_sums"][method][metric] for value in records) / denominator
            )
            for metric in METRICS
        }
    comparisons = {}
    reference = "fresh_full_a"
    for method_index, method in enumerate(methods):
        comparison = {}
        for metric_index, metric in enumerate(METRICS):
            reference_sums = np.asarray(
                [value["metric_sums"][reference][metric] for value in records],
                dtype=np.float64,
            )
            method_sums = np.asarray(
                [value["metric_sums"][method][metric] for value in records],
                dtype=np.float64,
            )
            oriented = (
                method_sums - reference_sums
                if metric == "cross_entropy"
                else reference_sums - method_sums
            )
            absolute = float(oriented.sum() / denominator)
            reference_value = endpoints[reference][metric]
            interval = _bootstrap_interval(
                targets,
                oriented,
                bootstrap_samples,
                bootstrap_seed + method_index * 101 + metric_index,
            )
            comparison[metric] = {
                "fresh_advantage_absolute": absolute,
                "fresh_advantage_relative_percent": (
                    100.0 * absolute / abs(reference_value) if reference_value else None
                ),
                "record_cluster_95_interval": interval,
                "fresh_advantage_positive_with_ci": interval[0] > 0,
            }
        comparisons[method] = comparison
    hidden = {
        method: {
            "mean_relative_error": float(
                np.mean([value["hidden_relative_error"][method] for value in records])
            ),
            "median_relative_error": float(
                np.median([value["hidden_relative_error"][method] for value in records])
            ),
            "maximum_relative_error": float(
                np.max([value["hidden_relative_error"][method] for value in records])
            ),
        }
        for method in methods
    }
    maximum_fresh_cache = max(
        float(value["fresh_duplicate"]["cache_maximum_absolute_error"]) for value in records
    )
    maximum_fresh_hidden = max(
        float(value["fresh_duplicate"]["hidden_maximum_absolute_error"]) for value in records
    )
    maximum_canonical_nll = max(
        float(value["canonical_equivalence"]["maximum_nll_absolute_error"]) for value in records
    )
    canonical_rank_equal = all(
        bool(value["canonical_equivalence"]["ranks_equal"]) for value in records
    )
    responsive = {
        method: hidden[method]["maximum_relative_error"] > 1e-7
        for method in ("zero_prefix", "no_prefix", "wrong_user_fresh")
    }
    finite = all(math.isfinite(value) for method in endpoints.values() for value in method.values())
    implementation_passed = bool(
        maximum_fresh_cache <= 1e-7
        and maximum_fresh_hidden <= 1e-6
        and maximum_canonical_nll <= 1e-6
        and canonical_rank_equal
        and finite
        and all(responsive.values())
    )
    return {
        "records": len(records),
        "positive_targets": int(denominator),
        "endpoints": endpoints,
        "fresh_reference_comparisons": comparisons,
        "hidden_response": hidden,
        "sanity": {
            "fresh_duplicate_cache_maximum_absolute_error": maximum_fresh_cache,
            "fresh_duplicate_hidden_maximum_absolute_error": maximum_fresh_hidden,
            "canonical_pair_nll_maximum_absolute_error": maximum_canonical_nll,
            "canonical_pair_ranks_equal": canonical_rank_equal,
            "strong_perturbation_hidden_response": responsive,
            "all_endpoints_finite": finite,
            "implementation_passed": implementation_passed,
        },
        "records_detail": records,
    }


@torch.no_grad()
def _evaluate(
    document: dict[str, object],
    corpus: QKStreamChainCorpus,
    records: list[int],
    source_caches: list[HSTUKVCache],
    spec,
    current_dense,
    current_embedding,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, object]:
    methods = list(document["interventions"]["methods"])
    target_chunk = int(document["quality"]["target_chunk"])
    item_chunk = int(document["quality"]["full_catalog_item_chunk"])
    shuffle_seed = int(document["interventions"]["shuffle_seed"])
    role_records = local_role_records(corpus, "stream_train", rank, world_size)
    local_results = []
    for ordinal, (record, source_cpu) in enumerate(zip(records, source_caches, strict=True)):
        prefix_items, _, _, prefix_length = prefix_inputs(corpus, record, 2)
        fresh_a = _current_cache(corpus, record, current_dense, current_embedding, device)
        fresh_b = _current_cache(corpus, record, current_dense, current_embedding, device)
        generator = torch.Generator().manual_seed(shuffle_seed + record)
        permutation = torch.randperm(prefix_length, generator=generator)
        donor = _donor_record(corpus, role_records, record, prefix_length)
        caches = {
            "fresh_full_a": fresh_a,
            "fresh_full_b": fresh_b,
            "stale_theta1": _device_fp32_cache(source_cpu, device),
            "zero_prefix": _zero_cache(fresh_a),
            "no_prefix": _empty_cache(fresh_a),
            "wrong_user_fresh": _current_cache(
                corpus,
                donor,
                current_dense,
                current_embedding,
                device,
                stop=prefix_length,
            ),
            "shuffled_prefix": _current_cache(
                corpus,
                record,
                current_dense,
                current_embedding,
                device,
                permutation=permutation,
            ),
        }
        for recent in document["interventions"]["recent_lengths"]:
            start = max(0, prefix_length - int(recent))
            caches[f"recent_{recent}"] = _current_cache(
                corpus,
                record,
                current_dense,
                current_embedding,
                device,
                start=start,
            )
        suffix_items, suffix_behaviors, suffix_deltas, targets, labels = evaluation_suffix(
            corpus, record, 2
        )
        suffix_lengths = torch.tensor([len(suffix_items)], dtype=torch.int64, device=device)
        suffix_vectors = current_embedding(suffix_items.unsqueeze(0).to(device), suffix_lengths)
        positive_mask = labels.to(device)
        positive_ids = targets.to(device)[positive_mask]
        hidden_values = {}
        for method in methods:
            hidden, updated = current_dense.core.forward_with_cache_from_item_embeddings(
                caches[method],
                suffix_vectors,
                suffix_behaviors.unsqueeze(0).to(device),
                suffix_deltas.unsqueeze(0).to(device),
            )
            hidden_values[method] = hidden[0][positive_mask]
            del hidden, updated
        hidden_by_method = torch.stack([hidden_values[method] for method in methods])
        maximum_targets = torch.tensor(len(positive_ids), dtype=torch.int64, device=device)
        dist.all_reduce(maximum_targets, op=dist.ReduceOp.MAX)
        nll_parts = []
        rank_parts = []
        canonical_error = 0.0
        canonical_ranks_equal = True
        metric_pairs = (
            ("stale_theta1", "fresh_full_a"),
            ("fresh_full_b", "zero_prefix"),
            ("no_prefix", "wrong_user_fresh"),
            ("shuffled_prefix", "recent_4"),
            ("recent_16", "recent_64"),
        )
        for step in range(math.ceil(int(maximum_targets.item()) / target_chunk)):
            start = step * target_chunk
            real = min(target_chunk, max(0, len(positive_ids) - start))
            padded_hidden = torch.zeros(
                (len(methods), target_chunk, spec.hidden_size),
                dtype=torch.float32,
                device=device,
            )
            padded_ids = torch.zeros(target_chunk, dtype=torch.int64, device=device)
            if real:
                padded_hidden[:, :real] = hidden_by_method[:, start : start + real]
                padded_ids[:real] = positive_ids[start : start + real]
            method_nll: list[torch.Tensor | None] = [None] * len(methods)
            method_ranks: list[torch.Tensor | None] = [None] * len(methods)
            canonical_hidden = None
            canonical_nll = None
            canonical_ranks = None
            for left, right in metric_pairs:
                pair_indices = [methods.index(left), methods.index(right)]
                pair_hidden = padded_hidden[pair_indices]
                pair_nll, pair_ranks = distributed_full_catalog_metrics(
                    current_embedding,
                    pair_hidden,
                    padded_ids,
                    real,
                    num_prediction_items=spec.num_prediction_items,
                    item_chunk=item_chunk,
                )
                for pair_row, method_index in enumerate(pair_indices):
                    method_nll[method_index] = pair_nll[pair_row]
                    method_ranks[method_index] = pair_ranks[pair_row]
                if left == "stale_theta1":
                    canonical_hidden = pair_hidden
                    canonical_nll = pair_nll
                    canonical_ranks = pair_ranks
            if any(value is None for value in (*method_nll, *method_ranks)):
                raise RuntimeError("QK root-cause metric pair coverage differs")
            check_nll, check_ranks = distributed_full_catalog_metrics(
                current_embedding,
                canonical_hidden,
                padded_ids,
                real,
                num_prediction_items=spec.num_prediction_items,
                item_chunk=item_chunk,
            )
            canonical_error = max(
                canonical_error,
                float(torch.max(torch.abs(check_nll - canonical_nll)).item()) if real else 0.0,
            )
            canonical_ranks_equal = bool(
                canonical_ranks_equal and torch.equal(check_ranks, canonical_ranks)
            )
            nll = torch.stack(method_nll)
            ranks = torch.stack(method_ranks)
            nll_parts.append(nll)
            rank_parts.append(ranks)
        nll = torch.cat(nll_parts, dim=1)
        ranks = torch.cat(rank_parts, dim=1)
        fresh_index = methods.index("fresh_full_a")
        metric_sums = {
            method: _metric_sums(nll[index], ranks[index]) for index, method in enumerate(methods)
        }
        hidden_relative_error = {
            method: _relative_tensor_error(hidden_values[method], hidden_values["fresh_full_a"])
            for method in methods
        }
        fresh_cache_error = max(
            float(torch.max(torch.abs(fresh_a.k - fresh_b.k)).item()),
            float(torch.max(torch.abs(fresh_a.v - fresh_b.v)).item()),
        )
        fresh_hidden_error = float(
            torch.max(
                torch.abs(hidden_values["fresh_full_a"] - hidden_values["fresh_full_b"])
            ).item()
        )
        local_results.append(
            {
                "record": record,
                "user_id": int(corpus.arrays["record_user_ids"][record]),
                "wrong_user_record": donor,
                "wrong_user_id": int(corpus.arrays["record_user_ids"][donor]),
                "prefix_length": prefix_length,
                "suffix_length": len(suffix_items),
                "targets": len(positive_ids),
                "metric_sums": metric_sums,
                "hidden_relative_error": hidden_relative_error,
                "mean_absolute_rank_shift_from_fresh": {
                    method: float(
                        torch.mean(
                            torch.abs(ranks[index].double() - ranks[fresh_index].double())
                        ).item()
                    )
                    for index, method in enumerate(methods)
                },
                "fresh_duplicate": {
                    "cache_maximum_absolute_error": fresh_cache_error,
                    "hidden_maximum_absolute_error": fresh_hidden_error,
                },
                "canonical_equivalence": {
                    "maximum_nll_absolute_error": canonical_error,
                    "ranks_equal": canonical_ranks_equal,
                },
            }
        )
        del (
            caches,
            suffix_vectors,
            hidden_values,
            hidden_by_method,
            nll,
            ranks,
        )
        print(
            f"phase=qk_root_cause_evaluate rank={rank} "
            f"record={ordinal + 1}/{len(records)} targets={len(positive_ids)}",
            flush=True,
        )
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local_results)
    if rank != 0:
        return {}
    combined = [value for part in gathered for value in part]
    return _aggregate(
        combined,
        methods,
        bootstrap_samples=int(document["quality"]["bootstrap_samples"]),
        bootstrap_seed=int(document["quality"]["bootstrap_seed"]),
    )


def run_qk_root_cause_sanity(config_path: Path) -> dict[str, object] | None:
    document = json.loads(config_path.read_text())
    _validate_document(document)
    rank, world_size, local_rank, device = _runtime()
    started = time.perf_counter()
    try:
        output = Path(document["outputs"]["result"])
        if output.exists():
            raise FileExistsError("QK root-cause sanity result already exists")
        campaign = document["campaign"]
        if file_sha256(Path(campaign["path"])) != campaign["sha256"]:
            raise ValueError("QK root-cause campaign hash differs")
        corpus = load_corpus(Path(document["data"]["corpus"]))
        if corpus.file_sha256 != document["data"]["corpus_sha256"]:
            raise ValueError("QK root-cause corpus hash differs")
        source = document["source_checkpoint"]
        current = document["current_checkpoint"]
        source_manifest = Path(source["root"]) / f"theta_{source['version']}" / "manifest.json"
        current_manifest = Path(current["root"]) / f"theta_{current['version']}" / "manifest.json"
        if (
            file_sha256(source_manifest) != source["manifest_sha256"]
            or file_sha256(current_manifest) != current["manifest_sha256"]
        ):
            raise ValueError("QK root-cause checkpoint hash differs")
        records = _selected_records(corpus, document, rank, world_size, device)
        _, source_dense, source_embedding, source_tracker, _ = _load_model(
            Path(source["root"]),
            int(source["version"]),
            rank=rank,
            world_size=world_size,
            device=device,
        )
        source_dense.eval()
        source_embedding.eval()
        source_caches = _snapshot_source_caches(
            corpus,
            records,
            source_dense,
            source_embedding,
            device,
            rank,
        )
        del source_dense, source_embedding, source_tracker
        gc.collect()
        torch.cuda.empty_cache()
        spec, current_dense, current_embedding, current_tracker, _ = _load_model(
            Path(current["root"]),
            int(current["version"]),
            rank=rank,
            world_size=world_size,
            device=device,
        )
        current_dense.eval()
        current_embedding.eval()
        aggregate = _evaluate(
            document,
            corpus,
            records,
            source_caches,
            spec,
            current_dense,
            current_embedding,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        result = {
            "protocol": PROTOCOL,
            "status": "complete_development_measurement",
            "scientific_result": False,
            "formal_result": False,
            "round_id": document["round_id"],
            "dataset": "tenrec-qk",
            "config": {
                "path": str(config_path),
                "sha256": file_sha256(config_path),
            },
            "programs": {
                "runner": {
                    "path": "src/hstu_kvcache/streaming/qk_root_cause_sanity.py",
                    "sha256": file_sha256(Path(__file__)),
                },
                "metric_primitive": {
                    "path": "src/hstu_kvcache/streaming/qk_stream_version.py",
                    "sha256": file_sha256(Path("src/hstu_kvcache/streaming/qk_stream_version.py")),
                },
            },
            "campaign": document["campaign"],
            "data": document["data"],
            "source_checkpoint": source,
            "current_checkpoint": current,
            "interventions": document["interventions"],
            "quality": document["quality"],
            "aggregate": aggregate,
            "execution": {
                "world_size": world_size,
                "local_rank": local_rank,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "runtime_seconds": time.perf_counter() - started,
                "qualification_consumed": False,
                "final_consumed": False,
                "hbm": {
                    "allocated_bytes": torch.cuda.memory_allocated(device),
                    "reserved_bytes": torch.cuda.memory_reserved(device),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                },
            },
        }
        if rank == 0:
            _atomic_json(output, result)
        dist.barrier()
        del current_dense, current_embedding, current_tracker
        return result if rank == 0 else None
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
