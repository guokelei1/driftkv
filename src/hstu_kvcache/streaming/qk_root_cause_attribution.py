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
from .qk_root_cause_sanity import (
    METRICS,
    _bootstrap_interval,
    _donor_record,
    _empty_cache,
    _metric_sums,
    _relative_tensor_error,
    _zero_cache,
)
from .qk_stream_runner import _atomic_json, _dense_state, _load_model, _runtime
from .qk_stream_version import (
    cache_relative_error,
    distributed_full_catalog_metrics,
    eligible_training_records,
    evaluation_suffix,
    file_sha256,
    fp16_storage_fp32_consumption,
    local_role_records,
    prefix_inputs,
    record_window,
)
from .sharded_edge import ExternalEmbeddingHSTU

PROTOCOL = "evokv_root_cause_qk_attribution_v0"


def _validate_document(document: dict[str, object]) -> None:
    data = document.get("data")
    checkpoints = document.get("checkpoints")
    interventions = document.get("interventions")
    quality = document.get("quality")
    execution = document.get("execution")
    outputs = document.get("outputs")
    methods = () if not isinstance(interventions, dict) else interventions.get("methods")
    pairs = () if not isinstance(interventions, dict) else interventions.get("metric_pairs")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scope") not in ("implementation_canary", "development_attribution")
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not all(
            isinstance(value, dict)
            for value in (data, checkpoints, interventions, quality, execution, outputs)
        )
        or data.get("role") != "stream_train"
        or data.get("edge") != 2
        or data.get("optimizer_participants_only") is not True
        or int(data.get("record_limit_per_rank", 0)) < 1
        or not isinstance(methods, list)
        or len(methods) != 16
        or len(set(methods)) != len(methods)
        or not isinstance(pairs, list)
        or sorted(value for pair in pairs for value in pair) != sorted(methods)
        or interventions.get("recent_lengths") != [4, 16, 64]
        or set(checkpoints) != {"theta0", "theta1", "theta2"}
        or [checkpoints[name].get("version") for name in ("theta0", "theta1", "theta2")]
        != [0, 1, 2]
        or int(quality.get("target_chunk", 0)) < 1
        or int(quality.get("full_catalog_item_chunk", 0)) < 1
        or int(quality.get("bootstrap_samples", 0)) < 1
        or execution.get("world_size") != 2
        or execution.get("cuda_visible_devices") != "0,1"
    ):
        raise ValueError("QK root-cause attribution config differs")


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
    eligible = []
    for raw_record in records:
        record = int(raw_record)
        if record not in participants:
            continue
        _, current, following = record_window(corpus, record, 2)
        offset = int(corpus.arrays["record_offsets"][record])
        positives = int(
            np.count_nonzero(corpus.arrays["label"][offset + current + 1 : offset + following + 1])
        )
        if bool(data["require_evaluation_positive"]) and not positives:
            continue
        eligible.append(record)
    generator = np.random.default_rng(int(data["sampling_seed"]) + rank)
    limit = int(data["record_limit_per_rank"])
    if len(eligible) < limit:
        raise RuntimeError("QK attribution selected record coverage differs")
    order = generator.permutation(len(eligible))[:limit]
    selected = sorted(eligible[int(value)] for value in order)
    count = torch.tensor([len(selected)], dtype=torch.int64, device=device)
    minimum = count.clone()
    maximum = count.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if minimum.item() != maximum.item():
        raise RuntimeError("QK attribution rank record counts differ")
    return selected


@torch.no_grad()
def _snapshot_projected_vectors(
    corpus: QKStreamChainCorpus,
    records: list[int],
    embedding,
    device: torch.device,
    batch_size: int,
    rank: int,
    phase: str,
) -> list[torch.Tensor]:
    snapshots = []
    for start in range(0, len(records), batch_size):
        selected = records[start : start + batch_size]
        values = [prefix_inputs(corpus, record, 2) for record in selected]
        width = max(value[3] for value in values)
        items = torch.zeros((len(values), width), dtype=torch.int64, device=device)
        lengths = torch.tensor([value[3] for value in values], dtype=torch.int64, device=device)
        for row, value in enumerate(values):
            items[row, : value[3]] = value[0].to(device)
        projected = embedding(items, lengths)
        snapshots.extend(
            projected[row, : value[3]].detach().cpu().clone() for row, value in enumerate(values)
        )
        print(
            f"phase={phase} rank={rank} records={min(start + batch_size, len(records))}/{len(records)}",
            flush=True,
        )
        del items, lengths, projected
    return snapshots


@torch.no_grad()
def _cache_from_vectors(
    dense,
    vectors: torch.Tensor,
    behaviors: torch.Tensor,
    deltas: torch.Tensor,
    device: torch.device,
) -> HSTUKVCache:
    length = len(vectors)
    lengths = torch.tensor([length], dtype=torch.int64, device=device)
    cache = dense.core.compute_kv_from_item_embeddings(
        vectors.unsqueeze(0).to(device),
        behaviors.unsqueeze(0).to(device),
        deltas.unsqueeze(0).to(device),
        lengths,
    )
    return fp16_storage_fp32_consumption(cache)


@torch.no_grad()
def _recursive_cache(
    dense0,
    dense1,
    vectors0: torch.Tensor,
    vectors1: torch.Tensor,
    behaviors: torch.Tensor,
    deltas: torch.Tensor,
    edge1_boundary: int,
    device: torch.device,
) -> HSTUKVCache:
    cache0 = _cache_from_vectors(
        dense0,
        vectors0[:edge1_boundary],
        behaviors[:edge1_boundary],
        deltas[:edge1_boundary],
        device,
    )
    _, updated = dense1.core.forward_with_cache_from_item_embeddings(
        cache0,
        vectors1[edge1_boundary:].unsqueeze(0).to(device),
        behaviors[edge1_boundary:].unsqueeze(0).to(device),
        deltas[edge1_boundary:].unsqueeze(0).to(device),
    )
    return fp16_storage_fp32_consumption(updated)


def _hybrid_dense(spec, source_state, target_state, predicate, device):
    model = ExternalEmbeddingHSTU(spec.hstu_config()).to(device)
    values = {
        name: target_state[name] if predicate(name) else value
        for name, value in source_state.items()
    }
    model.load_state_dict(values)
    model.eval()
    return model


def _is_kv_parameter(name: str) -> bool:
    return ".attn.k_proj." in name or ".attn.v_proj." in name


@torch.no_grad()
def _score_methods(
    embedding,
    hidden_values: dict[str, torch.Tensor],
    positive_ids: torch.Tensor,
    pairs: list[list[str]],
    *,
    hidden_size: int,
    num_prediction_items: int,
    target_chunk: int,
    item_chunk: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    maximum = torch.tensor(len(positive_ids), dtype=torch.int64, device=device)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    nll_parts = {name: [] for name in hidden_values}
    rank_parts = {name: [] for name in hidden_values}
    for step in range(math.ceil(int(maximum.item()) / target_chunk)):
        start = step * target_chunk
        real = min(target_chunk, max(0, len(positive_ids) - start))
        candidates = torch.zeros(target_chunk, dtype=torch.int64, device=device)
        if real:
            candidates[:real] = positive_ids[start : start + real]
        for pair in pairs:
            hidden = torch.zeros((2, target_chunk, hidden_size), dtype=torch.float32, device=device)
            if real:
                hidden[0, :real] = hidden_values[pair[0]][start : start + real]
                hidden[1, :real] = hidden_values[pair[1]][start : start + real]
            nll, ranks = distributed_full_catalog_metrics(
                embedding,
                hidden,
                candidates,
                real,
                num_prediction_items=num_prediction_items,
                item_chunk=item_chunk,
            )
            for row, method in enumerate(pair):
                nll_parts[method].append(nll[row])
                rank_parts[method].append(ranks[row])
    return (
        {name: torch.cat(values) for name, values in nll_parts.items()},
        {name: torch.cat(values) for name, values in rank_parts.items()},
    )


@torch.no_grad()
def _theta1_fresh_metrics(
    document,
    corpus,
    records,
    vectors1,
    spec,
    dense1,
    embedding1,
    *,
    rank,
    device,
) -> list[dict[str, object]]:
    results = []
    quality = document["quality"]
    for ordinal, (record, prefix_vectors) in enumerate(zip(records, vectors1, strict=True)):
        _, behaviors, deltas, _ = prefix_inputs(corpus, record, 2)
        cache = _cache_from_vectors(dense1, prefix_vectors, behaviors, deltas, device)
        suffix_items, suffix_behaviors, suffix_deltas, targets, labels = evaluation_suffix(
            corpus, record, 2
        )
        lengths = torch.tensor([len(suffix_items)], dtype=torch.int64, device=device)
        suffix_vectors = embedding1(suffix_items.unsqueeze(0).to(device), lengths)
        hidden, updated = dense1.core.forward_with_cache_from_item_embeddings(
            cache,
            suffix_vectors,
            suffix_behaviors.unsqueeze(0).to(device),
            suffix_deltas.unsqueeze(0).to(device),
        )
        mask = labels.to(device)
        positive_ids = targets.to(device)[mask]
        positive_hidden = hidden[0][mask]
        nll, ranks = _score_methods(
            embedding1,
            {"theta1_a": positive_hidden, "theta1_b": positive_hidden.clone()},
            positive_ids,
            [["theta1_a", "theta1_b"]],
            hidden_size=spec.hidden_size,
            num_prediction_items=spec.num_prediction_items,
            target_chunk=int(quality["target_chunk"]),
            item_chunk=int(quality["full_catalog_item_chunk"]),
            device=device,
        )
        if not torch.equal(ranks["theta1_a"], ranks["theta1_b"]):
            raise RuntimeError("QK theta1 duplicate ranks differ")
        duplicate_nll_error = float(torch.max(torch.abs(nll["theta1_a"] - nll["theta1_b"])).item())
        results.append(
            {
                "record": record,
                "targets": len(positive_ids),
                "metric_sums": _metric_sums(nll["theta1_a"], ranks["theta1_a"]),
                "duplicate_nll_maximum_absolute_error": duplicate_nll_error,
            }
        )
        del cache, suffix_vectors, hidden, updated, nll, ranks
        print(
            f"phase=qk_attribution_theta1_quality rank={rank} record={ordinal + 1}/{len(records)}",
            flush=True,
        )
    return results


def _metric_comparison(
    records,
    left_name,
    right_name,
    left_getter,
    right_getter,
    *,
    bootstrap_samples,
    bootstrap_seed,
):
    targets = np.asarray([value["targets"] for value in records], dtype=np.float64)
    denominator = float(targets.sum())
    output = {}
    for metric_index, metric in enumerate(METRICS):
        left = np.asarray([left_getter(value)[metric] for value in records], dtype=np.float64)
        right = np.asarray([right_getter(value)[metric] for value in records], dtype=np.float64)
        oriented = left - right if metric == "cross_entropy" else right - left
        absolute = float(oriented.sum() / denominator)
        left_mean = float(left.sum() / denominator)
        right_mean = float(right.sum() / denominator)
        interval = _bootstrap_interval(
            targets,
            oriented,
            bootstrap_samples,
            bootstrap_seed + metric_index,
        )
        output[metric] = {
            left_name: left_mean,
            right_name: right_mean,
            f"{right_name}_advantage_absolute": absolute,
            f"{right_name}_advantage_relative_percent": (
                100.0 * absolute / abs(right_mean) if right_mean else None
            ),
            "record_cluster_95_interval": interval,
            f"{right_name}_advantage_positive_with_ci": interval[0] > 0,
        }
    return output


def _aggregate(document, records, methods):
    records = sorted(records, key=lambda value: int(value["record"]))
    quality = document["quality"]
    targets = np.asarray([value["targets"] for value in records], dtype=np.float64)
    denominator = float(targets.sum())
    endpoints = {
        method: {
            metric: float(
                sum(value["metric_sums"][method][metric] for value in records) / denominator
            )
            for metric in METRICS
        }
        for method in methods
    }
    comparisons = {}
    for method_index, method in enumerate(methods):
        comparisons[method] = _metric_comparison(
            records,
            method,
            "fresh_theta2",
            lambda value, selected=method: value["metric_sums"][selected],
            lambda value: value["metric_sums"]["fresh_theta2"],
            bootstrap_samples=int(quality["bootstrap_samples"]),
            bootstrap_seed=int(quality["bootstrap_seed"]) + method_index * 101,
        )
    update_value = _metric_comparison(
        records,
        "theta1_fresh",
        "theta2_fresh",
        lambda value: value["theta1_fresh_metric_sums"],
        lambda value: value["metric_sums"]["fresh_theta2"],
        bootstrap_samples=int(quality["bootstrap_samples"]),
        bootstrap_seed=int(quality["bootstrap_seed"]) + 10001,
    )
    cache_errors = {}
    hidden_errors = {}
    for method in methods:
        cache_values = [value["cache_relative_error"].get(method) for value in records]
        cache_values = [value for value in cache_values if value is not None]
        cache_errors[method] = (
            {
                "records": len(cache_values),
                "mean": float(np.mean(cache_values)),
                "median": float(np.median(cache_values)),
                "maximum": float(np.max(cache_values)),
            }
            if cache_values
            else {"records": 0}
        )
        values = [value["hidden_relative_error"][method] for value in records]
        hidden_errors[method] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
        }
    parameter_methods = (
        "embedding_table_update_only",
        "projection_update_only",
        "embedding_projection_update",
        "kv_projection_update_only",
        "non_kv_dense_update_only",
    )
    parameter_paths = {}
    for method in parameter_methods:
        parameter_paths[method] = {}
        for metric in METRICS:
            stale_tax = comparisons["stale_theta1"][metric]["fresh_theta2_advantage_absolute"]
            stale_value = endpoints["stale_theta1"][metric]
            method_value = endpoints[method][metric]
            progress = (
                stale_value - method_value
                if metric == "cross_entropy"
                else method_value - stale_value
            )
            parameter_paths[method][metric] = {
                "stale_to_method_progress_absolute": progress,
                "stale_tax_absolute": stale_tax,
                "stale_tax_recovery_percent": (
                    100.0 * progress / stale_tax if abs(stale_tax) > 1e-12 else None
                ),
            }
    maximum_duplicate_error = max(
        value["theta1_duplicate_nll_maximum_absolute_error"] for value in records
    )
    finite = all(
        math.isfinite(value) for endpoint in endpoints.values() for value in endpoint.values()
    )
    return {
        "records": len(records),
        "positive_targets": int(denominator),
        "endpoints": endpoints,
        "fresh_theta2_comparisons": comparisons,
        "theta1_to_theta2_fresh_update_value": update_value,
        "cache_relative_error_from_fresh_theta2": cache_errors,
        "hidden_relative_error_from_fresh_theta2": hidden_errors,
        "parameter_path_recovery": parameter_paths,
        "sanity": {
            "theta1_duplicate_nll_maximum_absolute_error": maximum_duplicate_error,
            "all_endpoints_finite": finite,
            "implementation_passed": maximum_duplicate_error <= 1e-7 and finite,
        },
        "records_detail": records,
    }


@torch.no_grad()
def _evaluate_current(
    document,
    corpus,
    records,
    vectors0,
    vectors1,
    vectors_e1p2,
    vectors_e2p1,
    theta1_metrics,
    spec,
    dense0,
    dense1,
    dense2,
    dense_kv,
    dense_nonkv,
    embedding2,
    *,
    rank,
    world_size,
    device,
):
    methods = list(document["interventions"]["methods"])
    pairs = list(document["interventions"]["metric_pairs"])
    role_records = local_role_records(corpus, "stream_train", rank, world_size)
    quality = document["quality"]
    local_results = []
    for ordinal, values in enumerate(
        zip(
            records,
            vectors0,
            vectors1,
            vectors_e1p2,
            vectors_e2p1,
            theta1_metrics,
            strict=True,
        )
    ):
        record, vector0, vector1, vector_e1p2, vector_e2p1, theta1_value = values
        prefix_items, behaviors, deltas, prefix_length = prefix_inputs(corpus, record, 2)
        lengths = torch.tensor([prefix_length], dtype=torch.int64, device=device)
        vectors2 = embedding2(prefix_items.unsqueeze(0).to(device), lengths)[0]
        fresh = _cache_from_vectors(dense2, vectors2, behaviors, deltas, device)
        _, edge1_boundary, edge2_boundary = record_window(corpus, record, 1)
        if edge2_boundary != prefix_length:
            raise RuntimeError("QK recursive attribution boundary differs")
        recursive = _recursive_cache(
            dense0,
            dense1,
            vector0,
            vector1,
            behaviors,
            deltas,
            edge1_boundary,
            device,
        )
        generator = torch.Generator().manual_seed(
            int(document["interventions"]["shuffle_seed"]) + record
        )
        permutation = torch.randperm(prefix_length, generator=generator)
        permutation_device = permutation.to(device)
        donor = _donor_record(corpus, role_records, record, prefix_length)
        donor_items, donor_behaviors, donor_deltas, _ = prefix_inputs(corpus, donor, 2)
        donor_lengths = torch.tensor([prefix_length], dtype=torch.int64, device=device)
        donor_vectors = embedding2(
            donor_items[:prefix_length].unsqueeze(0).to(device), donor_lengths
        )[0]
        caches = {
            "fresh_theta2": fresh,
            "stale_theta1": _cache_from_vectors(dense1, vector1, behaviors, deltas, device),
            "direct_theta0": _cache_from_vectors(dense0, vector0, behaviors, deltas, device),
            "recursive_theta0_theta1": recursive,
            "zero_prefix": _zero_cache(fresh),
            "no_prefix": _empty_cache(fresh),
            "wrong_user_fresh": _cache_from_vectors(
                dense2,
                donor_vectors,
                donor_behaviors[:prefix_length],
                donor_deltas[:prefix_length],
                device,
            ),
            "shuffled_prefix": _cache_from_vectors(
                dense2,
                vectors2.index_select(0, permutation_device),
                behaviors.index_select(0, permutation),
                deltas,
                device,
            ),
            "embedding_table_update_only": _cache_from_vectors(
                dense1, vector_e2p1, behaviors, deltas, device
            ),
            "projection_update_only": _cache_from_vectors(
                dense1, vector_e1p2, behaviors, deltas, device
            ),
            "embedding_projection_update": _cache_from_vectors(
                dense1, vectors2, behaviors, deltas, device
            ),
            "kv_projection_update_only": _cache_from_vectors(
                dense_kv, vector1, behaviors, deltas, device
            ),
            "non_kv_dense_update_only": _cache_from_vectors(
                dense_nonkv, vector1, behaviors, deltas, device
            ),
        }
        for recent in document["interventions"]["recent_lengths"]:
            start = max(0, prefix_length - int(recent))
            caches[f"recent_{recent}"] = _cache_from_vectors(
                dense2,
                vectors2[start:],
                behaviors[start:],
                deltas[start:],
                device,
            )
        suffix_items, suffix_behaviors, suffix_deltas, targets, labels = evaluation_suffix(
            corpus, record, 2
        )
        suffix_lengths = torch.tensor([len(suffix_items)], dtype=torch.int64, device=device)
        suffix_vectors = embedding2(suffix_items.unsqueeze(0).to(device), suffix_lengths)
        mask = labels.to(device)
        positive_ids = targets.to(device)[mask]
        hidden_values = {}
        for method in methods:
            hidden, updated = dense2.core.forward_with_cache_from_item_embeddings(
                caches[method],
                suffix_vectors,
                suffix_behaviors.unsqueeze(0).to(device),
                suffix_deltas.unsqueeze(0).to(device),
            )
            hidden_values[method] = hidden[0][mask]
            del hidden, updated
        nll, ranks = _score_methods(
            embedding2,
            hidden_values,
            positive_ids,
            pairs,
            hidden_size=spec.hidden_size,
            num_prediction_items=spec.num_prediction_items,
            target_chunk=int(quality["target_chunk"]),
            item_chunk=int(quality["full_catalog_item_chunk"]),
            device=device,
        )
        cache_errors = {}
        for method in methods:
            cache_errors[method] = (
                cache_relative_error(caches[method], fresh)
                if caches[method].k.shape == fresh.k.shape
                else None
            )
        local_results.append(
            {
                "record": record,
                "user_id": int(corpus.arrays["record_user_ids"][record]),
                "wrong_user_record": donor,
                "wrong_user_id": int(corpus.arrays["record_user_ids"][donor]),
                "prefix_length": prefix_length,
                "edge1_boundary": edge1_boundary,
                "suffix_length": len(suffix_items),
                "targets": len(positive_ids),
                "metric_sums": {
                    method: _metric_sums(nll[method], ranks[method]) for method in methods
                },
                "theta1_fresh_metric_sums": theta1_value["metric_sums"],
                "theta1_duplicate_nll_maximum_absolute_error": theta1_value[
                    "duplicate_nll_maximum_absolute_error"
                ],
                "cache_relative_error": cache_errors,
                "hidden_relative_error": {
                    method: _relative_tensor_error(
                        hidden_values[method], hidden_values["fresh_theta2"]
                    )
                    for method in methods
                },
            }
        )
        del caches, suffix_vectors, hidden_values, nll, ranks, vectors2
        print(
            f"phase=qk_attribution_current rank={rank} "
            f"record={ordinal + 1}/{len(records)} targets={len(positive_ids)}",
            flush=True,
        )
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local_results)
    if rank != 0:
        return {}
    combined = [value for part in gathered for value in part]
    return _aggregate(document, combined, methods)


def run_qk_root_cause_attribution(config_path: Path) -> dict[str, object] | None:
    document = json.loads(config_path.read_text())
    _validate_document(document)
    rank, world_size, local_rank, device = _runtime()
    started = time.perf_counter()
    try:
        output = Path(document["outputs"]["result"])
        if output.exists():
            raise FileExistsError("QK root-cause attribution result already exists")
        campaign = document["campaign"]
        if file_sha256(Path(campaign["path"])) != campaign["sha256"]:
            raise ValueError("QK attribution campaign hash differs")
        corpus = load_corpus(Path(document["data"]["corpus"]))
        if corpus.file_sha256 != document["data"]["corpus_sha256"]:
            raise ValueError("QK attribution corpus hash differs")
        for name, checkpoint in document["checkpoints"].items():
            manifest = Path(checkpoint["root"]) / f"theta_{checkpoint['version']}" / "manifest.json"
            if file_sha256(manifest) != checkpoint["manifest_sha256"]:
                raise ValueError(f"QK attribution {name} hash differs")
        records = _selected_records(corpus, document, rank, world_size, device)
        batch_size = int(document["execution"]["snapshot_batch_size_per_rank"])
        checkpoint0 = document["checkpoints"]["theta0"]
        spec, loaded0, embedding0, tracker0, _ = _load_model(
            Path(checkpoint0["root"]),
            0,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        dense0_state = _dense_state(loaded0)
        loaded0.eval()
        embedding0.eval()
        vectors0 = _snapshot_projected_vectors(
            corpus,
            records,
            embedding0,
            device,
            batch_size,
            rank,
            "qk_attribution_theta0_vectors",
        )
        del loaded0, embedding0, tracker0
        gc.collect()
        torch.cuda.empty_cache()
        checkpoint1 = document["checkpoints"]["theta1"]
        spec1, loaded1, embedding1, tracker1, _ = _load_model(
            Path(checkpoint1["root"]),
            1,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        if spec1 != spec:
            raise RuntimeError("QK attribution theta1 specification differs")
        dense1_state = _dense_state(loaded1)
        loaded1.eval()
        embedding1.eval()
        projection1 = embedding1.projection_weight.detach().cpu().clone()
        vectors1 = _snapshot_projected_vectors(
            corpus,
            records,
            embedding1,
            device,
            batch_size,
            rank,
            "qk_attribution_theta1_vectors",
        )
        projection2_payload = torch.load(
            Path(document["checkpoints"]["theta2"]["root"]) / "theta_2" / "projection.pt",
            map_location="cpu",
            weights_only=True,
        )
        projection2 = projection2_payload["projection_weight"]
        with torch.no_grad():
            embedding1.projection_weight.copy_(projection2.to(device))
        vectors_e1p2 = _snapshot_projected_vectors(
            corpus,
            records,
            embedding1,
            device,
            batch_size,
            rank,
            "qk_attribution_e1p2_vectors",
        )
        with torch.no_grad():
            embedding1.projection_weight.copy_(projection1.to(device))
        theta1_metrics = _theta1_fresh_metrics(
            document,
            corpus,
            records,
            vectors1,
            spec,
            loaded1,
            embedding1,
            rank=rank,
            device=device,
        )
        del loaded1, embedding1, tracker1, projection2_payload
        gc.collect()
        torch.cuda.empty_cache()
        checkpoint2 = document["checkpoints"]["theta2"]
        spec2, dense2, embedding2, tracker2, _ = _load_model(
            Path(checkpoint2["root"]),
            2,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        if spec2 != spec:
            raise RuntimeError("QK attribution theta2 specification differs")
        dense2.eval()
        embedding2.eval()
        with torch.no_grad():
            embedding2.projection_weight.copy_(projection1.to(device))
        vectors_e2p1 = _snapshot_projected_vectors(
            corpus,
            records,
            embedding2,
            device,
            batch_size,
            rank,
            "qk_attribution_e2p1_vectors",
        )
        with torch.no_grad():
            embedding2.projection_weight.copy_(projection2.to(device))
        target_state = dense2.state_dict()
        dense0 = ExternalEmbeddingHSTU(spec.hstu_config()).to(device)
        dense0.load_state_dict(dense0_state)
        dense0.eval()
        dense1 = ExternalEmbeddingHSTU(spec.hstu_config()).to(device)
        dense1.load_state_dict(dense1_state)
        dense1.eval()
        dense_kv = _hybrid_dense(
            spec,
            dense1_state,
            target_state,
            _is_kv_parameter,
            device,
        )
        dense_nonkv = _hybrid_dense(
            spec,
            dense1_state,
            target_state,
            lambda name: not _is_kv_parameter(name),
            device,
        )
        aggregate = _evaluate_current(
            document,
            corpus,
            records,
            vectors0,
            vectors1,
            vectors_e1p2,
            vectors_e2p1,
            theta1_metrics,
            spec,
            dense0,
            dense1,
            dense2,
            dense_kv,
            dense_nonkv,
            embedding2,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        result = {
            "protocol": PROTOCOL,
            "status": "complete_development_measurement",
            "scope": document["scope"],
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
                    "path": "src/hstu_kvcache/streaming/qk_root_cause_attribution.py",
                    "sha256": file_sha256(Path(__file__)),
                },
                "metric_primitive": {
                    "path": "src/hstu_kvcache/streaming/qk_stream_version.py",
                    "sha256": file_sha256(Path("src/hstu_kvcache/streaming/qk_stream_version.py")),
                },
            },
            "campaign": campaign,
            "data": document["data"],
            "checkpoints": document["checkpoints"],
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
        del dense0, dense1, dense2, dense_kv, dense_nonkv, embedding2, tracker2
        return result if rank == 0 else None
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
