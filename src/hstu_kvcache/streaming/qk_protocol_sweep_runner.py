from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..data.qk_stream_chain import QKStreamChainCorpus, load_corpus
from .qk_stream_runner import _atomic_json, _dense_state, _load_model, _runtime
from .qk_stream_version import (
    distributed_projected_candidate_scores,
    eligible_training_records,
    evaluation_suffix,
    file_sha256,
    fp16_storage_fp32_consumption,
    local_role_records,
    prefix_inputs,
    prequential_evaluation_role_audit,
    record_window,
    snapshot_source_prefixes,
)
from .sharded_edge import ExternalEmbeddingHSTU
from .xp_projected_edge import TrainableProjectedModuloEmbedding

PROTOCOL = "evokv_qk_candidate_protocol_sweep_v0"
METRICS = (
    "cross_entropy",
    "ndcg_at_5",
    "ndcg_at_10",
    "mrr",
    "hit_rate_at_1",
    "hit_rate_at_5",
    "hit_rate_at_10",
)


def _validate_document(document: dict[str, object]) -> None:
    edge = document.get("edge")
    quality = document.get("quality")
    execution = document.get("execution")
    if (
        not isinstance(edge, dict)
        or not isinstance(quality, dict)
        or not isinstance(execution, dict)
    ):
        raise ValueError("QK protocol sweep config differs")
    source_version = edge.get("source_version")
    target_version = edge.get("target_version")
    edge_index = edge.get("edge")
    counts = quality.get("negative_counts")
    seeds = quality.get("uniform_candidate_seeds")
    variants = quality.get("popularity_variants")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_user_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not isinstance(source_version, int)
        or not isinstance(target_version, int)
        or not isinstance(edge_index, int)
        or source_version < 0
        or target_version != source_version + 1
        or edge_index != target_version
        or quality.get("primary_role") != "stream_train"
        or quality.get("supplemental_role") != "fit_tuning"
        or counts != [49, 99, 199, 499, 999]
        or not isinstance(seeds, list)
        or len(seeds) != 3
        or len(set(seeds)) != 3
        or variants != ["training_visible_exposure", "training_visible_positive"]
        or quality.get("metrics") != list(METRICS)
        or int(quality.get("target_chunk", 0)) < 1
        or int(quality.get("bootstrap_samples", 0)) < 1
        or int(quality.get("stable_minimum_negative_count", 0)) != 49
        or int(quality.get("preferred_negative_count", 0)) != 99
        or float(quality.get("preferred_relative_gap_percent_minimum", 0.0))
        != 5.0
        or float(quality.get("preferred_relative_gap_percent_maximum", 0.0))
        != 10.0
        or int(execution.get("world_size", 0)) != 2
    ):
        raise ValueError("QK protocol sweep config differs")


def _coprime_stride(size: int, seed: int) -> int:
    if size < 2:
        raise ValueError("candidate universe differs")
    value = 1 + 2 * (seed % max(1, size // 2))
    value %= size
    if value == 0:
        value = 1
    while math.gcd(value, size) != 1:
        value = (value + 2) % size
        if value == 0:
            value = 1
    return value


def nested_uniform_candidate_ids(
    positive_ids: torch.Tensor,
    *,
    num_prediction_items: int,
    maximum_negative_count: int,
    seed: int,
) -> torch.Tensor:
    positives = positive_ids.detach().cpu().long()
    if (
        positives.ndim != 1
        or maximum_negative_count < 1
        or maximum_negative_count >= num_prediction_items
        or bool(torch.any(positives < 1))
        or bool(torch.any(positives > num_prediction_items))
    ):
        raise ValueError("uniform candidate request differs")
    if positives.numel() == 0:
        return torch.empty(
            (0, maximum_negative_count + 1), dtype=torch.int64
        )
    stride = _coprime_stride(num_prediction_items, seed)
    inverse = pow(stride, -1, num_prediction_items)
    seed_mod = seed % num_prediction_items
    offsets = (
        positives.remainder(num_prediction_items) * 48_271
        + seed_mod * 69_621
    ).remainder(num_prediction_items)
    positive_zero = positives - 1
    positive_positions = (
        (positive_zero - offsets).remainder(num_prediction_items) * inverse
    ).remainder(num_prediction_items)
    positions = torch.arange(maximum_negative_count, dtype=torch.int64)
    positions = positions.unsqueeze(0).expand(len(positives), -1)
    positions = positions + (positions >= positive_positions.unsqueeze(1))
    negatives = (
        offsets.unsqueeze(1) + positions * stride
    ).remainder(num_prediction_items) + 1
    return torch.cat((positives.unsqueeze(1), negatives), dim=1)


def nested_popular_candidate_ids(
    positive_ids: torch.Tensor,
    popular_items: torch.Tensor,
    *,
    maximum_negative_count: int,
) -> torch.Tensor:
    positives = positive_ids.detach().cpu().long()
    popular = popular_items.detach().cpu().long()
    if (
        positives.ndim != 1
        or popular.ndim != 1
        or maximum_negative_count < 1
        or len(popular) < maximum_negative_count + 1
        or len(torch.unique(popular)) != len(popular)
    ):
        raise ValueError("popular candidate request differs")
    rows = []
    for positive in positives.tolist():
        selected = popular[popular != positive][:maximum_negative_count]
        if len(selected) != maximum_negative_count:
            raise ValueError("popular candidate coverage differs")
        rows.append(torch.cat((torch.tensor([positive]), selected)))
    if not rows:
        return torch.empty(
            (0, maximum_negative_count + 1), dtype=torch.int64
        )
    return torch.stack(rows)


def candidate_score_sums(scores_by_method: torch.Tensor) -> np.ndarray:
    if scores_by_method.ndim != 3 or scores_by_method.shape[0] != 2:
        raise ValueError("candidate score shape differs")
    targets = scores_by_method.shape[1]
    result = torch.zeros(
        (2, len(METRICS)), dtype=torch.float64, device=scores_by_method.device
    )
    if targets == 0:
        return result.cpu().numpy()
    scores = scores_by_method.double()
    ranks = 1 + (scores[:, :, 1:] >= scores[:, :, :1]).sum(dim=-1)
    result[:, 0] = (
        torch.logsumexp(scores, dim=-1) - scores[:, :, 0]
    ).sum(dim=1)
    for index, cutoff in ((1, 5), (2, 10)):
        result[:, index] = torch.where(
            ranks <= cutoff,
            torch.reciprocal(torch.log2(ranks.double() + 1.0)),
            torch.zeros_like(ranks, dtype=torch.float64),
        ).sum(dim=1)
    result[:, 3] = torch.reciprocal(ranks.double()).sum(dim=1)
    for index, cutoff in ((4, 1), (5, 5), (6, 10)):
        result[:, index] = (ranks <= cutoff).double().sum(dim=1)
    return result.cpu().numpy()


def training_visible_popularity(
    corpus: QKStreamChainCorpus,
    *,
    edge: int,
    num_prediction_items: int,
    maximum_negative_count: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    exposure = np.zeros(num_prediction_items + 1, dtype=np.int64)
    positive = np.zeros(num_prediction_items + 1, dtype=np.int64)
    offsets = corpus.arrays["record_offsets"]
    items = corpus.arrays["item_idx"]
    labels = corpus.arrays["label"]
    for raw_record in corpus.role_records("stream_train"):
        record = int(raw_record)
        _, current, _ = record_window(corpus, record, edge)
        start = int(offsets[record])
        stop = start + current + 1
        visible_items = items[start:stop].astype(np.int64, copy=False)
        prediction = visible_items <= num_prediction_items
        np.add.at(exposure, visible_items[prediction], 1)
        engaged = prediction & labels[start:stop].astype(np.bool_, copy=False)
        np.add.at(positive, visible_items[engaged], 1)
    item_ids = np.arange(1, num_prediction_items + 1, dtype=np.int64)
    rankings = {}
    metadata: dict[str, object] = {}
    for name, counts in (
        ("training_visible_exposure", exposure),
        ("training_visible_positive", positive),
    ):
        order = np.lexsort((item_ids, -counts[1:]))
        ranked = item_ids[order]
        if len(ranked) < maximum_negative_count + 1:
            raise ValueError("popularity candidate universe differs")
        rankings[name] = torch.from_numpy(ranked.copy())
        metadata[name] = {
            "nonzero_items": int(np.count_nonzero(counts[1:])),
            "visible_events": int(counts.sum()),
            "ranking_sha256": hashlib.sha256(
                ranked.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
            "top_items": ranked[:20].tolist(),
            "source": "stream_train events through theta1 training window only",
        }
    return rankings, metadata


def _bootstrap_summary(
    targets: np.ndarray,
    sums: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    records = len(targets)
    generator = np.random.default_rng(seed)
    probabilities = np.full(records, 1.0 / records, dtype=np.float64)
    weights = generator.multinomial(records, probabilities, size=samples)
    denominators = weights @ targets
    oriented = sums[:, :, :, 1, :] - sums[:, :, :, 0, :]
    oriented[:, :, :, 0] *= -1.0
    flat = oriented.reshape(records, -1)
    values = (weights @ flat) / denominators[:, None]
    lower = np.quantile(values, 0.025, axis=0).reshape(oriented.shape[1:])
    upper = np.quantile(values, 0.975, axis=0).reshape(oriented.shape[1:])
    return lower, upper


def summarize_candidate_matrix(
    payloads: list[dict[str, object]],
    *,
    variant_names: list[str],
    negative_counts: list[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    eligible = [value for value in payloads if int(value["targets"]) > 0]
    if not eligible:
        return {
            "records": len(payloads),
            "records_with_targets": 0,
            "positive_targets": 0,
        }
    targets = np.asarray([value["targets"] for value in eligible], dtype=np.float64)
    sums = np.stack([value["sums"] for value in eligible]).astype(
        np.float64, copy=False
    )
    if sums.shape[1:] != (
        len(variant_names),
        len(negative_counts),
        2,
        len(METRICS),
    ):
        raise ValueError("candidate summary matrix differs")
    denominator = float(targets.sum())
    endpoints = sums.sum(axis=0) / denominator
    lower, upper = _bootstrap_summary(
        targets,
        sums,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    protocols: dict[str, object] = {}
    for variant_index, variant in enumerate(variant_names):
        by_count: dict[str, object] = {}
        for count_index, count in enumerate(negative_counts):
            metrics: dict[str, object] = {}
            for metric_index, metric in enumerate(METRICS):
                reuse = float(endpoints[variant_index, count_index, 0, metric_index])
                recompute = float(
                    endpoints[variant_index, count_index, 1, metric_index]
                )
                gap = reuse - recompute if metric == "cross_entropy" else recompute - reuse
                metrics[metric] = {
                    "direction": (
                        "reuse_minus_recompute"
                        if metric == "cross_entropy"
                        else "recompute_minus_reuse"
                    ),
                    "reuse": reuse,
                    "recompute": recompute,
                    "absolute_gap": gap,
                    "relative_to_reuse_percent": (
                        100.0 * gap / reuse if reuse != 0.0 else None
                    ),
                    "record_cluster_bootstrap_95": {
                        "lower": float(lower[variant_index, count_index, metric_index]),
                        "upper": float(upper[variant_index, count_index, metric_index]),
                        "samples": bootstrap_samples,
                        "seed": bootstrap_seed,
                    },
                    "positive_direction_with_ci": bool(
                        gap > 0.0
                        and lower[variant_index, count_index, metric_index] > 0.0
                    ),
                }
            by_count[str(count)] = {
                "candidate_count": count + 1,
                "metrics": metrics,
            }
        protocols[variant] = {"negative_counts": by_count}
    return {
        "records": len(payloads),
        "records_with_targets": len(eligible),
        "positive_targets": int(denominator),
        "protocols": protocols,
    }


def _stable_gate(
    summary: dict[str, object],
    *,
    uniform_names: list[str],
    popularity_names: list[str],
    negative_counts: list[int],
    minimum_negative_count: int,
    preferred_negative_count: int,
    minimum_gap_percent: float,
    maximum_gap_percent: float,
) -> dict[str, object]:
    protocols = summary.get("protocols", {})
    rows = []
    for count in negative_counts:
        if count < minimum_negative_count:
            continue
        for family, names in (
            ("uniform_unique_three_seed", uniform_names),
            *[(name, [name]) for name in popularity_names],
        ):
            values = []
            for name in names:
                metrics = protocols[name]["negative_counts"][str(count)][
                    "metrics"
                ]
                ndcg = metrics["ndcg_at_10"]
                support = metrics["mrr"]
                values.append(
                    {
                        "protocol": name,
                        "ndcg_at_10_relative_percent": ndcg[
                            "relative_to_reuse_percent"
                        ],
                        "ndcg_at_10_positive_ci": ndcg[
                            "positive_direction_with_ci"
                        ],
                        "mrr_positive_ci": support["positive_direction_with_ci"],
                    }
                )
            passed = all(
                value["ndcg_at_10_relative_percent"] is not None
                and minimum_gap_percent
                <= value["ndcg_at_10_relative_percent"]
                <= maximum_gap_percent
                and value["ndcg_at_10_positive_ci"]
                and value["mrr_positive_ci"]
                for value in values
            )
            rows.append(
                {
                    "family": family,
                    "negative_count": count,
                    "preferred_negative_count": (
                        count == preferred_negative_count
                    ),
                    "stable_in_preferred_gap_range": passed,
                    "members": values,
                }
            )
    preference_order = [
        preferred_negative_count,
        *sorted(
            (
                value
                for value in negative_counts
                if value != preferred_negative_count
            ),
            key=lambda value: abs(math.log(value / preferred_negative_count)),
        ),
    ]
    admitted = sorted(
        (
            value
            for value in rows
            if value["stable_in_preferred_gap_range"]
        ),
        key=lambda value: (
            preference_order.index(value["negative_count"]),
            value["family"],
        ),
    )
    return {
        "criterion": {
            "minimum_negative_count": minimum_negative_count,
            "preferred_negative_count": preferred_negative_count,
            "ndcg_at_10_relative_gap_percent_range": [
                minimum_gap_percent,
                maximum_gap_percent,
            ],
            "ndcg_at_10_record_cluster_ci_lower_above_zero": True,
            "mrr_record_cluster_ci_lower_above_zero": True,
            "uniform_family_requires_all_three_candidate_seeds": True,
            "values_above_ten_percent_are_diagnostic_not_preferred": True,
        },
        "status": "admitted_protocol_found" if admitted else "no_admitted_protocol",
        "selected": admitted[0] if admitted else None,
        "admitted": admitted,
        "all_checked": rows,
    }


@torch.no_grad()
def _evaluate_role(
    document: dict[str, object],
    role: str,
    corpus: QKStreamChainCorpus,
    spec,
    current_dense: ExternalEmbeddingHSTU,
    current_embedding: TrainableProjectedModuloEmbedding,
    source_dense: ExternalEmbeddingHSTU,
    source_vectors: list[torch.Tensor],
    popular_rankings: dict[str, torch.Tensor],
    participants: set[int],
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, object]:
    quality = document["quality"]
    records = local_role_records(corpus, role, rank, world_size)
    if len(records) != len(source_vectors):
        raise ValueError("QK protocol sweep source coverage differs")
    counts = [int(value) for value in quality["negative_counts"]]
    maximum = max(counts)
    uniform_seeds = [int(value) for value in quality["uniform_candidate_seeds"]]
    uniform_names = [f"uniform_unique_seed_{index}" for index in range(len(uniform_seeds))]
    popularity_names = list(quality["popularity_variants"])
    variant_names = uniform_names + popularity_names
    segment_width = maximum + 1
    target_chunk = int(quality["target_chunk"])
    progress_every = int(document["execution"]["progress_every_records"])
    local_payloads = []
    started = time.perf_counter()
    for ordinal, (raw_record, old_vectors) in enumerate(
        zip(records, source_vectors, strict=True)
    ):
        record = int(raw_record)
        prefix_items, prefix_behaviors, prefix_deltas, prefix_length = prefix_inputs(
            corpus, record, int(document["edge"]["edge"])
        )
        lengths = torch.tensor([prefix_length], dtype=torch.int64, device=device)
        current_vectors = current_embedding(prefix_items.unsqueeze(0).to(device), lengths)
        old_cache = source_dense.core.compute_kv_from_item_embeddings(
            old_vectors.unsqueeze(0).to(device),
            prefix_behaviors.unsqueeze(0).to(device),
            prefix_deltas.unsqueeze(0).to(device),
            lengths,
        )
        recompute_cache = current_dense.core.compute_kv_from_item_embeddings(
            current_vectors,
            prefix_behaviors.unsqueeze(0).to(device),
            prefix_deltas.unsqueeze(0).to(device),
            lengths,
        )
        old_cache = fp16_storage_fp32_consumption(old_cache)
        recompute_cache = fp16_storage_fp32_consumption(recompute_cache)
        suffix_items, suffix_behaviors, suffix_deltas, targets, labels = evaluation_suffix(
            corpus, record, int(document["edge"]["edge"])
        )
        suffix_lengths = torch.tensor([len(suffix_items)], dtype=torch.int64, device=device)
        suffix_vectors = current_embedding(suffix_items.unsqueeze(0).to(device), suffix_lengths)
        reuse_hidden, _ = current_dense.core.forward_with_cache_from_item_embeddings(
            old_cache,
            suffix_vectors,
            suffix_behaviors.unsqueeze(0).to(device),
            suffix_deltas.unsqueeze(0).to(device),
        )
        recompute_hidden, _ = current_dense.core.forward_with_cache_from_item_embeddings(
            recompute_cache,
            suffix_vectors,
            suffix_behaviors.unsqueeze(0).to(device),
            suffix_deltas.unsqueeze(0).to(device),
        )
        mask = labels.to(device)
        positive_ids = targets.to(device)[mask]
        reuse_positive = reuse_hidden[0][mask]
        recompute_positive = recompute_hidden[0][mask]
        pools = [
            nested_uniform_candidate_ids(
                positive_ids,
                num_prediction_items=spec.num_prediction_items,
                maximum_negative_count=maximum,
                seed=seed + record * 1_000_003,
            )
            for seed in uniform_seeds
        ]
        pools.extend(
            nested_popular_candidate_ids(
                positive_ids,
                popular_rankings[name][: maximum + 1],
                maximum_negative_count=maximum,
            )
            for name in popularity_names
        )
        candidates = torch.cat(pools, dim=1)
        maximum_targets = torch.tensor(len(positive_ids), dtype=torch.int64, device=device)
        dist.all_reduce(maximum_targets, op=dist.ReduceOp.MAX)
        sums = np.zeros(
            (len(variant_names), len(counts), 2, len(METRICS)),
            dtype=np.float64,
        )
        steps = math.ceil(int(maximum_targets.item()) / target_chunk)
        for step in range(steps):
            start = step * target_chunk
            real = min(target_chunk, max(0, len(positive_ids) - start))
            padded_candidates = torch.zeros(
                (target_chunk, len(variant_names) * segment_width),
                dtype=torch.int64,
                device=device,
            )
            hidden = torch.zeros(
                (2, target_chunk, spec.hidden_size),
                dtype=torch.float32,
                device=device,
            )
            if real:
                padded_candidates[:real] = candidates[start : start + real].to(device)
                hidden[0, :real] = reuse_positive[start : start + real]
                hidden[1, :real] = recompute_positive[start : start + real]
            scores = distributed_projected_candidate_scores(
                current_embedding, hidden, padded_candidates, real
            )
            for variant_index in range(len(variant_names)):
                left = variant_index * segment_width
                for count_index, count in enumerate(counts):
                    selected = scores[:, :, left : left + count + 1]
                    sums[variant_index, count_index] += candidate_score_sums(selected)
        local_payloads.append(
            {
                "record": record,
                "targets": len(positive_ids),
                "participant": record in participants,
                "sums": sums,
            }
        )
        del (
            current_vectors,
            old_cache,
            recompute_cache,
            suffix_vectors,
            reuse_hidden,
            recompute_hidden,
            candidates,
        )
        if (
            ordinal == 0
            or (ordinal + 1) % progress_every == 0
            or ordinal + 1 == len(records)
        ):
            print(
                f"phase=qk_protocol_sweep role={role} rank={rank} "
                f"record={ordinal + 1}/{len(records)}",
                flush=True,
            )
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local_payloads)
    if rank != 0:
        return {}
    combined = [value for piece in gathered for value in piece]
    combined.sort(key=lambda value: int(value["record"]))
    samples = int(quality["bootstrap_samples"])
    seed = int(quality["bootstrap_seed"])
    all_summary = summarize_candidate_matrix(
        combined,
        variant_names=variant_names,
        negative_counts=counts,
        bootstrap_samples=samples,
        bootstrap_seed=seed + (0 if role == "stream_train" else 10_000_019),
    )
    result: dict[str, object] = {
        "role": role,
        "records": len(combined),
        "runtime_seconds": time.perf_counter() - started,
        "all": all_summary,
    }
    if role == "stream_train":
        participant = [value for value in combined if value["participant"]]
        nonparticipant = [value for value in combined if not value["participant"]]
        participant_summary = summarize_candidate_matrix(
            participant,
            variant_names=variant_names,
            negative_counts=counts,
            bootstrap_samples=samples,
            bootstrap_seed=seed + 20_000_033,
        )
        result["optimizer_participants"] = participant_summary
        result["nonparticipants"] = summarize_candidate_matrix(
            nonparticipant,
            variant_names=variant_names,
            negative_counts=counts,
            bootstrap_samples=samples,
            bootstrap_seed=seed + 30_000_047,
        )
        result["stable_gap_gate"] = _stable_gate(
            participant_summary,
            uniform_names=uniform_names,
            popularity_names=popularity_names,
            negative_counts=counts,
            minimum_negative_count=int(quality["stable_minimum_negative_count"]),
            preferred_negative_count=int(quality["preferred_negative_count"]),
            minimum_gap_percent=float(
                quality["preferred_relative_gap_percent_minimum"]
            ),
            maximum_gap_percent=float(
                quality["preferred_relative_gap_percent_maximum"]
            ),
        )
    return result


def run_qk_candidate_protocol_sweep(
    config_path: Path,
) -> dict[str, object] | None:
    document = json.loads(config_path.read_text())
    _validate_document(document)
    rank, world_size, local_rank, device = _runtime()
    started = time.perf_counter()
    try:
        output = Path(document["outputs"]["result"])
        if output.exists():
            raise FileExistsError("QK protocol sweep result already exists")
        corpus = load_corpus(document["data"]["corpus"])
        if corpus.file_sha256 != document["data"]["corpus_sha256"]:
            raise ValueError("QK protocol sweep corpus hash differs")
        edge = int(document["edge"]["edge"])
        source_version = int(document["edge"]["source_version"])
        target_version = int(document["edge"]["target_version"])
        audit = prequential_evaluation_role_audit(corpus, edge)
        source = document["source_checkpoint"]
        current = document["current_checkpoint"]
        source_root = Path(source["root"])
        current_root = Path(current["root"])
        source_manifest = (
            source_root / f"theta_{source_version}" / "manifest.json"
        )
        current_manifest = (
            current_root / f"theta_{target_version}" / "manifest.json"
        )
        if (
            file_sha256(source_manifest) != source["manifest_sha256"]
            or file_sha256(current_manifest) != current["manifest_sha256"]
        ):
            raise ValueError("QK protocol sweep checkpoint hash differs")
        spec, source_dense, source_embedding, _, _ = _load_model(
            source_root,
            source_version,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        source_dense_state = _dense_state(source_dense)
        snapshots = snapshot_source_prefixes(
            corpus,
            source_embedding,
            ("stream_train", "fit_tuning"),
            edge,
            rank,
            world_size,
            device,
            int(document["execution"]["snapshot_batch_size_per_rank"]),
        )
        del source_dense, source_embedding
        gc.collect()
        torch.cuda.empty_cache()
        spec, current_dense, current_embedding, _, _ = _load_model(
            current_root,
            target_version,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        source_dense = ExternalEmbeddingHSTU(spec.hstu_config()).to(device)
        source_dense.load_state_dict(source_dense_state)
        source_dense.eval()
        current_dense.eval()
        current_embedding.eval()
        maximum = max(int(value) for value in document["quality"]["negative_counts"])
        popular_rankings, popularity = training_visible_popularity(
            corpus,
            edge=edge,
            num_prediction_items=spec.num_prediction_items,
            maximum_negative_count=maximum,
        )
        participants = set(eligible_training_records(corpus, edge).tolist())
        primary = _evaluate_role(
            document,
            "stream_train",
            corpus,
            spec,
            current_dense,
            current_embedding,
            source_dense,
            snapshots["stream_train"],
            popular_rankings,
            participants,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        supplemental = _evaluate_role(
            document,
            "fit_tuning",
            corpus,
            spec,
            current_dense,
            current_embedding,
            source_dense,
            snapshots["fit_tuning"],
            popular_rankings,
            participants,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        result = {
            "protocol": PROTOCOL,
            "status": "complete_development_measurement",
            "scientific_result": False,
            "formal_result": False,
            "dataset": "tenrec-qk",
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "edge": document["edge"],
            "role_audit": audit,
            "source_checkpoint": source,
            "current_checkpoint": current,
            "candidate_protocol": {
                "negative_counts": document["quality"]["negative_counts"],
                "uniform_candidate_seeds": document["quality"][
                    "uniform_candidate_seeds"
                ],
                "uniform_generation": "seeded full-cycle permutation, positive removed, no duplicates",
                "nested_prefixes": True,
                "popularity": popularity,
                "evaluation_labels_used_for_candidates": False,
            },
            "quality": {
                "primary_update_local": primary,
                "supplemental_disjoint_user": supplemental,
                "qualification_consumed": False,
                "final_consumed": False,
                "evaluation_labels_used_for_role_or_protocol_selection": False,
                "full_matrix_retained_even_when_gate_fails": True,
            },
            "execution": {
                "world_size": world_size,
                "local_rank": local_rank,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "runtime_seconds": time.perf_counter() - started,
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
        return result if rank == 0 else None
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
