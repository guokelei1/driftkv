from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..data.qk_stream_chain import QKStreamChainCorpus, load_corpus
from .qk_protocol_sweep_runner import (
    METRICS,
    candidate_score_sums,
    nested_uniform_candidate_ids,
    summarize_candidate_matrix,
)
from .qk_stream_runner import _atomic_json, _dense_state, _load_model, _runtime
from .qk_stream_version import (
    distributed_full_catalog_metrics,
    distributed_projected_candidate_scores,
    eligible_training_records,
    evaluation_suffix,
    file_sha256,
    fp16_storage_fp32_consumption,
    local_role_records,
    paired_full_catalog_summary,
    prefix_inputs,
    prequential_evaluation_role_audit,
    record_window,
    snapshot_source_prefixes,
    summarize_full_catalog_record,
)
from .sharded_edge import ExternalEmbeddingHSTU
from .xp_projected_edge import TrainableProjectedModuloEmbedding

PROTOCOL = "evokv_qk_update_relevance_evaluation_v0"
MODES = ("rolling_next_item", "boundary_multi_positive")
COHORTS = (
    "all",
    "first_positive",
    "offset_lt_16",
    "context_h16_support_ge1",
    "context_h32_support_ge1",
    "context_h32_support_ge2",
    "context_h32_support_ge1_offset_lt_16",
    "context_h32_support_ge1_first_positive",
    "copositive_support_ge1",
    "successor_or_copositive",
)
RELATION_COHORTS = (
    "context_h16_support_ge1",
    "context_h32_support_ge1",
    "context_h32_support_ge2",
    "context_h32_support_ge1_offset_lt_16",
    "context_h32_support_ge1_first_positive",
    "copositive_support_ge1",
    "successor_or_copositive",
)
CANDIDATE_VARIANTS = ("uniform_unique", "context_successor_hard")


@dataclass(frozen=True)
class SparseRelationIndex:
    sources: np.ndarray
    offsets: np.ndarray
    targets: np.ndarray
    counts: np.ndarray

    def lookup(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        contexts = np.unique(contexts.astype(np.uint32, copy=False))
        pieces_targets = []
        pieces_counts = []
        positions = np.searchsorted(self.sources, contexts)
        for context, position in zip(contexts, positions, strict=True):
            if position >= len(self.sources) or self.sources[position] != context:
                continue
            start = int(self.offsets[position])
            stop = int(self.offsets[position + 1])
            pieces_targets.append(self.targets[start:stop])
            pieces_counts.append(self.counts[start:stop])
        if not pieces_targets:
            return (
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
            )
        targets = np.concatenate(pieces_targets)
        counts = np.concatenate(pieces_counts).astype(np.uint64, copy=False)
        unique, inverse = np.unique(targets, return_inverse=True)
        totals = np.zeros(len(unique), dtype=np.uint64)
        np.add.at(totals, inverse, counts)
        return unique, totals.astype(np.uint32)


def _relation_index(
    source_parts: list[np.ndarray],
    target_parts: list[np.ndarray],
) -> SparseRelationIndex:
    if not source_parts or len(source_parts) != len(target_parts):
        raise ValueError("QK update relation edge collection differs")
    sources = np.concatenate(source_parts).astype(np.uint64, copy=False)
    targets = np.concatenate(target_parts).astype(np.uint64, copy=False)
    if sources.shape != targets.shape or np.any(sources > np.iinfo(np.uint32).max):
        raise ValueError("QK update relation edge values differ")
    packed = (sources << np.uint64(32)) | targets
    unique, counts = np.unique(packed, return_counts=True)
    edge_sources = (unique >> np.uint64(32)).astype(np.uint32)
    edge_targets = (unique & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    relation_sources, starts = np.unique(edge_sources, return_index=True)
    offsets = np.concatenate(
        (starts.astype(np.int64), np.asarray([len(unique)], dtype=np.int64))
    )
    return SparseRelationIndex(
        sources=relation_sources,
        offsets=offsets,
        targets=edge_targets,
        counts=counts.astype(np.uint32),
    )


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def build_training_relations(
    corpus: QKStreamChainCorpus,
    *,
    edge: int,
    transition_horizon: int,
    num_prediction_items: int,
) -> tuple[SparseRelationIndex, SparseRelationIndex, np.ndarray, dict[str, object]]:
    if transition_horizon != 32 or num_prediction_items < 1:
        raise ValueError("QK update relation construction differs")
    offsets = corpus.arrays["record_offsets"]
    item_ids = corpus.arrays["item_idx"]
    labels = corpus.arrays["label"].astype(np.bool_, copy=False)
    successor_sources = []
    successor_targets = []
    copositive_sources = []
    copositive_targets = []
    positive_counts = np.zeros(num_prediction_items + 1, dtype=np.int64)
    participants = eligible_training_records(corpus, edge)
    positive_events = 0
    for raw_record in participants:
        record = int(raw_record)
        previous, current, _ = record_window(corpus, record, edge)
        start = int(offsets[record])
        values = item_ids[start + previous + 1 : start + current + 1].astype(
            np.uint32, copy=False
        )
        outcomes = labels[start + previous + 1 : start + current + 1]
        positives = values[outcomes]
        prediction = positives <= num_prediction_items
        np.add.at(positive_counts, positives[prediction], 1)
        positive_events += int(np.count_nonzero(prediction))
        for position in np.nonzero(outcomes)[0]:
            target = int(values[position])
            if target > num_prediction_items:
                continue
            context = np.unique(
                values[max(0, position - transition_horizon) : position]
            )
            if len(context):
                successor_sources.append(context)
                successor_targets.append(
                    np.full(len(context), target, dtype=np.uint32)
                )
        unique_positives = np.unique(positives[prediction])
        if len(unique_positives) > 1:
            left = np.repeat(unique_positives, len(unique_positives))
            right = np.tile(unique_positives, len(unique_positives))
            mask = left != right
            copositive_sources.append(left[mask])
            copositive_targets.append(right[mask])
    successor = _relation_index(successor_sources, successor_targets)
    copositive = _relation_index(copositive_sources, copositive_targets)
    prediction_ids = np.arange(1, num_prediction_items + 1, dtype=np.int64)
    popular_order = np.lexsort((prediction_ids, -positive_counts[1:]))
    popular = prediction_ids[popular_order]
    metadata = {
        "participants": len(participants),
        "training_window": edge,
        "transition_horizon": transition_horizon,
        "source": f"stream_train window{edge} only",
        "evaluation_window_events_used": False,
        "positive_events": positive_events,
        "successor_sources": len(successor.sources),
        "successor_edges": len(successor.targets),
        "copositive_sources": len(copositive.sources),
        "copositive_edges": len(copositive.targets),
        "successor_sha256": hashlib.sha256(
            successor.sources.tobytes()
            + successor.offsets.tobytes()
            + successor.targets.tobytes()
            + successor.counts.tobytes()
        ).hexdigest(),
        "copositive_sha256": hashlib.sha256(
            copositive.sources.tobytes()
            + copositive.offsets.tobytes()
            + copositive.targets.tobytes()
            + copositive.counts.tobytes()
        ).hexdigest(),
        "positive_popularity_sha256": _array_sha256(popular.astype("<i8")),
    }
    return successor, copositive, popular, metadata


def relation_cohort_masks(
    target_offsets: np.ndarray,
    target_orders: np.ndarray,
    h16_support: np.ndarray,
    h32_support: np.ndarray,
    copositive_support: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = (
        target_offsets,
        target_orders,
        h16_support,
        h32_support,
        copositive_support,
    )
    if any(value.ndim != 1 for value in arrays) or len({len(value) for value in arrays}) != 1:
        raise ValueError("QK update relation cohort inputs differ")
    first = target_orders == 0
    h16 = h16_support >= 1
    h32 = h32_support >= 1
    copositive = copositive_support >= 1
    return {
        "all": np.ones(len(target_offsets), dtype=np.bool_),
        "first_positive": first,
        "offset_lt_16": target_offsets < 16,
        "context_h16_support_ge1": h16,
        "context_h32_support_ge1": h32,
        "context_h32_support_ge2": h32_support >= 2,
        "context_h32_support_ge1_offset_lt_16": h32 & (target_offsets < 16),
        "context_h32_support_ge1_first_positive": h32 & first,
        "copositive_support_ge1": copositive,
        "successor_or_copositive": h32 | copositive,
    }


def _support_for_targets(
    relation_targets: np.ndarray,
    relation_counts: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    positions = np.searchsorted(relation_targets, targets)
    support = np.zeros(len(targets), dtype=np.uint32)
    valid = positions < len(relation_targets)
    matched = np.zeros(len(targets), dtype=np.bool_)
    matched[valid] = relation_targets[positions[valid]] == targets[valid]
    support[matched] = relation_counts[positions[matched]]
    return support


def record_relation_state(
    corpus: QKStreamChainCorpus,
    record: int,
    edge: int,
    successor: SparseRelationIndex,
    copositive: SparseRelationIndex,
) -> dict[str, np.ndarray]:
    previous, current, _ = record_window(corpus, record, edge)
    start = int(corpus.arrays["record_offsets"][record])
    values = corpus.arrays["item_idx"][
        start + previous + 1 : start + current + 1
    ].astype(np.uint32, copy=False)
    outcomes = corpus.arrays["label"][
        start + previous + 1 : start + current + 1
    ].astype(np.bool_, copy=False)
    h16_targets, h16_counts = successor.lookup(values[-16:])
    h32_targets, h32_counts = successor.lookup(values[-32:])
    copositive_targets, copositive_counts = copositive.lookup(values[outcomes])
    _, _, _, targets, labels = evaluation_suffix(corpus, record, edge)
    target_offsets = torch.nonzero(labels, as_tuple=False).flatten().numpy()
    positives = targets[labels].numpy().astype(np.uint32, copy=False)
    order = np.lexsort((h32_targets, -h32_counts.astype(np.int64)))
    return {
        "target_offsets": target_offsets.astype(np.int16, copy=False),
        "target_orders": np.arange(len(positives), dtype=np.int16),
        "positive_ids": positives,
        "h16_support": _support_for_targets(h16_targets, h16_counts, positives),
        "h32_support": _support_for_targets(h32_targets, h32_counts, positives),
        "copositive_support": _support_for_targets(
            copositive_targets, copositive_counts, positives
        ),
        "ranked_h32_targets": h32_targets[order],
    }


def context_hard_candidate_ids(
    positive_ids: torch.Tensor,
    ranked_context_items: np.ndarray,
    popular_items: np.ndarray,
    uniform_fallback: torch.Tensor,
    *,
    maximum_negative_count: int,
) -> tuple[torch.Tensor, int]:
    positives = positive_ids.detach().cpu().long()
    if (
        positives.ndim != 1
        or uniform_fallback.shape != (len(positives), maximum_negative_count + 1)
        or maximum_negative_count < 1
    ):
        raise ValueError("QK context hard candidate request differs")
    context = [int(value) for value in ranked_context_items]
    popular = [int(value) for value in popular_items]
    rows = []
    context_selected = 0
    for row, positive in enumerate(positives.tolist()):
        selected = [positive]
        seen = {positive}
        for item in context:
            if item not in seen:
                selected.append(item)
                seen.add(item)
                context_selected += 1
                if len(selected) == maximum_negative_count + 1:
                    break
        if len(selected) < maximum_negative_count + 1:
            for item in popular:
                if item not in seen:
                    selected.append(item)
                    seen.add(item)
                    if len(selected) == maximum_negative_count + 1:
                        break
        if len(selected) < maximum_negative_count + 1:
            for item in uniform_fallback[row, 1:].tolist():
                if item not in seen:
                    selected.append(item)
                    seen.add(item)
                    if len(selected) == maximum_negative_count + 1:
                        break
        if len(selected) != maximum_negative_count + 1:
            raise RuntimeError("QK context hard candidate coverage differs")
        rows.append(torch.tensor(selected, dtype=torch.int64))
    if not rows:
        return torch.empty((0, maximum_negative_count + 1), dtype=torch.int64), 0
    return torch.stack(rows), context_selected


def _validate_document(document: dict[str, object]) -> None:
    edge = document.get("edge")
    quality = document.get("quality")
    execution = document.get("execution")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_user_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not isinstance(edge, dict)
        or edge.get("source_version") != 1
        or edge.get("target_version") != 2
        or edge.get("edge") != 2
        or not isinstance(quality, dict)
        or quality.get("primary_role") != "stream_train"
        or quality.get("evaluation_users") != "optimizer_participants"
        or quality.get("modes") != list(MODES)
        or quality.get("cohorts") != list(COHORTS)
        or quality.get("relation_cohorts") != list(RELATION_COHORTS)
        or quality.get("candidate_variants") != list(CANDIDATE_VARIANTS)
        or quality.get("candidate_negative_counts") != [99, 499]
        or quality.get("transition_horizon") != 32
        or quality.get("preferred_relative_gap_percent_range") != [5.0, 10.0]
        or int(quality.get("minimum_cohort_targets", 0)) < 1
        or int(quality.get("bootstrap_samples", 0)) < 1
        or not isinstance(execution, dict)
        or execution.get("world_size") != 2
        or int(execution.get("record_limit_per_rank", -1)) < 0
    ):
        raise ValueError("QK update relevance config differs")


def _summarize_payloads(
    payloads: list[dict[str, object]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    negative_counts: list[int],
) -> dict[str, object]:
    full: dict[str, object] = {}
    candidates: dict[str, object] = {}
    ordinal = 0
    for mode in MODES:
        full_cohorts = {}
        candidate_cohorts = {}
        for cohort in COHORTS:
            records = [value["full"][mode][cohort] for value in payloads]
            if any(int(value["targets"]) > 0 for value in records):
                full_cohorts[cohort] = paired_full_catalog_summary(
                    records,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed + ordinal * 1_000_003,
                )
            else:
                full_cohorts[cohort] = {"records": len(records), "positive_targets": 0}
            ordinal += 1
            if cohort in RELATION_COHORTS:
                candidate_records = [
                    value["candidates"][mode][cohort] for value in payloads
                ]
                candidate_cohorts[cohort] = summarize_candidate_matrix(
                    candidate_records,
                    variant_names=list(CANDIDATE_VARIANTS),
                    negative_counts=negative_counts,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed + ordinal * 1_000_003,
                )
                ordinal += 1
        full[mode] = {"cohorts": full_cohorts}
        candidates[mode] = {"cohorts": candidate_cohorts}
    return {"full_catalog": full, "candidate_protocols": candidates}


def _admission_gate(
    summary: dict[str, object],
    *,
    candidates: list[dict[str, str]],
    minimum_targets: int,
    relative_range: list[float],
) -> dict[str, object]:
    rows = []
    for candidate in candidates:
        mode = candidate["mode"]
        cohort = candidate["cohort"]
        value = summary["full_catalog"][mode]["cohorts"][cohort]
        targets = int(value.get("positive_targets", 0))
        if targets:
            ndcg = value["gaps"]["ndcg_at_10"]
            mrr = value["gaps"]["mrr"]
            relative = ndcg["relative_percent"]
            passed = bool(
                targets >= minimum_targets
                and relative is not None
                and relative_range[0] <= relative <= relative_range[1]
                and ndcg["positive_direction_with_ci"]
                and mrr["positive_direction_with_ci"]
            )
        else:
            ndcg = None
            mrr = None
            relative = None
            passed = False
        rows.append(
            {
                "mode": mode,
                "cohort": cohort,
                "positive_targets": targets,
                "ndcg_at_10_relative_percent": relative,
                "ndcg_at_10_positive_ci": None if ndcg is None else ndcg["positive_direction_with_ci"],
                "mrr_positive_ci": None if mrr is None else mrr["positive_direction_with_ci"],
                "preferred_gap_passed": passed,
            }
        )
    admitted = [value for value in rows if value["preferred_gap_passed"]]
    return {
        "criterion": {
            "candidate_set": "all prediction items",
            "minimum_targets": minimum_targets,
            "ndcg_at_10_relative_gap_percent_range": relative_range,
            "ndcg_at_10_and_mrr_record_cluster_ci_lower_above_zero": True,
            "relations_built_from_training_window_only": True,
        },
        "status": "preferred_full_catalog_gap_found" if admitted else "no_preferred_full_catalog_gap",
        "admitted": admitted,
        "all_checked": rows,
    }


@torch.no_grad()
def _evaluate(
    document: dict[str, object],
    corpus: QKStreamChainCorpus,
    spec,
    current_dense: ExternalEmbeddingHSTU,
    current_embedding: TrainableProjectedModuloEmbedding,
    source_dense: ExternalEmbeddingHSTU,
    source_vectors: list[torch.Tensor],
    successor: SparseRelationIndex,
    copositive: SparseRelationIndex,
    popular_items: np.ndarray,
    participants: set[int],
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, object]:
    quality = document["quality"]
    records = local_role_records(corpus, "stream_train", rank, world_size)
    if len(records) != len(source_vectors):
        raise ValueError("QK update relevance source coverage differs")
    limit = int(document["execution"]["record_limit_per_rank"])
    target_chunk = int(quality["target_chunk"])
    item_chunk = int(quality["full_catalog_item_chunk"])
    negative_counts = [int(value) for value in quality["candidate_negative_counts"]]
    maximum_negative_count = max(negative_counts)
    progress_every = int(document["execution"]["progress_every_records"])
    local_payloads = []
    context_selected = 0
    candidate_slots = 0
    processed = 0
    started = time.perf_counter()
    for ordinal, (raw_record, old_vectors) in enumerate(zip(records, source_vectors, strict=True)):
        record = int(raw_record)
        if limit and processed >= limit:
            break
        processed += 1
        participant = record in participants
        relation = record_relation_state(corpus, record, 2, successor, copositive)
        masks = relation_cohort_masks(
            relation["target_offsets"],
            relation["target_orders"],
            relation["h16_support"],
            relation["h32_support"],
            relation["copositive_support"],
        )
        prefix_items, prefix_behaviors, prefix_deltas, prefix_length = prefix_inputs(
            corpus, record, 2
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
            corpus, record, 2
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
        positive_mask = labels.to(device)
        if not participant:
            positive_mask = torch.zeros_like(positive_mask)
        positive_ids = targets.to(device)[positive_mask]
        rolling_hidden = torch.stack(
            (reuse_hidden[0][positive_mask], recompute_hidden[0][positive_mask])
        )
        boundary_hidden = torch.stack(
            (
                reuse_hidden[0, 0].expand(len(positive_ids), -1),
                recompute_hidden[0, 0].expand(len(positive_ids), -1),
            )
        )
        all_hidden = torch.cat((rolling_hidden, boundary_hidden), dim=0)
        maximum_targets = torch.tensor(len(positive_ids), dtype=torch.int64, device=device)
        dist.all_reduce(maximum_targets, op=dist.ReduceOp.MAX)
        nll_parts = []
        rank_parts = []
        for step in range(math.ceil(int(maximum_targets.item()) / target_chunk)):
            left = step * target_chunk
            real = min(target_chunk, max(0, len(positive_ids) - left))
            hidden = torch.zeros((4, target_chunk, spec.hidden_size), device=device)
            candidates = torch.zeros(target_chunk, dtype=torch.int64, device=device)
            if real:
                hidden[:, :real] = all_hidden[:, left : left + real]
                candidates[:real] = positive_ids[left : left + real]
            mode_nll = []
            mode_ranks = []
            for mode_index in range(len(MODES)):
                nll, ranks = distributed_full_catalog_metrics(
                    current_embedding,
                    hidden[mode_index * 2 : mode_index * 2 + 2],
                    candidates,
                    real,
                    num_prediction_items=spec.num_prediction_items,
                    item_chunk=item_chunk,
                )
                mode_nll.append(nll)
                mode_ranks.append(ranks)
            nll_parts.append(torch.cat(mode_nll, dim=0))
            rank_parts.append(torch.cat(mode_ranks, dim=0))
        nll_by_method = (
            torch.cat(nll_parts, dim=1)
            if nll_parts
            else torch.empty((4, 0), dtype=torch.float32, device=device)
        )
        ranks_by_method = (
            torch.cat(rank_parts, dim=1)
            if rank_parts
            else torch.empty((4, 0), dtype=torch.int64, device=device)
        )
        full = {mode: {} for mode in MODES}
        if participant:
            for mode_index, mode in enumerate(MODES):
                method_indices = [mode_index * 2, mode_index * 2 + 1]
                for cohort in COHORTS:
                    selected = torch.from_numpy(masks[cohort]).to(device)
                    full[mode][cohort] = summarize_full_catalog_record(
                        nll_by_method[method_indices][:, selected],
                        ranks_by_method[method_indices][:, selected],
                    )
        relation_union = (
            masks["successor_or_copositive"]
            if participant
            else np.zeros(0, dtype=np.bool_)
        )
        selected_indices = np.nonzero(relation_union)[0]
        candidate_payload = {mode: {} for mode in MODES}
        if len(selected_indices):
            selected_tensor = torch.from_numpy(selected_indices).to(device)
            selected_ids = positive_ids.index_select(0, selected_tensor)
            uniform = nested_uniform_candidate_ids(
                selected_ids,
                num_prediction_items=spec.num_prediction_items,
                maximum_negative_count=maximum_negative_count,
                seed=int(quality["candidate_seed"]) + record * 1_000_003,
            )
            hard, selected_count = context_hard_candidate_ids(
                selected_ids,
                relation["ranked_h32_targets"],
                popular_items,
                uniform,
                maximum_negative_count=maximum_negative_count,
            )
            context_selected += selected_count
            candidate_slots += len(selected_ids) * maximum_negative_count
            pools = torch.cat((uniform, hard), dim=1)
            selected_hidden = all_hidden.index_select(1, selected_tensor)
        else:
            pools = torch.empty((0, 2 * (maximum_negative_count + 1)), dtype=torch.int64)
            selected_hidden = all_hidden[:, :0]
        maximum_selected = torch.tensor(len(selected_indices), dtype=torch.int64, device=device)
        dist.all_reduce(maximum_selected, op=dist.ReduceOp.MAX)
        candidate_scores = []
        segment_width = maximum_negative_count + 1
        for step in range(math.ceil(int(maximum_selected.item()) / target_chunk)):
            left = step * target_chunk
            real = min(target_chunk, max(0, len(selected_indices) - left))
            hidden = torch.zeros((4, target_chunk, spec.hidden_size), device=device)
            padded = torch.zeros(
                (target_chunk, len(CANDIDATE_VARIANTS) * segment_width),
                dtype=torch.int64,
                device=device,
            )
            if real:
                hidden[:, :real] = selected_hidden[:, left : left + real]
                padded[:real] = pools[left : left + real].to(device)
            candidate_scores.append(
                distributed_projected_candidate_scores(
                    current_embedding, hidden, padded, real
                )
            )
        scores = (
            torch.cat(candidate_scores, dim=1)
            if candidate_scores
            else torch.empty((4, 0, len(CANDIDATE_VARIANTS) * segment_width), device=device)
        )
        if participant:
            relation_masks = {
                cohort: masks[cohort][relation_union] for cohort in RELATION_COHORTS
            }
            for mode_index, mode in enumerate(MODES):
                method_scores = scores[mode_index * 2 : mode_index * 2 + 2]
                for cohort in RELATION_COHORTS:
                    cohort_mask = torch.from_numpy(relation_masks[cohort]).to(device)
                    sums = np.zeros(
                        (
                            len(CANDIDATE_VARIANTS),
                            len(negative_counts),
                            2,
                            len(METRICS),
                        ),
                        dtype=np.float64,
                    )
                    for variant_index in range(len(CANDIDATE_VARIANTS)):
                        segment = variant_index * segment_width
                        for count_index, count in enumerate(negative_counts):
                            selected_scores = method_scores[
                                :, cohort_mask, segment : segment + count + 1
                            ]
                            sums[variant_index, count_index] = candidate_score_sums(
                                selected_scores
                            )
                    candidate_payload[mode][cohort] = {
                        "targets": int(np.count_nonzero(relation_masks[cohort])),
                        "sums": sums,
                    }
            local_payloads.append(
                {"record": record, "full": full, "candidates": candidate_payload}
            )
        del (
            current_vectors,
            old_cache,
            recompute_cache,
            suffix_vectors,
            reuse_hidden,
            recompute_hidden,
            all_hidden,
            nll_by_method,
            ranks_by_method,
            scores,
        )
        if processed == 1 or processed % progress_every == 0:
            print(
                f"phase=qk_update_relevance rank={rank} record={processed} source_ordinal={ordinal + 1}",
                flush=True,
            )
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local_payloads)
    counters = torch.tensor([context_selected, candidate_slots], dtype=torch.int64, device=device)
    dist.all_reduce(counters, op=dist.ReduceOp.SUM)
    if rank != 0:
        return {}
    combined = [value for piece in gathered for value in piece]
    combined.sort(key=lambda value: int(value["record"]))
    summary = _summarize_payloads(
        combined,
        bootstrap_samples=int(quality["bootstrap_samples"]),
        bootstrap_seed=int(quality["bootstrap_seed"]),
        negative_counts=negative_counts,
    )
    gate = _admission_gate(
        summary,
        candidates=list(quality["gate_candidates"]),
        minimum_targets=int(quality["minimum_cohort_targets"]),
        relative_range=list(quality["preferred_relative_gap_percent_range"]),
    )
    return {
        "records": len(combined),
        "runtime_seconds": time.perf_counter() - started,
        "summary": summary,
        "primary_admission_gate": gate,
        "candidate_construction": {
            "context_successor_selected_negative_slots": int(counters[0].item()),
            "total_negative_slots": int(counters[1].item()),
            "context_successor_fraction": (
                float(counters[0].item() / counters[1].item())
                if counters[1].item()
                else 0.0
            ),
        },
    }


def run_qk_update_relevance_evaluation(
    config_path: Path,
) -> dict[str, object] | None:
    document = json.loads(config_path.read_text())
    _validate_document(document)
    rank, world_size, local_rank, device = _runtime()
    started = time.perf_counter()
    try:
        output = Path(document["outputs"]["result"])
        if output.exists():
            raise FileExistsError("QK update relevance result already exists")
        corpus = load_corpus(document["data"]["corpus"])
        if corpus.file_sha256 != document["data"]["corpus_sha256"]:
            raise ValueError("QK update relevance corpus hash differs")
        audit = prequential_evaluation_role_audit(corpus, 2)
        source = document["source_checkpoint"]
        current = document["current_checkpoint"]
        source_root = Path(source["root"])
        current_root = Path(current["root"])
        source_manifest = source_root / "theta_1" / "manifest.json"
        current_manifest = current_root / "theta_2" / "manifest.json"
        if (
            file_sha256(source_manifest) != source["manifest_sha256"]
            or file_sha256(current_manifest) != current["manifest_sha256"]
        ):
            raise ValueError("QK update relevance checkpoint hash differs")
        spec, source_dense, source_embedding, _, _ = _load_model(
            source_root, 1, rank=rank, world_size=world_size, device=device
        )
        source_dense_state = _dense_state(source_dense)
        snapshots = snapshot_source_prefixes(
            corpus,
            source_embedding,
            ("stream_train",),
            2,
            rank,
            world_size,
            device,
            int(document["execution"]["snapshot_batch_size_per_rank"]),
        )
        del source_dense, source_embedding
        gc.collect()
        torch.cuda.empty_cache()
        spec, current_dense, current_embedding, _, _ = _load_model(
            current_root, 2, rank=rank, world_size=world_size, device=device
        )
        source_dense = ExternalEmbeddingHSTU(spec.hstu_config()).to(device)
        source_dense.load_state_dict(source_dense_state)
        source_dense.eval()
        current_dense.eval()
        current_embedding.eval()
        successor, copositive, popular, relation_metadata = build_training_relations(
            corpus,
            edge=2,
            transition_horizon=int(document["quality"]["transition_horizon"]),
            num_prediction_items=spec.num_prediction_items,
        )
        participants = set(eligible_training_records(corpus, 2).tolist())
        evaluation = _evaluate(
            document,
            corpus,
            spec,
            current_dense,
            current_embedding,
            source_dense,
            snapshots["stream_train"],
            successor,
            copositive,
            popular,
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
            "candidate": document["candidate"],
            "role_audit": audit,
            "source_checkpoint": source,
            "current_checkpoint": current,
            "training_relations": relation_metadata,
            "quality": {
                "same_current_model": 2,
                "reuse_source_version": 1,
                "recompute_version": 2,
                "evaluation": evaluation,
                "evaluation_targets_used_for_training": False,
                "evaluation_targets_used_for_relation_construction": False,
                "evaluation_target_identity_used_only_to_apply_predeclared_cohort": True,
                "qualification_consumed": False,
                "final_consumed": False,
            },
            "execution": {
                "world_size": world_size,
                "local_rank": local_rank,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "record_limit_per_rank": document["execution"]["record_limit_per_rank"],
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
