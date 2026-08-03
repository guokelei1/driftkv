from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..data.qb_large_multifield import (
    PROTOCOL as DATA_PROTOCOL,
)
from ..data.qb_large_multifield import (
    QBLargeCatalog,
    artifact_sha256,
    file_sha256,
    load_catalog,
)
from ..models import HSTUKVCache
from .multifield_projected import lookup_multifield_projected
from .sharded_edge import ExternalEmbeddingHSTU, fixed_candidate_ids
from .trainer import build_next_item_targets
from .xp_projected_edge import (
    OptimizerActiveRowTracker,
    TrainableProjectedModuloEmbedding,
    tracked_sparse_optimizer_step,
)

PROTOCOL = "evokv_qb_large_multifield_stream_development_v0"
ROLE_CODES = {"train": 0, "tuning": 1, "qualification": 2}


@dataclass(frozen=True)
class QBLargeCorpus:
    path: Path
    catalog_path: Path
    arrays: dict[str, np.ndarray]
    metadata: dict[str, object]
    catalog: QBLargeCatalog
    file_sha256: str
    content_sha256: str

    def role_records(self, role: str) -> np.ndarray:
        if role not in ROLE_CODES:
            raise ValueError("QB role is invalid")
        return np.flatnonzero(self.arrays["role_record_role"] == ROLE_CODES[role])


@dataclass(frozen=True)
class QBEvaluationBatch:
    batch: dict[str, torch.Tensor]
    candidates: torch.Tensor
    candidate_sha256: str


def _validate_record_arrays(
    arrays: dict[str, np.ndarray],
    prefix: str,
    fields: int,
) -> None:
    offsets = arrays[f"{prefix}_record_offsets"]
    records = len(arrays[f"{prefix}_record_user_ids"])
    events = len(arrays[f"{prefix}_target_item_ids"])
    if (
        offsets.dtype != np.int64
        or offsets.shape != (records + 1,)
        or offsets[0] != 0
        or offsets[-1] != events
        or np.any(offsets[1:] <= offsets[:-1])
        or arrays[f"{prefix}_feature_ids"].shape != (events, fields)
        or arrays[f"{prefix}_feature_ids"].dtype != np.uint32
        or arrays[f"{prefix}_target_item_ids"].dtype != np.uint32
        or arrays[f"{prefix}_behavior"].dtype != np.uint8
        or arrays[f"{prefix}_label"].dtype != np.uint8
        or arrays[f"{prefix}_raw_ordinal"].dtype != np.uint16
    ):
        raise ValueError("QB corpus record layout differs")
    if any(
        arrays[f"{prefix}_{name}"].shape != (events,)
        for name in (
            "target_item_ids",
            "behavior",
            "raw_label",
            "label",
            "raw_ordinal",
            "is_prediction_item",
        )
    ):
        raise ValueError("QB corpus event layout differs")


def load_qb_large_corpus(
    path: str | Path,
    catalog_path: str | Path,
) -> QBLargeCorpus:
    resolved = Path(path)
    resolved_catalog = Path(catalog_path)
    with np.load(resolved, allow_pickle=False) as source:
        if "metadata_json" not in source.files:
            raise ValueError("QB corpus metadata is absent")
        arrays = {name: source[name].copy() for name in source.files if name != "metadata_json"}
        metadata = json.loads(str(source["metadata_json"].item()))
    catalog = load_catalog(resolved_catalog)
    content_hash = artifact_sha256(arrays)
    if (
        metadata.get("protocol") != DATA_PROTOCOL
        or metadata.get("dataset") != "tenrec-qb"
        or metadata.get("profile") != catalog.profile.name
        or metadata.get("feature_fields") != list(catalog.profile.fields)
        or int(metadata.get("embedding_width", -1)) != catalog.profile.embedding_width
        or int(metadata.get("num_embeddings", -1)) != catalog.num_embeddings
        or int(metadata.get("num_prediction_items", -1)) != catalog.num_prediction_items
        or metadata.get("catalog_content_sha256") != catalog.metadata.get("content_sha256")
        or metadata.get("content_sha256") != content_hash
        or metadata.get("roles_pairwise_disjoint") is not True
    ):
        raise ValueError("QB corpus binding differs")
    _validate_record_arrays(arrays, "base", catalog.profile.feature_count)
    _validate_record_arrays(arrays, "role", catalog.profile.feature_count)
    roles = arrays["role_record_role"]
    if (
        roles.dtype != np.uint8
        or roles.shape != arrays["role_record_user_ids"].shape
        or np.any(roles >= len(ROLE_CODES))
        or np.any(arrays["base_feature_ids"] >= catalog.num_embeddings)
        or np.any(arrays["role_feature_ids"] >= catalog.num_embeddings)
        or np.any(arrays["base_target_item_ids"] > catalog.num_prediction_items)
        or np.any(arrays["role_target_item_ids"] > catalog.num_prediction_items)
        or np.any(arrays["base_behavior"] > 5)
        or np.any(arrays["role_behavior"] > 5)
    ):
        raise ValueError("QB corpus values differ")
    anchors = arrays["cooccurrence_anchor_row"]
    positives = arrays["cooccurrence_positive_row"]
    users = arrays["cooccurrence_occurrence_user_id"]
    expected = np.arange(1, catalog.num_embeddings, dtype=np.uint32)
    if (
        anchors.dtype != np.uint32
        or positives.dtype != np.uint32
        or users.dtype != np.int64
        or not np.array_equal(anchors, expected)
        or positives.shape != anchors.shape
        or users.shape != anchors.shape
        or np.any(positives < 1)
        or np.any(positives >= catalog.num_embeddings)
        or np.any(positives == anchors)
    ):
        raise ValueError("QB cooccurrence coverage differs")
    return QBLargeCorpus(
        path=resolved,
        catalog_path=resolved_catalog,
        arrays=arrays,
        metadata=metadata,
        catalog=catalog,
        file_sha256=file_sha256(resolved),
        content_sha256=content_hash,
    )


def cooccurrence_negatives(corpus: QBLargeCorpus) -> np.ndarray:
    anchors = corpus.arrays["cooccurrence_anchor_row"]
    positives = corpus.arrays["cooccurrence_positive_row"]
    users = corpus.arrays["cooccurrence_occurrence_user_id"]
    count = len(anchors)
    indices = (np.arange(count, dtype=np.int64) + count // 2 + 1) % count
    invalid = (
        (users[indices] == users)
        | (positives[indices] == anchors)
        | (positives[indices] == positives)
    )
    attempts = 0
    while np.any(invalid) and attempts < 64:
        indices[invalid] = (indices[invalid] + 104_729) % count
        invalid = (
            (users[indices] == users)
            | (positives[indices] == anchors)
            | (positives[indices] == positives)
        )
        attempts += 1
    if np.any(invalid):
        raise RuntimeError("QB cooccurrence negative assignment failed")
    return positives[indices].astype(np.uint32, copy=False)


def _record_has_target(
    arrays: dict[str, np.ndarray],
    prefix: str,
    record: int,
    train_start: int,
    train_end: int,
) -> bool:
    offset = int(arrays[f"{prefix}_record_offsets"][record])
    stop = int(arrays[f"{prefix}_record_offsets"][record + 1])
    labels = arrays[f"{prefix}_label"][
        min(offset + train_start, stop) : min(offset + train_end, stop)
    ]
    return bool(np.any(labels > 0))


def _batch_records(
    corpus: QBLargeCorpus,
    *,
    prefix: str,
    records: np.ndarray,
    width: int,
    train_start: int,
    train_end: int,
    batch_size_per_rank: int,
    rank: int,
    world_size: int,
    maximum_steps: int = 0,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, object]]:
    if (
        prefix not in {"base", "role"}
        or width < 2
        or not 1 <= train_start < train_end <= width
        or batch_size_per_rank < 1
        or world_size < 1
        or not 0 <= rank < world_size
        or maximum_steps < 0
    ):
        raise ValueError("QB batch request differs")
    arrays = corpus.arrays
    offsets = arrays[f"{prefix}_record_offsets"]
    eligible = np.asarray(
        [
            int(record)
            for record in records
            if int(offsets[record + 1] - offsets[record]) >= 2
            and _record_has_target(
                arrays,
                prefix,
                int(record),
                train_start,
                train_end,
            )
        ],
        dtype=np.int64,
    )
    if len(eligible) == 0:
        raise RuntimeError("QB batch role has no positive target")
    global_batch = batch_size_per_rank * world_size
    steps = math.ceil(len(eligible) / global_batch)
    if maximum_steps:
        steps = min(steps, maximum_steps)
    fields = corpus.catalog.profile.feature_count
    batches = []
    local_real = 0
    local_targets = 0
    for step in range(steps):
        left = step * global_batch + rank * batch_size_per_rank
        selected = eligible[left : left + batch_size_per_rank]
        features = np.zeros((batch_size_per_rank, width, fields), dtype=np.int64)
        targets = np.zeros((batch_size_per_rank, width), dtype=np.int64)
        behaviors = np.zeros((batch_size_per_rank, width), dtype=np.int64)
        deltas = np.zeros((batch_size_per_rank, width), dtype=np.float32)
        labels = np.zeros((batch_size_per_rank, width), dtype=np.int64)
        train_mask = np.zeros((batch_size_per_rank, width), dtype=np.bool_)
        lengths = np.zeros(batch_size_per_rank, dtype=np.int64)
        record_indices = np.full(batch_size_per_rank, -1, dtype=np.int64)
        for row, record in enumerate(selected.tolist()):
            start = int(offsets[record])
            stop = min(start + width, int(offsets[record + 1]))
            length = stop - start
            features[row, :length] = arrays[f"{prefix}_feature_ids"][start:stop]
            targets[row, :length] = arrays[f"{prefix}_target_item_ids"][start:stop]
            behaviors[row, :length] = arrays[f"{prefix}_behavior"][start:stop]
            ordinals = arrays[f"{prefix}_raw_ordinal"][start:stop].astype(np.float32, copy=False)
            deltas[row, 1:length] = np.diff(ordinals)
            labels[row, :length] = arrays[f"{prefix}_label"][start:stop]
            train_mask[row, train_start : min(train_end, length)] = True
            lengths[row] = length
            record_indices[row] = record
        batch = {
            "feature_ids": torch.from_numpy(features),
            "target_item_ids": torch.from_numpy(targets),
            "behaviors": torch.from_numpy(behaviors),
            "time_deltas": torch.from_numpy(deltas),
            "labels": torch.from_numpy(labels),
            "train_mask": torch.from_numpy(train_mask),
            "lengths": torch.from_numpy(lengths),
            "record_indices": torch.from_numpy(record_indices),
        }
        _, valid = build_next_item_targets(
            batch["target_item_ids"],
            batch["lengths"],
            batch["labels"],
            batch["train_mask"],
        )
        local_real += len(selected)
        local_targets += int(valid.sum().item())
        batches.append(batch)
    return batches, {
        "prefix": prefix,
        "global_records": len(records),
        "global_eligible_records": len(eligible),
        "steps_per_rank": steps,
        "batch_size_per_rank": batch_size_per_rank,
        "local_real_records": local_real,
        "local_padding_records": steps * batch_size_per_rank - local_real,
        "local_positive_targets": local_targets,
        "width": width,
        "train_start": train_start,
        "train_end": train_end,
        "maximum_steps": maximum_steps,
    }


def build_base_batches(
    corpus: QBLargeCorpus,
    *,
    batch_size_per_rank: int,
    rank: int,
    world_size: int,
    maximum_steps: int = 0,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, object]]:
    records = np.arange(len(corpus.arrays["base_record_user_ids"]), dtype=np.int64)
    width = int(corpus.metadata["base_prefix"])
    return _batch_records(
        corpus,
        prefix="base",
        records=records,
        width=width,
        train_start=1,
        train_end=width,
        batch_size_per_rank=batch_size_per_rank,
        rank=rank,
        world_size=world_size,
        maximum_steps=maximum_steps,
    )


def build_role_batches(
    corpus: QBLargeCorpus,
    role: str,
    *,
    width: int,
    train_start: int,
    train_end: int,
    batch_size_per_rank: int,
    rank: int,
    world_size: int,
    maximum_steps: int = 0,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, object]]:
    return _batch_records(
        corpus,
        prefix="role",
        records=corpus.role_records(role),
        width=width,
        train_start=train_start,
        train_end=train_end,
        batch_size_per_rank=batch_size_per_rank,
        rank=rank,
        world_size=world_size,
        maximum_steps=maximum_steps,
    )


def prepare_evaluation_batches(
    batches: Iterable[dict[str, torch.Tensor]],
    *,
    num_prediction_items: int,
    negative_count: int,
    seed: int,
    rank: int,
    world_size: int,
) -> tuple[list[QBEvaluationBatch], str]:
    result = []
    digest = hashlib.sha256()
    for index, batch in enumerate(batches):
        targets, valid = build_next_item_targets(
            batch["target_item_ids"],
            batch["lengths"],
            batch["labels"],
            batch["train_mask"],
        )
        candidates = fixed_candidate_ids(
            targets[valid],
            num_prediction_items,
            negative_count,
            seed + index * world_size + rank,
        ).cpu()
        payload = candidates.contiguous().numpy().astype("<i8", copy=False)
        local_hash = hashlib.sha256(payload.tobytes()).hexdigest()
        digest.update(local_hash.encode())
        result.append(
            QBEvaluationBatch(
                batch=batch,
                candidates=candidates,
                candidate_sha256=local_hash,
            )
        )
    return result, digest.hexdigest()


def _all_reduce_dense_gradients(
    dense_model: nn.Module,
    process_group: dist.ProcessGroup | None,
) -> None:
    for parameter in dense_model.parameters():
        if parameter.grad is None or not bool(torch.all(torch.isfinite(parameter.grad))):
            raise RuntimeError("QB dense gradient is absent or nonfinite")
        if dist.is_initialized():
            dist.all_reduce(parameter.grad, group=process_group)


def train_multifield_step(
    dense_model: nn.Module,
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
    batch: dict[str, torch.Tensor],
    dense_optimizer: torch.optim.Optimizer,
    projection_optimizer: torch.optim.Optimizer,
    embedding_optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    num_prediction_items: int,
    negative_count: int,
    negative_seed: int,
    process_group: dist.ProcessGroup | None = None,
) -> tuple[float, int, int]:
    dense_model.train()
    embedding.train()
    features = batch["feature_ids"].to(device)
    targets = batch["target_item_ids"].to(device)
    behaviors = batch["behaviors"].to(device)
    deltas = batch["time_deltas"].to(device)
    lengths = batch["lengths"].to(device)
    labels = batch["labels"].to(device)
    train_mask = batch["train_mask"].to(device)
    dense_optimizer.zero_grad(set_to_none=True)
    projection_optimizer.zero_grad(set_to_none=True)
    embedding_optimizer.zero_grad(set_to_none=True)
    vectors = lookup_multifield_projected(embedding, features, lengths)
    hidden = dense_model(vectors, behaviors, deltas, lengths)
    next_items, valid = build_next_item_targets(targets, lengths, labels, train_mask)
    positive_ids = next_items[valid]
    candidates = fixed_candidate_ids(
        positive_ids,
        num_prediction_items,
        negative_count,
        negative_seed,
    )
    candidate_lengths = torch.full(
        (len(candidates),),
        candidates.shape[1],
        dtype=torch.int64,
        device=device,
    )
    candidate_vectors = embedding(candidates, candidate_lengths)
    if len(positive_ids):
        logits = torch.einsum("nh,nch->nc", hidden[:, :-1][valid], candidate_vectors)
        loss = F.cross_entropy(
            logits,
            torch.zeros(len(positive_ids), dtype=torch.int64, device=device),
        )
    else:
        loss = hidden.sum() * 0.0 + candidate_vectors.sum() * 0.0
    local_targets = torch.tensor(len(positive_ids), dtype=torch.float64, device=device)
    global_targets = local_targets.clone()
    if dist.is_initialized():
        dist.all_reduce(global_targets, group=process_group)
    scale = (
        local_targets / global_targets
        if global_targets.item() > 0
        else torch.zeros_like(global_targets)
    )
    (loss * scale.to(loss.dtype)).backward()
    _all_reduce_dense_gradients(dense_model, process_group)
    projection_gradient = embedding.projection_weight.grad
    if projection_gradient is None or not bool(torch.all(torch.isfinite(projection_gradient))):
        raise RuntimeError("QB projection gradient is absent or nonfinite")
    torch.nn.utils.clip_grad_norm_(dense_model.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_([embedding.projection_weight], 1.0)
    dense_optimizer.step()
    projection_optimizer.step()
    tracked_sparse_optimizer_step(embedding, embedding_optimizer, tracker)
    return float(loss.detach().item()), len(positive_ids), int(global_targets.item())


def train_cooccurrence_step(
    embedding: TrainableProjectedModuloEmbedding,
    tracker: OptimizerActiveRowTracker,
    anchor: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    projection_optimizer: torch.optim.Optimizer,
    embedding_optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    temperature: float,
    process_group: dist.ProcessGroup | None = None,
) -> tuple[float, int]:
    ids = torch.from_numpy(np.stack((anchor, positive, negative), axis=1).astype(np.int64)).to(
        device
    )
    lengths = torch.full((len(ids),), 3, dtype=torch.int64, device=device)
    projection_optimizer.zero_grad(set_to_none=True)
    embedding_optimizer.zero_grad(set_to_none=True)
    vectors = F.normalize(embedding(ids, lengths), dim=-1)
    logits = (
        torch.stack(
            (
                (vectors[:, 0] * vectors[:, 1]).sum(dim=-1),
                (vectors[:, 0] * vectors[:, 2]).sum(dim=-1),
            ),
            dim=1,
        )
        / temperature
    )
    local_count = torch.tensor(len(ids), dtype=torch.float64, device=device)
    global_count = local_count.clone()
    if dist.is_initialized():
        dist.all_reduce(global_count, group=process_group)
    loss_sum = F.cross_entropy(
        logits,
        torch.zeros(len(ids), dtype=torch.int64, device=device),
        reduction="sum",
    )
    loss = loss_sum / global_count.to(loss_sum.dtype)
    loss.backward()
    projection_gradient = embedding.projection_weight.grad
    if projection_gradient is None or not bool(torch.all(torch.isfinite(projection_gradient))):
        raise RuntimeError("QB contrastive projection gradient is nonfinite")
    torch.nn.utils.clip_grad_norm_([embedding.projection_weight], 1.0)
    projection_optimizer.step()
    tracked_sparse_optimizer_step(embedding, embedding_optimizer, tracker)
    return float(loss_sum.detach().item()), int(global_count.item())


def _float_cache(cache: HSTUKVCache, device: torch.device) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k.to(device=device, dtype=torch.float32),
        v=cache.v.to(device=device, dtype=torch.float32),
        seq_len=cache.seq_len,
    )


def _stored_cache(cache: HSTUKVCache) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k.detach().to(device="cpu", dtype=torch.float16),
        v=cache.v.detach().to(device="cpu", dtype=torch.float16),
        seq_len=cache.seq_len,
    )


@torch.no_grad()
def prefix_cache(
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    batch: dict[str, torch.Tensor],
    history_end: int,
    device: torch.device,
) -> HSTUKVCache:
    prefix_width = history_end - 1
    records = batch["record_indices"].to(device)
    lengths = torch.where(
        records >= 0,
        torch.full_like(records, prefix_width),
        torch.zeros_like(records),
    )
    features = batch["feature_ids"][:, :prefix_width].to(device)
    vectors = lookup_multifield_projected(embedding, features, lengths)
    cache = dense.core.compute_kv_from_item_embeddings(
        vectors,
        batch["behaviors"][:, :prefix_width].to(device),
        batch["time_deltas"][:, :prefix_width].to(device),
        lengths,
    )
    return _stored_cache(cache)


@torch.no_grad()
def scores_with_cache(
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    value: QBEvaluationBatch,
    cache: HSTUKVCache,
    history_end: int,
    device: torch.device,
) -> torch.Tensor:
    batch = value.batch
    prefix_width = history_end - 1
    records = batch["record_indices"].to(device)
    real = records >= 0
    suffix_features = batch["feature_ids"][:, prefix_width:-1].to(device)
    suffix_lengths = torch.where(
        real,
        torch.full_like(records, suffix_features.shape[1]),
        torch.zeros_like(records),
    )
    suffix_vectors = lookup_multifield_projected(embedding, suffix_features, suffix_lengths)
    hidden, _ = dense.core.forward_with_cache_from_item_embeddings(
        _float_cache(cache, device),
        suffix_vectors,
        batch["behaviors"][:, prefix_width:-1].to(device),
        batch["time_deltas"][:, prefix_width:-1].to(device),
    )
    targets, valid = build_next_item_targets(
        batch["target_item_ids"].to(device),
        batch["lengths"].to(device),
        batch["labels"].to(device),
        batch["train_mask"].to(device),
    )
    suffix_valid = valid[:, prefix_width:]
    positive_count = int(suffix_valid.sum().item())
    candidates = value.candidates.to(device)
    if candidates.shape[0] != positive_count:
        raise ValueError("QB evaluation candidates differ")
    candidate_lengths = torch.full(
        (positive_count,),
        candidates.shape[1],
        dtype=torch.int64,
        device=device,
    )
    candidate_vectors = embedding(candidates, candidate_lengths)
    return torch.einsum("nh,nch->nc", hidden[suffix_valid], candidate_vectors)


def score_sums(
    scores: torch.Tensor,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    count = len(scores)
    if count == 0:
        return torch.zeros(8, dtype=torch.float64, device=scores.device)
    positive = scores[:, :1]
    ranks = 1 + (scores[:, 1:] >= positive).sum(dim=1)
    hit = ranks <= 10
    ndcg = torch.where(
        hit,
        torch.reciprocal(torch.log2(ranks.double() + 1.0)),
        torch.zeros_like(ranks, dtype=torch.float64),
    )
    loss = F.cross_entropy(
        scores,
        torch.zeros(count, dtype=torch.int64, device=scores.device),
        reduction="sum",
    )
    if reference is None:
        cosine = torch.zeros((), dtype=torch.float64, device=scores.device)
        overlap = torch.zeros((), dtype=torch.float64, device=scores.device)
        agreement = torch.zeros((), dtype=torch.float64, device=scores.device)
    else:
        cosine = F.cosine_similarity(scores, reference, dim=1).double().sum()
        top = min(10, scores.shape[1])
        actual_top = torch.topk(scores, top, dim=1).indices
        reference_top = torch.topk(reference, top, dim=1).indices
        overlap = (actual_top[:, :, None] == reference_top[:, None, :]).any(
            dim=2
        ).double().sum() / top
        agreement = (torch.argmax(scores, dim=1) == torch.argmax(reference, dim=1)).double().sum()
    return torch.stack(
        (
            torch.tensor(float(count), dtype=torch.float64, device=scores.device),
            loss.double(),
            hit.double().sum(),
            ndcg.sum(),
            torch.reciprocal(ranks.double()).sum(),
            cosine,
            overlap,
            agreement,
        )
    )


def summarize_score_sums(values: torch.Tensor, *, relative: bool) -> dict[str, object]:
    count = int(values[0].item())
    if count < 1:
        raise RuntimeError("QB evaluation has no positive target")
    result: dict[str, object] = {
        "positive_targets": count,
        "sampled_cross_entropy": float((values[1] / count).item()),
        "hit_rate_at_10": float((values[2] / count).item()),
        "ndcg_at_10": float((values[3] / count).item()),
        "mean_reciprocal_rank": float((values[4] / count).item()),
    }
    if relative:
        result.update(
            {
                "score_cosine_to_exact": float((values[5] / count).item()),
                "top10_overlap_with_exact": float((values[6] / count).item()),
                "top1_agreement_with_exact": float((values[7] / count).item()),
            }
        )
    return result
