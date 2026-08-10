from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..data.qk_stream_chain import QKStreamChainCorpus
from ..models import HSTUKVCache
from .sharded_edge import fixed_candidate_ids
from .trainer import build_next_item_targets
from .xp_projected_edge import TrainableProjectedModuloEmbedding

PROTOCOL = "evokv_qk_stream_version_training_quality_v0"
FULL_CATALOG_PROTOCOL = "evokv_qk_stream_full_catalog_tuning_v1"
UPDATE_LOCAL_EVALUATION_ROLE = "stream_train"
DISJOINT_SUPPLEMENT_EVALUATION_ROLE = "fit_tuning"


@dataclass(frozen=True)
class QKStreamEdgeBinding:
    source_version: int
    target_version: int
    edge: int
    candidate_name: str
    dense_learning_rate: float
    projection_learning_rate: float
    embedding_learning_rate: float
    epochs: int
    train_negative_count: int
    quality_negative_count: int
    quality_epsilon_ce: float
    bootstrap_samples: int
    training_seed: int
    negative_seed: int
    quality_seed: int
    bootstrap_seed: int

    def __post_init__(self) -> None:
        if (
            self.source_version < 0
            or self.target_version != self.source_version + 1
            or self.edge != self.target_version
            or not self.candidate_name
            or min(
                self.dense_learning_rate,
                self.projection_learning_rate,
                self.embedding_learning_rate,
            )
            <= 0
            or min(
                self.epochs,
                self.train_negative_count,
                self.quality_negative_count,
                self.bootstrap_samples,
            )
            < 1
            or self.quality_epsilon_ce <= 0
            or min(
                self.training_seed,
                self.negative_seed,
                self.quality_seed,
                self.bootstrap_seed,
            )
            < 0
        ):
            raise ValueError("QK stream edge binding differs")


@dataclass(frozen=True)
class QKStreamFullCatalogBinding:
    source_version: int
    target_version: int
    edge: int
    candidate_name: str
    dense_learning_rate: float
    projection_learning_rate: float
    embedding_learning_rate: float
    epochs: int
    train_negative_count: int
    bootstrap_samples: int
    training_seed: int
    negative_seed: int
    bootstrap_seed: int
    full_catalog_item_chunk: int

    def __post_init__(self) -> None:
        if (
            self.source_version < 0
            or self.target_version != self.source_version + 1
            or self.edge != self.target_version
            or not self.candidate_name
            or min(
                self.dense_learning_rate,
                self.projection_learning_rate,
                self.embedding_learning_rate,
            )
            <= 0
            or min(
                self.epochs,
                self.train_negative_count,
                self.bootstrap_samples,
                self.full_catalog_item_chunk,
            )
            < 1
            or min(
                self.training_seed,
                self.negative_seed,
                self.bootstrap_seed,
            )
            < 0
        ):
            raise ValueError("QK stream full-catalog binding differs")


def validate_full_catalog_binding_document(
    document: dict[str, object],
    binding: QKStreamFullCatalogBinding,
) -> None:
    edge = document.get("edge")
    training = document.get("training")
    quality = document.get("quality")
    if (
        document.get("protocol") != FULL_CATALOG_PROTOCOL
        or document.get("status") != "ready_for_user_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not isinstance(edge, dict)
        or not isinstance(training, dict)
        or not isinstance(quality, dict)
        or edge
        != {
            "source_version": binding.source_version,
            "target_version": binding.target_version,
            "edge": binding.edge,
            "candidate_name": binding.candidate_name,
        }
        or training.get("dense_learning_rate")
        != binding.dense_learning_rate
        or training.get("projection_learning_rate")
        != binding.projection_learning_rate
        or training.get("embedding_learning_rate")
        != binding.embedding_learning_rate
        or training.get("epochs") != binding.epochs
        or training.get("negative_count") != binding.train_negative_count
        or training.get("seed") != binding.training_seed
        or training.get("negative_seed") != binding.negative_seed
        or quality.get("evaluation_role") != "fit_tuning"
        or quality.get("candidate_set") != "all_prediction_items"
        or quality.get("bootstrap_samples") != binding.bootstrap_samples
        or quality.get("bootstrap_seed") != binding.bootstrap_seed
        or quality.get("full_catalog_item_chunk")
        != binding.full_catalog_item_chunk
        or quality.get("decision_boundary") != "manual_after_tuning"
    ):
        raise ValueError(
            "QK stream full-catalog config differs from its entry point"
        )


def validate_binding_document(
    document: dict[str, object],
    binding: QKStreamEdgeBinding,
) -> None:
    edge = document.get("edge")
    training = document.get("training")
    quality = document.get("quality")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_user_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not isinstance(edge, dict)
        or not isinstance(training, dict)
        or not isinstance(quality, dict)
        or edge
        != {
            "source_version": binding.source_version,
            "target_version": binding.target_version,
            "edge": binding.edge,
            "candidate_name": binding.candidate_name,
        }
        or training.get("dense_learning_rate")
        != binding.dense_learning_rate
        or training.get("projection_learning_rate")
        != binding.projection_learning_rate
        or training.get("embedding_learning_rate")
        != binding.embedding_learning_rate
        or training.get("epochs") != binding.epochs
        or training.get("negative_count") != binding.train_negative_count
        or quality.get("negative_count") != binding.quality_negative_count
        or quality.get("epsilon_ce") != binding.quality_epsilon_ce
        or quality.get("bootstrap_samples") != binding.bootstrap_samples
        or training.get("seed") != binding.training_seed
        or training.get("negative_seed") != binding.negative_seed
        or quality.get("candidate_seed") != binding.quality_seed
        or quality.get("bootstrap_seed") != binding.bootstrap_seed
    ):
        raise ValueError("QK stream edge config differs from its entry point")


def record_window(
    corpus: QKStreamChainCorpus,
    record: int,
    edge: int,
) -> tuple[int, int, int]:
    boundaries = corpus.arrays["edge_last_ordinals"]
    if (
        edge < 1
        or edge + 1 >= boundaries.shape[1]
        or not 0 <= record < len(boundaries)
    ):
        raise ValueError("QK stream record window differs")
    previous = int(boundaries[record, edge - 1])
    current = int(boundaries[record, edge])
    following = int(boundaries[record, edge + 1])
    if not previous < current < following:
        raise ValueError("QK stream record boundaries differ")
    return previous, current, following


def eligible_training_records(
    corpus: QKStreamChainCorpus,
    edge: int,
) -> np.ndarray:
    result = []
    offsets = corpus.arrays["record_offsets"]
    labels = corpus.arrays["label"]
    for raw_record in corpus.role_records("stream_train"):
        record = int(raw_record)
        previous, current, _ = record_window(corpus, record, edge)
        start = int(offsets[record])
        if np.any(labels[start + previous + 1 : start + current + 1] > 0):
            result.append(record)
    if not result:
        raise RuntimeError("QK stream update has no training targets")
    return np.asarray(result, dtype=np.int64)


def prequential_evaluation_role_audit(
    corpus: QKStreamChainCorpus,
    edge: int,
) -> dict[str, object]:
    update_local = corpus.role_records(UPDATE_LOCAL_EVALUATION_ROLE)
    participants = eligible_training_records(corpus, edge)
    supplemental = corpus.role_records(DISJOINT_SUPPLEMENT_EVALUATION_ROLE)
    users = corpus.arrays["record_user_ids"]
    update_users = users[update_local]
    participant_users = users[participants]
    supplemental_users = users[supplemental]
    if (
        len(np.unique(update_users)) != len(update_users)
        or len(np.unique(participant_users)) != len(participant_users)
        or len(np.unique(supplemental_users)) != len(supplemental_users)
        or np.intersect1d(update_users, supplemental_users).size
        or not np.all(np.isin(participants, update_local))
    ):
        raise ValueError("QK prequential evaluation roles overlap")
    return {
        "primary_role": UPDATE_LOCAL_EVALUATION_ROLE,
        "primary_semantics": "same users, train window t then evaluate unseen window t+1",
        "primary_role_users": len(update_local),
        "optimizer_participant_users": len(participants),
        "optimizer_participant_fraction": len(participants)
        / len(update_local),
        "supplemental_role": DISJOINT_SUPPLEMENT_EVALUATION_ROLE,
        "supplemental_semantics": "user-disjoint next-window confirmation",
        "supplemental_users": len(supplemental),
        "supplemental_to_primary_fraction": len(supplemental)
        / len(update_local),
        "user_overlap": 0,
        "training_window": edge,
        "evaluation_window": edge + 1,
        "evaluation_targets_used_for_current_training": False,
    }


def training_record_order(
    corpus: QKStreamChainCorpus,
    edge: int,
    seed: int,
    epoch: int,
    bucket_records: int,
) -> np.ndarray:
    records = eligible_training_records(corpus, edge)
    if bucket_records < 1:
        raise ValueError("QK stream training bucket differs")
    generator = np.random.default_rng(seed + epoch * 1_000_003)
    generator.shuffle(records)
    boundaries = corpus.arrays["edge_last_ordinals"]
    pieces = []
    for start in range(0, len(records), bucket_records):
        block = records[start : start + bucket_records]
        lengths = boundaries[block, edge] + 1
        pieces.append(block[np.argsort(lengths, kind="stable")])
    return np.concatenate(pieces)


def _materialize_training_record(
    corpus: QKStreamChainCorpus,
    record: int,
    edge: int,
) -> dict[str, torch.Tensor]:
    previous, current, _ = record_window(corpus, record, edge)
    offset = int(corpus.arrays["record_offsets"][record])
    stop = current + 1
    items = corpus.arrays["item_idx"][offset : offset + stop]
    behaviors = corpus.arrays["behavior"][offset : offset + stop]
    labels = corpus.arrays["label"][offset : offset + stop]
    mask = np.zeros(stop, dtype=np.bool_)
    mask[previous + 1 : current + 1] = True
    deltas = np.ones(stop, dtype=np.float32)
    deltas[0] = 0.0
    return {
        "item_ids": torch.from_numpy(items.astype(np.int64, copy=True)),
        "behaviors": torch.from_numpy(
            behaviors.astype(np.int64, copy=True)
        ),
        "time_deltas": torch.from_numpy(deltas),
        "labels": torch.from_numpy(labels.astype(np.int64, copy=True)),
        "train_mask": torch.from_numpy(mask),
        "length": torch.tensor(stop, dtype=torch.int64),
        "record": torch.tensor(record, dtype=torch.int64),
    }


def build_training_batch(
    corpus: QKStreamChainCorpus,
    records: np.ndarray,
    batch_size: int,
    edge: int,
) -> dict[str, torch.Tensor]:
    if records.ndim != 1 or len(records) > batch_size or batch_size < 1:
        raise ValueError("QK stream training batch records differ")
    values = [
        _materialize_training_record(corpus, int(record), edge)
        for record in records
    ]
    width = max(
        [int(value["length"].item()) for value in values], default=2
    )
    while len(values) < batch_size:
        values.append(
            {
                "item_ids": torch.zeros(0, dtype=torch.int64),
                "behaviors": torch.zeros(0, dtype=torch.int64),
                "time_deltas": torch.zeros(0, dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "train_mask": torch.zeros(0, dtype=torch.bool),
                "length": torch.tensor(0, dtype=torch.int64),
                "record": torch.tensor(-1, dtype=torch.int64),
            }
        )
    batch: dict[str, torch.Tensor] = {}
    for name, dtype in (
        ("item_ids", torch.int64),
        ("behaviors", torch.int64),
        ("time_deltas", torch.float32),
        ("labels", torch.int64),
        ("train_mask", torch.bool),
    ):
        output = torch.zeros((batch_size, width), dtype=dtype)
        for row, value in enumerate(values):
            source = value[name]
            output[row, : len(source)] = source
        batch[name] = output
    batch["lengths"] = torch.stack([value["length"] for value in values])
    batch["record_indices"] = torch.stack(
        [value["record"] for value in values]
    )
    _, valid = build_next_item_targets(
        batch["item_ids"],
        batch["lengths"],
        batch["labels"],
        batch["train_mask"],
    )
    if len(records) and not bool(torch.any(valid)):
        raise RuntimeError("QK stream real training batch has no targets")
    return batch


def local_role_records(
    corpus: QKStreamChainCorpus,
    role: str,
    rank: int,
    world_size: int,
) -> np.ndarray:
    records = corpus.role_records(role)
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("QK stream role rank differs")
    if len(records) % world_size:
        raise ValueError("QK stream role is not rank balanced")
    return records[rank::world_size]


def prefix_inputs(
    corpus: QKStreamChainCorpus,
    record: int,
    edge: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    _, current, _ = record_window(corpus, record, edge)
    offset = int(corpus.arrays["record_offsets"][record])
    length = current
    items = torch.from_numpy(
        corpus.arrays["item_idx"][offset : offset + length].astype(
            np.int64, copy=True
        )
    )
    behaviors = torch.from_numpy(
        corpus.arrays["behavior"][offset : offset + length].astype(
            np.int64, copy=True
        )
    )
    deltas = torch.ones(length, dtype=torch.float32)
    deltas[0] = 0.0
    return items, behaviors, deltas, length


def evaluation_suffix(
    corpus: QKStreamChainCorpus,
    record: int,
    edge: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    _, current, following = record_window(corpus, record, edge)
    offset = int(corpus.arrays["record_offsets"][record])
    items = torch.from_numpy(
        corpus.arrays["item_idx"][
            offset + current : offset + following
        ].astype(np.int64, copy=True)
    )
    behaviors = torch.from_numpy(
        corpus.arrays["behavior"][
            offset + current : offset + following
        ].astype(np.int64, copy=True)
    )
    deltas = torch.ones(len(items), dtype=torch.float32)
    targets = torch.from_numpy(
        corpus.arrays["item_idx"][
            offset + current + 1 : offset + following + 1
        ].astype(np.int64, copy=True)
    )
    labels = torch.from_numpy(
        corpus.arrays["label"][
            offset + current + 1 : offset + following + 1
        ].astype(np.bool_, copy=True)
    )
    if not (
        len(items) == len(behaviors) == len(deltas) == len(targets) == len(labels)
        and len(items) > 0
    ):
        raise ValueError("QK stream evaluation suffix differs")
    return items, behaviors, deltas, targets, labels


@torch.no_grad()
def snapshot_source_prefixes(
    corpus: QKStreamChainCorpus,
    embedding: TrainableProjectedModuloEmbedding,
    roles: tuple[str, ...],
    edge: int,
    rank: int,
    world_size: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, list[torch.Tensor]]:
    if batch_size < 1:
        raise ValueError("QK stream snapshot batch size differs")
    result: dict[str, list[torch.Tensor]] = {}
    embedding.eval()
    for role in roles:
        records = local_role_records(corpus, role, rank, world_size)
        snapshots = []
        for start in range(0, len(records), batch_size):
            selected = records[start : start + batch_size]
            values = [
                prefix_inputs(corpus, int(record), edge)
                for record in selected
            ]
            width = max(value[3] for value in values)
            items = torch.zeros(
                (len(values), width), dtype=torch.int64, device=device
            )
            lengths = torch.tensor(
                [value[3] for value in values],
                dtype=torch.int64,
                device=device,
            )
            for row, value in enumerate(values):
                items[row, : value[3]] = value[0].to(device)
            projected = embedding(items, lengths)
            snapshots.extend(
                projected[row, : value[3]].detach().cpu().clone()
                for row, value in enumerate(values)
            )
            del projected, items, lengths
        if len(snapshots) != len(records):
            raise RuntimeError("QK stream source snapshot coverage differs")
        result[role] = snapshots
    return result


def fp16_storage_fp32_consumption(cache: HSTUKVCache) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k.to(torch.float16).to(torch.float32),
        v=cache.v.to(torch.float16).to(torch.float32),
        seq_len=cache.seq_len,
    )


def cache_relative_error(
    observed: HSTUKVCache,
    reference: HSTUKVCache,
) -> float:
    if observed.k.shape != reference.k.shape or observed.v.shape != reference.v.shape:
        raise ValueError("QK stream cache shapes differ")
    numerator = torch.sqrt(
        (observed.k - reference.k).double().square().sum()
        + (observed.v - reference.v).double().square().sum()
    )
    denominator = torch.sqrt(
        reference.k.double().square().sum()
        + reference.v.double().square().sum()
    ).clamp_min(1e-12)
    return float((numerator / denominator).item())


def _all_gather_tensor(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value.unsqueeze(0)
    values = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(values, value)
    return torch.stack(values)


@torch.no_grad()
def distributed_projected_candidate_scores(
    embedding: TrainableProjectedModuloEmbedding,
    hidden_by_method: torch.Tensor,
    candidates: torch.Tensor,
    real_targets: int,
) -> torch.Tensor:
    if (
        hidden_by_method.ndim != 3
        or candidates.ndim != 2
        or hidden_by_method.shape[1] != candidates.shape[0]
        or hidden_by_method.shape[2] != embedding.hidden_size
        or not 0 <= real_targets <= candidates.shape[0]
        or candidates.shape[1] < 2
    ):
        raise ValueError("QK stream candidate score request differs")
    projected = torch.matmul(
        hidden_by_method,
        embedding.projection_weight,
    )
    gathered_hidden = _all_gather_tensor(projected)
    gathered_candidates = _all_gather_tensor(candidates)
    world_size = gathered_hidden.shape[0]
    methods, target_slots, candidate_count = (
        hidden_by_method.shape[0],
        hidden_by_method.shape[1],
        candidates.shape[1],
    )
    output = torch.zeros(
        (world_size, methods, target_slots, candidate_count),
        dtype=hidden_by_method.dtype,
        device=hidden_by_method.device,
    )
    for requester in range(world_size):
        flat = gathered_candidates[requester].reshape(-1)
        owned = (flat > 0) & (flat.remainder(embedding.world_size) == embedding.rank)
        positions = torch.nonzero(owned, as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        row_ids = flat.index_select(0, positions).div(
            embedding.world_size, rounding_mode="floor"
        )
        target_ids = positions.div(candidate_count, rounding_mode="floor")
        weights = embedding.local_weight.index_select(0, row_ids)
        selected_hidden = gathered_hidden[requester].index_select(
            1, target_ids
        )
        scores = torch.einsum("mne,ne->mn", selected_hidden, weights)
        for method in range(methods):
            output[requester, method].view(-1).index_copy_(
                0, positions, scores[method]
            )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
    return output[embedding.rank, :, :real_targets]


def _local_prediction_row_bounds(
    num_prediction_items: int,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    if (
        num_prediction_items < 1
        or world_size < 1
        or not 0 <= rank < world_size
    ):
        raise ValueError("QK full-catalog prediction range differs")
    first_item = rank if rank > 0 else world_size
    if first_item > num_prediction_items:
        return 0, 0
    rows = (num_prediction_items - first_item) // world_size + 1
    first_row = first_item // world_size
    return first_row, first_row + rows


@torch.no_grad()
def distributed_full_catalog_topk(
    embedding: TrainableProjectedModuloEmbedding,
    hidden_by_method: torch.Tensor,
    *,
    num_prediction_items: int,
    maximum_k: int,
    item_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        hidden_by_method.shape != (2, embedding.hidden_size)
        or not 1 <= maximum_k <= num_prediction_items
        or not 1 <= num_prediction_items < embedding.num_embeddings
        or item_chunk < 1
    ):
        raise ValueError("QK full-catalog top-k request differs")
    projected = torch.matmul(
        hidden_by_method,
        embedding.projection_weight,
    )
    gathered_hidden = _all_gather_tensor(projected)
    world_size, methods, width = gathered_hidden.shape
    flat_hidden = gathered_hidden.reshape(-1, width)
    best_scores = torch.empty(
        (world_size * methods, 0),
        dtype=hidden_by_method.dtype,
        device=hidden_by_method.device,
    )
    best_ids = torch.empty(
        (world_size * methods, 0),
        dtype=torch.int64,
        device=hidden_by_method.device,
    )
    first_row, row_stop = _local_prediction_row_bounds(
        num_prediction_items,
        embedding.rank,
        embedding.world_size,
    )
    for start in range(first_row, row_stop, item_chunk):
        stop = min(start + item_chunk, row_stop)
        scores = torch.matmul(
            flat_hidden,
            embedding.local_weight[start:stop].t(),
        )
        ids = (
            torch.arange(
                start,
                stop,
                dtype=torch.int64,
                device=scores.device,
            )
            * embedding.world_size
            + embedding.rank
        ).expand(len(scores), -1)
        scores = torch.cat((best_scores, scores), dim=1)
        ids = torch.cat((best_ids, ids), dim=1)
        keep = min(maximum_k, scores.shape[1])
        best_scores, positions = torch.topk(
            scores,
            keep,
            dim=1,
            largest=True,
            sorted=True,
        )
        best_ids = ids.gather(1, positions)
    gathered_scores = _all_gather_tensor(
        best_scores.reshape(world_size, methods, -1)
    )
    gathered_ids = _all_gather_tensor(
        best_ids.reshape(world_size, methods, -1)
    )
    candidates = gathered_scores.permute(1, 2, 0, 3).reshape(
        world_size,
        methods,
        -1,
    )
    candidate_ids = gathered_ids.permute(1, 2, 0, 3).reshape(
        world_size,
        methods,
        -1,
    )
    scores, positions = torch.topk(
        candidates,
        maximum_k,
        dim=2,
        largest=True,
        sorted=True,
    )
    ids = candidate_ids.gather(2, positions)
    return scores[embedding.rank], ids[embedding.rank]


@torch.no_grad()
def distributed_full_catalog_metrics(
    embedding: TrainableProjectedModuloEmbedding,
    hidden_by_method: torch.Tensor,
    positive_ids: torch.Tensor,
    real_targets: int,
    *,
    num_prediction_items: int,
    item_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        hidden_by_method.ndim != 3
        or hidden_by_method.shape[0] < 1
        or hidden_by_method.shape[2] != embedding.hidden_size
        or positive_ids.shape != (hidden_by_method.shape[1],)
        or positive_ids.device != hidden_by_method.device
        or not 0 <= real_targets <= hidden_by_method.shape[1]
        or not 1 <= num_prediction_items < embedding.num_embeddings
        or item_chunk < 1
    ):
        raise ValueError("QK full-catalog metric request differs")
    if real_targets and (
        bool(torch.any(positive_ids[:real_targets] < 1))
        or bool(
            torch.any(
                positive_ids[:real_targets] > num_prediction_items
            )
        )
    ):
        raise ValueError("QK full-catalog positive item differs")
    projected = torch.matmul(
        hidden_by_method,
        embedding.projection_weight,
    )
    gathered_hidden = _all_gather_tensor(projected)
    gathered_positive = _all_gather_tensor(positive_ids.long())
    gathered_real = _all_gather_tensor(
        torch.tensor(real_targets, dtype=torch.int64, device=positive_ids.device)
    )
    world_size, methods, target_slots, width = gathered_hidden.shape
    if (
        world_size != embedding.world_size
        or methods < 1
        or width != embedding.embedding_width
        or gathered_positive.shape != (world_size, target_slots)
        or gathered_real.shape != (world_size,)
    ):
        raise RuntimeError("QK full-catalog gathered request differs")
    positive_scores = torch.zeros(
        (world_size, methods, target_slots),
        dtype=gathered_hidden.dtype,
        device=gathered_hidden.device,
    )
    for requester in range(world_size):
        count = int(gathered_real[requester].item())
        ids = gathered_positive[requester, :count]
        owned = ids.remainder(embedding.world_size) == embedding.rank
        positions = torch.nonzero(owned, as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        rows = ids.index_select(0, positions).div(
            embedding.world_size,
            rounding_mode="floor",
        )
        weights = embedding.local_weight.index_select(0, rows)
        hidden = gathered_hidden[requester].index_select(1, positions)
        scores = torch.einsum("mne,ne->mn", hidden, weights)
        positive_scores[requester, :, positions] = scores
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(positive_scores, op=dist.ReduceOp.SUM)
    local_lse = torch.full_like(positive_scores, -torch.inf)
    local_rank_counts = torch.zeros(
        positive_scores.shape,
        dtype=torch.int64,
        device=positive_scores.device,
    )
    first_row, row_stop = _local_prediction_row_bounds(
        num_prediction_items,
        embedding.rank,
        embedding.world_size,
    )
    flat_hidden = gathered_hidden.reshape(-1, width)
    for start in range(first_row, row_stop, item_chunk):
        stop = min(start + item_chunk, row_stop)
        weights = embedding.local_weight[start:stop]
        scores = torch.matmul(flat_hidden, weights.t()).reshape(
            world_size,
            methods,
            target_slots,
            stop - start,
        )
        local_ids = (
            torch.arange(
                start,
                stop,
                dtype=torch.int64,
                device=scores.device,
            )
            * embedding.world_size
            + embedding.rank
        )
        positive_mask = (
            gathered_positive[:, None, :, None]
            == local_ids[None, None, None, :]
        )
        local_lse = torch.logaddexp(
            local_lse,
            torch.logsumexp(scores, dim=-1),
        )
        local_rank_counts += (
            (scores >= positive_scores.unsqueeze(-1)) & ~positive_mask
        ).sum(dim=-1)
    gathered_lse = _all_gather_tensor(local_lse)
    global_lse = torch.logsumexp(gathered_lse, dim=0)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_rank_counts, op=dist.ReduceOp.SUM)
    count = real_targets
    nll = (
        global_lse[embedding.rank, :, :count]
        - positive_scores[embedding.rank, :, :count]
    )
    ranks = local_rank_counts[embedding.rank, :, :count] + 1
    if count and (
        not bool(torch.all(torch.isfinite(nll)).item())
        or bool(torch.any(ranks < 1).item())
        or bool(torch.any(ranks > num_prediction_items).item())
    ):
        raise RuntimeError("QK full-catalog metric output differs")
    return nll, ranks


def summarize_rank_sensitivity_record(
    nll_by_method: torch.Tensor,
    ranks_by_method: torch.Tensor,
    cutoffs: tuple[int, ...] = (10, 50, 200),
) -> dict[str, float | int]:
    if (
        nll_by_method.shape != ranks_by_method.shape
        or nll_by_method.ndim != 2
        or nll_by_method.shape[0] != 2
        or not cutoffs
        or any(value < 1 for value in cutoffs)
    ):
        raise ValueError("QK rank sensitivity record differs")
    reuse_nll, recompute_nll = nll_by_method.double()
    reuse_rank, recompute_rank = ranks_by_method.long()
    nll_delta = reuse_nll - recompute_nll
    rank_delta = reuse_rank - recompute_rank
    result: dict[str, float | int] = {
        "targets": nll_by_method.shape[1],
        "recompute_nll_wins": int(torch.count_nonzero(nll_delta > 0)),
        "reuse_nll_wins": int(torch.count_nonzero(nll_delta < 0)),
        "nll_ties": int(torch.count_nonzero(nll_delta == 0)),
        "recompute_rank_wins": int(torch.count_nonzero(rank_delta > 0)),
        "reuse_rank_wins": int(torch.count_nonzero(rank_delta < 0)),
        "rank_ties": int(torch.count_nonzero(rank_delta == 0)),
        "rank_improvement_sum": float(rank_delta.double().sum().item()),
        "log_rank_improvement_sum": float(
            (
                torch.log1p(reuse_rank.double())
                - torch.log1p(recompute_rank.double())
            ).sum().item()
        ),
        "absolute_log_rank_shift_sum": float(
            torch.abs(
                torch.log1p(reuse_rank.double())
                - torch.log1p(recompute_rank.double())
            ).sum().item()
        ),
    }
    for cutoff in cutoffs:
        reuse_hit = reuse_rank <= cutoff
        recompute_hit = recompute_rank <= cutoff
        result[f"recompute_rescues_at_{cutoff}"] = int(
            torch.count_nonzero(~reuse_hit & recompute_hit)
        )
        result[f"recompute_regressions_at_{cutoff}"] = int(
            torch.count_nonzero(reuse_hit & ~recompute_hit)
        )
        result[f"decision_flips_at_{cutoff}"] = int(
            torch.count_nonzero(reuse_hit != recompute_hit)
        )
    return result


def summarize_window_topk_record(
    topk_ids_by_method: torch.Tensor,
    positive_ids: torch.Tensor,
    cutoffs: tuple[int, ...] = (10, 50, 200),
) -> dict[str, float | int]:
    if (
        topk_ids_by_method.ndim != 2
        or topk_ids_by_method.shape[0] != 2
        or positive_ids.ndim != 1
        or not cutoffs
        or max(cutoffs) > topk_ids_by_method.shape[1]
    ):
        raise ValueError("QK next-window top-k record differs")
    positives = torch.unique(positive_ids.long())
    result: dict[str, float | int] = {
        "positive_items": len(positives),
    }
    for method, name in enumerate(("reuse", "recompute")):
        for cutoff in cutoffs:
            matched = torch.isin(
                topk_ids_by_method[method, :cutoff],
                positives,
            )
            hits = int(torch.count_nonzero(matched))
            positions = torch.nonzero(matched, as_tuple=False).flatten() + 1
            dcg = float(
                torch.reciprocal(
                    torch.log2(positions.double() + 1.0)
                ).sum().item()
            )
            ideal_count = min(len(positives), cutoff)
            ideal_positions = torch.arange(
                1,
                ideal_count + 1,
                dtype=torch.float64,
                device=topk_ids_by_method.device,
            )
            ideal = float(
                torch.reciprocal(torch.log2(ideal_positions + 1.0))
                .sum()
                .item()
            )
            result[f"{name}_hit_at_{cutoff}"] = int(hits > 0)
            result[f"{name}_recall_at_{cutoff}"] = (
                hits / len(positives) if len(positives) else 0.0
            )
            result[f"{name}_ndcg_at_{cutoff}"] = (
                dcg / ideal if ideal > 0 else 0.0
            )
    return result


def summarize_logged_window_record(
    scores_by_method: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float | int]:
    if (
        scores_by_method.ndim != 2
        or scores_by_method.shape[0] != 2
        or labels.shape != (scores_by_method.shape[1],)
        or bool(torch.any((labels != 0) & (labels != 1)))
    ):
        raise ValueError("QK logged-window record differs")
    labels = labels.bool()
    positives = int(torch.count_nonzero(labels))
    negatives = len(labels) - positives
    result: dict[str, float | int] = {
        "candidates": len(labels),
        "positives": positives,
        "negatives": negatives,
    }
    for method, name in enumerate(("reuse", "recompute")):
        scores = scores_by_method[method].double()
        if positives and negatives:
            differences = scores[labels, None] - scores[~labels][None, :]
            auc = float(
                (
                    (differences > 0).double()
                    + 0.5 * (differences == 0).double()
                ).mean().item()
            )
        else:
            auc = float("nan")
        order = torch.argsort(scores, descending=True, stable=True)
        ordered = labels[order]
        precision = torch.cumsum(ordered, dim=0).double() / torch.arange(
            1,
            len(ordered) + 1,
            dtype=torch.float64,
            device=ordered.device,
        )
        average_precision = float(
            precision[ordered].mean().item()
        ) if positives else float("nan")
        result[f"{name}_auc"] = auc
        result[f"{name}_average_precision"] = average_precision
    return result


def summarize_full_catalog_record(
    nll_by_method: torch.Tensor,
    ranks_by_method: torch.Tensor,
) -> dict[str, float | int]:
    if (
        nll_by_method.shape != ranks_by_method.shape
        or nll_by_method.ndim != 2
        or nll_by_method.shape[0] != 2
    ):
        raise ValueError("QK full-catalog paired metric shape differs")
    targets = nll_by_method.shape[1]
    result: dict[str, float | int] = {"targets": targets}
    for index, method in enumerate(("reuse", "recompute")):
        ranks = ranks_by_method[index].double()
        result[f"{method}_cross_entropy_sum"] = float(
            nll_by_method[index].double().sum().item()
        )
        result[f"{method}_ndcg_at_10_sum"] = float(
            torch.where(
                ranks <= 10,
                torch.reciprocal(torch.log2(ranks + 1.0)),
                torch.zeros_like(ranks),
            )
            .sum()
            .item()
        )
        result[f"{method}_mrr_sum"] = float(
            torch.reciprocal(ranks).sum().item()
        )
        for cutoff in (10, 50, 200):
            result[f"{method}_hit_rate_at_{cutoff}_sum"] = float(
                (ranks <= cutoff).double().sum().item()
            )
    return result


def _cluster_bootstrap_gap(
    targets: np.ndarray,
    oriented_record_gaps: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        count = min(256, samples - start)
        selected = generator.integers(
            0,
            len(targets),
            size=(count, len(targets)),
        )
        bootstrap[start : start + count] = (
            oriented_record_gaps[selected].sum(axis=1)
            / targets[selected].sum(axis=1)
        )
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return float(lower), float(upper)


def paired_full_catalog_summary(
    records: list[dict[str, object]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    eligible = [value for value in records if int(value["targets"]) > 0]
    if not eligible or bootstrap_samples < 1 or bootstrap_seed < 0:
        raise ValueError("QK full-catalog quality records differ")
    targets = np.asarray(
        [value["targets"] for value in eligible], dtype=np.float64
    )
    denominator = float(targets.sum())
    specifications = (
        ("cross_entropy", "cross_entropy_sum", "reuse_minus_recompute"),
        ("ndcg_at_10", "ndcg_at_10_sum", "recompute_minus_reuse"),
        ("mrr", "mrr_sum", "recompute_minus_reuse"),
        ("hit_rate_at_10", "hit_rate_at_10_sum", "recompute_minus_reuse"),
        ("hit_rate_at_50", "hit_rate_at_50_sum", "recompute_minus_reuse"),
        ("hit_rate_at_200", "hit_rate_at_200_sum", "recompute_minus_reuse"),
    )
    endpoints: dict[str, dict[str, float]] = {
        "reuse": {},
        "recompute": {},
    }
    gaps: dict[str, dict[str, object]] = {}
    for ordinal, (metric, suffix, direction) in enumerate(specifications):
        reuse = np.asarray(
            [value[f"reuse_{suffix}"] for value in eligible],
            dtype=np.float64,
        )
        recompute = np.asarray(
            [value[f"recompute_{suffix}"] for value in eligible],
            dtype=np.float64,
        )
        reuse_mean = float(reuse.sum() / denominator)
        recompute_mean = float(recompute.sum() / denominator)
        oriented = (
            reuse - recompute
            if direction == "reuse_minus_recompute"
            else recompute - reuse
        )
        gap = float(oriented.sum() / denominator)
        lower, upper = _cluster_bootstrap_gap(
            targets,
            oriented,
            samples=bootstrap_samples,
            seed=bootstrap_seed + ordinal * 1_000_003,
        )
        endpoints["reuse"][metric] = reuse_mean
        endpoints["recompute"][metric] = recompute_mean
        gaps[metric] = {
            "direction": direction,
            "absolute": gap,
            "relative_to_reuse": (
                gap / reuse_mean if reuse_mean != 0 else None
            ),
            "relative_percent": (
                100.0 * gap / reuse_mean if reuse_mean != 0 else None
            ),
            "record_cluster_bootstrap_95": {
                "lower": lower,
                "upper": upper,
                "samples": bootstrap_samples,
                "seed": bootstrap_seed + ordinal * 1_000_003,
            },
            "positive_direction_with_ci": bool(gap > 0 and lower > 0),
        }
    reuse_ce = endpoints["reuse"]["cross_entropy"]
    recompute_ce = endpoints["recompute"]["cross_entropy"]
    endpoints["reuse"]["perplexity"] = math.exp(reuse_ce)
    endpoints["recompute"]["perplexity"] = math.exp(recompute_ce)
    ce_lower = gaps["cross_entropy"]["record_cluster_bootstrap_95"][
        "lower"
    ]
    ce_upper = gaps["cross_entropy"]["record_cluster_bootstrap_95"][
        "upper"
    ]
    ce_gap = gaps["cross_entropy"]["absolute"]
    gaps["perplexity"] = {
        "direction": "reuse_over_recompute_ratio_minus_one",
        "relative_ratio": math.exp(ce_gap),
        "penalty_percent": 100.0 * math.expm1(ce_gap),
        "record_cluster_bootstrap_95_penalty_percent": {
            "lower": 100.0 * math.expm1(ce_lower),
            "upper": 100.0 * math.expm1(ce_upper),
        },
    }
    return {
        "protocol": "evokv_qk_full_catalog_reuse_recompute_metrics_v1",
        "candidate_set": "all prediction item ids [1, num_prediction_items]",
        "methods": ["reuse", "recompute"],
        "records": len(records),
        "records_with_targets": len(eligible),
        "positive_targets": int(denominator),
        "reuse": endpoints["reuse"],
        "recompute": endpoints["recompute"],
        "gaps": gaps,
        "decision": "manual_after_tuning",
    }


def candidate_batches(
    positive_ids: torch.Tensor,
    *,
    num_prediction_items: int,
    negative_count: int,
    seed: int,
    target_chunk: int,
    device: torch.device,
) -> tuple[list[tuple[torch.Tensor, int]], str]:
    if target_chunk < 1:
        raise ValueError("QK stream candidate target chunk differs")
    candidates = fixed_candidate_ids(
        positive_ids.cpu(),
        num_prediction_items,
        negative_count,
        seed,
    )
    digest = hashlib.sha256(
        np.asarray(candidates, dtype="<i8").tobytes()
    ).hexdigest()
    batches = []
    for start in range(0, len(candidates), target_chunk):
        value = candidates[start : start + target_chunk]
        real = len(value)
        padded = torch.zeros(
            (target_chunk, negative_count + 1), dtype=torch.int64
        )
        padded[:real] = value
        batches.append((padded.to(device), real))
    return batches, digest


def summarize_record_scores(
    reuse_scores: torch.Tensor,
    exact_scores: torch.Tensor,
) -> dict[str, float | int]:
    if reuse_scores.shape != exact_scores.shape or reuse_scores.ndim != 2:
        raise ValueError("QK stream paired scores differ")
    targets = len(reuse_scores)
    if targets < 1:
        return {
            "targets": 0,
            "reuse_loss_sum": 0.0,
            "exact_loss_sum": 0.0,
            "reuse_hit_sum": 0.0,
            "exact_hit_sum": 0.0,
            "reuse_ndcg_sum": 0.0,
            "exact_ndcg_sum": 0.0,
        }
    labels = torch.zeros(targets, dtype=torch.int64, device=reuse_scores.device)
    reuse_loss = torch.nn.functional.cross_entropy(
        reuse_scores, labels, reduction="sum"
    )
    exact_loss = torch.nn.functional.cross_entropy(
        exact_scores, labels, reduction="sum"
    )

    def ranking(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ranks = 1 + (scores[:, 1:] >= scores[:, :1]).sum(dim=1)
        hit = (ranks <= 10).double()
        ndcg = torch.where(
            ranks <= 10,
            torch.reciprocal(torch.log2(ranks.double() + 1.0)),
            torch.zeros_like(ranks, dtype=torch.float64),
        )
        return hit, ndcg

    reuse_hit, reuse_ndcg = ranking(reuse_scores)
    exact_hit, exact_ndcg = ranking(exact_scores)
    return {
        "targets": targets,
        "reuse_loss_sum": float(reuse_loss.item()),
        "exact_loss_sum": float(exact_loss.item()),
        "reuse_hit_sum": float(reuse_hit.sum().item()),
        "exact_hit_sum": float(exact_hit.sum().item()),
        "reuse_ndcg_sum": float(reuse_ndcg.sum().item()),
        "exact_ndcg_sum": float(exact_ndcg.sum().item()),
    }


def paired_quality_summary(
    records: list[dict[str, object]],
    *,
    epsilon_ce: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    eligible = [value for value in records if int(value["targets"]) > 0]
    if not eligible:
        raise RuntimeError("QK stream quality role has no targets")
    targets = np.asarray([value["targets"] for value in eligible], dtype=np.float64)
    reuse_loss = np.asarray(
        [value["reuse_loss_sum"] for value in eligible], dtype=np.float64
    )
    exact_loss = np.asarray(
        [value["exact_loss_sum"] for value in eligible], dtype=np.float64
    )
    denominator = float(targets.sum())
    reuse_ce = float(reuse_loss.sum() / denominator)
    exact_ce = float(exact_loss.sum() / denominator)
    gap = reuse_ce - exact_ce
    generator = np.random.default_rng(bootstrap_seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for start in range(0, bootstrap_samples, 256):
        count = min(256, bootstrap_samples - start)
        selected = generator.integers(
            0, len(eligible), size=(count, len(eligible))
        )
        sampled_targets = targets[selected].sum(axis=1)
        bootstrap[start : start + count] = (
            (reuse_loss[selected] - exact_loss[selected]).sum(axis=1)
            / sampled_targets
        )
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    total_targets = int(denominator)
    result = {
        "records": len(records),
        "records_with_targets": len(eligible),
        "positive_targets": total_targets,
        "reuse": {
            "sampled_cross_entropy": reuse_ce,
            "hit_rate_at_10": sum(
                float(value["reuse_hit_sum"]) for value in eligible
            )
            / denominator,
            "ndcg_at_10": sum(
                float(value["reuse_ndcg_sum"]) for value in eligible
            )
            / denominator,
        },
        "exact": {
            "sampled_cross_entropy": exact_ce,
            "hit_rate_at_10": sum(
                float(value["exact_hit_sum"]) for value in eligible
            )
            / denominator,
            "ndcg_at_10": sum(
                float(value["exact_ndcg_sum"]) for value in eligible
            )
            / denominator,
        },
        "reuse_minus_exact_ce": gap,
        "record_cluster_bootstrap_95": {
            "lower": float(lower),
            "upper": float(upper),
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "epsilon_ce": epsilon_ce,
    }
    result["existence_gate_passed"] = bool(gap > 0 and lower > 0)
    result["practical_gate_passed"] = bool(
        gap >= epsilon_ce and lower > 0
    )
    return result


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
