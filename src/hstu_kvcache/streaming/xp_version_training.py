from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from ..data.qk_xp_edge_inputs import (
    ROLE_NAMES,
    array_sha256,
    artifact_sha256,
)
from .sharded_edge import (
    FixedHeldoutBatch,
    fixed_candidate_ids,
    make_fixed_heldout_batch,
)
from .trainer import build_next_item_targets
from .xp_projected_edge import (
    OptimizerActiveRowTracker,
    TrainableProjectedModuloEmbedding,
    active_row_ids_sha256,
    active_row_update_counts_sha256,
)

XP_VERSION_TRAINING_PROTOCOL = (
    "evokv_xp_two_edge_training_development_v0"
)
ROLE_CODES = {
    name: index for index, name in enumerate(ROLE_NAMES)
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class XPFixedEdgeCorpus:
    path: Path
    summary_path: Path
    arrays: dict[str, np.ndarray]
    metadata: dict[str, object]
    summary: dict[str, object]
    file_sha256: str
    summary_sha256: str
    content_sha256: str

    def role_records(self, role: str) -> np.ndarray:
        if role not in ROLE_CODES:
            raise ValueError("XP edge role is invalid")
        return np.flatnonzero(
            self.arrays["record_role"] == ROLE_CODES[role]
        )


def _required_arrays() -> dict[str, np.dtype]:
    return {
        "record_user_ids": np.dtype(np.int64),
        "record_role": np.dtype(np.uint8),
        "record_offsets": np.dtype(np.int64),
        "record_history_start": np.dtype(np.uint16),
        "record_history_end": np.dtype(np.uint16),
        "record_update_start": np.dtype(np.uint16),
        "record_update_end": np.dtype(np.uint16),
        "item_idx": np.dtype(np.uint32),
        "behavior": np.dtype(np.uint8),
        "raw_label": np.dtype(np.uint8),
        "label": np.dtype(np.uint8),
        "raw_ordinal": np.dtype(np.uint16),
        "is_prediction_item": np.dtype(np.uint8),
        "is_stream_only_fallback": np.dtype(np.uint8),
    }


def _validate_role_binding(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    summary: dict[str, object],
) -> None:
    frozen = metadata.get("frozen_roles")
    if (
        not isinstance(frozen, dict)
        or frozen.get("included") != list(ROLE_NAMES)
        or frozen.get("excluded") != ["fit", "profile", "final"]
        or not isinstance(
            frozen.get("included_user_ids_sha256"),
            dict,
        )
    ):
        raise ValueError("XP fixed edge role exclusion differs")
    observed_users = []
    observed_counts = {}
    for role in ROLE_NAMES:
        records = np.flatnonzero(
            arrays["record_role"] == ROLE_CODES[role]
        )
        users = arrays["record_user_ids"][records]
        if (
            len(np.unique(users)) != len(users)
            or array_sha256(users)
            != frozen["included_user_ids_sha256"].get(role)
        ):
            raise ValueError(
                f"XP fixed edge user binding differs for {role}"
            )
        observed_users.append(users)
        observed_counts[role] = len(records)
    if len(np.unique(np.concatenate(observed_users))) != sum(
        len(values) for values in observed_users
    ):
        raise ValueError("XP fixed edge roles overlap")
    if summary.get("records") != observed_counts:
        raise ValueError("XP fixed edge summary role counts differ")


def _validate_record_layout(
    arrays: dict[str, np.ndarray],
) -> None:
    records = len(arrays["record_user_ids"])
    offsets = arrays["record_offsets"]
    if (
        offsets.shape != (records + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(arrays["item_idx"])
        or np.any(offsets[1:] <= offsets[:-1])
        or any(
            arrays[name].shape != (records,)
            for name in (
                "record_role",
                "record_history_start",
                "record_history_end",
                "record_update_start",
                "record_update_end",
            )
        )
        or np.any(arrays["record_role"] >= len(ROLE_NAMES))
        or np.any(arrays["record_history_start"] != 0)
        or np.any(
            arrays["record_history_end"]
            != arrays["record_update_start"]
        )
        or np.any(
            arrays["record_update_start"]
            >= arrays["record_update_end"]
        )
        or not np.array_equal(
            np.diff(offsets),
            arrays["record_update_end"].astype(np.int64),
        )
    ):
        raise ValueError("XP fixed edge record layout differs")
    for record in range(records):
        start = int(offsets[record])
        stop = int(offsets[record + 1])
        if not np.array_equal(
            arrays["raw_ordinal"][start:stop],
            np.arange(stop - start, dtype=np.uint16),
        ):
            raise ValueError("XP fixed edge ordinal layout differs")


def load_xp_fixed_edge_corpus(
    path: str | Path,
    summary_path: str | Path,
    *,
    num_embeddings: int,
    num_prediction_items: int,
    num_behaviors: int,
) -> XPFixedEdgeCorpus:
    resolved = Path(path)
    resolved_summary = Path(summary_path)
    with np.load(resolved, allow_pickle=False) as source:
        missing = set(_required_arrays()) - set(source.files)
        if missing or "metadata_json" not in source.files:
            raise ValueError(
                f"XP fixed edge artifact is incomplete: {sorted(missing)}"
            )
        arrays = {
            name: source[name].copy()
            for name in source.files
            if name != "metadata_json"
        }
        metadata = json.loads(
            str(source["metadata_json"].item())
        )
    summary = json.loads(resolved_summary.read_text())
    for name, dtype in _required_arrays().items():
        if arrays[name].dtype != dtype:
            raise ValueError(
                f"XP fixed edge dtype differs for {name}"
            )
    content_hash = artifact_sha256(arrays)
    artifact = summary.get("artifact")
    catalog = metadata.get("catalog")
    if (
        metadata.get("dataset") != "tenrec-qk"
        or metadata.get("scientific_result") is not False
        or metadata.get("formal_result") is not False
        or metadata.get("content_sha256") != content_hash
        or summary.get("content_sha256") != content_hash
        or summary.get("status") != "pass"
        or not isinstance(artifact, dict)
        or artifact.get("content_sha256") != content_hash
        or artifact.get("file_sha256")
        != file_sha256(resolved)
        or not isinstance(catalog, dict)
        or int(catalog.get("base_entity_rows", -1)) + 1
        != num_embeddings
        or int(catalog.get("prediction_rows", -1))
        != num_prediction_items
    ):
        raise ValueError("XP fixed edge artifact binding differs")
    _validate_record_layout(arrays)
    _validate_role_binding(arrays, metadata, summary)
    token_count = len(arrays["item_idx"])
    if (
        any(
            arrays[name].shape != (token_count,)
            for name in (
                "behavior",
                "raw_label",
                "label",
                "raw_ordinal",
                "is_prediction_item",
                "is_stream_only_fallback",
            )
        )
        or np.any(arrays["item_idx"] < 1)
        or np.any(arrays["item_idx"] >= num_embeddings)
        or np.any(arrays["behavior"] < 1)
        or np.any(arrays["behavior"] > num_behaviors)
        or np.any(arrays["raw_label"] > 1)
        or np.any(arrays["label"] > 1)
        or np.any(arrays["is_prediction_item"] > 1)
        or np.any(arrays["is_stream_only_fallback"] > 1)
        or np.any(arrays["label"] > arrays["raw_label"])
        or np.any(
            arrays["label"]
            > arrays["is_prediction_item"]
        )
        or np.any(
            arrays["item_idx"][arrays["label"].astype(bool)]
            > num_prediction_items
        )
    ):
        raise ValueError("XP fixed edge token semantics differ")
    boundaries = metadata.get("boundaries")
    if not isinstance(boundaries, dict):
        raise ValueError("XP fixed edge boundaries are absent")
    for role in ROLE_NAMES:
        records = np.flatnonzero(
            arrays["record_role"] == ROLE_CODES[role]
        )
        boundary = boundaries.get(role)
        if (
            not isinstance(boundary, dict)
            or boundary.get("history")
            != [
                int(arrays["record_history_start"][records[0]]),
                int(arrays["record_history_end"][records[0]]),
            ]
            or boundary.get("update")
            != [
                int(arrays["record_update_start"][records[0]]),
                int(arrays["record_update_end"][records[0]]),
            ]
            or np.any(
                arrays["record_history_end"][records]
                != arrays["record_history_end"][records[0]]
            )
            or np.any(
                arrays["record_update_end"][records]
                != arrays["record_update_end"][records[0]]
            )
        ):
            raise ValueError(
                f"XP fixed edge boundary differs for {role}"
            )
    return XPFixedEdgeCorpus(
        path=resolved,
        summary_path=resolved_summary,
        arrays=arrays,
        metadata=metadata,
        summary=summary,
        file_sha256=file_sha256(resolved),
        summary_sha256=file_sha256(resolved_summary),
        content_sha256=content_hash,
    )


def _record_has_target(
    corpus: XPFixedEdgeCorpus,
    record: int,
) -> bool:
    arrays = corpus.arrays
    offset = int(arrays["record_offsets"][record])
    update_start = int(
        arrays["record_update_start"][record]
    )
    update_end = int(arrays["record_update_end"][record])
    target_labels = arrays["label"][
        offset + update_start : offset + update_end
    ]
    source_items = arrays["item_idx"][
        offset + update_start - 1 : offset + update_end - 1
    ]
    return bool(np.any((target_labels > 0) & (source_items > 0)))


def _materialize_record(
    corpus: XPFixedEdgeCorpus,
    record: int,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    arrays = corpus.arrays
    offset = int(arrays["record_offsets"][record])
    update_start = int(
        arrays["record_update_start"][record]
    )
    update_end = int(arrays["record_update_end"][record])
    window_start = max(0, update_end - max_seq_len)
    start = offset + window_start
    stop = offset + update_end
    width = stop - start
    ordinals = arrays["raw_ordinal"][start:stop].astype(
        np.float32,
        copy=True,
    )
    time_deltas = np.empty(width, dtype=np.float32)
    if window_start == 0:
        time_deltas[0] = 0.0
    else:
        time_deltas[0] = (
            ordinals[0]
            - float(arrays["raw_ordinal"][start - 1])
        )
    time_deltas[1:] = np.diff(ordinals)
    train_mask = np.zeros(width, dtype=np.bool_)
    train_mask[
        update_start - window_start : update_end - window_start
    ] = True
    return {
        "item_ids": torch.from_numpy(
            arrays["item_idx"][start:stop].astype(
                np.int64,
                copy=True,
            )
        ),
        "behaviors": torch.from_numpy(
            arrays["behavior"][start:stop].astype(
                np.int64,
                copy=True,
            )
        ),
        "time_deltas": torch.from_numpy(time_deltas),
        "labels": torch.from_numpy(
            arrays["label"][start:stop].astype(
                np.int64,
                copy=True,
            )
        ),
        "train_mask": torch.from_numpy(train_mask),
        "length": torch.tensor(width, dtype=torch.int64),
        "record": torch.tensor(record, dtype=torch.int64),
        "window_start": torch.tensor(
            window_start,
            dtype=torch.int64,
        ),
    }


def build_role_batches(
    corpus: XPFixedEdgeCorpus,
    role: str,
    *,
    max_seq_len: int,
    batch_size_per_rank: int,
    rank: int,
    world_size: int,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, object]]:
    if (
        role not in ROLE_CODES
        or max_seq_len < 2
        or batch_size_per_rank < 1
        or world_size < 1
        or not 0 <= rank < world_size
    ):
        raise ValueError("XP role batch request differs")
    records = corpus.role_records(role)
    eligible = np.asarray(
        [
            int(record)
            for record in records
            if _record_has_target(corpus, int(record))
        ],
        dtype=np.int64,
    )
    if len(eligible) == 0:
        raise RuntimeError(f"XP edge role {role} has no targets")
    global_batch = batch_size_per_rank * world_size
    steps = (len(eligible) + global_batch - 1) // global_batch
    materialized = {
        int(record): _materialize_record(
            corpus,
            int(record),
            max_seq_len,
        )
        for record in eligible
    }
    first = materialized[int(eligible[0])]
    width = len(first["item_ids"])
    batches = []
    local_real_records = 0
    local_targets = 0
    local_tokens = 0
    for step in range(steps):
        left = step * global_batch + rank * batch_size_per_rank
        right = min(left + batch_size_per_rank, len(eligible))
        selected = eligible[left:right]
        values = [materialized[int(record)] for record in selected]
        local_real_records += len(values)
        while len(values) < batch_size_per_rank:
            values.append(
                {
                    "item_ids": torch.zeros(
                        width,
                        dtype=torch.int64,
                    ),
                    "behaviors": torch.zeros(
                        width,
                        dtype=torch.int64,
                    ),
                    "time_deltas": torch.zeros(
                        width,
                        dtype=torch.float32,
                    ),
                    "labels": torch.zeros(
                        width,
                        dtype=torch.int64,
                    ),
                    "train_mask": torch.zeros(
                        width,
                        dtype=torch.bool,
                    ),
                    "length": torch.tensor(
                        0,
                        dtype=torch.int64,
                    ),
                    "record": torch.tensor(
                        -1,
                        dtype=torch.int64,
                    ),
                    "window_start": torch.tensor(
                        0,
                        dtype=torch.int64,
                    ),
                }
            )
        batch = {
            name: torch.stack([value[name] for value in values])
            for name in (
                "item_ids",
                "behaviors",
                "time_deltas",
                "labels",
                "train_mask",
            )
        }
        batch["lengths"] = torch.stack(
            [value["length"] for value in values]
        )
        batch["record_indices"] = torch.stack(
            [value["record"] for value in values]
        )
        batch["window_starts"] = torch.stack(
            [value["window_start"] for value in values]
        )
        _, valid = build_next_item_targets(
            batch["item_ids"],
            batch["lengths"],
            batch["labels"],
            batch["train_mask"],
        )
        local_targets += int(valid.sum().item())
        local_tokens += int(batch["lengths"].sum().item())
        batches.append(batch)
    return batches, {
        "role": role,
        "global_records": len(records),
        "global_eligible_records": len(eligible),
        "global_zero_target_records_removed": (
            len(records) - len(eligible)
        ),
        "steps_per_rank": steps,
        "batch_size_per_rank": batch_size_per_rank,
        "local_real_records": local_real_records,
        "local_padding_records": (
            steps * batch_size_per_rank - local_real_records
        ),
        "local_tokens": local_tokens,
        "local_targets": local_targets,
        "maximum_model_context": max_seq_len,
        "physical_sequence_width": width,
        "causal_window_start": int(
            first["window_start"].item()
        ),
        "time_delta_semantics": (
            "adjacent within-user raw-ordinal delta; the uncropped "
            "first event is zero and a cropped first event retains its "
            "delta from the immediately preceding source event"
        ),
    }


def prepare_fixed_qualification(
    batches: list[dict[str, torch.Tensor]],
    *,
    num_prediction_items: int,
    negative_count: int,
    seed: int,
    rank: int,
    world_size: int,
) -> tuple[list[FixedHeldoutBatch], str]:
    heldout = [
        make_fixed_heldout_batch(
            batch,
            num_prediction_items,
            negative_count,
            seed + index * world_size + rank,
        )
        for index, batch in enumerate(batches)
    ]
    digest = hashlib.sha256()
    for value in heldout:
        candidates = value.candidates.contiguous().numpy()
        digest.update(
            np.asarray(candidates.shape, dtype="<i8").tobytes()
        )
        digest.update(
            np.asarray(candidates, dtype="<i8").tobytes()
        )
    return heldout, digest.hexdigest()


def _manual_all_reduce_dense_gradients(
    dense_model: nn.Module,
    process_group: dist.ProcessGroup | None,
) -> None:
    for parameter in dense_model.parameters():
        if parameter.grad is None:
            raise RuntimeError("XP dense gradient is absent")
        if not bool(torch.all(torch.isfinite(parameter.grad))):
            raise RuntimeError("XP dense gradient is not finite")
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(
                parameter.grad,
                op=dist.ReduceOp.SUM,
                group=process_group,
            )


def _tracked_sparse_step_delta(
    embedding: TrainableProjectedModuloEmbedding,
    optimizer: torch.optim.Optimizer,
    tracker: OptimizerActiveRowTracker,
) -> tuple[int, ...]:
    gradient = embedding.local_weight.grad
    if gradient is None or not gradient.is_sparse:
        raise RuntimeError("XP embedding gradient must be sparse")
    coalesced = gradient.coalesce()
    values = coalesced.values()
    if not bool(torch.all(torch.isfinite(values))):
        raise RuntimeError("XP embedding gradient is not finite")
    nonzero = (
        torch.any(values != 0, dim=1)
        if values.numel()
        else torch.zeros(
            0,
            dtype=torch.bool,
            device=values.device,
        )
    )
    local_rows = coalesced.indices()[0][nonzero]
    optimizer.step()
    tracker.mark_local_rows(local_rows)
    if embedding.rank == 0 and embedding.local_rows:
        with torch.no_grad():
            embedding.local_weight[0].zero_()
    return tuple(
        int(value)
        for value in (
            local_rows.detach().cpu() * embedding.world_size
            + embedding.rank
        ).tolist()
        if 0 < int(value) < embedding.num_embeddings
    )


def xp_projected_next_item_train_step(
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
) -> tuple[float, int, int, tuple[int, ...]]:
    dense_model.train()
    embedding.train()
    item_ids = batch["item_ids"].to(device)
    behaviors = batch["behaviors"].to(device)
    time_deltas = batch["time_deltas"].to(device)
    lengths = batch["lengths"].to(device)
    labels = batch["labels"].to(device)
    train_mask = batch["train_mask"].to(device)
    dense_optimizer.zero_grad(set_to_none=True)
    projection_optimizer.zero_grad(set_to_none=True)
    embedding_optimizer.zero_grad(set_to_none=True)
    item_vectors = embedding(item_ids, lengths)
    hidden = dense_model(
        item_vectors,
        behaviors,
        time_deltas,
        lengths,
    )
    targets, valid = build_next_item_targets(
        item_ids,
        lengths,
        labels,
        train_mask,
    )
    positive_ids = targets[valid]
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
    candidate_vectors = embedding(
        candidates,
        candidate_lengths,
    )
    if len(positive_ids):
        logits = torch.einsum(
            "nh,nch->nc",
            hidden[:, :-1][valid],
            candidate_vectors,
        )
        loss = torch.nn.functional.cross_entropy(
            logits,
            torch.zeros(
                len(positive_ids),
                dtype=torch.int64,
                device=device,
            ),
        )
    else:
        loss = (
            hidden.sum() * 0.0
            + candidate_vectors.sum() * 0.0
        )
    local_targets = torch.tensor(
        len(positive_ids),
        dtype=torch.float64,
        device=device,
    )
    global_targets = local_targets.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(
            global_targets,
            op=dist.ReduceOp.SUM,
            group=process_group,
        )
    scale = (
        local_targets / global_targets
        if global_targets.item() > 0
        else torch.zeros_like(global_targets)
    )
    (loss * scale.to(loss.dtype)).backward()
    _manual_all_reduce_dense_gradients(
        dense_model,
        process_group,
    )
    projection_gradient = embedding.projection_weight.grad
    if (
        projection_gradient is None
        or not bool(torch.all(torch.isfinite(projection_gradient)))
    ):
        raise RuntimeError("XP projection gradient is not finite")
    dense_parameters = list(dense_model.parameters())
    torch.nn.utils.clip_grad_norm_(dense_parameters, 1.0)
    torch.nn.utils.clip_grad_norm_(
        [embedding.projection_weight],
        1.0,
    )
    dense_optimizer.step()
    projection_optimizer.step()
    updated_rows = _tracked_sparse_step_delta(
        embedding,
        embedding_optimizer,
        tracker,
    )
    return (
        float(loss.detach().item()),
        len(positive_ids),
        int(global_targets.item()),
        updated_rows,
    )


def tracker_count_snapshot(
    tracker: OptimizerActiveRowTracker,
) -> torch.Tensor:
    return tracker.local_update_counts.clone()


def global_tracker_delta(
    tracker: OptimizerActiveRowTracker,
    before: torch.Tensor,
    *,
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, object]:
    if before.shape != tracker.local_update_counts.shape:
        raise ValueError("XP tracker delta shape differs")
    delta = tracker.local_update_counts - before
    if bool(torch.any(delta < 0)):
        raise ValueError("XP tracker counts moved backwards")
    local_rows = torch.nonzero(
        delta > 0,
        as_tuple=False,
    ).flatten()
    local_ids = (
        local_rows * tracker.world_size + tracker.rank
    )
    valid = (
        (local_ids > 0)
        & (local_ids < tracker.num_embeddings)
    )
    ids = [
        int(value)
        for value in local_ids[valid].tolist()
    ]
    counts = [
        int(value)
        for value in delta.index_select(
            0,
            local_rows,
        )[valid].tolist()
    ]
    if dist.is_available() and dist.is_initialized():
        gathered: list[object] = [None] * tracker.world_size
        dist.all_gather_object(
            gathered,
            (ids, counts),
            group=process_group,
        )
    else:
        gathered = [(ids, counts)]
    ordered = sorted(
        (
            (int(row), int(count))
            for rank_ids, rank_counts in gathered
            for row, count in zip(
                rank_ids,
                rank_counts,
                strict=True,
            )
        ),
        key=lambda value: value[0],
    )
    global_ids = [value[0] for value in ordered]
    global_counts = [value[1] for value in ordered]
    if len(global_ids) != len(set(global_ids)):
        raise RuntimeError("XP tracker delta shards overlap")
    return {
        "definition": (
            "rows with a finite nonzero sparse embedding gradient "
            "recorded after a successful optimizer step on this edge"
        ),
        "global_updated_rows": len(global_ids),
        "global_optimizer_update_events": sum(global_counts),
        "global_row_ids_sha256": (
            active_row_ids_sha256(global_ids)
        ),
        "global_row_update_counts_sha256": (
            active_row_update_counts_sha256(
                global_ids,
                global_counts,
            )
        ),
        "update_count_minimum": (
            min(global_counts) if global_counts else 0
        ),
        "update_count_maximum": (
            max(global_counts) if global_counts else 0
        ),
        "update_count_mean": (
            sum(global_counts) / len(global_counts)
            if global_counts
            else 0.0
        ),
        "per_rank_updated_rows": [
            len(rank_ids)
            for rank_ids, _ in gathered
        ],
        "padding_row_excluded": True,
    }
