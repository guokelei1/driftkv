"""Shared causal data and checkpoint helpers for v0/v1/v2 training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FoundationBatch:
    item_ids: torch.Tensor
    behaviors: torch.Tensor
    time_deltas: torch.Tensor
    lengths: torch.Tensor
    candidate_ids: torch.Tensor
    query_time_deltas: torch.Tensor
    base_features: torch.Tensor | None
    labels: torch.Tensor
    weights: torch.Tensor


class FoundationHistoryIndex:
    """In-memory chronological listen index with strict `< query time` lookup."""

    def __init__(self, rows: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        self.rows = rows

    @classmethod
    def from_columns(
        cls,
        uids: np.ndarray,
        timestamps: np.ndarray,
        item_ids: np.ndarray,
        behaviors: np.ndarray,
    ) -> "FoundationHistoryIndex":
        order = np.lexsort((item_ids, timestamps, uids))
        uids, timestamps = np.asarray(uids)[order], np.asarray(timestamps)[order]
        item_ids, behaviors = np.asarray(item_ids)[order], np.asarray(behaviors)[order]
        rows = {}
        for uid in np.unique(uids):
            mask = uids == uid
            rows[int(uid)] = (
                timestamps[mask].astype(np.int64),
                item_ids[mask].astype(np.int64),
                behaviors[mask].astype(np.int64),
            )
        return cls(rows)

    def prefix(self, uid: int, query_timestamp: int, max_history: int = 512):
        timestamps, items, behaviors = self.rows.get(
            int(uid),
            (np.empty(0, dtype=np.int64),) * 3,
        )
        stop = int(np.searchsorted(timestamps, int(query_timestamp), side="left"))
        start = max(0, stop - int(max_history))
        timestamps = timestamps[start:stop]
        return items[start:stop], behaviors[start:stop], timestamps


def collate_foundation_batch(
    requests: list[dict], history: FoundationHistoryIndex, *, device: torch.device
) -> FoundationBatch:
    if not requests:
        raise ValueError("empty foundation batch")
    prefixes = [history.prefix(row["uid"], row["query_timestamp"]) for row in requests]
    if any(len(prefix[0]) == 0 for prefix in prefixes):
        raise ValueError("foundation request has an empty strict-prior prefix")
    width = max(len(prefix[0]) for prefix in prefixes)
    batch = len(requests)
    items = np.zeros((batch, width), dtype=np.int64)
    behaviors = np.zeros((batch, width), dtype=np.int64)
    deltas = np.zeros((batch, width), dtype=np.float32)
    lengths = np.empty(batch, dtype=np.int64)
    query_deltas = np.empty(batch, dtype=np.float32)
    for index, (row, (prefix_items, prefix_behaviors, prefix_times)) in enumerate(
        zip(requests, prefixes, strict=True)
    ):
        length = len(prefix_items)
        lengths[index] = length
        items[index, :length] = prefix_items
        behaviors[index, :length] = prefix_behaviors
        if length > 1:
            deltas[index, 1:length] = np.diff(prefix_times)
        query_deltas[index] = float(int(row["query_timestamp"]) - int(prefix_times[-1]))
    return FoundationBatch(
        item_ids=torch.as_tensor(items, device=device),
        behaviors=torch.as_tensor(behaviors, device=device),
        time_deltas=torch.as_tensor(deltas, device=device),
        lengths=torch.as_tensor(lengths, device=device),
        candidate_ids=torch.as_tensor(
            [[int(row["item_idx"])] for row in requests], device=device
        ),
        query_time_deltas=torch.as_tensor(query_deltas, device=device),
        # Historical residual experiments carry Base features; HSTU-native
        # training/evaluation intentionally does not materialize them.
        base_features=(
            torch.as_tensor(np.asarray([row["base_features"] for row in requests], dtype=np.float32), device=device)[:, None, :]
            if all("base_features" in row for row in requests) else None
        ),
        labels=torch.as_tensor(
            [float(row["label"]) for row in requests], dtype=torch.float32, device=device
        ),
        weights=torch.as_tensor(
            [float(row["weight"]) for row in requests], dtype=torch.float32, device=device
        ),
    )


def cache_producer_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash only parameters capable of changing persistent prefix K/V."""
    digest = hashlib.sha256()
    excluded = ("query_encoder.", "cc_score_head.", "output_emb.")
    for name in sorted(state_dict):
        if name.startswith(excluded):
            continue
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
