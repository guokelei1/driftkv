from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_movielens_hard(path: str | Path, split: str = "train") -> list[dict]:
    """Load LRM1-standardised movielens_1m_hard_v5 records (generative-rec format).

    Each record: history_items (list[str]), candidate_items (list[str]),
    labels.positive_item_ids (list[str]). We map item strings -> contiguous
    int ids using the item_catalog. Designed for the small `pilot20` set for
    fast feasibility checks; `main100` is the larger variant.
    """
    path = Path(path)
    with open(path / "item_catalog.jsonl") as f:
        catalog = [json.loads(line) for line in f]
    item_map = {row["item_id"]: i + 1 for i, row in enumerate(catalog)}

    records = []
    with open(path / f"records_{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            hist = [item_map[x] for x in r["history_items"]]
            cands = [item_map[x] for x in r["candidate_items"]]
            pos = set(item_map[x] for x in r["labels"]["positive_item_ids"])
            labels = np.array([1 if c in pos else 0 for c in cands], dtype=np.int64)
            records.append({
                "history": np.array(hist, dtype=np.int64),
                "candidates": np.array(cands, dtype=np.int64),
                "labels": labels,
            })
    records.sort(key=lambda r: len(r["history"]))
    return records


def collate_grec_batch(records: list[dict], max_hist: int = 256, max_cands: int = 20) -> dict:
    """Collate generative-rec records into batched tensors.

    Returns history [B, H], time_deltas (zeros - ML1m has no ts), behaviors (1s),
    candidates [B, C], labels [B, C]. This format lets the same HSTU forward
    path be reused for both KuaiRand (streaming) and ML1m (static eval).
    """
    B = len(records)
    H = min(max(len(r["history"]) for r in records), max_hist)
    C = min(max(len(r["candidates"]) for r in records), max_cands)
    history = np.zeros((B, H), dtype=np.int64)
    behaviors = np.ones((B, H), dtype=np.int64)
    time_deltas = np.zeros((B, H), dtype=np.float32)
    candidates = np.zeros((B, C), dtype=np.int64)
    labels = np.zeros((B, C), dtype=np.int64)
    for i, r in enumerate(records):
        h = r["history"][-H:]
        history[i, : len(h)] = h
        c = r["candidates"][:C]
        candidates[i, : len(c)] = c
        labels[i, : len(c)] = r["labels"][:C]
    return {
        "item_ids": torch.from_numpy(history),
        "behaviors": torch.from_numpy(behaviors),
        "time_deltas": torch.from_numpy(time_deltas),
        "candidates": torch.from_numpy(candidates),
        "labels": torch.from_numpy(labels),
    }
