#!/usr/bin/env python3
"""Tiny theta0 training sanity on the frozen Yambda canary manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from hstu_kvcache.models import HSTU, HSTUConfig


def main() -> None:
    torch.manual_seed(37)
    manifest_path = Path("data/manifests/yambda50m_v2_canary_candidates.jsonl")
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    uids = {int(row["uid"]) for row in rows}
    histories: dict[int, list[tuple[int, int, int]]] = {uid: [] for uid in uids}
    path = Path("data/raw/yambda/flat/50m/listens.parquet")
    base_end = 210 * 86_400

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id", "is_organic"]):
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        ts = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        items = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        organic = batch.column("is_organic").to_numpy(zero_copy_only=False).astype(np.int64)
        mask = np.isin(uid, list(uids)) & (ts < base_end)
        for u, t, item, org in zip(uid[mask], ts[mask], items[mask], organic[mask]):
            history = histories[int(u)]
            delta = 0 if not history else min(max(int(t - history[-1][1]), 0), 86_400 * 7)
            history.append((int(item), int(t), int(1 + (1 - org))))
            if len(history) > 64:
                del history[0]

    raw_ids = set()
    for row in rows:
        raw_ids.update(row["candidate_item_ids"])
        raw_ids.update(item for item, _, _ in histories[int(row["uid"])])
    item_map = {raw: index + 1 for index, raw in enumerate(sorted(raw_ids))}
    sequences = []
    candidates = []
    for row in rows:
        history = histories[int(row["uid"])]
        if len(history) < 5:
            continue
        timestamps = np.asarray([x[1] for x in history], dtype=np.int64)
        deltas = np.zeros(len(history), dtype=np.float32)
        if len(history) > 1:
            deltas[1:] = np.diff(timestamps).clip(0, 86_400 * 7)
        sequences.append({
            "item_ids": np.asarray([item_map[x[0]] for x in history], dtype=np.int64),
            "behaviors": np.asarray([x[2] for x in history], dtype=np.int64),
            "time_deltas": deltas,
            "length": len(history),
        })
        candidates.append(np.asarray([item_map[x] for x in row["candidate_item_ids"]], dtype=np.int64))

    if len(sequences) < 64:
        raise RuntimeError(f"too few canary histories: {len(sequences)}")
    length = max(x["length"] for x in sequences)
    batch_items = np.zeros((len(sequences), length), dtype=np.int64)
    batch_behaviors = np.zeros_like(batch_items)
    batch_deltas = np.zeros((len(sequences), length), dtype=np.float32)
    lengths = np.zeros(len(sequences), dtype=np.int64)
    for i, sequence in enumerate(sequences):
        n = sequence["length"]
        batch_items[i, :n] = sequence["item_ids"]
        batch_behaviors[i, :n] = sequence["behaviors"]
        batch_deltas[i, :n] = sequence["time_deltas"]
        lengths[i] = n
    item_tensor = torch.from_numpy(batch_items)
    behavior_tensor = torch.from_numpy(batch_behaviors)
    delta_tensor = torch.from_numpy(batch_deltas)
    length_tensor = torch.from_numpy(lengths)
    candidate_tensor = torch.from_numpy(np.stack(candidates))

    cfg = HSTUConfig(
        num_items=len(item_map),
        num_behaviors=3,
        hidden_size=32,
        num_layers=2,
        num_heads=2,
        max_seq_len=max(64, length),
        input_dropout=0.0,
    )
    model = HSTU(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    def evaluate() -> tuple[float, float, float]:
        model.eval()
        with torch.no_grad():
            hidden, _ = model(item_tensor, behavior_tensor, delta_tensor, lengths=length_tensor)
            scores = model.score_candidates(hidden, candidate_tensor, length_tensor)
            no_history = torch.zeros_like(item_tensor[:, :1])
            no_behavior = torch.zeros_like(no_history)
            no_delta = torch.zeros_like(no_history, dtype=torch.float32)
            no_hidden, _ = model(no_history, no_behavior, no_delta, lengths=torch.zeros(len(sequences), dtype=torch.long))
            no_scores = model.score_candidates(no_hidden, candidate_tensor, torch.zeros(len(sequences), dtype=torch.long))
            loss = F.cross_entropy(scores, torch.zeros(len(sequences), dtype=torch.long))
            margin = (scores[:, 0] - scores[:, 1:].max(dim=1).values).mean()
            score_rms = (scores - no_scores).pow(2).mean().sqrt()
        return float(loss), float(margin), float(score_rms)

    before = evaluate()
    model.train()
    for _ in range(30):
        optimizer.zero_grad()
        hidden, _ = model(item_tensor, behavior_tensor, delta_tensor, lengths=length_tensor)
        scores = model.score_candidates(hidden, candidate_tensor, length_tensor)
        loss = F.cross_entropy(scores, torch.zeros(len(sequences), dtype=torch.long))
        loss.backward()
        optimizer.step()
    after = evaluate()
    result = {
        "status": "canary_only_not_formal_quality_result",
        "users": len(sequences),
        "history_max_len": int(length),
        "candidate_size": int(candidate_tensor.shape[1]),
        "theta0_steps": 30,
        "before": {"loss": before[0], "target_margin": before[1], "full_vs_no_history_score_rms": before[2]},
        "after": {"loss": after[0], "target_margin": after[1], "full_vs_no_history_score_rms": after[2]},
    }
    output = Path("results/data_audit/yambda50m_v2/theta0_canary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not after[0] < before[0]:
        raise AssertionError("theta0 canary loss did not improve")


if __name__ == "__main__":
    main()
