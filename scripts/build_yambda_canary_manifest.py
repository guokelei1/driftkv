#!/usr/bin/env python3
"""Build a small, deterministic base-popularity reranking manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DAY = 86_400
BASE_DAYS = 210
GAP = 1_800
WINDOW = DAY


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/yambda"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/yambda50m_v2_canary_candidates.jsonl"))
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--update-index", type=int, default=0)
    args = parser.parse_args()
    path = args.root / "flat/50m/listens.parquet"
    base_end = BASE_DAYS * DAY
    update_start = base_end + GAP
    update_start += args.update_index * (WINDOW + GAP)
    update_end = update_start + WINDOW
    future_start = update_end + GAP
    future_end = future_start + WINDOW

    popularity = np.zeros(9_390_624, dtype=np.int64)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=262_144, columns=["timestamp", "item_id", "played_ratio_pct"]):
        ts = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        played = batch.column("played_ratio_pct").to_numpy(zero_copy_only=False).astype(np.int64)
        mask = (ts < base_end) & (played > 50)
        popularity += np.bincount(item[mask], minlength=len(popularity))
    popular_items = np.flatnonzero(popularity)
    popular_items = popular_items[np.argsort(-popularity[popular_items], kind="stable")]

    rows = []
    current_uid = None
    base_seen: set[int] = set()
    future_positive = None

    def consume(uid: int | None) -> None:
        if uid is None or future_positive is None or len(base_seen) < 5:
            return
        target, request_timestamp = future_positive
        target = int(target)
        if popularity[target] == 0:
            return
        candidates = [target]
        for item in popular_items:
            item = int(item)
            if item not in base_seen and item != target:
                candidates.append(item)
            if len(candidates) == 100:
                break
        if len(candidates) < 100:
            return
        rows.append({
            "request_id": f"yambda50m-v2-canary-{len(rows):05d}",
            "uid": uid,
            "request_timestamp": int(request_timestamp),
            "positive_item_id": target,
            "candidate_item_ids": candidates,
            "candidate_size": 100,
            "retriever_version": "base_popularity_v1",
            "retriever_cutoff_timestamp": base_end,
            "target_injected": True,
            "candidate_protocol": "conditional_reranking",
            "candidate_hash_seed": 37,
        })

    for batch in parquet.iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id", "played_ratio_pct"]):
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        ts = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        played = batch.column("played_ratio_pct").to_numpy(zero_copy_only=False).astype(np.int64)
        boundaries = np.flatnonzero(uid[1:] != uid[:-1]) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, len(uid)]
        for start, end in zip(starts, ends):
            group_uid = int(uid[start])
            if current_uid is not None and group_uid != current_uid:
                consume(current_uid)
                if args.max_users is not None and len(rows) >= args.max_users:
                    break
                base_seen, future_positive = set(), None
            current_uid = group_uid
            group_ts, group_item, group_played = ts[start:end], item[start:end], played[start:end]
            base_seen.update(group_item[(group_ts < base_end) & (group_played > 50)].tolist())
            future_mask = (group_ts >= future_start) & (group_ts < future_end) & (group_played > 50)
            if future_positive is None and future_mask.any():
                first_future = np.flatnonzero(future_mask)[0]
                future_positive = (int(group_item[first_future]), int(group_ts[first_future]))
        if args.max_users is not None and len(rows) >= args.max_users:
            break
    if args.max_users is None or len(rows) < args.max_users:
        consume(current_uid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for row in rows if args.max_users is None else rows[:args.max_users]:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({
        "rows": len(rows) if args.max_users is None else min(len(rows), args.max_users),
        "base_days": BASE_DAYS,
        "release_gap_seconds": GAP,
        "update_window": "1d",
        "update_index": args.update_index,
        "future_gap_seconds": GAP,
        "candidate_size": 100,
        "path": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
