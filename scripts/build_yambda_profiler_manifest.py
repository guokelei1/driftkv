#!/usr/bin/env python3
"""Create a target-independent candidate manifest for compatibility profiling."""

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


def uid_bucket(uid: int, modulus: int) -> int:
    return ((uid * 2654435761) & 0xFFFFFFFF) % modulus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/yambda"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--bucket-modulus", type=int, default=16)
    parser.add_argument("--include-buckets", type=str, default="0,1,2,3")
    args = parser.parse_args()
    include_buckets = {int(value) for value in args.include_buckets.split(",") if value}
    if any(bucket < 0 or bucket >= args.bucket_modulus for bucket in include_buckets):
        raise ValueError("include-buckets must be in [0, bucket-modulus)")

    base_end = BASE_DAYS * DAY
    update_start = base_end + GAP + args.update_index * (WINDOW + GAP)
    update_end = update_start + WINDOW
    release_timestamp = update_end + GAP
    path = args.root / "flat/50m/listens.parquet"

    popularity = np.zeros(9_390_624, dtype=np.int64)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=262_144, columns=["timestamp", "item_id", "played_ratio_pct"]):
        timestamp = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        played = batch.column("played_ratio_pct").to_numpy(zero_copy_only=False).astype(np.int64)
        mask = (timestamp < base_end) & (played > 50)
        popularity += np.bincount(item[mask], minlength=len(popularity))
    popular_items = np.flatnonzero(popularity)
    popular_items = popular_items[np.argsort(-popularity[popular_items], kind="stable")]

    rows = []
    current_uid = None
    base_seen: set[int] = set()

    def consume(uid: int | None) -> None:
        if uid is None or uid_bucket(uid, args.bucket_modulus) not in include_buckets or len(base_seen) < 5:
            return
        candidates = []
        for item in popular_items:
            item = int(item)
            if item not in base_seen:
                candidates.append(item)
            if len(candidates) == 100:
                break
        if len(candidates) != 100:
            return
        rows.append({
            "request_id": f"yambda50m-v2-profiler-e{args.update_index + 1}-{len(rows):05d}",
            "uid": uid,
            "request_timestamp": release_timestamp,
            "candidate_item_ids": candidates,
            "candidate_size": 100,
            "retriever_version": "base_popularity_v1",
            "retriever_cutoff_timestamp": base_end,
            "target_injected": False,
            "candidate_protocol": "target_independent_compatibility_profiling",
            "candidate_hash_seed": 37,
            "selection_rule": f"uid_hash_bucket in {sorted(include_buckets)}; base_history_at_least_5",
        })

    for batch in parquet.iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id", "played_ratio_pct"]):
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        timestamp = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        played = batch.column("played_ratio_pct").to_numpy(zero_copy_only=False).astype(np.int64)
        boundaries = np.flatnonzero(uid[1:] != uid[:-1]) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, len(uid)]
        for start, end in zip(starts, ends):
            group_uid = int(uid[start])
            if current_uid is not None and group_uid != current_uid:
                consume(current_uid)
                base_seen = set()
            current_uid = group_uid
            group_timestamp = timestamp[start:end]
            group_item = item[start:end]
            group_played = played[start:end]
            base_seen.update(group_item[(group_timestamp < base_end) & (group_played > 50)].tolist())
    consume(current_uid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({
        "rows": len(rows),
        "update_index": args.update_index,
        "release_timestamp": release_timestamp,
        "candidate_protocol": "target_independent_compatibility_profiling",
        "target_injected": False,
        "path": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
