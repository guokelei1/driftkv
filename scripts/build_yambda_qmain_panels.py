#!/usr/bin/env python3
"""Materialize independent target-free panels from frozen Q_main."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from build_yambda_release_snapshot import EDGE, FOUNDATION_END, prepare_catalog_and_popularity


PANEL_COUNT = 32
PANEL_SIZE = 100
POOL_SIZE = 1000
RANK_DECAY_POWER = 0.5
GLOBAL_SEED = 20260818


def seed_for(edge: str, uid: int) -> int:
    raw = f"{GLOBAL_SEED}:{edge}:{uid}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge", choices=sorted(EDGE), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = Path("data/raw/yambda/flat/50m/listens.parquet")
    snapshot = pq.read_table(f"data/manifests/yambda50m_v2_release_snapshot_{args.edge}.parquet", columns=["uid", "state_hash"]).to_pydict()
    states = {int(uid): value for uid, value in zip(snapshot["uid"], snapshot["state_hash"])}
    _, popular = prepare_catalog_and_popularity(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    def flush(rows):
        nonlocal writer
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(args.output, table.schema, compression="zstd")
        writer.write_table(table)

    rows, current_uid, seen = [], None, set()
    def consume(uid: int, base_seen: set[int]) -> None:
        nonlocal rows
        if uid not in states:
            return
        pool = [int(item) for item in popular if int(item) not in base_seen][:POOL_SIZE]
        if len(pool) != POOL_SIZE:
            raise RuntimeError("insufficient Q_main pool")
        weights = np.arange(1, POOL_SIZE + 1, dtype=np.float64) ** (-RANK_DECAY_POWER)
        weights /= weights.sum()
        # Exponential races are exactly weighted sampling without replacement:
        # selecting the smallest -log(U)/w keys is equivalent to repeatedly
        # sampling from the remaining categorical proposal.  Vectorising all
        # 32 races keeps panel generation tractable without changing Q_main.
        keys = np.random.default_rng(seed_for(args.edge, uid)).exponential(size=(PANEL_COUNT, POOL_SIZE)) / weights
        selected = np.argpartition(keys, PANEL_SIZE - 1, axis=1)[:, :PANEL_SIZE]
        for panel_id, indices in enumerate(selected):
            chosen = np.asarray(pool, dtype=np.int64)[indices]
            rows.append({"edge_id": args.edge, "uid": uid, "panel_id": panel_id, "candidate_item_ids": chosen.astype(np.int64).tolist(), "snapshot_state_hash": states[uid]})
        if len(rows) >= 16_384:
            flush(rows)
            rows = []

    parquet = pq.ParquetFile(raw)
    for batch in parquet.iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id", "played_ratio_pct"]):
        for uid, timestamp, item, played in zip(batch.column("uid").to_numpy(zero_copy_only=False), batch.column("timestamp").to_numpy(zero_copy_only=False), batch.column("item_id").to_numpy(zero_copy_only=False), batch.column("played_ratio_pct").to_numpy(zero_copy_only=False)):
            uid, timestamp, item, played = int(uid), int(timestamp), int(item), int(played)
            if current_uid is not None and uid != current_uid:
                consume(current_uid, seen)
                seen = set()
            current_uid = uid
            if timestamp < FOUNDATION_END and played > 50:
                seen.add(item)
    if current_uid is not None:
        consume(current_uid, seen)
    flush(rows)
    if writer is None:
        raise RuntimeError("no panels written")
    writer.close()
    metadata = {"edge": args.edge, "distribution": "Q_main_rank_decay_v1", "global_seed": GLOBAL_SEED, "pool_size": POOL_SIZE, "panel_size": PANEL_SIZE, "panel_count": PANEL_COUNT, "rank_decay_power": RANK_DECAY_POWER, "target_injected": False, "catalog_cutoff": FOUNDATION_END}
    args.output.with_suffix(".meta.json").write_text(__import__("json").dumps(metadata, indent=2) + "\n")
    print(metadata)


if __name__ == "__main__":
    main()
