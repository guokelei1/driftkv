#!/usr/bin/env python3
"""Build deterministic, target-free popularity panel B for frozen snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from build_yambda_release_snapshot import EDGE, FOUNDATION_END, prepare_catalog_and_popularity


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def panel_b(popular: np.ndarray, seen: set[int]) -> list[int]:
    candidates, eligible_seen = [], 0
    for raw in popular:
        item = int(raw)
        if item in seen:
            continue
        if eligible_seen < 100:
            eligible_seen += 1
            continue
        candidates.append(item)
        if len(candidates) == 100:
            return candidates
    raise RuntimeError("insufficient disjoint popularity candidates")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge", choices=sorted(EDGE), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release = EDGE[args.edge][0]
    raw = Path("data/raw/yambda/flat/50m/listens.parquet")
    snapshot = pq.read_table(f"data/manifests/yambda50m_v2_release_snapshot_{args.edge}.parquet", columns=["uid", "state_hash"]).to_pydict()
    states = {int(uid): state_hash for uid, state_hash in zip(snapshot["uid"], snapshot["state_hash"])}
    _, popular = prepare_catalog_and_popularity(raw)
    rows, current_uid, seen = [], None, set()

    def consume(uid: int, base_seen: set[int]) -> None:
        if uid not in states:
            return
        candidates = panel_b(popular, base_seen)
        rows.append({
            "request_id": f"yambda50m-v2-cutover-panel-b-{args.edge}-{uid}",
            "uid": uid,
            "request_timestamp": release,
            "candidate_item_ids": candidates,
            "candidate_size": 100,
            "retriever_version": "base_popularity_panel_b_v1",
            "retriever_cutoff_timestamp": FOUNDATION_END,
            "target_injected": False,
            "candidate_protocol": "target_independent_cutover_probe_panel_b",
            "snapshot_state_hash": states[uid],
            "candidate_hash": digest(candidates),
        })

    parquet = pq.ParquetFile(raw)
    for batch in parquet.iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id", "played_ratio_pct"]):
        for uid, timestamp, item, played in zip(
            batch.column("uid").to_numpy(zero_copy_only=False),
            batch.column("timestamp").to_numpy(zero_copy_only=False),
            batch.column("item_id").to_numpy(zero_copy_only=False),
            batch.column("played_ratio_pct").to_numpy(zero_copy_only=False),
        ):
            uid, timestamp, item, played = int(uid), int(timestamp), int(item), int(played)
            if current_uid is not None and uid != current_uid:
                consume(current_uid, seen)
                seen = set()
            current_uid = uid
            if timestamp < FOUNDATION_END and played > 50:
                seen.add(item)
    if current_uid is not None:
        consume(current_uid, seen)
    if len(rows) != len(states):
        raise AssertionError("panel B must cover every frozen state")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({"edge": args.edge, "states": len(rows), "panel_hash": hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
