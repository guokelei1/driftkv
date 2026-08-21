#!/usr/bin/env python3
"""Freeze all materialized parent-KV states at a Yambda release cutoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from train_yambda_theta0_medium import DAY, FOUNDATION_END, MAX_HISTORY
from train_yambda_two_edges import THETA1_RELEASE, THETA2_RELEASE


EDGE = {
    "theta0_theta1": (THETA1_RELEASE, "theta0", "checkpoints/yambda50m_v2_theta0_medium_batchfix_v3.pt"),
    "theta1_theta2": (THETA2_RELEASE, "theta1", "checkpoints/yambda50m_v2_theta1_medium_batchfix_v3.pt"),
}
MAX_ITEM_ID = 9_390_624


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_object(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def prepare_catalog_and_popularity(listens: Path):
    catalog = np.zeros(MAX_ITEM_ID, dtype=bool)
    popularity = np.zeros(MAX_ITEM_ID, dtype=np.int64)
    parquet = pq.ParquetFile(listens)
    for batch in parquet.iter_batches(batch_size=262_144, columns=["timestamp", "item_id", "played_ratio_pct"]):
        timestamp = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        played = batch.column("played_ratio_pct").to_numpy(zero_copy_only=False).astype(np.int64)
        base = timestamp < FOUNDATION_END
        catalog[item[base]] = True
        popular = base & (played > 50)
        popularity += np.bincount(item[popular], minlength=MAX_ITEM_ID)
    ordered = np.flatnonzero(popularity)
    return catalog, ordered[np.argsort(-popularity[ordered], kind="stable")]


def candidate_panel(popular_items, base_seen: set[int]) -> list[int]:
    candidates = []
    for raw_item in popular_items:
        item = int(raw_item)
        if item not in base_seen:
            candidates.append(item)
        if len(candidates) == 100:
            return candidates
    raise RuntimeError("insufficient base-popularity candidates")


def enrich_artist_counts(tmp_snapshot: Path, mapping: Path) -> dict[int, int]:
    connection = duckdb.connect()
    query = """
        SELECT snapshot.uid, COUNT(DISTINCT mapping.artist_id)::BIGINT AS artists
        FROM read_parquet(?) AS snapshot
        CROSS JOIN UNNEST(snapshot.recent_item_ids_last_7d) AS items(item_id)
        JOIN read_parquet(?) AS mapping ON mapping.item_id = items.item_id
        GROUP BY snapshot.uid
    """
    try:
        return {int(uid): int(count) for uid, count in connection.execute(query, [str(tmp_snapshot), str(mapping)]).fetchall()}
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge", choices=sorted(EDGE), required=True)
    parser.add_argument("--root", type=Path, default=Path("data/raw/yambda"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-output", type=Path, required=True)
    args = parser.parse_args()
    release, parent_version, checkpoint_name = EDGE[args.edge]
    listens = args.root / "flat/50m/listens.parquet"
    artist_mapping = args.root / "artist_item_mapping.parquet"
    checkpoint = Path(checkpoint_name)
    catalog, popular_items = prepare_catalog_and_popularity(listens)
    checkpoint_hash = sha256_file(checkpoint)
    rows, probe_rows = [], []

    def consume(uid: int, events: list[tuple[int, int, int, int]]) -> None:
        prefix = [event for event in events if event[1] < release]
        if not prefix:
            return
        model_prefix = [event for event in prefix if catalog[event[0]]]
        if not model_prefix:
            return
        effective = model_prefix[-MAX_HISTORY:]
        raw_last = prefix[-1]
        base_seen = {event[0] for event in prefix if event[1] < FOUNDATION_END and event[3] > 50}
        candidates = candidate_panel(popular_items, base_seen)
        recent_1d = [event for event in prefix if event[1] >= release - DAY]
        recent_7d = [event for event in prefix if event[1] >= release - 7 * DAY]
        recent_30d = [event for event in prefix if event[1] >= release - 30 * DAY]
        recent_items = [event[0] for event in recent_7d]
        unique_recent = len(set(recent_items))
        state_events = [[event[0], event[1], event[2]] for event in effective]
        state_hash = digest_object(state_events)
        panel_hash = digest_object(candidates)
        row = {
            "edge_id": args.edge,
            "uid": uid,
            "release_cutoff": release,
            "parent_model_version": parent_version,
            "state_timestamp": effective[-1][1],
            "last_activity_timestamp": raw_last[1],
            "last_activity_age_seconds": release - raw_last[1],
            "raw_prefix_length": len(prefix),
            "effective_prefix_length": len(effective),
            "state_hash": state_hash,
            "parent_checkpoint_hash": checkpoint_hash,
            "events_last_1d": len(recent_1d),
            "events_last_7d": len(recent_7d),
            "events_last_30d": len(recent_30d),
            "unique_items_last_7d": unique_recent,
            "repeat_ratio_last_7d": 0.0 if not recent_7d else 1.0 - unique_recent / len(recent_7d),
            "organic_ratio_last_7d": 0.0 if not recent_7d else float(np.mean([event[2] == 1 for event in recent_7d])),
            "exact_token_layer_work": len(effective) * 4,
            "probe_candidate_hash": panel_hash,
            "recent_item_ids_last_7d": recent_items,
        }
        rows.append(row)
        probe_rows.append({
            "request_id": f"yambda50m-v2-cutover-{args.edge}-{uid}",
            "uid": uid,
            "request_timestamp": release,
            "candidate_item_ids": candidates,
            "candidate_size": len(candidates),
            "retriever_version": "base_popularity_v1",
            "retriever_cutoff_timestamp": FOUNDATION_END,
            "target_injected": False,
            "candidate_protocol": "target_independent_cutover_probe",
            "snapshot_state_hash": state_hash,
            "candidate_hash": panel_hash,
        })

    current_uid = None
    current_events: list[tuple[int, int, int, int]] = []
    parquet = pq.ParquetFile(listens)
    for batch in parquet.iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id", "is_organic", "played_ratio_pct"]):
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        timestamp = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        organic = batch.column("is_organic").to_numpy(zero_copy_only=False).astype(np.int64)
        played = batch.column("played_ratio_pct").to_numpy(zero_copy_only=False).astype(np.int64)
        for u, t, i, o, p in zip(uid, timestamp, item, organic, played):
            u = int(u)
            if current_uid is not None and u != current_uid:
                consume(current_uid, current_events)
                current_events = []
            current_uid = u
            if int(t) < release:
                current_events.append((int(i), int(t), int(1 + (1 - int(o))), int(p)))
    if current_uid is not None:
        consume(current_uid, current_events)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".artist_tmp.parquet")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    artists = enrich_artist_counts(temporary, artist_mapping)
    for row in rows:
        row["unique_artists_last_7d"] = artists.get(row["uid"], 0)
        del row["recent_item_ids_last_7d"]
    pq.write_table(pa.Table.from_pylist(rows), args.output, compression="zstd")
    temporary.unlink()
    with args.probe_output.open("w") as stream:
        for row in probe_rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    metadata = {
        "edge_id": args.edge,
        "release_cutoff": release,
        "primary_population": "all_materialized_states_at_release",
        "eligibility": {
            "state_timestamp_lte_release_cutoff": True,
            "state_model_version": "exact_parent_version",
            "min_effective_prefix_tokens": 1,
            "require_future_request": False,
            "require_future_append": False,
            "require_future_target": False,
        },
        "history_cap": MAX_HISTORY,
        "state_count": len(rows),
        "snapshot_hash": sha256_file(args.output),
        "probe_manifest_hash": sha256_file(args.probe_output),
        "parent_checkpoint_hash": checkpoint_hash,
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
