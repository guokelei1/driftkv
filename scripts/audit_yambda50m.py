#!/usr/bin/env python3
"""Run the first read-only Yambda-50M data opportunity audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


BIN_SECONDS = 5
DAY_BINS = 24 * 60 * 60 // BIN_SECONDS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    nonzero = values[values > 0]
    if len(nonzero) == 0:
        return {f"p{p}": 0.0 for p in (50, 90, 95, 99)}
    return {f"p{p}": float(np.percentile(nonzero, p)) for p in (50, 90, 95, 99)}


def user_opportunity_metrics(
    parquet: pq.ParquetFile,
    artist_by_item: np.ndarray,
    min_ts: int,
) -> list[dict[str, float | int]]:
    """Compute label-free recent-history overlap metrics one user at a time."""
    windows = (1, 3, 7)
    anchor_days = (300, 450, 600, 750, 900, 1050, 1200)
    observations = {
        (anchor, days): {
            "users": 0,
            "item_overlap": [],
            "artist_overlap": [],
            "previous_item_overlap": [],
            "previous_artist_overlap": [],
            "previous_users": 0,
            "last_item_hit": 0,
            "last_artist_hit": 0,
        }
        for anchor in anchor_days
        for days in windows
    }
    current_uid = None
    current_ts: list[np.ndarray] = []
    current_items: list[np.ndarray] = []

    def consume(uid: int, ts_parts: list[np.ndarray], item_parts: list[np.ndarray]) -> None:
        if not ts_parts:
            return
        ts = np.concatenate(ts_parts)
        items = np.concatenate(item_parts)
        for anchor in anchor_days:
            base_end = min_ts + anchor * DAY_BINS
            for days in windows:
                width = days * DAY_BINS
                base = items[ts < base_end]
                previous = items[(ts >= base_end - width) & (ts < base_end)]
                update = items[(ts >= base_end) & (ts < base_end + width)]
                future = items[(ts >= base_end + width) & (ts < base_end + 2 * width)]
                if len(base) == 0 or len(update) == 0 or len(future) == 0:
                    continue
                future_unique = np.unique(future)
                update_unique = np.unique(update)
                item_overlap = np.intersect1d(future_unique, update_unique).size / len(future_unique)
                update_artist = np.unique(artist_by_item[update_unique])
                future_artist = np.unique(artist_by_item[future_unique])
                update_artist = update_artist[update_artist >= 0]
                future_artist = future_artist[future_artist >= 0]
                artist_overlap = (
                    np.intersect1d(future_artist, update_artist).size / len(future_artist)
                    if len(future_artist) else 0.0
                )
                obs = observations[(anchor, days)]
                obs["users"] += 1
                obs["item_overlap"].append(float(item_overlap))
                obs["artist_overlap"].append(float(artist_overlap))
                if len(previous):
                    previous_unique = np.unique(previous)
                    previous_artist = np.unique(artist_by_item[previous_unique])
                    previous_artist = previous_artist[previous_artist >= 0]
                    obs["previous_users"] += 1
                    obs["previous_item_overlap"].append(
                        float(np.intersect1d(future_unique, previous_unique).size / len(future_unique))
                    )
                    obs["previous_artist_overlap"].append(
                        float(np.intersect1d(future_artist, previous_artist).size / len(future_artist))
                        if len(future_artist) else 0.0
                    )
                obs["last_item_hit"] += int(update[-1] in future_unique)
                last_artist = int(artist_by_item[update[-1]])
                obs["last_artist_hit"] += int(last_artist >= 0 and last_artist in future_artist)

    for batch in parquet.iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id"]):
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        ts = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        items = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        boundaries = np.flatnonzero(uid[1:] != uid[:-1]) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, len(uid)]
        for start, end in zip(starts, ends):
            group_uid = int(uid[start])
            if current_uid is not None and group_uid != current_uid:
                consume(current_uid, current_ts, current_items)
                current_ts, current_items = [], []
            current_uid = group_uid
            current_ts.append(ts[start:end])
            current_items.append(items[start:end])
    if current_uid is not None:
        consume(current_uid, current_ts, current_items)

    output = []
    for anchor in anchor_days:
        for days in windows:
            obs = observations[(anchor, days)]
            n = int(obs["users"])
            previous_n = int(obs["previous_users"])
            output.append({
                "anchor_days": anchor,
                "window_days": days,
                "users_with_base_update_future": n,
                "mean_future_item_overlap": float(np.mean(obs["item_overlap"])) if n else 0.0,
                "mean_future_artist_overlap": float(np.mean(obs["artist_overlap"])) if n else 0.0,
                "users_with_previous_update_future": previous_n,
                "mean_future_previous_item_overlap": float(np.mean(obs["previous_item_overlap"])) if previous_n else 0.0,
                "mean_future_previous_artist_overlap": float(np.mean(obs["previous_artist_overlap"])) if previous_n else 0.0,
                "last_item_hit_rate": obs["last_item_hit"] / n if n else 0.0,
                "last_artist_hit_rate": obs["last_artist_hit"] / n if n else 0.0,
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/yambda"))
    parser.add_argument("--output", type=Path, default=Path("results/data_audit/yambda50m_v1"))
    args = parser.parse_args()

    listens = args.root / "flat/50m/listens.parquet"
    if not listens.exists():
        raise SystemExit(f"missing input: {listens}")

    parquet = pq.ParquetFile(listens)
    user_counts = np.zeros(1_000_001, dtype=np.int64)
    item_counts = np.zeros(9_390_624, dtype=np.int64)
    organic_counts = np.zeros(2, dtype=np.int64)
    daily_events: dict[int, int] = defaultdict(int)
    daily_users: dict[int, set[int]] = defaultdict(set)
    daily_items: dict[int, set[int]] = defaultdict(set)
    min_ts, max_ts = None, None
    rows = 0
    sort_violations = 0
    previous_uid, previous_ts = None, None

    for batch in parquet.iter_batches(
        batch_size=262_144,
        columns=["uid", "timestamp", "item_id", "is_organic", "played_ratio_pct"],
    ):
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        ts = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        organic = batch.column("is_organic").to_numpy(zero_copy_only=False).astype(np.int64)
        rows += len(uid)
        user_counts += np.bincount(uid, minlength=len(user_counts))
        item_counts += np.bincount(item, minlength=len(item_counts))
        organic_counts += np.bincount(organic, minlength=2)
        min_ts = int(ts.min()) if min_ts is None else min(min_ts, int(ts.min()))
        max_ts = int(ts.max()) if max_ts is None else max(max_ts, int(ts.max()))
        if previous_uid is not None:
            sort_violations += int((uid[0] < previous_uid) or (uid[0] == previous_uid and ts[0] < previous_ts))
        if len(uid) > 1:
            sort_violations += int(np.sum((uid[1:] < uid[:-1]) | ((uid[1:] == uid[:-1]) & (ts[1:] < ts[:-1]))))
        previous_uid, previous_ts = int(uid[-1]), int(ts[-1])
        days = ts // DAY_BINS
        for day in np.unique(days):
            mask = days == day
            day_int = int(day)
            daily_events[day_int] += int(mask.sum())
            daily_users[day_int].update(uid[mask].tolist())
            daily_items[day_int].update(item[mask].tolist())

    if min_ts is None or max_ts is None:
        raise SystemExit("listens.parquet is empty")

    mapping_path = args.root / "artist_item_mapping.parquet"
    mapping_table = pq.read_table(mapping_path, columns=["artist_id", "item_id"])
    artist_by_item = np.full(9_390_624, -1, dtype=np.int64)
    mapping_items = mapping_table.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
    mapping_artists = mapping_table.column("artist_id").to_numpy(zero_copy_only=False).astype(np.int64)
    artist_by_item[mapping_items] = mapping_artists
    opportunity = user_opportunity_metrics(parquet, artist_by_item, min_ts)

    base_end = min_ts + 300 * DAY_BINS
    windows = []
    for days in (1, 3, 7):
        width = days * DAY_BINS
        base_users: set[int] = set()
        update_users: set[int] = set()
        future_users: set[int] = set()
        update_events = future_events = 0
        for day, users in daily_users.items():
            start = day * DAY_BINS
            if start < base_end:
                base_users.update(users)
            elif base_end <= start < base_end + width:
                update_users.update(users)
                update_events += daily_events[day]
            elif base_end + width <= start < base_end + 2 * width:
                future_users.update(users)
                future_events += daily_events[day]
        windows.append({
            "window_days": days,
            "base_users": len(base_users),
            "update_users": len(update_users),
            "future_users": len(future_users),
            "users_in_all_three": len(base_users & update_users & future_users),
            "update_events": update_events,
            "future_events": future_events,
        })

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "audit_version": "yambda50m_v1",
        "input": {
            "path": str(listens),
            "rows": rows,
            "bytes": listens.stat().st_size,
            "sha256": sha256(listens),
            "row_groups": parquet.num_row_groups,
        },
        "schema_checks": {
            "sort_violations_uid_timestamp": sort_violations,
            "timestamp_bin_seconds_assumed": BIN_SECONDS,
            "positive_rule": "played_ratio_pct > 50",
        },
        "time": {
            "min_timestamp_bin": min_ts,
            "max_timestamp_bin": max_ts,
            "span_days_at_5s_bins": (max_ts - min_ts) * BIN_SECONDS / 86400,
            "base_300d_end_bin": base_end,
        },
        "coverage": {
            "users": int(np.count_nonzero(user_counts)),
            "items": int(np.count_nonzero(item_counts)),
            "user_history_length": quantiles(user_counts),
            "organic_counts": {str(i): int(v) for i, v in enumerate(organic_counts)},
            "daily": [
                {"timestamp_bin_day": day, "events": daily_events[day], "users": len(daily_users[day]), "items": len(daily_items[day])}
                for day in sorted(daily_events)
            ],
        },
        "window_candidates": windows,
        "opportunity_metrics": opportunity,
        "status": "audit_observation_only_window_not_frozen",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "window_candidates.csv").open("w", newline="") as stream:
        fields = list(windows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(windows)
    print(json.dumps({"rows": rows, "time": report["time"], "windows": windows}, indent=2))


if __name__ == "__main__":
    main()
