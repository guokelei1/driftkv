#!/usr/bin/env python3
"""Timestamp-corrected, user-paired Yambda-50M opportunity audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DAY_SECONDS = 86_400
RELEASE_GAP_SECONDS = 1_800
ANCHORS_DAYS = (180, 210, 225, 240, 255, 270)
WINDOWS = (("12h", 12 * 3600), ("1d", 24 * 3600), ("3d", 3 * 24 * 3600), ("7d", 7 * 24 * 3600))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    nonzero = values[values > 0]
    return {f"p{p}": float(np.percentile(nonzero, p)) if len(nonzero) else 0.0 for p in (50, 90, 95, 99)}


def bootstrap_ci(values: list[float], seed: int = 37, reps: int = 2000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        means[i] = x[rng.integers(0, len(x), len(x))].mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/yambda"))
    parser.add_argument("--output", type=Path, default=Path("results/data_audit/yambda50m_v2"))
    args = parser.parse_args()
    listens_path = args.root / "flat/50m/listens.parquet"
    if not listens_path.exists():
        raise SystemExit(f"missing input: {listens_path}")

    parquet = pq.ParquetFile(listens_path)
    event_user_sets = {}
    for event_name in ("listens", "likes", "dislikes"):
        event_path = args.root / f"flat/50m/{event_name}.parquet"
        if event_path.exists():
            event_user_sets[event_name] = set(
                pq.read_table(event_path, columns=["uid"]).column("uid").to_numpy().tolist()
            )
    user_counts = np.zeros(1_000_001, dtype=np.int64)
    item_counts = np.zeros(9_390_624, dtype=np.int64)
    organic_counts = np.zeros(2, dtype=np.int64)
    positive_count = 0
    timestamp_mod_5 = 0
    daily_events: dict[int, int] = defaultdict(int)
    daily_users: dict[int, set[int]] = defaultdict(set)
    daily_items: dict[int, set[int]] = defaultdict(set)
    min_ts = max_ts = None
    rows = 0
    sort_violations = 0
    previous_uid = previous_ts = None

    for batch in parquet.iter_batches(
        batch_size=262_144,
        columns=["uid", "timestamp", "item_id", "is_organic", "played_ratio_pct"],
    ):
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        ts = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        items = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        organic = batch.column("is_organic").to_numpy(zero_copy_only=False).astype(np.int64)
        played = batch.column("played_ratio_pct").to_numpy(zero_copy_only=False).astype(np.int64)
        rows += len(uid)
        user_counts += np.bincount(uid, minlength=len(user_counts))
        item_counts += np.bincount(items, minlength=len(item_counts))
        organic_counts += np.bincount(organic, minlength=2)
        positive_count += int((played > 50).sum())
        timestamp_mod_5 += int((ts % 5 == 0).sum())
        min_ts = int(ts.min()) if min_ts is None else min(min_ts, int(ts.min()))
        max_ts = int(ts.max()) if max_ts is None else max(max_ts, int(ts.max()))
        if previous_uid is not None:
            sort_violations += int(uid[0] < previous_uid or (uid[0] == previous_uid and ts[0] < previous_ts))
        if len(uid) > 1:
            sort_violations += int(np.sum((uid[1:] < uid[:-1]) | ((uid[1:] == uid[:-1]) & (ts[1:] < ts[:-1]))))
        previous_uid, previous_ts = int(uid[-1]), int(ts[-1])
        day_ids = ts // DAY_SECONDS
        for day in np.unique(day_ids):
            mask = day_ids == day
            day = int(day)
            daily_events[day] += int(mask.sum())
            daily_users[day].update(uid[mask].tolist())
            daily_items[day].update(items[mask].tolist())

    if min_ts is None or max_ts is None:
        raise SystemExit("empty listens parquet")

    mapping_path = args.root / "artist_item_mapping.parquet"
    mapping = pq.read_table(mapping_path, columns=["artist_id", "item_id"])
    artist_by_item = np.full(9_390_624, -1, dtype=np.int64)
    mapping_items = mapping.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
    mapping_artists = mapping.column("artist_id").to_numpy(zero_copy_only=False).astype(np.int64)
    artist_by_item[mapping_items] = mapping_artists
    observed_items = np.flatnonzero(item_counts)
    observed_artists = np.unique(artist_by_item[observed_items])
    observed_artists = observed_artists[observed_artists >= 0]

    observations: dict[tuple[int, str], dict[str, list | int]] = {}
    for anchor in ANCHORS_DAYS:
        for label, width in WINDOWS:
            observations[(anchor, label)] = {
                "users": 0, "previous_users": 0, "update_events": [], "future_events": [],
                "update_unique_items": [], "future_unique_items": [], "item_delta": [], "artist_delta": [],
                "last_item_hit": 0, "last_artist_hit": 0,
            }

    current_uid = None
    ts_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []

    def consume(user_ts_parts: list[np.ndarray], user_item_parts: list[np.ndarray]) -> None:
        if not user_ts_parts:
            return
        ts = np.concatenate(user_ts_parts)
        items = np.concatenate(user_item_parts)
        for anchor in ANCHORS_DAYS:
            anchor_end = min_ts + anchor * DAY_SECONDS
            for label, width in WINDOWS:
                previous = items[(ts >= anchor_end - width) & (ts < anchor_end)]
                update_start = anchor_end + RELEASE_GAP_SECONDS
                update = items[(ts >= update_start) & (ts < update_start + width)]
                future = items[(ts >= update_start + width) & (ts < update_start + 2 * width)]
                base = items[ts < anchor_end]
                if len(base) == 0 or len(update) == 0 or len(future) == 0:
                    continue
                update_unique = np.unique(update)
                future_unique = np.unique(future)
                update_artists = np.unique(artist_by_item[update_unique])
                future_artists = np.unique(artist_by_item[future_unique])
                update_artists = update_artists[update_artists >= 0]
                future_artists = future_artists[future_artists >= 0]
                current_item_overlap = np.intersect1d(future_unique, update_unique).size / len(future_unique)
                current_artist_overlap = (
                    np.intersect1d(future_artists, update_artists).size / len(future_artists)
                    if len(future_artists) else 0.0
                )
                obs = observations[(anchor, label)]
                obs["users"] += 1
                obs["update_events"].append(len(update))
                obs["future_events"].append(len(future))
                obs["update_unique_items"].append(len(update_unique))
                obs["future_unique_items"].append(len(future_unique))
                if len(previous):
                    previous_unique = np.unique(previous)
                    previous_artists = np.unique(artist_by_item[previous_unique])
                    previous_artists = previous_artists[previous_artists >= 0]
                    previous_item_overlap = np.intersect1d(future_unique, previous_unique).size / len(future_unique)
                    previous_artist_overlap = (
                        np.intersect1d(future_artists, previous_artists).size / len(future_artists)
                        if len(future_artists) else 0.0
                    )
                    obs["previous_users"] += 1
                    obs["item_delta"].append(float(current_item_overlap - previous_item_overlap))
                    obs["artist_delta"].append(float(current_artist_overlap - previous_artist_overlap))
                obs["last_item_hit"] += int(update[-1] in future_unique)
                last_artist = int(artist_by_item[update[-1]])
                obs["last_artist_hit"] += int(last_artist >= 0 and last_artist in future_artists)

    for batch in parquet.iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id"]):
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        ts = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        items = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        boundaries = np.flatnonzero(uid[1:] != uid[:-1]) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, len(uid)]
        for start, end in zip(starts, ends):
            if current_uid is not None and int(uid[start]) != current_uid:
                consume(ts_parts, item_parts)
                ts_parts, item_parts = [], []
            current_uid = int(uid[start])
            ts_parts.append(ts[start:end])
            item_parts.append(items[start:end])
    consume(ts_parts, item_parts)

    candidate_rows = []
    for anchor in ANCHORS_DAYS:
        for label, width in WINDOWS:
            obs = observations[(anchor, label)]
            n = int(obs["users"])
            pn = int(obs["previous_users"])
            item_delta = obs["item_delta"]
            artist_delta = obs["artist_delta"]
            candidate_rows.append({
                "anchor_days": anchor,
                "window": label,
                "window_hours": width / 3600,
                "users_with_base_update_future": n,
                "users_with_previous_update_future": pn,
                "mean_item_delta": float(np.mean(item_delta)) if pn else 0.0,
                "median_item_delta": float(np.median(item_delta)) if pn else 0.0,
                "item_delta_positive_fraction": float(np.mean(np.asarray(item_delta) > 0)) if pn else 0.0,
                "item_delta_bootstrap_ci95_low": bootstrap_ci(item_delta, seed=37 + anchor)[0] if pn else 0.0,
                "item_delta_bootstrap_ci95_high": bootstrap_ci(item_delta, seed=37 + anchor)[1] if pn else 0.0,
                "mean_artist_delta": float(np.mean(artist_delta)) if pn else 0.0,
                "median_artist_delta": float(np.median(artist_delta)) if pn else 0.0,
                "artist_delta_positive_fraction": float(np.mean(np.asarray(artist_delta) > 0)) if pn else 0.0,
                "artist_delta_bootstrap_ci95_low": bootstrap_ci(artist_delta, seed=73 + anchor)[0] if pn else 0.0,
                "artist_delta_bootstrap_ci95_high": bootstrap_ci(artist_delta, seed=73 + anchor)[1] if pn else 0.0,
                "mean_update_events": float(np.mean(obs["update_events"])) if n else 0.0,
                "mean_future_events": float(np.mean(obs["future_events"])) if n else 0.0,
                "mean_update_unique_items": float(np.mean(obs["update_unique_items"])) if n else 0.0,
                "mean_future_unique_items": float(np.mean(obs["future_unique_items"])) if n else 0.0,
                "last_item_hit_rate": obs["last_item_hit"] / n if n else 0.0,
                "last_artist_hit_rate": obs["last_artist_hit"] / n if n else 0.0,
            })

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "audit_version": "yambda50m_v2",
        "input": {"path": str(listens_path), "rows": rows, "bytes": listens_path.stat().st_size, "sha256": sha256(listens_path), "row_groups": parquet.num_row_groups},
        "timestamp_contract": {
            "stored_value_unit": "seconds",
            "precision_seconds": 5,
            "min_timestamp": min_ts,
            "max_timestamp": max_ts,
            "span_days": (max_ts - min_ts) / DAY_SECONDS,
            "timestamp_multiple_of_5_fraction": timestamp_mod_5 / rows,
            "day_id": "timestamp // 86400",
        },
        "schema_checks": {"sort_violations_uid_timestamp": sort_violations, "positive_rule": "played_ratio_pct > 50"},
        "coverage": {
            "raw_listen_users": int(np.count_nonzero(user_counts)),
            "event_user_counts": {name: len(users) for name, users in event_user_sets.items()},
            "all_downloaded_event_users": len(set().union(*event_user_sets.values())),
            "observed_listen_items": int(len(observed_items)),
            "observed_listen_artists": int(len(observed_artists)),
            "user_history_length": quantiles(user_counts),
            "positive_listen_events": positive_count,
            "positive_listen_fraction": positive_count / rows,
            "organic_counts": {str(i): int(v) for i, v in enumerate(organic_counts)},
            "daily_summary": {
                "num_days": len(daily_events),
                "events_min": min(daily_events.values()),
                "events_median": float(np.median(list(daily_events.values()))),
                "events_max": max(daily_events.values()),
                "users_min": min(map(len, daily_users.values())),
                "users_median": float(np.median([len(x) for x in daily_users.values()])),
                "users_max": max(map(len, daily_users.values())),
                "missing_day_bins": int(max(daily_events) - min(daily_events) + 1 - len(daily_events)),
            },
        },
        "release_gap_seconds": RELEASE_GAP_SECONDS,
        "anchors_days": list(ANCHORS_DAYS),
        "window_candidates": candidate_rows,
        "status": "timestamp_corrected_observation_protocol_pending_final_freeze",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "window_candidates.csv").open("w", newline="") as stream:
        fields = list(candidate_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate_rows)
    print(json.dumps({"timestamp_contract": report["timestamp_contract"], "daily_summary": report["coverage"]["daily_summary"], "windows": candidate_rows}, indent=2))


if __name__ == "__main__":
    main()
