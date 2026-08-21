#!/usr/bin/env python3
"""P7.4 user-level split coverage audit; no model fitting or scoring."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/yambda"
OUTPUT = ROOT / "results/data_audit/yambda50m_p7/split_coverage_audit_v1.json"
DAY = 86_400
RECENT = 32
CONTEXT = 512
RETURN_GAP = 3 * DAY
SPLITS = {
    "base_fit": (0, 180 * DAY),
    "residual_train": (180 * DAY, 203 * DAY),
    "development": (203 * DAY, 210 * DAY),
    "qualification": (210 * DAY, 217 * DAY),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_name(timestamp: int) -> str | None:
    for name, (start, end) in SPLITS.items():
        if start <= timestamp < end:
            return name
    return None


@dataclass
class IntegerDistribution:
    maximum: int = CONTEXT
    counts: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.counts = np.zeros(self.maximum + 1, dtype=np.int64)

    def add(self, value: int, count: int = 1) -> None:
        if not 0 <= value <= self.maximum:
            raise ValueError(f"integer statistic {value} is outside [0, {self.maximum}]")
        self.counts[value] += count

    def summary(self) -> dict:
        total = int(self.counts.sum())
        nonzero = np.flatnonzero(self.counts)
        result = {"count": total, "max": int(nonzero[-1]) if len(nonzero) else None}
        cumulative = np.cumsum(self.counts)
        for q in (50, 90, 95, 99):
            threshold = int(np.ceil(total * q / 100))
            result[f"p{q}"] = int(np.searchsorted(cumulative, threshold)) if total else None
        return result


@dataclass
class CoverageBucket:
    query_count: int = 0
    user_query_counts: list[int] = field(default_factory=list)
    rankable_user_query_counts: list[int] = field(default_factory=list)
    history_lengths: IntegerDistribution = field(default_factory=IntegerDistribution)
    candidate_counts: IntegerDistribution = field(default_factory=IntegerDistribution)
    target_strata: Counter = field(default_factory=Counter)
    feature_candidate_rows: int = 0
    artist_missing_rows: int = 0
    zero_base_popularity_rows: int = 0
    cohorts: Counter = field(default_factory=Counter)

    def add_user_count(self, count: int) -> None:
        if count:
            self.user_query_counts.append(count)

    def add_rankable_user_count(self, count: int) -> None:
        if count:
            self.rankable_user_query_counts.append(count)

    def summary(self, *, feature_scope: str) -> dict:
        counts = np.asarray(self.user_query_counts, dtype=np.int64)
        rankable_counts = np.asarray(self.rankable_user_query_counts, dtype=np.int64)
        strata_total = sum(self.target_strata.values())
        return {
            "query_count": self.query_count,
            "unique_users": len(counts),
            "queries_per_participating_user": {
                f"p{q}": float(np.percentile(counts, q)) if len(counts) else None
                for q in (50, 90, 99)
            },
            "rankable_unique_users": len(rankable_counts),
            "rankable_queries_per_participating_user": {
                f"p{q}": float(np.percentile(rankable_counts, q))
                if len(rankable_counts)
                else None
                for q in (50, 90, 99)
            },
            "history_length": self.history_lengths.summary(),
            "candidate_count": self.candidate_counts.summary(),
            "target_strata": dict(self.target_strata),
            "target_strata_fractions": {
                key: value / strata_total if strata_total else None
                for key, value in self.target_strata.items()
            },
            "base_feature_coverage": {
                "scope": feature_scope,
                "candidate_rows": self.feature_candidate_rows,
                "artist_missing_rate": (
                    self.artist_missing_rows / self.feature_candidate_rows
                    if self.feature_candidate_rows
                    else None
                ),
                "zero_global_popularity_at_base_fit_cutoff_rate": (
                    self.zero_base_popularity_rows / self.feature_candidate_rows
                    if self.feature_candidate_rows
                    else None
                ),
            },
            "cohorts": dict(self.cohorts),
            "cohort_fractions": {
                key: value / self.query_count if self.query_count else None
                for key, value in self.cohorts.items()
            },
        }


def load_artist_map() -> np.ndarray:
    table = pq.read_table(RAW / "artist_item_mapping.parquet", columns=["artist_id", "item_id"])
    items = table.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
    artists = table.column("artist_id").to_numpy(zero_copy_only=False).astype(np.int64)
    output = np.full(int(items.max()) + 1, -1, dtype=np.int64)
    output[items] = artists
    return output


def base_fit_popularity(size: int) -> np.ndarray:
    output = np.zeros(size, dtype=np.int64)
    parquet = pq.ParquetFile(RAW / "flat/50m/listens.parquet")
    for batch in parquet.iter_batches(batch_size=524_288, columns=["timestamp", "item_id"]):
        timestamp = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        items = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        selected = items[timestamp < 180 * DAY]
        output += np.bincount(selected, minlength=size)
    return output


def load_feedback() -> dict[int, list[tuple[int, int, int, int]]]:
    result: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for label, name in ((1, "likes"), (0, "dislikes")):
        table = pq.read_table(
            RAW / f"flat/50m/{name}.parquet",
            columns=["uid", "timestamp", "item_id", "is_organic"],
        )
        columns = [table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names]
        for uid, timestamp, item, organic in zip(*columns, strict=True):
            result[int(uid)].append((int(timestamp), int(item), label, int(organic)))
    for values in result.values():
        values.sort()
    return result


def target_stratum(previous_position: int | None, prefix_end: int) -> str:
    if previous_position is None:
        return "never_seen"
    if previous_position >= prefix_end - RECENT:
        return "recent_seen"
    if previous_position >= prefix_end - CONTEXT:
        return "old_seen"
    return "seen_only_before_512"


def process_feedback(
    uid: int,
    timestamps: np.ndarray,
    items: np.ndarray,
    events: list[tuple[int, int, int, int]],
    artist_by_item: np.ndarray,
    popularity: np.ndarray,
    buckets: dict[str, CoverageBucket],
) -> None:
    if not events:
        return
    positions_by_item: dict[int, list[int]] = defaultdict(list)
    for position, item in enumerate(items):
        positions_by_item[int(item)].append(position)
    local = Counter()
    for timestamp, candidate, label, organic in events:
        split = split_name(timestamp)
        if split is None:
            continue
        prefix_end = int(np.searchsorted(timestamps, timestamp, side="left"))
        if prefix_end == 0:
            continue
        positions = positions_by_item.get(candidate, ())
        earlier_count = bisect.bisect_left(positions, prefix_end)
        previous_position = positions[earlier_count - 1] if earlier_count else None
        bucket = buckets[split]
        bucket.query_count += 1
        local[split] += 1
        bucket.history_lengths.add(min(prefix_end, CONTEXT))
        bucket.candidate_counts.add(1)
        bucket.target_strata[target_stratum(previous_position, prefix_end)] += 1
        bucket.feature_candidate_rows += 1
        artist = int(artist_by_item[candidate]) if candidate < len(artist_by_item) else -1
        bucket.artist_missing_rows += int(artist < 0)
        bucket.zero_base_popularity_rows += int(candidate >= len(popularity) or popularity[candidate] == 0)
        bucket.cohorts["like" if label else "dislike"] += 1
        bucket.cohorts["organic" if organic else "recommendation_driven"] += 1
        latest = int(items[prefix_end - 1]) == candidate
        prior_30m = previous_position is not None and timestamp - timestamps[previous_position] <= 1_800
        bucket.cohorts["latest_item"] += int(latest)
        bucket.cohorts["prior_30m_same_item"] += int(prior_30m)
        bucket.cohorts["non_prior_30m_same_item"] += int(not prior_30m)
    for split, count in local.items():
        buckets[split].add_user_count(count)


def process_listen_user(
    timestamps: np.ndarray,
    items: np.ndarray,
    artist_by_item: np.ndarray,
    popularity: np.ndarray,
    n_buckets: dict[str, CoverageBucket],
    r_buckets: dict[str, CoverageBucket],
) -> None:
    n_local, r_local, r_rankable_local = Counter(), Counter(), Counter()
    window: deque[tuple[int, int]] = deque()
    window_counts: Counter[int] = Counter()
    last_position: dict[int, int] = {}
    boundaries = np.r_[0, np.flatnonzero(timestamps[1:] != timestamps[:-1]) + 1, len(timestamps)]
    for boundary_index in range(len(boundaries) - 1):
        start, end = int(boundaries[boundary_index]), int(boundaries[boundary_index + 1])
        timestamp = int(timestamps[start])
        split = split_name(timestamp)
        if split is not None and start > 0:
            bucket = n_buckets[split]
            for position in range(start, end):
                target = int(items[position])
                bucket.query_count += 1
                n_local[split] += 1
                bucket.history_lengths.add(min(start, CONTEXT))
                bucket.candidate_counts.add(100)
                bucket.target_strata[target_stratum(last_position.get(target), start)] += 1
                bucket.feature_candidate_rows += 1
                artist = int(artist_by_item[target]) if target < len(artist_by_item) else -1
                bucket.artist_missing_rows += int(artist < 0)
                bucket.zero_base_popularity_rows += int(
                    target >= len(popularity) or popularity[target] == 0
                )

        previous_timestamp = int(timestamps[start - 1]) if start else None
        gap = timestamp - previous_timestamp if previous_timestamp is not None else 0
        if split is not None and gap >= RETURN_GAP and len(window_counts) >= 2:
            bucket = r_buckets[split]
            target = int(items[start])
            bucket.query_count += 1
            r_local[split] += 1
            bucket.history_lengths.add(len(window))
            bucket.candidate_counts.add(len(window_counts))
            stratum = target_stratum(last_position.get(target), start)
            bucket.target_strata[stratum] += 1
            bucket.cohorts["all_eligible_session_starts"] += 1
            rankable = target in window_counts
            bucket.cohorts["rankable_familiar_return"] += int(rankable)
            r_rankable_local[split] += int(rankable)
            for candidate in window_counts:
                bucket.feature_candidate_rows += 1
                artist = (
                    int(artist_by_item[candidate]) if candidate < len(artist_by_item) else -1
                )
                bucket.artist_missing_rows += int(artist < 0)
                bucket.zero_base_popularity_rows += int(
                    candidate >= len(popularity) or popularity[candidate] == 0
                )

        for position in range(start, end):
            item = int(items[position])
            window.append((item, position))
            window_counts[item] += 1
            last_position[item] = position
            if len(window) > CONTEXT:
                old_item, _ = window.popleft()
                window_counts[old_item] -= 1
                if window_counts[old_item] == 0:
                    del window_counts[old_item]

    for split, count in n_local.items():
        n_buckets[split].add_user_count(count)
    for split, count in r_local.items():
        r_buckets[split].add_user_count(count)
    for split, count in r_rankable_local.items():
        r_buckets[split].add_rankable_user_count(count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args()

    artist_by_item = load_artist_map()
    popularity = base_fit_popularity(len(artist_by_item))
    feedback = load_feedback()
    buckets = {
        workload: {split: CoverageBucket() for split in SPLITS}
        for workload in ("N", "R", "F")
    }
    parquet = pq.ParquetFile(RAW / "flat/50m/listens.parquet")
    current_uid: int | None = None
    timestamp_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []
    users = 0

    def consume() -> bool:
        nonlocal users
        if current_uid is None:
            return True
        timestamps = np.concatenate(timestamp_parts).astype(np.int64, copy=False)
        items = np.concatenate(item_parts).astype(np.int64, copy=False)
        process_listen_user(timestamps, items, artist_by_item, popularity, buckets["N"], buckets["R"])
        process_feedback(
            current_uid,
            timestamps,
            items,
            feedback.pop(current_uid, []),
            artist_by_item,
            popularity,
            buckets["F"],
        )
        users += 1
        return args.max_users is None or users < args.max_users

    stop = False
    for batch in parquet.iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id"]):
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        timestamp = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        boundaries = np.flatnonzero(uid[1:] != uid[:-1]) + 1
        starts, ends = np.r_[0, boundaries], np.r_[boundaries, len(uid)]
        for start, end in zip(starts, ends, strict=True):
            user = int(uid[start])
            if current_uid is not None and user != current_uid:
                if not consume():
                    stop = True
                    break
                timestamp_parts.clear()
                item_parts.clear()
            current_uid = user
            timestamp_parts.append(timestamp[start:end])
            item_parts.append(item[start:end])
        if stop:
            break
    if not stop:
        consume()

    feature_scopes = {
        "N": "observed target proxy; full Q_main candidate rows require P7.5 canary",
        "R": "complete familiar candidate universe",
        "F": "observed feedback candidate",
    }
    report = {
        "contract": "p7_4_training_contract_v1",
        "stage": "split_coverage_audit_v1",
        "status": "completed_full_audit" if args.max_users is None else "canary_only",
        "users_with_listens": users,
        "max_users": args.max_users,
        "splits_seconds": {key: list(value) for key, value in SPLITS.items()},
        "base_popularity_cutoff_seconds": 180 * DAY,
        "workloads": {
            workload: {
                split: bucket.summary(feature_scope=feature_scopes[workload])
                for split, bucket in split_buckets.items()
            }
            for workload, split_buckets in buckets.items()
        },
        "remaining_feedback_users_without_listens": len(feedback),
        "source_sha256": {
            name: sha256(path)
            for name, path in {
                "listens": RAW / "flat/50m/listens.parquet",
                "likes": RAW / "flat/50m/likes.parquet",
                "dislikes": RAW / "flat/50m/dislikes.parquet",
                "artist_mapping": RAW / "artist_item_mapping.parquet",
            }.items()
        },
        "limitations": [
            "N candidate-level feature coverage is deferred to the frozen Q_main canary; target-side coverage is reported here.",
            "Counts are descriptive coverage evidence and do not authorize HSTU training or open qualification.",
            "Request counts never replace users as the inferential unit.",
        ],
    }
    for split in SPLITS:
        r = report["workloads"]["R"][split]
        eligible = r["cohorts"].get("all_eligible_session_starts", 0)
        rankable = r["cohorts"].get("rankable_familiar_return", 0)
        r["rankable_coverage"] = rankable / eligible if eligible else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "users": users,
                "R_residual_train_queries": report["workloads"]["R"]["residual_train"]["query_count"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
