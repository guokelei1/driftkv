#!/usr/bin/env python3
"""P7.1 read-only feasibility audit for Yambda stateful workloads.

This script never trains a model and never writes candidate manifests.  It
selects neither a workload nor a threshold from H/S: the only automatic
decisions implement the coverage/legality rules frozen in the P7 contract.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import subprocess
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/yambda"
OUTPUT = ROOT / "results/data_audit/yambda50m_p7/stateful_workload_feasibility_v1.json"
DAY = 86_400
GAPS = (1_800, 7_200, 21_600, 86_400, 259_200)
RECENT = 32
CONTEXT = 512
SPLITS = {
    "train": (0, 203 * DAY),
    "development": (203 * DAY, 210 * DAY),
    "qualification": (210 * DAY, 217 * DAY),
    "outside_registered_splits": (217 * DAY, np.iinfo(np.int64).max),
}
RESERVOIR_SIZE = 100_000
SEED = 20260819


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def split_name(timestamp: int) -> str:
    for name, (start, end) in SPLITS.items():
        if start <= timestamp < end:
            return name
    return "before_registered_origin"


@dataclass
class RunningStat:
    """Exact count/mean plus a deterministic reservoir for quantiles."""

    seed: int
    count: int = 0
    total: float = 0.0
    sample: list[float] = field(default_factory=list)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def add(self, value: float) -> None:
        self.count += 1
        self.total += float(value)
        if len(self.sample) < RESERVOIR_SIZE:
            self.sample.append(float(value))
            return
        replacement = int(self.rng.integers(0, self.count))
        if replacement < RESERVOIR_SIZE:
            self.sample[replacement] = float(value)

    def summary(self) -> dict:
        values = np.asarray(self.sample, dtype=np.float64)
        return {
            "count": self.count,
            "mean": self.total / self.count if self.count else None,
            "quantiles_from_deterministic_reservoir": {
                f"p{q}": float(np.percentile(values, q)) if len(values) else None
                for q in (0, 25, 50, 75, 90, 95, 99, 100)
            },
            "reservoir_size": len(values),
        }


def new_return_bucket(seed: int) -> dict:
    return {
        "session_starts": 0,
        "target_recent_seen": 0,
        "target_old_seen": 0,
        "target_outside_512": 0,
        "target_seen_only_before_512": 0,
        "target_never_seen": 0,
        "rankable_requests": 0,
        "requests_with_old_only_candidates": 0,
        "familiar_candidates": RunningStat(seed + 1),
        "recent_candidates": RunningStat(seed + 2),
        "old_only_candidates": RunningStat(seed + 3),
        "target_item_count_midrank": RunningStat(seed + 4),
        "target_item_recency_midrank": RunningStat(seed + 5),
        "target_artist_count_midrank": RunningStat(seed + 6),
        "target_artist_recency_midrank": RunningStat(seed + 7),
        "optimistic_best_simple_feature_midrank": RunningStat(seed + 8),
    }


def rank_fraction(values: list[float], target: float, *, higher_is_better: bool) -> float:
    array = np.asarray(values, dtype=np.float64)
    better = array > target if higher_is_better else array < target
    tied = array == target
    # Midrank normalized to (0, 1], where lower is better.
    return float((better.sum() + 0.5 * tied.sum()) / len(array))


def summarize_return_bucket(bucket: dict) -> dict:
    starts = bucket["session_starts"]
    result = {
        key: value
        for key, value in bucket.items()
        if not isinstance(value, RunningStat)
    }
    result["target_position_fractions"] = {
        key: bucket[key] / starts if starts else None
        for key in (
            "target_recent_seen",
            "target_old_seen",
            "target_outside_512",
            "target_seen_only_before_512",
            "target_never_seen",
        )
    }
    result["old_only_candidate_positive_fraction"] = (
        bucket["requests_with_old_only_candidates"] / starts if starts else None
    )
    for key, value in bucket.items():
        if isinstance(value, RunningStat):
            result[key] = value.summary()
    return result


def load_artist_map() -> np.ndarray:
    table = pq.read_table(RAW / "artist_item_mapping.parquet", columns=["artist_id", "item_id"])
    items = table.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
    artists = table.column("artist_id").to_numpy(zero_copy_only=False).astype(np.int64)
    output = np.full(int(items.max()) + 1, -1, dtype=np.int64)
    output[items] = artists
    return output


def load_feedback() -> dict[int, list[tuple[int, int, int]]]:
    by_user: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for label, name in ((1, "likes"), (0, "dislikes")):
        table = pq.read_table(RAW / f"flat/50m/{name}.parquet", columns=["uid", "timestamp", "item_id"])
        uid = table.column("uid").to_numpy(zero_copy_only=False)
        timestamp = table.column("timestamp").to_numpy(zero_copy_only=False)
        item = table.column("item_id").to_numpy(zero_copy_only=False)
        for user, ts, candidate in zip(uid, timestamp, item, strict=True):
            by_user[int(user)].append((int(ts), int(candidate), label))
    for events in by_user.values():
        events.sort()
    return by_user


def new_feedback_bucket() -> dict:
    return Counter(
        total=0,
        causal=0,
        like=0,
        dislike=0,
        causal_like=0,
        causal_dislike=0,
        candidate_recent_seen=0,
        candidate_old_seen=0,
        candidate_seen_only_before_512=0,
        candidate_unseen=0,
        latest_item=0,
        target_listen_within_prior_30m=0,
        coincident_target_listen=0,
    )


def process_feedback_user(
    uid: int,
    timestamps: np.ndarray,
    items: np.ndarray,
    feedback_by_user: dict[int, list[tuple[int, int, int]]],
    buckets: dict[str, Counter],
    labels_per_user: dict[str, RunningStat],
) -> None:
    events = feedback_by_user.pop(uid, ())
    if not events:
        return
    causal_labels = 0
    positions_by_item: dict[int, list[int]] = defaultdict(list)
    for position, item in enumerate(items):
        positions_by_item[int(item)].append(position)
    for timestamp, candidate, label in events:
        name = split_name(timestamp)
        if name not in buckets:
            buckets[name] = new_feedback_bucket()
        bucket = buckets[name]
        bucket["total"] += 1
        bucket["like" if label else "dislike"] += 1
        prefix_end = int(np.searchsorted(timestamps, timestamp, side="left"))
        same_end = int(np.searchsorted(timestamps, timestamp, side="right"))
        positions = positions_by_item.get(candidate, ())
        earlier_count = bisect.bisect_left(positions, prefix_end)
        same_time_end = bisect.bisect_left(positions, same_end)
        if same_time_end > earlier_count:
            bucket["coincident_target_listen"] += 1
        if prefix_end == 0:
            continue
        bucket["causal"] += 1
        bucket["causal_like" if label else "causal_dislike"] += 1
        causal_labels += 1
        last_position = positions[earlier_count - 1] if earlier_count else None
        if last_position is not None and last_position >= prefix_end - RECENT:
            bucket["candidate_recent_seen"] += 1
        elif last_position is not None and last_position >= prefix_end - CONTEXT:
            bucket["candidate_old_seen"] += 1
        elif last_position is not None:
            bucket["candidate_seen_only_before_512"] += 1
        else:
            bucket["candidate_unseen"] += 1
        if int(items[prefix_end - 1]) == candidate:
            bucket["latest_item"] += 1
        if last_position is not None and timestamp - int(timestamps[last_position]) <= 1_800:
            bucket["target_listen_within_prior_30m"] += 1
    labels_per_user["all_feedback"].add(len(events))
    if causal_labels:
        labels_per_user["causal_feedback"].add(causal_labels)


def process_user(
    uid: int,
    timestamps: np.ndarray,
    items: np.ndarray,
    artist_by_item: np.ndarray,
    return_buckets: dict[int, dict[str, dict]],
    gap_stats: RunningStat,
) -> None:
    window: deque[tuple[int, int, int]] = deque()
    item_counts: Counter[int] = Counter()
    artist_counts: Counter[int] = Counter()
    last_item: dict[int, int] = {}
    last_artist: dict[int, int] = {}
    lifetime_seen: set[int] = set()
    previous_timestamp: int | None = None

    for timestamp_value, item_value in zip(timestamps, items, strict=True):
        timestamp, item = int(timestamp_value), int(item_value)
        artist = int(artist_by_item[item]) if item < len(artist_by_item) else -1
        gap = timestamp - previous_timestamp if previous_timestamp is not None else 0
        if gap >= GAPS[0]:
            gap_stats.add(gap)
            recent_events = list(window)[-RECENT:]
            recent_items = {event[0] for event in recent_events}
            old_only_items = set(item_counts) - recent_items
            target_recent = item in recent_items
            target_old = item in old_only_items
            target_seen_before_cap = item in lifetime_seen and not (target_recent or target_old)
            target_never_seen = item not in lifetime_seen
            ranks: tuple[float, float, float, float] | None = None
            if item in item_counts and len(item_counts) >= 2:
                candidates = list(item_counts)
                item_count_values = [item_counts[value] for value in candidates]
                item_recencies = [timestamp - last_item[value] for value in candidates]
                artist_count_values = [artist_counts[int(artist_by_item[value])] for value in candidates]
                artist_recencies = [timestamp - last_artist[int(artist_by_item[value])] for value in candidates]
                ranks = (
                    rank_fraction(item_count_values, item_counts[item], higher_is_better=True),
                    rank_fraction(item_recencies, timestamp - last_item[item], higher_is_better=False),
                    rank_fraction(artist_count_values, artist_counts[artist], higher_is_better=True),
                    rank_fraction(artist_recencies, timestamp - last_artist[artist], higher_is_better=False),
                )
            for gap_threshold in GAPS:
                if gap < gap_threshold:
                    continue
                name = split_name(timestamp)
                bucket = return_buckets[gap_threshold][name]
                bucket["session_starts"] += 1
                bucket["target_recent_seen"] += int(target_recent)
                bucket["target_old_seen"] += int(target_old)
                bucket["target_seen_only_before_512"] += int(target_seen_before_cap)
                bucket["target_never_seen"] += int(target_never_seen)
                bucket["target_outside_512"] += int(not (target_recent or target_old))
                bucket["familiar_candidates"].add(len(item_counts))
                bucket["recent_candidates"].add(len(recent_items))
                bucket["old_only_candidates"].add(len(old_only_items))
                bucket["requests_with_old_only_candidates"] += int(bool(old_only_items))
                if ranks is None:
                    continue
                bucket["rankable_requests"] += 1
                for key, value in zip(
                    (
                        "target_item_count_midrank",
                        "target_item_recency_midrank",
                        "target_artist_count_midrank",
                        "target_artist_recency_midrank",
                    ),
                    ranks,
                    strict=True,
                ):
                    bucket[key].add(value)
                bucket["optimistic_best_simple_feature_midrank"].add(min(ranks))

        window.append((item, artist, timestamp))
        item_counts[item] += 1
        artist_counts[artist] += 1
        last_item[item] = timestamp
        last_artist[artist] = timestamp
        lifetime_seen.add(item)
        if len(window) > CONTEXT:
            old_item, old_artist, _ = window.popleft()
            item_counts[old_item] -= 1
            artist_counts[old_artist] -= 1
            if item_counts[old_item] == 0:
                del item_counts[old_item]
            if artist_counts[old_artist] == 0:
                del artist_counts[old_artist]
        previous_timestamp = timestamp


def feedback_summary(bucket: Counter) -> dict:
    causal = bucket["causal"]
    output = dict(bucket)
    output["causal_fractions"] = {
        key: bucket[key] / causal if causal else None
        for key in (
            "candidate_recent_seen",
            "candidate_old_seen",
            "candidate_seen_only_before_512",
            "candidate_unseen",
            "latest_item",
            "target_listen_within_prior_30m",
        )
    }
    output["coincident_target_listen_fraction_of_all_feedback"] = (
        bucket["coincident_target_listen"] / bucket["total"] if bucket["total"] else None
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-users", type=int, default=None, help="Canary only; omitted for the frozen full audit")
    args = parser.parse_args()

    artist_by_item = load_artist_map()
    feedback_by_user = load_feedback()
    return_buckets = {
        gap: {
            name: new_return_bucket(SEED + gap + index * 100)
            for index, name in enumerate(SPLITS)
        }
        for gap in GAPS
    }
    feedback_buckets = {name: new_feedback_bucket() for name in SPLITS}
    labels_per_user = {
        "all_feedback": RunningStat(SEED + 900),
        "causal_feedback": RunningStat(SEED + 901),
    }
    gap_stats = RunningStat(SEED + 902)

    parquet = pq.ParquetFile(RAW / "flat/50m/listens.parquet")
    current_uid: int | None = None
    timestamp_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []
    users_processed = 0

    def consume() -> bool:
        nonlocal users_processed
        if current_uid is None or not timestamp_parts:
            return True
        timestamps = np.concatenate(timestamp_parts).astype(np.int64, copy=False)
        items = np.concatenate(item_parts).astype(np.int64, copy=False)
        process_user(current_uid, timestamps, items, artist_by_item, return_buckets, gap_stats)
        process_feedback_user(
            current_uid, timestamps, items, feedback_by_user, feedback_buckets, labels_per_user
        )
        users_processed += 1
        return args.max_users is None or users_processed < args.max_users

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

    return_report = {
        str(gap): {name: summarize_return_bucket(bucket) for name, bucket in split.items()}
        for gap, split in return_buckets.items()
    }
    eligible_gaps = []
    for gap in GAPS:
        counts = return_buckets[gap]
        if (
            counts["train"]["rankable_requests"] >= 10_000
            and counts["development"]["rankable_requests"] >= 500
            and counts["qualification"]["rankable_requests"] >= 500
        ):
            eligible_gaps.append(gap)
    selected_gap = max(eligible_gaps) if eligible_gaps else None

    development = feedback_buckets["development"]
    qualification = feedback_buckets["qualification"]
    feedback_pass = (
        feedback_buckets["train"]["causal"] >= 10_000
        and development["causal"] >= 500
        and qualification["causal"] >= 500
        and development["causal_like"] >= 100
        and development["causal_dislike"] >= 100
        and qualification["causal_like"] >= 100
        and qualification["causal_dislike"] >= 100
        and development["latest_item"] / max(1, development["causal"]) <= 0.80
        and qualification["latest_item"] / max(1, qualification["causal"]) <= 0.80
    )

    paths = {
        name: RAW / relative
        for name, relative in {
            "listens": "flat/50m/listens.parquet",
            "likes": "flat/50m/likes.parquet",
            "dislikes": "flat/50m/dislikes.parquet",
            "artist_mapping": "artist_item_mapping.parquet",
        }.items()
    }
    result = {
        "contract": "p7_yambda_stateful_suite_v1",
        "stage": "P7.1",
        "status": "completed_full_audit" if args.max_users is None else "canary_only",
        "read_only_no_model_training": True,
        "users_processed": users_processed,
        "max_users": args.max_users,
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
        "time_splits_seconds": {name: list(bounds) for name, bounds in SPLITS.items()},
        "return_to_familiar": {
            "gap_distribution_for_gaps_ge_30m": gap_stats.summary(),
            "gap_grid_seconds": list(GAPS),
            "recent_tokens": RECENT,
            "persistent_context_tokens": CONTEXT,
            "by_gap_and_split": return_report,
            "coverage_only_gap_decision": {
                "eligible_gaps_seconds": eligible_gaps,
                "selected_largest_gap_seconds": selected_gap,
                "rule": "rankable train>=10000 and development>=500 and qualification>=500",
                "did_not_use_h_or_s": True,
            },
            "rank_diagnostic_semantics": "normalized target midrank within familiar candidates; optimistic best-feature is diagnostic, not a deployable fitted ranker",
        },
        "explicit_feedback": {
            "by_split": {name: feedback_summary(bucket) for name, bucket in feedback_buckets.items()},
            "labels_per_user": {name: value.summary() for name, value in labels_per_user.items()},
            "unmatched_feedback_users_after_listen_scan": len(feedback_by_user),
            "selection_decision": {
                "explicit_feedback_selected": feedback_pass,
                "fallback": None if feedback_pass else "future_window_preference_requires_separate_contract",
                "did_not_use_h_or_s": True,
            },
            "causal_rule": "listen timestamp strictly less than feedback timestamp",
        },
        "limitations": [
            "Simple-feature midranks are an opportunity/shortcut audit, not fitted base-ranker quality.",
            "Other familiar candidates are competing candidates, not observed negative feedback.",
            "P7.1 does not authorize workload manifests, M0/M1 training, or a version chain.",
        ],
        "seed": SEED,
        "code_commit": code_commit(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "users_processed": users_processed,
                "selected_return_gap_seconds": selected_gap,
                "explicit_feedback_selected": feedback_pass,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
