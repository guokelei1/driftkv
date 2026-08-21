#!/usr/bin/env python3
"""Coverage-only audit for prospective P8 release/update windows."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/yambda/flat/50m"
OUTPUT = ROOT / "results/data_audit/yambda50m_p8/release_window_coverage_v1.json"
DAY = 86_400
WINDOWS = {
    "update1": (217 * DAY, 231 * DAY),
    "update1_train": (217 * DAY, 229 * DAY),
    "update1_admission_dev": (229 * DAY, 231 * DAY),
    "eval1": (231 * DAY, 238 * DAY),
    "update2": (231 * DAY, 245 * DAY),
    "update2_train": (231 * DAY, 243 * DAY),
    "update2_admission_dev": (243 * DAY, 245 * DAY),
    "eval2": (245 * DAY, 252 * DAY),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def window(timestamp: int) -> list[str]:
    return [name for name, (start, end) in WINDOWS.items() if start <= timestamp < end]


def feedback_rows() -> dict[int, list[tuple[int, int, int]]]:
    output: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for label, filename in ((1, "likes.parquet"), (0, "dislikes.parquet")):
        table = pq.read_table(RAW / filename, columns=["uid", "timestamp", "item_id"])
        for uid, timestamp, item in zip(
            table["uid"].to_numpy(zero_copy_only=False),
            table["timestamp"].to_numpy(zero_copy_only=False),
            table["item_id"].to_numpy(zero_copy_only=False),
            strict=True,
        ):
            if window(int(timestamp)):
                output[int(uid)].append((int(timestamp), int(item), label))
    for rows in output.values():
        rows.sort()
    return output


def main() -> None:
    feedback = feedback_rows()
    counts = {name: Counter() for name in WINDOWS}
    users = {name: defaultdict(set) for name in WINDOWS}
    strata = {name: Counter() for name in WINDOWS}
    parquet = pq.ParquetFile(RAW / "listens.parquet")
    current_uid: int | None = None
    ts_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []

    def consume() -> None:
        if current_uid is None:
            return
        timestamps = np.concatenate(ts_parts).astype(np.int64, copy=False)
        items = np.concatenate(item_parts).astype(np.int64, copy=False)
        last: dict[int, int] = {}
        boundaries = np.r_[0, np.flatnonzero(timestamps[1:] != timestamps[:-1]) + 1, len(timestamps)]
        for group in range(len(boundaries) - 1):
            start, end = int(boundaries[group]), int(boundaries[group + 1])
            timestamp = int(timestamps[start])
            for name in window(timestamp):
                if start:
                    counts[name]["N_queries"] += end - start
                    users[name]["N"].add(current_uid)
                gap_eligible = start and timestamp - int(timestamps[start - 1]) >= 3 * DAY
                familiar = (
                    set(items[max(0, start - 512) : start].tolist()) if gap_eligible else set()
                )
                if gap_eligible and len(familiar) >= 2:
                    counts[name]["R_eligible"] += 1
                    users[name]["R_eligible"].add(current_uid)
                    if int(items[start]) in familiar:
                        counts[name]["R_rankable"] += 1
                        users[name]["R_rankable"].add(current_uid)
            for position in range(start, end):
                last[int(items[position])] = position
        for timestamp, item, label in feedback.get(current_uid, ()):
            prefix = int(np.searchsorted(timestamps, timestamp, side="left"))
            if not prefix:
                continue
            for name in window(timestamp):
                counts[name]["F_queries"] += 1
                counts[name]["F_likes" if label else "F_dislikes"] += 1
                users[name]["F"].add(current_uid)
                position = last_before(items, item, prefix)
                if position is None:
                    stratum = "never_seen"
                elif position >= prefix - 32:
                    stratum = "recent_seen"
                elif position >= prefix - 512:
                    stratum = "old_seen"
                else:
                    stratum = "seen_only_before_512"
                strata[name][stratum] += 1

    for batch in parquet.iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id"]):
        batch_uid = batch["uid"].to_numpy(zero_copy_only=False).astype(np.int64)
        batch_ts = batch["timestamp"].to_numpy(zero_copy_only=False).astype(np.int64)
        batch_item = batch["item_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        boundaries = np.flatnonzero(batch_uid[1:] != batch_uid[:-1]) + 1
        for start, end in zip(np.r_[0, boundaries], np.r_[boundaries, len(batch_uid)], strict=True):
            uid = int(batch_uid[start])
            if current_uid is not None and uid != current_uid:
                consume()
                ts_parts.clear()
                item_parts.clear()
            current_uid = uid
            ts_parts.append(batch_ts[start:end])
            item_parts.append(batch_item[start:end])
    consume()
    payload = {
        "status": "coverage_only_no_model_scores",
        "windows_seconds": {name: list(bounds) for name, bounds in WINDOWS.items()},
        "windows_days": {name: [start // DAY, end // DAY] for name, (start, end) in WINDOWS.items()},
        "coverage": {
            name: {
                **dict(counts[name]),
                "unique_users": {key: len(value) for key, value in users[name].items()},
                "feedback_history_strata_v2": dict(strata[name]),
            }
            for name in WINDOWS
        },
        "raw_hashes": {
            filename: sha256_file(RAW / filename)
            for filename in ("listens.parquet", "likes.parquet", "dislikes.parquet")
        },
        "selection_uses_H_or_S": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["coverage"], indent=2))


def last_before(items: np.ndarray, target: int, end: int) -> int | None:
    matches = np.flatnonzero(items[:end] == target)
    return None if not len(matches) else int(matches[-1])


if __name__ == "__main__":
    main()
