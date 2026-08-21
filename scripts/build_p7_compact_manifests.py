#!/usr/bin/env python3
"""Materialize and seal P7.5-Full compact manifests without model scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hstu_kvcache.data.stateful_workloads import build_return_to_familiar_request

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/yambda"
OUTPUT = ROOT / "data/manifests/p7_full_v1"
DAY = 86_400
CONTEXT = 512
RETURN_GAP = 3 * DAY
K = 3939
POOL_SIZE = 1000
PANEL_SIZE = 100
DECAY = 0.5
SPLITS = {
    "base_fit": (0, 180 * DAY),
    "residual_train": (180 * DAY, 203 * DAY),
    "development": (203 * DAY, 210 * DAY),
    "qualification": (210 * DAY, 217 * DAY),
}
BLOCKS = tuple((index * 45 * DAY, (index + 1) * 45 * DAY) for index in range(4))
FEATURE_SCHEMAS = {
    "N": (
        "log1p_item_count",
        "log1p_artist_count",
        "log1p_item_recency_seconds",
        "log1p_artist_recency_seconds",
        "log1p_global_popularity_at_base_fit_cutoff",
        "item_history_missing",
        "artist_history_or_mapping_missing",
    ),
    "R": (
        "log1p_item_count",
        "log1p_artist_count",
        "log1p_item_recency_seconds",
        "log1p_artist_recency_seconds",
        "log1p_global_popularity_at_base_fit_cutoff",
        "log1p_causal_proposal_rank",
        "artist_missing",
    ),
    "F": (
        "log1p_item_count",
        "log1p_artist_count",
        "log1p_item_recency_seconds",
        "log1p_artist_recency_seconds",
        "log1p_global_popularity_at_base_fit_cutoff",
        "item_history_missing",
        "artist_history_or_mapping_missing",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def code_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def hash_score(namespace: str, uid: int, timestamp: int = 0, row: int = 0) -> int:
    payload = f"{namespace}:{uid}:{timestamp}:{row}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def split_name(timestamp: int) -> str | None:
    for name, (start, end) in SPLITS.items():
        if start <= timestamp < end:
            return name
    return None


def base_block(timestamp: int) -> int | None:
    for index, (start, end) in enumerate(BLOCKS):
        if start <= timestamp < end:
            return index
    return None


@dataclass(frozen=True)
class Spec:
    workload: str
    split: str
    uid: int
    query_timestamp: int
    prefix_end_local: int
    raw_user_row_start: int
    source_row: int
    target_item: int | None = None
    label: int | None = None
    is_organic: int | None = None
    rankable: bool = False
    target_stratum: str | None = None

    @property
    def key(self) -> tuple:
        return (
            self.workload,
            self.split,
            int(self.uid),
            int(self.query_timestamp),
            int(self.source_row),
        )


class ArrayLookup:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def get(self, key: int, default: int = 0) -> int:
        return int(self.values[key]) if 0 <= key < len(self.values) else default


def catalog_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mapping = pq.read_table(RAW / "artist_item_mapping.parquet", columns=["artist_id", "item_id"])
    mapping_items = mapping.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
    mapping_artists = mapping.column("artist_id").to_numpy(zero_copy_only=False).astype(np.int64)
    size = int(mapping_items.max()) + 1
    artist = np.full(size, -1, dtype=np.int64)
    artist[mapping_items] = mapping_artists
    first_seen = np.full(size, np.iinfo(np.int64).max, dtype=np.int64)
    popularity = np.zeros(size, dtype=np.int64)
    parquet = pq.ParquetFile(RAW / "flat/50m/listens.parquet")
    for batch in parquet.iter_batches(batch_size=524_288, columns=["timestamp", "item_id"]):
        timestamp = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        items = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        visible = timestamp < SPLITS["qualification"][1]
        np.minimum.at(first_seen, items[visible], timestamp[visible])
        base = items[timestamp < SPLITS["base_fit"][1]]
        popularity += np.bincount(base, minlength=size)
    catalog = np.flatnonzero(first_seen < np.iinfo(np.int64).max)
    order = catalog[np.argsort(first_seen[catalog], kind="stable")]
    return artist, popularity, first_seen, order


def load_feedback() -> dict[int, list[tuple[int, int, int, int]]]:
    result: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for label, name in ((1, "likes"), (0, "dislikes")):
        table = pq.read_table(
            RAW / f"flat/50m/{name}.parquet",
            columns=["uid", "timestamp", "item_id", "is_organic"],
        )
        columns = [table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names]
        for uid, timestamp, item, organic in zip(*columns, strict=True):
            if int(timestamp) < SPLITS["qualification"][1]:
                result[int(uid)].append((int(timestamp), int(item), label, int(organic)))
    for values in result.values():
        values.sort()
    return result


def stratum(last_position: int | None, prefix_end: int) -> str:
    if last_position is None:
        return "never_seen"
    if last_position >= prefix_end - 32:
        return "recent_seen"
    if last_position >= prefix_end - CONTEXT:
        return "old_seen"
    return "seen_only_before_512"


def choose(current: Spec | None, candidate: Spec, namespace: str) -> Spec:
    if current is None:
        return candidate
    old = hash_score(namespace, current.uid, current.query_timestamp, current.source_row)
    new = hash_score(namespace, candidate.uid, candidate.query_timestamp, candidate.source_row)
    return candidate if new < old else current


def load_qualification_exclusions() -> tuple[set[tuple[str, int, int]], str]:
    path = ROOT / "data/manifests/p7_canary/qualification_exclusions_v1.json"
    payload = json.loads(path.read_text())
    excluded = {
        (str(row["workload"]), int(row["uid"]), int(row["query_timestamp"]))
        for row in payload["conservative_exact_request_exclusions"]
    }
    return excluded, sha256_file(path)


def select_specs(
    feedback: dict[int, list[tuple[int, int, int, int]]],
    catalog_first_seen_ordered: np.ndarray,
    qualification_exclusions: set[tuple[str, int, int]],
) -> tuple[list[Spec], dict]:
    n_base: dict[tuple[int, int], Spec] = {}
    n_other: dict[tuple[str, int], Spec] = {}
    r_base: dict[tuple[int, int], Spec] = {}
    r_all: list[Spec] = []
    f_base: dict[tuple[int, int], Spec] = {}
    f_residual: dict[int, Spec] = {}
    f_eval: list[Spec] = []
    parquet = pq.ParquetFile(RAW / "flat/50m/listens.parquet")
    current_uid: int | None = None
    timestamp_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []
    user_row_start = 0
    global_end = 0

    def consume() -> None:
        if current_uid is None:
            return
        timestamps = np.concatenate(timestamp_parts).astype(np.int64, copy=False)
        items = np.concatenate(item_parts).astype(np.int64, copy=False)
        last_position: dict[int, int] = {}
        window: deque[int] = deque()
        counts: Counter[int] = Counter()
        boundaries = np.r_[0, np.flatnonzero(timestamps[1:] != timestamps[:-1]) + 1, len(timestamps)]
        for group_index in range(len(boundaries) - 1):
            start, end = int(boundaries[group_index]), int(boundaries[group_index + 1])
            timestamp = int(timestamps[start])
            split = split_name(timestamp)
            visible_catalog = int(
                np.searchsorted(catalog_first_seen_ordered, timestamp, side="left")
            )
            if split is not None and start > 0 and visible_catalog >= PANEL_SIZE:
                for position in range(start, end):
                    target = int(items[position])
                    spec = Spec(
                        "N",
                        split,
                        current_uid,
                        timestamp,
                        start,
                        user_row_start,
                        user_row_start + position,
                        target_item=target,
                        target_stratum=stratum(last_position.get(target), start),
                    )
                    if ("N", current_uid, timestamp) in qualification_exclusions:
                        continue
                    if split == "base_fit":
                        block = base_block(timestamp)
                        assert block is not None
                        key = (current_uid, block)
                        n_base[key] = choose(n_base.get(key), spec, f"p7-base-N-block{block}")
                    else:
                        key = (split, current_uid)
                        n_other[key] = choose(n_other.get(key), spec, f"p7-{split}-N-v1")

            gap = timestamp - int(timestamps[start - 1]) if start else 0
            if split is not None and gap >= RETURN_GAP and len(counts) >= 2:
                target = int(items[start])
                previous = last_position.get(target)
                rankable = target in counts
                spec = Spec(
                    "R",
                    split,
                    current_uid,
                    timestamp,
                    start,
                    user_row_start,
                    user_row_start + start,
                    target_item=target,
                    rankable=rankable,
                    target_stratum=stratum(previous, start),
                )
                if split == "base_fit":
                    if rankable:
                        block = base_block(timestamp)
                        assert block is not None
                        key = (current_uid, block)
                        r_base[key] = choose(r_base.get(key), spec, f"p7-base-R-block{block}")
                else:
                    r_all.append(spec)

            for position in range(start, end):
                item = int(items[position])
                window.append(item)
                counts[item] += 1
                last_position[item] = position
                if len(window) > CONTEXT:
                    removed = window.popleft()
                    counts[removed] -= 1
                    if counts[removed] == 0:
                        del counts[removed]

        for feedback_index, (timestamp, item, label, organic) in enumerate(
            feedback.get(current_uid, ())
        ):
            split = split_name(timestamp)
            if split is None:
                continue
            prefix_end = int(np.searchsorted(timestamps, timestamp, side="left"))
            if prefix_end == 0:
                continue
            spec = Spec(
                "F",
                split,
                current_uid,
                timestamp,
                prefix_end,
                user_row_start,
                -(current_uid * 1_000_000 + feedback_index + 1),
                target_item=item,
                label=label,
                is_organic=organic,
            )
            if split == "base_fit":
                block = base_block(timestamp)
                assert block is not None
                key = (current_uid, block)
                f_base[key] = choose(f_base.get(key), spec, f"p7-base-F-block{block}")
            elif split == "residual_train":
                f_residual[current_uid] = choose(
                    f_residual.get(current_uid), spec, "p7-residual-F-v1"
                )
            else:
                f_eval.append(spec)

    for batch in parquet.iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id"]):
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        timestamp = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        boundaries = np.flatnonzero(uid[1:] != uid[:-1]) + 1
        starts, ends = np.r_[0, boundaries], np.r_[boundaries, len(uid)]
        for start, end in zip(starts, ends, strict=True):
            user = int(uid[start])
            if current_uid is not None and user != current_uid:
                consume()
                timestamp_parts.clear()
                item_parts.clear()
                user_row_start = global_end
            current_uid = user
            timestamp_parts.append(timestamp[start:end])
            item_parts.append(item[start:end])
            global_end += end - start
    consume()

    n_residual = [value for (split, _), value in n_other.items() if split == "residual_train"]
    n_residual.sort(key=lambda value: hash_score("p7-residual-N-users-v1", value.uid))
    n_selected = n_residual[:K]
    f_values = list(f_residual.values())
    f_values.sort(key=lambda value: hash_score("p7-residual-F-users-v1", value.uid))
    f_selected = f_values[:K]
    if len(n_selected) != K or len(f_selected) != K:
        raise RuntimeError("N/F residual selection does not meet frozen K")
    specs = list(n_base.values()) + n_selected
    specs += [value for (split, _), value in n_other.items() if split in {"development", "qualification"}]
    specs += list(r_base.values()) + r_all + list(f_base.values()) + f_selected + f_eval
    specs.sort(key=lambda value: (value.uid, value.query_timestamp, value.workload, value.source_row))
    by_workload_split = Counter((value.workload, value.split) for value in specs)
    selection = {
        "specs": len(specs),
        "by_workload_split": {
            f"{workload}:{split}": count
            for (workload, split), count in sorted(by_workload_split.items())
        },
        "selection_digest": digest([value.key for value in specs]),
        "qualification_exact_request_exclusions_applied": len(qualification_exclusions),
    }
    return specs, selection


def ranked_pool(
    history: list[tuple[int, int, int]],
    query_timestamp: int,
    first_seen: np.ndarray,
    catalog_order: np.ndarray,
    catalog_first_seen_ordered: np.ndarray,
    *,
    exclude: int | None,
) -> list[int]:
    counts = Counter(item for item, _, _ in history)
    last = {item: index for index, (item, _, _) in enumerate(history)}
    seen = sorted(counts, key=lambda item: (-counts[item], -last[item], item))
    pool, included = [], set()
    for item in seen:
        if item != exclude and first_seen[item] < query_timestamp:
            pool.append(item)
            included.add(item)
    visible = int(np.searchsorted(catalog_first_seen_ordered, query_timestamp, side="left"))
    for raw_item in catalog_order[:visible]:
        item = int(raw_item)
        if item == exclude or item in included:
            continue
        pool.append(item)
        included.add(item)
        if len(pool) >= POOL_SIZE:
            break
    return pool[:POOL_SIZE]


def sample_panel(pool: list[int], count: int, namespace: str, uid: int, timestamp: int) -> list[int]:
    if len(pool) < count:
        raise ValueError("Q_main pool is smaller than the frozen panel")
    ranks = np.arange(1, len(pool) + 1, dtype=np.float64)
    weights = ranks ** (-DECAY)
    weights /= weights.sum()
    seed = hash_score(namespace, uid, timestamp)
    keys = np.random.default_rng(seed).exponential(size=len(pool)) / weights
    selected = np.argpartition(keys, count - 1)[:count]
    return [pool[index] for index in np.sort(selected).tolist()]


def generic_features(
    history: list[tuple[int, int, int]],
    candidates: list[int],
    query_timestamp: int,
    artist: np.ndarray,
    popularity: np.ndarray,
) -> list[list[float]]:
    item_counts = Counter(item for item, _, _ in history)
    item_last = {item: timestamp for item, timestamp, _ in history}
    artist_counts: Counter[int] = Counter()
    artist_last: dict[int, int] = {}
    for item, timestamp, _ in history:
        value = int(artist[item]) if item < len(artist) else -1
        if value >= 0:
            artist_counts[value] += 1
            artist_last[value] = timestamp
    output = []
    for item in candidates:
        value = int(artist[item]) if item < len(artist) else -1
        item_missing = item not in item_last
        artist_missing = value < 0 or value not in artist_last
        output.append(
            [
                math.log1p(item_counts[item]),
                math.log1p(artist_counts[value]) if not artist_missing else 0.0,
                math.log1p(query_timestamp - item_last[item]) if not item_missing else 0.0,
                math.log1p(query_timestamp - artist_last[value]) if not artist_missing else 0.0,
                math.log1p(popularity[item]) if item < len(popularity) else 0.0,
                float(item_missing),
                float(artist_missing),
            ]
        )
    return output


REQUEST_SCHEMA = pa.schema(
    [
        ("workload", pa.string()),
        ("split", pa.string()),
        ("manifest_kind", pa.string()),
        ("request_id", pa.string()),
        ("candidate_set_id", pa.string()),
        ("uid", pa.int64()),
        ("query_timestamp", pa.int64()),
        ("raw_user_row_start", pa.int64()),
        ("raw_prefix_end_exclusive", pa.int64()),
        ("effective_prefix_length", pa.int32()),
        ("candidate_offset", pa.int64()),
        ("candidate_count", pa.int32()),
        ("base_feature_schema_hash", pa.string()),
        ("request_weight", pa.float64()),
        ("label", pa.int8()),
        ("target_index", pa.int32()),
        ("rankable", pa.bool_()),
        ("target_stratum", pa.string()),
        ("is_organic", pa.int8()),
        ("prior_30m_same_item", pa.bool_()),
        ("latest_item", pa.bool_()),
        ("source_row", pa.int64()),
    ]
)
CANDIDATE_SCHEMA = pa.schema(
    [
        ("candidate_global_index", pa.int64()),
        ("workload", pa.string()),
        ("candidate_set_id", pa.string()),
        ("query_timestamp", pa.int64()),
        ("candidate_position", pa.int32()),
        ("candidate_item_id", pa.int64()),
        ("base_features", pa.list_(pa.float32(), 7)),
    ]
)


class SplitWriter:
    def __init__(self, root: Path, split: str) -> None:
        self.root = root / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.request_buffer: list[dict] = []
        self.candidate_buffer: list[dict] = []
        self.request_shards: list[dict] = []
        self.candidate_shards: list[dict] = []
        self.candidate_offset = 0
        self.views: Counter = Counter()
        self.view_users: dict[tuple[str, str], set[int]] = defaultdict(set)
        self.history_tokens: Counter = Counter()
        self.candidate_rows: Counter = Counter()
        self.candidate_counts: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.user_view_counts: Counter = Counter()

    def add_candidate_set(
        self,
        workload: str,
        candidate_set_id: str,
        query_timestamp: int,
        candidates: list[int],
        features: list[list[float]],
    ) -> tuple[int, int]:
        offset = self.candidate_offset
        for position, (item, values) in enumerate(zip(candidates, features, strict=True)):
            self.candidate_buffer.append(
                {
                    "candidate_global_index": self.candidate_offset,
                    "workload": workload,
                    "candidate_set_id": candidate_set_id,
                    "query_timestamp": query_timestamp,
                    "candidate_position": position,
                    "candidate_item_id": item,
                    "base_features": np.asarray(values, dtype=np.float32).tolist(),
                }
            )
            self.candidate_offset += 1
        if len(self.candidate_buffer) >= 200_000:
            self.flush_candidates()
        return offset, len(candidates)

    def add_request(self, row: dict) -> None:
        self.request_buffer.append(row)
        key = (row["workload"], row["manifest_kind"])
        self.views[key] += 1
        self.view_users[key].add(row["uid"])
        self.history_tokens[key] += row["effective_prefix_length"]
        self.candidate_rows[key] += row["candidate_count"]
        self.candidate_counts[key].append(row["candidate_count"])
        self.user_view_counts[(row["workload"], row["manifest_kind"], row["uid"])] += 1
        if len(self.request_buffer) >= 50_000:
            self.flush_requests()

    def _write(self, rows: list[dict], schema: pa.Schema, prefix: str, shards: list[dict]) -> None:
        if not rows:
            return
        path = self.root / f"{prefix}-{len(shards):05d}.parquet"
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, path, compression="zstd", use_dictionary=True)
        shards.append(
            {
                "path": path.name,
                "rows": len(rows),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "min_timestamp": int(min(row["query_timestamp"] for row in rows)),
                "max_timestamp": int(max(row["query_timestamp"] for row in rows)),
                "schema_hash": digest(str(schema)),
            }
        )
        rows.clear()

    def flush_candidates(self) -> None:
        self._write(self.candidate_buffer, CANDIDATE_SCHEMA, "candidates", self.candidate_shards)

    def flush_requests(self) -> None:
        self._write(self.request_buffer, REQUEST_SCHEMA, "requests", self.request_shards)

    def finish(self, common: dict) -> Path:
        self.flush_candidates()
        self.flush_requests()
        views = {}
        for key, count in sorted(self.views.items()):
            candidates = np.asarray(self.candidate_counts[key], dtype=np.int64)
            views[f"{key[0]}:{key[1]}"] = {
                "queries": count,
                "users": len(self.view_users[key]),
                "candidate_rows": self.candidate_rows[key],
                "history_tokens": self.history_tokens[key],
                "candidate_count": {
                    "p50": float(np.percentile(candidates, 50)),
                    "p90": float(np.percentile(candidates, 90)),
                    "p95": float(np.percentile(candidates, 95)),
                    "p99": float(np.percentile(candidates, 99)),
                    "max": int(candidates.max()),
                },
            }
        index = {
            **common,
            "split": self.split,
            "request_shards": self.request_shards,
            "candidate_shards": self.candidate_shards,
            "views": views,
            "total_request_rows": sum(shard["rows"] for shard in self.request_shards),
            "total_candidate_rows": self.candidate_offset,
            "total_users": len(
                {
                    uid
                    for (_, _, uid), count in self.user_view_counts.items()
                    if count > 0
                }
            ),
            "request_schema_hash": digest(str(REQUEST_SCHEMA)),
            "candidate_schema_hash": digest(str(CANDIDATE_SCHEMA)),
        }
        path = self.root / "manifest.index.json"
        path.write_text(json.dumps(index, indent=2) + "\n")
        return path


def qmain_candidates(
    spec: Spec,
    history: list[tuple[int, int, int]],
    first_seen: np.ndarray,
    catalog_order: np.ndarray,
    catalog_first_seen_ordered: np.ndarray,
) -> tuple[tuple[list[int], int], tuple[list[int], None]]:
    assert spec.target_item is not None
    quality_pool = ranked_pool(
        history,
        spec.query_timestamp,
        first_seen,
        catalog_order,
        catalog_first_seen_ordered,
        exclude=spec.target_item,
    )
    quality = [spec.target_item] + sample_panel(
        quality_pool, PANEL_SIZE - 1, "p7-N-quality-v1", spec.uid, spec.query_timestamp
    )
    fidelity_pool = ranked_pool(
        history,
        spec.query_timestamp,
        first_seen,
        catalog_order,
        catalog_first_seen_ordered,
        exclude=None,
    )
    fidelity = sample_panel(
        fidelity_pool, PANEL_SIZE, "p7-N-fidelity-v1", spec.uid, spec.query_timestamp
    )
    return (quality, 0), (fidelity, None)


def materialize(
    specs: list[Spec],
    artist: np.ndarray,
    popularity: np.ndarray,
    first_seen: np.ndarray,
    catalog_order: np.ndarray,
    qualification_exclusion_hash: str,
    output: Path,
) -> tuple[dict[str, Path], dict]:
    writers = {split: SplitWriter(output, split) for split in SPLITS}
    by_uid: dict[int, list[Spec]] = defaultdict(list)
    for spec in specs:
        by_uid[spec.uid].append(spec)
    view_counts: Counter = Counter()
    for spec in specs:
        if spec.workload == "N":
            kinds = ("quality", "fidelity")
        elif spec.workload == "R":
            kinds = ("fidelity_all_eligible",) + (
                ("quality_rankable", "fidelity_rankable_companion") if spec.rankable else ()
            )
        else:
            kinds = ("quality", "fidelity")
        for kind in kinds:
            view_counts[(spec.workload, spec.split, kind, spec.uid)] += 1

    parquet = pq.ParquetFile(RAW / "flat/50m/listens.parquet")
    current_uid: int | None = None
    timestamp_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []
    organic_parts: list[np.ndarray] = []
    global_end = 0
    expanded_samples: dict[tuple[str, str, str], dict] = {}

    def add_view(
        spec: Spec,
        kind: str,
        candidate_set_id: str,
        offset: int,
        count: int,
        history: list[tuple[int, int, int]],
        candidates: list[int],
        features: list[list[float]],
        *,
        label: int | None,
        target_index: int | None,
        prior_30m: bool | None = None,
        latest: bool | None = None,
    ) -> None:
        writer = writers[spec.split]
        request_id = digest([candidate_set_id, kind])[:24]
        weight = 1.0 / view_counts[(spec.workload, spec.split, kind, spec.uid)]
        row = {
            "workload": spec.workload,
            "split": spec.split,
            "manifest_kind": kind,
            "request_id": request_id,
            "candidate_set_id": candidate_set_id,
            "uid": spec.uid,
            "query_timestamp": spec.query_timestamp,
            "raw_user_row_start": spec.raw_user_row_start,
            "raw_prefix_end_exclusive": spec.raw_user_row_start + spec.prefix_end_local,
            "effective_prefix_length": len(history),
            "candidate_offset": offset,
            "candidate_count": count,
            "base_feature_schema_hash": digest(FEATURE_SCHEMAS[spec.workload]),
            "request_weight": weight,
            "label": label,
            "target_index": target_index,
            "rankable": spec.rankable if kind == "quality_rankable" else None,
            "target_stratum": spec.target_stratum if "quality" in kind else None,
            "is_organic": spec.is_organic if kind == "quality" else None,
            "prior_30m_same_item": prior_30m if kind == "quality" else None,
            "latest_item": latest if kind == "quality" else None,
            "source_row": spec.source_row,
        }
        writer.add_request(row)
        sample_key = (spec.workload, spec.split, kind)
        current = expanded_samples.get(sample_key)
        if current is None or hash_score("roundtrip", spec.uid, spec.query_timestamp, spec.source_row) < current["score"]:
            expanded_samples[sample_key] = {
                "score": hash_score("roundtrip", spec.uid, spec.query_timestamp, spec.source_row),
                "request_id": request_id,
                "history": history,
                "candidate_set_id": candidate_set_id,
                "candidates": candidates,
                "base_features": np.asarray(features, dtype=np.float32).tolist(),
                "label": label,
                "target_index": target_index,
                "weight": weight,
            }

    def consume() -> None:
        if current_uid is None or current_uid not in by_uid:
            return
        timestamps = np.concatenate(timestamp_parts).astype(np.int64, copy=False)
        items = np.concatenate(item_parts).astype(np.int64, copy=False)
        organic = np.concatenate(organic_parts).astype(np.int64, copy=False)
        for spec in by_uid[current_uid]:
            start = max(0, spec.prefix_end_local - CONTEXT)
            history = [
                (int(items[index]), int(timestamps[index]), 1 + (1 - int(organic[index])))
                for index in range(start, spec.prefix_end_local)
                if int(timestamps[index]) < spec.query_timestamp
            ]
            if not history:
                raise RuntimeError("selected request has empty causal history")
            writer = writers[spec.split]
            if spec.workload == "N":
                quality, fidelity = qmain_candidates(
                    spec,
                    history,
                    first_seen,
                    catalog_order,
                    first_seen[catalog_order],
                )
                for kind, (candidates, target_index) in (
                    ("quality", quality),
                    ("fidelity", fidelity),
                ):
                    features = generic_features(
                        history, candidates, spec.query_timestamp, artist, popularity
                    )
                    candidate_set_id = digest([spec.key, kind, candidates])[:24]
                    offset, count = writer.add_candidate_set(
                        spec.workload,
                        candidate_set_id,
                        spec.query_timestamp,
                        candidates,
                        features,
                    )
                    add_view(
                        spec,
                        kind,
                        candidate_set_id,
                        offset,
                        count,
                        history,
                        candidates,
                        features,
                        label=None,
                        target_index=target_index,
                    )
            elif spec.workload == "R":
                request = build_return_to_familiar_request(
                    history,
                    spec.query_timestamp,
                    ArrayLookup(artist),
                    ArrayLookup(popularity),
                )
                candidates = list(request.item_ids)
                features = [list(candidate.base_features()) for candidate in request.candidates]
                candidate_set_id = digest([spec.key, candidates])[:24]
                offset, count = writer.add_candidate_set(
                    spec.workload,
                    candidate_set_id,
                    spec.query_timestamp,
                    candidates,
                    features,
                )
                add_view(
                    spec,
                    "fidelity_all_eligible",
                    candidate_set_id,
                    offset,
                    count,
                    history,
                    candidates,
                    features,
                    label=None,
                    target_index=None,
                )
                if spec.rankable:
                    assert spec.target_item is not None
                    target_index = request.quality_target_index(spec.target_item)
                    if target_index is None or candidates.count(spec.target_item) != 1:
                        raise RuntimeError("rankable R target is not unique in complete universe")
                    add_view(
                        spec,
                        "quality_rankable",
                        candidate_set_id,
                        offset,
                        count,
                        history,
                        candidates,
                        features,
                        label=None,
                        target_index=target_index,
                    )
                    add_view(
                        spec,
                        "fidelity_rankable_companion",
                        candidate_set_id,
                        offset,
                        count,
                        history,
                        candidates,
                        features,
                        label=None,
                        target_index=None,
                    )
            else:
                assert spec.target_item is not None and spec.label is not None
                candidates = [spec.target_item]
                features = generic_features(
                    history, candidates, spec.query_timestamp, artist, popularity
                )
                candidate_set_id = digest([spec.key, candidates])[:24]
                offset, count = writer.add_candidate_set(
                    spec.workload,
                    candidate_set_id,
                    spec.query_timestamp,
                    candidates,
                    features,
                )
                previous_positions = [
                    index for index, (item, _, _) in enumerate(history) if item == spec.target_item
                ]
                prior_30m = bool(
                    previous_positions
                    and spec.query_timestamp - history[previous_positions[-1]][1] <= 1_800
                )
                latest = history[-1][0] == spec.target_item
                add_view(
                    spec,
                    "quality",
                    candidate_set_id,
                    offset,
                    count,
                    history,
                    candidates,
                    features,
                    label=spec.label,
                    target_index=None,
                    prior_30m=prior_30m,
                    latest=latest,
                )
                add_view(
                    spec,
                    "fidelity",
                    candidate_set_id,
                    offset,
                    count,
                    history,
                    candidates,
                    features,
                    label=None,
                    target_index=None,
                )

    for batch in parquet.iter_batches(
        batch_size=262_144, columns=["uid", "timestamp", "item_id", "is_organic"]
    ):
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        timestamp = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        organic = batch.column("is_organic").to_numpy(zero_copy_only=False).astype(np.int64)
        boundaries = np.flatnonzero(uid[1:] != uid[:-1]) + 1
        starts, ends = np.r_[0, boundaries], np.r_[boundaries, len(uid)]
        for start, end in zip(starts, ends, strict=True):
            user = int(uid[start])
            if current_uid is not None and user != current_uid:
                consume()
                timestamp_parts.clear()
                item_parts.clear()
                organic_parts.clear()
            current_uid = user
            timestamp_parts.append(timestamp[start:end])
            item_parts.append(item[start:end])
            organic_parts.append(organic[start:end])
            global_end += end - start
    consume()

    common = {
        "contract": "p7_5_materialization_contract_v1",
        "status": "sealed_unscored",
        "raw_source_hashes": {
            name: sha256_file(path)
            for name, path in {
                "listens": RAW / "flat/50m/listens.parquet",
                "likes": RAW / "flat/50m/likes.parquet",
                "dislikes": RAW / "flat/50m/dislikes.parquet",
                "artist_mapping": RAW / "artist_item_mapping.parquet",
            }.items()
        },
        "materialization_contract_hash": sha256_file(
            ROOT / "configs/contracts/p7_5_materialization_contract_v1.yaml"
        ),
        "canary_exclusion_hash": qualification_exclusion_hash,
        "materializer_code_hash": sha256_file(Path(__file__)),
        "code_commit": code_commit(),
        "qualification_scored": False,
    }
    indices = {split: writer.finish(common) for split, writer in writers.items()}
    roundtrip_path = output / "expanded_reference_samples.json"
    roundtrip_path.write_text(json.dumps({str(key): value for key, value in expanded_samples.items()}, indent=2) + "\n")
    return indices, {
        "expanded_reference_samples": str(roundtrip_path.relative_to(ROOT)),
        "expanded_reference_hash": sha256_file(roundtrip_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output already exists and is non-empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    artist, popularity, first_seen, catalog_order = catalog_arrays()
    feedback = load_feedback()
    qualification_exclusions, qualification_exclusion_hash = load_qualification_exclusions()
    specs, selection = select_specs(
        feedback, first_seen[catalog_order], qualification_exclusions
    )
    indices, reference = materialize(
        specs,
        artist,
        popularity,
        first_seen,
        catalog_order,
        qualification_exclusion_hash,
        args.output,
    )
    result = {
        "contract": "p7_5_materialization_contract_v1",
        "stage": "P7.5-Full",
        "status": "materialized_unverified_unscored",
        "selection": selection,
        "indices": {split: str(path.relative_to(ROOT)) for split, path in indices.items()},
        "index_hashes": {split: sha256_file(path) for split, path in indices.items()},
        **reference,
        "qualification_scored": False,
        "base_fitted": False,
        "hstu_trained": False,
    }
    path = args.output / "materialization_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "specs": selection["specs"], "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
