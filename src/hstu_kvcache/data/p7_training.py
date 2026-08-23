"""Compact-manifest loader for the frozen P7 theta0 training contract."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .compact_manifest import QualificationUnlock, load_compact_index

DAY = 86_400
KINDS = {"N": "quality", "R": "quality_rankable", "F": "quality"}
QUERY_TYPES = {"N": 0, "R": 1, "F": 2}


@dataclass(frozen=True)
class P7Request:
    request_id: str
    workload: str
    uid: int
    query_timestamp: int
    history_items: np.ndarray
    history_behaviors: np.ndarray
    history_time_deltas: np.ndarray
    query_time_delta: float
    candidate_ids: np.ndarray
    base_features: np.ndarray
    target_index: int | None
    label: int | None
    request_weight: float
    manifest_kind: str = "quality"
    is_organic: int | None = None
    prior_30m_same_item: bool | None = None
    latest_item: bool | None = None
    target_stratum: str | None = None
    history_timestamps: np.ndarray | None = None

    def __post_init__(self) -> None:
        length = len(self.history_items)
        if not 1 <= length <= 1024:
            raise ValueError("stateful-workload histories must contain 1..1024 events")
        if self.history_behaviors.shape != (length,) or self.history_time_deltas.shape != (length,):
            raise ValueError("history arrays differ")
        if self.history_timestamps is not None and self.history_timestamps.shape != (length,):
            raise ValueError("history timestamps differ")
        if self.candidate_ids.ndim != 1 or not len(self.candidate_ids):
            raise ValueError("candidate IDs must be non-empty")
        if self.base_features.shape != (len(self.candidate_ids), 7):
            raise ValueError("candidate features differ")
        is_fidelity = "fidelity" in self.manifest_kind
        if is_fidelity:
            if self.target_index is not None or self.label is not None:
                raise ValueError("fidelity requests must not carry targets or labels")
        elif self.workload in {"N", "R"}:
            if self.target_index is None or not 0 <= self.target_index < len(self.candidate_ids):
                raise ValueError("ranking target differs")
            if self.label is not None:
                raise ValueError("ranking requests must not carry a binary label")
        elif self.workload == "F":
            if self.label not in {0, 1} or self.target_index is not None:
                raise ValueError("feedback label differs")
        else:
            raise ValueError("unknown P7 workload")
        if not np.isfinite(self.base_features).all() or not np.isfinite(self.request_weight):
            raise ValueError("request contains non-finite values")


class _RawRowReader:
    def __init__(self, path: Path) -> None:
        self.parquet = pq.ParquetFile(path)
        sizes = [
            self.parquet.metadata.row_group(index).num_rows
            for index in range(self.parquet.num_row_groups)
        ]
        self.ends = np.cumsum(sizes).tolist()
        self.cache: dict[int, pa.Table] = {}

    def rows(self, start: int, end: int) -> pa.Table:
        pieces = []
        cursor = start
        while cursor < end:
            group = bisect.bisect_right(self.ends, cursor)
            group_start = 0 if group == 0 else self.ends[group - 1]
            group_end = self.ends[group]
            table = self.cache.get(group)
            if table is None:
                table = self.parquet.read_row_group(
                    group, columns=["uid", "timestamp", "item_id", "is_organic"]
                )
                self.cache[group] = table
            local_start = cursor - group_start
            count = min(end, group_end) - cursor
            pieces.append(table.slice(local_start, count))
            cursor += count
        return pa.concat_tables(pieces)


def _concatenate(paths: list[Path]) -> pa.Table:
    return pa.concat_tables([pq.read_table(path) for path in paths])


def _candidate_payloads(
    index_path: Path,
    index: dict,
    offsets: np.ndarray,
    counts: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    candidate_ids: list[np.ndarray] = [None] * len(offsets)  # type: ignore[list-item]
    features: list[np.ndarray] = [None] * len(offsets)  # type: ignore[list-item]
    request_cursor = 0
    shard_start = 0
    for shard in index["candidate_shards"]:
        shard_end = shard_start + int(shard["rows"])
        begin = request_cursor
        while request_cursor < len(offsets) and offsets[request_cursor] < shard_end:
            if offsets[request_cursor] < shard_start:
                raise AssertionError("candidate offsets are not monotonic")
            if offsets[request_cursor] + counts[request_cursor] > shard_end:
                raise AssertionError("candidate set crosses shard")
            request_cursor += 1
        if request_cursor > begin:
            table = pq.read_table(
                index_path.parent / shard["path"],
                columns=["candidate_item_id", "base_features"],
            )
            for request_index in range(begin, request_cursor):
                local = int(offsets[request_index] - shard_start)
                count = int(counts[request_index])
                candidate_ids[request_index] = (
                    table["candidate_item_id"]
                    .slice(local, count)
                    .to_numpy(zero_copy_only=False)
                    .astype(np.int64)
                )
                features[request_index] = np.asarray(
                    table["base_features"].slice(local, count).to_pylist(),
                    dtype=np.float32,
                )
        shard_start = shard_end
    if request_cursor != len(offsets):
        raise AssertionError("not every candidate set was reconstructed")
    return candidate_ids, features


def load_p7_requests(
    manifest_root: str | Path,
    raw_listens: str | Path,
    split: str,
    workload: str,
    *,
    manifest_kind: str | None = None,
    qualification_unlock: QualificationUnlock | None = None,
    history_limit: int | None = None,
) -> list[P7Request]:
    """Load one compact P7 view while enforcing the qualification guard.

    ``history_limit=None`` preserves the sealed manifest's materialized history
    length.  A larger explicit limit is reserved for prospective scale
    reproduction: it reconstructs more causal history from the same raw user
    pointer without changing query, candidates, labels, weights, or Base
    features.
    """
    if history_limit is not None and not 1 <= history_limit <= 1024:
        raise ValueError("history_limit must be within 1..1024")
    manifest_root = Path(manifest_root)
    index_path = manifest_root / split / "manifest.index.json"
    index = load_compact_index(index_path, qualification_unlock=qualification_unlock)
    selected_kind = KINDS[workload] if manifest_kind is None else manifest_kind
    request_paths = [index_path.parent / shard["path"] for shard in index["request_shards"]]
    table = _concatenate(request_paths)
    table = table.filter(
            pc.and_(
                pc.equal(table["workload"], workload),
                pc.equal(table["manifest_kind"], selected_kind),
        )
    ).sort_by([("candidate_offset", "ascending")])
    offsets = table["candidate_offset"].to_numpy(zero_copy_only=False).astype(np.int64)
    counts = table["candidate_count"].to_numpy(zero_copy_only=False).astype(np.int64)
    candidates, base_features = _candidate_payloads(index_path, index, offsets, counts)
    raw = _RawRowReader(Path(raw_listens))
    output = []
    for index, row in enumerate(table.to_pylist()):
        manifest_length = int(row["effective_prefix_length"])
        end = int(row["raw_prefix_end_exclusive"])
        if history_limit is None:
            length = manifest_length
        else:
            user_start = int(row["raw_user_row_start"])
            if not user_start <= end:
                raise AssertionError("raw user history pointer is invalid")
            length = min(int(history_limit), end - user_start)
            if length < manifest_length:
                raise ValueError("history_limit may not truncate the sealed manifest history")
        history = raw.rows(end - length, end)
        history_uids = history["uid"].to_numpy(zero_copy_only=False).astype(np.int64)
        if not np.all(history_uids == int(row["uid"])):
            raise AssertionError("raw history pointer crosses users")
        timestamps = history["timestamp"].to_numpy(zero_copy_only=False).astype(np.int64)
        if not np.all(timestamps < int(row["query_timestamp"])):
            raise AssertionError("training history is not strictly causal")
        deltas = np.zeros(length, dtype=np.float32)
        if length > 1:
            deltas[1:] = np.diff(timestamps).clip(0, 7 * DAY).astype(np.float32)
        items = history["item_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        organic = history["is_organic"].to_numpy(zero_copy_only=False).astype(np.int64)
        behaviors = (1 + (1 - organic)).astype(np.int64)
        target = row["target_index"]
        label = row["label"]
        output.append(
            P7Request(
                request_id=str(row["request_id"]),
                workload=workload,
                uid=int(row["uid"]),
                query_timestamp=int(row["query_timestamp"]),
                history_items=items,
                history_behaviors=behaviors,
                history_time_deltas=deltas,
                query_time_delta=float(
                    np.clip(int(row["query_timestamp"]) - int(timestamps[-1]), 0, 7 * DAY)
                ),
                candidate_ids=candidates[index],
                base_features=base_features[index],
                target_index=None if target is None else int(target),
                label=None if label is None else int(label),
                request_weight=float(row["request_weight"]),
                manifest_kind=str(row["manifest_kind"]),
                is_organic=None if row["is_organic"] is None else int(row["is_organic"]),
                prior_30m_same_item=None
                if row["prior_30m_same_item"] is None
                else bool(row["prior_30m_same_item"]),
                latest_item=None if row["latest_item"] is None else bool(row["latest_item"]),
                target_stratum=None
                if row["target_stratum"] is None
                else str(row["target_stratum"]),
                history_timestamps=timestamps,
            )
        )
    return output
