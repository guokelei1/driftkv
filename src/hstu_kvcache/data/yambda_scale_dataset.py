"""Windowed loader for the unified Yambda-500M S/M/L processed store."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import duckdb
import pyarrow as pa
import yaml


FeedbackPurpose = Literal["train", "evaluation"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


class YambdaScaleDataset:
    """Read one logical scale from the shared daily Parquet store.

    The dataset manifest fixes UID membership, mapping and default feedback
    access bounds. Listen histories remain readable across the full raw span;
    feedback outside the unlocked train/evaluation boundary requires a separate
    contract bound to this exact dataset manifest.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        feedback_unlock_contract: str | Path | None = None,
        threads: int = 4,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.root = self.manifest_path.parent
        self.rank_limit = int(self.manifest["rank_limit"])
        self.listens_glob = (self.root / self.manifest["shared_listens_glob"]).resolve()
        self.feedback_glob = (self.root / self.manifest["shared_feedback_glob"]).resolve()
        self.item_mapping = (self.root / self.manifest["item_mapping_path"]).resolve()
        self.threads = int(threads)
        if self.threads < 1:
            raise ValueError("threads must be positive")
        access = self.manifest["feedback_access"]
        self.feedback_ends = {
            "train": int(access["default_training_end_exclusive"]),
            "evaluation": int(access["default_evaluation_end_exclusive"]),
        }
        if feedback_unlock_contract is not None:
            self._apply_unlock(Path(feedback_unlock_contract).resolve())

    def _apply_unlock(self, path: Path) -> None:
        text = path.read_text()
        value = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        expected = value.get("dataset_manifest_sha256")
        if expected != _sha256(self.manifest_path):
            raise ValueError("feedback unlock contract targets a different dataset manifest")
        if value.get("theta3_authorized") is not True:
            raise ValueError("feedback unlock contract does not authorize theta3")
        for purpose in ("train", "evaluation"):
            key = f"{purpose}_end_exclusive"
            if key in value:
                self.feedback_ends[purpose] = int(value[key])

    @staticmethod
    def _validate_window(start: int, end: int) -> tuple[int, int, int, int]:
        start, end = int(start), int(end)
        if start < 0 or end <= start:
            raise ValueError("time window must be non-empty and non-negative")
        bucket_seconds = 7 * 86_400
        return start, end, start // bucket_seconds, (end - 1) // bucket_seconds

    def _connection(self) -> duckdb.DuckDBPyConnection:
        connection = duckdb.connect()
        connection.execute(f"PRAGMA threads={self.threads}")
        return connection

    def _listen_sql(self, start: int, end: int) -> str:
        start, end, first_week, last_week = self._validate_window(start, end)
        return f"""
            SELECT e.uid, e.selector_rank, e.timestamp,
                   coalesce(m.item_idx, 0)::UBIGINT AS item_idx,
                   e.raw_item_id, e.behavior, e.is_organic,
                   e.played_ratio_pct, e.track_length_seconds
            FROM read_parquet('{_sql_path(self.listens_glob)}', hive_partitioning=true) e
            LEFT JOIN read_parquet('{_sql_path(self.item_mapping)}') m USING (raw_item_id)
            WHERE e.week BETWEEN {first_week} AND {last_week}
              AND e.selector_rank <= {self.rank_limit}
              AND e.timestamp >= {start} AND e.timestamp < {end}
            ORDER BY e.uid, e.timestamp, e.raw_item_id
        """

    def _feedback_sql(
        self,
        start: int,
        end: int,
        *,
        purpose: FeedbackPurpose,
        require_known_item: bool,
    ) -> str:
        start, end, first_week, last_week = self._validate_window(start, end)
        if purpose not in self.feedback_ends:
            raise ValueError(f"unknown feedback purpose: {purpose}")
        if end > self.feedback_ends[purpose]:
            raise PermissionError(
                f"{purpose} feedback is locked at {self.feedback_ends[purpose]}; "
                "a bound prospective unlock contract is required"
            )
        known = "AND m.item_idx IS NOT NULL" if require_known_item else ""
        return f"""
            SELECT e.uid, e.selector_rank, e.timestamp,
                   coalesce(m.item_idx, 0)::UBIGINT AS item_idx,
                   e.raw_item_id, e.label, e.is_organic,
                   (m.item_idx IS NOT NULL) AS target_known
            FROM read_parquet('{_sql_path(self.feedback_glob)}', hive_partitioning=true) e
            LEFT JOIN read_parquet('{_sql_path(self.item_mapping)}') m USING (raw_item_id)
            WHERE e.week BETWEEN {first_week} AND {last_week}
              AND e.selector_rank <= {self.rank_limit}
              AND e.timestamp >= {start} AND e.timestamp < {end}
              {known}
            ORDER BY e.uid, e.timestamp, e.raw_item_id, e.label
        """

    def _batches(self, sql: str, batch_size: int) -> Iterator[pa.RecordBatch]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        connection = self._connection()
        try:
            reader = connection.execute(sql).fetch_record_batch(rows_per_batch=batch_size)
            yield from reader
        finally:
            connection.close()

    def iter_listens(self, start: int, end: int, *, batch_size: int = 262_144) -> Iterator[pa.RecordBatch]:
        yield from self._batches(self._listen_sql(start, end), batch_size)

    def iter_feedback(
        self,
        start: int,
        end: int,
        *,
        purpose: FeedbackPurpose,
        require_known_item: bool = True,
        batch_size: int = 262_144,
    ) -> Iterator[pa.RecordBatch]:
        yield from self._batches(
            self._feedback_sql(
                start,
                end,
                purpose=purpose,
                require_known_item=require_known_item,
            ),
            batch_size,
        )

    def count_listens(self, start: int, end: int) -> int:
        sql = self._listen_sql(start, end)
        connection = self._connection()
        try:
            return int(connection.execute(f"SELECT count(*) FROM ({sql})").fetchone()[0])
        finally:
            connection.close()

    def count_feedback(
        self,
        start: int,
        end: int,
        *,
        purpose: FeedbackPurpose,
        require_known_item: bool = True,
    ) -> int:
        sql = self._feedback_sql(
            start,
            end,
            purpose=purpose,
            require_known_item=require_known_item,
        )
        connection = self._connection()
        try:
            return int(connection.execute(f"SELECT count(*) FROM ({sql})").fetchone()[0])
        finally:
            connection.close()
