"""Contract-driven bounded Yambda listen-history loading."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

from hstu_kvcache.data.oov import apply_stable_oov_buckets
from hstu_kvcache.training.foundation import FoundationHistoryIndex


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def load_yambda_histories(
    dataset_manifest: Path,
    uids: list[int],
    *,
    known_vocab_size: int,
    oov_buckets: int,
    end_timestamp: int,
    start_timestamp: int | None = None,
    max_pre_events: int | None = None,
    threads: int = 4,
) -> FoundationHistoryIndex:
    """Load exact history, optionally bounding the prefix before a window.

    With ``start_timestamp`` and ``max_pre_events``, the result contains the last
    ``max_pre_events`` listens strictly before the window plus every listen in
    ``[start_timestamp, end_timestamp)``.  This exactly supports a model whose
    causal context is bounded by ``max_pre_events`` without loading future events.
    """
    if end_timestamp <= 0:
        raise ValueError("end_timestamp must be positive")
    if (start_timestamp is None) != (max_pre_events is None):
        raise ValueError("start_timestamp and max_pre_events must be supplied together")
    if start_timestamp is not None and not 0 <= start_timestamp < end_timestamp:
        raise ValueError("history window must be nonempty and nonnegative")
    if max_pre_events is not None and max_pre_events < 1:
        raise ValueError("max_pre_events must be positive")
    if not uids:
        return FoundationHistoryIndex({})

    dataset_manifest = dataset_manifest.resolve()
    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    root = dataset_manifest.parent
    listens = (root / dataset["shared_listens_glob"]).resolve()
    mapping = (root / dataset["item_mapping_path"]).resolve()
    placeholders = ",".join("?" for _ in uids)
    selected = [int(uid) for uid in uids]
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={int(threads)}")
    if start_timestamp is None:
        query = f"""
            SELECT l.uid,l.timestamp,l.raw_item_id,coalesce(m.item_idx,0) AS item_idx,l.behavior
            FROM read_parquet('{_sql_path(listens)}', hive_partitioning=true) l
            LEFT JOIN read_parquet('{_sql_path(mapping)}') m USING(raw_item_id)
            WHERE l.uid IN ({placeholders}) AND l.timestamp < ?
            ORDER BY l.uid,l.timestamp,l.raw_item_id,l.behavior
        """
        parameters = [*selected, int(end_timestamp)]
    else:
        query = f"""
            WITH pre AS (
              SELECT uid,timestamp,raw_item_id,behavior,is_organic
              FROM read_parquet('{_sql_path(listens)}', hive_partitioning=true)
              WHERE uid IN ({placeholders}) AND timestamp < ?
              QUALIFY row_number() OVER (
                PARTITION BY uid ORDER BY timestamp DESC,raw_item_id DESC,behavior DESC
              ) <= ?
            ), post AS (
              SELECT uid,timestamp,raw_item_id,behavior,is_organic
              FROM read_parquet('{_sql_path(listens)}', hive_partitioning=true)
              WHERE uid IN ({placeholders}) AND timestamp >= ? AND timestamp < ?
            ), bounded AS (
              SELECT * FROM pre UNION ALL SELECT * FROM post
            )
            SELECT l.uid,l.timestamp,l.raw_item_id,coalesce(m.item_idx,0) AS item_idx,l.behavior
            FROM bounded l LEFT JOIN read_parquet('{_sql_path(mapping)}') m USING(raw_item_id)
            ORDER BY l.uid,l.timestamp,l.raw_item_id,l.behavior
        """
        parameters = [
            *selected, int(start_timestamp), int(max_pre_events),
            *selected, int(start_timestamp), int(end_timestamp),
        ]
    try:
        table = connection.execute(query, parameters).fetch_arrow_table()
    finally:
        connection.close()
    item_ids = apply_stable_oov_buckets(
        table["raw_item_id"].to_numpy(), table["item_idx"].to_numpy(),
        known_vocab_size=int(known_vocab_size), buckets=int(oov_buckets),
    )
    return FoundationHistoryIndex.from_columns(
        table["uid"].to_numpy(), table["timestamp"].to_numpy(), np.asarray(item_ids),
        table["behavior"].to_numpy(),
    )
