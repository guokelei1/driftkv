from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def event_time_deltas(
    events: list[tuple[int, int, int]],
    *,
    previous_timestamp: int | None = None,
    max_delta_seconds: float = 86400.0 * 7.0,
) -> np.ndarray:
    """Return model time deltas for a contiguous event slice.

    ``forward_with_cache`` receives only the newly appended tokens, but the
    first appended token is still temporally adjacent to the final token in
    the materialized prefix.  Callers must therefore pass that prefix
    timestamp as ``previous_timestamp``.  Resetting the first delta to zero
    changes the appended token embedding and makes Full and Append follow
    different input contracts.
    """
    if not events:
        return np.empty(0, dtype=np.float32)
    timestamps = np.asarray([event[1] for event in events], dtype=np.int64)
    deltas = np.zeros(len(events), dtype=np.float32)
    if previous_timestamp is not None:
        deltas[0] = np.clip(timestamps[0] - int(previous_timestamp), 0, max_delta_seconds)
    if len(events) > 1:
        deltas[1:] = np.diff(timestamps).clip(0, max_delta_seconds)
    return deltas


@dataclass
class YambdaTrace:
    """Flat Yambda listening trace with contiguous model-facing IDs."""

    interactions: pd.DataFrame
    num_users: int
    num_items: int
    num_behaviors: int = 3
    user_map: dict[int, int] | None = None
    item_map: dict[int, int] | None = None


def load_yambda_listens(
    path: str | Path,
    *,
    min_interactions_per_user: int = 5,
    max_users: int | None = None,
    max_items: int | None = None,
    max_rows: int | None = None,
) -> YambdaTrace:
    """Load flat listens and prepare leak-free model-facing columns.

    This is intentionally a small loading primitive, not an experiment runner.
    Catalog maps are fitted on the earliest observed rows in the loaded input;
    callers constructing a formal release contract must pass a base-only file
    or implement an explicit base cutoff before fitting maps.
    """
    frame = pd.read_parquet(path, columns=[
        "uid", "timestamp", "item_id", "is_organic", "played_ratio_pct", "track_length_seconds",
    ])
    if max_rows is not None:
        frame = frame.iloc[:max_rows].copy()
    frame = frame.sort_values(["uid", "timestamp"], kind="stable").reset_index(drop=True)
    counts = frame.groupby("uid", sort=False).size()
    users = counts[counts >= min_interactions_per_user].index
    if max_users is not None and len(users) > max_users:
        users = counts.loc[users].sort_values(ascending=False, kind="stable").head(max_users).index
    frame = frame[frame["uid"].isin(users)].copy()
    user_ids = sorted(frame["uid"].unique().tolist())
    user_map = {int(value): index + 1 for index, value in enumerate(user_ids)}
    frame["user_idx"] = frame["uid"].map(user_map).astype(np.int64)

    item_counts = frame.groupby("item_id", sort=False).size()
    item_ids = item_counts.sort_values(ascending=False, kind="stable").index
    if max_items is not None:
        item_ids = item_ids[:max_items]
    item_map = {int(value): index + 1 for index, value in enumerate(item_ids)}
    frame = frame[frame["item_id"].isin(item_map)].copy()
    frame["item_idx"] = frame["item_id"].map(item_map).astype(np.int64)
    frame["behavior"] = 1 + (1 - frame["is_organic"].astype(np.int64))
    frame["label"] = (frame["played_ratio_pct"] > 50).astype(np.int8)
    # Yambda stores seconds rounded to a five-second precision; the values
    # themselves are already seconds and must not be multiplied by five.
    frame["time_delta"] = (
        frame.groupby("user_idx")["timestamp"].diff().fillna(0).astype(np.float32)
    ).clip(lower=0.0, upper=86400.0 * 7.0)
    return YambdaTrace(
        interactions=frame.reset_index(drop=True),
        num_users=len(user_map),
        num_items=len(item_map),
        user_map=user_map,
        item_map=item_map,
    )
