from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass
class KuaiRandTrace:
    """Streaming-ready KuaiRand behaviour trace.

    KuaiRand-1K ships two standard-log windows (Apr 8-21, Apr 22-May 8) with
    millisecond timestamps, allowing each stream window to produce a distinct
    model-version checkpoint.

    Columns of interest: user_id, video_id, time_ms, date, hourmin, and the
    multi-behaviour flags (is_click/like/follow/comment/forward/hate/long_view).
    """

    interactions: pd.DataFrame  # sorted by time_ms, reindexed user/video ids
    num_users: int
    num_items: int
    num_behaviors: int
    user_map: dict
    item_map: dict
    num_prediction_items: int | None = None
    context_hash_buckets: int = 0

    def __post_init__(self) -> None:
        if self.num_prediction_items is None:
            self.num_prediction_items = self.num_items
        if not 1 <= self.num_prediction_items <= self.num_items:
            raise ValueError("num_prediction_items must be in [1, num_items]")
        if self.context_hash_buckets < 0:
            raise ValueError("context_hash_buckets must be non-negative")


BEHAVIOR_NAMES = ("click", "like", "follow", "comment", "forward", "hate", "long_view")


def stable_context_hash(values: np.ndarray) -> np.ndarray:
    hashed = values.astype(np.uint64, copy=True)
    hashed = hashed + np.uint64(0x9E3779B97F4A7C15)
    hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    return hashed ^ (hashed >> np.uint64(31))


def load_kuairand(
    csv_paths: Iterable[str | Path],
    min_interactions_per_user: int = 5,
    max_seq_len: int = 512,
    max_items: int | None = 50000,
    max_users: int | None = None,
    fit_num_days: int | None = None,
    context_hash_buckets: int = 0,
) -> KuaiRandTrace:
    """Load one or more KuaiRand log CSVs into a single sorted trace.

    Args:
        csv_paths: log_standard_*.csv files (the dense recommendation logs).
            log_random_* is uniform-exposure and better suited for evaluation.
        min_interactions_per_user: drop cold users.
        max_seq_len: cap sequence length (older events truncated) for memory.
        max_items: keep only the top-N most frequent items (KuaiRand has ~2M
            unique videos; a tractable prediction catalog is required to fit
            the item embedding table on 4xA40). Without context hash buckets,
            interactions with dropped items are removed.
        max_users: optionally cap the user catalog (for fast feasibility runs).
    """
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p, usecols=[
            "user_id", "video_id", "time_ms", "date", "hourmin",
            "is_click", "is_like", "is_follow", "is_comment",
            "is_forward", "is_hate", "long_view", "play_time_ms", "duration_ms",
        ])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("time_ms").reset_index(drop=True)

    fit_df = df
    if fit_num_days is not None:
        dates = sorted(df["date"].astype(str).unique())[:fit_num_days]
        fit_df = df[df["date"].astype(str).isin(dates)]

    if context_hash_buckets < 0:
        raise ValueError("context_hash_buckets must be non-negative")

    if max_items is not None:
        item_counts = fit_df["video_id"].value_counts()
        keep_items = item_counts.head(max_items).index
        if context_hash_buckets == 0:
            df = df[df["video_id"].isin(keep_items)].reset_index(drop=True)
            fit_df = fit_df[fit_df["video_id"].isin(keep_items)]
    elif fit_num_days is not None:
        keep_items = fit_df["video_id"].unique()
        if context_hash_buckets == 0:
            df = df[df["video_id"].isin(keep_items)].reset_index(drop=True)
    else:
        keep_items = fit_df["video_id"].unique()

    behavior = np.ones(len(df), dtype=np.int8)
    for column, value in (
        ("long_view", 8),
        ("is_click", 2),
        ("is_like", 3),
        ("is_follow", 4),
        ("is_comment", 5),
        ("is_forward", 6),
        ("is_hate", 7),
    ):
        behavior[df[column].to_numpy(dtype=bool)] = value
    df["behavior"] = behavior
    df["label"] = (
        df["is_click"]
        | df["is_like"]
        | df["is_follow"]
        | df["is_comment"]
        | df["is_forward"]
        | df["long_view"]
    ).astype(int)

    # remap user/item ids to contiguous 1..N (0 reserved for padding)
    active_users = fit_df.groupby("user_id").size()
    keep_users = active_users[active_users >= min_interactions_per_user].index
    df = df[df["user_id"].isin(keep_users)]
    fit_df = fit_df[fit_df["user_id"].isin(keep_users)]
    user_ids = fit_df["user_id"].unique()
    fitted_items = set(keep_items)
    item_ids = fit_df.loc[fit_df["video_id"].isin(fitted_items), "video_id"].unique()
    if max_users is not None and len(user_ids) > max_users:
        ranked_users = (
            active_users.loc[user_ids]
            .rename("count")
            .reset_index()
            .sort_values(["count", "user_id"], ascending=[False, True], kind="stable")
        )
        user_ids = ranked_users["user_id"].to_numpy()[:max_users]
        df = df[df["user_id"].isin(user_ids)].reset_index(drop=True)
    user_map = {u: i + 1 for i, u in enumerate(user_ids)}
    item_map = {v: i + 1 for i, v in enumerate(item_ids)}
    df["user_idx"] = df["user_id"].map(user_map)
    known_items = df["video_id"].isin(item_map)
    if context_hash_buckets:
        item_idx = df["video_id"].map(item_map).to_numpy(
            dtype=np.float64,
            copy=True,
        )
        unknown = ~known_items.to_numpy()
        hashed = stable_context_hash(
            df.loc[unknown, "video_id"].to_numpy(dtype=np.int64)
        )
        item_idx[unknown] = (
            len(item_map)
            + 1
            + (hashed % np.uint64(context_hash_buckets)).astype(np.int64)
        )
        df["item_idx"] = item_idx.astype(np.int64)
        df.loc[~known_items, "label"] = 0
    else:
        df["item_idx"] = df["video_id"].map(item_map).astype(np.int64)
    df["is_prediction_item"] = known_items.to_numpy()

    # per-user time delta (seconds) between consecutive events
    df = df.sort_values(["user_idx", "time_ms"]).reset_index(drop=True)
    df["time_delta"] = df.groupby("user_idx")["time_ms"].diff().fillna(0.0) / 1000.0
    df["time_delta"] = df["time_delta"].clip(lower=0.0, upper=86400.0 * 7.0)

    return KuaiRandTrace(
        interactions=df,
        num_users=len(user_map),
        num_items=len(item_map) + context_hash_buckets,
        num_behaviors=len(BEHAVIOR_NAMES) + 2,
        user_map=user_map,
        item_map=item_map,
        num_prediction_items=len(item_map),
        context_hash_buckets=context_hash_buckets,
    )


def build_user_sequences(
    trace: KuaiRandTrace, max_seq_len: int = 512
) -> dict[int, dict]:
    """Materialise per-user sequences: {item_ids, behaviors, time_deltas, labels}."""
    df = trace.interactions
    out: dict[int, dict] = {}
    for u, grp in df.groupby("user_idx"):
        grp = grp.sort_values("time_ms")
        if len(grp) > max_seq_len:
            grp = grp.iloc[-max_seq_len:]
        out[int(u)] = {
            "item_ids": grp["item_idx"].to_numpy(dtype=np.int64),
            "behaviors": grp["behavior"].to_numpy(dtype=np.int64),
            "time_deltas": grp["time_delta"].to_numpy(dtype=np.float32),
            "labels": grp["label"].to_numpy(dtype=np.int64),
            "timestamps": grp["time_ms"].to_numpy(dtype=np.int64),
        }
    return out


def split_by_time(
    trace: KuaiRandTrace, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/dev/test split of the global interaction stream."""
    df = trace.interactions.sort_values("time_ms").reset_index(drop=True)
    n = len(df)
    n_tr = int(n * fractions[0])
    n_dv = int(n * fractions[1])
    return df.iloc[:n_tr], df.iloc[n_tr : n_tr + n_dv], df.iloc[n_tr + n_dv :]


def collate_batch(
    sequences: list[dict], max_seq_len: int = 512, pad_to: int | None = None
) -> dict:
    """Pad a list of user-sequence dicts into batched tensors (left-truncate)."""
    L = pad_to or max(len(s["item_ids"]) for s in sequences)
    L = min(L, max_seq_len)
    B = len(sequences)
    item_ids = np.zeros((B, L), dtype=np.int64)
    behaviors = np.zeros((B, L), dtype=np.int64)
    time_deltas = np.zeros((B, L), dtype=np.float32)
    has_labels = "labels" in sequences[0]
    has_train_mask = "train_mask" in sequences[0]
    labels = np.zeros((B, L), dtype=np.int64) if has_labels else None
    train_mask = np.zeros((B, L), dtype=np.bool_) if has_train_mask else None
    lengths = np.zeros(B, dtype=np.int64)
    for i, s in enumerate(sequences):
        arr = s["item_ids"][-L:]
        n = len(arr)
        item_ids[i, :n] = arr
        behaviors[i, :n] = s["behaviors"][-L:]
        time_deltas[i, :n] = s["time_deltas"][-L:]
        if has_labels:
            labels[i, :n] = s["labels"][-L:]
        if has_train_mask:
            train_mask[i, :n] = s["train_mask"][-L:]
        lengths[i] = n
    out = {
        "item_ids": torch.from_numpy(item_ids),
        "behaviors": torch.from_numpy(behaviors),
        "time_deltas": torch.from_numpy(time_deltas),
        "lengths": torch.from_numpy(lengths),
    }
    if has_labels:
        out["labels"] = torch.from_numpy(labels)
    if has_train_mask:
        out["train_mask"] = torch.from_numpy(train_mask)
    return out
