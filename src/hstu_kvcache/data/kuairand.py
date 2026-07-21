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
    millisecond timestamps - ideal for streaming drift: each window yields a
    distinct theta checkpoint when the model is retrained on the new window.

    Columns of interest: user_id, video_id, time_ms, date, hourmin, and the
    multi-behaviour flags (is_click/like/follow/comment/forward/hate/long_view).
    """

    interactions: pd.DataFrame  # sorted by time_ms, reindexed user/video ids
    num_users: int
    num_items: int
    num_behaviors: int
    user_map: dict
    item_map: dict


BEHAVIOR_NAMES = ("click", "like", "follow", "comment", "forward", "hate", "long_view")


def load_kuairand(
    csv_paths: Iterable[str | Path],
    min_interactions_per_user: int = 5,
    max_seq_len: int = 512,
    max_items: int | None = 50000,
    max_users: int | None = None,
) -> KuaiRandTrace:
    """Load one or more KuaiRand log CSVs into a single sorted trace.

    Args:
        csv_paths: log_standard_*.csv files (the dense recommendation logs).
            log_random_* is uniform-exposure and better suited for evaluation.
        min_interactions_per_user: drop cold users.
        max_seq_len: cap sequence length (older events truncated) for memory.
        max_items: keep only the top-N most frequent items (KuaiRand has ~2M
            unique videos; filtering to a tractable catalog is required to fit
            the item embedding table on 4xA40). Interactions with dropped items
            are removed.
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

    # optional catalog truncation: keep top-N most frequent items
    if max_items is not None:
        item_counts = df["video_id"].value_counts()
        keep_items = item_counts.head(max_items).index
        df = df[df["video_id"].isin(keep_items)].reset_index(drop=True)

    # encode the multi-behaviour signal into a single behaviour id:
    # 0 = padding, 1 = view(no-engage), 2=click, 3=like, 4=follow, 5=comment,
    # 6=forward, 7=hate, 8=long_view. Priority: most informative first.
    def _behavior_id(row: pd.Series) -> int:
        if row["is_hate"]:
            return 7
        if row["is_forward"]:
            return 6
        if row["is_comment"]:
            return 5
        if row["is_follow"]:
            return 4
        if row["is_like"]:
            return 3
        if row["is_click"]:
            return 2
        if row["long_view"]:
            return 8
        return 1

    df["behavior"] = df.apply(_behavior_id, axis=1)
    # positive label = any positive engagement (exclude hate)
    df["label"] = ((df["is_click"] | df["is_like"] | df["is_follow"] | df["is_comment"] | df["is_forward"] | df["long_view"]).astype(int))

    # remap user/item ids to contiguous 1..N (0 reserved for padding)
    active_users = df.groupby("user_id").size()
    keep_users = active_users[active_users >= min_interactions_per_user].index
    df = df[df["user_id"].isin(keep_users)]
    user_ids = df["user_id"].unique()
    item_ids = df["video_id"].unique()
    if max_users is not None and len(user_ids) > max_users:
        user_ids = user_ids[:max_users]
        df = df[df["user_id"].isin(user_ids)].reset_index(drop=True)
    user_map = {u: i + 1 for i, u in enumerate(user_ids)}
    item_map = {v: i + 1 for i, v in enumerate(item_ids)}
    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["video_id"].map(item_map)

    # per-user time delta (seconds) between consecutive events
    df = df.sort_values(["user_idx", "time_ms"]).reset_index(drop=True)
    df["time_delta"] = df.groupby("user_idx")["time_ms"].diff().fillna(0.0) / 1000.0
    df["time_delta"] = df["time_delta"].clip(lower=0.0, upper=86400.0 * 7.0)

    return KuaiRandTrace(
        interactions=df,
        num_users=len(user_map),
        num_items=len(item_map),
        num_behaviors=len(BEHAVIOR_NAMES) + 2,
        user_map=user_map,
        item_map=item_map,
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
    labels = np.zeros((B, L), dtype=np.int64) if has_labels else None
    lengths = np.zeros(B, dtype=np.int64)
    for i, s in enumerate(sequences):
        arr = s["item_ids"][-L:]
        n = len(arr)
        item_ids[i, :n] = arr
        behaviors[i, :n] = s["behaviors"][-L:]
        time_deltas[i, :n] = s["time_deltas"][-L:]
        if has_labels:
            labels[i, :n] = s["labels"][-L:]
        lengths[i] = n
    out = {
        "item_ids": torch.from_numpy(item_ids),
        "behaviors": torch.from_numpy(behaviors),
        "time_deltas": torch.from_numpy(time_deltas),
        "lengths": torch.from_numpy(lengths),
    }
    if has_labels:
        out["labels"] = torch.from_numpy(labels)
    return out
