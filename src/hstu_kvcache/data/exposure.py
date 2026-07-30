from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .kuairand import KuaiRandTrace
from .streaming_plan import StreamingDataPlan


def load_prepared_exposure_plan(
    path: str | Path,
    max_seq_len: int,
) -> tuple[StreamingDataPlan, dict]:
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        user_idx = source["user_idx"].astype(np.int64, copy=True)
        item_idx = source["item_idx"].astype(np.int64, copy=True)
        behavior = source["behavior"].astype(np.int64, copy=True)
        label = source["label"].astype(np.int64, copy=True)
        time_ms = source["time_ms"].astype(np.int64, copy=True)
        window_index = source["window_index"].astype(np.int8, copy=True)
    lengths = {
        len(user_idx),
        len(item_idx),
        len(behavior),
        len(label),
        len(time_ms),
        len(window_index),
    }
    if len(lengths) != 1:
        raise ValueError("prepared exposure arrays have inconsistent lengths")
    window_count = int(metadata["window_count"])
    if np.any((window_index < -1) | (window_index >= window_count)):
        raise ValueError("prepared exposure window index is out of range")
    if len(user_idx) == 0:
        raise ValueError("prepared exposure stream is empty")
    if user_idx.min() < 1 or user_idx.max() > int(metadata["selected_users"]):
        raise ValueError("prepared exposure user index is out of range")
    if item_idx.min() < 1 or item_idx.max() > int(metadata["fitted_items"]):
        raise ValueError("prepared exposure item index is out of range")
    num_prediction_items = int(
        metadata.get("num_prediction_items", metadata["fitted_items"])
    )
    context_hash_buckets = int(metadata.get("context_hash_buckets", 0))
    if (
        not 1 <= num_prediction_items <= int(metadata["fitted_items"])
        or context_hash_buckets < 0
        or num_prediction_items + context_hash_buckets
        != int(metadata["fitted_items"])
    ):
        raise ValueError("prepared exposure item roles are invalid")
    if np.any((item_idx > num_prediction_items) & (label > 0)):
        raise ValueError(
            "prepared exposure context-only items cannot be positive targets"
        )
    names = np.array(
        ["base", *[f"window_{index}" for index in range(window_count)]],
        dtype=object,
    )
    frame = pd.DataFrame(
        {
            "date": names[window_index.astype(np.int64) + 1],
            "user_idx": user_idx,
            "item_idx": item_idx,
            "behavior": behavior,
            "label": label,
            "time_ms": time_ms,
        }
    )
    trace = KuaiRandTrace(
        interactions=frame,
        num_users=int(metadata["selected_users"]),
        num_items=int(metadata["fitted_items"]),
        num_behaviors=int(metadata["num_behaviors"]),
        user_map={},
        item_map={},
        num_prediction_items=num_prediction_items,
        context_hash_buckets=context_hash_buckets,
    )
    plan = StreamingDataPlan(
        trace=trace,
        base_dates=["base"],
        stream_dates=[f"window_{index}" for index in range(window_count)],
        max_seq_len=max_seq_len,
        max_items=trace.num_items,
    )
    return plan, metadata
