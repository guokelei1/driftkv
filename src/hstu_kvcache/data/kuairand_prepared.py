from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .kuairand import KuaiRandTrace
from .streaming_plan import StreamingDataPlan

PREPARED_PROTOCOL = "kuairand_long_context_8plus8_data_v2"
SUPPORTED_BASE_DAYS = (4, 6, 8)


def _expected_protocol(base_days: int, online_days: int) -> str:
    if base_days not in SUPPORTED_BASE_DAYS or base_days + online_days != 16:
        raise ValueError(
            "prepared long-context protocol requires a supported 16-day split"
        )
    if base_days == 8:
        return PREPARED_PROTOCOL
    return (
        f"kuairand_long_context_{base_days}plus{online_days}"
        "_data_exploration_v1"
    )


def _ordered_raw_ids(mapping: dict) -> np.ndarray:
    return np.asarray(
        [
            raw_id
            for raw_id, _ in sorted(
                mapping.items(),
                key=lambda value: value[1],
            )
        ],
        dtype=np.int64,
    )


def prepared_metadata(
    plan: StreamingDataPlan,
    source_paths: list[str],
    protocol: str = PREPARED_PROTOCOL,
) -> dict:
    dates = plan.base_dates + plan.stream_dates
    frames = [plan.daily_segments[date] for date in dates]
    frame = pd.concat(frames, ignore_index=True)
    counts = frame.groupby(frame["date"].astype(str)).size()
    return {
        "protocol": protocol,
        "schema_version": 1,
        "source_paths": source_paths,
        "total_dates": len(dates),
        "base_date_count": len(plan.base_dates),
        "online_date_count": len(plan.stream_dates),
        "base_dates": plan.base_dates,
        "online_dates": plan.stream_dates,
        "max_seq_len": plan.max_seq_len,
        "history_window_days": plan.history_window_days,
        "max_prediction_items": plan.max_items,
        "num_users": plan.num_users,
        "num_context_items": plan.num_items,
        "num_prediction_items": plan.num_prediction_items,
        "context_hash_buckets": plan.trace.context_hash_buckets,
        "context_hash_function": "splitmix64_v1",
        "num_behaviors": plan.num_behaviors,
        "selected_rows": len(frame),
        "prediction_catalog_rows": int(frame["is_prediction_item"].sum()),
        "eligible_engaged_targets": int(frame["label"].sum()),
        "rows_per_date": {
            date: int(counts.loc[date])
            for date in dates
        },
        "cohort_rule": "base-only users with at least five base-period interactions",
        "prediction_catalog_rule": "base-only top-50000 raw video ids",
        "context_rule": (
            "all selected exposures retained; out-of-catalog video ids use "
            "262144 stable context-only hash buckets"
        ),
    }


def save_prepared_kuairand_plan(
    plan: StreamingDataPlan,
    path: str | Path,
    source_paths: list[str],
    overwrite: bool = False,
    protocol: str = PREPARED_PROTOCOL,
) -> dict:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"prepared artifact already exists: {path}")
    dates = plan.base_dates + plan.stream_dates
    expected_protocol = _expected_protocol(
        len(plan.base_dates),
        len(plan.stream_dates),
    )
    if protocol != expected_protocol:
        raise ValueError(
            f"prepared protocol must be {expected_protocol!r} for this date split"
        )
    date_to_index = {date: index for index, date in enumerate(dates)}
    frame = pd.concat(
        [plan.daily_segments[date] for date in dates],
        ignore_index=True,
    )
    date_index = frame["date"].astype(str).map(date_to_index).to_numpy(dtype=np.int8)
    if np.any(date_index < 0):
        raise ValueError("interaction date is outside the selected 16-date horizon")
    item_idx = frame["item_idx"].to_numpy(dtype=np.int32)
    labels = frame["label"].to_numpy(dtype=np.uint8)
    if np.any((item_idx > plan.num_prediction_items) & (labels > 0)):
        raise ValueError("context-only hash buckets cannot be prediction targets")
    metadata = prepared_metadata(plan, source_paths, protocol)
    user_ids = _ordered_raw_ids(plan.trace.user_map)
    prediction_video_ids = _ordered_raw_ids(plan.trace.item_map)
    if len(user_ids) != plan.num_users:
        raise ValueError("user mapping and trace user count differ")
    if len(prediction_video_ids) != plan.num_prediction_items:
        raise ValueError("item mapping and prediction catalog size differ")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        user_idx=frame["user_idx"].to_numpy(dtype=np.int16),
        item_idx=item_idx,
        behavior=frame["behavior"].to_numpy(dtype=np.uint8),
        label=labels,
        time_ms=frame["time_ms"].to_numpy(dtype=np.int64),
        date_index=date_index,
        user_ids_by_index=user_ids,
        prediction_video_ids_by_index=prediction_video_ids,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(path)
    return metadata


def load_prepared_kuairand_plan(
    path: str | Path,
) -> tuple[StreamingDataPlan, dict]:
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        user_idx = source["user_idx"].astype(np.int64, copy=True)
        item_idx = source["item_idx"].astype(np.int64, copy=True)
        behavior = source["behavior"].astype(np.int64, copy=True)
        label = source["label"].astype(np.int64, copy=True)
        time_ms = source["time_ms"].astype(np.int64, copy=True)
        date_index = source["date_index"].astype(np.int64, copy=True)
        user_ids = source["user_ids_by_index"].astype(np.int64, copy=True)
        prediction_video_ids = source["prediction_video_ids_by_index"].astype(
            np.int64,
            copy=True,
        )
    base_count = int(metadata["base_date_count"])
    online_count = int(metadata["online_date_count"])
    expected_protocol = _expected_protocol(base_count, online_count)
    if metadata.get("protocol") != expected_protocol:
        raise ValueError(
            "prepared artifact protocol does not match its recorded date split"
        )
    lengths = {
        len(user_idx),
        len(item_idx),
        len(behavior),
        len(label),
        len(time_ms),
        len(date_index),
    }
    if lengths != {int(metadata["selected_rows"])}:
        raise ValueError("prepared artifact arrays have inconsistent lengths")
    dates = metadata["base_dates"] + metadata["online_dates"]
    if (
        len(dates) != 16
        or len(metadata["base_dates"]) != base_count
        or len(metadata["online_dates"]) != online_count
    ):
        raise ValueError("prepared artifact date counts are inconsistent")
    if np.any((date_index < 0) | (date_index >= len(dates))):
        raise ValueError("prepared artifact date index is out of range")
    if user_idx.min() < 1 or user_idx.max() > int(metadata["num_users"]):
        raise ValueError("prepared artifact user index is out of range")
    if item_idx.min() < 1 or item_idx.max() > int(metadata["num_context_items"]):
        raise ValueError("prepared artifact context item index is out of range")
    prediction_items = int(metadata["num_prediction_items"])
    if np.any((item_idx > prediction_items) & (label > 0)):
        raise ValueError("prepared artifact has a context-only prediction target")
    if len(user_ids) != int(metadata["num_users"]):
        raise ValueError("prepared user id table has the wrong size")
    if len(prediction_video_ids) != prediction_items:
        raise ValueError("prepared item id table has the wrong size")
    frame = pd.DataFrame(
        {
            "date": np.asarray(dates, dtype=object)[date_index],
            "user_idx": user_idx,
            "item_idx": item_idx,
            "behavior": behavior,
            "label": label,
            "time_ms": time_ms,
            "is_prediction_item": item_idx <= prediction_items,
        }
    )
    user_map = {
        int(raw_id): index + 1
        for index, raw_id in enumerate(user_ids)
    }
    item_map = {
        int(raw_id): index + 1
        for index, raw_id in enumerate(prediction_video_ids)
    }
    trace = KuaiRandTrace(
        interactions=frame,
        num_users=int(metadata["num_users"]),
        num_items=int(metadata["num_context_items"]),
        num_behaviors=int(metadata["num_behaviors"]),
        user_map=user_map,
        item_map=item_map,
        num_prediction_items=prediction_items,
        context_hash_buckets=int(metadata["context_hash_buckets"]),
    )
    plan = StreamingDataPlan(
        trace=trace,
        base_dates=list(metadata["base_dates"]),
        stream_dates=list(metadata["online_dates"]),
        max_seq_len=int(metadata["max_seq_len"]),
        max_items=int(metadata["max_prediction_items"]),
        history_window_days=int(metadata["history_window_days"]),
    )
    return plan, metadata
