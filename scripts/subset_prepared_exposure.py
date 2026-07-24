from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--protocol", default="motivation_joint_scale_prepared_v1")
    parser.add_argument("--catalog-size", type=int, required=True)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--base-prefix", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--min-window-events", type=int, default=1)
    return parser.parse_args()


def build_subset(
    arrays: dict[str, np.ndarray],
    metadata: dict,
    catalog_size: int,
    requested_users: int,
    base_prefix: int,
    window_size: int,
    windows: int,
    min_window_events: int,
) -> tuple[dict[str, np.ndarray], dict]:
    if catalog_size <= 0 or catalog_size > int(metadata["fitted_items"]):
        raise ValueError("catalog size exceeds the source vocabulary")
    if requested_users <= 0:
        raise ValueError("users must be positive")
    if min_window_events < 0:
        raise ValueError("minimum window events cannot be negative")
    user_idx = arrays["user_idx"].astype(np.int64, copy=False)
    item_idx = arrays["item_idx"].astype(np.int64, copy=False)
    positions = arrays["time_ms"].astype(np.int64, copy=False) // 1000
    limit = base_prefix + window_size * windows
    item_mask = (item_idx <= catalog_size) & (positions < limit)
    window_index = np.where(
        positions < base_prefix,
        -1,
        (positions - base_prefix) // window_size,
    ).astype(np.int8)
    source_users = int(metadata["selected_users"])
    base_counts = np.bincount(
        user_idx[item_mask & (window_index < 0)],
        minlength=source_users + 1,
    )
    window_counts = np.zeros((source_users + 1, windows), dtype=np.int32)
    stream_mask = item_mask & (window_index >= 0) & (window_index < windows)
    np.add.at(
        window_counts,
        (user_idx[stream_mask], window_index[stream_mask]),
        1,
    )
    eligible = np.flatnonzero(
        (base_counts > 0)
        & (window_counts >= min_window_events).all(axis=1)
    )
    eligible = eligible[eligible > 0]
    eligible = eligible[
        np.lexsort((eligible, -base_counts[eligible].astype(np.int64)))
    ]
    selected = eligible[:requested_users]
    if len(selected) < requested_users:
        raise ValueError(
            f"requested {requested_users} users but only {len(selected)} are eligible"
        )
    user_map = np.zeros(source_users + 1, dtype=np.int32)
    user_map[selected] = np.arange(1, len(selected) + 1, dtype=np.int32)
    row_mask = item_mask & (user_map[user_idx] > 0)
    output = {
        name: values[row_mask].copy()
        for name, values in arrays.items()
        if name not in ("original_user_ids", "original_item_ids", "window_index")
    }
    output["user_idx"] = user_map[user_idx[row_mask]]
    output["window_index"] = window_index[row_mask]
    source_original_users = arrays["original_user_ids"]
    output["original_user_ids"] = source_original_users[selected - 1].copy()
    output["original_item_ids"] = arrays["original_item_ids"][:catalog_size].copy()
    split_rows = {
        "base": int(np.count_nonzero(output["window_index"] == -1)),
        **{
            f"window_{index}": int(
                np.count_nonzero(output["window_index"] == index)
            )
            for index in range(windows)
        },
    }
    split_positive = {
        name: int(
            output["label"][
                output["window_index"]
                == (-1 if name == "base" else int(name[7:]))
            ].sum()
        )
        for name in split_rows
    }
    output_metadata = {
        **metadata,
        "protocol": "motivation_joint_scale_prepared_v1",
        "source_prepared": metadata.get("output"),
        "requested_items": catalog_size,
        "fitted_items": catalog_size,
        "requested_users": requested_users,
        "selected_users": int(len(selected)),
        "base_prefix": base_prefix,
        "window_size": window_size,
        "window_count": windows,
        "minimum_retained_events_per_window": min_window_events,
        "cohort_selection": (
            "complete retained-exposure horizon, then highest retained base activity, "
            "user id as deterministic tie-break; labels are not used"
        ),
        "cohort_selection_mode": "complete_retained_horizon",
        "catalog_fit": (
            f"top-{catalog_size} prefix of the source base-only item-frequency order"
        ),
        "rows": int(len(output["user_idx"])),
        "positive_rows": int(output["label"].sum()),
        "split_rows": split_rows,
        "split_positive_rows": split_positive,
        "retained_base_events": {
            "min": int(base_counts[selected].min()),
            "mean": float(base_counts[selected].mean()),
            "max": int(base_counts[selected].max()),
        },
    }
    return output, output_metadata


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    with np.load(source, allow_pickle=False) as loaded:
        arrays = {
            name: loaded[name].copy()
            for name in loaded.files
            if name != "metadata_json"
        }
        metadata = json.loads(str(loaded["metadata_json"].item()))
    output, metadata = build_subset(
        arrays,
        metadata,
        args.catalog_size,
        args.users,
        args.base_prefix,
        args.window_size,
        args.windows,
        args.min_window_events,
    )
    metadata["protocol"] = args.protocol
    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata["source_prepared"] = str(source)
    metadata["output"] = str(output_path)
    np.savez_compressed(
        output_path,
        **output,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    metadata["output_bytes"] = output_path.stat().st_size
    save_json(metadata, metadata_path)
    print(metadata_path)
    print(output_path)


if __name__ == "__main__":
    main()
