from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from exposure_data_audit import (
    DEFAULT_INPUTS,
    DEFAULT_MEMBERS,
    chunk_reader,
    first_pass,
    group_layout,
    positive_mask,
    update_group_counts,
)

from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DEFAULT_INPUTS), required=True)
    parser.add_argument("--input")
    parser.add_argument("--member")
    parser.add_argument("--output")
    parser.add_argument("--metadata-output")
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--base-prefix", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--catalog-size", type=int, default=50000)
    parser.add_argument("--users", type=int, default=5000)
    parser.add_argument("--min-base-events", type=int, default=48)
    parser.add_argument(
        "--cohort-selection",
        choices=("base_activity", "complete_horizon"),
        default="base_activity",
    )
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def resolve_outputs(args: argparse.Namespace) -> tuple[Path, Path]:
    stem = args.dataset.replace("-", "_")
    default = f"{stem}_top{args.catalog_size}_users{args.users}"
    output = Path(args.output or f"data/processed/{default}.npz")
    metadata = Path(
        args.metadata_output or f"results/dataset_audit/{default}_prepared.json"
    )
    return output, metadata


def retained_base_counts(
    dataset: str,
    path: Path,
    member: str | None,
    chunk_size: int,
    max_rows: int | None,
    base_prefix: int,
    keep_items: np.ndarray,
    user_capacity: int,
) -> np.ndarray:
    output = np.zeros(user_capacity, dtype=np.uint16)
    seen_counts = np.zeros(user_capacity, dtype=np.int32)
    previous_user: int | None = None
    rows = 0
    started = time.perf_counter()
    with chunk_reader(dataset, path, member, chunk_size, max_rows) as reader:
        for chunk_index, chunk in enumerate(reader, start=1):
            users = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
            items = chunk["item_id"].to_numpy(dtype=np.int64, copy=False)
            starts, _, group_users, _, positions, _ = group_layout(
                users, seen_counts, previous_user
            )
            selected = (positions < base_prefix) & keep_items[items]
            update_group_counts(output, selected, starts, group_users)
            rows += len(chunk)
            previous_user = int(users[-1])
            print(
                f"pass=cohort dataset={dataset} chunks={chunk_index} rows={rows:,} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return output


def behavior_values(dataset: str, chunk, selected: np.ndarray) -> np.ndarray:
    if dataset.startswith("tenrec"):
        output = np.ones(int(np.count_nonzero(selected)), dtype=np.int8)
        for column, value in (("click", 2), ("like", 3), ("follow", 4), ("share", 5)):
            active = chunk[column].to_numpy(dtype=np.bool_, copy=False)[selected]
            output[active] = value
        return output
    positive = chunk["click_timestamp"].to_numpy(dtype=np.int64, copy=False)[selected] > 0
    return np.where(positive, 2, 1).astype(np.int8)


def materialize(
    dataset: str,
    path: Path,
    member: str | None,
    chunk_size: int,
    max_rows: int | None,
    base_prefix: int,
    window_size: int,
    windows: int,
    user_map: np.ndarray,
    item_map: np.ndarray,
) -> dict[str, np.ndarray]:
    columns: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "user_idx",
            "item_idx",
            "behavior",
            "label",
            "time_ms",
            "window_index",
        )
    }
    seen_counts = np.zeros(len(user_map), dtype=np.int32)
    previous_user: int | None = None
    limit = base_prefix + window_size * windows
    rows = 0
    kept = 0
    started = time.perf_counter()
    with chunk_reader(dataset, path, member, chunk_size, max_rows) as reader:
        for chunk_index, chunk in enumerate(reader, start=1):
            users = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
            items = chunk["item_id"].to_numpy(dtype=np.int64, copy=False)
            _, _, _, _, positions, _ = group_layout(users, seen_counts, previous_user)
            mapped_users = user_map[users]
            mapped_items = item_map[items]
            selected = (mapped_users > 0) & (mapped_items > 0) & (positions < limit)
            if selected.any():
                selected_positions = positions[selected]
                positive = positive_mask(dataset, chunk)[selected]
                window_index = np.where(
                    selected_positions < base_prefix,
                    -1,
                    (selected_positions - base_prefix) // window_size,
                ).astype(np.int8)
                if dataset == "zhihurec":
                    timestamp = chunk["impression_timestamp"].to_numpy(
                        dtype=np.int64, copy=False
                    )[selected]
                    time_ms = timestamp * 1000 + selected_positions % 1000
                else:
                    time_ms = selected_positions * 1000
                columns["user_idx"].append(mapped_users[selected].astype(np.int32))
                columns["item_idx"].append(mapped_items[selected].astype(np.int32))
                columns["behavior"].append(behavior_values(dataset, chunk, selected))
                columns["label"].append(positive.astype(np.int8))
                columns["time_ms"].append(time_ms.astype(np.int64))
                columns["window_index"].append(window_index)
                kept += int(np.count_nonzero(selected))
            rows += len(chunk)
            previous_user = int(users[-1])
            print(
                f"pass=materialize dataset={dataset} chunks={chunk_index} rows={rows:,} "
                f"kept={kept:,} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return {
        name: np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        for name, parts in columns.items()
    }


def main() -> None:
    args = parse_args()
    if args.max_rows is not None and args.output is None:
        raise ValueError("--output is required when --max-rows is set")
    path = Path(args.input or DEFAULT_INPUTS[args.dataset])
    member = args.member or DEFAULT_MEMBERS.get(args.dataset)
    output, metadata_output = resolve_outputs(args)
    raw = first_pass(
        args.dataset,
        path,
        member,
        args.chunk_size,
        args.max_rows,
        args.base_prefix,
        args.window_size,
        args.windows,
    )
    arrays = raw.pop("arrays")
    base_item_counts = arrays["base_item_counts"]
    base_items = np.flatnonzero(base_item_counts)
    item_order = base_items[
        np.argsort(-base_item_counts[base_items], kind="stable")
    ][: args.catalog_size]
    keep_items = np.zeros(len(base_item_counts), dtype=np.bool_)
    keep_items[item_order] = True
    base_counts = retained_base_counts(
        args.dataset,
        path,
        member,
        args.chunk_size,
        args.max_rows,
        args.base_prefix,
        keep_items,
        len(arrays["user_counts"]),
    )
    candidate_mask = base_counts >= args.min_base_events
    limit = args.base_prefix + args.window_size * args.windows
    if args.cohort_selection == "complete_horizon":
        candidate_mask &= arrays["user_counts"] >= limit
    candidates = np.flatnonzero(candidate_mask)
    candidates = candidates[
        np.lexsort((candidates, -base_counts[candidates].astype(np.int64)))
    ]
    selected_users = candidates[: args.users]
    if len(selected_users) == 0:
        raise ValueError("no users satisfy the base-only cohort rule")
    user_map = np.zeros(len(base_counts), dtype=np.int32)
    user_map[selected_users] = np.arange(1, len(selected_users) + 1, dtype=np.int32)
    item_map = np.zeros(len(base_item_counts), dtype=np.int32)
    item_map[item_order] = np.arange(1, len(item_order) + 1, dtype=np.int32)
    prepared = materialize(
        args.dataset,
        path,
        member,
        args.chunk_size,
        args.max_rows,
        args.base_prefix,
        args.window_size,
        args.windows,
        user_map,
        item_map,
    )
    split_rows = {
        "base": int(np.count_nonzero(prepared["window_index"] == -1)),
        **{
            f"window_{index}": int(
                np.count_nonzero(prepared["window_index"] == index)
            )
            for index in range(args.windows)
        },
    }
    split_positive = {
        name: int(
            prepared["label"][prepared["window_index"] == (-1 if name == "base" else int(name[7:]))].sum()
        )
        for name in split_rows
    }
    metadata = {
        "protocol": "ordered_exposure_prepared_v1",
        "dataset": args.dataset,
        "source": str(path),
        "member": member,
        "source_rows_scanned": raw["rows"],
        "catalog_fit": f"top-{args.catalog_size} from first {args.base_prefix} raw exposures per user",
        "requested_items": args.catalog_size,
        "fitted_items": int(len(item_order)),
        "cohort_selection": (
            "raw exposure length covers the complete horizon, then highest retained "
            "base activity, user id as deterministic tie-break"
            if args.cohort_selection == "complete_horizon"
            else "highest retained base activity, user id as deterministic tie-break"
        ),
        "cohort_selection_mode": args.cohort_selection,
        "minimum_raw_horizon_events": (
            limit if args.cohort_selection == "complete_horizon" else None
        ),
        "requested_users": args.users,
        "selected_users": int(len(selected_users)),
        "minimum_retained_base_events": args.min_base_events,
        "base_prefix": args.base_prefix,
        "window_size": args.window_size,
        "window_count": args.windows,
        "num_behaviors": 5 if args.dataset.startswith("tenrec") else 2,
        "positive_rule": (
            "click or like or follow or share"
            if args.dataset.startswith("tenrec")
            else "click_timestamp > 0"
        ),
        "ordering": (
            "stable official within-user file order"
            if args.dataset.startswith("tenrec")
            else "impression timestamp with raw ordinal tie-break"
        ),
        "rows": int(len(prepared["user_idx"])),
        "positive_rows": int(prepared["label"].sum()),
        "split_rows": split_rows,
        "split_positive_rows": split_positive,
        "retained_base_events": {
            "min": int(base_counts[selected_users].min()),
            "mean": float(base_counts[selected_users].mean()),
            "max": int(base_counts[selected_users].max()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **prepared,
        original_user_ids=selected_users.astype(np.int64),
        original_item_ids=item_order.astype(np.int64),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    metadata["output"] = str(output)
    metadata["output_bytes"] = output.stat().st_size
    save_json(metadata, metadata_output)
    print(metadata_output, flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
