from __future__ import annotations

import argparse
import gzip
import time
import zipfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hstu_kvcache.utils import save_json

TENREC_COLUMNS = [
    "user_id",
    "item_id",
    "click",
    "follow",
    "like",
    "share",
    "watching_times",
]
ZHIHU_COLUMNS = ["user_id", "item_id", "impression_timestamp", "click_timestamp"]
EXPECTED_ROWS = {
    "tenrec-qk": 493_458_970,
    "tenrec-qb": 2_442_299,
    "zhihurec": 99_978_523,
}
DEFAULT_INPUTS = {
    "tenrec-qk": "data/tenrec/Tenrec.zip",
    "tenrec-qb": "data/tenrec/Tenrec.zip",
    "zhihurec": "data/zhihurec/inter_impression.csv.gz",
}
DEFAULT_MEMBERS = {
    "tenrec-qk": "Tenrec/QK-video.csv",
    "tenrec-qb": "Tenrec/QB-video.csv",
}
DEFAULT_OUTPUTS = {
    "tenrec-qk": "results/dataset_audit/tenrec_qk.json",
    "tenrec-qb": "results/dataset_audit/tenrec_qb.json",
    "zhihurec": "results/dataset_audit/zhihurec.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DEFAULT_INPUTS), required=True)
    parser.add_argument("--input")
    parser.add_argument("--member")
    parser.add_argument("--output")
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--base-prefix", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument(
        "--catalog-sizes",
        type=int,
        nargs="+",
        default=[5000, 20000, 50000, 100000, 250000],
    )
    parser.add_argument("--primary-catalog", type=int, default=50000)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def grow_vector(array: np.ndarray, required: int) -> np.ndarray:
    if len(array) >= required:
        return array
    size = max(required, max(1024, len(array) * 2))
    output = np.zeros(size, dtype=array.dtype)
    output[: len(array)] = array
    return output


def grow_matrix(array: np.ndarray, required: int) -> np.ndarray:
    if array.shape[1] >= required:
        return array
    size = max(required, max(1024, array.shape[1] * 2))
    output = np.zeros((array.shape[0], size), dtype=array.dtype)
    output[:, : array.shape[1]] = array
    return output


def distribution(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"count": 0}
    quantiles = np.quantile(values, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "min": int(np.min(values)),
        "p25": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": int(np.max(values)),
    }


def source_metadata(dataset: str, path: Path, member: str | None) -> dict:
    output = {"path": str(path), "compressed_file_size_bytes": path.stat().st_size}
    if dataset.startswith("tenrec"):
        if member is None:
            raise ValueError("Tenrec member is required")
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo(member)
        output.update(
            {
                "member": member,
                "member_uncompressed_size_bytes": info.file_size,
                "member_compressed_size_bytes": info.compress_size,
                "member_crc32": f"{info.CRC:08x}",
            }
        )
    else:
        with open(path, "rb") as source:
            source.seek(-4, 2)
            gzip_isize = int.from_bytes(source.read(4), "little")
        output["gzip_isize_modulo_2_32"] = gzip_isize
    return output


@contextmanager
def chunk_reader(
    dataset: str,
    path: Path,
    member: str | None,
    chunk_size: int,
    max_rows: int | None,
):
    if dataset.startswith("tenrec"):
        if member is None:
            raise ValueError("Tenrec member is required")
        with zipfile.ZipFile(path) as archive, archive.open(member) as source:
            yield pd.read_csv(
                source,
                usecols=TENREC_COLUMNS,
                dtype={column: "int64" for column in TENREC_COLUMNS},
                chunksize=chunk_size,
                nrows=max_rows,
            )
    else:
        with gzip.open(path, "rb") as source:
            yield pd.read_csv(
                source,
                names=ZHIHU_COLUMNS,
                header=None,
                dtype={column: "int64" for column in ZHIHU_COLUMNS},
                chunksize=chunk_size,
                nrows=max_rows,
            )


def group_layout(
    users: np.ndarray,
    seen_counts: np.ndarray,
    previous_user: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    starts = np.concatenate(
        [np.array([0], dtype=np.int64), np.flatnonzero(users[1:] != users[:-1]) + 1]
    )
    lengths = np.diff(np.append(starts, len(users))).astype(np.int64, copy=False)
    group_users = users[starts]
    continuation = previous_user is not None and int(group_users[0]) == previous_user
    row_groups = np.repeat(np.arange(len(starts), dtype=np.int32), lengths)
    positions = np.empty(len(users), dtype=np.int64)
    reopened = 0
    if len(np.unique(group_users)) == len(group_users):
        seen = seen_counts[group_users].astype(np.int64, copy=False)
        reopened = int(np.count_nonzero(seen))
        if continuation:
            reopened -= 1
        offsets = seen - starts
        positions[:] = np.arange(len(users), dtype=np.int64) + np.repeat(
            offsets, lengths
        )
        seen_counts[group_users] += lengths.astype(seen_counts.dtype, copy=False)
    else:
        preceding = (
            pd.Series(lengths)
            .groupby(group_users, sort=False)
            .cumsum()
            .to_numpy(dtype=np.int64)
            - lengths
        )
        seen = seen_counts[group_users].astype(np.int64, copy=False) + preceding
        reopened = int(np.count_nonzero(seen))
        if continuation:
            reopened -= 1
        offsets = seen - starts
        positions[:] = np.arange(len(users), dtype=np.int64) + np.repeat(
            offsets, lengths
        )
        np.add.at(
            seen_counts,
            group_users,
            lengths.astype(seen_counts.dtype, copy=False),
        )
    return starts, lengths, group_users, row_groups, positions, reopened


def positive_mask(dataset: str, chunk: pd.DataFrame) -> np.ndarray:
    if dataset.startswith("tenrec"):
        values = np.zeros(len(chunk), dtype=np.bool_)
        for column in ("click", "follow", "like", "share"):
            values |= chunk[column].to_numpy(dtype=np.bool_, copy=False)
        return values
    return chunk["click_timestamp"].to_numpy(dtype=np.int64, copy=False) > 0


def update_group_counts(
    target: np.ndarray,
    values: np.ndarray,
    starts: np.ndarray,
    group_users: np.ndarray,
) -> None:
    counts = np.add.reduceat(values, starts)
    np.add.at(target, group_users, counts.astype(target.dtype, copy=False))


def update_group_matrix(
    target: np.ndarray,
    group_users: np.ndarray,
    values: np.ndarray,
) -> None:
    if len(np.unique(group_users)) == len(group_users):
        target[:, group_users] += values.T.astype(target.dtype, copy=False)
        return
    for index in range(target.shape[0]):
        np.add.at(
            target[index],
            group_users,
            values[:, index].astype(target.dtype, copy=False),
        )


def nested_group_counts(
    keys: np.ndarray,
    selected: np.ndarray,
    group_count: int,
    catalog_count: int,
) -> np.ndarray:
    shape = (group_count, catalog_count + 1)
    histogram = np.bincount(
        keys[selected], minlength=shape[0] * shape[1]
    ).reshape(shape)
    return np.cumsum(histogram[:, :catalog_count], axis=1)


def cohort_metrics(
    selected: np.ndarray,
    base_counts: np.ndarray,
    stream_counts: np.ndarray,
    base_positive: np.ndarray,
    stream_positive: np.ndarray,
) -> dict:
    user_ids = np.flatnonzero(selected)
    if len(user_ids) == 0:
        return {"users": 0}
    base = base_counts[user_ids].astype(np.int64, copy=False)
    stream = stream_counts[user_ids].astype(np.int64, copy=False)
    total = base + stream
    base_pos = base_positive[user_ids].astype(np.int64, copy=False)
    stream_pos = stream_positive[user_ids].astype(np.int64, copy=False)
    return {
        "users": int(len(user_ids)),
        "base_events": int(base.sum()),
        "stream_events": int(stream.sum()),
        "all_events": int(total.sum()),
        "base_positive_events": int(base_pos.sum()),
        "stream_positive_events": int(stream_pos.sum()),
        "users_with_stream_events": int(np.count_nonzero(stream)),
        "users_with_stream_positive_events": int(np.count_nonzero(stream_pos)),
        "base_events_per_user": distribution(base),
        "stream_events_per_user": distribution(stream),
        "all_events_per_user": distribution(total),
        "sequence_length_coverage": {
            str(length): int(np.count_nonzero(total >= length))
            for length in (64, 128, 256, 512)
        },
    }


def cohort_summaries(
    base_counts: np.ndarray,
    stream_counts: np.ndarray,
    base_positive: np.ndarray,
    stream_positive: np.ndarray,
) -> dict:
    thresholds = (8, 16, 32, 48, 64)
    active_ids = np.flatnonzero(base_counts > 0)
    order = active_ids[
        np.lexsort((active_ids, -base_counts[active_ids].astype(np.int64)))
    ]
    minimum = {
        str(threshold): cohort_metrics(
            base_counts >= threshold,
            base_counts,
            stream_counts,
            base_positive,
            stream_positive,
        )
        for threshold in thresholds
    }
    top = {}
    for requested in (1000, 5000, 10000, 50000):
        selected = np.zeros(len(base_counts), dtype=np.bool_)
        selected[order[: min(requested, len(order))]] = True
        top[str(requested)] = cohort_metrics(
            selected,
            base_counts,
            stream_counts,
            base_positive,
            stream_positive,
        )
    return {
        "selection_uses_base_prefix_only": True,
        "all_base_active_users": cohort_metrics(
            base_counts > 0,
            base_counts,
            stream_counts,
            base_positive,
            stream_positive,
        ),
        "minimum_retained_base_event_cohorts": minimum,
        "top_base_activity_cohorts": top,
    }


def first_pass(
    dataset: str,
    path: Path,
    member: str | None,
    chunk_size: int,
    max_rows: int | None,
    base_prefix: int,
    window_size: int,
    windows: int,
) -> dict:
    user_counts = np.zeros(0, dtype=np.int32)
    user_positive = np.zeros(0, dtype=np.int32)
    item_counts = np.zeros(0, dtype=np.int64)
    base_item_counts = np.zeros(0, dtype=np.int64)
    window_user_seen = np.zeros((windows, 0), dtype=np.bool_)
    window_positive_user_seen = np.zeros((windows, 0), dtype=np.bool_)
    seen_counts = np.zeros(0, dtype=np.int32)
    window_rows = np.zeros(windows, dtype=np.int64)
    window_positive_rows = np.zeros(windows, dtype=np.int64)
    feedback = {name: 0 for name in ("click", "follow", "like", "share")}
    feedback.update({"watching_times_zero": 0, "watching_times_positive": 0})
    rows = 0
    positive_rows = 0
    user_id_reversals = 0
    non_contiguous_user_blocks = 0
    within_user_timestamp_reversals = 0
    click_before_impression = 0
    min_timestamp: int | None = None
    max_timestamp: int | None = None
    daily_rows: dict[int, int] = {}
    daily_positive_rows: dict[int, int] = {}
    previous_user: int | None = None
    previous_timestamp: int | None = None
    started = time.perf_counter()
    with chunk_reader(dataset, path, member, chunk_size, max_rows) as reader:
        for chunk_index, chunk in enumerate(reader, start=1):
            users = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
            items = chunk["item_id"].to_numpy(dtype=np.int64, copy=False)
            if np.any(users < 0) or np.any(items < 0):
                raise ValueError("negative user or item identifier")
            user_required = int(users.max()) + 1
            item_required = int(items.max()) + 1
            user_counts = grow_vector(user_counts, user_required)
            user_positive = grow_vector(user_positive, user_required)
            seen_counts = grow_vector(seen_counts, user_required)
            window_user_seen = grow_matrix(window_user_seen, user_required)
            window_positive_user_seen = grow_matrix(
                window_positive_user_seen, user_required
            )
            item_counts = grow_vector(item_counts, item_required)
            base_item_counts = grow_vector(base_item_counts, item_required)
            starts, lengths, group_users, _, positions, reopened = group_layout(
                users, seen_counts, previous_user
            )
            non_contiguous_user_blocks += reopened
            positive = positive_mask(dataset, chunk)
            np.add.at(
                user_counts,
                group_users,
                lengths.astype(user_counts.dtype, copy=False),
            )
            update_group_counts(user_positive, positive, starts, group_users)
            item_chunk_counts = np.bincount(items)
            item_counts[: len(item_chunk_counts)] += item_chunk_counts
            base = positions < base_prefix
            if base.any():
                base_counts = np.bincount(items[base])
                base_item_counts[: len(base_counts)] += base_counts
            relative = positions - base_prefix
            window_index = relative // window_size
            for index in range(windows):
                selected = window_index == index
                window_rows[index] += int(np.count_nonzero(selected))
                window_positive_rows[index] += int(np.count_nonzero(selected & positive))
                group_selected = np.add.reduceat(selected, starts) > 0
                if group_selected.any():
                    window_user_seen[index, group_users[group_selected]] = True
                positive_selected = selected & positive
                group_positive = np.add.reduceat(positive_selected, starts) > 0
                if group_positive.any():
                    window_positive_user_seen[
                        index, group_users[group_positive]
                    ] = True
            if previous_user is not None:
                user_id_reversals += int(users[0] < previous_user)
            user_id_reversals += int(np.count_nonzero(users[1:] < users[:-1]))
            if dataset == "zhihurec":
                timestamps = chunk["impression_timestamp"].to_numpy(
                    dtype=np.int64, copy=False
                )
                clicks = chunk["click_timestamp"].to_numpy(dtype=np.int64, copy=False)
                if previous_user is not None and int(users[0]) == previous_user:
                    within_user_timestamp_reversals += int(
                        timestamps[0] < previous_timestamp
                    )
                same_user = users[1:] == users[:-1]
                within_user_timestamp_reversals += int(
                    np.count_nonzero(same_user & (timestamps[1:] < timestamps[:-1]))
                )
                click_before_impression += int(
                    np.count_nonzero((clicks > 0) & (clicks < timestamps))
                )
                chunk_min = int(timestamps.min())
                chunk_max = int(timestamps.max())
                min_timestamp = chunk_min if min_timestamp is None else min(min_timestamp, chunk_min)
                max_timestamp = chunk_max if max_timestamp is None else max(max_timestamp, chunk_max)
                days = timestamps // 86400
                for day, count in zip(*np.unique(days, return_counts=True), strict=True):
                    key = int(day)
                    daily_rows[key] = daily_rows.get(key, 0) + int(count)
                positive_days = days[positive]
                for day, count in zip(
                    *np.unique(positive_days, return_counts=True), strict=True
                ):
                    key = int(day)
                    daily_positive_rows[key] = daily_positive_rows.get(key, 0) + int(count)
                previous_timestamp = int(timestamps[-1])
            else:
                for name in ("click", "follow", "like", "share"):
                    feedback[name] += int(chunk[name].sum())
                watching = chunk["watching_times"].to_numpy(dtype=np.int64, copy=False)
                feedback["watching_times_zero"] += int(np.count_nonzero(watching == 0))
                feedback["watching_times_positive"] += int(np.count_nonzero(watching > 0))
            rows += len(chunk)
            positive_rows += int(np.count_nonzero(positive))
            previous_user = int(users[-1])
            print(
                f"pass=raw dataset={dataset} chunks={chunk_index} rows={rows:,} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    active_users = user_counts > 0
    active_items = item_counts > 0
    user_end = int(np.flatnonzero(active_users)[-1]) + 1
    item_end = int(np.flatnonzero(active_items)[-1]) + 1
    user_counts = user_counts[:user_end]
    user_positive = user_positive[:user_end]
    item_counts = item_counts[:item_end]
    base_item_counts = base_item_counts[:item_end]
    window_user_seen = window_user_seen[:, :user_end]
    window_positive_user_seen = window_positive_user_seen[:, :user_end]
    active_users = active_users[:user_end]
    active_items = active_items[:item_end]
    raw_base = np.minimum(user_counts.astype(np.int64), base_prefix)
    raw_stream = np.maximum(user_counts.astype(np.int64) - base_prefix, 0)
    calendar = None
    if dataset == "zhihurec":
        calendar = {
            "min_impression_timestamp": min_timestamp,
            "max_impression_timestamp": max_timestamp,
            "min_impression_utc": datetime.fromtimestamp(min_timestamp, tz=UTC).isoformat(),
            "max_impression_utc": datetime.fromtimestamp(max_timestamp, tz=UTC).isoformat(),
            "within_user_timestamp_reversals": within_user_timestamp_reversals,
            "clicks_before_impression": click_before_impression,
            "daily": [
                {
                    "date_utc": datetime.fromtimestamp(day * 86400, tz=UTC).date().isoformat(),
                    "rows": daily_rows[day],
                    "positive_rows": daily_positive_rows.get(day, 0),
                }
                for day in sorted(daily_rows)
            ],
        }
    return {
        "rows": rows,
        "expected_rows": EXPECTED_ROWS[dataset],
        "complete_row_count": rows == EXPECTED_ROWS[dataset] if max_rows is None else None,
        "users": int(np.count_nonzero(active_users)),
        "items": int(np.count_nonzero(active_items)),
        "max_user_id": int(np.flatnonzero(active_users)[-1]),
        "max_item_id": int(np.flatnonzero(active_items)[-1]),
        "positive_rows": positive_rows,
        "negative_rows": rows - positive_rows,
        "positive_row_fraction": positive_rows / rows,
        "feedback": feedback if dataset.startswith("tenrec") else {"click": positive_rows},
        "per_user": {
            "events": distribution(user_counts[active_users]),
            "positive_events": distribution(user_positive[active_users]),
            "sequence_length_coverage": {
                str(length): int(np.count_nonzero(user_counts >= length))
                for length in (64, 128, 256, 512, 1024)
            },
        },
        "per_item": {"exposures": distribution(item_counts[active_items])},
        "order": {
            "user_id_reversals": user_id_reversals,
            "non_contiguous_user_blocks": non_contiguous_user_blocks,
            "within_user_order_source": (
                "official per-user order without timestamps"
                if dataset.startswith("tenrec")
                else "impression_timestamp"
            ),
        },
        "raw_ordered_replay": {
            "base_prefix_events_per_user": base_prefix,
            "base_rows": int(raw_base.sum()),
            "stream_rows": int(raw_stream.sum()),
            "users_reaching_base_prefix": int(np.count_nonzero(user_counts >= base_prefix)),
            "windows": [
                {
                    "index": index,
                    "ordinal_start_inclusive": base_prefix + index * window_size,
                    "ordinal_end_exclusive": base_prefix + (index + 1) * window_size,
                    "rows": int(window_rows[index]),
                    "positive_rows": int(window_positive_rows[index]),
                    "users": int(window_user_seen[index].sum()),
                    "positive_users": int(window_positive_user_seen[index].sum()),
                }
                for index in range(windows)
            ],
        },
        "calendar": calendar,
        "arrays": {
            "user_counts": user_counts,
            "user_positive": user_positive,
            "item_counts": item_counts,
            "base_item_counts": base_item_counts,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def catalog_pass(
    dataset: str,
    path: Path,
    member: str | None,
    chunk_size: int,
    max_rows: int | None,
    base_prefix: int,
    window_size: int,
    windows: int,
    catalog_sizes: list[int],
    primary_catalog: int,
    user_capacity: int,
    item_capacity: int,
    base_item_counts: np.ndarray,
) -> dict:
    sizes = sorted(set(catalog_sizes + [primary_catalog]))
    base_items = np.flatnonzero(base_item_counts)
    item_order = base_items[
        np.argsort(-base_item_counts[base_items], kind="stable")
    ]
    ranks = np.full(item_capacity, max(sizes) + 1, dtype=np.int32)
    ranks[item_order] = np.arange(len(item_order), dtype=np.int32)
    catalog_count = len(sizes)
    base_counts = np.zeros((catalog_count, user_capacity), dtype=np.uint16)
    stream_counts = np.zeros((catalog_count, user_capacity), dtype=np.uint32)
    base_positive = np.zeros((catalog_count, user_capacity), dtype=np.uint16)
    stream_positive = np.zeros((catalog_count, user_capacity), dtype=np.uint32)
    primary_window_counts = np.zeros((windows, user_capacity), dtype=np.uint8)
    primary_window_positive = np.zeros((windows, user_capacity), dtype=np.uint8)
    seen_counts = np.zeros(user_capacity, dtype=np.int32)
    previous_user: int | None = None
    rows = 0
    started = time.perf_counter()
    with chunk_reader(dataset, path, member, chunk_size, max_rows) as reader:
        for chunk_index, chunk in enumerate(reader, start=1):
            users = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
            items = chunk["item_id"].to_numpy(dtype=np.int64, copy=False)
            starts, lengths, group_users, row_groups, positions, _ = group_layout(
                users, seen_counts, previous_user
            )
            positive = positive_mask(dataset, chunk)
            item_ranks = ranks[items]
            buckets = np.searchsorted(sizes, item_ranks, side="right")
            keys = row_groups.astype(np.int64) * (catalog_count + 1) + buckets
            in_base = positions < base_prefix
            in_stream = ~in_base
            base_values = nested_group_counts(
                keys, in_base, len(starts), catalog_count
            )
            stream_values = nested_group_counts(
                keys, in_stream, len(starts), catalog_count
            )
            base_positive_values = nested_group_counts(
                keys, in_base & positive, len(starts), catalog_count
            )
            stream_positive_values = nested_group_counts(
                keys, in_stream & positive, len(starts), catalog_count
            )
            update_group_matrix(base_counts, group_users, base_values)
            update_group_matrix(stream_counts, group_users, stream_values)
            update_group_matrix(base_positive, group_users, base_positive_values)
            update_group_matrix(stream_positive, group_users, stream_positive_values)
            primary_selected = item_ranks < primary_catalog
            relative = positions - base_prefix
            window_index = relative // window_size
            valid_window = (
                primary_selected & (window_index >= 0) & (window_index < windows)
            )
            window_keys = row_groups[valid_window].astype(np.int64) * windows + window_index[
                valid_window
            ]
            window_histogram = np.bincount(
                window_keys, minlength=len(starts) * windows
            ).reshape(len(starts), windows)
            positive_window = valid_window & positive
            positive_keys = (
                row_groups[positive_window].astype(np.int64) * windows
                + window_index[positive_window]
            )
            positive_histogram = np.bincount(
                positive_keys, minlength=len(starts) * windows
            ).reshape(len(starts), windows)
            update_group_matrix(primary_window_counts, group_users, window_histogram)
            update_group_matrix(
                primary_window_positive, group_users, positive_histogram
            )
            rows += len(chunk)
            previous_user = int(users[-1])
            print(
                f"pass=catalog dataset={dataset} chunks={chunk_index} rows={rows:,} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    raw_base_rows = int(np.minimum(seen_counts.astype(np.int64), base_prefix).sum())
    raw_stream_rows = int(np.maximum(seen_counts.astype(np.int64) - base_prefix, 0).sum())
    catalogs = {}
    for index, size in enumerate(sizes):
        base = base_counts[index]
        stream = stream_counts[index]
        base_pos = base_positive[index]
        stream_pos = stream_positive[index]
        catalogs[str(size)] = {
            "requested_items": size,
            "fitted_items": int(min(size, len(item_order))),
            "retained_base_rows": int(base.sum()),
            "retained_base_row_fraction": float(base.sum() / raw_base_rows),
            "retained_stream_rows": int(stream.sum()),
            "retained_stream_row_fraction": float(stream.sum() / raw_stream_rows)
            if raw_stream_rows
            else 0.0,
            "retained_all_rows": int(base.sum(dtype=np.uint64) + stream.sum(dtype=np.uint64)),
            "retained_all_row_fraction": float(
                (base.sum(dtype=np.uint64) + stream.sum(dtype=np.uint64)) / rows
            ),
            "retained_positive_rows": int(
                base_pos.sum(dtype=np.uint64) + stream_pos.sum(dtype=np.uint64)
            ),
            "cohorts": cohort_summaries(base, stream, base_pos, stream_pos),
        }
    primary_index = sizes.index(primary_catalog)
    plans = {}
    for threshold in (8, 16, 32, 48, 64):
        selected = base_counts[primary_index] >= threshold
        update_plan = []
        for model_t in range(1, windows):
            update_counts = primary_window_counts[model_t - 1, selected]
            update_positive = primary_window_positive[model_t - 1, selected]
            eval_counts = primary_window_counts[model_t, selected]
            eval_positive = primary_window_positive[model_t, selected]
            update_plan.append(
                {
                    "model_t": model_t,
                    "update_window": model_t - 1,
                    "eval_window": model_t,
                    "update_rows": int(update_counts.sum()),
                    "update_positive_rows": int(update_positive.sum()),
                    "update_users": int(np.count_nonzero(update_counts)),
                    "eval_rows": int(eval_counts.sum()),
                    "eval_positive_rows": int(eval_positive.sum()),
                    "eval_users": int(np.count_nonzero(eval_counts)),
                    "eval_positive_users": int(np.count_nonzero(eval_positive)),
                }
            )
        eval_presence = primary_window_positive[1:, selected] > 0
        plans[str(threshold)] = {
            "selected_users": int(np.count_nonzero(selected)),
            "users_with_positive_target_in_every_eval_window": int(
                np.count_nonzero(np.all(eval_presence, axis=0))
            ),
            "users_with_positive_target_in_any_eval_window": int(
                np.count_nonzero(np.any(eval_presence, axis=0))
            ),
            "five_update_plan": update_plan,
        }
    return {
        "base_catalog_fit_rule": f"first {base_prefix} raw exposures per user only",
        "stream_boundary_rule": f"raw per-user ordinal >= {base_prefix}",
        "catalogs": catalogs,
        "primary_catalog": primary_catalog,
        "primary_catalog_ordered_replay": {
            "window_size_raw_exposures_per_user": window_size,
            "window_count": windows,
            "cohort_plans": plans,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    if args.max_rows is not None and args.output is None:
        raise ValueError("--output is required when --max-rows is set")
    path = Path(args.input or DEFAULT_INPUTS[args.dataset])
    member = args.member or DEFAULT_MEMBERS.get(args.dataset)
    output = Path(args.output or DEFAULT_OUTPUTS[args.dataset])
    windows = args.updates + 1
    metadata = source_metadata(args.dataset, path, member)
    raw = first_pass(
        args.dataset,
        path,
        member,
        args.chunk_size,
        args.max_rows,
        args.base_prefix,
        args.window_size,
        windows,
    )
    if args.max_rows is None and not raw["complete_row_count"]:
        raise ValueError(
            f"expected {EXPECTED_ROWS[args.dataset]:,} rows, found {raw['rows']:,}"
        )
    arrays = raw.pop("arrays")
    catalogs = catalog_pass(
        args.dataset,
        path,
        member,
        args.chunk_size,
        args.max_rows,
        args.base_prefix,
        args.window_size,
        windows,
        args.catalog_sizes,
        args.primary_catalog,
        len(arrays["user_counts"]),
        len(arrays["item_counts"]),
        arrays["base_item_counts"],
    )
    result = {
        "protocol": "ordered_exposure_data_audit_v1",
        "dataset": args.dataset,
        "source": metadata,
        "scan": raw,
        "catalog_filtered": catalogs,
        "interpretation_guardrails": {
            "negative_rows_are_observed_exposures": True,
            "catalog_and_cohort_selection_use_base_prefix_only": True,
            "ordered_replay_is_not_claimed_as_calendar_time": args.dataset.startswith(
                "tenrec"
            ),
            "no_model_training_or_quality_selection_performed": True,
        },
    }
    save_json(result, output)
    print(output)


if __name__ == "__main__":
    main()
