from __future__ import annotations

import argparse
import hashlib
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from hstu_kvcache.utils import save_json

COLUMNS = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
BEHAVIORS = ("pv", "buy", "cart", "fav")
STRONG_BEHAVIORS = ("buy", "cart", "fav")
START_TIMESTAMP = 1511539200
END_TIMESTAMP = 1512316800
UTC_FIRST_DATE_TIMESTAMP = 1511568000
DAY_SECONDS = 86400
EXPECTED_ROWS = 100150807
EXPECTED_FILE_SIZE = 3672347465
EXPECTED_SHA256 = "46fdd7d389c1ddc7922eb7d9014af5573a4a3045da28c6c46197636873d8f1a9"
COHORT_THRESHOLDS = (20, 50, 100, 128, 200, 256, 512)
COHORT_SIZES = (1000, 5000, 10000, 50000)
COHORT_CATALOG_SIZES = (50000, 100000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/taobao/UserBehavior.csv")
    parser.add_argument("--output", default="results/taobao/data_audit.json")
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--base-days", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def grow_vector(array: np.ndarray, required: int, fill: int | bool = 0) -> np.ndarray:
    if len(array) >= required:
        return array
    size = max(required, max(1024, len(array) * 2))
    output = np.full(size, fill, dtype=array.dtype)
    output[: len(array)] = array
    return output


def grow_matrix(array: np.ndarray, required: int, fill: int | bool = 0) -> np.ndarray:
    if array.shape[1] >= required:
        return array
    size = max(required, max(1024, array.shape[1] * 2))
    output = np.full((array.shape[0], size), fill, dtype=array.dtype)
    output[:, : array.shape[1]] = array
    return output


def add_counts(target: np.ndarray, values: np.ndarray) -> None:
    if len(values) == 0:
        return
    counts = np.bincount(values)
    target[: len(counts)] += counts.astype(target.dtype, copy=False)


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


def shanghai_date_labels(count: int) -> list[str]:
    start = datetime.fromtimestamp(START_TIMESTAMP, tz=ZoneInfo("Asia/Shanghai")).date()
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def utc_date_labels(count: int) -> list[str]:
    start = datetime.fromtimestamp(START_TIMESTAMP, tz=UTC).date()
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def catalog_summary(
    base_counts: np.ndarray,
    total_counts: np.ndarray,
    total_rows: int,
    max_items_values: list[int],
) -> dict:
    base_rows = int(base_counts.sum())
    stream_rows = total_rows - base_rows
    item_ids = np.flatnonzero(base_counts)
    order = item_ids[np.argsort(base_counts[item_ids], kind="stable")[::-1]]
    output = {}
    for requested in max_items_values:
        selected = order[: min(requested, len(order))]
        selected_base = int(base_counts[selected].sum())
        selected_total = int(total_counts[selected].sum())
        selected_stream = selected_total - selected_base
        output[str(requested)] = {
            "fitted_items": int(len(selected)),
            "retained_base_rows": selected_base,
            "retained_base_row_fraction": selected_base / base_rows if base_rows else 0.0,
            "retained_stream_rows": selected_stream,
            "retained_stream_row_fraction": selected_stream / stream_rows if stream_rows else 0.0,
            "retained_all_rows": selected_total,
            "retained_all_row_fraction": selected_total / total_rows if total_rows else 0.0,
        }
    selected_base = int(base_counts[order].sum())
    selected_total = int(total_counts[order].sum())
    output["all_base_items"] = {
        "fitted_items": int(len(order)),
        "retained_base_rows": selected_base,
        "retained_base_row_fraction": selected_base / base_rows if base_rows else 0.0,
        "retained_stream_rows": selected_total - selected_base,
        "retained_stream_row_fraction": (
            (selected_total - selected_base) / stream_rows if stream_rows else 0.0
        ),
        "retained_all_rows": selected_total,
        "retained_all_row_fraction": selected_total / total_rows if total_rows else 0.0,
    }
    return output


def cohort_slice_summary(
    selected: np.ndarray,
    base_counts: np.ndarray,
    total_counts: np.ndarray,
    day_counts: np.ndarray,
    day_strong: np.ndarray,
    base_days: int,
    labels: list[str],
) -> dict:
    user_ids = np.flatnonzero(selected)
    if len(user_ids) == 0:
        return {"users": 0}
    selected_base = base_counts[user_ids]
    selected_total = total_counts[user_ids]
    selected_stream = selected_total - selected_base
    selected_day_counts = day_counts[:, user_ids]
    active_by_day = selected_day_counts > 0
    return {
        "users": int(len(user_ids)),
        "base_events": int(selected_base.sum()),
        "stream_events": int(selected_stream.sum()),
        "all_events": int(selected_total.sum()),
        "base_events_per_user": distribution(selected_base),
        "stream_events_per_user": distribution(selected_stream),
        "all_events_per_user": distribution(selected_total),
        "active_days_per_user": distribution(np.count_nonzero(active_by_day, axis=0)),
        "sequence_length_coverage": {
            str(length): {
                "base_users": int(np.count_nonzero(selected_base >= length)),
                "all_days_users": int(np.count_nonzero(selected_total >= length)),
            }
            for length in (128, 256, 512)
        },
        "daily": [
            {
                "date_asia_shanghai": label,
                "events": int(selected_day_counts[day].sum()),
                "active_users": int(active_by_day[day].sum()),
            }
            for day, label in enumerate(labels)
        ],
        "users_active_on_every_post_base_day": int(
            np.count_nonzero(np.all(active_by_day[base_days:], axis=0))
        ),
        "final_day_users_with_any_behavior": int(active_by_day[-1].sum()),
        "final_day_users_with_strong_feedback": int(
            day_strong[-1, user_ids].sum()
        ),
    }


def cohort_summary(
    total_counts: np.ndarray,
    day_counts: np.ndarray,
    day_strong: np.ndarray,
    base_days: int,
    labels: list[str],
) -> dict:
    base_counts = day_counts[:base_days].sum(axis=0, dtype=np.int64)
    base_active = base_counts > 0
    user_ids = np.flatnonzero(base_active)
    order = user_ids[
        np.lexsort((user_ids, -base_counts[user_ids]))
    ]
    threshold_cohorts = {}
    for threshold in COHORT_THRESHOLDS:
        selected = base_counts >= threshold
        threshold_cohorts[str(threshold)] = cohort_slice_summary(
            selected,
            base_counts,
            total_counts,
            day_counts,
            day_strong,
            base_days,
            labels,
        )
    top_cohorts = {}
    for requested in COHORT_SIZES:
        selected = np.zeros(len(base_counts), dtype=np.bool_)
        selected[order[: min(requested, len(order))]] = True
        top_cohorts[str(requested)] = cohort_slice_summary(
            selected,
            base_counts,
            total_counts,
            day_counts,
            day_strong,
            base_days,
            labels,
        )
    return {
        "selection_uses_base_window_only": True,
        "all_base_active_users": cohort_slice_summary(
            base_active,
            base_counts,
            total_counts,
            day_counts,
            day_strong,
            base_days,
            labels,
        ),
        "minimum_base_event_cohorts": threshold_cohorts,
        "top_base_activity_cohorts": top_cohorts,
    }


def catalog_user_audit(
    path: Path,
    chunk_size: int,
    max_rows: int | None,
    days: int,
    base_days: int,
    item_base_counts: np.ndarray,
    user_capacity: int,
    labels: list[str],
) -> dict:
    item_ids = np.flatnonzero(item_base_counts)
    order = item_ids[np.argsort(item_base_counts[item_ids], kind="stable")[::-1]]
    lookups = {}
    user_counts = {}
    user_day_counts = {}
    user_day_strong = {}
    for requested in COHORT_CATALOG_SIZES:
        lookup = np.zeros(len(item_base_counts), dtype=np.bool_)
        lookup[order[: min(requested, len(order))]] = True
        lookups[requested] = lookup
        user_counts[requested] = np.zeros(user_capacity, dtype=np.int32)
        user_day_counts[requested] = np.zeros((days, user_capacity), dtype=np.int32)
        user_day_strong[requested] = np.zeros((days, user_capacity), dtype=np.bool_)
    reader = pd.read_csv(
        path,
        names=COLUMNS,
        header=None,
        usecols=["user_id", "item_id", "behavior_type", "timestamp"],
        dtype={
            "user_id": "int32",
            "item_id": "int32",
            "behavior_type": "category",
            "timestamp": "int64",
        },
        chunksize=chunk_size,
        nrows=max_rows,
        memory_map=True,
    )
    scanned = 0
    started = time.perf_counter()
    for chunk_index, chunk in enumerate(reader, start=1):
        scanned += len(chunk)
        users = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
        items = chunk["item_id"].to_numpy(dtype=np.int64, copy=False)
        timestamps = chunk["timestamp"].to_numpy(dtype=np.int64, copy=False)
        behaviors = chunk["behavior_type"].astype("string").to_numpy()
        in_range = (timestamps >= START_TIMESTAMP) & (timestamps < END_TIMESTAMP)
        known = np.isin(behaviors, BEHAVIORS)
        usable = in_range & known
        day_index = ((timestamps - START_TIMESTAMP) // DAY_SECONDS).astype(np.int64)
        strong = np.isin(behaviors, STRONG_BEHAVIORS)
        for requested, lookup in lookups.items():
            selected = usable.copy()
            selected[usable] = lookup[items[usable]]
            if not selected.any():
                continue
            selected_users = users[selected]
            selected_days = day_index[selected]
            selected_strong = strong[selected]
            add_counts(user_counts[requested], selected_users)
            for day in range(days):
                on_day = selected_days == day
                if not on_day.any():
                    continue
                add_counts(user_day_counts[requested][day], selected_users[on_day])
                strong_users = np.unique(selected_users[on_day & selected_strong])
                user_day_strong[requested][day, strong_users] = True
        print(
            f"catalog-pass chunks={chunk_index} rows={scanned:,} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return {
        str(requested): {
            "fitted_items": int(min(requested, len(order))),
            "cohorts": cohort_summary(
                user_counts[requested],
                user_day_counts[requested],
                user_day_strong[requested],
                base_days,
                labels,
            ),
        }
        for requested in COHORT_CATALOG_SIZES
    }


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    started = time.perf_counter()
    source_sha256 = sha256(path)
    days = (END_TIMESTAMP - START_TIMESTAMP) // DAY_SECONDS
    user_counts = np.zeros(0, dtype=np.int32)
    user_strong_counts = np.zeros(0, dtype=np.int32)
    user_day_counts = np.zeros((days, 0), dtype=np.int32)
    user_day_strong = np.zeros((days, 0), dtype=np.bool_)
    item_counts = np.zeros(0, dtype=np.int64)
    item_base_counts = {
        base_days: np.zeros(0, dtype=np.int64) for base_days in args.base_days
    }
    category_seen = np.zeros(0, dtype=np.bool_)
    behavior_counts = np.zeros(len(BEHAVIORS), dtype=np.int64)
    behavior_day_counts = np.zeros((days, len(BEHAVIORS)), dtype=np.int64)
    shanghai_day_counts = np.zeros(days, dtype=np.int64)
    utc_calendar_day_counts = np.zeros(days + 1, dtype=np.int64)
    rows = 0
    usable_rows = 0
    rows_before_range = 0
    rows_after_range = 0
    timezone_shifted_rows = 0
    min_timestamp: int | None = None
    max_timestamp: int | None = None
    invalid_behavior_counts: dict[str, int] = {}
    invalid_time_samples = []
    user_order_reversals = 0
    within_user_time_reversals = 0
    previous_user: int | None = None
    previous_timestamp: int | None = None
    dtypes = {
        "user_id": "int32",
        "item_id": "int32",
        "category_id": "int32",
        "behavior_type": "category",
        "timestamp": "int64",
    }
    reader = pd.read_csv(
        path,
        names=COLUMNS,
        header=None,
        dtype=dtypes,
        chunksize=args.chunk_size,
        nrows=args.max_rows,
        memory_map=True,
    )
    for chunk_index, chunk in enumerate(reader, start=1):
        rows += len(chunk)
        users_raw = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
        items_raw = chunk["item_id"].to_numpy(dtype=np.int64, copy=False)
        categories_raw = chunk["category_id"].to_numpy(dtype=np.int64, copy=False)
        timestamps_raw = chunk["timestamp"].to_numpy(dtype=np.int64, copy=False)
        behaviors_raw = chunk["behavior_type"].astype("string").to_numpy()
        if np.any(users_raw < 0) or np.any(items_raw < 0) or np.any(categories_raw < 0):
            raise ValueError("negative identifier in source data")
        chunk_min = int(timestamps_raw.min())
        chunk_max = int(timestamps_raw.max())
        min_timestamp = chunk_min if min_timestamp is None else min(min_timestamp, chunk_min)
        max_timestamp = chunk_max if max_timestamp is None else max(max_timestamp, chunk_max)
        if previous_user is not None:
            user_order_reversals += int(users_raw[0] < previous_user)
            within_user_time_reversals += int(
                users_raw[0] == previous_user and timestamps_raw[0] < previous_timestamp
            )
        user_order_reversals += int(np.count_nonzero(users_raw[1:] < users_raw[:-1]))
        same_user = users_raw[1:] == users_raw[:-1]
        within_user_time_reversals += int(
            np.count_nonzero(same_user & (timestamps_raw[1:] < timestamps_raw[:-1]))
        )
        previous_user = int(users_raw[-1])
        previous_timestamp = int(timestamps_raw[-1])
        known_behavior = np.zeros(len(chunk), dtype=np.bool_)
        for behavior in BEHAVIORS:
            known_behavior |= behaviors_raw == behavior
        unknown = behaviors_raw[~known_behavior]
        if len(unknown):
            values, counts = np.unique(unknown, return_counts=True)
            for value, count in zip(values, counts, strict=True):
                key = str(value)
                invalid_behavior_counts[key] = invalid_behavior_counts.get(key, 0) + int(count)
        before = timestamps_raw < START_TIMESTAMP
        after = timestamps_raw >= END_TIMESTAMP
        rows_before_range += int(before.sum())
        rows_after_range += int(after.sum())
        invalid_time = before | after
        if invalid_time.any() and len(invalid_time_samples) < 10:
            sample = chunk.loc[invalid_time, COLUMNS].head(10 - len(invalid_time_samples))
            invalid_time_samples.extend(sample.to_dict(orient="records"))
        usable = (~invalid_time) & known_behavior
        if not usable.any():
            continue
        users = users_raw[usable]
        items = items_raw[usable]
        categories = categories_raw[usable]
        timestamps = timestamps_raw[usable]
        behaviors = behaviors_raw[usable]
        day_index = ((timestamps - START_TIMESTAMP) // DAY_SECONDS).astype(np.int8)
        usable_rows += len(users)
        user_required = int(users.max()) + 1
        item_required = int(items.max()) + 1
        category_required = int(categories.max()) + 1
        user_counts = grow_vector(user_counts, user_required)
        user_strong_counts = grow_vector(user_strong_counts, user_required)
        user_day_counts = grow_matrix(user_day_counts, user_required)
        user_day_strong = grow_matrix(user_day_strong, user_required)
        item_counts = grow_vector(item_counts, item_required)
        category_seen = grow_vector(category_seen, category_required)
        for base_days in item_base_counts:
            item_base_counts[base_days] = grow_vector(
                item_base_counts[base_days], item_required
            )
        add_counts(user_counts, users)
        add_counts(item_counts, items)
        category_seen[np.unique(categories)] = True
        strong = np.isin(behaviors, STRONG_BEHAVIORS)
        add_counts(user_strong_counts, users[strong])
        for day in range(days):
            on_day = day_index == day
            if not on_day.any():
                continue
            shanghai_day_counts[day] += int(on_day.sum())
            add_counts(user_day_counts[day], users[on_day])
            strong_users = np.unique(users[on_day & strong])
            user_day_strong[day, strong_users] = True
            for behavior_index, behavior in enumerate(BEHAVIORS):
                count = int(np.count_nonzero(on_day & (behaviors == behavior)))
                behavior_counts[behavior_index] += count
                behavior_day_counts[day, behavior_index] += count
        for base_days, counts in item_base_counts.items():
            add_counts(counts, items[day_index < base_days])
        utc_calendar_day_index = (
            (timestamps - UTC_FIRST_DATE_TIMESTAMP) // DAY_SECONDS
        ).astype(np.int8)
        timezone_shifted_rows += int(np.count_nonzero(utc_calendar_day_index != day_index))
        utc_counts = np.bincount(utc_calendar_day_index + 1, minlength=days + 1)
        utc_calendar_day_counts += utc_counts[: days + 1]
        print(
            f"chunks={chunk_index} rows={rows:,} usable={usable_rows:,} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    active_users = user_counts > 0
    active_user_counts = user_counts[active_users]
    strong_user_counts = user_strong_counts[active_users]
    active_days = np.count_nonzero(user_day_counts[:, active_users], axis=0)
    labels = shanghai_date_labels(days)
    daily = []
    cumulative_history = np.zeros(user_day_counts.shape[1], dtype=np.bool_)
    for day, label in enumerate(labels):
        active = user_day_counts[day] > 0
        strong_active = user_day_strong[day]
        eligible = cumulative_history & active
        strong_eligible = cumulative_history & strong_active
        daily.append(
            {
                "date_asia_shanghai": label,
                "rows": int(shanghai_day_counts[day]),
                "users": int(active.sum()),
                "strong_feedback_users": int(strong_active.sum()),
                "users_with_prior_history": int(eligible.sum()),
                "strong_feedback_users_with_prior_history": int(strong_eligible.sum()),
                "behaviors": {
                    behavior: int(behavior_day_counts[day, index])
                    for index, behavior in enumerate(BEHAVIORS)
                },
            }
        )
        cumulative_history |= active
    overlaps = []
    for day in range(1, days):
        previous = user_day_counts[day - 1] > 0
        current = user_day_counts[day] > 0
        intersection = int(np.count_nonzero(previous & current))
        union = int(np.count_nonzero(previous | current))
        overlaps.append(
            {
                "from": labels[day - 1],
                "to": labels[day],
                "shared_users": intersection,
                "previous_user_retention": intersection / int(previous.sum()),
                "current_users_with_previous_day": intersection / int(current.sum()),
                "jaccard": intersection / union,
            }
        )
    base_protocols = {}
    for base_days in args.base_days:
        if base_days <= 0 or base_days >= days:
            raise ValueError("base-days must be between 1 and 8")
        base_user_counts = user_day_counts[:base_days].sum(axis=0, dtype=np.int64)
        base_active = base_user_counts > 0
        evaluation_days = []
        history = base_active.copy()
        for day in range(base_days, days):
            active = user_day_counts[day] > 0
            strong_active = user_day_strong[day]
            evaluation_days.append(
                {
                    "date_asia_shanghai": labels[day],
                    "users_with_history_and_any_behavior": int(np.count_nonzero(history & active)),
                    "users_with_history_and_strong_feedback": int(
                        np.count_nonzero(history & strong_active)
                    ),
                }
            )
            history |= active
        base_protocols[str(base_days)] = {
            "base_dates_asia_shanghai": labels[:base_days],
            "update_or_eval_dates_asia_shanghai": labels[base_days:],
            "base_rows": int(base_user_counts.sum()),
            "base_users": int(base_active.sum()),
            "base_events_per_user": distribution(base_user_counts[base_active]),
            "raw_cohorts": cohort_summary(
                user_counts,
                user_day_counts,
                user_day_strong,
                base_days,
                labels,
            ),
            "evaluation_eligibility": evaluation_days,
            "catalogs": catalog_summary(
                item_base_counts[base_days],
                item_counts,
                usable_rows,
                [5000, 20000, 50000, 100000, 250000],
            ),
        }
    cohort_catalog_base_days = min(args.base_days)
    filtered_cohorts = catalog_user_audit(
        path,
        args.chunk_size,
        args.max_rows,
        days,
        cohort_catalog_base_days,
        item_base_counts[cohort_catalog_base_days],
        len(user_counts),
        labels,
    )
    result = {
        "protocol": "taobao_userbehavior_data_audit_v1",
        "source": {
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "expected_file_size_bytes": EXPECTED_FILE_SIZE,
            "sha256": source_sha256,
            "expected_sha256": EXPECTED_SHA256,
            "sha256_verified": source_sha256 == EXPECTED_SHA256,
        },
        "scan": {
            "rows": rows,
            "expected_rows": EXPECTED_ROWS,
            "complete_row_count": rows == EXPECTED_ROWS if args.max_rows is None else None,
            "usable_rows_in_official_range": usable_rows,
            "rows_before_official_range": rows_before_range,
            "rows_after_official_range": rows_after_range,
            "invalid_behavior_counts": invalid_behavior_counts,
            "min_timestamp": min_timestamp,
            "max_timestamp": max_timestamp,
            "user_order_reversals": user_order_reversals,
            "within_user_time_reversals": within_user_time_reversals,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "official_range": {
            "start_timestamp_inclusive": START_TIMESTAMP,
            "end_timestamp_exclusive": END_TIMESTAMP,
            "calendar_timezone": "Asia/Shanghai",
            "dates": labels,
            "invalid_time_samples": invalid_time_samples,
        },
        "cardinality": {
            "users": int(active_users.sum()),
            "items": int(np.count_nonzero(item_counts)),
            "categories": int(category_seen.sum()),
        },
        "behaviors": {
            behavior: int(behavior_counts[index])
            for index, behavior in enumerate(BEHAVIORS)
        },
        "per_user": {
            "events": distribution(active_user_counts),
            "strong_feedback_events": distribution(strong_user_counts),
            "active_days": distribution(active_days),
        },
        "daily_asia_shanghai": daily,
        "consecutive_day_user_overlap": overlaps,
        "timezone_sensitivity": {
            "rows_whose_utc_and_asia_shanghai_dates_differ": timezone_shifted_rows,
            "row_fraction": timezone_shifted_rows / usable_rows,
            "utc_daily_rows_from_november_24": {
                label: int(count)
                for label, count in zip(
                    utc_date_labels(days + 1), utc_calendar_day_counts, strict=True
                )
            },
        },
        "base_protocol_candidates": base_protocols,
        "catalog_filtered_cohorts": {
            "base_days": cohort_catalog_base_days,
            "base_dates_asia_shanghai": labels[:cohort_catalog_base_days],
            "catalogs": filtered_cohorts,
        },
    }
    save_json(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
