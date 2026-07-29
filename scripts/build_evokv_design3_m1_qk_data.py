from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hstu_kvcache.utils import save_json

PROTOCOL = "evokv_design3_m1_qk_data_development_v1"
DEFAULT_SOURCE = Path("data/tenrec/Tenrec.zip")
DEFAULT_MEMBER = "Tenrec/QK-video.csv"
DEFAULT_CACHE_DIR = Path("data/processed/evokv_d3_m1_qk_cache")
DEFAULT_OUTPUT = Path("data/processed/evokv_d3_m1_qk_8704.npz")
DEFAULT_MANIFEST = Path("configs/evokv_d3/m1/qk_m1_manifest.json")
DEFAULT_COHORT_IDS = Path("configs/evokv_d3/m1/qk_m1_cohorts.json")
DEFAULT_HASH_SALT = "evokv-d3-qk-m1-v1"
TENREC_COLUMNS = ("user_id", "item_id", "click", "follow", "like", "share")


@dataclass(frozen=True)
class BuildConfig:
    source: Path = DEFAULT_SOURCE
    member: str = DEFAULT_MEMBER
    cache_dir: Path = DEFAULT_CACHE_DIR
    output: Path = DEFAULT_OUTPUT
    manifest: Path = DEFAULT_MANIFEST
    cohort_ids: Path = DEFAULT_COHORT_IDS
    catalog_size: int = 250_000
    base_prefix: int = 64
    history_length: int = 512
    slide: int = 32
    fit_calibration_users: int = 512
    cohort_sizes: tuple[int, ...] = (2_048, 4_096, 8_192)
    hash_salt: str = DEFAULT_HASH_SALT
    chunk_size: int = 2_000_000
    refresh: bool = False

    @property
    def required_events(self) -> int:
        return self.history_length + self.slide

    @property
    def benchmark_users(self) -> int:
        return max(self.cohort_sizes)

    @property
    def selected_users(self) -> int:
        return self.fit_calibration_users + self.benchmark_users

    @property
    def catalog_cache(self) -> Path:
        return self.cache_dir / (
            f"catalog_top{self.catalog_size}_base{self.base_prefix}.npz"
        )

    @property
    def cohort_cache(self) -> Path:
        return self.cache_dir / (
            f"cohort_top{self.catalog_size}_events{self.required_events}_"
            f"fit{self.fit_calibration_users}_bench{self.benchmark_users}.npz"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--member", default=DEFAULT_MEMBER)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cohort-ids", type=Path, default=DEFAULT_COHORT_IDS)
    parser.add_argument("--catalog-size", type=int, default=250_000)
    parser.add_argument("--base-prefix", type=int, default=64)
    parser.add_argument("--history-length", type=int, default=512)
    parser.add_argument("--slide", type=int, default=32)
    parser.add_argument("--fit-calibration-users", type=int, default=512)
    parser.add_argument(
        "--cohort-sizes",
        type=int,
        nargs="+",
        default=[2_048, 4_096, 8_192],
    )
    parser.add_argument("--hash-salt", default=DEFAULT_HASH_SALT)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> BuildConfig:
    return BuildConfig(
        source=args.source,
        member=args.member,
        cache_dir=args.cache_dir,
        output=args.output,
        manifest=args.manifest,
        cohort_ids=args.cohort_ids,
        catalog_size=args.catalog_size,
        base_prefix=args.base_prefix,
        history_length=args.history_length,
        slide=args.slide,
        fit_calibration_users=args.fit_calibration_users,
        cohort_sizes=tuple(sorted(set(args.cohort_sizes))),
        hash_salt=args.hash_salt,
        chunk_size=args.chunk_size,
        refresh=args.refresh,
    )


def validate_config(config: BuildConfig) -> None:
    if config.catalog_size < 1:
        raise ValueError("catalog size must be positive")
    if config.base_prefix < 1:
        raise ValueError("base prefix must be positive")
    if config.history_length < 1 or config.slide < 1:
        raise ValueError("history length and slide must be positive")
    if config.fit_calibration_users < 0:
        raise ValueError("fit/calibration user count cannot be negative")
    if not config.cohort_sizes or min(config.cohort_sizes) < 1:
        raise ValueError("cohort sizes must be positive")
    if tuple(sorted(set(config.cohort_sizes))) != config.cohort_sizes:
        raise ValueError("cohort sizes must be sorted and unique")
    if config.chunk_size < 1:
        raise ValueError("chunk size must be positive")


def source_fingerprint(config: BuildConfig) -> dict:
    with zipfile.ZipFile(config.source) as archive:
        info = archive.getinfo(config.member)
    return {
        "path": str(config.source.resolve()),
        "archive_size_bytes": config.source.stat().st_size,
        "member": config.member,
        "member_size_bytes": info.file_size,
        "member_compressed_size_bytes": info.compress_size,
        "member_crc32": f"{info.CRC:08x}",
    }


def array_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def grow_vector(values: np.ndarray, required: int) -> np.ndarray:
    if len(values) >= required:
        return values
    size = max(required, max(1_024, len(values) * 2))
    output = np.zeros(size, dtype=values.dtype)
    output[: len(values)] = values
    return output


def consume_user_positions(
    users: np.ndarray,
    seen_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(users) == 0:
        return np.empty(0, dtype=np.int64), seen_counts
    seen_counts = grow_vector(seen_counts, int(users.max()) + 1)
    starts = np.concatenate(
        [np.array([0], dtype=np.int64), np.flatnonzero(users[1:] != users[:-1]) + 1]
    )
    lengths = np.diff(np.append(starts, len(users))).astype(np.int64, copy=False)
    group_users = users[starts].astype(np.int64, copy=False)
    if len(np.unique(group_users)) == len(group_users):
        preceding = np.zeros(len(group_users), dtype=np.int64)
    else:
        preceding = (
            pd.Series(lengths)
            .groupby(group_users, sort=False)
            .cumsum()
            .to_numpy(dtype=np.int64)
            - lengths
        )
    offsets = seen_counts[group_users].astype(np.int64, copy=False) + preceding - starts
    positions = np.arange(len(users), dtype=np.int64) + np.repeat(offsets, lengths)
    if len(np.unique(group_users)) == len(group_users):
        seen_counts[group_users] += lengths.astype(seen_counts.dtype, copy=False)
    else:
        np.add.at(
            seen_counts,
            group_users,
            lengths.astype(seen_counts.dtype, copy=False),
        )
    return positions, seen_counts


def read_chunks(
    config: BuildConfig,
    columns: tuple[str, ...],
):
    dtype = {
        "user_id": "int32",
        "item_id": "int32",
        "click": "int8",
        "follow": "int8",
        "like": "int8",
        "share": "int8",
    }
    with zipfile.ZipFile(config.source) as archive, archive.open(config.member) as source:
        yield from pd.read_csv(
            source,
            usecols=list(columns),
            dtype={name: dtype[name] for name in columns},
            chunksize=config.chunk_size,
        )


def load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {
            name: source[name].copy()
            for name in source.files
            if name != "metadata_json"
        }
        metadata = json.loads(str(source["metadata_json"].item()))
    return arrays, metadata


def cache_matches(metadata: dict, expected: dict) -> bool:
    return all(metadata.get(key) == value for key, value in expected.items())


def save_npz(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )


def catalog_cache_key(config: BuildConfig, fingerprint: dict) -> dict:
    return {
        "protocol": PROTOCOL,
        "phase": "catalog",
        "source": fingerprint,
        "base_prefix_raw_events": config.base_prefix,
        "catalog_size": config.catalog_size,
    }


def build_catalog_cache(
    config: BuildConfig,
    fingerprint: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    seen_counts = np.zeros(0, dtype=np.int32)
    base_item_counts = np.zeros(0, dtype=np.int64)
    rows = 0
    started = time.perf_counter()
    for chunk_index, chunk in enumerate(
        read_chunks(config, ("user_id", "item_id")),
        start=1,
    ):
        users = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
        items = chunk["item_id"].to_numpy(dtype=np.int64, copy=False)
        positions, seen_counts = consume_user_positions(users, seen_counts)
        base_selected = positions < config.base_prefix
        if base_selected.any():
            counts = np.bincount(items[base_selected])
            base_item_counts = grow_vector(base_item_counts, len(counts))
            base_item_counts[: len(counts)] += counts
        rows += len(chunk)
        print(
            f"phase=catalog chunks={chunk_index} rows={rows:,} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    active_items = np.flatnonzero(base_item_counts)
    order = active_items[
        np.argsort(-base_item_counts[active_items], kind="stable")
    ][: config.catalog_size]
    if len(order) < config.catalog_size:
        raise ValueError(
            f"requested {config.catalog_size} items but only {len(order)} occur in the base prefix"
        )
    arrays = {
        "original_item_ids": order.astype(np.int64),
        "base_item_frequencies": base_item_counts[order].astype(np.int64),
    }
    metadata = {
        **catalog_cache_key(config, fingerprint),
        "source_rows_scanned": rows,
        "fitted_items": int(len(order)),
        "catalog_item_ids_sha256": array_sha256(order),
        "ordering": "frequency descending, original item id ascending for ties",
    }
    save_npz(config.catalog_cache, arrays, metadata)
    return arrays, metadata


def load_or_build_catalog(
    config: BuildConfig,
    fingerprint: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    expected = catalog_cache_key(config, fingerprint)
    if config.catalog_cache.exists() and not config.refresh:
        arrays, metadata = load_npz(config.catalog_cache)
        if cache_matches(metadata, expected):
            return arrays, metadata
    return build_catalog_cache(config, fingerprint)


def stable_user_order(user_ids: np.ndarray, salt: str) -> np.ndarray:
    ordered = sorted(
        (int(user_id) for user_id in user_ids),
        key=lambda user_id: (
            hashlib.sha256(f"{salt}:{user_id}".encode()).digest(),
            user_id,
        ),
    )
    return np.asarray(ordered, dtype=np.int64)


def cohort_cache_key(
    config: BuildConfig,
    fingerprint: dict,
    catalog_metadata: dict,
) -> dict:
    return {
        "protocol": PROTOCOL,
        "phase": "cohort",
        "source": fingerprint,
        "catalog_item_ids_sha256": catalog_metadata["catalog_item_ids_sha256"],
        "required_filtered_events": config.required_events,
        "fit_calibration_users": config.fit_calibration_users,
        "cohort_sizes": list(config.cohort_sizes),
        "hash_salt": config.hash_salt,
    }


def build_cohort_cache(
    config: BuildConfig,
    fingerprint: dict,
    catalog_arrays: dict[str, np.ndarray],
    catalog_metadata: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    item_ids = catalog_arrays["original_item_ids"]
    keep_items = np.zeros(int(item_ids.max()) + 1, dtype=np.bool_)
    keep_items[item_ids] = True
    filtered_counts = np.zeros(0, dtype=np.int32)
    retained_rows = 0
    rows = 0
    started = time.perf_counter()
    for chunk_index, chunk in enumerate(
        read_chunks(config, ("user_id", "item_id")),
        start=1,
    ):
        users = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
        items = chunk["item_id"].to_numpy(dtype=np.int64, copy=False)
        in_range = items < len(keep_items)
        selected = np.zeros(len(items), dtype=np.bool_)
        selected[in_range] = keep_items[items[in_range]]
        if selected.any():
            counts = np.bincount(users[selected])
            filtered_counts = grow_vector(filtered_counts, len(counts))
            filtered_counts[: len(counts)] += counts.astype(
                filtered_counts.dtype,
                copy=False,
            )
        retained_rows += int(np.count_nonzero(selected))
        rows += len(chunk)
        print(
            f"phase=cohort chunks={chunk_index} rows={rows:,} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    eligible = np.flatnonzero(filtered_counts >= config.required_events)
    ordered = stable_user_order(eligible, config.hash_salt)
    if len(ordered) < config.selected_users:
        raise ValueError(
            f"need {config.selected_users} eligible users but found {len(ordered)}"
        )
    fit_calibration = ordered[: config.fit_calibration_users]
    benchmark = ordered[
        config.fit_calibration_users : config.fit_calibration_users
        + config.benchmark_users
    ]
    selected = np.concatenate([fit_calibration, benchmark])
    arrays = {
        "eligible_user_ids": eligible.astype(np.int64),
        "eligible_filtered_lengths": filtered_counts[eligible].astype(np.int32),
        "fit_calibration_user_ids": fit_calibration,
        "benchmark_user_ids": benchmark,
        "selected_user_ids": selected,
    }
    metadata = {
        **cohort_cache_key(config, fingerprint, catalog_metadata),
        "source_rows_scanned": rows,
        "retained_rows": retained_rows,
        "eligible_users": int(len(eligible)),
        "selected_users": int(len(selected)),
        "fit_calibration_user_ids_sha256": array_sha256(fit_calibration),
        "benchmark_user_ids_sha256": array_sha256(benchmark),
        "selected_user_ids_sha256": array_sha256(selected),
        "selection": (
            "filter by retained length, sort by SHA-256 of "
            f"{config.hash_salt}:<original_user_id>, reserve fit/calibration, "
            "then take one benchmark pool whose prefixes form nested cohorts"
        ),
    }
    save_npz(config.cohort_cache, arrays, metadata)
    return arrays, metadata


def load_or_build_cohort(
    config: BuildConfig,
    fingerprint: dict,
    catalog_arrays: dict[str, np.ndarray],
    catalog_metadata: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    expected = cohort_cache_key(config, fingerprint, catalog_metadata)
    if config.cohort_cache.exists() and not config.refresh:
        arrays, metadata = load_npz(config.cohort_cache)
        if cache_matches(metadata, expected):
            return arrays, metadata
    return build_cohort_cache(
        config,
        fingerprint,
        catalog_arrays,
        catalog_metadata,
    )


def behavior_values(chunk: pd.DataFrame, selected: np.ndarray) -> np.ndarray:
    output = np.ones(int(np.count_nonzero(selected)), dtype=np.int8)
    for column, value in (("click", 2), ("like", 3), ("follow", 4), ("share", 5)):
        active = chunk[column].to_numpy(dtype=np.bool_, copy=False)[selected]
        output[active] = value
    return output


def positive_values(chunk: pd.DataFrame, selected: np.ndarray) -> np.ndarray:
    output = np.zeros(int(np.count_nonzero(selected)), dtype=np.bool_)
    for column in ("click", "follow", "like", "share"):
        output |= chunk[column].to_numpy(dtype=np.bool_, copy=False)[selected]
    return output.astype(np.int8)


def materialize_selected(
    config: BuildConfig,
    fingerprint: dict,
    catalog_arrays: dict[str, np.ndarray],
    catalog_metadata: dict,
    cohort_arrays: dict[str, np.ndarray],
    cohort_metadata: dict,
) -> dict:
    item_ids = catalog_arrays["original_item_ids"]
    selected_user_ids = cohort_arrays["selected_user_ids"]
    item_map = np.zeros(int(item_ids.max()) + 1, dtype=np.int32)
    item_map[item_ids] = np.arange(1, len(item_ids) + 1, dtype=np.int32)
    user_map = np.zeros(int(selected_user_ids.max()) + 1, dtype=np.int32)
    user_map[selected_user_ids] = np.arange(
        1,
        len(selected_user_ids) + 1,
        dtype=np.int32,
    )
    raw_seen = np.zeros(0, dtype=np.int32)
    filtered_seen = np.zeros(len(user_map), dtype=np.int32)
    columns: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "user_idx",
            "item_idx",
            "behavior",
            "label",
            "time_ms",
            "raw_ordinal",
            "filtered_position",
            "window_index",
        )
    }
    rows = 0
    kept = 0
    started = time.perf_counter()
    for chunk_index, chunk in enumerate(
        read_chunks(config, TENREC_COLUMNS),
        start=1,
    ):
        users = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
        items = chunk["item_id"].to_numpy(dtype=np.int64, copy=False)
        raw_positions, raw_seen = consume_user_positions(users, raw_seen)
        user_in_range = users < len(user_map)
        item_in_range = items < len(item_map)
        selected = user_in_range & item_in_range
        selected[user_in_range & item_in_range] &= (
            user_map[users[user_in_range & item_in_range]] > 0
        ) & (item_map[items[user_in_range & item_in_range]] > 0)
        if selected.any():
            selected_users = users[selected]
            filtered_positions, filtered_seen = consume_user_positions(
                selected_users,
                filtered_seen,
            )
            within_horizon = filtered_positions < config.required_events
            if within_horizon.any():
                chosen_rows = np.flatnonzero(selected)[within_horizon]
                chosen = np.zeros(len(chunk), dtype=np.bool_)
                chosen[chosen_rows] = True
                chosen_users = users[chosen]
                chosen_items = items[chosen]
                chosen_positions = filtered_positions[within_horizon]
                columns["user_idx"].append(user_map[chosen_users])
                columns["item_idx"].append(item_map[chosen_items])
                columns["behavior"].append(behavior_values(chunk, chosen))
                columns["label"].append(positive_values(chunk, chosen))
                columns["time_ms"].append(
                    raw_positions[chosen].astype(np.int64, copy=False) * 1_000
                )
                columns["raw_ordinal"].append(
                    raw_positions[chosen].astype(np.int32, copy=False)
                )
                columns["filtered_position"].append(
                    chosen_positions.astype(np.int16, copy=False)
                )
                columns["window_index"].append(
                    np.where(
                        chosen_positions < config.history_length,
                        -1,
                        0,
                    ).astype(np.int8)
                )
                kept += int(np.count_nonzero(within_horizon))
        rows += len(chunk)
        print(
            f"phase=materialize chunks={chunk_index} rows={rows:,} kept={kept:,} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if np.count_nonzero(filtered_seen[selected_user_ids] >= config.required_events) == len(
            selected_user_ids
        ):
            break
    arrays = {
        name: np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        for name, parts in columns.items()
    }
    expected_rows = len(selected_user_ids) * config.required_events
    if len(arrays["user_idx"]) != expected_rows:
        raise ValueError(
            f"materialized {len(arrays['user_idx'])} rows but expected {expected_rows}"
        )
    per_user = np.bincount(
        arrays["user_idx"].astype(np.int64, copy=False),
        minlength=len(selected_user_ids) + 1,
    )[1:]
    if not np.all(per_user == config.required_events):
        raise ValueError("selected users do not have exactly the requested retained horizon")
    arrays["original_user_ids"] = selected_user_ids.astype(np.int64, copy=True)
    arrays["original_item_ids"] = item_ids.astype(np.int64, copy=True)
    metadata = {
        "protocol": PROTOCOL,
        "scientific_result": False,
        "dataset": "tenrec-qk",
        "source": fingerprint,
        "source_rows_scanned_until_complete": rows,
        "catalog_size": config.catalog_size,
        "fitted_items": len(item_ids),
        "catalog_fit": f"top-{config.catalog_size} from each user's first {config.base_prefix} raw exposures",
        "catalog_item_ids_sha256": catalog_metadata["catalog_item_ids_sha256"],
        "selected_user_ids_sha256": cohort_metadata["selected_user_ids_sha256"],
        "selected_users": len(selected_user_ids),
        "fit_calibration_users": config.fit_calibration_users,
        "benchmark_users": config.benchmark_users,
        "nested_benchmark_prefixes": list(config.cohort_sizes),
        "history_length": config.history_length,
        "slide": config.slide,
        "required_filtered_events": config.required_events,
        "base_prefix": config.history_length,
        "window_size": config.slide,
        "window_count": 1,
        "old_window_filtered_positions": [0, config.history_length],
        "target_window_filtered_positions": [
            config.slide,
            config.slide + config.history_length,
        ],
        "base_window_rows": int(np.count_nonzero(arrays["window_index"] == -1)),
        "update_window_rows": int(np.count_nonzero(arrays["window_index"] == 0)),
        "split_rows": {
            "base": int(np.count_nonzero(arrays["window_index"] == -1)),
            "window_0": int(np.count_nonzero(arrays["window_index"] == 0)),
        },
        "split_positive_rows": {
            "base": int(arrays["label"][arrays["window_index"] == -1].sum()),
            "window_0": int(arrays["label"][arrays["window_index"] == 0].sum()),
        },
        "rows": int(len(arrays["user_idx"])),
        "positive_rows": int(arrays["label"].sum()),
        "num_behaviors": 5,
        "positive_rule": "click or follow or like or share",
        "ordering": "stable official within-user file order without calendar-time semantics",
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output,
        **arrays,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    return {
        **metadata,
        "output": str(config.output),
        "output_bytes": config.output.stat().st_size,
    }


def write_control_files(
    config: BuildConfig,
    fingerprint: dict,
    catalog_metadata: dict,
    cohort_arrays: dict[str, np.ndarray],
    cohort_metadata: dict,
    materialized: dict | None,
) -> dict:
    fit_calibration = cohort_arrays["fit_calibration_user_ids"]
    benchmark = cohort_arrays["benchmark_user_ids"]
    nested = {
        str(size): {
            "prefix_length": size,
            "user_ids_sha256": array_sha256(benchmark[:size]),
        }
        for size in config.cohort_sizes
    }
    cohort_ids = {
        "protocol": PROTOCOL,
        "hash_salt": config.hash_salt,
        "selection": "fit/calibration prefix followed by one nested benchmark pool",
        "fit_calibration_user_ids": fit_calibration.tolist(),
        "benchmark_user_ids": benchmark.tolist(),
        "nested_benchmark_prefixes": nested,
    }
    save_json(cohort_ids, config.cohort_ids)
    manifest = {
        "protocol": PROTOCOL,
        "status": "materialized" if materialized is not None else "audit_only",
        "scientific_result": False,
        "source": fingerprint,
        "catalog": {
            "size": config.catalog_size,
            "base_prefix_raw_events": config.base_prefix,
            "item_ids_sha256": catalog_metadata["catalog_item_ids_sha256"],
            "cache": str(config.catalog_cache),
        },
        "history": {
            "old": [0, config.history_length],
            "target": [config.slide, config.slide + config.history_length],
            "required_filtered_events": config.required_events,
        },
        "cohorts": {
            "eligible_users": cohort_metadata["eligible_users"],
            "fit_calibration_users": config.fit_calibration_users,
            "benchmark_users": config.benchmark_users,
            "nested_benchmark_prefixes": nested,
            "selected_user_ids_sha256": cohort_metadata[
                "selected_user_ids_sha256"
            ],
            "cache": str(config.cohort_cache),
            "ids": str(config.cohort_ids),
        },
        "output": materialized,
        "scope": (
            "label-free ordinal QK capacity and one model-edge development input; "
            "not a calendar-time or paper-result claim"
        ),
    }
    save_json(manifest, config.manifest)
    return manifest


def run(config: BuildConfig, audit_only: bool = False) -> dict:
    validate_config(config)
    fingerprint = source_fingerprint(config)
    catalog_arrays, catalog_metadata = load_or_build_catalog(config, fingerprint)
    cohort_arrays, cohort_metadata = load_or_build_cohort(
        config,
        fingerprint,
        catalog_arrays,
        catalog_metadata,
    )
    materialized = None
    if not audit_only:
        materialized = materialize_selected(
            config,
            fingerprint,
            catalog_arrays,
            catalog_metadata,
            cohort_arrays,
            cohort_metadata,
        )
    return write_control_files(
        config,
        fingerprint,
        catalog_metadata,
        cohort_arrays,
        cohort_metadata,
        materialized,
    )


def main() -> None:
    args = parse_args()
    manifest = run(config_from_args(args), audit_only=args.audit_only)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
