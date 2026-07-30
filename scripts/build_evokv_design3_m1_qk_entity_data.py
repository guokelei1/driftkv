from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hstu_kvcache.utils import save_json

_SHARED_NAME = "_evokv_design3_m1_qk_data_shared"
_SHARED_PATH = Path(__file__).with_name(
    "build_evokv_design3_m1_qk_data.py"
)
if _SHARED_NAME in sys.modules:
    _SHARED = sys.modules[_SHARED_NAME]
else:
    _SHARED_SPEC = importlib.util.spec_from_file_location(
        _SHARED_NAME,
        _SHARED_PATH,
    )
    if _SHARED_SPEC is None or _SHARED_SPEC.loader is None:
        raise RuntimeError("cannot load shared QK builder")
    _SHARED = importlib.util.module_from_spec(_SHARED_SPEC)
    sys.modules[_SHARED_NAME] = _SHARED
    _SHARED_SPEC.loader.exec_module(_SHARED)

TENREC_COLUMNS = _SHARED.TENREC_COLUMNS
array_sha256 = _SHARED.array_sha256
behavior_values = _SHARED.behavior_values
cache_matches = _SHARED.cache_matches
consume_user_positions = _SHARED.consume_user_positions
grow_vector = _SHARED.grow_vector
load_npz = _SHARED.load_npz
positive_values = _SHARED.positive_values
read_chunks = _SHARED.read_chunks
save_npz = _SHARED.save_npz
source_fingerprint = _SHARED.source_fingerprint
splitmix64 = _SHARED.splitmix64
stable_user_order = _SHARED.stable_user_order

PROTOCOL = "evokv_design3_m1_qk_base_entity_data_development_v0"
DEFAULT_SOURCE = Path("data/tenrec/Tenrec.zip")
DEFAULT_MEMBER = "Tenrec/QK-video.csv"
DEFAULT_CACHE_DIR = Path(
    "data/processed/evokv_d3_m1_qk_entity_cache"
)
DEFAULT_OUTPUT = Path(
    "data/processed/evokv_d3_m1_qk_entity_2560.npz"
)
DEFAULT_MANIFEST = Path(
    "configs/evokv_d3/m1/qk_entity_manifest.json"
)
DEFAULT_COHORT_IDS = Path(
    "configs/evokv_d3/m1/qk_entity_cohorts.json"
)
DEFAULT_HASH_SALT = "evokv-d3-qk-m1-v1"


@dataclass(frozen=True)
class BuildConfig:
    source: Path = DEFAULT_SOURCE
    member: str = DEFAULT_MEMBER
    cache_dir: Path = DEFAULT_CACHE_DIR
    output: Path = DEFAULT_OUTPUT
    manifest: Path = DEFAULT_MANIFEST
    cohort_ids: Path = DEFAULT_COHORT_IDS
    prediction_catalog_size: int = 250_000
    base_prefix: int = 64
    history_length: int = 512
    slide: int = 32
    fit_calibration_users: int = 512
    cohort_sizes: tuple[int, ...] = (512, 1_024, 2_048)
    primary_cohort_size: int = 2_048
    embedding_width: int = 1_536
    model_layers: int = 24
    model_heads: int = 24
    model_head_dim: int = 64
    hash_salt: str = DEFAULT_HASH_SALT
    chunk_size: int = 2_000_000
    refresh: bool = False

    @property
    def required_events(self) -> int:
        return self.history_length + 2 * self.slide

    @property
    def benchmark_users(self) -> int:
        return max(self.cohort_sizes)

    @property
    def selected_users(self) -> int:
        return self.fit_calibration_users + self.benchmark_users

    @property
    def catalog_cache(self) -> Path:
        return self.cache_dir / (
            f"entity_catalog_base{self.base_prefix}_"
            f"top{self.prediction_catalog_size}.npz"
        )

    @property
    def cohort_cache(self) -> Path:
        return self.cache_dir / (
            f"entity_cohort_base{self.base_prefix}_"
            f"top{self.prediction_catalog_size}_"
            f"events{self.required_events}_"
            f"fit{self.fit_calibration_users}_"
            f"bench{self.benchmark_users}.npz"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--member", default=DEFAULT_MEMBER)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cohort-ids", type=Path, default=DEFAULT_COHORT_IDS)
    parser.add_argument(
        "--prediction-catalog-size",
        type=int,
        default=250_000,
    )
    parser.add_argument("--base-prefix", type=int, default=64)
    parser.add_argument("--history-length", type=int, default=512)
    parser.add_argument("--slide", type=int, default=32)
    parser.add_argument("--fit-calibration-users", type=int, default=512)
    parser.add_argument(
        "--cohort-sizes",
        type=int,
        nargs="+",
        default=[512, 1_024, 2_048],
    )
    parser.add_argument(
        "--primary-cohort-size",
        type=int,
        default=2_048,
    )
    parser.add_argument("--embedding-width", type=int, default=1_536)
    parser.add_argument("--model-layers", type=int, default=24)
    parser.add_argument("--model-heads", type=int, default=24)
    parser.add_argument("--model-head-dim", type=int, default=64)
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
        prediction_catalog_size=args.prediction_catalog_size,
        base_prefix=args.base_prefix,
        history_length=args.history_length,
        slide=args.slide,
        fit_calibration_users=args.fit_calibration_users,
        cohort_sizes=tuple(sorted(set(args.cohort_sizes))),
        primary_cohort_size=args.primary_cohort_size,
        embedding_width=args.embedding_width,
        model_layers=args.model_layers,
        model_heads=args.model_heads,
        model_head_dim=args.model_head_dim,
        hash_salt=args.hash_salt,
        chunk_size=args.chunk_size,
        refresh=args.refresh,
    )


def validate_config(config: BuildConfig) -> None:
    if config.prediction_catalog_size < 1:
        raise ValueError("prediction catalog size must be positive")
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
    if config.primary_cohort_size not in config.cohort_sizes:
        raise ValueError("primary cohort size must be a nested cohort size")
    if (
        config.embedding_width < 1
        or config.model_layers < 1
        or config.model_heads < 1
        or config.model_head_dim < 1
    ):
        raise ValueError("planned model dimensions must be positive")
    if (
        config.model_heads * config.model_head_dim
        != config.embedding_width
    ):
        raise ValueError("heads times head dimension must equal hidden size")
    if config.chunk_size < 1:
        raise ValueError("chunk size must be positive")


def entity_context_rule(config: BuildConfig) -> str:
    return (
        "every item observed in each user's first "
        f"{config.base_prefix} raw exposures receives one stable row; "
        f"the top {config.prediction_catalog_size} base-frequency rows are "
        "prediction items, remaining rows are context-only entities, and an "
        "item first observed after the base prefix maps by SplitMix64 into "
        "the existing context-entity rows without extending the table"
    )


def catalog_cache_key(config: BuildConfig, fingerprint: dict) -> dict:
    return {
        "protocol": PROTOCOL,
        "phase": "base_entity_catalog",
        "source": fingerprint,
        "base_prefix_raw_events": config.base_prefix,
        "prediction_catalog_size": config.prediction_catalog_size,
        "ordering": (
            "base frequency descending, original item id ascending for ties"
        ),
    }


def build_catalog_cache(
    config: BuildConfig,
    fingerprint: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    seen_counts = np.zeros(0, dtype=np.int32)
    base_item_counts = np.zeros(0, dtype=np.int64)
    rows = 0
    base_rows = 0
    started = time.perf_counter()
    for chunk_index, chunk in enumerate(
        read_chunks(config, ("user_id", "item_id")),
        start=1,
    ):
        users = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
        items = chunk["item_id"].to_numpy(dtype=np.int64, copy=False)
        positions, seen_counts = consume_user_positions(
            users,
            seen_counts,
        )
        selected = positions < config.base_prefix
        if selected.any():
            counts = np.bincount(items[selected])
            base_item_counts = grow_vector(base_item_counts, len(counts))
            base_item_counts[: len(counts)] += counts
            base_rows += int(np.count_nonzero(selected))
        rows += len(chunk)
        print(
            f"phase=entity_catalog chunks={chunk_index} rows={rows:,} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    active_items = np.flatnonzero(base_item_counts)
    order = active_items[
        np.argsort(-base_item_counts[active_items], kind="stable")
    ]
    if len(order) <= config.prediction_catalog_size:
        raise ValueError(
            "base-active catalog must contain prediction and context rows"
        )
    prediction = order[: config.prediction_catalog_size]
    context = order[config.prediction_catalog_size :]
    arrays = {
        "original_item_ids": prediction.astype(np.int64),
        "base_entity_original_item_ids": order.astype(np.int64),
        "base_item_frequencies": base_item_counts[order].astype(np.int64),
        "is_prediction_item": np.concatenate(
            [
                np.ones(len(prediction), dtype=np.int8),
                np.zeros(len(context), dtype=np.int8),
            ]
        ),
    }
    metadata = {
        **catalog_cache_key(config, fingerprint),
        "source_rows_scanned": rows,
        "base_rows": base_rows,
        "base_entity_items": int(len(order)),
        "num_prediction_items": int(len(prediction)),
        "context_entity_rows": int(len(context)),
        "base_entity_item_ids_sha256": array_sha256(order),
        "prediction_item_ids_sha256": array_sha256(prediction),
        "context_entity_item_ids_sha256": array_sha256(context),
        "base_entity_frequency_sum": int(
            base_item_counts[order].sum()
        ),
        "minimum_base_entity_frequency": int(
            base_item_counts[order].min()
        ),
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


def build_dense_entity_map(
    original_item_ids: np.ndarray,
) -> np.ndarray:
    if len(original_item_ids) == 0:
        raise ValueError("entity catalog cannot be empty")
    if np.any(original_item_ids < 0):
        raise ValueError("entity catalog contains negative item ids")
    item_map = np.zeros(
        int(original_item_ids.max()) + 1,
        dtype=np.int32,
    )
    item_map[original_item_ids] = np.arange(
        1,
        len(original_item_ids) + 1,
        dtype=np.int32,
    )
    return item_map


def map_entity_item_ids(
    items: np.ndarray,
    item_map: np.ndarray,
    config: BuildConfig,
    context_entity_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if context_entity_rows < 1:
        raise ValueError("stream fallback requires context entity rows")
    in_range = (items >= 0) & (items < len(item_map))
    direct = np.zeros(len(items), dtype=np.int32)
    direct[in_range] = item_map[items[in_range]]
    base_seen = direct > 0
    predicted = base_seen & (
        direct <= config.prediction_catalog_size
    )
    exact_context = base_seen & ~predicted
    stream_only = ~base_seen
    mapped = direct.copy()
    if stream_only.any():
        mapped[stream_only] = (
            config.prediction_catalog_size
            + 1
            + (
                splitmix64(items[stream_only])
                % np.uint64(context_entity_rows)
            ).astype(np.int64)
        ).astype(np.int32)
    return mapped, predicted, exact_context, stream_only


def cohort_cache_key(
    config: BuildConfig,
    fingerprint: dict,
    catalog_metadata: dict,
) -> dict:
    return {
        "protocol": PROTOCOL,
        "phase": "entity_cohort",
        "source": fingerprint,
        "base_entity_item_ids_sha256": catalog_metadata[
            "base_entity_item_ids_sha256"
        ],
        "required_events": config.required_events,
        "fit_calibration_users": config.fit_calibration_users,
        "cohort_sizes": list(config.cohort_sizes),
        "primary_cohort_size": config.primary_cohort_size,
        "hash_salt": config.hash_salt,
        "cohort_length_basis": (
            "all raw exposure events; stream-only items use a context-entity "
            "fallback and no event is dropped"
        ),
    }


def build_cohort_cache(
    config: BuildConfig,
    fingerprint: dict,
    catalog_arrays: dict[str, np.ndarray],
    catalog_metadata: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    original_item_ids = catalog_arrays[
        "base_entity_original_item_ids"
    ]
    item_map = build_dense_entity_map(original_item_ids)
    context_entity_rows = int(
        catalog_metadata["context_entity_rows"]
    )
    user_counts = np.zeros(0, dtype=np.int32)
    prediction_items_seen = np.zeros(
        config.prediction_catalog_size,
        dtype=np.bool_,
    )
    context_entities_seen = np.zeros(
        context_entity_rows,
        dtype=np.bool_,
    )
    fallback_entities_seen = np.zeros(
        context_entity_rows,
        dtype=np.bool_,
    )
    stream_only_items_seen = np.zeros(0, dtype=np.bool_)
    prediction_rows = 0
    exact_context_rows = 0
    stream_only_rows = 0
    rows = 0
    started = time.perf_counter()
    for chunk_index, chunk in enumerate(
        read_chunks(config, ("user_id", "item_id")),
        start=1,
    ):
        users = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
        items = chunk["item_id"].to_numpy(dtype=np.int64, copy=False)
        counts = np.bincount(users)
        user_counts = grow_vector(user_counts, len(counts))
        user_counts[: len(counts)] += counts.astype(
            user_counts.dtype,
            copy=False,
        )
        mapped, predicted, exact_context, stream_only = (
            map_entity_item_ids(
                items,
                item_map,
                config,
                context_entity_rows,
            )
        )
        if predicted.any():
            prediction_items_seen[
                mapped[predicted] - 1
            ] = True
        if exact_context.any():
            context_entities_seen[
                mapped[exact_context]
                - config.prediction_catalog_size
                - 1
            ] = True
        if stream_only.any():
            stream_items = items[stream_only]
            required = int(stream_items.max()) + 1
            stream_only_items_seen = grow_vector(
                stream_only_items_seen,
                required,
            )
            stream_only_items_seen[stream_items] = True
            fallback_entities_seen[
                mapped[stream_only]
                - config.prediction_catalog_size
                - 1
            ] = True
        prediction_rows += int(np.count_nonzero(predicted))
        exact_context_rows += int(
            np.count_nonzero(exact_context)
        )
        stream_only_rows += int(np.count_nonzero(stream_only))
        rows += len(chunk)
        print(
            f"phase=entity_cohort chunks={chunk_index} rows={rows:,} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    eligible = np.flatnonzero(
        user_counts >= config.required_events
    )
    ordered = stable_user_order(eligible, config.hash_salt)
    if len(ordered) < config.selected_users:
        raise ValueError(
            f"need {config.selected_users} eligible users but "
            f"found {len(ordered)}"
        )
    fit_calibration = ordered[: config.fit_calibration_users]
    benchmark = ordered[
        config.fit_calibration_users :
        config.fit_calibration_users + config.benchmark_users
    ]
    selected = np.concatenate([fit_calibration, benchmark])
    arrays = {
        "eligible_user_ids": eligible.astype(np.int64),
        "eligible_raw_lengths": user_counts[eligible].astype(
            np.int32
        ),
        "fit_calibration_user_ids": fit_calibration,
        "benchmark_user_ids": benchmark,
        "selected_user_ids": selected,
    }
    metadata = {
        **cohort_cache_key(
            config,
            fingerprint,
            catalog_metadata,
        ),
        "source_rows_scanned": rows,
        "retained_rows": rows,
        "prediction_rows": prediction_rows,
        "exact_context_rows": exact_context_rows,
        "stream_only_fallback_rows": stream_only_rows,
        "context_rows": exact_context_rows + stream_only_rows,
        "dropped_rows": 0,
        "unique_prediction_entities_seen": int(
            np.count_nonzero(prediction_items_seen)
        ),
        "unique_exact_context_entities_seen": int(
            np.count_nonzero(context_entities_seen)
        ),
        "unique_stream_only_original_items_seen": int(
            np.count_nonzero(stream_only_items_seen)
        ),
        "unique_fallback_context_entities_touched": int(
            np.count_nonzero(fallback_entities_seen)
        ),
        "fallback_collisions": int(
            np.count_nonzero(stream_only_items_seen)
            - np.count_nonzero(fallback_entities_seen)
        ),
        "base_entity_rows_seen_directly": int(
            np.count_nonzero(prediction_items_seen)
            + np.count_nonzero(context_entities_seen)
        ),
        "base_entity_direct_coverage": float(
            (
                np.count_nonzero(prediction_items_seen)
                + np.count_nonzero(context_entities_seen)
            )
            / len(original_item_ids)
        ),
        "eligible_users": int(len(eligible)),
        "selected_users": int(len(selected)),
        "fit_calibration_user_ids_sha256": array_sha256(
            fit_calibration
        ),
        "benchmark_user_ids_sha256": array_sha256(benchmark),
        "selected_user_ids_sha256": array_sha256(selected),
        "selection": (
            "filter by raw retained length, sort by SHA-256 of "
            f"{config.hash_salt}:<original_user_id>, reserve "
            "fit/calibration, then take one benchmark pool whose "
            "prefixes form nested cohorts"
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
    expected = cohort_cache_key(
        config,
        fingerprint,
        catalog_metadata,
    )
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


def materialize_selected(
    config: BuildConfig,
    fingerprint: dict,
    catalog_arrays: dict[str, np.ndarray],
    catalog_metadata: dict,
    cohort_arrays: dict[str, np.ndarray],
    cohort_metadata: dict,
) -> dict:
    original_item_ids = catalog_arrays[
        "base_entity_original_item_ids"
    ]
    selected_user_ids = cohort_arrays["selected_user_ids"]
    item_map = build_dense_entity_map(original_item_ids)
    context_entity_rows = int(
        catalog_metadata["context_entity_rows"]
    )
    user_map = np.zeros(
        int(selected_user_ids.max()) + 1,
        dtype=np.int32,
    )
    user_map[selected_user_ids] = np.arange(
        1,
        len(selected_user_ids) + 1,
        dtype=np.int32,
    )
    raw_seen = np.zeros(0, dtype=np.int32)
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
            "is_prediction_item",
            "is_stream_only_fallback",
        )
    }
    mapped_rows_seen = np.zeros(
        len(original_item_ids) + 1,
        dtype=np.bool_,
    )
    direct_rows_seen = np.zeros(
        len(original_item_ids) + 1,
        dtype=np.bool_,
    )
    prediction_rows_seen = np.zeros(
        config.prediction_catalog_size,
        dtype=np.bool_,
    )
    exact_context_rows_seen = np.zeros(
        context_entity_rows,
        dtype=np.bool_,
    )
    fallback_context_rows_seen = np.zeros(
        context_entity_rows,
        dtype=np.bool_,
    )
    stream_only_items_seen = np.zeros(0, dtype=np.bool_)
    rows = 0
    kept = 0
    started = time.perf_counter()
    for chunk_index, chunk in enumerate(
        read_chunks(config, TENREC_COLUMNS),
        start=1,
    ):
        users = chunk["user_id"].to_numpy(
            dtype=np.int64,
            copy=False,
        )
        items = chunk["item_id"].to_numpy(
            dtype=np.int64,
            copy=False,
        )
        raw_positions, raw_seen = consume_user_positions(
            users,
            raw_seen,
        )
        mapped, predicted, exact_context, stream_only = (
            map_entity_item_ids(
                items,
                item_map,
                config,
                context_entity_rows,
            )
        )
        user_in_range = (users >= 0) & (users < len(user_map))
        selected = np.zeros(len(users), dtype=np.bool_)
        selected[user_in_range] = (
            user_map[users[user_in_range]] > 0
        )
        chosen = selected & (
            raw_positions < config.required_events
        )
        if chosen.any():
            chosen_users = users[chosen]
            chosen_items = items[chosen]
            chosen_positions = raw_positions[chosen]
            chosen_mapped = mapped[chosen]
            chosen_prediction = predicted[chosen]
            chosen_exact_context = exact_context[chosen]
            chosen_stream_only = stream_only[chosen]
            chosen_labels = positive_values(chunk, chosen)
            chosen_labels[~chosen_prediction] = 0
            columns["user_idx"].append(user_map[chosen_users])
            columns["item_idx"].append(chosen_mapped)
            columns["behavior"].append(behavior_values(chunk, chosen))
            columns["label"].append(chosen_labels)
            columns["time_ms"].append(
                chosen_positions.astype(np.int64, copy=False)
                * 1_000
            )
            columns["raw_ordinal"].append(
                chosen_positions.astype(np.int32, copy=False)
            )
            columns["filtered_position"].append(
                chosen_positions.astype(np.int32, copy=False)
            )
            columns["window_index"].append(
                np.where(
                    chosen_positions < config.history_length,
                    -1,
                    np.where(
                        chosen_positions
                        < config.history_length + config.slide,
                        0,
                        1,
                    ),
                ).astype(np.int8)
            )
            columns["is_prediction_item"].append(
                chosen_prediction.astype(np.int8, copy=False)
            )
            columns["is_stream_only_fallback"].append(
                chosen_stream_only.astype(np.int8, copy=False)
            )
            mapped_rows_seen[chosen_mapped] = True
            direct = ~chosen_stream_only
            direct_rows_seen[chosen_mapped[direct]] = True
            prediction_rows_seen[
                chosen_mapped[chosen_prediction] - 1
            ] = True
            exact_context_rows_seen[
                chosen_mapped[chosen_exact_context]
                - config.prediction_catalog_size
                - 1
            ] = True
            fallback_context_rows_seen[
                chosen_mapped[chosen_stream_only]
                - config.prediction_catalog_size
                - 1
            ] = True
            stream_items = chosen_items[chosen_stream_only]
            if len(stream_items):
                required = int(stream_items.max()) + 1
                stream_only_items_seen = grow_vector(
                    stream_only_items_seen,
                    required,
                )
                stream_only_items_seen[stream_items] = True
            kept += int(np.count_nonzero(chosen))
        rows += len(chunk)
        print(
            f"phase=entity_materialize chunks={chunk_index} "
            f"rows={rows:,} kept={kept:,} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if (
            len(raw_seen) > int(selected_user_ids.max())
            and np.all(
                raw_seen[selected_user_ids]
                >= config.required_events
            )
        ):
            break
    arrays = {
        name: (
            np.concatenate(parts)
            if parts
            else np.empty(0, dtype=np.int64)
        )
        for name, parts in columns.items()
    }
    expected_rows = (
        len(selected_user_ids) * config.required_events
    )
    if len(arrays["user_idx"]) != expected_rows:
        raise ValueError(
            f"materialized {len(arrays['user_idx'])} rows but "
            f"expected {expected_rows}"
        )
    per_user = np.bincount(
        arrays["user_idx"].astype(np.int64, copy=False),
        minlength=len(selected_user_ids) + 1,
    )[1:]
    if not np.all(per_user == config.required_events):
        raise ValueError(
            "selected users do not have exactly the requested horizon"
        )
    if np.any(
        (arrays["item_idx"] > config.prediction_catalog_size)
        & (arrays["label"] > 0)
    ):
        raise ValueError("context-only rows contain positive targets")
    arrays["original_user_ids"] = selected_user_ids.astype(
        np.int64,
        copy=True,
    )
    arrays["original_item_ids"] = catalog_arrays[
        "original_item_ids"
    ].astype(
        np.int64,
        copy=True,
    )
    arrays["base_entity_original_item_ids"] = original_item_ids.astype(
        np.int64,
        copy=True,
    )
    prediction_rows = int(
        arrays["is_prediction_item"].sum()
    )
    context_rows = int(
        len(arrays["is_prediction_item"]) - prediction_rows
    )
    fallback_rows = int(
        arrays["is_stream_only_fallback"].sum()
    )
    base_rows = int(
        np.count_nonzero(arrays["window_index"] == -1)
    )
    update_rows = int(
        np.count_nonzero(arrays["window_index"] == 0)
    )
    heldout_rows = int(
        np.count_nonzero(arrays["window_index"] == 1)
    )
    metadata = {
        "protocol": PROTOCOL,
        "scientific_result": False,
        "formal_design3": False,
        "dataset": "tenrec-qk",
        "source": fingerprint,
        "source_rows_scanned_until_complete": rows,
        "base_entity_items": len(original_item_ids),
        "num_prediction_items": config.prediction_catalog_size,
        "context_entity_rows": context_entity_rows,
        "context_hash_buckets": context_entity_rows,
        "context_hash_buckets_role": (
            "loader compatibility count; base context entities have exact "
            "rows and only stream-only originals use hashing"
        ),
        "fitted_items": len(original_item_ids),
        "context_rule": entity_context_rule(config),
        "catalog_fit": (
            f"all distinct items from each user's first "
            f"{config.base_prefix} raw exposures; top "
            f"{config.prediction_catalog_size} by base frequency are "
            "prediction rows"
        ),
        "base_entity_item_ids_sha256": catalog_metadata[
            "base_entity_item_ids_sha256"
        ],
        "prediction_item_ids_sha256": catalog_metadata[
            "prediction_item_ids_sha256"
        ],
        "context_entity_item_ids_sha256": catalog_metadata[
            "context_entity_item_ids_sha256"
        ],
        "selected_user_ids_sha256": cohort_metadata[
            "selected_user_ids_sha256"
        ],
        "selected_users": len(selected_user_ids),
        "fit_calibration_users": config.fit_calibration_users,
        "benchmark_users": config.benchmark_users,
        "primary_benchmark_users": config.primary_cohort_size,
        "nested_benchmark_prefixes": list(config.cohort_sizes),
        "history_length": config.history_length,
        "slide": config.slide,
        "required_filtered_events": config.required_events,
        "base_prefix": config.history_length,
        "window_size": config.slide,
        "window_count": 2,
        "old_window_filtered_positions": [
            0,
            config.history_length,
        ],
        "target_window_filtered_positions": [
            config.slide,
            config.slide + config.history_length,
        ],
        "update_window_filtered_positions": [
            config.history_length,
            config.history_length + config.slide,
        ],
        "heldout_window_filtered_positions": [
            config.history_length + config.slide,
            config.history_length + 2 * config.slide,
        ],
        "base_window_rows": base_rows,
        "update_window_rows": update_rows,
        "heldout_window_rows": heldout_rows,
        "split_rows": {
            "base": base_rows,
            "window_0": update_rows,
            "window_1": heldout_rows,
        },
        "split_positive_rows": {
            "base": int(
                arrays["label"][
                    arrays["window_index"] == -1
                ].sum()
            ),
            "window_0": int(
                arrays["label"][
                    arrays["window_index"] == 0
                ].sum()
            ),
            "window_1": int(
                arrays["label"][
                    arrays["window_index"] == 1
                ].sum()
            ),
        },
        "rows": int(len(arrays["user_idx"])),
        "positive_rows": int(arrays["label"].sum()),
        "prediction_rows": prediction_rows,
        "context_rows": context_rows,
        "exact_context_rows": context_rows - fallback_rows,
        "stream_only_fallback_rows": fallback_rows,
        "unique_prediction_entity_rows_accessed": int(
            np.count_nonzero(prediction_rows_seen)
        ),
        "unique_exact_context_entity_rows_accessed": int(
            np.count_nonzero(exact_context_rows_seen)
        ),
        "unique_stream_only_original_items_accessed": int(
            np.count_nonzero(stream_only_items_seen)
        ),
        "unique_fallback_context_entity_rows_accessed": int(
            np.count_nonzero(fallback_context_rows_seen)
        ),
        "unique_mapped_entity_rows_accessed": int(
            np.count_nonzero(mapped_rows_seen)
        ),
        "unique_direct_base_entity_rows_accessed": int(
            np.count_nonzero(direct_rows_seen)
        ),
        "mapped_entity_row_coverage": float(
            np.count_nonzero(mapped_rows_seen)
            / len(original_item_ids)
        ),
        "direct_base_entity_row_coverage": float(
            np.count_nonzero(direct_rows_seen)
            / len(original_item_ids)
        ),
        "source_mapping_stats": {
            "rows": cohort_metadata["source_rows_scanned"],
            "prediction_rows": cohort_metadata["prediction_rows"],
            "exact_context_rows": cohort_metadata[
                "exact_context_rows"
            ],
            "stream_only_fallback_rows": cohort_metadata[
                "stream_only_fallback_rows"
            ],
            "context_rows": cohort_metadata["context_rows"],
            "dropped_rows": cohort_metadata["dropped_rows"],
            "unique_prediction_entities_seen": cohort_metadata[
                "unique_prediction_entities_seen"
            ],
            "unique_exact_context_entities_seen": cohort_metadata[
                "unique_exact_context_entities_seen"
            ],
            "unique_stream_only_original_items_seen": cohort_metadata[
                "unique_stream_only_original_items_seen"
            ],
            "unique_fallback_context_entities_touched": cohort_metadata[
                "unique_fallback_context_entities_touched"
            ],
            "fallback_collisions": cohort_metadata[
                "fallback_collisions"
            ],
            "base_entity_direct_coverage": cohort_metadata[
                "base_entity_direct_coverage"
            ],
        },
        "num_behaviors": 5,
        "positive_rule": (
            "click or follow or like or share for prediction rows; "
            "every exact-context and stream-fallback row is forced to zero"
        ),
        "ordering": (
            "stable official within-user file order without "
            "calendar-time semantics"
        ),
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
    fit_calibration = cohort_arrays[
        "fit_calibration_user_ids"
    ]
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
        "selection": (
            "fit/calibration prefix followed by one nested "
            "benchmark pool"
        ),
        "primary_benchmark_users": config.primary_cohort_size,
        "base_entity_item_ids_sha256": catalog_metadata[
            "base_entity_item_ids_sha256"
        ],
        "fit_calibration_user_ids": fit_calibration.tolist(),
        "benchmark_user_ids": benchmark.tolist(),
        "nested_benchmark_prefixes": nested,
    }
    save_json(cohort_ids, config.cohort_ids)
    embedding_rows = (
        int(catalog_metadata["base_entity_items"]) + 1
    )
    embedding_bytes = (
        embedding_rows * config.embedding_width * 4
    )
    per_record_per_version_kv_bytes = (
        config.model_layers
        * 2
        * config.history_length
        * config.embedding_width
        * 2
    )
    primary_old_target_kv_bytes = (
        config.primary_cohort_size
        * per_record_per_version_kv_bytes
        * 2
    )
    manifest = {
        "protocol": PROTOCOL,
        "status": (
            "materialized"
            if materialized is not None
            else "audit_only"
        ),
        "scientific_result": False,
        "formal_design3": False,
        "source": fingerprint,
        "catalog": {
            "base_prefix_raw_events": config.base_prefix,
            "base_entity_items": catalog_metadata[
                "base_entity_items"
            ],
            "num_prediction_items": catalog_metadata[
                "num_prediction_items"
            ],
            "context_entity_rows": catalog_metadata[
                "context_entity_rows"
            ],
            "context_rule": entity_context_rule(config),
            "base_entity_item_ids_sha256": catalog_metadata[
                "base_entity_item_ids_sha256"
            ],
            "prediction_item_ids_sha256": catalog_metadata[
                "prediction_item_ids_sha256"
            ],
            "context_entity_item_ids_sha256": catalog_metadata[
                "context_entity_item_ids_sha256"
            ],
            "base_rows": catalog_metadata["base_rows"],
            "cache": str(config.catalog_cache),
            "planned_embedding": {
                "rows_including_padding": embedding_rows,
                "hidden_size": config.embedding_width,
                "dtype": "float32",
                "bytes": embedding_bytes,
                "gibibytes": embedding_bytes / (2**30),
            },
        },
        "planned_model_shape": {
            "hidden_size": config.embedding_width,
            "num_layers": config.model_layers,
            "num_heads": config.model_heads,
            "head_dim": config.model_head_dim,
            "max_seq_len": config.history_length,
        },
        "planned_primary_kv": {
            "records": config.primary_cohort_size,
            "layers": config.model_layers,
            "history_tokens": config.history_length,
            "kv_width": config.embedding_width,
            "dtype": "float16",
            "per_record_per_version_bytes": (
                per_record_per_version_kv_bytes
            ),
            "old_and_private_target_bytes": (
                primary_old_target_kv_bytes
            ),
            "old_and_private_target_gibibytes": (
                primary_old_target_kv_bytes / (2**30)
            ),
        },
        "history": {
            "old": [0, config.history_length],
            "target": [
                config.slide,
                config.slide + config.history_length,
            ],
            "theta1_update": [
                config.history_length,
                config.history_length + config.slide,
            ],
            "heldout_evaluation": [
                config.history_length + config.slide,
                config.history_length + 2 * config.slide,
            ],
            "required_events": config.required_events,
        },
        "cohorts": {
            "eligible_users": cohort_metadata[
                "eligible_users"
            ],
            "length_basis": cohort_metadata[
                "cohort_length_basis"
            ],
            "fit_calibration_users": (
                config.fit_calibration_users
            ),
            "benchmark_users": config.benchmark_users,
            "primary_benchmark_users": (
                config.primary_cohort_size
            ),
            "nested_benchmark_prefixes": nested,
            "selected_user_ids_sha256": cohort_metadata[
                "selected_user_ids_sha256"
            ],
            "cache": str(config.cohort_cache),
            "ids": str(config.cohort_ids),
        },
        "source_mapping_stats": {
            "rows": cohort_metadata["source_rows_scanned"],
            "prediction_rows": cohort_metadata[
                "prediction_rows"
            ],
            "exact_context_rows": cohort_metadata[
                "exact_context_rows"
            ],
            "stream_only_fallback_rows": cohort_metadata[
                "stream_only_fallback_rows"
            ],
            "context_rows": cohort_metadata["context_rows"],
            "dropped_rows": cohort_metadata["dropped_rows"],
            "unique_prediction_entities_seen": cohort_metadata[
                "unique_prediction_entities_seen"
            ],
            "unique_exact_context_entities_seen": cohort_metadata[
                "unique_exact_context_entities_seen"
            ],
            "unique_stream_only_original_items_seen": cohort_metadata[
                "unique_stream_only_original_items_seen"
            ],
            "unique_fallback_context_entities_touched": cohort_metadata[
                "unique_fallback_context_entities_touched"
            ],
            "fallback_collisions": cohort_metadata[
                "fallback_collisions"
            ],
            "base_entity_direct_coverage": cohort_metadata[
                "base_entity_direct_coverage"
            ],
        },
        "output": materialized,
        "scope": (
            "non-scientific two-version QK D2/D3 mechanism-development "
            "input; ordinal order is not calendar time"
        ),
    }
    save_json(manifest, config.manifest)
    return manifest


def run(
    config: BuildConfig,
    audit_only: bool = False,
) -> dict:
    validate_config(config)
    fingerprint = source_fingerprint(config)
    catalog_arrays, catalog_metadata = load_or_build_catalog(
        config,
        fingerprint,
    )
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
    manifest = run(
        config_from_args(args),
        audit_only=args.audit_only,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
