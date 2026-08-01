from __future__ import annotations

import hashlib
import json
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL = "evokv_qk_successor_foundation_development_v1"
LENGTH_CACHE_PROTOCOL = "evokv_qk_successor_foundation_development_v0"
TENREC_COLUMNS = ("user_id", "item_id")
GIB = 1 << 30


@dataclass(frozen=True)
class FoundationConfig:
    source: Path
    member: str
    catalog_cache: Path
    length_cache: Path
    output: Path
    summary: Path
    roles: Path
    upstream_prepared: Path | None = None
    hash_salt: str = "evokv-qk-successor-foundation-v1"
    theta12_users: int = 2_048
    theta01_users: int = 2_560
    fit_users: int = 512
    profile_users: int = 512
    qualification_users: int = 512
    final_users: int = 65_536
    minimum_events: int = 96
    theta12_minimum_events: int = 1_024
    history_horizon: int = 544
    target_horizon: int = 512
    append_events: int = 32
    model_layers: int = 24
    model_hidden_size: int = 1_536
    kv_element_bytes: int = 2
    embedding_width: int = 4_096
    embedding_element_bytes: int = 4
    embedding_padding_rows: int = 1
    dense_parameter_count: int = 291_863_040
    single_card_torch_allocatable_bytes: int = 47_699_722_240
    capacity_gib: tuple[float, ...] = (36, 72, 144, 288, 576, 720)
    chunk_size: int = 2_000_000
    refresh_lengths: bool = False
    require_all_roles: bool = True
    require_all_capacities: bool = True

    @property
    def kv_bytes_per_token(self) -> int:
        return (
            2
            * self.model_layers
            * self.model_hidden_size
            * self.kv_element_bytes
        )

    @property
    def capacity_bytes(self) -> np.ndarray:
        return np.asarray(
            [round(value * GIB) for value in self.capacity_gib],
            dtype=np.int64,
        )


@dataclass(frozen=True)
class CatalogMapping:
    original_item_ids: np.ndarray
    base_item_frequencies: np.ndarray
    prediction_rows: int
    context_rows: int
    dense_map: np.ndarray
    metadata: dict

    @property
    def rows(self) -> int:
        return len(self.original_item_ids)

    def map_items(self, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        items = np.asarray(items, dtype=np.int64)
        direct = np.zeros(len(items), dtype=np.int32)
        in_range = (items >= 0) & (items < len(self.dense_map))
        direct[in_range] = self.dense_map[items[in_range]]
        fallback = direct == 0
        if fallback.any():
            direct[fallback] = (
                self.prediction_rows
                + 1
                + (
                    splitmix64(items[fallback])
                    % np.uint64(self.context_rows)
                ).astype(np.int64)
            ).astype(np.int32)
        return direct, fallback


def validate_config(config: FoundationConfig) -> None:
    counts = (
        config.theta12_users,
        config.theta01_users,
        config.fit_users,
        config.profile_users,
        config.qualification_users,
        config.final_users,
    )
    if min(counts) < 0 or config.final_users < 1:
        raise ValueError("role counts must be nonnegative and final_users positive")
    if config.minimum_events < config.append_events:
        raise ValueError("minimum_events must cover append_events")
    theta12_required_events = (
        config.history_horizon + config.append_events
    )
    if config.theta12_minimum_events < theta12_required_events:
        raise ValueError(
            "theta12 preferred minimum must cover its fixed edge"
        )
    if config.history_horizon != config.target_horizon + config.append_events:
        raise ValueError("history horizon must equal target horizon plus append")
    if min(
        config.model_layers,
        config.model_hidden_size,
        config.kv_element_bytes,
        config.embedding_width,
        config.embedding_element_bytes,
        config.embedding_padding_rows,
        config.dense_parameter_count,
        config.single_card_torch_allocatable_bytes,
        config.chunk_size,
    ) < 1:
        raise ValueError("model geometry and chunk size must be positive")
    if not config.capacity_gib or min(config.capacity_gib) <= 0:
        raise ValueError("capacity points must be positive")
    if tuple(sorted(set(config.capacity_gib))) != config.capacity_gib:
        raise ValueError("capacity points must be sorted and unique")
    if not config.hash_salt:
        raise ValueError("hash salt cannot be empty")


def source_fingerprint(config: FoundationConfig) -> dict:
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


def read_qk_chunks(
    config: FoundationConfig,
    columns: tuple[str, ...] = TENREC_COLUMNS,
) -> Iterator[pd.DataFrame]:
    dtypes = {"user_id": "int32", "item_id": "int32"}
    with (
        zipfile.ZipFile(config.source) as archive,
        archive.open(config.member) as stream,
    ):
        yield from pd.read_csv(
            stream,
            usecols=list(columns),
            dtype={name: dtypes[name] for name in columns},
            chunksize=config.chunk_size,
        )


def splitmix64(values: np.ndarray) -> np.ndarray:
    hashed = values.astype(np.uint64, copy=True)
    hashed = hashed + np.uint64(0x9E3779B97F4A7C15)
    hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    return hashed ^ (hashed >> np.uint64(31))


def array_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_user_order(user_ids: np.ndarray, salt: str) -> np.ndarray:
    ordered = sorted(
        (int(user_id) for user_id in user_ids),
        key=lambda user_id: (
            hashlib.sha256(f"{salt}:{user_id}".encode()).digest(),
            user_id,
        ),
    )
    return np.asarray(ordered, dtype=np.int64)


def stable_owner_ranks(
    user_ids: np.ndarray,
    salt: str,
    ranks: int,
) -> np.ndarray:
    if ranks < 1:
        raise ValueError("ranks must be positive")
    seed = np.uint64(
        int.from_bytes(
            hashlib.sha256(f"{salt}:owner".encode()).digest()[:8],
            "little",
        )
    )
    owner_hash = splitmix64(
        np.asarray(user_ids, dtype=np.uint64) ^ seed
    )
    return (owner_hash % np.uint64(ranks)).astype(np.int16)


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
        [
            np.array([0], dtype=np.int64),
            np.flatnonzero(users[1:] != users[:-1]) + 1,
        ]
    )
    lengths = np.diff(np.append(starts, len(users))).astype(
        np.int64,
        copy=False,
    )
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
    offsets = (
        seen_counts[group_users].astype(np.int64, copy=False)
        + preceding
        - starts
    )
    positions = np.arange(len(users), dtype=np.int64) + np.repeat(
        offsets,
        lengths,
    )
    if len(np.unique(group_users)) == len(group_users):
        seen_counts[group_users] += lengths.astype(
            seen_counts.dtype,
            copy=False,
        )
    else:
        np.add.at(
            seen_counts,
            group_users,
            lengths.astype(seen_counts.dtype, copy=False),
        )
    return positions, seen_counts


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {
            name: source[name].copy()
            for name in source.files
            if name != "metadata_json"
        }
        metadata = json.loads(str(source["metadata_json"].item()))
    return arrays, metadata


def _save_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def load_catalog(
    path: Path,
    fingerprint: dict | None = None,
) -> CatalogMapping:
    arrays, metadata = _load_npz(path)
    required = {
        "base_entity_original_item_ids",
        "base_item_frequencies",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"catalog is missing arrays: {sorted(missing)}")
    original = np.asarray(
        arrays["base_entity_original_item_ids"],
        dtype=np.int64,
    )
    frequencies = np.asarray(
        arrays["base_item_frequencies"],
        dtype=np.int64,
    )
    if len(original) == 0 or len(original) != len(frequencies):
        raise ValueError("catalog rows and frequencies differ")
    if len(np.unique(original)) != len(original):
        raise ValueError("catalog original item ids are not unique")
    prediction_rows = int(
        metadata.get(
            "num_prediction_items",
            metadata.get("prediction_catalog_size", 0),
        )
    )
    context_rows = int(metadata.get("context_entity_rows", 0))
    if prediction_rows < 1 or prediction_rows + context_rows != len(original):
        raise ValueError("catalog prediction/context partition is invalid")
    if fingerprint is not None and "source" in metadata:
        source_identity = metadata["source"]
        fields = (
            "member",
            "member_size_bytes",
            "member_compressed_size_bytes",
            "member_crc32",
        )
        if any(source_identity.get(name) != fingerprint.get(name) for name in fields):
            raise ValueError("catalog and workload source identities differ")
    dense = np.zeros(int(original.max()) + 1, dtype=np.int32)
    dense[original] = np.arange(1, len(original) + 1, dtype=np.int32)
    return CatalogMapping(
        original_item_ids=original,
        base_item_frequencies=frequencies,
        prediction_rows=prediction_rows,
        context_rows=context_rows,
        dense_map=dense,
        metadata=metadata,
    )


def _length_cache_identity(fingerprint: dict) -> dict:
    return {
        "protocol": LENGTH_CACHE_PROTOCOL,
        "phase": "pass_a_full_user_lengths",
        "source": fingerprint,
        "length_basis": "all raw QK exposure rows",
    }


def build_user_lengths(
    config: FoundationConfig,
    fingerprint: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    counts = np.zeros(0, dtype=np.int32)
    rows = 0
    started = time.perf_counter()
    for chunk_index, chunk in enumerate(read_qk_chunks(config), start=1):
        users = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
        if len(users):
            chunk_counts = np.bincount(users)
            counts = grow_vector(counts, len(chunk_counts))
            counts[: len(chunk_counts)] += chunk_counts.astype(
                counts.dtype,
                copy=False,
            )
        rows += len(chunk)
        print(
            f"phase=pass_a chunks={chunk_index} rows={rows:,} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    user_ids = np.flatnonzero(counts).astype(np.int64)
    lengths = counts[user_ids].astype(np.int32)
    metadata = {
        **_length_cache_identity(fingerprint),
        "scientific_result": False,
        "source_rows_scanned": rows,
        "users": len(user_ids),
        "user_ids_sha256": array_sha256(user_ids),
        "length_sum": int(lengths.astype(np.int64).sum()),
        "minimum_length": int(lengths.min()),
        "maximum_length": int(lengths.max()),
    }
    _save_npz(
        config.length_cache,
        {"user_ids": user_ids, "raw_lengths": lengths},
        metadata,
    )
    return user_ids, lengths, metadata


def load_or_build_user_lengths(
    config: FoundationConfig,
    fingerprint: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    expected = _length_cache_identity(fingerprint)
    if config.length_cache.exists() and not config.refresh_lengths:
        arrays, metadata = _load_npz(config.length_cache)
        if all(metadata.get(name) == value for name, value in expected.items()):
            return (
                np.asarray(arrays["user_ids"], dtype=np.int64),
                np.asarray(arrays["raw_lengths"], dtype=np.int32),
                metadata,
            )
    return build_user_lengths(config, fingerprint)


def load_upstream_training_users(
    config: FoundationConfig,
    fingerprint: dict,
) -> tuple[np.ndarray, dict]:
    if config.upstream_prepared is None:
        return np.empty(0, dtype=np.int64), {
            "enabled": False,
            "excluded_users": 0,
            "definition": "no inherited upstream model initialization",
        }
    arrays, metadata = _load_npz(config.upstream_prepared)
    if "original_user_ids" not in arrays:
        raise ValueError("upstream prepared data lacks original_user_ids")
    source_identity = metadata.get("source", {})
    fields = (
        "member",
        "member_size_bytes",
        "member_compressed_size_bytes",
        "member_crc32",
    )
    if any(
        source_identity.get(name) != fingerprint.get(name)
        for name in fields
    ):
        raise ValueError("upstream prepared data source differs")
    users = np.asarray(arrays["original_user_ids"], dtype=np.int64)
    if (
        len(users) == 0
        or np.any(users < 0)
        or len(np.unique(users)) != len(users)
    ):
        raise ValueError("upstream training user ids are invalid")
    selected_users = metadata.get("selected_users")
    if selected_users is not None and int(selected_users) != len(users):
        raise ValueError("upstream prepared user count differs")
    return users, {
        "enabled": True,
        "path": str(config.upstream_prepared),
        "file_sha256": file_sha256(config.upstream_prepared),
        "source_protocol": metadata.get("protocol"),
        "excluded_users": len(users),
        "excluded_user_ids_sha256": array_sha256(users),
        "array": "original_user_ids",
        "scope": (
            "all users inherited by the upstream theta0 model "
            "initialization"
        ),
    }


def allocate_roles(
    user_ids: np.ndarray,
    raw_lengths: np.ndarray,
    config: FoundationConfig,
    upstream_training_user_ids: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    if len(user_ids) != len(raw_lengths):
        raise ValueError("user ids and lengths differ")
    excluded = (
        np.empty(0, dtype=np.int64)
        if upstream_training_user_ids is None
        else np.asarray(upstream_training_user_ids, dtype=np.int64)
    )
    if (
        np.any(excluded < 0)
        or len(np.unique(excluded)) != len(excluded)
    ):
        raise ValueError("upstream training user ids are invalid")
    eligible_before = raw_lengths >= config.minimum_events
    excluded_dense = np.zeros(
        max(
            int(user_ids.max()) + 1,
            int(excluded.max()) + 1 if len(excluded) else 0,
        ),
        dtype=np.bool_,
    )
    excluded_dense[excluded] = True
    excluded_in_source = excluded_dense[user_ids]
    eligible_mask = eligible_before & ~excluded_in_source
    eligible_ids = user_ids[eligible_mask]
    eligible_lengths = raw_lengths[eligible_mask]
    ordered_ids = stable_user_order(eligible_ids, config.hash_salt)
    length_by_user = np.zeros(int(user_ids.max()) + 1, dtype=np.int32)
    length_by_user[user_ids] = raw_lengths
    theta12_preferred = ordered_ids[
        length_by_user[ordered_ids] >= config.theta12_minimum_events
    ]
    theta12_required_events = (
        config.history_horizon + config.append_events
    )
    theta12_backfill_candidates = ordered_ids[
        (length_by_user[ordered_ids] >= theta12_required_events)
        & (length_by_user[ordered_ids] < config.theta12_minimum_events)
    ]
    preferred_take = min(
        len(theta12_preferred),
        config.theta12_users,
    )
    missing = config.theta12_users - preferred_take
    if (
        config.require_all_roles
        and len(theta12_backfill_candidates) < missing
    ):
        raise ValueError("insufficient theta12 fixed-edge candidates")
    backfill_take = (
        missing
        if config.require_all_roles
        else min(missing, len(theta12_backfill_candidates))
    )
    theta12 = np.concatenate(
        [
            theta12_preferred[:preferred_take],
            theta12_backfill_candidates[:backfill_take],
        ]
    )
    reserved = np.zeros(len(length_by_user), dtype=np.bool_)
    reserved[theta12] = True
    remaining = ordered_ids[~reserved[ordered_ids]]
    role_sizes = (
        ("theta01", config.theta01_users),
        ("fit", config.fit_users),
        ("profile", config.profile_users),
        ("qualification", config.qualification_users),
        ("final", config.final_users),
    )
    roles: dict[str, np.ndarray] = {"theta12": theta12}
    cursor = 0
    for name, requested in role_sizes:
        available = max(0, len(remaining) - cursor)
        take = requested if config.require_all_roles else min(requested, available)
        if available < take:
            raise ValueError(f"insufficient users for role {name}")
        roles[name] = remaining[cursor : cursor + take]
        cursor += take
    all_selected = np.concatenate(list(roles.values()))
    if len(np.unique(all_selected)) != len(all_selected):
        raise ValueError("post-base role assignment is not disjoint")
    role_audit = {
        name: {
            "count": len(values),
            "user_ids_sha256": array_sha256(values),
            "minimum_raw_length": int(length_by_user[values].min())
            if len(values)
            else None,
            "maximum_raw_length": int(length_by_user[values].max())
            if len(values)
            else None,
        }
        for name, values in roles.items()
    }
    audit = {
        "eligible_users": int(len(eligible_ids)),
        "eligible_user_ids_sha256": array_sha256(eligible_ids),
        "eligible_users_before_upstream_exclusion": int(
            np.count_nonzero(eligible_before)
        ),
        "upstream_training_exclusion": {
            "excluded_users_declared": len(excluded),
            "excluded_user_ids_sha256": array_sha256(excluded),
            "excluded_users_present_in_source": int(
                np.count_nonzero(excluded_in_source)
            ),
            "excluded_minimum_event_eligible_users": int(
                np.count_nonzero(eligible_before & excluded_in_source)
            ),
            "selected_role_overlap": int(
                np.count_nonzero(np.isin(all_selected, excluded))
            ),
            "applied_before_stable_role_assignment": True,
        },
        "eligible_minimum_events": config.minimum_events,
        "theta12_candidate_users": int(
            np.count_nonzero(
                eligible_lengths >= theta12_required_events
            )
        ),
        "theta12_required_events": theta12_required_events,
        "theta12_minimum_events": config.theta12_minimum_events,
        "theta12_preferred_minimum_events": (
            config.theta12_minimum_events
        ),
        "theta12_preferred_candidate_users": len(theta12_preferred),
        "theta12_preferred_selected_users": preferred_take,
        "theta12_backfill_candidate_users": len(
            theta12_backfill_candidates
        ),
        "theta12_backfill_selected_users": backfill_take,
        "assignment_order": [
            "theta12",
            "theta01",
            "fit",
            "profile",
            "qualification",
            "final",
        ],
        "roles": role_audit,
        "post_base_roles_pairwise_disjoint": True,
        "base_builder": {
            "role": "common_upstream",
            "user_exclusion": False,
            "allowed_data": "base-period histories only",
            "forbidden_data": "post-base update and final windows",
        },
    }
    return roles, audit


def layout_from_lengths(
    raw_lengths: np.ndarray,
    config: FoundationConfig,
) -> dict[str, np.ndarray]:
    boundary = np.minimum(
        raw_lengths.astype(np.int64),
        config.history_horizon,
    )
    old_start = np.maximum(0, boundary - config.history_horizon)
    old_end = boundary - config.append_events
    target_start = np.maximum(0, boundary - config.target_horizon)
    target_end = boundary
    retained_start = np.maximum(old_start, target_start)
    retained_end = np.minimum(old_end, target_end)
    old_length = old_end - old_start
    target_length = target_end - target_start
    retained_length = np.maximum(0, retained_end - retained_start)
    evicted_length = old_length - retained_length
    append_length = target_length - retained_length
    if np.any(old_length < 0) or np.any(target_length < 1):
        raise ValueError("invalid HET extent lengths")
    old_valid_bytes = (
        old_length.astype(np.int64) * config.kv_bytes_per_token
    )
    target_valid_bytes = (
        target_length.astype(np.int64) * config.kv_bytes_per_token
    )
    hom_allocated_bytes = np.full(
        len(raw_lengths),
        config.target_horizon * config.kv_bytes_per_token,
        dtype=np.int64,
    )
    return {
        "boundary_b": boundary.astype(np.int16),
        "old_start": old_start.astype(np.int16),
        "old_end": old_end.astype(np.int16),
        "target_start": target_start.astype(np.int16),
        "target_end": target_end.astype(np.int16),
        "old_length": old_length.astype(np.int16),
        "target_length": target_length.astype(np.int16),
        "retained_length": retained_length.astype(np.int16),
        "evicted_length": evicted_length.astype(np.int16),
        "append_length": append_length.astype(np.int16),
        "hom_old_left_padding": (
            config.target_horizon - old_length
        ).astype(np.int16),
        "hom_target_left_padding": (
            config.target_horizon - target_length
        ).astype(np.int16),
        "hom_old_allocated_length": np.full(
            len(raw_lengths),
            config.target_horizon,
            dtype=np.int16,
        ),
        "hom_target_allocated_length": np.full(
            len(raw_lengths),
            config.target_horizon,
            dtype=np.int16,
        ),
        "het_old_valid_kv_bytes": old_valid_bytes,
        "het_target_valid_kv_bytes": target_valid_bytes,
        "hom_old_allocated_kv_bytes": hom_allocated_bytes.copy(),
        "hom_target_allocated_kv_bytes": hom_allocated_bytes,
    }


def capacity_cohorts(
    target_lengths: np.ndarray,
    config: FoundationConfig,
) -> dict[str, np.ndarray]:
    target_bytes = (
        target_lengths.astype(np.int64) * config.kv_bytes_per_token
    )
    cumulative = np.cumsum(target_bytes, dtype=np.int64)
    thresholds = config.capacity_bytes
    ends = np.searchsorted(cumulative, thresholds, side="left") + 1
    reached = ends <= len(target_lengths)
    clipped = np.minimum(ends, len(target_lengths))
    actual = np.zeros(len(thresholds), dtype=np.int64)
    nonzero = clipped > 0
    actual[nonzero] = cumulative[clipped[nonzero] - 1]
    overshoot = np.where(reached, actual - thresholds, -1)
    if config.require_all_capacities and not np.all(reached):
        missing = [
            config.capacity_gib[index]
            for index in np.flatnonzero(~reached)
        ]
        raise ValueError(f"final universe cannot reach capacity points {missing}")
    return {
        "record_target_valid_kv_bytes": target_bytes,
        "record_cumulative_target_valid_kv_bytes": cumulative,
        "capacity_target_bytes": thresholds,
        "capacity_prefix_records": clipped.astype(np.int32),
        "capacity_actual_valid_kv_bytes": actual,
        "capacity_overshoot_bytes": overshoot,
        "capacity_reached": reached.astype(np.uint8),
    }


def _per_rank_values(
    owners: np.ndarray,
    values: np.ndarray,
    ranks: int,
    end: int,
) -> np.ndarray:
    totals = np.zeros(ranks, dtype=np.int64)
    np.add.at(
        totals,
        owners[:end].astype(np.int64, copy=False),
        values[:end].astype(np.int64, copy=False),
    )
    return totals


def _per_rank_counts(
    owners: np.ndarray,
    ranks: int,
    end: int,
) -> np.ndarray:
    return np.bincount(
        owners[:end].astype(np.int64, copy=False),
        minlength=ranks,
    ).astype(np.int64)


def build_rank_ledgers(
    user_ids: np.ndarray,
    layout: dict[str, np.ndarray],
    cohorts: dict[str, np.ndarray],
    config: FoundationConfig,
) -> tuple[dict[str, np.ndarray], dict]:
    owners_by_ranks = {
        ranks: stable_owner_ranks(user_ids, config.hash_salt, ranks)
        for ranks in (2, 4)
    }
    byte_fields = (
        "het_old_valid_kv_bytes",
        "het_target_valid_kv_bytes",
        "hom_old_allocated_kv_bytes",
        "hom_target_allocated_kv_bytes",
    )
    arrays: dict[str, np.ndarray] = {
        "owner_rank_2": owners_by_ranks[2],
        "owner_rank_4": owners_by_ranks[4],
    }
    total = {
        field: int(layout[field].astype(np.int64).sum())
        for field in byte_fields
    }
    for field, value in total.items():
        arrays[f"total_{field}"] = np.asarray(value, dtype=np.int64)
    summary: dict[str, object] = {
        "owner_rule": (
            "splitmix64(original_user_id xor "
            "sha256(hash_salt:owner)[0:8]) modulo rank_count"
        ),
        "full_universe": {"records": len(user_ids), **total},
        "ranks": {},
        "capacity_cohorts": [],
    }
    for ranks, owners in owners_by_ranks.items():
        counts = _per_rank_counts(owners, ranks, len(user_ids))
        rank_values = {
            field: _per_rank_values(
                owners,
                layout[field],
                ranks,
                len(user_ids),
            )
            for field in byte_fields
        }
        arrays[f"owner{ranks}_record_count"] = counts
        for field, values in rank_values.items():
            arrays[f"owner{ranks}_{field}"] = values
        summary["ranks"][str(ranks)] = _rank_summary(
            counts,
            rank_values,
        )
        summary["ranks"][str(ranks)][
            "owner_rank_ids_sha256"
        ] = array_sha256(owners)
    for cohort_index, end_value in enumerate(
        cohorts["capacity_prefix_records"]
    ):
        end = int(end_value)
        cohort_summary: dict[str, object] = {
            "target_gib": config.capacity_gib[cohort_index],
            "prefix_records": end,
            "ranks": {},
        }
        for ranks, owners in owners_by_ranks.items():
            counts = _per_rank_counts(owners, ranks, end)
            rank_values = {
                field: _per_rank_values(
                    owners,
                    layout[field],
                    ranks,
                    end,
                )
                for field in byte_fields
            }
            arrays.setdefault(
                f"capacity_owner{ranks}_record_count",
                np.empty(
                    (len(config.capacity_gib), ranks),
                    dtype=np.int64,
                ),
            )[cohort_index] = counts
            for field, values in rank_values.items():
                arrays.setdefault(
                    f"capacity_owner{ranks}_{field}",
                    np.empty(
                        (len(config.capacity_gib), ranks),
                        dtype=np.int64,
                    ),
                )[cohort_index] = values
            cohort_summary["ranks"][str(ranks)] = _rank_summary(
                counts,
                rank_values,
            )
        summary["capacity_cohorts"].append(cohort_summary)
    return arrays, summary


def _imbalance(values: np.ndarray) -> dict[str, float | int]:
    mean = float(np.mean(values))
    return {
        "max_over_mean": float(values.max() / mean) if mean else 1.0,
        "max_minus_min": int(values.max() - values.min()),
    }


def _rank_summary(
    counts: np.ndarray,
    rank_values: dict[str, np.ndarray],
) -> dict:
    return {
        "record_count": [int(value) for value in counts],
        **{
            field: [int(value) for value in values]
            for field, values in rank_values.items()
        },
        "imbalance": {
            "record_count": _imbalance(counts),
            **{
                field: _imbalance(values)
                for field, values in rank_values.items()
            },
        },
    }


def xp_embedding_ledger(
    catalog: CatalogMapping,
    config: FoundationConfig,
) -> dict:
    physical_rows = catalog.rows + config.embedding_padding_rows
    global_bytes = (
        physical_rows
        * config.embedding_width
        * config.embedding_element_bytes
    )
    per_rank: dict[str, dict] = {}
    for ranks in (2, 4):
        base = physical_rows // ranks
        remainder = physical_rows % ranks
        row_counts = [
            base + int(rank < remainder)
            for rank in range(ranks)
        ]
        byte_counts = [
            rows
            * config.embedding_width
            * config.embedding_element_bytes
            for rows in row_counts
        ]
        per_rank[str(ranks)] = {
            "row_counts": row_counts,
            "bytes": byte_counts,
        }
    projection_bytes = (
        config.embedding_width
        * config.model_hidden_size
        * config.embedding_element_bytes
    )
    dense_bytes = (
        config.dense_parameter_count * config.embedding_element_bytes
    )
    fixed_state_bytes = global_bytes + dense_bytes
    row_bytes = config.embedding_width * config.embedding_element_bytes
    required_active_rows = (
        max(
            0,
            config.single_card_torch_allocatable_bytes - dense_bytes,
        )
        // row_bytes
        + 1
    )
    return {
        "semantic_rows": catalog.rows,
        "padding_rows": config.embedding_padding_rows,
        "physical_rows": physical_rows,
        "width": config.embedding_width,
        "element_bytes": config.embedding_element_bytes,
        "global_bytes": global_bytes,
        "per_rank": per_rank,
        "owner_projection_shape": [
            config.embedding_width,
            config.model_hidden_size,
        ],
        "owner_projection_fp32_bytes": projection_bytes,
        "dense_parameter_count_including_projection": (
            config.dense_parameter_count
        ),
        "dense_fp32_bytes_including_projection": dense_bytes,
        "global_embedding_plus_dense_bytes": fixed_state_bytes,
        "single_card_torch_allocatable_bytes": (
            config.single_card_torch_allocatable_bytes
        ),
        "full_replication_capacity_admitted": (
            fixed_state_bytes
            <= config.single_card_torch_allocatable_bytes
        ),
        "minimum_optimizer_active_rows_to_force_sharding": (
            required_active_rows
        ),
        "minimum_optimizer_active_fraction_of_semantic_rows": (
            required_active_rows / catalog.rows
        ),
        "optimizer_active_gate": "pending",
    }


def materialize_histories(
    config: FoundationConfig,
    final_user_ids: np.ndarray,
    final_raw_lengths: np.ndarray,
    catalog: CatalogMapping,
) -> tuple[dict[str, np.ndarray], dict]:
    layout = layout_from_lengths(final_raw_lengths, config)
    boundary = layout["boundary_b"].astype(np.int64)
    offsets = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.cumsum(boundary, dtype=np.int64),
        ]
    )
    item_idx = np.empty(int(offsets[-1]), dtype=np.int32)
    filled = np.zeros(int(offsets[-1]), dtype=np.bool_)
    user_to_record = np.zeros(int(final_user_ids.max()) + 1, dtype=np.int32)
    user_to_record[final_user_ids] = np.arange(
        1,
        len(final_user_ids) + 1,
        dtype=np.int32,
    )
    raw_seen = np.zeros(0, dtype=np.int32)
    rows = 0
    selected_rows = 0
    fallback_rows = 0
    started = time.perf_counter()
    for chunk_index, chunk in enumerate(read_qk_chunks(config), start=1):
        users = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
        items = chunk["item_id"].to_numpy(dtype=np.int64, copy=False)
        positions, raw_seen = consume_user_positions(users, raw_seen)
        in_range = (users >= 0) & (users < len(user_to_record))
        records = np.full(len(users), -1, dtype=np.int64)
        records[in_range] = user_to_record[users[in_range]].astype(
            np.int64,
            copy=False,
        ) - 1
        selected = records >= 0
        selected &= positions < boundary[np.maximum(records, 0)]
        if selected.any():
            selected_records = records[selected]
            selected_positions = positions[selected]
            destinations = offsets[selected_records] + selected_positions
            mapped, fallback = catalog.map_items(items[selected])
            if np.any(filled[destinations]):
                raise ValueError("duplicate final-history positions")
            item_idx[destinations] = mapped
            filled[destinations] = True
            selected_rows += len(mapped)
            fallback_rows += int(np.count_nonzero(fallback))
        rows += len(chunk)
        print(
            f"phase=pass_b chunks={chunk_index} rows={rows:,} "
            f"selected={selected_rows:,} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if selected_rows == len(item_idx):
            break
    if not np.all(filled):
        raise ValueError(
            f"materialized {np.count_nonzero(filled)} of {len(filled)} "
            "final-history rows"
        )
    arrays = {
        "record_user_ids": final_user_ids.astype(np.int64, copy=False),
        "record_raw_lengths": final_raw_lengths.astype(
            np.int32,
            copy=False,
        ),
        "history_offsets": offsets,
        "history_item_idx": item_idx,
        **layout,
    }
    audit = {
        "source_rows_scanned_until_complete": rows,
        "history_rows": len(item_idx),
        "history_item_idx_sha256": array_sha256(item_idx),
        "stream_only_fallback_rows": fallback_rows,
        "stream_only_fallback_fraction": fallback_rows / len(item_idx),
    }
    return arrays, audit


def semantic_request_union(
    histories: dict[str, np.ndarray],
    catalog: CatalogMapping,
) -> tuple[dict[str, np.ndarray], dict]:
    offsets = histories["history_offsets"].astype(np.int64, copy=False)
    target_start = histories["target_start"].astype(np.int64, copy=False)
    target_end = histories["target_end"].astype(np.int64, copy=False)
    items = histories["history_item_idx"]
    target_parts = [
        items[offsets[index] + target_start[index] : offsets[index] + target_end[index]]
        for index in range(len(target_start))
    ]
    target_items = np.concatenate(target_parts)
    unique_items = np.unique(target_items).astype(np.int32, copy=False)
    eligible_catalog = catalog.base_item_frequencies > 0
    eligible = eligible_catalog[unique_items.astype(np.int64) - 1]
    identity = array_sha256(unique_items)
    audit = {
        "definition": (
            "unique mapped item rows from all-exact valid targets at the "
            "same frozen trace boundary for both model edges"
        ),
        "model_edges": ["theta0_to_theta1", "theta1_to_theta2"],
        "target_tokens_per_edge": len(target_items),
        "target_tokens_across_edges": 2 * len(target_items),
        "unique_rows": len(unique_items),
        "unique_rows_sha256": identity,
        "edge_unique_rows_sha256": {
            "theta0_to_theta1": identity,
            "theta1_to_theta2": identity,
        },
        "eligible_for_update_rows": int(np.count_nonzero(eligible)),
        "all_request_rows_eligible_for_update": bool(np.all(eligible)),
        "optimizer_active_gate": "pending",
        "catalog_frequency_is_optimizer_activity": False,
    }
    arrays = {
        "semantic_request_union_item_idx": unique_items,
        "semantic_request_union_eligible_for_update": eligible.astype(
            np.uint8
        ),
    }
    return arrays, audit


def _lengths_for_users(
    all_user_ids: np.ndarray,
    all_lengths: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    dense = np.zeros(int(all_user_ids.max()) + 1, dtype=np.int32)
    dense[all_user_ids] = all_lengths
    return dense[selected]


def _quantiles(values: np.ndarray) -> dict[str, float]:
    points = (0, 0.25, 0.5, 0.75, 0.95, 0.99, 1)
    return {
        str(point): float(np.quantile(values, point))
        for point in points
    }


def role_document(
    config: FoundationConfig,
    fingerprint: dict,
    roles: dict[str, np.ndarray],
    audit: dict,
) -> dict:
    return {
        "protocol": PROTOCOL,
        "scientific_result": False,
        "source": fingerprint,
        "hash_salt": config.hash_salt,
        "role_assignment": audit,
        "roles": {
            name: {
                "count": len(values),
                "user_ids_sha256": array_sha256(values),
                "user_ids": [int(value) for value in values],
            }
            for name, values in roles.items()
        },
    }


def run(config: FoundationConfig, audit_only: bool = False) -> dict:
    validate_config(config)
    fingerprint = source_fingerprint(config)
    catalog = load_catalog(config.catalog_cache, fingerprint)
    user_ids, raw_lengths, length_metadata = load_or_build_user_lengths(
        config,
        fingerprint,
    )
    upstream_users, upstream_audit = load_upstream_training_users(
        config,
        fingerprint,
    )
    roles, role_audit = allocate_roles(
        user_ids,
        raw_lengths,
        config,
        upstream_users,
    )
    role_audit["upstream_training_exclusion"].update(upstream_audit)
    if (
        role_audit["upstream_training_exclusion"][
            "selected_role_overlap"
        ]
        != 0
    ):
        raise ValueError("upstream training users leaked into roles")
    role_value = role_document(
        config,
        fingerprint,
        roles,
        role_audit,
    )
    _write_json(config.roles, role_value)
    final_lengths = _lengths_for_users(
        user_ids,
        raw_lengths,
        roles["final"],
    )
    layout = layout_from_lengths(final_lengths, config)
    cohorts = capacity_cohorts(layout["target_length"], config)
    rank_arrays, rank_ledger = build_rank_ledgers(
        roles["final"],
        layout,
        cohorts,
        config,
    )
    eligible_rows = catalog.base_item_frequencies > 0
    summary = {
        "protocol": PROTOCOL,
        "scientific_result": False,
        "formal_protocol": False,
        "formal_result": False,
        "status": "audit_only" if audit_only else "materializing",
        "source": fingerprint,
        "catalog": {
            "cache": str(config.catalog_cache),
            "base_entity_rows": catalog.rows,
            "base_entity_item_ids_sha256": catalog.metadata.get(
                "base_entity_item_ids_sha256",
                array_sha256(catalog.original_item_ids),
            ),
            "prediction_rows": catalog.prediction_rows,
            "context_rows": catalog.context_rows,
            "eligible_for_update_definition": "base_item_frequency > 0",
            "eligible_for_update_rows": int(
                np.count_nonzero(eligible_rows)
            ),
            "eligible_for_update_row_ids_sha256": array_sha256(
                np.flatnonzero(eligible_rows) + 1
            ),
            "catalog_frequency_is_optimizer_activity": False,
            "optimizer_active_gate": "pending",
        },
        "xp_embedding": xp_embedding_ledger(catalog, config),
        "pass_a": {
            "cache": str(config.length_cache),
            "users": len(user_ids),
            "source_rows_scanned": length_metadata[
                "source_rows_scanned"
            ],
            "minimum_events_eligible_users": role_audit[
                "eligible_users"
            ],
            "theta12_candidate_users": role_audit[
                "theta12_candidate_users"
            ],
        },
        "roles": role_audit,
        "workload": {
            "identity": "X-QK-HET",
            "record_count": len(roles["final"]),
            "record_user_ids_sha256": array_sha256(roles["final"]),
            "raw_length_quantiles": _quantiles(final_lengths),
            "old_length_quantiles": _quantiles(layout["old_length"]),
            "target_length_quantiles": _quantiles(
                layout["target_length"]
            ),
            "retained_length_quantiles": _quantiles(
                layout["retained_length"]
            ),
            "evicted_length_quantiles": _quantiles(
                layout["evicted_length"]
            ),
            "append_length_quantiles": _quantiles(
                layout["append_length"]
            ),
            "target_512_saturation_fraction": float(
                np.mean(layout["target_length"] == config.target_horizon)
            ),
            "het_boundary": (
                "b=min(n,544); old=[max(0,b-544),b-32); "
                "target=[max(0,b-512),b)"
            ),
            "hom_control": (
                "same records and valid histories, masked left padding "
                "to 512 old and target slots"
            ),
            "kv_bytes_per_valid_token": config.kv_bytes_per_token,
        },
        "capacity_cohorts": [
            {
                "target_gib": config.capacity_gib[index],
                "target_bytes": int(cohorts["capacity_target_bytes"][index]),
                "prefix_records": int(
                    cohorts["capacity_prefix_records"][index]
                ),
                "actual_valid_kv_bytes": int(
                    cohorts["capacity_actual_valid_kv_bytes"][index]
                ),
                "overshoot_bytes": int(
                    cohorts["capacity_overshoot_bytes"][index]
                ),
                "reached": bool(cohorts["capacity_reached"][index]),
            }
            for index in range(len(config.capacity_gib))
        ],
        "byte_and_owner_ledger": rank_ledger,
        "semantic_request_union": {
            "status": "pending_pass_b",
            "definition": (
                "all-exact valid target item rows at one trace boundary "
                "for theta0_to_theta1 and theta1_to_theta2"
            ),
        },
        "gates": {
            "foundation_workload": "pending_pass_b"
            if audit_only
            else "running",
            "optimizer_active": "pending",
            "benchmark_qualification": "not_run",
        },
        "artifacts": {
            "roles": str(config.roles),
            "workload": str(config.output),
            "summary": str(config.summary),
        },
    }
    if audit_only:
        _write_json(config.summary, summary)
        return summary
    histories, materialization_audit = materialize_histories(
        config,
        roles["final"],
        final_lengths,
        catalog,
    )
    request_arrays, request_audit = semantic_request_union(
        histories,
        catalog,
    )
    arrays = {
        **histories,
        **cohorts,
        **rank_arrays,
        **request_arrays,
    }
    metadata = {
        "protocol": PROTOCOL,
        "scientific_result": False,
        "formal_protocol": False,
        "formal_result": False,
        "dataset": "tenrec-qk",
        "workload": "X-QK-HET",
        "source": fingerprint,
        "catalog_cache": str(config.catalog_cache),
        "catalog_identity": summary["catalog"],
        "hash_salt": config.hash_salt,
        "role_identity": {
            name: details["user_ids_sha256"]
            for name, details in role_audit["roles"].items()
        },
        "upstream_training_exclusion": role_audit[
            "upstream_training_exclusion"
        ],
        "layout": summary["workload"],
        "capacity_gib": list(config.capacity_gib),
        "xp_embedding": summary["xp_embedding"],
        "byte_and_owner_ledger": rank_ledger,
        "hom_descriptor": {
            "same_record": True,
            "allocated_old_slots": config.target_horizon,
            "allocated_target_slots": config.target_horizon,
            "padding_side": "left",
            "padding_masked": True,
            "capacity_uses_allocated_bytes": True,
        },
        "semantic_request_union": request_audit,
        "optimizer_active_gate": "pending",
    }
    _save_npz(config.output, arrays, metadata)
    summary["status"] = "materialized"
    summary["pass_b"] = materialization_audit
    summary["semantic_request_union"] = request_audit
    summary["gates"]["foundation_workload"] = "pass"
    summary["artifact_bytes"] = config.output.stat().st_size
    _write_json(config.summary, summary)
    return summary
