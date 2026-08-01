from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL = "evokv_qk_xp_base_row_coverage_development_v0"


@dataclass(frozen=True)
class CoverageConfig:
    source: Path
    member: str
    catalog_cache: Path
    user_length_cache: Path
    cache_dir: Path
    output: Path
    summary: Path
    base_prefix: int = 64
    chunk_size: int = 2_000_000
    checkpoint_every_chunks: int = 16
    derive_user_block: int = 100_000
    refresh: bool = False

    @property
    def matrix_path(self) -> Path:
        return self.cache_dir / "base_mapped_rows.npy"

    @property
    def working_seen_path(self) -> Path:
        return self.cache_dir / "working_seen_counts.npy"

    @property
    def state_path(self) -> Path:
        return self.cache_dir / "scan_state.json"

    def seen_checkpoint_path(self, chunk: int) -> Path:
        return self.cache_dir / f"seen_counts_chunk_{chunk:06d}.npy"


@dataclass(frozen=True)
class Catalog:
    original_item_ids: np.ndarray
    frequencies: np.ndarray
    dense_map: np.ndarray
    metadata: dict
    file_sha256: str

    @property
    def rows(self) -> int:
        return len(self.original_item_ids)


def validate_config(config: CoverageConfig) -> None:
    if min(
        config.base_prefix,
        config.chunk_size,
        config.checkpoint_every_chunks,
        config.derive_user_block,
    ) < 1:
        raise ValueError("coverage builder dimensions must be positive")


def source_fingerprint(config: CoverageConfig) -> dict:
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    value = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def artifact_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _source_matches(left: dict, right: dict) -> bool:
    fields = (
        "member",
        "member_size_bytes",
        "member_compressed_size_bytes",
        "member_crc32",
    )
    return all(left.get(name) == right.get(name) for name in fields)


def load_catalog(
    config: CoverageConfig,
    fingerprint: dict,
) -> Catalog:
    arrays, metadata = _load_npz(config.catalog_cache)
    required = {
        "base_entity_original_item_ids",
        "base_item_frequencies",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"catalog arrays are missing: {sorted(missing)}")
    original = np.asarray(
        arrays["base_entity_original_item_ids"],
        dtype=np.int64,
    )
    frequencies = np.asarray(
        arrays["base_item_frequencies"],
        dtype=np.int64,
    )
    if (
        len(original) == 0
        or len(original) != len(frequencies)
        or len(np.unique(original)) != len(original)
        or np.any(frequencies < 1)
    ):
        raise ValueError("base entity catalog is invalid")
    if int(metadata.get("base_prefix_raw_events", -1)) != config.base_prefix:
        raise ValueError("catalog base prefix differs")
    if not _source_matches(metadata.get("source", {}), fingerprint):
        raise ValueError("catalog source differs")
    dense = np.zeros(int(original.max()) + 1, dtype=np.int32)
    dense[original] = np.arange(1, len(original) + 1, dtype=np.int32)
    return Catalog(
        original_item_ids=original,
        frequencies=frequencies,
        dense_map=dense,
        metadata=metadata,
        file_sha256=file_sha256(config.catalog_cache),
    )


def load_user_capacity(
    config: CoverageConfig,
    fingerprint: dict,
) -> tuple[int, dict]:
    arrays, metadata = _load_npz(config.user_length_cache)
    if not _source_matches(metadata.get("source", {}), fingerprint):
        raise ValueError("user-length cache source differs")
    user_ids = np.asarray(arrays["user_ids"], dtype=np.int64)
    if len(user_ids) == 0 or np.any(user_ids < 0):
        raise ValueError("user-length cache is invalid")
    return int(user_ids.max()) + 1, {
        "path": str(config.user_length_cache),
        "file_sha256": file_sha256(config.user_length_cache),
        "users": len(user_ids),
        "user_ids_sha256": metadata.get("user_ids_sha256"),
        "use": "allocation capacity only",
        "post_base_values_used_for_pairs": False,
    }


def read_qk_chunks(config: CoverageConfig) -> Iterator[pd.DataFrame]:
    with (
        zipfile.ZipFile(config.source) as archive,
        archive.open(config.member) as stream,
    ):
        yield from pd.read_csv(
            stream,
            usecols=["user_id", "item_id"],
            dtype={"user_id": "int32", "item_id": "int32"},
            chunksize=config.chunk_size,
        )


def consume_user_positions(
    users: np.ndarray,
    seen_counts: np.ndarray,
) -> np.ndarray:
    if len(users) == 0:
        return np.empty(0, dtype=np.int64)
    if int(users.min()) < 0 or int(users.max()) >= len(seen_counts):
        raise ValueError("QK user id exceeds cached user capacity")
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
    return positions


def _cache_identity(
    config: CoverageConfig,
    fingerprint: dict,
    catalog: Catalog,
    user_capacity: int,
) -> dict:
    return {
        "protocol": PROTOCOL,
        "phase": "base_prefix_scan_cache",
        "source": fingerprint,
        "catalog_file_sha256": catalog.file_sha256,
        "catalog_row_ids_sha256": catalog.metadata.get(
            "base_entity_item_ids_sha256",
            array_sha256(catalog.original_item_ids),
        ),
        "base_prefix": config.base_prefix,
        "user_capacity": user_capacity,
        "matrix_shape": [user_capacity, config.base_prefix],
        "matrix_dtype": "uint32",
    }


def _state_matches(state: dict, identity: dict) -> bool:
    return all(state.get(name) == value for name, value in identity.items())


def _initialize_cache(
    config: CoverageConfig,
    identity: dict,
    user_capacity: int,
) -> tuple[np.memmap, np.memmap, dict]:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        config.matrix_path,
        mode="w+",
        dtype=np.uint32,
        shape=(user_capacity, config.base_prefix),
    )
    matrix.fill(0)
    matrix.flush()
    seen = np.lib.format.open_memmap(
        config.working_seen_path,
        mode="w+",
        dtype=np.int32,
        shape=(user_capacity,),
    )
    seen.fill(0)
    seen.flush()
    checkpoint = config.seen_checkpoint_path(0)
    frozen = np.lib.format.open_memmap(
        checkpoint,
        mode="w+",
        dtype=np.int32,
        shape=(user_capacity,),
    )
    frozen[:] = seen
    frozen.flush()
    del frozen
    state = {
        **identity,
        "scientific_result": False,
        "complete": False,
        "completed_chunks": 0,
        "source_rows_scanned": 0,
        "base_rows_retained": 0,
        "post_base_rows_ignored": 0,
        "seen_checkpoint": str(checkpoint),
    }
    _write_json(config.state_path, state)
    return matrix, seen, state


def _open_cache(
    config: CoverageConfig,
    identity: dict,
    user_capacity: int,
) -> tuple[np.memmap, np.memmap, dict]:
    if config.refresh or not config.state_path.exists():
        return _initialize_cache(
            config,
            identity,
            user_capacity,
        )
    state = json.loads(config.state_path.read_text())
    if not _state_matches(state, identity):
        raise ValueError("base-row cache identity differs; use --refresh")
    required = (
        config.matrix_path,
        config.working_seen_path,
        Path(state["seen_checkpoint"]),
    )
    if any(not path.is_file() for path in required):
        raise ValueError("base-row cache is incomplete; use --refresh")
    matrix = np.load(config.matrix_path, mmap_mode="r+")
    seen = np.load(config.working_seen_path, mmap_mode="r+")
    frozen = np.load(Path(state["seen_checkpoint"]), mmap_mode="r")
    if (
        matrix.shape != (user_capacity, config.base_prefix)
        or seen.shape != (user_capacity,)
        or frozen.shape != (user_capacity,)
    ):
        raise ValueError("base-row cache shape differs")
    seen[:] = frozen
    seen.flush()
    return matrix, seen, state


def _checkpoint(
    config: CoverageConfig,
    identity: dict,
    matrix: np.memmap,
    seen: np.memmap,
    completed_chunks: int,
    source_rows: int,
    base_rows: int,
    complete: bool,
) -> dict:
    matrix.flush()
    seen.flush()
    checkpoint_path = config.seen_checkpoint_path(completed_chunks)
    checkpoint = np.lib.format.open_memmap(
        checkpoint_path,
        mode="w+",
        dtype=np.int32,
        shape=seen.shape,
    )
    checkpoint[:] = seen
    checkpoint.flush()
    del checkpoint
    state = {
        **identity,
        "scientific_result": False,
        "complete": complete,
        "completed_chunks": completed_chunks,
        "source_rows_scanned": source_rows,
        "base_rows_retained": base_rows,
        "post_base_rows_ignored": source_rows - base_rows,
        "seen_checkpoint": str(checkpoint_path),
    }
    _write_json(config.state_path, state)
    return state


def scan_base_prefix(
    config: CoverageConfig,
    catalog: Catalog,
    fingerprint: dict,
    user_capacity: int,
) -> tuple[np.memmap, dict]:
    identity = _cache_identity(
        config,
        fingerprint,
        catalog,
        user_capacity,
    )
    matrix, seen, state = _open_cache(
        config,
        identity,
        user_capacity,
    )
    if state["complete"]:
        return matrix, state
    completed = int(state["completed_chunks"])
    rows = int(state["source_rows_scanned"])
    base_rows = int(state["base_rows_retained"])
    started = time.perf_counter()
    last_chunk = completed
    for chunk_index, chunk in enumerate(read_qk_chunks(config), start=1):
        if chunk_index <= completed:
            continue
        users = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
        items = chunk["item_id"].to_numpy(dtype=np.int64, copy=False)
        positions = consume_user_positions(users, seen)
        selected = positions < config.base_prefix
        if selected.any():
            chosen_items = items[selected]
            in_range = (
                (chosen_items >= 0)
                & (chosen_items < len(catalog.dense_map))
            )
            mapped = np.zeros(len(chosen_items), dtype=np.int32)
            mapped[in_range] = catalog.dense_map[
                chosen_items[in_range]
            ]
            if np.any(mapped == 0):
                raise ValueError(
                    "base-prefix occurrence is absent from base catalog"
                )
            matrix[
                users[selected],
                positions[selected],
            ] = mapped.astype(np.uint32, copy=False)
            base_rows += len(mapped)
        rows += len(chunk)
        last_chunk = chunk_index
        print(
            f"phase=base_scan chunks={chunk_index} rows={rows:,} "
            f"base_rows={base_rows:,} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if chunk_index % config.checkpoint_every_chunks == 0:
            state = _checkpoint(
                config,
                identity,
                matrix,
                seen,
                chunk_index,
                rows,
                base_rows,
                False,
            )
    state = _checkpoint(
        config,
        identity,
        matrix,
        seen,
        last_chunk,
        rows,
        base_rows,
        True,
    )
    return matrix, state


def derive_pairs(
    config: CoverageConfig,
    catalog: Catalog,
    matrix: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    maximum = np.iinfo(np.int64).max
    frequency = np.zeros(catalog.rows, dtype=np.int64)
    neighbor_occurrence = np.full(
        catalog.rows,
        maximum,
        dtype=np.int64,
    )
    isolated_occurrence = np.full(
        catalog.rows,
        maximum,
        dtype=np.int64,
    )
    started = time.perf_counter()
    for start in range(0, len(matrix), config.derive_user_block):
        stop = min(start + config.derive_user_block, len(matrix))
        block = np.asarray(matrix[start:stop])
        valid = block > 0
        flat = block.reshape(-1)
        valid_indices = np.flatnonzero(flat)
        rows = flat[valid_indices].astype(np.int64, copy=False) - 1
        frequency += np.bincount(
            rows,
            minlength=catalog.rows,
        ).astype(np.int64, copy=False)
        local_users = valid_indices // config.base_prefix
        positions = valid_indices % config.base_prefix
        user_ids = local_users + start
        encoded = (
            user_ids.astype(np.int64) * config.base_prefix + positions
        )
        lengths = valid.sum(axis=1)
        has_neighbor = lengths[local_users] >= 2
        np.minimum.at(
            neighbor_occurrence,
            rows[has_neighbor],
            encoded[has_neighbor],
        )
        np.minimum.at(
            isolated_occurrence,
            rows[~has_neighbor],
            encoded[~has_neighbor],
        )
        print(
            f"phase=derive users={stop:,}/{len(matrix):,} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    if not np.array_equal(frequency, catalog.frequencies):
        mismatch = int(np.count_nonzero(frequency != catalog.frequencies))
        raise ValueError(
            f"derived base frequencies differ for {mismatch} rows"
        )
    has_neighbor = neighbor_occurrence != maximum
    selected = np.where(
        has_neighbor,
        neighbor_occurrence,
        isolated_occurrence,
    )
    if np.any(selected == maximum):
        raise ValueError("base catalog contains uncovered semantic rows")
    occurrence_user = selected // config.base_prefix
    occurrence_position = selected % config.base_prefix
    positive_position = np.where(
        has_neighbor,
        np.where(occurrence_position > 0, occurrence_position - 1, 1),
        occurrence_position,
    )
    positive_row = matrix[
        occurrence_user,
        positive_position,
    ].astype(np.int32)
    anchor_row = np.arange(1, catalog.rows + 1, dtype=np.int32)
    observed_anchor = matrix[
        occurrence_user,
        occurrence_position,
    ].astype(np.int32)
    if not np.array_equal(anchor_row, observed_anchor):
        raise ValueError("frozen anchor occurrence differs from row id")
    if np.any(has_neighbor & (positive_row == 0)):
        raise ValueError("neighbor-capable anchor has no positive row")
    positive_row[~has_neighbor] = anchor_row[~has_neighbor]
    arrays = {
        "anchor_row": anchor_row,
        "positive_row": positive_row,
        "base_frequency": frequency,
        "occurrence_user_id": occurrence_user.astype(np.int32),
        "occurrence_position": occurrence_position.astype(np.uint8),
        "positive_position": positive_position.astype(np.uint8),
        "has_same_user_neighbor": has_neighbor.astype(np.uint8),
    }
    audit = {
        "semantic_rows": catalog.rows,
        "covered_rows": int(np.count_nonzero(frequency)),
        "frequency_sum": int(frequency.sum()),
        "neighbor_rows": int(np.count_nonzero(has_neighbor)),
        "isolated_fallback_rows": int(
            np.count_nonzero(~has_neighbor)
        ),
        "isolated_policy": (
            "self row is stored as a masked placeholder; "
            "has_same_user_neighbor=0 forbids co-occurrence training"
        ),
        "all_neighbor_flags_valid": bool(
            np.all(
                positive_row[has_neighbor] > 0
            )
        ),
        "catalog_frequency_exact_match": True,
    }
    return arrays, audit


def run(config: CoverageConfig, audit_only: bool = False) -> dict:
    run_started = time.perf_counter()
    previous_summary = (
        json.loads(config.summary.read_text())
        if config.summary.is_file()
        else {}
    )
    cache_complete_before = False
    if config.state_path.is_file() and not config.refresh:
        prior_state = json.loads(config.state_path.read_text())
        cache_complete_before = bool(prior_state.get("complete", False))
    validate_config(config)
    fingerprint = source_fingerprint(config)
    catalog = load_catalog(config, fingerprint)
    user_capacity, user_capacity_audit = load_user_capacity(
        config,
        fingerprint,
    )
    matrix, scan_state = scan_base_prefix(
        config,
        catalog,
        fingerprint,
        user_capacity,
    )
    arrays, coverage = derive_pairs(config, catalog, matrix)
    content_sha256 = artifact_sha256(arrays)
    hashes = {
        name: array_sha256(value)
        for name, value in arrays.items()
    }
    metadata = {
        "protocol": PROTOCOL,
        "scientific_result": False,
        "formal_result": False,
        "dataset": "tenrec-qk",
        "purpose": "XP base-only row coverage and co-occurrence initialization",
        "source": fingerprint,
        "catalog": {
            "path": str(config.catalog_cache),
            "file_sha256": catalog.file_sha256,
            "semantic_rows": catalog.rows,
            "row_ids_sha256": catalog.metadata.get(
                "base_entity_item_ids_sha256",
                array_sha256(catalog.original_item_ids),
            ),
        },
        "base_only_boundary": {
            "per_user_positions": [0, config.base_prefix],
            "post_base_rows_used": False,
            "post_base_labels_used": False,
            "d1_actions_used": False,
            "final_outcomes_used": False,
        },
        "user_capacity_cache": user_capacity_audit,
        "scan": {
            "cache_state": str(config.state_path),
            "complete": bool(scan_state["complete"]),
            "completed_chunks": int(scan_state["completed_chunks"]),
            "source_rows_scanned": int(
                scan_state["source_rows_scanned"]
            ),
            "base_rows_retained": int(
                scan_state["base_rows_retained"]
            ),
            "post_base_rows_ignored": int(
                scan_state["post_base_rows_ignored"]
            ),
            "recoverable_seen_checkpoint": scan_state[
                "seen_checkpoint"
            ],
        },
        "coverage": coverage,
        "array_sha256": hashes,
        "content_sha256": content_sha256,
        "optimizer_active_gate": "pending_training",
    }
    summary = {
        **metadata,
        "status": "audit_only" if audit_only else "materialized",
        "artifact": {
            "path": str(config.output),
            "written": not audit_only,
            "bytes": None,
            "file_sha256": None,
        },
    }
    if not audit_only:
        _save_npz(config.output, arrays, metadata)
        summary["artifact"]["bytes"] = config.output.stat().st_size
        summary["artifact"]["file_sha256"] = file_sha256(
            config.output
        )
    invocation_seconds = time.perf_counter() - run_started
    runtime = dict(previous_summary.get("runtime", {}))
    runtime.update(
        {
            "invocation_wall_seconds": invocation_seconds,
            "scan_cache_complete": bool(scan_state["complete"]),
            "scan_cache_reused_for_this_invocation": (
                cache_complete_before
            ),
        }
    )
    if not cache_complete_before:
        runtime["initial_full_build_wall_seconds"] = invocation_seconds
    summary["runtime"] = runtime
    _write_json(config.summary, summary)
    return summary
