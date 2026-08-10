from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROTOCOL = "evokv_qk_theta0_next_item_corpus_v0"
SOURCE_COLUMNS = (
    "user_id",
    "item_id",
    "click",
    "follow",
    "like",
    "share",
)


@dataclass(frozen=True)
class QKTheta0CorpusConfig:
    source: Path
    member: str
    catalog: Path
    user_lengths: Path
    cache_dir: Path
    output: Path
    summary: Path
    base_prefix: int = 64
    prediction_rows: int = 250_000
    representative_users: int = 16_384
    minimum_eligible_rows: int = 2_840_105
    eligible_row_margin: int = 0
    selection_seed: int = 2026080501
    chunk_size: int = 2_000_000
    checkpoint_every_chunks: int = 16
    derive_user_block: int = 100_000
    refresh: bool = False

    @property
    def item_cache(self) -> Path:
        return self.cache_dir / "base_item_rows.npy"

    @property
    def behavior_cache(self) -> Path:
        return self.cache_dir / "base_behaviors.npy"

    @property
    def label_cache(self) -> Path:
        return self.cache_dir / "base_effective_labels.npy"

    @property
    def seen_cache(self) -> Path:
        return self.cache_dir / "seen_counts.npy"

    @property
    def state_path(self) -> Path:
        return self.cache_dir / "state.json"

    def seen_checkpoint(self, chunk: int) -> Path:
        return self.cache_dir / f"seen_counts_chunk_{chunk:06d}.npy"


@dataclass(frozen=True)
class QKTheta0Corpus:
    path: Path
    arrays: dict[str, np.ndarray]
    metadata: dict[str, object]
    file_sha256: str
    content_sha256: str

    @property
    def records(self) -> int:
        return len(self.arrays["record_user_ids"])

    @property
    def tokens(self) -> int:
        return len(self.arrays["item_idx"])


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    resolved = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(resolved.dtype).encode())
    digest.update(json.dumps(list(resolved.shape)).encode())
    digest.update(resolved.view(np.uint8))
    return digest.hexdigest()


def artifact_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode())
        digest.update(array_sha256(arrays[name]).encode())
    return digest.hexdigest()


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


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {
            name: source[name].copy()
            for name in source.files
            if name != "metadata_json"
        }
        metadata = json.loads(str(source["metadata_json"].item()))
    return arrays, metadata


def _source_identity(config: QKTheta0CorpusConfig) -> dict[str, object]:
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


def _source_matches(left: object, right: dict[str, object]) -> bool:
    if not isinstance(left, dict):
        return False
    keys = (
        "member",
        "member_size_bytes",
        "member_compressed_size_bytes",
        "member_crc32",
    )
    return all(left.get(name) == right[name] for name in keys)


def _validate_config(config: QKTheta0CorpusConfig) -> None:
    values = (
        config.base_prefix,
        config.prediction_rows,
        config.representative_users,
        config.minimum_eligible_rows,
        config.chunk_size,
        config.checkpoint_every_chunks,
        config.derive_user_block,
    )
    if min(values) < 1:
        raise ValueError("QK theta0 corpus dimensions must be positive")
    if config.base_prefix > np.iinfo(np.uint16).max:
        raise ValueError("QK theta0 base prefix exceeds ordinal storage")
    if config.eligible_row_margin < 0:
        raise ValueError("QK theta0 eligible row margin cannot be negative")


def _load_catalog(
    config: QKTheta0CorpusConfig,
    source_identity: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    arrays, metadata = _load_npz(config.catalog)
    required = {
        "base_entity_original_item_ids",
        "base_item_frequencies",
    }
    if not required.issubset(arrays):
        raise ValueError("QK theta0 catalog is incomplete")
    original = np.asarray(
        arrays["base_entity_original_item_ids"], dtype=np.int64
    )
    frequencies = np.asarray(arrays["base_item_frequencies"], dtype=np.int64)
    if (
        len(original) < config.prediction_rows
        or len(original) != len(frequencies)
        or len(np.unique(original)) != len(original)
        or np.any(original < 0)
        or np.any(frequencies < 1)
        or int(metadata.get("base_prefix_raw_events", -1)) != config.base_prefix
        or int(metadata.get("num_prediction_items", -1))
        != config.prediction_rows
        or not _source_matches(metadata.get("source"), source_identity)
    ):
        raise ValueError("QK theta0 catalog binding differs")
    dense = np.zeros(int(original.max()) + 1, dtype=np.int32)
    dense[original] = np.arange(1, len(original) + 1, dtype=np.int32)
    return dense, {
        "path": str(config.catalog),
        "file_sha256": file_sha256(config.catalog),
        "semantic_rows": len(original),
        "num_embeddings": len(original) + 1,
        "prediction_rows": config.prediction_rows,
        "base_entity_item_ids_sha256": metadata.get(
            "base_entity_item_ids_sha256"
        ),
        "base_frequency_sum": int(frequencies.sum()),
    }


def _load_users(
    config: QKTheta0CorpusConfig,
    source_identity: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    arrays, metadata = _load_npz(config.user_lengths)
    user_ids = np.asarray(arrays.get("user_ids"), dtype=np.int64)
    raw_lengths = np.asarray(arrays.get("raw_lengths"), dtype=np.int32)
    if (
        len(user_ids) == 0
        or len(user_ids) != len(raw_lengths)
        or len(np.unique(user_ids)) != len(user_ids)
        or np.any(user_ids < 0)
        or np.any(raw_lengths < 1)
        or not _source_matches(metadata.get("source"), source_identity)
    ):
        raise ValueError("QK theta0 user-length binding differs")
    raw_to_compact = np.full(int(user_ids.max()) + 1, -1, dtype=np.int32)
    raw_to_compact[user_ids] = np.arange(len(user_ids), dtype=np.int32)
    base_lengths = np.minimum(raw_lengths, config.base_prefix).astype(np.uint16)
    return user_ids, base_lengths, raw_to_compact, {
        "path": str(config.user_lengths),
        "file_sha256": file_sha256(config.user_lengths),
        "users": len(user_ids),
        "raw_rows": int(raw_lengths.astype(np.int64).sum()),
        "base_rows": int(base_lengths.astype(np.int64).sum()),
        "user_ids_sha256": metadata.get("user_ids_sha256"),
    }


def _scan_identity(
    config: QKTheta0CorpusConfig,
    source_identity: dict[str, object],
    catalog: dict[str, object],
    users: dict[str, object],
) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "phase": "base_prefix_scan",
        "source": source_identity,
        "catalog_file_sha256": catalog["file_sha256"],
        "user_lengths_file_sha256": users["file_sha256"],
        "users": users["users"],
        "base_prefix": config.base_prefix,
        "prediction_rows": config.prediction_rows,
        "shape": [users["users"], config.base_prefix],
    }


def _open_scan_cache(
    config: QKTheta0CorpusConfig,
    identity: dict[str, object],
) -> tuple[np.memmap, np.memmap, np.memmap, np.memmap, dict[str, object]]:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        config.item_cache,
        config.behavior_cache,
        config.label_cache,
        config.seen_cache,
    )
    if config.refresh or not config.state_path.is_file():
        shape = tuple(identity["shape"])
        item = np.lib.format.open_memmap(
            config.item_cache, mode="w+", dtype=np.uint32, shape=shape
        )
        behavior = np.lib.format.open_memmap(
            config.behavior_cache, mode="w+", dtype=np.uint8, shape=shape
        )
        label = np.lib.format.open_memmap(
            config.label_cache, mode="w+", dtype=np.uint8, shape=shape
        )
        seen = np.lib.format.open_memmap(
            config.seen_cache,
            mode="w+",
            dtype=np.int32,
            shape=(int(identity["users"]),),
        )
        item.fill(0)
        behavior.fill(0)
        label.fill(0)
        seen.fill(0)
        for value in (item, behavior, label, seen):
            value.flush()
        checkpoint = config.seen_checkpoint(0)
        with checkpoint.open("wb") as target:
            np.save(target, np.asarray(seen))
        state = {
            **identity,
            "complete": False,
            "completed_chunks": 0,
            "source_rows_scanned": 0,
            "base_rows_materialized": 0,
            "raw_positive_rows": 0,
            "effective_positive_rows": 0,
            "seen_checkpoint": str(checkpoint),
        }
        _atomic_json(config.state_path, state)
        return item, behavior, label, seen, state
    if any(not path.is_file() for path in paths):
        raise ValueError("QK theta0 scan cache is incomplete; use refresh")
    state = json.loads(config.state_path.read_text())
    if any(state.get(name) != value for name, value in identity.items()):
        raise ValueError("QK theta0 scan cache identity differs; use refresh")
    item = np.load(config.item_cache, mmap_mode="r+")
    behavior = np.load(config.behavior_cache, mmap_mode="r+")
    label = np.load(config.label_cache, mmap_mode="r+")
    seen = np.load(config.seen_cache, mmap_mode="r+")
    shape = tuple(identity["shape"])
    if (
        item.shape != shape
        or behavior.shape != shape
        or label.shape != shape
        or seen.shape != (int(identity["users"]),)
    ):
        raise ValueError("QK theta0 scan cache shape differs")
    checkpoint_path = Path(str(state.get("seen_checkpoint", "")))
    if not checkpoint_path.is_file():
        raise ValueError("QK theta0 seen-count checkpoint is absent")
    if not bool(state["complete"]):
        frozen_seen = np.load(checkpoint_path, mmap_mode="r")
        if frozen_seen.shape != seen.shape or frozen_seen.dtype != seen.dtype:
            raise ValueError("QK theta0 seen-count checkpoint differs")
        width = int(identity["base_prefix"])
        columns = np.arange(width).reshape(1, -1)
        block_size = 100_000
        for start in range(0, len(seen), block_size):
            stop = min(start + block_size, len(seen))
            keep = np.minimum(
                np.asarray(frozen_seen[start:stop], dtype=np.int64), width
            )
            stale = columns >= keep.reshape(-1, 1)
            item[start:stop][stale] = 0
            behavior[start:stop][stale] = 0
            label[start:stop][stale] = 0
        seen[:] = frozen_seen
        for value in (item, behavior, label, seen):
            value.flush()
    return item, behavior, label, seen, state


def _read_chunks(config: QKTheta0CorpusConfig):
    dtypes = {
        "user_id": "int32",
        "item_id": "int32",
        "click": "int8",
        "follow": "int8",
        "like": "int8",
        "share": "int8",
    }
    with zipfile.ZipFile(config.source) as archive, archive.open(
        config.member
    ) as source:
        yield from pd.read_csv(
            source,
            usecols=list(SOURCE_COLUMNS),
            dtype=dtypes,
            chunksize=config.chunk_size,
        )


def _positions(users: np.ndarray, seen: np.ndarray) -> np.ndarray:
    if len(users) == 0:
        return np.empty(0, dtype=np.int64)
    starts = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.flatnonzero(users[1:] != users[:-1]) + 1]
    )
    lengths = np.diff(np.append(starts, len(users))).astype(np.int64, copy=False)
    group_users = users[starts]
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
    offsets = seen[group_users].astype(np.int64, copy=False) + preceding - starts
    positions = np.arange(len(users), dtype=np.int64) + np.repeat(offsets, lengths)
    if len(np.unique(group_users)) == len(group_users):
        seen[group_users] += lengths.astype(seen.dtype, copy=False)
    else:
        np.add.at(seen, group_users, lengths.astype(seen.dtype, copy=False))
    return positions


def _behaviors(frame: pd.DataFrame) -> np.ndarray:
    output = np.ones(len(frame), dtype=np.uint8)
    for name, value in (("click", 2), ("like", 3), ("follow", 4), ("share", 5)):
        output[frame[name].to_numpy(dtype=np.uint8, copy=False) > 0] = value
    return output


def _raw_positive(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["click"].to_numpy(dtype=np.uint8, copy=False)
        | frame["follow"].to_numpy(dtype=np.uint8, copy=False)
        | frame["like"].to_numpy(dtype=np.uint8, copy=False)
        | frame["share"].to_numpy(dtype=np.uint8, copy=False)
    ) > 0


def _flush_scan(
    config: QKTheta0CorpusConfig,
    identity: dict[str, object],
    arrays: tuple[np.memmap, ...],
    *,
    complete: bool,
    chunks: int,
    source_rows: int,
    base_rows: int,
    raw_positives: int,
    effective_positives: int,
) -> dict[str, object]:
    for value in arrays:
        value.flush()
    seen = arrays[-1]
    checkpoint = config.seen_checkpoint(chunks)
    temporary = checkpoint.with_name(f".{checkpoint.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as target:
        np.save(target, np.asarray(seen))
    temporary.replace(checkpoint)
    state = {
        **identity,
        "complete": complete,
        "completed_chunks": chunks,
        "source_rows_scanned": source_rows,
        "base_rows_materialized": base_rows,
        "raw_positive_rows": raw_positives,
        "effective_positive_rows": effective_positives,
        "seen_checkpoint": str(checkpoint),
    }
    _atomic_json(config.state_path, state)
    return state


def _scan_base(
    config: QKTheta0CorpusConfig,
    identity: dict[str, object],
    dense_item_map: np.ndarray,
    raw_to_compact: np.ndarray,
) -> tuple[np.memmap, np.memmap, np.memmap, dict[str, object]]:
    item, behavior, label, seen, state = _open_scan_cache(config, identity)
    if bool(state["complete"]):
        return item, behavior, label, state
    completed = int(state["completed_chunks"])
    source_rows = int(state["source_rows_scanned"])
    base_rows = int(state["base_rows_materialized"])
    raw_positives = int(state["raw_positive_rows"])
    effective_positives = int(state["effective_positive_rows"])
    last_chunk = completed
    started = time.perf_counter()
    for chunk_index, frame in enumerate(_read_chunks(config), start=1):
        if chunk_index <= completed:
            continue
        raw_users = frame["user_id"].to_numpy(dtype=np.int64, copy=False)
        if np.any(raw_users < 0) or np.any(raw_users >= len(raw_to_compact)):
            raise ValueError("QK theta0 source user id exceeds frozen users")
        compact = raw_to_compact[raw_users]
        if np.any(compact < 0):
            raise ValueError("QK theta0 source user is absent from length cache")
        ordinal = _positions(compact, seen)
        selected = ordinal < config.base_prefix
        if selected.any():
            destination_user = compact[selected]
            destination_ordinal = ordinal[selected]
            original_items = frame["item_id"].to_numpy(dtype=np.int64, copy=False)[
                selected
            ]
            in_range = (original_items >= 0) & (original_items < len(dense_item_map))
            mapped = np.zeros(len(original_items), dtype=np.int32)
            mapped[in_range] = dense_item_map[original_items[in_range]]
            if np.any(mapped == 0):
                raise ValueError("QK base-prefix item is absent from base catalog")
            all_behavior = _behaviors(frame)[selected]
            all_raw_positive = _raw_positive(frame)[selected]
            all_effective = all_raw_positive & (mapped <= config.prediction_rows)
            if np.any(item[destination_user, destination_ordinal] != 0):
                raise ValueError("QK theta0 base event destination is duplicated")
            item[destination_user, destination_ordinal] = mapped.astype(
                np.uint32, copy=False
            )
            behavior[destination_user, destination_ordinal] = all_behavior
            label[destination_user, destination_ordinal] = all_effective.astype(
                np.uint8, copy=False
            )
            base_rows += len(mapped)
            raw_positives += int(all_raw_positive.sum())
            effective_positives += int(all_effective.sum())
        source_rows += len(frame)
        last_chunk = chunk_index
        print(
            f"phase=qk_theta0_scan chunk={chunk_index} rows={source_rows:,} "
            f"base={base_rows:,} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if chunk_index % config.checkpoint_every_chunks == 0:
            state = _flush_scan(
                config,
                identity,
                (item, behavior, label, seen),
                complete=False,
                chunks=chunk_index,
                source_rows=source_rows,
                base_rows=base_rows,
                raw_positives=raw_positives,
                effective_positives=effective_positives,
            )
    state = _flush_scan(
        config,
        identity,
        (item, behavior, label, seen),
        complete=True,
        chunks=last_chunk,
        source_rows=source_rows,
        base_rows=base_rows,
        raw_positives=raw_positives,
        effective_positives=effective_positives,
    )
    return item, behavior, label, state


def training_ends(labels: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    if labels.ndim != 2 or lengths.shape != (len(labels),):
        raise ValueError("QK theta0 training-end input differs")
    width = labels.shape[1]
    valid = np.arange(width).reshape(1, -1) < lengths.reshape(-1, 1)
    eligible = (labels > 0) & valid
    positions = np.where(eligible, np.arange(width).reshape(1, -1), -1)
    last = positions.max(axis=1)
    return np.where(last >= 1, last + 1, 0).astype(np.uint16)


def select_training_users(
    item_rows: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    user_ids: np.ndarray,
    *,
    semantic_rows: int,
    representative_users: int,
    minimum_eligible_rows: int,
    selection_seed: int,
    eligible_row_margin: int = 0,
    block_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if (
        item_rows.ndim != 2
        or labels.shape != item_rows.shape
        or lengths.shape != (len(item_rows),)
        or user_ids.shape != (len(item_rows),)
        or semantic_rows < 1
        or representative_users < 1
        or minimum_eligible_rows < 1
        or minimum_eligible_rows > semantic_rows
        or eligible_row_margin < 0
        or block_size < 1
    ):
        raise ValueError("QK theta0 user-selection input differs")
    ends = training_ends(labels, lengths)
    frequency = np.zeros(semantic_rows + 1, dtype=np.int64)
    for start in range(0, len(item_rows), block_size):
        stop = min(start + block_size, len(item_rows))
        block = np.asarray(item_rows[start:stop])
        mask = np.arange(block.shape[1]).reshape(1, -1) < ends[
            start:stop
        ].reshape(-1, 1)
        rows = block[mask].astype(np.int64, copy=False)
        frequency += np.bincount(rows, minlength=semantic_rows + 1)
    eligible_rows = frequency[1:] > 0
    eligible_count = int(eligible_rows.sum())
    if eligible_count < minimum_eligible_rows:
        raise RuntimeError(
            "QK theta0 base data cannot satisfy the optimizer-active row gate: "
            f"eligible={eligible_count:,} required={minimum_eligible_rows:,}"
        )
    target_covered_rows = min(
        eligible_count, minimum_eligible_rows + eligible_row_margin
    )
    inverse = np.zeros_like(frequency, dtype=np.float64)
    inverse[frequency > 0] = 1.0 / frequency[frequency > 0]
    scores = np.zeros(len(item_rows), dtype=np.float64)
    for start in range(0, len(item_rows), block_size):
        stop = min(start + block_size, len(item_rows))
        block = np.asarray(item_rows[start:stop])
        mask = np.arange(block.shape[1]).reshape(1, -1) < ends[
            start:stop
        ].reshape(-1, 1)
        scores[start:stop] = (inverse[block] * mask).sum(axis=1)
    hashes = splitmix64(
        user_ids.astype(np.uint64, copy=False) ^ np.uint64(selection_seed)
    )
    eligible_users = np.flatnonzero(ends >= 2)
    representative_order = eligible_users[
        np.lexsort((user_ids[eligible_users], hashes[eligible_users]))
    ]
    representative_order = representative_order[:representative_users]
    selected = np.zeros(len(item_rows), dtype=np.uint8)
    selected[representative_order] |= np.uint8(1)
    covered = np.zeros(semantic_rows + 1, dtype=np.bool_)
    for user in representative_order:
        end = int(ends[user])
        covered[np.unique(item_rows[user, :end])] = True
    covered[0] = False
    coverage_order = eligible_users[
        np.lexsort(
            (
                user_ids[eligible_users],
                hashes[eligible_users],
                -scores[eligible_users],
            )
        )
    ]
    coverage_count = int(covered.sum())
    for user in coverage_order:
        if coverage_count >= target_covered_rows:
            break
        end = int(ends[user])
        rows = np.unique(item_rows[user, :end])
        gain = int(np.count_nonzero(~covered[rows]))
        if gain:
            selected[user] |= np.uint8(2)
            covered[rows] = True
            coverage_count += gain
    selected_users = np.flatnonzero(selected > 0)
    training_order = selected_users[
        np.lexsort((user_ids[selected_users], hashes[selected_users]))
    ]
    selected_tokens = int(ends[training_order].astype(np.int64).sum())
    return training_order, ends, {
        "eligible_users": len(eligible_users),
        "eligible_semantic_rows": eligible_count,
        "eligible_semantic_fraction": eligible_count / semantic_rows,
        "required_semantic_rows": minimum_eligible_rows,
        "required_semantic_fraction": minimum_eligible_rows / semantic_rows,
        "target_covered_rows": target_covered_rows,
        "eligible_row_margin_requested": eligible_row_margin,
        "representative_users_requested": representative_users,
        "representative_users_selected": len(representative_order),
        "coverage_users_selected": int(np.count_nonzero(selected & 2)),
        "selected_users": len(training_order),
        "selected_tokens": selected_tokens,
        "selected_covered_rows": coverage_count,
        "selected_covered_fraction": coverage_count / semantic_rows,
        "selection_code": selected[training_order],
        "frequency_minimum": int(frequency[eligible_rows.nonzero()[0] + 1].min()),
        "frequency_maximum": int(frequency.max()),
    }


def _materialize_corpus(
    config: QKTheta0CorpusConfig,
    item: np.ndarray,
    behavior: np.ndarray,
    label: np.ndarray,
    user_ids: np.ndarray,
    selected: np.ndarray,
    ends: np.ndarray,
    selection: dict[str, object],
    metadata: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    record_lengths = ends[selected].astype(np.uint16, copy=True)
    offsets = np.zeros(len(selected) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(record_lengths, dtype=np.int64)
    item_idx = np.empty(int(offsets[-1]), dtype=np.uint32)
    behaviors = np.empty(int(offsets[-1]), dtype=np.uint8)
    labels = np.empty(int(offsets[-1]), dtype=np.uint8)
    raw_ordinal = np.empty(int(offsets[-1]), dtype=np.uint16)
    for record, user in enumerate(selected):
        start = int(offsets[record])
        stop = int(offsets[record + 1])
        length = stop - start
        item_idx[start:stop] = item[user, :length]
        behaviors[start:stop] = behavior[user, :length]
        labels[start:stop] = label[user, :length]
        raw_ordinal[start:stop] = np.arange(length, dtype=np.uint16)
    selection_code = np.asarray(selection.pop("selection_code"), dtype=np.uint8)
    arrays = {
        "record_user_ids": user_ids[selected].astype(np.int64, copy=True),
        "record_offsets": offsets,
        "record_lengths": record_lengths,
        "record_selection": selection_code,
        "item_idx": item_idx,
        "behavior": behaviors,
        "label": labels,
        "raw_ordinal": raw_ordinal,
    }
    content_hash = artifact_sha256(arrays)
    corpus_metadata = {
        **metadata,
        "protocol": PROTOCOL,
        "scientific_result": False,
        "formal_result": False,
        "objective": (
            "sampled next-item cross entropy over real within-user base-prefix "
            "sequences; every retained sequence ends at its last effective "
            "engaged prediction target"
        ),
        "base_only_boundary": {
            "base_prefix": config.base_prefix,
            "post_base_rows_materialized": False,
            "vocabulary_fit": "base prefix only",
        },
        "selection": selection,
        "content_sha256": content_hash,
    }
    return arrays, corpus_metadata


def build_qk_theta0_corpus(config: QKTheta0CorpusConfig) -> dict[str, object]:
    _validate_config(config)
    started = time.perf_counter()
    source_identity = _source_identity(config)
    dense_item_map, catalog = _load_catalog(config, source_identity)
    user_ids, base_lengths, raw_to_compact, users = _load_users(
        config, source_identity
    )
    identity = _scan_identity(config, source_identity, catalog, users)
    item, behavior, label, scan = _scan_base(
        config, identity, dense_item_map, raw_to_compact
    )
    if int(scan["source_rows_scanned"]) != int(users["raw_rows"]):
        raise ValueError("QK theta0 full source scan row count differs")
    if int(scan["base_rows_materialized"]) != int(users["base_rows"]):
        raise ValueError("QK theta0 base-prefix row count differs")
    selected, ends, selection = select_training_users(
        item,
        label,
        base_lengths,
        user_ids,
        semantic_rows=int(catalog["semantic_rows"]),
        representative_users=config.representative_users,
        minimum_eligible_rows=config.minimum_eligible_rows,
        selection_seed=config.selection_seed,
        eligible_row_margin=config.eligible_row_margin,
        block_size=config.derive_user_block,
    )
    arrays, corpus_metadata = _materialize_corpus(
        config,
        item,
        behavior,
        label,
        user_ids,
        selected,
        ends,
        selection,
        {
            "dataset": "tenrec-qk",
            "source": source_identity,
            "catalog": catalog,
            "user_lengths": users,
            "scan": scan,
            "selection_seed": config.selection_seed,
        },
    )
    config.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output.with_name(f".{config.output.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as target:
        np.savez_compressed(
            target,
            **arrays,
            metadata_json=np.asarray(json.dumps(corpus_metadata, sort_keys=True)),
        )
    temporary.replace(config.output)
    summary = {
        **corpus_metadata,
        "status": "pass",
        "artifact": {
            "path": str(config.output),
            "bytes": config.output.stat().st_size,
            "file_sha256": file_sha256(config.output),
            "content_sha256": corpus_metadata["content_sha256"],
        },
        "records": len(arrays["record_user_ids"]),
        "tokens": len(arrays["item_idx"]),
        "effective_targets": int(
            arrays["label"].sum()
            - arrays["label"][arrays["record_offsets"][:-1]].sum()
        ),
        "lengths": {
            "minimum": int(arrays["record_lengths"].min()),
            "median": float(np.median(arrays["record_lengths"])),
            "p95": float(np.quantile(arrays["record_lengths"], 0.95)),
            "maximum": int(arrays["record_lengths"].max()),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(config.summary, summary)
    return summary


def build_canary_from_fixed_edges(
    edge_path: Path,
    output: Path,
    summary: Path,
    *,
    maximum_records: int,
) -> dict[str, object]:
    if maximum_records < 1:
        raise ValueError("QK theta0 canary record count must be positive")
    with np.load(edge_path, allow_pickle=False) as source:
        edge = {
            name: source[name].copy()
            for name in source.files
            if name != "metadata_json"
        }
        edge_metadata = json.loads(str(source["metadata_json"].item()))
    records = np.flatnonzero(edge["record_role"] == 0)[:maximum_records]
    parts: dict[str, list[np.ndarray]] = {
        "item_idx": [],
        "behavior": [],
        "label": [],
        "raw_ordinal": [],
    }
    users = []
    selection = []
    for record in records:
        start = int(edge["record_offsets"][record])
        end = start + int(edge["record_history_end"][record])
        labels = edge["label"][start:end]
        target_positions = np.flatnonzero(labels > 0)
        target_positions = target_positions[target_positions >= 1]
        if len(target_positions) == 0:
            continue
        stop = start + int(target_positions[-1]) + 1
        for name, source_name in (
            ("item_idx", "item_idx"),
            ("behavior", "behavior"),
            ("label", "label"),
            ("raw_ordinal", "raw_ordinal"),
        ):
            parts[name].append(edge[source_name][start:stop])
        users.append(int(edge["record_user_ids"][record]))
        selection.append(1)
    if not users:
        raise RuntimeError("QK theta0 canary has no effective next-item targets")
    lengths = np.asarray([len(value) for value in parts["item_idx"]], dtype=np.uint16)
    offsets = np.zeros(len(users) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    arrays = {
        "record_user_ids": np.asarray(users, dtype=np.int64),
        "record_offsets": offsets,
        "record_lengths": lengths,
        "record_selection": np.asarray(selection, dtype=np.uint8),
        **{name: np.concatenate(values) for name, values in parts.items()},
    }
    content_hash = artifact_sha256(arrays)
    metadata = {
        "protocol": PROTOCOL,
        "dataset": "tenrec-qk",
        "scientific_result": False,
        "formal_result": False,
        "development_canary": True,
        "source_fixed_edge": {
            "path": str(edge_path),
            "file_sha256": file_sha256(edge_path),
            "content_sha256": edge_metadata.get("content_sha256"),
        },
        "catalog": edge_metadata["catalog"],
        "objective": "sampled next-item cross entropy on base-only real user prefixes",
        "base_only_boundary": {
            "base_prefix": 64,
            "post_base_rows_materialized": False,
            "vocabulary_fit": "inherited frozen base-only catalog",
        },
        "selection": {
            "selected_users": len(users),
            "selected_tokens": len(arrays["item_idx"]),
            "selected_covered_rows": len(np.unique(arrays["item_idx"])),
        },
        "content_sha256": content_hash,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as target:
        np.savez_compressed(
            target,
            **arrays,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    result = {
        **metadata,
        "status": "pass",
        "artifact": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "file_sha256": file_sha256(output),
            "content_sha256": content_hash,
        },
        "records": len(users),
        "tokens": len(arrays["item_idx"]),
        "effective_targets": int(arrays["label"].sum()),
    }
    _atomic_json(summary, result)
    return result


def load_qk_theta0_corpus(
    path: str | Path,
    *,
    num_embeddings: int,
    num_prediction_items: int,
) -> QKTheta0Corpus:
    resolved = Path(path)
    arrays, metadata = _load_npz(resolved)
    required = {
        "record_user_ids",
        "record_offsets",
        "record_lengths",
        "record_selection",
        "item_idx",
        "behavior",
        "label",
        "raw_ordinal",
    }
    if not required.issubset(arrays):
        raise ValueError("QK theta0 corpus arrays are incomplete")
    records = len(arrays["record_user_ids"])
    offsets = arrays["record_offsets"]
    lengths = arrays["record_lengths"]
    tokens = len(arrays["item_idx"])
    content_hash = artifact_sha256(arrays)
    catalog = metadata.get("catalog")
    catalog_rows = int(catalog.get("semantic_rows", catalog.get("base_entity_rows", -1)))
    if (
        metadata.get("protocol") != PROTOCOL
        or metadata.get("dataset") != "tenrec-qk"
        or metadata.get("scientific_result") is not False
        or metadata.get("base_only_boundary", {}).get(
            "post_base_rows_materialized"
        )
        is not False
        or metadata.get("content_sha256") != content_hash
        or catalog_rows + 1 != num_embeddings
        or int(catalog.get("prediction_rows", -1)) != num_prediction_items
        or offsets.shape != (records + 1,)
        or offsets[0] != 0
        or offsets[-1] != tokens
        or np.any(offsets[1:] <= offsets[:-1])
        or lengths.shape != (records,)
        or not np.array_equal(np.diff(offsets), lengths.astype(np.int64))
        or arrays["record_selection"].shape != (records,)
        or len(np.unique(arrays["record_user_ids"])) != records
        or any(
            arrays[name].shape != (tokens,)
            for name in ("behavior", "label", "raw_ordinal")
        )
        or np.any(arrays["item_idx"] < 1)
        or np.any(arrays["item_idx"] >= num_embeddings)
        or np.any(arrays["behavior"] < 1)
        or np.any(arrays["behavior"] > 5)
        or np.any(arrays["label"] > 1)
        or np.any(arrays["item_idx"][arrays["label"].astype(bool)] > num_prediction_items)
    ):
        raise ValueError("QK theta0 corpus semantics differ")
    for record in range(records):
        start = int(offsets[record])
        stop = int(offsets[record + 1])
        if (
            lengths[record] < 2
            or arrays["label"][start + 1 : stop].sum() < 1
            or not np.array_equal(
                arrays["raw_ordinal"][start:stop],
                np.arange(stop - start, dtype=arrays["raw_ordinal"].dtype),
            )
        ):
            raise ValueError("QK theta0 record is not a causal base sequence")
    return QKTheta0Corpus(
        path=resolved,
        arrays=arrays,
        metadata=metadata,
        file_sha256=file_sha256(resolved),
        content_sha256=content_hash,
    )


def build_rank_batch(
    corpus: QKTheta0Corpus,
    records: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    if batch_size < 1 or len(records) > batch_size:
        raise ValueError("QK theta0 rank batch request differs")
    real_lengths = [
        int(corpus.arrays["record_lengths"][record]) for record in records
    ]
    width = max([2, *real_lengths])
    item_ids = np.zeros((batch_size, width), dtype=np.int64)
    behaviors = np.zeros((batch_size, width), dtype=np.int64)
    labels = np.zeros((batch_size, width), dtype=np.int64)
    time_deltas = np.zeros((batch_size, width), dtype=np.float32)
    lengths = np.zeros(batch_size, dtype=np.int64)
    record_indices = np.full(batch_size, -1, dtype=np.int64)
    for row, record in enumerate(records):
        start = int(corpus.arrays["record_offsets"][record])
        stop = int(corpus.arrays["record_offsets"][record + 1])
        length = stop - start
        item_ids[row, :length] = corpus.arrays["item_idx"][start:stop]
        behaviors[row, :length] = corpus.arrays["behavior"][start:stop]
        labels[row, :length] = corpus.arrays["label"][start:stop]
        ordinals = corpus.arrays["raw_ordinal"][start:stop].astype(
            np.float32, copy=False
        )
        if length > 1:
            time_deltas[row, 1:length] = np.diff(ordinals)
        lengths[row] = length
        record_indices[row] = record
    train_mask = np.arange(width).reshape(1, -1) < lengths.reshape(-1, 1)
    return {
        "item_ids": torch.from_numpy(item_ids),
        "behaviors": torch.from_numpy(behaviors),
        "time_deltas": torch.from_numpy(time_deltas),
        "lengths": torch.from_numpy(lengths),
        "labels": torch.from_numpy(labels),
        "train_mask": torch.from_numpy(train_mask),
        "record_indices": torch.from_numpy(record_indices),
    }


def epoch_record_order(
    corpus: QKTheta0Corpus,
    *,
    seed: int,
    epoch: int,
    bucket_size: int,
) -> np.ndarray:
    if epoch < 0 or bucket_size < 1:
        raise ValueError("QK theta0 epoch-order request differs")
    generator = np.random.default_rng(seed + epoch * 1_000_003)
    order = generator.permutation(corpus.records)
    lengths = corpus.arrays["record_lengths"]
    for start in range(0, len(order), bucket_size):
        stop = min(start + bucket_size, len(order))
        block = order[start:stop]
        order[start:stop] = block[
            np.argsort(lengths[block], kind="stable")
        ]
    return order.astype(np.int64, copy=False)
