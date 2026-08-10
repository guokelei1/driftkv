from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .qk_xp_edge_inputs import (
    EdgeInputConfig,
    action_masks,
    artifact_sha256,
    behavior_values,
    consume_user_positions,
    file_sha256,
    load_catalog,
    read_qk_chunks,
    source_fingerprint,
    typed_array_sha256,
)

PROTOCOL = "evokv_qk_stream_chain_data_v0"
ROLE_NAMES = (
    "stream_train",
    "fit_tuning",
    "qualification",
    "final",
    "short_diagnostic",
)
ROLE_CODES = {name: index for index, name in enumerate(ROLE_NAMES)}
LONG_ROLE_NAMES = ROLE_NAMES[:4]


@dataclass(frozen=True)
class QKStreamChainConfig:
    source: Path
    member: str
    catalog: Path
    user_lengths: Path
    theta0_corpus: Path
    final_workload: Path
    roles_output: Path
    corpus_output: Path
    summary_output: Path
    base_prefix: int = 64
    maximum_sequence_length: int = 512
    update_count: int = 7
    stream_train_users: int = 16_384
    fit_tuning_users: int = 2_048
    qualification_users: int = 4_096
    final_users: int = 65_536
    short_diagnostic_users: int = 4_096
    minimum_long_events: int = 96
    short_minimum_events: int = 32
    short_maximum_events: int = 63
    selection_salt: str = "evokv-qk-next-item-theta0-theta7-20260805-v0"
    chunk_size: int = 2_000_000


@dataclass(frozen=True)
class QKStreamChainCorpus:
    path: Path
    arrays: dict[str, np.ndarray]
    metadata: dict[str, object]
    file_sha256: str
    content_sha256: str

    def role_records(self, role: str) -> np.ndarray:
        if role not in ROLE_CODES:
            raise ValueError("QK stream role differs")
        return np.flatnonzero(
            self.arrays["record_role"] == ROLE_CODES[role]
        )

    def record_slice(self, record: int) -> slice:
        offsets = self.arrays["record_offsets"]
        return slice(int(offsets[record]), int(offsets[record + 1]))


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


def _seed(salt: str) -> np.uint64:
    return np.uint64(
        int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], "little")
    )


def _splitmix64(values: np.ndarray) -> np.ndarray:
    hashed = values.astype(np.uint64, copy=True)
    hashed = hashed + np.uint64(0x9E3779B97F4A7C15)
    hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    return hashed ^ (hashed >> np.uint64(31))


def stable_user_order(user_ids: np.ndarray, salt: str) -> np.ndarray:
    values = np.asarray(user_ids, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0):
        raise ValueError("QK stream user ids differ")
    hashes = _splitmix64(values.astype(np.uint64) ^ _seed(salt))
    return values[np.lexsort((values, hashes))]


def chain_boundaries(
    valid_lengths: np.ndarray,
    *,
    base_prefix: int,
    update_count: int,
) -> np.ndarray:
    lengths = np.asarray(valid_lengths, dtype=np.int64)
    intervals = update_count + 1
    base_last = base_prefix - 1
    final = lengths - 1
    if (
        lengths.ndim != 1
        or len(lengths) < 1
        or base_prefix < 2
        or update_count < 1
        or np.any(final - base_last < intervals)
    ):
        raise ValueError("QK stream boundary lengths differ")
    result = np.empty((len(lengths), intervals + 1), dtype=np.uint16)
    result[:, 0] = base_last
    width = final - base_last
    for boundary in range(1, intervals):
        result[:, boundary] = (
            base_last + np.floor_divide(width * boundary, intervals)
        ).astype(np.uint16)
    result[:, intervals] = final.astype(np.uint16)
    if np.any(result[:, 1:] <= result[:, :-1]):
        raise ValueError("QK stream boundaries are not increasing")
    return result


def _load_user_lengths(
    config: QKStreamChainConfig,
    fingerprint: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    arrays, metadata = _load_npz(config.user_lengths)
    user_ids = np.asarray(arrays.get("user_ids"), dtype=np.int64)
    lengths = np.asarray(arrays.get("raw_lengths"), dtype=np.int32)
    source = metadata.get("source", {})
    keys = (
        "member",
        "member_size_bytes",
        "member_compressed_size_bytes",
        "member_crc32",
    )
    if (
        len(user_ids) < 1
        or user_ids.shape != lengths.shape
        or len(np.unique(user_ids)) != len(user_ids)
        or np.any(user_ids < 0)
        or np.any(lengths < 1)
        or any(source.get(name) != fingerprint[name] for name in keys)
    ):
        raise ValueError("QK stream user-length binding differs")
    return user_ids, lengths, {
        "path": str(config.user_lengths),
        "sha256": file_sha256(config.user_lengths),
        "users": len(user_ids),
        "rows": int(lengths.astype(np.int64).sum()),
    }


def _dense_lengths(
    user_ids: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    dense = np.zeros(int(user_ids.max()) + 1, dtype=np.int32)
    dense[user_ids] = lengths
    return dense


def _final_users(
    config: QKStreamChainConfig,
    dense_lengths: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    arrays, metadata = _load_npz(config.final_workload)
    users = np.asarray(arrays.get("record_user_ids"), dtype=np.int64)
    raw_lengths = np.asarray(arrays.get("record_raw_lengths"), dtype=np.int32)
    if (
        metadata.get("dataset") != "tenrec-qk"
        or len(users) != config.final_users
        or users.shape != raw_lengths.shape
        or len(np.unique(users)) != len(users)
        or np.any(users < 0)
        or np.any(users >= len(dense_lengths))
        or not np.array_equal(dense_lengths[users], raw_lengths)
        or np.any(raw_lengths < config.minimum_long_events)
    ):
        raise ValueError("QK final workload binding differs")
    return users, {
        "path": str(config.final_workload),
        "sha256": file_sha256(config.final_workload),
        "protocol": metadata.get("protocol"),
        "user_ids_sha256": typed_array_sha256(users),
    }


def _bucket_codes(lengths: np.ndarray) -> np.ndarray:
    return np.searchsorted(
        np.asarray([128, 192, 384], dtype=np.int64),
        lengths,
        side="right",
    ).astype(np.uint8)


def _quotas(count: int) -> tuple[int, int, int, int]:
    base, remainder = divmod(count, 4)
    return tuple(base + int(index < remainder) for index in range(4))


def _select_long_roles(
    config: QKStreamChainConfig,
    user_ids: np.ndarray,
    lengths: np.ndarray,
    final_users: np.ndarray,
) -> dict[str, np.ndarray]:
    final_mask = np.zeros(int(user_ids.max()) + 1, dtype=np.bool_)
    final_mask[final_users] = True
    eligible = (lengths >= config.minimum_long_events) & ~final_mask[user_ids]
    candidates = user_ids[eligible]
    candidate_lengths = lengths[eligible]
    buckets = _bucket_codes(candidate_lengths)
    ordered = {
        bucket: stable_user_order(
            candidates[buckets == bucket],
            f"{config.selection_salt}:long:{bucket}",
        )
        for bucket in range(4)
    }
    cursors = {bucket: 0 for bucket in range(4)}
    requested = {
        "stream_train": config.stream_train_users,
        "fit_tuning": config.fit_tuning_users,
        "qualification": config.qualification_users,
    }
    result: dict[str, np.ndarray] = {}
    for role, count in requested.items():
        pieces = []
        for bucket, quota in enumerate(_quotas(count)):
            start = cursors[bucket]
            stop = start + quota
            if stop > len(ordered[bucket]):
                raise RuntimeError(f"QK stream bucket {bucket} is too small")
            pieces.append(ordered[bucket][start:stop])
            cursors[bucket] = stop
        result[role] = stable_user_order(
            np.concatenate(pieces),
            f"{config.selection_salt}:{role}:merge",
        )
    result["final"] = final_users.copy()
    return result


def _select_short_role(
    config: QKStreamChainConfig,
    user_ids: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    theta0, _ = _load_npz(config.theta0_corpus)
    theta0_users = np.asarray(theta0.get("record_user_ids"), dtype=np.int64)
    excluded = np.zeros(int(user_ids.max()) + 1, dtype=np.bool_)
    excluded[theta0_users] = True
    eligible = (
        (lengths >= config.short_minimum_events)
        & (lengths <= config.short_maximum_events)
        & ~excluded[user_ids]
    )
    ordered = stable_user_order(
        user_ids[eligible],
        f"{config.selection_salt}:short",
    )
    if len(ordered) < config.short_diagnostic_users:
        raise RuntimeError("QK short diagnostic role is too small")
    return ordered[: config.short_diagnostic_users]


def _role_record(
    values: np.ndarray,
    dense_lengths: np.ndarray,
) -> dict[str, object]:
    lengths = dense_lengths[values]
    buckets = _bucket_codes(lengths) if np.all(lengths >= 96) else None
    return {
        "count": len(values),
        "user_ids": [int(value) for value in values],
        "user_ids_sha256": typed_array_sha256(values),
        "minimum_raw_length": int(lengths.min()),
        "maximum_raw_length": int(lengths.max()),
        "length_bucket_counts": (
            {
                name: int(np.count_nonzero(buckets == index))
                for index, name in enumerate(
                    ("96_127", "128_191", "192_383", "384_plus")
                )
            }
            if buckets is not None
            else None
        ),
    }


def build_roles(
    config: QKStreamChainConfig,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    read_config = EdgeInputConfig(
        source=config.source,
        member=config.member,
        catalog_cache=config.catalog,
        roles=config.roles_output,
        output=config.corpus_output,
        summary=config.summary_output,
        chunk_size=config.chunk_size,
    )
    fingerprint = source_fingerprint(read_config)
    user_ids, lengths, length_binding = _load_user_lengths(
        config, fingerprint
    )
    dense = _dense_lengths(user_ids, lengths)
    final_users, final_binding = _final_users(config, dense)
    roles = _select_long_roles(
        config, user_ids, lengths, final_users
    )
    roles["short_diagnostic"] = _select_short_role(
        config, user_ids, lengths
    )
    combined = np.concatenate([roles[name] for name in ROLE_NAMES])
    if len(np.unique(combined)) != len(combined):
        raise ValueError("QK stream roles overlap")
    document = {
        "protocol": PROTOCOL,
        "status": "frozen",
        "scientific_result": False,
        "formal_result": False,
        "selection_salt": config.selection_salt,
        "source": fingerprint,
        "user_lengths": length_binding,
        "final_workload": final_binding,
        "theta0_corpus": {
            "path": str(config.theta0_corpus),
            "sha256": file_sha256(config.theta0_corpus),
            "short_users_excluded": True,
        },
        "base_prefix": config.base_prefix,
        "maximum_sequence_length": config.maximum_sequence_length,
        "update_count": config.update_count,
        "model_versions": list(range(config.update_count + 1)),
        "roles_pairwise_disjoint": True,
        "post_base_selection_uses_labels": False,
        "roles": {
            name: _role_record(roles[name], dense)
            for name in ROLE_NAMES
        },
    }
    return roles, document


def _layout(
    config: QKStreamChainConfig,
    roles: dict[str, np.ndarray],
    dense_lengths: np.ndarray,
) -> dict[str, np.ndarray]:
    users = np.concatenate([roles[name] for name in ROLE_NAMES])
    role = np.concatenate(
        [
            np.full(len(roles[name]), ROLE_CODES[name], dtype=np.uint8)
            for name in ROLE_NAMES
        ]
    )
    raw_lengths = dense_lengths[users]
    valid_lengths = np.minimum(
        raw_lengths, config.maximum_sequence_length
    ).astype(np.uint16)
    offsets = np.concatenate(
        [
            np.zeros(1, dtype=np.int64),
            np.cumsum(valid_lengths, dtype=np.int64),
        ]
    )
    boundaries = np.zeros(
        (len(users), config.update_count + 2), dtype=np.uint16
    )
    long = role < len(LONG_ROLE_NAMES)
    boundaries[long] = chain_boundaries(
        valid_lengths[long],
        base_prefix=config.base_prefix,
        update_count=config.update_count,
    )
    short_history_end = np.zeros(len(users), dtype=np.uint16)
    short_target_end = np.zeros(len(users), dtype=np.uint16)
    short = role == ROLE_CODES["short_diagnostic"]
    short_history_end[short] = np.maximum(
        config.short_minimum_events // 2,
        valid_lengths[short].astype(np.int64) - 8,
    ).astype(np.uint16)
    short_target_end[short] = valid_lengths[short] - 1
    return {
        "record_user_ids": users.astype(np.int64, copy=False),
        "record_role": role,
        "record_raw_lengths": raw_lengths.astype(np.int32, copy=False),
        "record_valid_lengths": valid_lengths,
        "record_offsets": offsets,
        "edge_last_ordinals": boundaries,
        "short_history_end": short_history_end,
        "short_target_end": short_target_end,
    }


def _event_arrays(rows: int) -> dict[str, np.ndarray]:
    return {
        "item_idx": np.zeros(rows, dtype=np.uint32),
        "behavior": np.zeros(rows, dtype=np.uint8),
        "label": np.zeros(rows, dtype=np.uint8),
        "raw_label": np.zeros(rows, dtype=np.uint8),
        "is_prediction_item": np.zeros(rows, dtype=np.uint8),
        "is_stream_only_fallback": np.zeros(rows, dtype=np.uint8),
        "raw_ordinal": np.zeros(rows, dtype=np.uint16),
    }


def _materialize(
    config: QKStreamChainConfig,
    layout: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object]]:
    read_config = EdgeInputConfig(
        source=config.source,
        member=config.member,
        catalog_cache=config.catalog,
        roles=config.roles_output,
        output=config.corpus_output,
        summary=config.summary_output,
        chunk_size=config.chunk_size,
    )
    fingerprint = source_fingerprint(read_config)
    catalog = load_catalog(read_config, fingerprint)
    users = layout["record_user_ids"]
    offsets = layout["record_offsets"]
    horizons = layout["record_valid_lengths"].astype(np.int64)
    user_to_record = np.full(int(users.max()) + 1, -1, dtype=np.int32)
    user_to_record[users] = np.arange(len(users), dtype=np.int32)
    events = _event_arrays(int(offsets[-1]))
    filled = np.zeros(int(offsets[-1]), dtype=np.bool_)
    seen = np.zeros(0, dtype=np.int32)
    source_rows = 0
    selected_rows = 0
    started = time.perf_counter()
    for chunk_index, chunk in enumerate(read_qk_chunks(read_config), start=1):
        raw_users = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
        positions, seen = consume_user_positions(raw_users, seen)
        in_range = (raw_users >= 0) & (raw_users < len(user_to_record))
        records = np.full(len(raw_users), -1, dtype=np.int32)
        records[in_range] = user_to_record[raw_users[in_range]]
        selected = records >= 0
        if selected.any():
            indices = np.flatnonzero(selected)
            chosen_records = records[selected].astype(np.int64, copy=False)
            chosen_positions = positions[selected]
            within = chosen_positions < horizons[chosen_records]
            if within.any():
                indices = indices[within]
                chosen_records = chosen_records[within]
                chosen_positions = chosen_positions[within]
                destination = offsets[chosen_records] + chosen_positions
                if filled[destination].any():
                    raise ValueError("QK stream event destination overlaps")
                original = chunk["item_id"].to_numpy(
                    dtype=np.int64, copy=False
                )[indices]
                mapped, predicted, _, stream_only = catalog.map(original)
                click = chunk["click"].to_numpy(
                    dtype=np.uint8, copy=False
                )[indices]
                follow = chunk["follow"].to_numpy(
                    dtype=np.uint8, copy=False
                )[indices]
                like = chunk["like"].to_numpy(
                    dtype=np.uint8, copy=False
                )[indices]
                share = chunk["share"].to_numpy(
                    dtype=np.uint8, copy=False
                )[indices]
                raw_label = action_masks(click, follow, like, share) > 0
                events["item_idx"][destination] = mapped.astype(
                    np.uint32, copy=False
                )
                events["behavior"][destination] = behavior_values(
                    click, follow, like, share
                )
                events["label"][destination] = (
                    raw_label & predicted
                ).astype(np.uint8, copy=False)
                events["raw_label"][destination] = raw_label.astype(
                    np.uint8, copy=False
                )
                events["is_prediction_item"][destination] = predicted.astype(
                    np.uint8, copy=False
                )
                events["is_stream_only_fallback"][destination] = (
                    stream_only.astype(np.uint8, copy=False)
                )
                events["raw_ordinal"][destination] = chosen_positions.astype(
                    np.uint16, copy=False
                )
                filled[destination] = True
                selected_rows += len(destination)
        source_rows += len(chunk)
        print(
            f"phase=qk_stream_materialize chunk={chunk_index} "
            f"rows={source_rows:,} selected={selected_rows:,}/"
            f"{len(filled):,} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    if not filled.all():
        raise RuntimeError(
            f"QK stream corpus is missing {np.count_nonzero(~filled):,} rows"
        )
    return events, {
        "source_rows_scanned": source_rows,
        "materialized_rows": len(filled),
        "complete": True,
        "wall_seconds": time.perf_counter() - started,
    }, {
        "path": str(config.catalog),
        "sha256": catalog.file_sha256,
        "prediction_rows": catalog.prediction_rows,
        "context_rows": catalog.context_rows,
        "num_embeddings": len(catalog.original_item_ids) + 1,
        "mapping": (
            "base rows are direct and post-base unseen identities use "
            "SplitMix64 over the frozen context namespace"
        ),
    }


def _window_stats(
    arrays: dict[str, np.ndarray],
    role: str,
    update_count: int,
) -> list[dict[str, object]]:
    records = np.flatnonzero(arrays["record_role"] == ROLE_CODES[role])
    offsets = arrays["record_offsets"]
    boundaries = arrays["edge_last_ordinals"]
    labels = arrays["label"]
    result = []
    for edge in range(1, update_count + 1):
        targets = 0
        records_with_targets = 0
        widths = []
        for record in records:
            left = int(boundaries[record, edge - 1]) + 1
            right = int(boundaries[record, edge]) + 1
            start = int(offsets[record])
            count = int(labels[start + left : start + right].sum())
            targets += count
            records_with_targets += int(count > 0)
            widths.append(right - left)
        width = np.asarray(widths, dtype=np.int64)
        result.append(
            {
                "edge": edge,
                "targets": targets,
                "records": len(records),
                "records_with_targets": records_with_targets,
                "record_coverage": records_with_targets / len(records),
                "window_length": {
                    "minimum": int(width.min()),
                    "median": float(np.median(width)),
                    "p95": float(np.quantile(width, 0.95)),
                    "maximum": int(width.max()),
                },
            }
        )
    return result


def validate_arrays(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    required = {
        "record_user_ids",
        "record_role",
        "record_raw_lengths",
        "record_valid_lengths",
        "record_offsets",
        "edge_last_ordinals",
        "short_history_end",
        "short_target_end",
        "item_idx",
        "behavior",
        "label",
        "raw_label",
        "is_prediction_item",
        "is_stream_only_fallback",
        "raw_ordinal",
    }
    if required.difference(arrays):
        raise ValueError("QK stream corpus arrays are incomplete")
    records = len(arrays["record_user_ids"])
    offsets = arrays["record_offsets"]
    rows = int(offsets[-1]) if len(offsets) else -1
    update_count = int(metadata.get("update_count", -1))
    long = arrays["record_role"] < len(LONG_ROLE_NAMES)
    short = arrays["record_role"] == ROLE_CODES["short_diagnostic"]
    if (
        metadata.get("protocol") != PROTOCOL
        or records < 1
        or update_count < 1
        or arrays["record_user_ids"].dtype != np.int64
        or len(np.unique(arrays["record_user_ids"])) != records
        or arrays["record_role"].shape != (records,)
        or arrays["record_role"].dtype != np.uint8
        or np.any(arrays["record_role"] >= len(ROLE_NAMES))
        or arrays["record_raw_lengths"].shape != (records,)
        or arrays["record_raw_lengths"].dtype != np.int32
        or arrays["record_valid_lengths"].shape != (records,)
        or arrays["record_valid_lengths"].dtype != np.uint16
        or offsets.dtype != np.int64
        or offsets.shape != (records + 1,)
        or offsets[0] != 0
        or not np.array_equal(np.diff(offsets), arrays["record_valid_lengths"])
        or arrays["edge_last_ordinals"].shape
        != (records, update_count + 2)
        or arrays["edge_last_ordinals"].dtype != np.uint16
        or np.any(
            arrays["edge_last_ordinals"][long, 1:]
            <= arrays["edge_last_ordinals"][long, :-1]
        )
        or np.any(arrays["edge_last_ordinals"][short] != 0)
        or np.any(
            arrays["edge_last_ordinals"][long, -1]
            != arrays["record_valid_lengths"][long] - 1
        )
        or np.any(arrays["short_history_end"][long] != 0)
        or np.any(arrays["short_target_end"][long] != 0)
        or np.any(
            arrays["short_history_end"][short]
            >= arrays["short_target_end"][short]
        )
        or any(
            arrays[name].shape != (rows,)
            for name in (
                "item_idx",
                "behavior",
                "label",
                "raw_label",
                "is_prediction_item",
                "is_stream_only_fallback",
                "raw_ordinal",
            )
        )
        or np.any(arrays["item_idx"] < 1)
        or np.any(arrays["behavior"] < 1)
        or np.any(arrays["behavior"] > 5)
        or np.any(arrays["label"] > arrays["raw_label"])
        or np.any(arrays["label"] > arrays["is_prediction_item"])
        or np.any(arrays["raw_label"] > 1)
        or np.any(arrays["is_prediction_item"] > 1)
        or np.any(arrays["is_stream_only_fallback"] > 1)
    ):
        raise ValueError("QK stream corpus layout differs")
    expected = chain_boundaries(
        arrays["record_valid_lengths"][long],
        base_prefix=int(metadata["base_prefix"]),
        update_count=update_count,
    )
    if not np.array_equal(arrays["edge_last_ordinals"][long], expected):
        raise ValueError("QK stream boundary binding differs")
    for record in range(records):
        start = int(offsets[record])
        stop = int(offsets[record + 1])
        if not np.array_equal(
            arrays["raw_ordinal"][start:stop],
            np.arange(stop - start, dtype=np.uint16),
        ):
            raise ValueError("QK stream ordinal coverage differs")


def build_corpus(config: QKStreamChainConfig) -> dict[str, object]:
    if min(
        config.base_prefix,
        config.maximum_sequence_length,
        config.update_count,
        config.stream_train_users,
        config.fit_tuning_users,
        config.qualification_users,
        config.final_users,
        config.short_diagnostic_users,
        config.minimum_long_events,
        config.short_minimum_events,
        config.chunk_size,
    ) < 1:
        raise ValueError("QK stream configuration differs")
    if (
        config.base_prefix >= config.minimum_long_events
        or config.minimum_long_events > config.maximum_sequence_length
        or config.short_minimum_events > config.short_maximum_events
        or config.short_maximum_events >= config.base_prefix
    ):
        raise ValueError("QK stream length contract differs")
    if config.corpus_output.exists():
        corpus = load_corpus(config.corpus_output)
        if not config.summary_output.is_file() or not config.roles_output.is_file():
            raise ValueError("QK stream durable artifacts are incomplete")
        summary = json.loads(config.summary_output.read_text())
        if (
            summary.get("status") != "pass"
            or summary.get("content_sha256") != corpus.content_sha256
            or summary.get("artifact", {}).get("sha256")
            != corpus.file_sha256
        ):
            raise ValueError("QK stream existing summary differs")
        return summary
    if config.summary_output.exists():
        raise FileExistsError(config.summary_output)
    roles, role_document = build_roles(config)
    if config.roles_output.exists():
        existing = json.loads(config.roles_output.read_text())
        if existing != role_document:
            raise ValueError("QK stream frozen roles differ")
    else:
        _atomic_json(config.roles_output, role_document)
    fingerprint = role_document["source"]
    user_ids, lengths, _ = _load_user_lengths(config, fingerprint)
    dense_lengths = _dense_lengths(user_ids, lengths)
    layout = _layout(config, roles, dense_lengths)
    started = time.perf_counter()
    events, scan, catalog = _materialize(config, layout)
    arrays = {**layout, **events}
    metadata = {
        "protocol": PROTOCOL,
        "scientific_result": False,
        "formal_result": False,
        "dataset": "tenrec-qk",
        "source": fingerprint,
        "catalog": catalog,
        "roles": {
            "path": str(config.roles_output),
            "sha256": file_sha256(config.roles_output),
            "selection_salt": config.selection_salt,
            "pairwise_disjoint": True,
        },
        "base_prefix": config.base_prefix,
        "maximum_sequence_length": config.maximum_sequence_length,
        "update_count": config.update_count,
        "model_versions": list(range(config.update_count + 1)),
        "ordering": "official within-user ordinal order",
        "positive_target": "OR(click, follow, like, share) in base-fitted prediction rows",
        "stream_only_mapping": "SplitMix64 into the frozen context namespace",
        "window_semantics": {
            "boundary": "last observed target ordinal",
            "theta_t_training_targets": "(c_(t-1), c_t] on stream_train",
            "theta_t_evaluation_targets": "(c_t, c_(t+1)] on held-out roles",
            "prequential_identity": "evaluation targets for theta_t become training targets for theta_(t+1) only after evaluation",
        },
    }
    validate_arrays(arrays, metadata)
    content_hash = artifact_sha256(arrays)
    metadata["content_sha256"] = content_hash
    config.corpus_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.corpus_output.with_name(
        f".{config.corpus_output.name}.{os.getpid()}.tmp.npz"
    )
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(config.corpus_output)
    summary = {
        **metadata,
        "status": "pass",
        "artifact": {
            "path": str(config.corpus_output),
            "bytes": config.corpus_output.stat().st_size,
            "sha256": file_sha256(config.corpus_output),
            "content_sha256": content_hash,
        },
        "records": {
            name: int(np.count_nonzero(layout["record_role"] == code))
            for name, code in ROLE_CODES.items()
        },
        "rows": len(events["item_idx"]),
        "effective_positive_rows": int(events["label"].sum()),
        "raw_positive_rows": int(events["raw_label"].sum()),
        "stream_train_windows": _window_stats(
            arrays, "stream_train", config.update_count
        ),
        "fit_tuning_windows": _window_stats(
            arrays, "fit_tuning", config.update_count
        ),
        "qualification_windows": _window_stats(
            arrays, "qualification", config.update_count
        ),
        "final_windows": _window_stats(arrays, "final", config.update_count),
        "integrity": {
            "array_sha256": {
                name: typed_array_sha256(value)
                for name, value in sorted(arrays.items())
            },
            "roles_pairwise_disjoint": True,
            "ordinal_coverage_contiguous": True,
            "post_base_selection_uses_labels": False,
            "theta_t_evaluation_precedes_theta_t_plus_1_training": True,
        },
        "scan": scan,
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(config.summary_output, summary)
    return summary


def load_corpus(path: str | Path) -> QKStreamChainCorpus:
    resolved = Path(path)
    arrays, metadata = _load_npz(resolved)
    validate_arrays(arrays, metadata)
    content = artifact_sha256(arrays)
    if metadata.get("content_sha256") != content:
        raise ValueError("QK stream corpus content hash differs")
    return QKStreamChainCorpus(
        path=resolved,
        arrays=arrays,
        metadata=metadata,
        file_sha256=file_sha256(resolved),
        content_sha256=content,
    )
