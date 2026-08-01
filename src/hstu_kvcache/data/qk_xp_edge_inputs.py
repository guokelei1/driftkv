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

PROTOCOL = "evokv_qk_xp_fixed_edge_inputs_development_v1"
ROLE_NAMES = ("theta01", "theta12", "qualification")
FORBIDDEN_ROLE_NAMES = ("fit", "profile", "final")
TENREC_COLUMNS = (
    "user_id",
    "item_id",
    "click",
    "follow",
    "like",
    "share",
)


@dataclass(frozen=True)
class EdgeInputConfig:
    source: Path
    member: str
    catalog_cache: Path
    roles: Path
    output: Path
    summary: Path
    hash_salt: str = "evokv-qk-successor-foundation-v1"
    prediction_catalog_size: int = 250_000
    base_prefix: int = 64
    theta01_history_end: int = 64
    theta01_update_end: int = 96
    theta12_history_end: int = 544
    theta12_update_end: int = 576
    qualification_history_end: int = 64
    qualification_update_end: int = 96
    theta01_users: int = 2_560
    theta12_users: int = 2_048
    qualification_users: int = 512
    chunk_size: int = 2_000_000


@dataclass(frozen=True)
class Catalog:
    original_item_ids: np.ndarray
    dense_map: np.ndarray
    prediction_rows: int
    context_rows: int
    metadata: dict
    file_sha256: str

    def map(
        self,
        original_item_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        in_range = (
            (original_item_ids >= 0)
            & (original_item_ids < len(self.dense_map))
        )
        direct = np.zeros(len(original_item_ids), dtype=np.int32)
        direct[in_range] = self.dense_map[original_item_ids[in_range]]
        base_seen = direct > 0
        predicted = base_seen & (direct <= self.prediction_rows)
        exact_context = base_seen & ~predicted
        stream_only = ~base_seen
        mapped = direct.copy()
        if stream_only.any():
            mapped[stream_only] = (
                self.prediction_rows
                + 1
                + (
                    splitmix64(original_item_ids[stream_only])
                    % np.uint64(self.context_rows)
                ).astype(np.int64)
            ).astype(np.int32)
        return mapped, predicted, exact_context, stream_only


def validate_config(config: EdgeInputConfig) -> None:
    values = (
        config.prediction_catalog_size,
        config.base_prefix,
        config.theta01_history_end,
        config.theta01_update_end,
        config.theta12_history_end,
        config.theta12_update_end,
        config.qualification_history_end,
        config.qualification_update_end,
        config.theta01_users,
        config.theta12_users,
        config.qualification_users,
        config.chunk_size,
    )
    if min(values) < 1:
        raise ValueError("edge input dimensions must be positive")
    boundaries = (
        (
            config.theta01_history_end,
            config.theta01_update_end,
        ),
        (
            config.theta12_history_end,
            config.theta12_update_end,
        ),
        (
            config.qualification_history_end,
            config.qualification_update_end,
        ),
    )
    if any(history >= update for history, update in boundaries):
        raise ValueError("every edge must have a nonempty update window")
    if (
        config.theta01_history_end != config.base_prefix
        or config.qualification_history_end != config.base_prefix
    ):
        raise ValueError("theta01 and qualification must start from base")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def typed_array_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode())
    digest.update(json.dumps(list(canonical.shape)).encode())
    digest.update(canonical.view(np.uint8))
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


def source_fingerprint(config: EdgeInputConfig) -> dict:
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


def source_matches(left: dict, right: dict) -> bool:
    keys = (
        "member",
        "member_size_bytes",
        "member_compressed_size_bytes",
        "member_crc32",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {
            name: source[name].copy()
            for name in source.files
            if name != "metadata_json"
        }
        metadata = json.loads(str(source["metadata_json"].item()))
    return arrays, metadata


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def load_catalog(
    config: EdgeInputConfig,
    fingerprint: dict,
) -> Catalog:
    arrays, metadata = load_npz(config.catalog_cache)
    if "base_entity_original_item_ids" not in arrays:
        raise ValueError("catalog lacks base entity item ids")
    original = np.asarray(
        arrays["base_entity_original_item_ids"],
        dtype=np.int64,
    )
    if (
        len(original) <= config.prediction_catalog_size
        or np.any(original < 0)
        or len(np.unique(original)) != len(original)
    ):
        raise ValueError("base entity catalog is invalid")
    if int(metadata.get("base_prefix_raw_events", -1)) != config.base_prefix:
        raise ValueError("catalog base prefix differs")
    if not source_matches(metadata.get("source", {}), fingerprint):
        raise ValueError("catalog source differs")
    prediction_rows = int(metadata.get("num_prediction_items", -1))
    context_rows = int(metadata.get("context_entity_rows", -1))
    if prediction_rows != config.prediction_catalog_size:
        raise ValueError("catalog prediction row count differs")
    if prediction_rows + context_rows != len(original):
        raise ValueError("catalog prediction/context partition is invalid")
    dense = np.zeros(int(original.max()) + 1, dtype=np.int32)
    dense[original] = np.arange(1, len(original) + 1, dtype=np.int32)
    return Catalog(
        original_item_ids=original,
        dense_map=dense,
        prediction_rows=prediction_rows,
        context_rows=context_rows,
        metadata=metadata,
        file_sha256=file_sha256(config.catalog_cache),
    )


def load_roles(config: EdgeInputConfig) -> tuple[dict[str, np.ndarray], dict]:
    document = json.loads(config.roles.read_text())
    if document.get("hash_salt") != config.hash_salt:
        raise ValueError("frozen role salt differs")
    source_roles = document.get("roles", {})
    required = set(ROLE_NAMES + FORBIDDEN_ROLE_NAMES)
    if not required.issubset(source_roles):
        raise ValueError("frozen role file is incomplete")
    roles = {
        name: np.asarray(source_roles[name]["user_ids"], dtype=np.int64)
        for name in required
    }
    expected_counts = {
        "theta01": config.theta01_users,
        "theta12": config.theta12_users,
        "qualification": config.qualification_users,
    }
    for name, values in roles.items():
        record = source_roles[name]
        if len(values) != int(record["count"]):
            raise ValueError(f"frozen role count differs for {name}")
        if array_sha256(values) != record["user_ids_sha256"]:
            raise ValueError(f"frozen role hash differs for {name}")
        if len(np.unique(values)) != len(values) or np.any(values < 0):
            raise ValueError(f"frozen role ids are invalid for {name}")
    for name, count in expected_counts.items():
        if len(roles[name]) != count:
            raise ValueError(f"required role count differs for {name}")
    combined = np.concatenate([roles[name] for name in required])
    if len(np.unique(combined)) != len(combined):
        raise ValueError("post-base roles are not pairwise disjoint")
    return roles, document


def read_qk_chunks(config: EdgeInputConfig) -> Iterator[pd.DataFrame]:
    dtypes = {
        "user_id": "int32",
        "item_id": "int32",
        "click": "int8",
        "follow": "int8",
        "like": "int8",
        "share": "int8",
    }
    with (
        zipfile.ZipFile(config.source) as archive,
        archive.open(config.member) as stream,
    ):
        yield from pd.read_csv(
            stream,
            usecols=list(TENREC_COLUMNS),
            dtype=dtypes,
            chunksize=config.chunk_size,
        )


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


def behavior_values(
    click: np.ndarray,
    follow: np.ndarray,
    like: np.ndarray,
    share: np.ndarray,
) -> np.ndarray:
    output = np.ones(len(click), dtype=np.uint8)
    output[click.astype(np.bool_, copy=False)] = 2
    output[like.astype(np.bool_, copy=False)] = 3
    output[follow.astype(np.bool_, copy=False)] = 4
    output[share.astype(np.bool_, copy=False)] = 5
    return output


def action_masks(
    click: np.ndarray,
    follow: np.ndarray,
    like: np.ndarray,
    share: np.ndarray,
) -> np.ndarray:
    return (
        click.astype(np.uint8, copy=False)
        | (follow.astype(np.uint8, copy=False) << np.uint8(1))
        | (like.astype(np.uint8, copy=False) << np.uint8(2))
        | (share.astype(np.uint8, copy=False) << np.uint8(3))
    )


def record_layout(
    config: EdgeInputConfig,
    roles: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    role_code = np.concatenate(
        [
            np.full(len(roles[name]), index, dtype=np.uint8)
            for index, name in enumerate(ROLE_NAMES)
        ]
    )
    user_ids = np.concatenate([roles[name] for name in ROLE_NAMES])
    history_end_values = (
        config.theta01_history_end,
        config.theta12_history_end,
        config.qualification_history_end,
    )
    update_end_values = (
        config.theta01_update_end,
        config.theta12_update_end,
        config.qualification_update_end,
    )
    history_end = np.asarray(
        [history_end_values[int(code)] for code in role_code],
        dtype=np.uint16,
    )
    update_end = np.asarray(
        [update_end_values[int(code)] for code in role_code],
        dtype=np.uint16,
    )
    offsets = np.zeros(len(user_ids) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(update_end, dtype=np.int64)
    return {
        "record_user_ids": user_ids,
        "record_role": role_code,
        "record_offsets": offsets,
        "record_history_start": np.zeros(len(user_ids), dtype=np.uint16),
        "record_history_end": history_end,
        "record_update_start": history_end.copy(),
        "record_update_end": update_end,
    }


def initialize_event_arrays(rows: int) -> dict[str, np.ndarray]:
    return {
        "item_idx": np.zeros(rows, dtype=np.uint32),
        "original_item_id": np.zeros(rows, dtype=np.int32),
        "behavior": np.zeros(rows, dtype=np.uint8),
        "action_mask": np.zeros(rows, dtype=np.uint8),
        "raw_label": np.zeros(rows, dtype=np.uint8),
        "label": np.zeros(rows, dtype=np.uint8),
        "raw_ordinal": np.zeros(rows, dtype=np.uint16),
        "is_prediction_item": np.zeros(rows, dtype=np.uint8),
        "is_stream_only_fallback": np.zeros(rows, dtype=np.uint8),
    }


def materialize(
    config: EdgeInputConfig,
    catalog: Catalog,
    roles: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict]:
    layout = record_layout(config, roles)
    record_users = layout["record_user_ids"]
    offsets = layout["record_offsets"]
    horizons = layout["record_update_end"].astype(np.int64)
    user_to_record = np.full(
        int(record_users.max()) + 1,
        -1,
        dtype=np.int32,
    )
    user_to_record[record_users] = np.arange(
        len(record_users),
        dtype=np.int32,
    )
    events = initialize_event_arrays(int(offsets[-1]))
    filled = np.zeros(int(offsets[-1]), dtype=np.bool_)
    seen = np.zeros(0, dtype=np.int32)
    source_rows = 0
    selected_rows_seen = 0
    post_window_ignored = 0
    started = time.perf_counter()
    for chunk_index, chunk in enumerate(read_qk_chunks(config), start=1):
        users = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
        positions, seen = consume_user_positions(users, seen)
        in_range = (users >= 0) & (users < len(user_to_record))
        records = np.full(len(users), -1, dtype=np.int32)
        records[in_range] = user_to_record[users[in_range]]
        selected = records >= 0
        selected_rows_seen += int(np.count_nonzero(selected))
        if selected.any():
            selected_indices = np.flatnonzero(selected)
            selected_records = records[selected]
            selected_positions = positions[selected]
            within = selected_positions < horizons[selected_records]
            post_window_ignored += int(np.count_nonzero(~within))
            if within.any():
                chosen = selected_indices[within]
                chosen_records = selected_records[within].astype(
                    np.int64,
                    copy=False,
                )
                chosen_positions = selected_positions[within]
                destination = offsets[chosen_records] + chosen_positions
                if filled[destination].any():
                    raise ValueError("duplicate event destination")
                original_items = chunk["item_id"].to_numpy(
                    dtype=np.int64,
                    copy=False,
                )[chosen]
                mapped, predicted, _, stream_only = catalog.map(
                    original_items
                )
                click = chunk["click"].to_numpy(
                    dtype=np.uint8,
                    copy=False,
                )[chosen]
                follow = chunk["follow"].to_numpy(
                    dtype=np.uint8,
                    copy=False,
                )[chosen]
                like = chunk["like"].to_numpy(
                    dtype=np.uint8,
                    copy=False,
                )[chosen]
                share = chunk["share"].to_numpy(
                    dtype=np.uint8,
                    copy=False,
                )[chosen]
                actions = action_masks(click, follow, like, share)
                raw_label = actions > 0
                events["item_idx"][destination] = mapped.astype(
                    np.uint32,
                    copy=False,
                )
                events["original_item_id"][destination] = (
                    original_items.astype(np.int32, copy=False)
                )
                events["behavior"][destination] = behavior_values(
                    click,
                    follow,
                    like,
                    share,
                )
                events["action_mask"][destination] = actions
                events["raw_label"][destination] = raw_label.astype(
                    np.uint8,
                    copy=False,
                )
                events["label"][destination] = (
                    raw_label & predicted
                ).astype(np.uint8, copy=False)
                events["raw_ordinal"][destination] = (
                    chosen_positions.astype(np.uint16, copy=False)
                )
                events["is_prediction_item"][destination] = (
                    predicted.astype(np.uint8, copy=False)
                )
                events["is_stream_only_fallback"][destination] = (
                    stream_only.astype(np.uint8, copy=False)
                )
                filled[destination] = True
        source_rows += len(chunk)
        print(
            f"phase=xp_edge_inputs chunks={chunk_index} "
            f"source_rows={source_rows:,} "
            f"materialized={int(filled.sum()):,}/{len(filled):,} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    if not filled.all():
        missing = int(np.count_nonzero(~filled))
        raise ValueError(f"fixed edge inputs are missing {missing} rows")
    for record in range(len(record_users)):
        start = int(offsets[record])
        end = int(offsets[record + 1])
        expected = np.arange(end - start, dtype=np.uint16)
        if not np.array_equal(events["raw_ordinal"][start:end], expected):
            raise ValueError("record ordinal coverage is not contiguous")
    arrays = {**layout, **events}
    audit = {
        "source_rows_scanned": source_rows,
        "selected_role_rows_seen": selected_rows_seen,
        "selected_post_window_rows_ignored": post_window_ignored,
        "materialized_rows": len(filled),
        "materialized_complete": bool(filled.all()),
        "post_window_values_materialized": False,
        "forbidden_role_values_materialized": False,
        "wall_seconds": time.perf_counter() - started,
    }
    return arrays, audit


def phase_counts(
    arrays: dict[str, np.ndarray],
) -> dict:
    offsets = arrays["record_offsets"]
    role_codes = arrays["record_role"]
    history_end = arrays["record_history_end"]
    result = {}
    for code, name in enumerate(ROLE_NAMES):
        records = np.flatnonzero(role_codes == code)
        role_result = {}
        for phase in ("history", "update"):
            indices = []
            for record in records:
                start = int(offsets[record])
                split = start + int(history_end[record])
                end = int(offsets[record + 1])
                left, right = (
                    (start, split) if phase == "history" else (split, end)
                )
                indices.append(np.arange(left, right, dtype=np.int64))
            selected = (
                np.concatenate(indices)
                if indices
                else np.empty(0, dtype=np.int64)
            )
            role_result[phase] = {
                "rows": len(selected),
                "raw_positive_rows": int(
                    arrays["raw_label"][selected].sum()
                ),
                "effective_positive_rows": int(
                    arrays["label"][selected].sum()
                ),
                "prediction_rows": int(
                    arrays["is_prediction_item"][selected].sum()
                ),
                "stream_only_fallback_rows": int(
                    arrays["is_stream_only_fallback"][selected].sum()
                ),
            }
        result[name] = role_result
    return result


def run(config: EdgeInputConfig) -> dict:
    validate_config(config)
    fingerprint = source_fingerprint(config)
    catalog = load_catalog(config, fingerprint)
    roles, role_document = load_roles(config)
    started = time.perf_counter()
    arrays, scan = materialize(config, catalog, roles)
    content_hash = artifact_sha256(arrays)
    metadata = {
        "protocol": PROTOCOL,
        "scientific_result": False,
        "formal_result": False,
        "dataset": "tenrec-qk",
        "source": fingerprint,
        "catalog": {
            "path": str(config.catalog_cache),
            "file_sha256": catalog.file_sha256,
            "base_entity_rows": len(catalog.original_item_ids),
            "prediction_rows": catalog.prediction_rows,
            "context_rows": catalog.context_rows,
            "base_entity_item_ids_sha256": catalog.metadata[
                "base_entity_item_ids_sha256"
            ],
            "mapping": (
                "base entity rows are direct; items first seen after the "
                "base window use SplitMix64 over existing context rows"
            ),
        },
        "frozen_roles": {
            "path": str(config.roles),
            "file_sha256": file_sha256(config.roles),
            "hash_salt": config.hash_salt,
            "source_protocol": role_document["protocol"],
            "included": list(ROLE_NAMES),
            "excluded": list(FORBIDDEN_ROLE_NAMES),
            "included_user_ids_sha256": {
                name: array_sha256(roles[name]) for name in ROLE_NAMES
            },
        },
        "boundaries": {
            "theta01": {
                "history": [0, config.theta01_history_end],
                "update": [
                    config.theta01_history_end,
                    config.theta01_update_end,
                ],
            },
            "theta12": {
                "history": [0, config.theta12_history_end],
                "update": [
                    config.theta12_history_end,
                    config.theta12_update_end,
                ],
            },
            "qualification": {
                "history": [0, config.qualification_history_end],
                "update": [
                    config.qualification_history_end,
                    config.qualification_update_end,
                ],
            },
        },
        "semantics": {
            "ordering": "official within-user ordinal order",
            "behavior": (
                "1 exposure, 2 click, 3 like, 4 follow, 5 share with "
                "later listed active feedback taking precedence"
            ),
            "action_mask": (
                "bit 0 click, bit 1 follow, bit 2 like, bit 3 share"
            ),
            "raw_label": "OR(click, follow, like, share)",
            "label": (
                "raw_label for prediction rows and zero for base-context "
                "or stream-only fallback rows"
            ),
            "post_window_values_materialized": False,
        },
        "content_sha256": content_hash,
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    mapped = arrays["item_idx"]
    prediction = arrays["is_prediction_item"].astype(np.bool_)
    fallback = arrays["is_stream_only_fallback"].astype(np.bool_)
    direct = ~fallback
    exact_context = direct & ~prediction
    summary = {
        **metadata,
        "status": "pass",
        "artifact": {
            "path": str(config.output),
            "bytes": config.output.stat().st_size,
            "file_sha256": file_sha256(config.output),
            "content_sha256": content_hash,
        },
        "records": {
            name: int(np.count_nonzero(arrays["record_role"] == code))
            for code, name in enumerate(ROLE_NAMES)
        },
        "rows": {
            "total": len(mapped),
            "prediction": int(np.count_nonzero(prediction)),
            "exact_base_context": int(np.count_nonzero(exact_context)),
            "stream_only_fallback": int(np.count_nonzero(fallback)),
            "raw_positive": int(arrays["raw_label"].sum()),
            "effective_positive": int(arrays["label"].sum()),
        },
        "phase_counts": phase_counts(arrays),
        "coverage": {
            "unique_mapped_rows": int(len(np.unique(mapped))),
            "unique_direct_base_rows": int(len(np.unique(mapped[direct]))),
            "unique_prediction_rows": int(
                len(np.unique(mapped[prediction]))
            ),
            "unique_exact_base_context_rows": int(
                len(np.unique(mapped[exact_context]))
            ),
            "unique_stream_only_original_items": int(
                len(np.unique(arrays["original_item_id"][fallback]))
            ),
            "unique_fallback_context_rows": int(
                len(np.unique(mapped[fallback]))
            ),
            "mapped_catalog_fraction": float(
                len(np.unique(mapped)) / len(catalog.original_item_ids)
            ),
        },
        "integrity": {
            "array_sha256": {
                name: typed_array_sha256(value)
                for name, value in sorted(arrays.items())
            },
            "included_roles_pairwise_disjoint": True,
            "included_roles_disjoint_from_fit_profile_final": True,
            "all_records_complete": scan["materialized_complete"],
            "all_ordinals_contiguous_from_zero": True,
            "post_window_values_materialized": False,
            "forbidden_role_values_materialized": False,
        },
        "scan": scan,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(config.summary, summary)
    return summary
