from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

PROTOCOL: Final = "evokv_qb_large_multifield_data_development_v0"
SOURCE_COLUMNS: Final = (
    "user_id",
    "item_id",
    "click",
    "follow",
    "like",
    "share",
    "video_category",
    "watching_times",
    "gender",
    "age",
)
FIELD_COLUMNS: Final = {
    "item": ("item_id",),
    "user": ("user_id",),
    "user_item": ("user_id", "item_id"),
    "user_category": ("user_id", "video_category_code"),
    "item_demographic": ("item_id", "gender", "age"),
    "item_behavior": ("item_id", "behavior_signature"),
    "item_watch": ("item_id", "watch_bucket"),
    "user_watch": ("user_id", "watch_bucket"),
    "user_item_behavior": ("user_id", "item_id", "behavior_signature"),
}
PROFILE_DEFINITIONS: Final = {
    "mf5_e8192": {
        "fields": (
            "item",
            "user",
            "user_item",
            "user_category",
            "item_demographic",
        ),
        "embedding_width": 8192,
    },
    "mf8_e6656": {
        "fields": tuple(FIELD_COLUMNS)[:8],
        "embedding_width": 6656,
    },
    "mf9_e4096": {
        "fields": (
            "item",
            "user",
            "user_item",
            "user_category",
            "item_demographic",
            "item_behavior",
            "item_watch",
            "user_watch",
            "user_item_behavior",
        ),
        "embedding_width": 4096,
    },
}
ROLE_NAMES: Final = ("train", "tuning", "qualification")


@dataclass(frozen=True)
class QBLargeProfile:
    name: str
    fields: tuple[str, ...]
    embedding_width: int

    @property
    def feature_count(self) -> int:
        return len(self.fields)


@dataclass(frozen=True)
class QBLargeCatalog:
    profile: QBLargeProfile
    keys: dict[str, np.ndarray]
    offsets: dict[str, int]
    metadata: dict[str, object]

    @property
    def semantic_rows(self) -> int:
        return sum(len(self.keys[field]) for field in self.profile.fields)

    @property
    def num_embeddings(self) -> int:
        return self.semantic_rows + 1

    @property
    def num_prediction_items(self) -> int:
        return len(self.keys["item"])

    def map_field(
        self,
        field: str,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if field not in self.profile.fields:
            raise ValueError("QB feature field is outside the profile")
        queries = np.ascontiguousarray(values, dtype=np.int64)
        catalog = self.keys[field]
        if queries.ndim != 2 or queries.shape[1] != catalog.shape[1]:
            raise ValueError("QB feature query width differs")
        catalog_view = structured_view(catalog)
        query_view = structured_view(queries)
        locations = np.searchsorted(catalog_view, query_view)
        bounded = np.minimum(locations, len(catalog) - 1)
        direct = (locations < len(catalog)) & np.all(
            catalog[bounded] == queries,
            axis=1,
        )
        fallback = field_hash(queries, field) % np.uint64(len(catalog))
        local = np.where(direct, bounded, fallback.astype(np.int64))
        rows = self.offsets[field] + local
        return rows.astype(np.uint32), direct

    def map_frame(
        self,
        frame: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feature_rows = []
        item_rows = None
        item_direct = None
        for field in self.profile.fields:
            values = frame.loc[:, list(FIELD_COLUMNS[field])].to_numpy(
                dtype=np.int64,
                copy=False,
            )
            rows, direct = self.map_field(field, values)
            feature_rows.append(rows)
            if field == "item":
                item_rows = rows
                item_direct = direct
        if item_rows is None or item_direct is None:
            raise ValueError("QB feature profile lacks the item namespace")
        return (
            np.stack(feature_rows, axis=1),
            item_rows,
            item_direct,
        )


def profile_from_name(name: str) -> QBLargeProfile:
    if name not in PROFILE_DEFINITIONS:
        raise ValueError("QB large profile is unknown")
    value = PROFILE_DEFINITIONS[name]
    return QBLargeProfile(
        name=name,
        fields=tuple(value["fields"]),
        embedding_width=int(value["embedding_width"]),
    )


def splitmix64(values: np.ndarray) -> np.ndarray:
    hashed = values.astype(np.uint64, copy=True)
    hashed = hashed + np.uint64(0x9E3779B97F4A7C15)
    hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return hashed ^ (hashed >> np.uint64(31))


def field_hash(values: np.ndarray, field: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(field.encode()).digest()[:8], "little")
    hashed = np.full(len(values), np.uint64(seed), dtype=np.uint64)
    for column in range(values.shape[1]):
        component = splitmix64(values[:, column].astype(np.uint64, copy=False))
        hashed = splitmix64(hashed ^ component)
    return hashed


def structured_view(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype="<i8")
    dtype = np.dtype([(f"k{index}", "<i8") for index in range(values.shape[1])])
    return contiguous.view(dtype).reshape(-1)


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(list(contiguous.shape)).encode())
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def artifact_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode())
        digest.update(array_sha256(arrays[name]).encode())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_identity(path: Path, member: str) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
    return {
        "path": str(path.resolve()),
        "archive_bytes": path.stat().st_size,
        "member": member,
        "member_bytes": info.file_size,
        "member_compressed_bytes": info.compress_size,
        "member_crc32": f"{info.CRC:08x}",
    }


def load_qb_frame(
    path: Path,
    member: str,
    watch_bucket_maximum: int = 15,
) -> pd.DataFrame:
    dtypes = {
        "user_id": "int64",
        "item_id": "int64",
        "click": "int8",
        "follow": "int8",
        "like": "int8",
        "share": "int8",
        "video_category": "string",
        "watching_times": "int64",
        "gender": "int8",
        "age": "int8",
    }
    with zipfile.ZipFile(path) as archive, archive.open(member) as source:
        frame = pd.read_csv(source, usecols=SOURCE_COLUMNS, dtype=dtypes)
    frame["raw_ordinal"] = frame.groupby("user_id", sort=False).cumcount()
    user_lengths = frame.groupby("user_id", sort=False).size()
    frame["user_length"] = frame["user_id"].map(user_lengths)
    category = pd.to_numeric(frame["video_category"], errors="coerce")
    frame["video_category_code"] = category.fillna(2).astype(np.int8)
    frame["watch_bucket"] = np.minimum(
        frame["watching_times"].to_numpy(dtype=np.int64, copy=False),
        watch_bucket_maximum,
    )
    frame["behavior_signature"] = (
        frame["click"].to_numpy(dtype=np.int64, copy=False)
        | (frame["follow"].to_numpy(dtype=np.int64, copy=False) << 1)
        | (frame["like"].to_numpy(dtype=np.int64, copy=False) << 2)
        | (frame["share"].to_numpy(dtype=np.int64, copy=False) << 3)
    )
    return frame


def behavior_values(frame: pd.DataFrame) -> np.ndarray:
    result = np.ones(len(frame), dtype=np.uint8)
    for column, value in (("click", 2), ("like", 3), ("follow", 4), ("share", 5)):
        result[frame[column].to_numpy(dtype=np.uint8, copy=False) > 0] = value
    return result


def positive_values(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["click"].to_numpy(dtype=np.uint8, copy=False)
        | frame["follow"].to_numpy(dtype=np.uint8, copy=False)
        | frame["like"].to_numpy(dtype=np.uint8, copy=False)
        | frame["share"].to_numpy(dtype=np.uint8, copy=False)
    ) > 0


def build_catalog(
    frame: pd.DataFrame,
    profile: QBLargeProfile,
    base_prefix: int,
) -> QBLargeCatalog:
    base = frame[frame["raw_ordinal"] < base_prefix]
    eligible = base[base["raw_ordinal"] + 1 < np.minimum(base["user_length"], base_prefix)]
    keys = {}
    offsets = {}
    offset = 1
    for field in profile.fields:
        source = eligible
        if field == "item":
            source = pd.concat(
                (eligible, base.loc[positive_values(base)]),
                ignore_index=True,
            )
        values = source.loc[:, list(FIELD_COLUMNS[field])].to_numpy(dtype=np.int64, copy=False)
        unique = np.unique(values, axis=0)
        if len(unique) == 0:
            raise ValueError("QB feature namespace is empty")
        keys[field] = np.ascontiguousarray(unique, dtype=np.int64)
        offsets[field] = offset
        offset += len(unique)
    metadata = {
        "protocol": PROTOCOL,
        "scientific_result": False,
        "formal_result": False,
        "profile": profile.name,
        "fields": list(profile.fields),
        "embedding_width": profile.embedding_width,
        "base_prefix": base_prefix,
        "base_rows": len(base),
        "optimizer_eligible_base_rows": len(eligible),
        "semantic_rows": offset - 1,
        "num_embeddings": offset,
        "num_prediction_items": len(keys["item"]),
        "field_offsets": offsets,
        "field_rows": {field: len(keys[field]) for field in profile.fields},
        "field_keys_sha256": {field: array_sha256(keys[field]) for field in profile.fields},
        "unseen_mapping": "field-local SplitMix64 into base-only direct rows",
        "video_category_mapping": {"0": 0, "1": 1, "missing_or_N": 2},
    }
    return QBLargeCatalog(
        profile=profile,
        keys=keys,
        offsets=offsets,
        metadata=metadata,
    )


def save_catalog(path: Path, catalog: QBLargeCatalog) -> dict[str, object]:
    arrays = {f"keys_{field}": catalog.keys[field] for field in catalog.profile.fields}
    content_hash = artifact_sha256(arrays)
    metadata = {**catalog.metadata, "content_sha256": content_hash}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "content_sha256": content_hash,
    }


def load_catalog(path: Path) -> QBLargeCatalog:
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        profile = profile_from_name(str(metadata["profile"]))
        keys = {
            field: np.asarray(source[f"keys_{field}"], dtype=np.int64) for field in profile.fields
        }
    arrays = {f"keys_{field}": keys[field] for field in profile.fields}
    if (
        metadata.get("protocol") != PROTOCOL
        or metadata.get("content_sha256") != artifact_sha256(arrays)
        or metadata.get("fields") != list(profile.fields)
        or int(metadata.get("embedding_width", -1)) != profile.embedding_width
    ):
        raise ValueError("QB feature catalog binding differs")
    offsets = {field: int(metadata["field_offsets"][field]) for field in profile.fields}
    catalog = QBLargeCatalog(profile=profile, keys=keys, offsets=offsets, metadata=metadata)
    if catalog.num_embeddings != int(metadata["num_embeddings"]):
        raise ValueError("QB feature catalog size differs")
    return catalog


def stable_role_users(
    frame: pd.DataFrame,
    required_horizon: int,
    users: int,
    train_users: int,
    tuning_users: int,
    qualification_users: int,
    salt: str,
) -> dict[str, np.ndarray]:
    lengths = frame.groupby("user_id", sort=False).size()
    eligible = lengths[lengths >= required_horizon].index.to_numpy(dtype=np.int64)
    if users != train_users + tuning_users + qualification_users or len(eligible) < users:
        raise ValueError("QB role counts exceed the fixed-horizon cohort")
    salt_value = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], "little")
    hashes = splitmix64(eligible.astype(np.uint64) ^ np.uint64(salt_value))
    selected = eligible[np.argsort(hashes, kind="stable")[:users]]
    return {
        "train": selected[:train_users],
        "tuning": selected[train_users : train_users + tuning_users],
        "qualification": selected[train_users + tuning_users :],
    }


def _ordered_records(
    frame: pd.DataFrame,
    user_ids: np.ndarray,
    horizon: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    order = {int(user): index for index, user in enumerate(user_ids.tolist())}
    selected = frame[frame["user_id"].isin(user_ids) & (frame["raw_ordinal"] < horizon)].copy()
    selected["record_order"] = selected["user_id"].map(order)
    selected.sort_values(["record_order", "raw_ordinal"], inplace=True, kind="stable")
    counts = selected.groupby("record_order", sort=True).size().to_numpy(dtype=np.int64)
    if len(counts) != len(user_ids) or np.any(counts != horizon):
        raise ValueError("QB fixed-horizon materialization is incomplete")
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    return selected, offsets


def _base_records(
    frame: pd.DataFrame,
    base_prefix: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    lengths = frame.groupby("user_id", sort=False).size()
    users = lengths[lengths >= 2].index.to_numpy(dtype=np.int64)
    order = {int(user): index for index, user in enumerate(users.tolist())}
    selected = frame[frame["user_id"].isin(users) & (frame["raw_ordinal"] < base_prefix)].copy()
    selected["record_order"] = selected["user_id"].map(order)
    selected.sort_values(["record_order", "raw_ordinal"], inplace=True, kind="stable")
    counts = selected.groupby("record_order", sort=True).size().to_numpy(dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    return selected, users, offsets


def _event_arrays(
    frame: pd.DataFrame,
    catalog: QBLargeCatalog,
) -> dict[str, np.ndarray]:
    feature_ids, target_item_ids, item_direct = catalog.map_frame(frame)
    raw_label = positive_values(frame)
    return {
        "feature_ids": feature_ids.astype(np.uint32, copy=False),
        "target_item_ids": target_item_ids.astype(np.uint32, copy=False),
        "behavior": behavior_values(frame),
        "raw_label": raw_label.astype(np.uint8),
        "label": (raw_label & item_direct).astype(np.uint8),
        "raw_ordinal": frame["raw_ordinal"].to_numpy(dtype=np.uint16, copy=True),
        "is_prediction_item": item_direct.astype(np.uint8),
    }


def build_cooccurrence(
    frame: pd.DataFrame,
    catalog: QBLargeCatalog,
    base_prefix: int,
) -> dict[str, np.ndarray]:
    base = frame[frame["raw_ordinal"] < base_prefix]
    eligible = frame[
        (frame["raw_ordinal"] < base_prefix)
        & (frame["raw_ordinal"] + 1 < np.minimum(frame["user_length"], base_prefix))
    ]
    feature_ids, _, direct = catalog.map_frame(eligible)
    if not direct.all():
        raise ValueError("QB base cooccurrence contains a fallback item row")
    item_index = catalog.profile.fields.index("item")
    user_index = catalog.profile.fields.index("user")
    positive = np.zeros(catalog.num_embeddings, dtype=np.uint32)
    users = np.zeros(catalog.num_embeddings, dtype=np.int64)
    event_users = eligible["user_id"].to_numpy(dtype=np.int64, copy=False)
    for field_index in range(catalog.profile.feature_count):
        rows = feature_ids[:, field_index]
        _, first = np.unique(rows, return_index=True)
        selected_rows = rows[first]
        selected_positive = feature_ids[
            first,
            user_index if field_index == item_index else item_index,
        ]
        positive[selected_rows] = selected_positive
        users[selected_rows] = event_users[first]
    target_events = base.loc[positive_values(base)]
    target_features, _, target_direct = catalog.map_frame(target_events)
    if not target_direct.all():
        raise ValueError("QB base target cooccurrence contains a fallback item row")
    target_items = target_features[:, item_index]
    missing = positive[target_items] == 0
    if np.any(missing):
        selected_items = target_items[missing]
        selected_users = target_features[missing, user_index]
        selected_event_users = target_events["user_id"].to_numpy(dtype=np.int64, copy=False)[
            missing
        ]
        _, first = np.unique(selected_items, return_index=True)
        positive[selected_items[first]] = selected_users[first]
        users[selected_items[first]] = selected_event_users[first]
    anchor = np.arange(1, catalog.num_embeddings, dtype=np.uint32)
    if np.any(positive[1:] == 0) or np.any(positive[1:] == anchor):
        raise ValueError("QB base cooccurrence does not cover every semantic row")
    return {
        "anchor_row": anchor,
        "positive_row": positive[1:],
        "occurrence_user_id": users[1:],
    }


def materialize_corpus(
    frame: pd.DataFrame,
    catalog: QBLargeCatalog,
    *,
    base_prefix: int = 64,
    required_horizon: int = 104,
    users: int = 5000,
    train_users: int = 3500,
    tuning_users: int = 500,
    qualification_users: int = 1000,
    role_salt: str = "evokv-qb-large-multifield-v0",
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    roles = stable_role_users(
        frame,
        required_horizon,
        users,
        train_users,
        tuning_users,
        qualification_users,
        role_salt,
    )
    base_frame, base_users, base_offsets = _base_records(frame, base_prefix)
    base_arrays = _event_arrays(base_frame, catalog)
    role_users = np.concatenate([roles[name] for name in ROLE_NAMES])
    role_frame, role_offsets = _ordered_records(frame, role_users, required_horizon)
    role_arrays = _event_arrays(role_frame, catalog)
    role_codes = np.concatenate(
        [np.full(len(roles[name]), index, dtype=np.uint8) for index, name in enumerate(ROLE_NAMES)]
    )
    cooccurrence = build_cooccurrence(frame, catalog, base_prefix)
    arrays = {
        "base_record_user_ids": base_users,
        "base_record_offsets": base_offsets,
        **{f"base_{name}": value for name, value in base_arrays.items()},
        "role_record_user_ids": role_users,
        "role_record_role": role_codes,
        "role_record_offsets": role_offsets,
        **{f"role_{name}": value for name, value in role_arrays.items()},
        **{f"cooccurrence_{name}": value for name, value in cooccurrence.items()},
    }
    metadata = {
        "protocol": PROTOCOL,
        "scientific_result": False,
        "formal_result": False,
        "dataset": "tenrec-qb",
        "profile": catalog.profile.name,
        "embedding_width": catalog.profile.embedding_width,
        "feature_fields": list(catalog.profile.fields),
        "num_embeddings": catalog.num_embeddings,
        "num_prediction_items": catalog.num_prediction_items,
        "base_prefix": base_prefix,
        "required_horizon": required_horizon,
        "roles": {
            name: {
                "users": len(roles[name]),
                "user_ids_sha256": array_sha256(roles[name]),
            }
            for name in ROLE_NAMES
        },
        "roles_pairwise_disjoint": len(np.unique(role_users)) == len(role_users),
        "role_salt": role_salt,
        "base_users": len(base_users),
        "base_events": len(base_frame),
        "role_events": len(role_frame),
        "cooccurrence_rows": len(cooccurrence["anchor_row"]),
        "catalog_content_sha256": catalog.metadata["content_sha256"],
        "content_sha256": artifact_sha256(arrays),
    }
    return arrays, metadata


def extend_role_horizon(
    frame: pd.DataFrame,
    catalog: QBLargeCatalog,
    parent_arrays: dict[str, np.ndarray],
    parent_metadata: dict[str, object],
    required_horizon: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    parent_horizon = int(parent_metadata.get("required_horizon", -1))
    role_users = parent_arrays["role_record_user_ids"]
    role_codes = parent_arrays["role_record_role"]
    parent_offsets = parent_arrays["role_record_offsets"]
    if (
        parent_metadata.get("protocol") != PROTOCOL
        or parent_metadata.get("catalog_content_sha256")
        != catalog.metadata.get("content_sha256")
        or required_horizon <= parent_horizon
        or parent_horizon < 2
        or len(role_users) != len(role_codes)
        or parent_offsets.shape != (len(role_users) + 1,)
        or np.any(np.diff(parent_offsets) != parent_horizon)
    ):
        raise ValueError("QB parent corpus cannot be extended")
    order = {int(user): index for index, user in enumerate(role_users.tolist())}
    selected = frame[
        frame["user_id"].isin(role_users) & (frame["raw_ordinal"] < required_horizon)
    ].copy()
    selected["record_order"] = selected["user_id"].map(order)
    selected.sort_values(["record_order", "raw_ordinal"], inplace=True, kind="stable")
    counts = (
        selected.groupby("record_order", sort=True)
        .size()
        .reindex(np.arange(len(role_users)), fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    if np.any(counts < parent_horizon) or np.any(counts > required_horizon):
        raise ValueError("QB extended role lengths differ")
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    role_arrays = _event_arrays(selected, catalog)
    prefix_indices = np.concatenate(
        [
            np.arange(offsets[index], offsets[index] + parent_horizon, dtype=np.int64)
            for index in range(len(role_users))
        ]
    )
    for name in (
        "feature_ids",
        "target_item_ids",
        "behavior",
        "raw_label",
        "label",
        "raw_ordinal",
        "is_prediction_item",
    ):
        if not np.array_equal(
            parent_arrays[f"role_{name}"],
            role_arrays[name][prefix_indices],
        ):
            raise ValueError(f"QB extended role prefix differs: {name}")
    arrays = {
        name: value
        for name, value in parent_arrays.items()
        if not name.startswith("role_")
    }
    arrays.update(
        {
            "role_record_user_ids": role_users,
            "role_record_role": role_codes,
            "role_record_offsets": offsets,
            **{f"role_{name}": value for name, value in role_arrays.items()},
        }
    )
    full_horizon = counts == required_horizon
    metadata = {
        **parent_metadata,
        "formal_result": False,
        "scientific_result": False,
        "required_horizon": required_horizon,
        "role_events": len(selected),
        "parent_required_horizon": parent_horizon,
        "parent_corpus_content_sha256": parent_metadata["content_sha256"],
        "extension_semantics": "same role users and byte-identical per-user parent prefix",
        "role_length": {
            "minimum": int(counts.min()),
            "maximum": int(counts.max()),
            "mean": float(counts.mean()),
            "users_reaching_required_horizon": int(full_horizon.sum()),
            "users_reaching_required_horizon_by_role": {
                name: int(np.sum(full_horizon & (role_codes == code)))
                for code, name in enumerate(ROLE_NAMES)
            },
        },
        "content_sha256": artifact_sha256(arrays),
    }
    return arrays, metadata


def save_corpus(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> dict[str, object]:
    if metadata.get("content_sha256") != artifact_sha256(arrays):
        raise ValueError("QB corpus content binding differs")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "content_sha256": metadata["content_sha256"],
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
