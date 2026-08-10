from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROTOCOL = "evokv_large_variable_inference_v0"
ROLE_CODES = {"fit": 0, "probe": 1, "qualification": 2}


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def artifact_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode())
        digest.update(array_sha256(arrays[name]).encode())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_order(values: np.ndarray, salt: str) -> np.ndarray:
    return np.asarray(
        sorted(
            (int(value) for value in values),
            key=lambda value: (
                hashlib.sha256(f"{salt}:{value}".encode()).digest(),
                value,
            ),
        ),
        dtype=np.int64,
    )


def prefix_schedule(
    valid_lengths: np.ndarray,
    edge_count: int,
    minimum_initial_tokens: int = 64,
) -> np.ndarray:
    lengths = np.asarray(valid_lengths, dtype=np.int64)
    if (
        lengths.ndim != 1
        or len(lengths) < 1
        or edge_count < 1
        or minimum_initial_tokens < 2
        or np.any(lengths <= minimum_initial_tokens + edge_count)
    ):
        raise ValueError("variable inference lengths differ")
    final = lengths - 1
    initial = np.maximum(minimum_initial_tokens, final // 2)
    initial = np.minimum(initial, final - edge_count)
    schedule = np.empty((len(lengths), edge_count + 1), dtype=np.int16)
    schedule[:, 0] = initial.astype(np.int16)
    remaining = final - initial
    for edge in range(1, edge_count):
        value = initial + np.floor_divide(remaining * edge, edge_count)
        schedule[:, edge] = value.astype(np.int16)
    schedule[:, edge_count] = final.astype(np.int16)
    if (
        np.any(schedule < 2)
        or np.any(schedule[:, 1:] <= schedule[:, :-1])
        or np.any(schedule[:, -1] >= lengths)
    ):
        raise ValueError("variable inference schedule differs")
    return schedule


@dataclass(frozen=True)
class VariableInferenceCorpus:
    path: Path
    arrays: dict[str, np.ndarray]
    metadata: dict[str, object]
    file_sha256: str
    content_sha256: str

    @property
    def dataset(self) -> str:
        return str(self.metadata["dataset"])

    @property
    def edge_count(self) -> int:
        return int(self.metadata["edge_count"])

    @property
    def feature_fields(self) -> int:
        return int(self.metadata["feature_fields"])

    def role_records(self, role: str) -> np.ndarray:
        if role not in ROLE_CODES:
            raise ValueError("variable inference role differs")
        return np.flatnonzero(self.arrays["record_role"] == ROLE_CODES[role])

    def record_slice(self, record: int) -> slice:
        offsets = self.arrays["record_offsets"]
        return slice(int(offsets[record]), int(offsets[record + 1]))


def validate_arrays(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    required = {
        "record_source_ids",
        "record_user_ids",
        "record_role",
        "record_offsets",
        "record_valid_lengths",
        "edge_prefix_lengths",
        "feature_ids",
        "target_item_ids",
        "behaviors",
        "time_deltas",
        "labels",
        "is_prediction_item",
    }
    if required.difference(arrays):
        raise ValueError("variable inference arrays are incomplete")
    records = len(arrays["record_user_ids"])
    offsets = arrays["record_offsets"]
    events = int(offsets[-1]) if len(offsets) else -1
    edge_count = int(metadata.get("edge_count", -1))
    fields = int(metadata.get("feature_fields", -1))
    if (
        metadata.get("protocol") != PROTOCOL
        or metadata.get("dataset") not in {"qk", "qb"}
        or records < 1
        or edge_count < 1
        or fields < 1
        or arrays["record_source_ids"].dtype != np.int64
        or arrays["record_source_ids"].shape != (records,)
        or len(np.unique(arrays["record_source_ids"])) != records
        or arrays["record_user_ids"].dtype != np.int64
        or arrays["record_user_ids"].shape != (records,)
        or len(np.unique(arrays["record_user_ids"])) != records
        or arrays["record_role"].shape != (records,)
        or arrays["record_role"].dtype != np.uint8
        or np.any(arrays["record_role"] >= len(ROLE_CODES))
        or offsets.dtype != np.int64
        or offsets.shape != (records + 1,)
        or offsets[0] != 0
        or np.any(offsets[1:] <= offsets[:-1])
        or arrays["record_valid_lengths"].dtype != np.int64
        or arrays["record_valid_lengths"].shape != (records,)
        or not np.array_equal(arrays["record_valid_lengths"], np.diff(offsets))
        or arrays["edge_prefix_lengths"].dtype != np.int16
        or arrays["edge_prefix_lengths"].shape != (records, edge_count + 1)
        or arrays["feature_ids"].shape != (events, fields)
        or arrays["target_item_ids"].shape != (events,)
        or arrays["behaviors"].shape != (events,)
        or arrays["time_deltas"].shape != (events,)
        or arrays["labels"].shape != (events,)
        or arrays["is_prediction_item"].shape != (events,)
        or arrays["feature_ids"].dtype != np.uint32
        or arrays["target_item_ids"].dtype != np.uint32
        or arrays["behaviors"].dtype != np.uint8
        or arrays["time_deltas"].dtype != np.float32
        or arrays["labels"].dtype != np.uint8
        or arrays["is_prediction_item"].dtype != np.uint8
        or np.any(arrays["feature_ids"] < 1)
        or np.any(arrays["target_item_ids"] < 1)
        or np.any(arrays["behaviors"] < 1)
        or np.any(arrays["behaviors"] > 5)
        or np.any(~np.isfinite(arrays["time_deltas"]))
        or np.any(arrays["time_deltas"] < 0)
        or np.any(arrays["is_prediction_item"] > 1)
        or np.any(arrays["labels"] > arrays["is_prediction_item"])
        or np.any(arrays["edge_prefix_lengths"][:, 1:] <= arrays["edge_prefix_lengths"][:, :-1])
        or np.any(arrays["edge_prefix_lengths"][:, -1] >= arrays["record_valid_lengths"])
    ):
        raise ValueError("variable inference corpus differs")
    expected_schedule = prefix_schedule(
        arrays["record_valid_lengths"],
        edge_count,
        int(metadata["minimum_initial_tokens"]),
    )
    if not np.array_equal(arrays["edge_prefix_lengths"], expected_schedule):
        raise ValueError("variable inference prefix schedule binding differs")


def write_corpus(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> dict[str, object]:
    output = Path(path)
    if output.exists():
        raise FileExistsError(output)
    value = {**metadata, "protocol": PROTOCOL}
    validate_arrays(arrays, value)
    content = artifact_sha256(arrays)
    value["content_sha256"] = content
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.asarray(json.dumps(value, sort_keys=True)),
    )
    temporary.replace(output)
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": file_sha256(output),
        "content_sha256": content,
    }


def load_corpus(path: str | Path) -> VariableInferenceCorpus:
    resolved = Path(path)
    with np.load(resolved, allow_pickle=False) as source:
        if "metadata_json" not in source.files:
            raise ValueError("variable inference metadata is absent")
        arrays = {
            name: source[name].copy()
            for name in source.files
            if name != "metadata_json"
        }
        metadata = json.loads(str(source["metadata_json"].item()))
    validate_arrays(arrays, metadata)
    content = artifact_sha256(arrays)
    if metadata.get("content_sha256") != content:
        raise ValueError("variable inference content hash differs")
    return VariableInferenceCorpus(
        path=resolved,
        arrays=arrays,
        metadata=metadata,
        file_sha256=file_sha256(resolved),
        content_sha256=content,
    )
