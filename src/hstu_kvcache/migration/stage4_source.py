from __future__ import annotations

import hashlib
import io
import json
import math
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

SOURCE_MANIFEST_PROTOCOL = "cohortkv_stage4_source_manifest_v1"
SOURCE_SHARD_PROTOCOL = "cohortkv_stage4_source_shard_v1"
SOURCE_REPRESENTATIONS = (
    "normalized_capsule_fp16",
    "old_kv_fp16",
    "raw_history",
    "residual_hidden_suffix_bf16",
)
_SHA256_LENGTH = 64


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        + b"\n"
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_nbytes(values: Mapping[str, torch.Tensor]) -> int:
    return sum(value.numel() * value.element_size() for value in values.values())


def _validate_relative_path(value: str) -> None:
    path = Path(value)
    if path.is_absolute() or not value or ".." in path.parts:
        raise ValueError("source shard path must be a safe relative path")


@dataclass(frozen=True)
class SourceShardDescriptor:
    representation: str
    path: str
    sha256: str
    physical_bytes: int
    logical_bytes: int

    def __post_init__(self) -> None:
        if self.representation not in SOURCE_REPRESENTATIONS:
            raise ValueError("unsupported source representation")
        _validate_relative_path(self.path)
        if len(self.sha256) != _SHA256_LENGTH:
            raise ValueError("source shard hash is invalid")
        if self.physical_bytes < 1 or self.logical_bytes < 1:
            raise ValueError("source shard byte counts must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "representation": self.representation,
            "path": self.path,
            "sha256": self.sha256,
            "physical_bytes": self.physical_bytes,
            "logical_bytes": self.logical_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SourceShardDescriptor:
        return cls(
            representation=str(value["representation"]),
            path=str(value["path"]),
            sha256=str(value["sha256"]),
            physical_bytes=int(value["physical_bytes"]),
            logical_bytes=int(value["logical_bytes"]),
        )


@dataclass(frozen=True)
class SourceRecordDescriptor:
    record_id: int
    user_id: int
    evaluation_role: str
    source_version: str
    target_version: str
    prefix_tokens: int
    shards: tuple[SourceShardDescriptor, ...]

    def __post_init__(self) -> None:
        if self.record_id < 0 or self.user_id < 0 or self.prefix_tokens < 1:
            raise ValueError("source record identity is invalid")
        if not self.evaluation_role or not self.source_version or not self.target_version:
            raise ValueError("source record metadata is incomplete")
        representations = [value.representation for value in self.shards]
        if len(set(representations)) != len(representations):
            raise ValueError("source record representations must be unique")
        required = {
            "normalized_capsule_fp16",
            "old_kv_fp16",
            "raw_history",
        }
        if not required.issubset(representations):
            raise ValueError("source record is missing a primary representation")

    @property
    def shard_map(self) -> dict[str, SourceShardDescriptor]:
        return {value.representation: value for value in self.shards}

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "user_id": self.user_id,
            "evaluation_role": self.evaluation_role,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "prefix_tokens": self.prefix_tokens,
            "shards": [value.to_dict() for value in self.shards],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SourceRecordDescriptor:
        return cls(
            record_id=int(value["record_id"]),
            user_id=int(value["user_id"]),
            evaluation_role=str(value["evaluation_role"]),
            source_version=str(value["source_version"]),
            target_version=str(value["target_version"]),
            prefix_tokens=int(value["prefix_tokens"]),
            shards=tuple(
                SourceShardDescriptor.from_dict(item)
                for item in value["shards"]
            ),
        )


@dataclass(frozen=True)
class Stage4SourceManifest:
    workload_content_sha256: str
    workload_file_sha256: str
    num_layers: int
    hidden_size: int
    kv_width: int
    records: tuple[SourceRecordDescriptor, ...]
    creation: Mapping[str, object]
    protocol: str = SOURCE_MANIFEST_PROTOCOL

    def __post_init__(self) -> None:
        if self.protocol != SOURCE_MANIFEST_PROTOCOL:
            raise ValueError("unsupported Stage 4 source manifest protocol")
        if (
            len(self.workload_content_sha256) != _SHA256_LENGTH
            or len(self.workload_file_sha256) != _SHA256_LENGTH
        ):
            raise ValueError("source manifest workload hashes are invalid")
        if min(self.num_layers, self.hidden_size, self.kv_width) < 1:
            raise ValueError("source manifest model dimensions are invalid")
        if not self.records:
            raise ValueError("source manifest must contain records")
        record_ids = [value.record_id for value in self.records]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("source manifest record IDs must be unique")
        if tuple(sorted(record_ids)) != tuple(record_ids):
            raise ValueError("source manifest records must be ordered by record ID")
        targets = {value.target_version for value in self.records}
        if len(targets) != 1:
            raise ValueError("source manifest target version must be unique")

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def prefix_tokens(self) -> int:
        return sum(value.prefix_tokens for value in self.records)

    @property
    def target_version(self) -> str:
        return self.records[0].target_version

    @property
    def record_map(self) -> dict[int, SourceRecordDescriptor]:
        return {value.record_id: value for value in self.records}

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "workload_content_sha256": self.workload_content_sha256,
            "workload_file_sha256": self.workload_file_sha256,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "kv_width": self.kv_width,
            "record_count": self.record_count,
            "prefix_tokens": self.prefix_tokens,
            "records": [value.to_dict() for value in self.records],
            "creation": dict(self.creation),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Stage4SourceManifest:
        manifest = cls(
            protocol=str(value["protocol"]),
            workload_content_sha256=str(value["workload_content_sha256"]),
            workload_file_sha256=str(value["workload_file_sha256"]),
            num_layers=int(value["num_layers"]),
            hidden_size=int(value["hidden_size"]),
            kv_width=int(value["kv_width"]),
            records=tuple(
                SourceRecordDescriptor.from_dict(item)
                for item in value["records"]
            ),
            creation=dict(value.get("creation", {})),
        )
        if int(value["record_count"]) != manifest.record_count:
            raise ValueError("source manifest record count is invalid")
        if int(value["prefix_tokens"]) != manifest.prefix_tokens:
            raise ValueError("source manifest token count is invalid")
        return manifest

    @classmethod
    def load(cls, path: Path | str) -> Stage4SourceManifest:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def write(self, path: Path | str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = _canonical_json(self.to_dict())
        temporary = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.tmp"
        )
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)


def _validate_source_tensors(
    representation: str,
    tensors: Mapping[str, torch.Tensor],
    prefix_tokens: int,
    metadata: Mapping[str, object],
) -> None:
    if any(value.device.type != "cpu" for value in tensors.values()):
        raise ValueError("source shard tensors must be CPU resident")
    if any(not value.is_contiguous() for value in tensors.values()):
        raise ValueError("source shard tensors must be contiguous")
    if representation == "normalized_capsule_fp16":
        value = tensors.get("normed")
        if (
            set(tensors) != {"normed"}
            or value is None
            or value.ndim != 3
            or value.shape[1] != prefix_tokens
            or value.dtype != torch.float16
        ):
            raise ValueError("normalized capsule shard is invalid")
    elif representation == "old_kv_fp16":
        k = tensors.get("k")
        v = tensors.get("v")
        if (
            set(tensors) != {"k", "v"}
            or k is None
            or v is None
            or k.ndim != 3
            or k.shape != v.shape
            or k.shape[1] != prefix_tokens
            or k.dtype != torch.float16
            or v.dtype != torch.float16
        ):
            raise ValueError("old K/V shard is invalid")
    elif representation == "raw_history":
        expected = {"item_ids", "behaviors", "time_deltas"}
        if set(tensors) != expected:
            raise ValueError("raw history fields are invalid")
        if any(value.ndim != 1 or value.shape[0] != prefix_tokens for value in tensors.values()):
            raise ValueError("raw history shapes are invalid")
        if tensors["item_ids"].dtype != torch.long:
            raise ValueError("raw item IDs must be int64")
        if tensors["behaviors"].dtype != torch.long:
            raise ValueError("raw behaviors must be int64")
        if tensors["time_deltas"].dtype != torch.float32:
            raise ValueError("raw time deltas must be float32")
    elif representation == "residual_hidden_suffix_bf16":
        value = tensors.get("hidden_states")
        start_layer = int(metadata.get("start_layer", -1))
        num_layers = int(metadata.get("num_layers", -1))
        if (
            set(tensors) != {"hidden_states"}
            or value is None
            or value.ndim != 3
            or value.shape[0] != num_layers - start_layer
            or value.shape[1] != prefix_tokens
            or value.dtype != torch.bfloat16
            or not 1 <= start_layer < num_layers
        ):
            raise ValueError("residual hidden suffix shard is invalid")
    else:
        raise ValueError("unsupported source representation")


def write_source_shard(
    root: Path | str,
    relative_path: str,
    representation: str,
    record_id: int,
    source_version: str,
    target_version: str,
    prefix_tokens: int,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, object] | None = None,
) -> SourceShardDescriptor:
    _validate_relative_path(relative_path)
    metadata = dict(metadata or {})
    values = {
        name: value.detach().cpu().contiguous()
        for name, value in tensors.items()
    }
    _validate_source_tensors(
        representation,
        values,
        prefix_tokens,
        metadata,
    )
    payload = {
        "protocol": SOURCE_SHARD_PROTOCOL,
        "representation": representation,
        "record_id": record_id,
        "source_version": source_version,
        "target_version": target_version,
        "prefix_tokens": prefix_tokens,
        "metadata": metadata,
        "tensors": values,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    encoded = buffer.getvalue()
    destination = Path(root) / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return SourceShardDescriptor(
        representation=representation,
        path=relative_path,
        sha256=_sha256(encoded),
        physical_bytes=len(encoded),
        logical_bytes=_tensor_nbytes(values),
    )


@dataclass(frozen=True)
class Stage4SourceBatch:
    record_ids: tuple[int, ...]
    migration_anchor_version: str
    served_kv_target: str
    lengths: torch.Tensor
    sequence_width: int
    normed: torch.Tensor | None = None
    old_k: torch.Tensor | None = None
    old_v: torch.Tensor | None = None
    item_ids: torch.Tensor | None = None
    behaviors: torch.Tensor | None = None
    time_deltas: torch.Tensor | None = None
    residual_hidden_states: torch.Tensor | None = None
    residual_start_layer: int | None = None

    def __post_init__(self) -> None:
        if not self.record_ids or len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError("source batch record IDs must be nonempty and unique")
        if not self.migration_anchor_version or not self.served_kv_target:
            raise ValueError("source batch versions must be nonempty")
        if self.lengths.shape != (len(self.record_ids),):
            raise ValueError("source batch lengths are invalid")
        if self.sequence_width < 1:
            raise ValueError("source batch sequence width must be positive")
        tensors = self.tensors
        if not tensors:
            raise ValueError("source batch contains no representation")
        devices = {value.device for value in tensors.values()}
        if len(devices) != 1:
            raise ValueError("source batch tensors must share one device")
        batch = len(self.record_ids)
        if self.normed is not None and self.normed.shape[1:3] != (
            batch,
            self.sequence_width,
        ):
            raise ValueError("source batch normalized capsule shape is invalid")
        if self.old_k is not None:
            if (
                self.old_v is None
                or self.old_k.shape != self.old_v.shape
                or self.old_k.shape[1:3] != (batch, self.sequence_width)
            ):
                raise ValueError("source batch old K/V shape is invalid")
        raw = (self.item_ids, self.behaviors, self.time_deltas)
        if any(value is not None for value in raw) and (
            any(value is None for value in raw)
            or any(value.shape != (batch, self.sequence_width) for value in raw)
        ):
            raise ValueError("source batch raw history shape is invalid")
        if self.residual_hidden_states is not None:
            if (
                self.residual_hidden_states.shape[1:3]
                != (batch, self.sequence_width)
                or self.residual_start_layer is None
            ):
                raise ValueError("source batch residual suffix shape is invalid")
        if self.device.type == "cpu" and (
            bool(torch.any(self.lengths < 1))
            or bool(torch.any(self.lengths > self.sequence_width))
        ):
            raise ValueError("source batch lengths exceed its sequence width")

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        values = {"lengths": self.lengths}
        for name in (
            "normed",
            "old_k",
            "old_v",
            "item_ids",
            "behaviors",
            "time_deltas",
            "residual_hidden_states",
        ):
            value = getattr(self, name)
            if value is not None:
                values[name] = value
        return values

    @property
    def device(self) -> torch.device:
        return self.lengths.device

    @property
    def nbytes(self) -> int:
        return _tensor_nbytes(self.tensors)

    @property
    def token_count(self) -> int:
        return int(self.lengths.sum().item())

    @property
    def is_pinned(self) -> bool:
        return self.device.type == "cpu" and all(
            value.is_pinned() for value in self.tensors.values()
        )

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ) -> Stage4SourceBatch:
        values = {
            name: None
            if getattr(self, name) is None
            else getattr(self, name).to(device, non_blocking=non_blocking)
            for name in (
                "normed",
                "old_k",
                "old_v",
                "item_ids",
                "behaviors",
                "time_deltas",
                "residual_hidden_states",
            )
        }
        return Stage4SourceBatch(
            record_ids=self.record_ids,
            migration_anchor_version=self.migration_anchor_version,
            served_kv_target=self.served_kv_target,
            lengths=self.lengths.to(device, non_blocking=non_blocking),
            sequence_width=self.sequence_width,
            residual_start_layer=self.residual_start_layer,
            **values,
        )


@dataclass(frozen=True)
class Stage4ExtentSpec:
    extent_id: str
    records: tuple[SourceRecordDescriptor, ...]
    representations: tuple[str, ...]
    sequence_width: int
    logical_input_bytes: int
    physical_input_bytes: int
    logical_output_bytes: int
    padding_tokens: int

    def __post_init__(self) -> None:
        if not self.extent_id or not self.records:
            raise ValueError("extent specification is incomplete")
        sources = {value.source_version for value in self.records}
        targets = {value.target_version for value in self.records}
        if len(sources) != 1 or len(targets) != 1:
            raise ValueError("extent records must share source and target versions")
        if (
            self.sequence_width < max(value.prefix_tokens for value in self.records)
            or min(
                self.logical_input_bytes,
                self.physical_input_bytes,
                self.logical_output_bytes,
            )
            < 1
            or self.padding_tokens < 0
        ):
            raise ValueError("extent byte or padding metadata is invalid")

    @property
    def record_ids(self) -> tuple[int, ...]:
        return tuple(value.record_id for value in self.records)

    @property
    def migration_anchor_version(self) -> str:
        return self.records[0].source_version

    @property
    def served_kv_target(self) -> str:
        return self.records[0].target_version

    @property
    def token_count(self) -> int:
        return sum(value.prefix_tokens for value in self.records)

    @property
    def placement_weight_bytes(self) -> int:
        return self.logical_input_bytes + self.logical_output_bytes


@dataclass(frozen=True)
class SourceReadMetrics:
    physical_bytes: int
    logical_bytes: int
    peak_source_resident_bytes: int


def build_stage4_extents(
    manifest: Stage4SourceManifest,
    record_ids: tuple[int, ...],
    representations_by_source: Mapping[str, tuple[str, ...]],
    batch_size: int,
    bucket_width: int,
) -> tuple[Stage4ExtentSpec, ...]:
    if batch_size < 1 or bucket_width < 1:
        raise ValueError("batch size and bucket width must be positive")
    if not record_ids or len(set(record_ids)) != len(record_ids):
        raise ValueError("extent planner record IDs must be nonempty and unique")
    record_map = manifest.record_map
    try:
        selected = [record_map[value] for value in record_ids]
    except KeyError as exc:
        raise ValueError("extent planner record is absent from source manifest") from exc
    grouped: dict[tuple[str, int, tuple[str, ...]], list[SourceRecordDescriptor]] = {}
    for record in selected:
        try:
            representations = tuple(representations_by_source[record.source_version])
        except KeyError as exc:
            raise ValueError("source version has no representation contract") from exc
        if not representations or len(set(representations)) != len(representations):
            raise ValueError("source representations must be nonempty and unique")
        missing = set(representations) - set(record.shard_map)
        if missing:
            raise ValueError(
                f"record {record.record_id} is missing representations {sorted(missing)}"
            )
        key = (
            record.source_version,
            math.ceil(record.prefix_tokens / bucket_width),
            representations,
        )
        grouped.setdefault(key, []).append(record)
    extents = []
    extent_index = 0
    for key in sorted(grouped):
        cohort = sorted(
            grouped[key],
            key=lambda value: (value.prefix_tokens, value.record_id),
        )
        representations = key[2]
        for start in range(0, len(cohort), batch_size):
            records = tuple(cohort[start : start + batch_size])
            width = math.ceil(
                max(value.prefix_tokens for value in records) / bucket_width
            ) * bucket_width
            width = min(2047, width)
            logical_input = sum(
                record.shard_map[representation].logical_bytes
                for record in records
                for representation in representations
            )
            physical_input = sum(
                record.shard_map[representation].physical_bytes
                for record in records
                for representation in representations
            )
            tokens = sum(value.prefix_tokens for value in records)
            output_bytes = (
                2
                * manifest.num_layers
                * tokens
                * manifest.kv_width
                * torch.tensor([], dtype=torch.float16).element_size()
            )
            extents.append(
                Stage4ExtentSpec(
                    extent_id=f"extent-{extent_index:06d}",
                    records=records,
                    representations=representations,
                    sequence_width=width,
                    logical_input_bytes=logical_input,
                    physical_input_bytes=physical_input,
                    logical_output_bytes=output_bytes,
                    padding_tokens=len(records) * width - tokens,
                )
            )
            extent_index += 1
    covered = tuple(
        record_id
        for extent in extents
        for record_id in extent.record_ids
    )
    if len(covered) != len(record_ids) or set(covered) != set(record_ids):
        raise ValueError("extent plan does not cover every selected record once")
    return tuple(extents)


def place_stage4_extents_lpt(
    extents: tuple[Stage4ExtentSpec, ...],
    gpu_count: int,
) -> tuple[tuple[Stage4ExtentSpec, ...], ...]:
    if not extents or gpu_count < 1:
        raise ValueError("LPT placement requires extents and devices")
    assignments: list[list[Stage4ExtentSpec]] = [[] for _ in range(gpu_count)]
    assigned_bytes = [0] * gpu_count
    for extent in sorted(
        extents,
        key=lambda value: (-value.placement_weight_bytes, value.extent_id),
    ):
        index = min(range(gpu_count), key=lambda value: (assigned_bytes[value], value))
        assignments[index].append(extent)
        assigned_bytes[index] += extent.placement_weight_bytes
    return tuple(
        tuple(sorted(values, key=lambda value: value.extent_id))
        for values in assignments
    )


class LazyStage4SourceReader:
    def __init__(
        self,
        manifest_path: Path | str,
        expected_workload_content_sha256: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        encoded = self.manifest_path.read_bytes()
        self.manifest_file_sha256 = _sha256(encoded)
        self.manifest_physical_bytes = len(encoded)
        self.manifest = Stage4SourceManifest.from_dict(json.loads(encoded))
        if (
            expected_workload_content_sha256 is not None
            and self.manifest.workload_content_sha256
            != expected_workload_content_sha256
        ):
            raise ValueError("source and workload manifest identities differ")

    def _read_shard(
        self,
        record: SourceRecordDescriptor,
        representation: str,
    ) -> tuple[dict[str, torch.Tensor], dict[str, object], int]:
        descriptor = record.shard_map[representation]
        path = (self.root / descriptor.path).resolve()
        if self.root not in path.parents:
            raise ValueError("source shard resolves outside its root")
        encoded = path.read_bytes()
        if (
            len(encoded) != descriptor.physical_bytes
            or _sha256(encoded) != descriptor.sha256
        ):
            raise ValueError("source shard integrity check failed")
        payload = torch.load(
            io.BytesIO(encoded),
            map_location="cpu",
            weights_only=True,
        )
        if (
            payload.get("protocol") != SOURCE_SHARD_PROTOCOL
            or payload.get("representation") != representation
            or int(payload.get("record_id", -1)) != record.record_id
            or payload.get("source_version") != record.source_version
            or payload.get("target_version") != record.target_version
            or int(payload.get("prefix_tokens", -1)) != record.prefix_tokens
        ):
            raise ValueError("source shard identity check failed")
        tensors = dict(payload["tensors"])
        metadata = dict(payload.get("metadata", {}))
        _validate_source_tensors(
            representation,
            tensors,
            record.prefix_tokens,
            metadata,
        )
        if _tensor_nbytes(tensors) != descriptor.logical_bytes:
            raise ValueError("source shard logical byte count is invalid")
        return tensors, metadata, len(encoded)

    def read_extent(
        self,
        extent: Stage4ExtentSpec,
        pin_memory: bool = True,
    ) -> tuple[Stage4SourceBatch, SourceReadMetrics]:
        batch_size = len(extent.records)
        width = extent.sequence_width
        options = {"device": "cpu", "pin_memory": pin_memory}
        lengths = torch.tensor(
            [value.prefix_tokens for value in extent.records],
            dtype=torch.long,
            **options,
        )
        values: dict[str, torch.Tensor | None] = {
            "normed": None,
            "old_k": None,
            "old_v": None,
            "item_ids": None,
            "behaviors": None,
            "time_deltas": None,
            "residual_hidden_states": None,
        }
        manifest = self.manifest
        if "normalized_capsule_fp16" in extent.representations:
            values["normed"] = torch.zeros(
                manifest.num_layers,
                batch_size,
                width,
                manifest.hidden_size,
                dtype=torch.float16,
                **options,
            )
        if "old_kv_fp16" in extent.representations:
            shape = (
                manifest.num_layers,
                batch_size,
                width,
                manifest.kv_width,
            )
            values["old_k"] = torch.zeros(shape, dtype=torch.float16, **options)
            values["old_v"] = torch.zeros(shape, dtype=torch.float16, **options)
        if "raw_history" in extent.representations:
            shape = (batch_size, width)
            values["item_ids"] = torch.zeros(shape, dtype=torch.long, **options)
            values["behaviors"] = torch.zeros(shape, dtype=torch.long, **options)
            values["time_deltas"] = torch.zeros(
                shape,
                dtype=torch.float32,
                **options,
            )
        residual_start = None
        physical_bytes = 0
        logical_bytes = 0
        max_transient = 0
        for row, record in enumerate(extent.records):
            length = record.prefix_tokens
            for representation in extent.representations:
                tensors, metadata, encoded_bytes = self._read_shard(
                    record,
                    representation,
                )
                decoded_bytes = _tensor_nbytes(tensors)
                physical_bytes += encoded_bytes
                logical_bytes += decoded_bytes
                max_transient = max(max_transient, encoded_bytes + decoded_bytes)
                if representation == "normalized_capsule_fp16":
                    values["normed"][:, row, :length].copy_(tensors["normed"])
                elif representation == "old_kv_fp16":
                    values["old_k"][:, row, :length].copy_(tensors["k"])
                    values["old_v"][:, row, :length].copy_(tensors["v"])
                elif representation == "raw_history":
                    values["item_ids"][row, :length].copy_(tensors["item_ids"])
                    values["behaviors"][row, :length].copy_(tensors["behaviors"])
                    values["time_deltas"][row, :length].copy_(
                        tensors["time_deltas"]
                    )
                elif representation == "residual_hidden_suffix_bf16":
                    start_layer = int(metadata["start_layer"])
                    if residual_start is None:
                        residual_start = start_layer
                        values["residual_hidden_states"] = torch.zeros(
                            manifest.num_layers - start_layer,
                            batch_size,
                            width,
                            manifest.hidden_size,
                            dtype=torch.bfloat16,
                            **options,
                        )
                    if start_layer != residual_start:
                        raise ValueError("extent residual suffix starts differ")
                    values["residual_hidden_states"][:, row, :length].copy_(
                        tensors["hidden_states"]
                    )
        if (
            physical_bytes != extent.physical_input_bytes
            or logical_bytes != extent.logical_input_bytes
        ):
            raise ValueError("source extent byte totals differ from its plan")
        batch = Stage4SourceBatch(
            record_ids=extent.record_ids,
            migration_anchor_version=extent.migration_anchor_version,
            served_kv_target=extent.served_kv_target,
            lengths=lengths,
            sequence_width=width,
            residual_start_layer=residual_start,
            **values,
        )
        return batch, SourceReadMetrics(
            physical_bytes=physical_bytes,
            logical_bytes=logical_bytes,
            peak_source_resident_bytes=batch.nbytes + max_transient,
        )
