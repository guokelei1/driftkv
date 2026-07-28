from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import torch

from .cohort_jagged import JaggedMigratedKVBatch

DESTINATION_MANIFEST_PROTOCOL = "streamkv_destination_manifest_v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _validate_identifier(value: str, field: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} contains unsupported characters")


def _version_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _canonical_metadata_json(value: dict[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _serialize_batch(batch: JaggedMigratedKVBatch) -> bytes:
    payload = {
        "k": batch.k.detach().contiguous(),
        "v": batch.v.detach().contiguous(),
        "lengths": batch.lengths.detach().contiguous(),
        "offsets": batch.offsets.detach().contiguous(),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _deserialize_batch(
    payload: bytes,
    extent: PublishedKVExtent,
) -> JaggedMigratedKVBatch:
    values = torch.load(
        io.BytesIO(payload),
        map_location="cpu",
        weights_only=True,
    )
    return JaggedMigratedKVBatch(
        record_ids=extent.record_ids,
        migration_anchor_version=extent.migration_anchor_version,
        served_kv_target=extent.served_kv_target,
        k=values["k"],
        v=values["v"],
        lengths=values["lengths"],
        offsets=values["offsets"],
    )


def _clone_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.device.type == "cpu" and value.is_pinned():
        output = torch.empty_like(value, pin_memory=True)
        output.copy_(value)
        return output
    return value.clone()


def _clone_batch(
    batch: JaggedMigratedKVBatch,
) -> JaggedMigratedKVBatch:
    return JaggedMigratedKVBatch(
        record_ids=batch.record_ids,
        migration_anchor_version=batch.migration_anchor_version,
        served_kv_target=batch.served_kv_target,
        k=_clone_tensor(batch.k),
        v=_clone_tensor(batch.v),
        lengths=_clone_tensor(batch.lengths),
        offsets=_clone_tensor(batch.offsets),
    )


class DestinationKind(str, Enum):
    HBM = "hbm"
    DRAM = "dram"
    LOCAL_SSD = "local_ssd"
    REMOTE = "remote"


class PublicationMode(str, Enum):
    DIRECT_DEVICE = "direct_device"
    HOST_STAGED = "host_staged"


@dataclass(frozen=True)
class DestinationCapabilities:
    kind: DestinationKind
    publication_mode: PublicationMode
    durable: bool
    atomic_manifest: bool
    supports_readback: bool


@dataclass(frozen=True)
class PublishedKVExtent:
    extent_id: str
    record_ids: tuple[int, ...]
    migration_anchor_version: str
    served_kv_target: str
    num_layers: int
    token_count: int
    kv_width: int
    dtype: str
    payload_bytes: int
    location: str
    device: str | None = None
    checksum_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.extent_id, "extent_id")
        if not self.record_ids or len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError("extent record IDs must be nonempty and unique")
        if not self.migration_anchor_version or not self.served_kv_target:
            raise ValueError("extent versions must be nonempty")
        if self.num_layers < 1 or self.token_count < 1 or self.kv_width < 1:
            raise ValueError("extent tensor dimensions must be positive")
        if not self.dtype or self.payload_bytes < 1 or not self.location:
            raise ValueError("extent storage metadata is incomplete")

    def to_dict(self) -> dict[str, object]:
        return {
            "extent_id": self.extent_id,
            "record_ids": list(self.record_ids),
            "migration_anchor_version": self.migration_anchor_version,
            "served_kv_target": self.served_kv_target,
            "num_layers": self.num_layers,
            "token_count": self.token_count,
            "kv_width": self.kv_width,
            "dtype": self.dtype,
            "payload_bytes": self.payload_bytes,
            "location": self.location,
            "device": self.device,
            "checksum_sha256": self.checksum_sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> PublishedKVExtent:
        return cls(
            extent_id=str(value["extent_id"]),
            record_ids=tuple(int(item) for item in value["record_ids"]),
            migration_anchor_version=str(value["migration_anchor_version"]),
            served_kv_target=str(value["served_kv_target"]),
            num_layers=int(value["num_layers"]),
            token_count=int(value["token_count"]),
            kv_width=int(value["kv_width"]),
            dtype=str(value["dtype"]),
            payload_bytes=int(value["payload_bytes"]),
            location=str(value["location"]),
            device=None if value["device"] is None else str(value["device"]),
            checksum_sha256=(
                None
                if value["checksum_sha256"] is None
                else str(value["checksum_sha256"])
            ),
        )


@dataclass(frozen=True)
class KVVersionManifest:
    job_id: str
    target_version: str
    destination_id: str
    destination_kind: DestinationKind
    publication_mode: PublicationMode
    extents: tuple[PublishedKVExtent, ...]
    record_count: int
    token_count: int
    payload_bytes: int
    metadata_sha256: str | None = None
    metadata_json: str | None = None
    protocol: str = DESTINATION_MANIFEST_PROTOCOL

    def __post_init__(self) -> None:
        _validate_identifier(self.job_id, "job_id")
        _validate_identifier(self.destination_id, "destination_id")
        if not self.target_version:
            raise ValueError("manifest target version must be nonempty")
        if self.protocol != DESTINATION_MANIFEST_PROTOCOL:
            raise ValueError("unsupported destination manifest protocol")
        if not self.extents:
            raise ValueError("manifest must publish at least one extent")
        extent_ids = [extent.extent_id for extent in self.extents]
        if len(set(extent_ids)) != len(extent_ids):
            raise ValueError("manifest extent IDs must be unique")
        record_ids = [
            record_id
            for extent in self.extents
            for record_id in extent.record_ids
        ]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("manifest record IDs must be globally unique")
        if any(
            extent.served_kv_target != self.target_version
            for extent in self.extents
        ):
            raise ValueError("manifest extents have different target versions")
        if self.record_count != len(record_ids):
            raise ValueError("manifest record count differs from its extents")
        if self.token_count != sum(extent.token_count for extent in self.extents):
            raise ValueError("manifest token count differs from its extents")
        if self.payload_bytes != sum(
            extent.payload_bytes for extent in self.extents
        ):
            raise ValueError("manifest payload bytes differ from its extents")
        if (self.metadata_sha256 is None) != (self.metadata_json is None):
            raise ValueError("manifest metadata payload and SHA-256 must coexist")
        if self.metadata_sha256 is not None and (
            not re.fullmatch(r"[0-9a-f]{64}", self.metadata_sha256)
            or not isinstance(json.loads(self.metadata_json), dict)
            or _canonical_metadata_json(json.loads(self.metadata_json))
            != self.metadata_json
            or hashlib.sha256(self.metadata_json.encode("utf-8")).hexdigest()
            != self.metadata_sha256
        ):
            raise ValueError("manifest metadata SHA-256 is invalid")

    @property
    def record_ids(self) -> tuple[int, ...]:
        return tuple(
            record_id
            for extent in self.extents
            for record_id in extent.record_ids
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "job_id": self.job_id,
            "target_version": self.target_version,
            "destination_id": self.destination_id,
            "destination_kind": self.destination_kind.value,
            "publication_mode": self.publication_mode.value,
            "extents": [extent.to_dict() for extent in self.extents],
            "record_count": self.record_count,
            "token_count": self.token_count,
            "payload_bytes": self.payload_bytes,
            "metadata_sha256": self.metadata_sha256,
            "metadata": (
                None
                if self.metadata_json is None
                else json.loads(self.metadata_json)
            ),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> KVVersionManifest:
        return cls(
            protocol=str(value["protocol"]),
            job_id=str(value["job_id"]),
            target_version=str(value["target_version"]),
            destination_id=str(value["destination_id"]),
            destination_kind=DestinationKind(str(value["destination_kind"])),
            publication_mode=PublicationMode(str(value["publication_mode"])),
            extents=tuple(
                PublishedKVExtent.from_dict(item)
                for item in value["extents"]
            ),
            record_count=int(value["record_count"]),
            token_count=int(value["token_count"]),
            payload_bytes=int(value["payload_bytes"]),
            metadata_sha256=(
                None
                if value.get("metadata_sha256") is None
                else str(value["metadata_sha256"])
            ),
            metadata_json=(
                None
                if value.get("metadata") is None
                else _canonical_metadata_json(value["metadata"])
            ),
        )


class KVPublicationTransaction:
    def __init__(
        self,
        destination: KVUpdateDestination,
        transaction_id: str,
        job_id: str,
        target_version: str,
        expected_record_ids: tuple[int, ...],
        metadata: dict[str, object] | None,
    ) -> None:
        _validate_identifier(job_id, "job_id")
        if not target_version:
            raise ValueError("target_version must be nonempty")
        if not expected_record_ids:
            raise ValueError("expected_record_ids must be nonempty")
        if len(set(expected_record_ids)) != len(expected_record_ids):
            raise ValueError("expected_record_ids must be unique")
        metadata_json = (
            None if metadata is None else _canonical_metadata_json(metadata)
        )
        self.destination = destination
        self.transaction_id = transaction_id
        self.job_id = job_id
        self.target_version = target_version
        self.expected_record_ids = expected_record_ids
        self.metadata_json = metadata_json
        self.metadata_sha256 = (
            None
            if metadata_json is None
            else hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()
        )
        self._expected = set(expected_record_ids)
        self._staged_records: set[int] = set()
        self._extents: list[PublishedKVExtent] = []
        self._closed = False
        self._lock = threading.Lock()

    @property
    def staged_extent_count(self) -> int:
        return len(self._extents)

    @property
    def staged_record_count(self) -> int:
        return len(self._staged_records)

    def stage(
        self,
        extent_id: str,
        batch: JaggedMigratedKVBatch,
    ) -> PublishedKVExtent:
        with self._lock:
            if self._closed:
                raise RuntimeError("publication transaction is closed")
            _validate_identifier(extent_id, "extent_id")
            if any(extent.extent_id == extent_id for extent in self._extents):
                raise ValueError("extent_id is already staged")
            if batch.served_kv_target != self.target_version:
                raise ValueError("batch target differs from publication target")
            records = set(batch.record_ids)
            if not records.issubset(self._expected):
                raise ValueError("batch contains records outside the publication job")
            if records.intersection(self._staged_records):
                raise ValueError("record is already staged in this publication")
            extent = self.destination._stage(
                transaction_id=self.transaction_id,
                job_id=self.job_id,
                target_version=self.target_version,
                extent_id=extent_id,
                batch=batch,
            )
            self._staged_records.update(records)
            self._extents.append(extent)
            return extent

    def commit(self) -> KVVersionManifest:
        with self._lock:
            if self._closed:
                raise RuntimeError("publication transaction is closed")
            if self._staged_records != self._expected:
                missing = sorted(self._expected - self._staged_records)
                raise ValueError(f"publication is missing {len(missing)} records")
            manifest = KVVersionManifest(
                job_id=self.job_id,
                target_version=self.target_version,
                destination_id=self.destination.destination_id,
                destination_kind=self.destination.capabilities.kind,
                publication_mode=self.destination.capabilities.publication_mode,
                extents=tuple(self._extents),
                record_count=len(self._staged_records),
                token_count=sum(extent.token_count for extent in self._extents),
                payload_bytes=sum(extent.payload_bytes for extent in self._extents),
                metadata_sha256=self.metadata_sha256,
                metadata_json=self.metadata_json,
            )
            self.destination._commit(self.transaction_id, manifest)
            self._closed = True
            return manifest

    def abort(self) -> None:
        with self._lock:
            if not self._closed:
                self.destination._abort(self.transaction_id)
                self._closed = True

    def __enter__(self) -> KVPublicationTransaction:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._closed:
            self.abort()


class KVUpdateDestination(ABC):
    def __init__(
        self,
        destination_id: str,
        capabilities: DestinationCapabilities,
    ) -> None:
        _validate_identifier(destination_id, "destination_id")
        self.destination_id = destination_id
        self.capabilities = capabilities

    def begin(
        self,
        job_id: str,
        target_version: str,
        expected_record_ids: tuple[int, ...],
        metadata: dict[str, object] | None = None,
    ) -> KVPublicationTransaction:
        transaction_id = uuid.uuid4().hex
        transaction = KVPublicationTransaction(
            destination=self,
            transaction_id=transaction_id,
            job_id=job_id,
            target_version=target_version,
            expected_record_ids=expected_record_ids,
            metadata=metadata,
        )
        self._begin(transaction_id, job_id, target_version)
        return transaction

    @abstractmethod
    def _begin(
        self,
        transaction_id: str,
        job_id: str,
        target_version: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _stage(
        self,
        transaction_id: str,
        job_id: str,
        target_version: str,
        extent_id: str,
        batch: JaggedMigratedKVBatch,
    ) -> PublishedKVExtent:
        raise NotImplementedError

    @abstractmethod
    def _commit(
        self,
        transaction_id: str,
        manifest: KVVersionManifest,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _abort(self, transaction_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def manifest(self, target_version: str) -> KVVersionManifest:
        raise NotImplementedError

    @abstractmethod
    def load_extent(
        self,
        target_version: str,
        extent_id: str,
    ) -> JaggedMigratedKVBatch:
        raise NotImplementedError

    @abstractmethod
    def staging_exists(self, transaction_id: str) -> bool:
        raise NotImplementedError


class DRAMKVUpdateDestination(KVUpdateDestination):
    def __init__(
        self,
        destination_id: str = "dram",
        require_pinned: bool = False,
    ) -> None:
        super().__init__(
            destination_id,
            DestinationCapabilities(
                kind=DestinationKind.DRAM,
                publication_mode=PublicationMode.HOST_STAGED,
                durable=False,
                atomic_manifest=True,
                supports_readback=True,
            ),
        )
        self._lock = threading.Lock()
        self._staging: dict[str, dict[str, JaggedMigratedKVBatch]] = {}
        self._committed: dict[str, dict[str, JaggedMigratedKVBatch]] = {}
        self._manifests: dict[str, KVVersionManifest] = {}
        self.require_pinned = require_pinned

    def _begin(
        self,
        transaction_id: str,
        job_id: str,
        target_version: str,
    ) -> None:
        with self._lock:
            if target_version in self._manifests:
                raise FileExistsError("target version is already published")
            self._staging[transaction_id] = {}

    def _stage(
        self,
        transaction_id: str,
        job_id: str,
        target_version: str,
        extent_id: str,
        batch: JaggedMigratedKVBatch,
    ) -> PublishedKVExtent:
        if batch.k.device.type != "cpu":
            raise ValueError("DRAM publication requires CPU-resident K/V")
        if self.require_pinned and not all(
            value.is_pinned()
            for value in (batch.k, batch.v, batch.lengths, batch.offsets)
        ):
            raise ValueError("DRAM publication requires pinned retained tensors")
        with self._lock:
            staged = self._staging[transaction_id]
            if extent_id in staged:
                raise ValueError("extent is already staged")
            staged[extent_id] = _clone_batch(batch)
        return PublishedKVExtent(
            extent_id=extent_id,
            record_ids=batch.record_ids,
            migration_anchor_version=batch.migration_anchor_version,
            served_kv_target=batch.served_kv_target,
            num_layers=batch.k.shape[0],
            token_count=batch.token_count,
            kv_width=batch.k.shape[2],
            dtype=_dtype_name(batch.k.dtype),
            payload_bytes=batch.nbytes,
            location=(
                f"dram://{self.destination_id}/"
                f"{_version_token(target_version)}/{extent_id}"
            ),
            device="cpu",
        )

    def _commit(
        self,
        transaction_id: str,
        manifest: KVVersionManifest,
    ) -> None:
        with self._lock:
            if manifest.target_version in self._manifests:
                raise FileExistsError("target version is already published")
            staged = self._staging.pop(transaction_id)
            if set(staged) != {
                extent.extent_id for extent in manifest.extents
            }:
                raise RuntimeError("staged DRAM extents differ from manifest")
            self._committed[manifest.target_version] = staged
            self._manifests[manifest.target_version] = manifest

    def _abort(self, transaction_id: str) -> None:
        with self._lock:
            self._staging.pop(transaction_id, None)

    def manifest(self, target_version: str) -> KVVersionManifest:
        with self._lock:
            try:
                return self._manifests[target_version]
            except KeyError as exc:
                raise KeyError("target version is not published") from exc

    def load_extent(
        self,
        target_version: str,
        extent_id: str,
    ) -> JaggedMigratedKVBatch:
        with self._lock:
            try:
                return _clone_batch(
                    self._committed[target_version][extent_id]
                )
            except KeyError as exc:
                raise KeyError("published DRAM extent is unavailable") from exc

    def staging_exists(self, transaction_id: str) -> bool:
        with self._lock:
            return transaction_id in self._staging


class HBMKVUpdateDestination(KVUpdateDestination):
    def __init__(
        self,
        devices: tuple[torch.device | str, ...],
        destination_id: str = "hbm",
        capacity_bytes: int | None = None,
    ) -> None:
        if not devices:
            raise ValueError("HBM destination requires at least one device")
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        resolved = []
        for value in devices:
            device = torch.device(value)
            if device.type != "cuda":
                raise ValueError("HBM destination devices must be CUDA devices")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            if device.index >= torch.cuda.device_count():
                raise ValueError("HBM destination device is unavailable")
            resolved.append(device)
        if len(set(resolved)) != len(resolved):
            raise ValueError("HBM destination devices must be unique")
        if capacity_bytes is not None and capacity_bytes < 1:
            raise ValueError("capacity_bytes must be positive")
        super().__init__(
            destination_id,
            DestinationCapabilities(
                kind=DestinationKind.HBM,
                publication_mode=PublicationMode.DIRECT_DEVICE,
                durable=False,
                atomic_manifest=True,
                supports_readback=True,
            ),
        )
        self.devices = tuple(resolved)
        self.capacity_bytes = capacity_bytes
        self._lock = threading.Lock()
        self._staging: dict[str, dict[str, JaggedMigratedKVBatch]] = {}
        self._committed: dict[str, dict[str, JaggedMigratedKVBatch]] = {}
        self._manifests: dict[str, KVVersionManifest] = {}

    def _begin(
        self,
        transaction_id: str,
        job_id: str,
        target_version: str,
    ) -> None:
        with self._lock:
            if target_version in self._manifests:
                raise FileExistsError("target version is already published")
            self._staging[transaction_id] = {}

    def _stage(
        self,
        transaction_id: str,
        job_id: str,
        target_version: str,
        extent_id: str,
        batch: JaggedMigratedKVBatch,
    ) -> PublishedKVExtent:
        if batch.k.device not in self.devices:
            raise ValueError("HBM extent is not on an allowed destination device")
        with self._lock:
            staged = self._staging[transaction_id]
            if extent_id in staged:
                raise ValueError("extent is already staged")
            staged[extent_id] = _clone_batch(batch)
        return PublishedKVExtent(
            extent_id=extent_id,
            record_ids=batch.record_ids,
            migration_anchor_version=batch.migration_anchor_version,
            served_kv_target=batch.served_kv_target,
            num_layers=batch.k.shape[0],
            token_count=batch.token_count,
            kv_width=batch.k.shape[2],
            dtype=_dtype_name(batch.k.dtype),
            payload_bytes=batch.nbytes,
            location=(
                f"hbm://{self.destination_id}/{batch.k.device}/"
                f"{_version_token(target_version)}/{extent_id}"
            ),
            device=str(batch.k.device),
        )

    def _commit(
        self,
        transaction_id: str,
        manifest: KVVersionManifest,
    ) -> None:
        with self._lock:
            if manifest.target_version in self._manifests:
                raise FileExistsError("target version is already published")
            committed_bytes = sum(
                value.payload_bytes for value in self._manifests.values()
            )
            if (
                self.capacity_bytes is not None
                and committed_bytes + manifest.payload_bytes > self.capacity_bytes
            ):
                raise MemoryError("HBM destination capacity is exceeded")
            staged = self._staging.pop(transaction_id)
            if set(staged) != {
                extent.extent_id for extent in manifest.extents
            }:
                raise RuntimeError("staged HBM extents differ from manifest")
            self._committed[manifest.target_version] = staged
            self._manifests[manifest.target_version] = manifest

    def _abort(self, transaction_id: str) -> None:
        with self._lock:
            self._staging.pop(transaction_id, None)

    def manifest(self, target_version: str) -> KVVersionManifest:
        with self._lock:
            try:
                return self._manifests[target_version]
            except KeyError as exc:
                raise KeyError("target version is not published") from exc

    def load_extent(
        self,
        target_version: str,
        extent_id: str,
    ) -> JaggedMigratedKVBatch:
        with self._lock:
            try:
                return _clone_batch(
                    self._committed[target_version][extent_id]
                )
            except KeyError as exc:
                raise KeyError("published HBM extent is unavailable") from exc

    def staging_exists(self, transaction_id: str) -> bool:
        with self._lock:
            return transaction_id in self._staging


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes(path: Path, payload: bytes, durable: bool) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        if durable:
            os.fsync(handle.fileno())
    os.replace(temporary, path)
    if durable:
        _fsync_directory(path.parent)


class FilesystemKVUpdateDestination(KVUpdateDestination):
    def __init__(
        self,
        root: Path | str,
        destination_id: str = "local-ssd",
        durable: bool = True,
    ) -> None:
        super().__init__(
            destination_id,
            DestinationCapabilities(
                kind=DestinationKind.LOCAL_SSD,
                publication_mode=PublicationMode.HOST_STAGED,
                durable=durable,
                atomic_manifest=True,
                supports_readback=True,
            ),
        )
        self.root = Path(root).expanduser().resolve()
        self.staging_root = self.root / ".staging"
        self.versions_root = self.root / "versions"
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.versions_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._staging: dict[str, Path] = {}

    def _version_path(self, target_version: str) -> Path:
        return self.versions_root / _version_token(target_version)

    def _begin(
        self,
        transaction_id: str,
        job_id: str,
        target_version: str,
    ) -> None:
        final = self._version_path(target_version)
        with self._lock:
            if final.exists():
                raise FileExistsError("target version is already published")
            staging = self.staging_root / transaction_id
            staging.mkdir()
            self._staging[transaction_id] = staging

    def _stage(
        self,
        transaction_id: str,
        job_id: str,
        target_version: str,
        extent_id: str,
        batch: JaggedMigratedKVBatch,
    ) -> PublishedKVExtent:
        if batch.k.device.type != "cpu":
            raise ValueError("filesystem publication requires CPU-resident K/V")
        payload = _serialize_batch(batch)
        checksum = hashlib.sha256(payload).hexdigest()
        with self._lock:
            staging = self._staging[transaction_id]
        path = staging / f"{extent_id}.pt"
        if path.exists():
            raise ValueError("extent is already staged")
        _write_bytes(path, payload, self.capabilities.durable)
        final = self._version_path(target_version) / path.name
        return PublishedKVExtent(
            extent_id=extent_id,
            record_ids=batch.record_ids,
            migration_anchor_version=batch.migration_anchor_version,
            served_kv_target=batch.served_kv_target,
            num_layers=batch.k.shape[0],
            token_count=batch.token_count,
            kv_width=batch.k.shape[2],
            dtype=_dtype_name(batch.k.dtype),
            payload_bytes=batch.nbytes,
            location=str(final),
            device="cpu",
            checksum_sha256=checksum,
        )

    def _commit(
        self,
        transaction_id: str,
        manifest: KVVersionManifest,
    ) -> None:
        with self._lock:
            staging = self._staging[transaction_id]
        manifest_payload = json.dumps(
            manifest.to_dict(),
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        _write_bytes(
            staging / "manifest.json",
            manifest_payload,
            self.capabilities.durable,
        )
        final = self._version_path(manifest.target_version)
        with self._lock:
            if final.exists():
                raise FileExistsError("target version is already published")
            staging.rename(final)
            self._staging.pop(transaction_id)
        if self.capabilities.durable:
            _fsync_directory(self.versions_root)

    def _abort(self, transaction_id: str) -> None:
        with self._lock:
            staging = self._staging.pop(transaction_id, None)
        if staging is not None and staging.exists():
            shutil.rmtree(staging)

    def manifest(self, target_version: str) -> KVVersionManifest:
        path = self._version_path(target_version) / "manifest.json"
        if not path.exists():
            raise KeyError("target version is not published")
        return KVVersionManifest.from_dict(json.loads(path.read_text()))

    def load_extent(
        self,
        target_version: str,
        extent_id: str,
    ) -> JaggedMigratedKVBatch:
        manifest = self.manifest(target_version)
        try:
            extent = next(
                value
                for value in manifest.extents
                if value.extent_id == extent_id
            )
        except StopIteration as exc:
            raise KeyError("published filesystem extent is unavailable") from exc
        payload = Path(extent.location).read_bytes()
        if (
            extent.checksum_sha256 is not None
            and hashlib.sha256(payload).hexdigest() != extent.checksum_sha256
        ):
            raise RuntimeError("filesystem extent checksum mismatch")
        return _deserialize_batch(payload, extent)

    def staging_exists(self, transaction_id: str) -> bool:
        with self._lock:
            staging = self._staging.get(transaction_id)
        return staging is not None and staging.exists()


class RemoteObjectStore(Protocol):
    @property
    def durable(self) -> bool:
        ...

    def put_if_absent(self, key: str, payload: bytes) -> bool:
        ...

    def get(self, key: str) -> bytes:
        ...

    def delete(self, key: str) -> None:
        ...


class InMemoryRemoteObjectStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._objects: dict[str, bytes] = {}

    @property
    def durable(self) -> bool:
        return False

    def put_if_absent(self, key: str, payload: bytes) -> bool:
        with self._lock:
            if key in self._objects:
                return False
            self._objects[key] = payload
            return True

    def get(self, key: str) -> bytes:
        with self._lock:
            try:
                return self._objects[key]
            except KeyError as exc:
                raise KeyError("remote object is unavailable") from exc

    def delete(self, key: str) -> None:
        with self._lock:
            self._objects.pop(key, None)


class RemoteKVUpdateDestination(KVUpdateDestination):
    def __init__(
        self,
        store: RemoteObjectStore,
        destination_id: str = "remote",
        prefix: str = "streamkv",
    ) -> None:
        _validate_identifier(prefix, "prefix")
        super().__init__(
            destination_id,
            DestinationCapabilities(
                kind=DestinationKind.REMOTE,
                publication_mode=PublicationMode.HOST_STAGED,
                durable=store.durable,
                atomic_manifest=True,
                supports_readback=True,
            ),
        )
        self.store = store
        self.prefix = prefix
        self._lock = threading.Lock()
        self._staging: dict[str, list[str]] = {}

    def _manifest_key(self, target_version: str) -> str:
        return f"{self.prefix}/versions/{_version_token(target_version)}/manifest.json"

    def _begin(
        self,
        transaction_id: str,
        job_id: str,
        target_version: str,
    ) -> None:
        try:
            self.store.get(self._manifest_key(target_version))
        except KeyError:
            pass
        else:
            raise FileExistsError("target version is already published")
        with self._lock:
            self._staging[transaction_id] = []

    def _stage(
        self,
        transaction_id: str,
        job_id: str,
        target_version: str,
        extent_id: str,
        batch: JaggedMigratedKVBatch,
    ) -> PublishedKVExtent:
        if batch.k.device.type != "cpu":
            raise ValueError("remote publication requires CPU-resident K/V")
        payload = _serialize_batch(batch)
        checksum = hashlib.sha256(payload).hexdigest()
        key = f"{self.prefix}/objects/{transaction_id}/{extent_id}.pt"
        if not self.store.put_if_absent(key, payload):
            raise FileExistsError("remote extent object already exists")
        with self._lock:
            self._staging[transaction_id].append(key)
        return PublishedKVExtent(
            extent_id=extent_id,
            record_ids=batch.record_ids,
            migration_anchor_version=batch.migration_anchor_version,
            served_kv_target=batch.served_kv_target,
            num_layers=batch.k.shape[0],
            token_count=batch.token_count,
            kv_width=batch.k.shape[2],
            dtype=_dtype_name(batch.k.dtype),
            payload_bytes=batch.nbytes,
            location=f"remote://{self.destination_id}/{key}",
            device="cpu",
            checksum_sha256=checksum,
        )

    def _commit(
        self,
        transaction_id: str,
        manifest: KVVersionManifest,
    ) -> None:
        payload = json.dumps(
            manifest.to_dict(),
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        if not self.store.put_if_absent(
            self._manifest_key(manifest.target_version),
            payload,
        ):
            raise FileExistsError("target version is already published")
        with self._lock:
            self._staging.pop(transaction_id)

    def _abort(self, transaction_id: str) -> None:
        with self._lock:
            keys = self._staging.pop(transaction_id, [])
        for key in keys:
            self.store.delete(key)

    def manifest(self, target_version: str) -> KVVersionManifest:
        try:
            payload = self.store.get(self._manifest_key(target_version))
        except KeyError as exc:
            raise KeyError("target version is not published") from exc
        return KVVersionManifest.from_dict(json.loads(payload))

    def load_extent(
        self,
        target_version: str,
        extent_id: str,
    ) -> JaggedMigratedKVBatch:
        manifest = self.manifest(target_version)
        try:
            extent = next(
                value
                for value in manifest.extents
                if value.extent_id == extent_id
            )
        except StopIteration as exc:
            raise KeyError("published remote extent is unavailable") from exc
        location_prefix = f"remote://{self.destination_id}/"
        if not extent.location.startswith(location_prefix):
            raise RuntimeError("remote extent location belongs to another destination")
        payload = self.store.get(extent.location.removeprefix(location_prefix))
        if (
            extent.checksum_sha256 is not None
            and hashlib.sha256(payload).hexdigest() != extent.checksum_sha256
        ):
            raise RuntimeError("remote extent checksum mismatch")
        return _deserialize_batch(payload, extent)

    def staging_exists(self, transaction_id: str) -> bool:
        with self._lock:
            return transaction_id in self._staging
