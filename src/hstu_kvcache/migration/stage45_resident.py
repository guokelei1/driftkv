from __future__ import annotations

import hashlib
import io
import os
import statistics
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from .cohort_jagged import JaggedMigratedKVBatch
from .destination import (
    HBMKVUpdateDestination,
    KVVersionManifest,
)
from .stage4_engine import Stage4RuntimeConfig, Stage4Transform
from .stage4_source import (
    SOURCE_SHARD_PROTOCOL,
    LazyStage4SourceReader,
    SourceReadMetrics,
    Stage4ExtentSpec,
    Stage4SourceBatch,
    _tensor_nbytes,
    _validate_source_tensors,
    build_stage4_extents,
    place_stage4_extents_lpt,
)

STAGE45_RESIDENT_PROTOCOL = "cohortkv_stage4_5_resident_ceiling_v1"
STAGE45_SOURCE_TIERS = ("hbm_resident", "dram_resident")
STAGE45_METHODS = ("compiled", "exact", "no_transform")


def _available_host_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")


def _padded_source_bytes(extent: Stage4ExtentSpec) -> int:
    total = len(extent.records) * torch.tensor(
        [],
        dtype=torch.long,
    ).element_size()
    for record in extent.records:
        for representation in extent.representations:
            logical = record.shard_map[representation].logical_bytes
            if logical % record.prefix_tokens:
                raise ValueError("Stage 4.5 source bytes are not token-linear")
            total += logical // record.prefix_tokens * extent.sequence_width
    return total


def _extent_output_bytes(extent: Stage4ExtentSpec) -> int:
    index_bytes = torch.tensor([], dtype=torch.long).element_size()
    return extent.logical_output_bytes + (2 * len(extent.records) + 1) * index_bytes


def _make_device_output(
    spec: Stage4ExtentSpec,
    num_layers: int,
    kv_width: int,
    device: torch.device,
) -> JaggedMigratedKVBatch:
    lengths = torch.tensor(
        [value.prefix_tokens for value in spec.records],
        dtype=torch.long,
        device=device,
    )
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            lengths.cumsum(0),
        )
    )
    shape = (num_layers, spec.token_count, kv_width)
    return JaggedMigratedKVBatch(
        record_ids=spec.record_ids,
        migration_anchor_version=spec.migration_anchor_version,
        served_kv_target=spec.served_kv_target,
        k=torch.empty(shape, dtype=torch.float16, device=device),
        v=torch.empty(shape, dtype=torch.float16, device=device),
        lengths=lengths,
        offsets=offsets,
    )


def _pin_batch(batch: Stage4SourceBatch) -> Stage4SourceBatch:
    values = {
        name: (
            None
            if getattr(batch, name) is None
            else getattr(batch, name).pin_memory()
        )
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
        record_ids=batch.record_ids,
        migration_anchor_version=batch.migration_anchor_version,
        served_kv_target=batch.served_kv_target,
        lengths=batch.lengths.pin_memory(),
        sequence_width=batch.sequence_width,
        residual_start_layer=batch.residual_start_layer,
        **values,
    )


@dataclass
class _ProfiledSourceRead:
    file_read_seconds: float = 0.0
    integrity_seconds: float = 0.0
    deserialize_validate_seconds: float = 0.0
    pageable_allocation_seconds: float = 0.0
    batch_pack_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return (
            self.file_read_seconds
            + self.integrity_seconds
            + self.deserialize_validate_seconds
            + self.pageable_allocation_seconds
            + self.batch_pack_seconds
        )

    def merge(self, other: _ProfiledSourceRead) -> None:
        self.file_read_seconds += other.file_read_seconds
        self.integrity_seconds += other.integrity_seconds
        self.deserialize_validate_seconds += (
            other.deserialize_validate_seconds
        )
        self.pageable_allocation_seconds += (
            other.pageable_allocation_seconds
        )
        self.batch_pack_seconds += other.batch_pack_seconds


def _read_shard_profiled(
    reader: LazyStage4SourceReader,
    record,
    representation: str,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, object],
    int,
    _ProfiledSourceRead,
]:
    descriptor = record.shard_map[representation]
    path = (reader.root / descriptor.path).resolve()
    if reader.root not in path.parents:
        raise ValueError("Stage 4.5 source shard resolves outside its root")
    profile = _ProfiledSourceRead()
    started = time.perf_counter()
    encoded = path.read_bytes()
    profile.file_read_seconds = time.perf_counter() - started
    started = time.perf_counter()
    digest = hashlib.sha256(encoded).hexdigest()
    if len(encoded) != descriptor.physical_bytes or digest != descriptor.sha256:
        raise ValueError("Stage 4.5 source shard integrity check failed")
    profile.integrity_seconds = time.perf_counter() - started
    started = time.perf_counter()
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
        raise ValueError("Stage 4.5 source shard identity check failed")
    tensors = dict(payload["tensors"])
    metadata = dict(payload.get("metadata", {}))
    _validate_source_tensors(
        representation,
        tensors,
        record.prefix_tokens,
        metadata,
    )
    if _tensor_nbytes(tensors) != descriptor.logical_bytes:
        raise ValueError("Stage 4.5 source shard byte count is invalid")
    profile.deserialize_validate_seconds = time.perf_counter() - started
    return tensors, metadata, len(encoded), profile


def _read_extent_profiled(
    reader: LazyStage4SourceReader,
    extent: Stage4ExtentSpec,
) -> tuple[Stage4SourceBatch, SourceReadMetrics, _ProfiledSourceRead]:
    profile = _ProfiledSourceRead()
    started = time.perf_counter()
    batch_size = len(extent.records)
    width = extent.sequence_width
    lengths = torch.tensor(
        [value.prefix_tokens for value in extent.records],
        dtype=torch.long,
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
    manifest = reader.manifest
    if "normalized_capsule_fp16" in extent.representations:
        values["normed"] = torch.zeros(
            manifest.num_layers,
            batch_size,
            width,
            manifest.hidden_size,
            dtype=torch.float16,
        )
    if "old_kv_fp16" in extent.representations:
        shape = (
            manifest.num_layers,
            batch_size,
            width,
            manifest.kv_width,
        )
        values["old_k"] = torch.zeros(shape, dtype=torch.float16)
        values["old_v"] = torch.zeros(shape, dtype=torch.float16)
    if "raw_history" in extent.representations:
        shape = (batch_size, width)
        values["item_ids"] = torch.zeros(shape, dtype=torch.long)
        values["behaviors"] = torch.zeros(shape, dtype=torch.long)
        values["time_deltas"] = torch.zeros(shape, dtype=torch.float32)
    profile.pageable_allocation_seconds += time.perf_counter() - started
    residual_start = None
    physical_bytes = 0
    logical_bytes = 0
    maximum_transient = 0
    for row, record in enumerate(extent.records):
        length = record.prefix_tokens
        for representation in extent.representations:
            tensors, metadata, encoded_bytes, shard_profile = (
                _read_shard_profiled(
                    reader,
                    record,
                    representation,
                )
            )
            profile.merge(shard_profile)
            decoded_bytes = _tensor_nbytes(tensors)
            physical_bytes += encoded_bytes
            logical_bytes += decoded_bytes
            maximum_transient = max(
                maximum_transient,
                encoded_bytes + decoded_bytes,
            )
            started = time.perf_counter()
            if representation == "normalized_capsule_fp16":
                values["normed"][:, row, :length].copy_(tensors["normed"])
            elif representation == "old_kv_fp16":
                values["old_k"][:, row, :length].copy_(tensors["k"])
                values["old_v"][:, row, :length].copy_(tensors["v"])
            elif representation == "raw_history":
                values["item_ids"][row, :length].copy_(tensors["item_ids"])
                values["behaviors"][row, :length].copy_(
                    tensors["behaviors"]
                )
                values["time_deltas"][row, :length].copy_(
                    tensors["time_deltas"]
                )
            elif representation == "residual_hidden_suffix_bf16":
                start_layer = int(metadata["start_layer"])
                if residual_start is None:
                    allocation_started = time.perf_counter()
                    residual_start = start_layer
                    values["residual_hidden_states"] = torch.zeros(
                        manifest.num_layers - start_layer,
                        batch_size,
                        width,
                        manifest.hidden_size,
                        dtype=torch.bfloat16,
                    )
                    profile.pageable_allocation_seconds += (
                        time.perf_counter() - allocation_started
                    )
                    started = time.perf_counter()
                if start_layer != residual_start:
                    raise ValueError(
                        "Stage 4.5 residual suffix starts differ"
                    )
                values["residual_hidden_states"][:, row, :length].copy_(
                    tensors["hidden_states"]
                )
            profile.batch_pack_seconds += time.perf_counter() - started
    if (
        physical_bytes != extent.physical_input_bytes
        or logical_bytes != extent.logical_input_bytes
    ):
        raise ValueError("Stage 4.5 profiled source totals differ")
    batch = Stage4SourceBatch(
        record_ids=extent.record_ids,
        migration_anchor_version=extent.migration_anchor_version,
        served_kv_target=extent.served_kv_target,
        lengths=lengths,
        sequence_width=width,
        residual_start_layer=residual_start,
        **values,
    )
    return (
        batch,
        SourceReadMetrics(
            physical_bytes=physical_bytes,
            logical_bytes=logical_bytes,
            peak_source_resident_bytes=batch.nbytes + maximum_transient,
        ),
        profile,
    )


def _validate_transforms(
    transforms: tuple[Stage4Transform, ...],
    runtime_config: Stage4RuntimeConfig,
) -> tuple[str, str]:
    if not transforms:
        raise ValueError("Stage 4.5 requires at least one transform")
    methods = {value.method for value in transforms}
    targets = {value.target_version for value in transforms}
    devices = tuple(value.device for value in transforms)
    if len(methods) != 1 or next(iter(methods)) not in STAGE45_METHODS:
        raise ValueError("Stage 4.5 transforms have an unsupported method")
    if len(targets) != 1:
        raise ValueError("Stage 4.5 transforms must share one target")
    if len(set(devices)) != len(devices) or any(
        value.type != "cuda" for value in devices
    ):
        raise ValueError("Stage 4.5 transforms require unique CUDA devices")
    if len(transforms) not in {1, 2, 4}:
        raise ValueError("Stage 4.5 GPU count must be one, two, or four")
    method = next(iter(methods))
    if method == "compiled" and (
        runtime_config.compiled_operator is None
        or any(
            getattr(value, "runtime_variant", None)
            != runtime_config.compiled_operator
            for value in transforms
        )
    ):
        raise ValueError("Stage 4.5 compiled runtime variant differs")
    if method == "exact" and (
        runtime_config.exact_compute is None
        or any(
            getattr(value, "runtime_variant", None)
            != runtime_config.exact_compute
            for value in transforms
        )
    ):
        raise ValueError("Stage 4.5 exact runtime variant differs")
    return method, next(iter(targets))


@dataclass(frozen=True)
class Stage45ResidentPlan:
    source_manifest_path: Path
    source_manifest_sha256: str
    workload_content_sha256: str
    method: str
    target_version: str
    source_tier: str
    runtime_config: Stage4RuntimeConfig
    record_ids: tuple[int, ...]
    num_layers: int
    kv_width: int
    extents: tuple[Stage4ExtentSpec, ...]
    assignments: tuple[tuple[Stage4ExtentSpec, ...], ...]
    protocol: str = STAGE45_RESIDENT_PROTOCOL

    @property
    def record_count(self) -> int:
        return len(self.record_ids)

    @property
    def prefix_tokens(self) -> int:
        return sum(value.token_count for value in self.extents)

    @property
    def logical_source_bytes(self) -> int:
        return sum(value.logical_input_bytes for value in self.extents)

    @property
    def physical_source_bytes(self) -> int:
        return sum(value.physical_input_bytes for value in self.extents)

    @property
    def resident_source_bytes(self) -> int:
        return sum(_padded_source_bytes(value) for value in self.extents)

    @property
    def logical_output_bytes(self) -> int:
        return sum(value.logical_output_bytes for value in self.extents)

    @property
    def physical_output_bytes(self) -> int:
        return sum(_extent_output_bytes(value) for value in self.extents)

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "method": self.method,
            "target_version": self.target_version,
            "source_tier": self.source_tier,
            "source_manifest_path": str(self.source_manifest_path),
            "source_manifest_sha256": self.source_manifest_sha256,
            "workload_content_sha256": self.workload_content_sha256,
            "runtime_config": self.runtime_config.to_dict(),
            "record_count": self.record_count,
            "prefix_tokens": self.prefix_tokens,
            "extent_count": len(self.extents),
            "logical_source_bytes": self.logical_source_bytes,
            "physical_source_bytes": self.physical_source_bytes,
            "resident_source_bytes": self.resident_source_bytes,
            "logical_output_bytes": self.logical_output_bytes,
            "physical_output_bytes": self.physical_output_bytes,
            "per_gpu": [
                {
                    "index": index,
                    "extent_count": len(assignment),
                    "record_count": sum(
                        len(value.records) for value in assignment
                    ),
                    "prefix_tokens": sum(
                        value.token_count for value in assignment
                    ),
                    "resident_source_bytes": sum(
                        _padded_source_bytes(value) for value in assignment
                    ),
                    "physical_output_bytes": sum(
                        _extent_output_bytes(value) for value in assignment
                    ),
                }
                for index, assignment in enumerate(self.assignments)
            ],
        }


def build_stage45_resident_plan(
    source_manifest_path: Path | str,
    transforms: tuple[Stage4Transform, ...],
    source_tier: str,
    runtime_config: Stage4RuntimeConfig,
    record_ids: tuple[int, ...] | None = None,
    expected_source_manifest_sha256: str | None = None,
    expected_workload_content_sha256: str | None = None,
) -> Stage45ResidentPlan:
    if source_tier not in STAGE45_SOURCE_TIERS:
        raise ValueError("unsupported Stage 4.5 source tier")
    method, target_version = _validate_transforms(
        transforms,
        runtime_config,
    )
    reader = LazyStage4SourceReader(
        source_manifest_path,
        expected_workload_content_sha256,
    )
    if (
        expected_source_manifest_sha256 is not None
        and reader.manifest_file_sha256
        != expected_source_manifest_sha256
    ):
        raise ValueError("Stage 4.5 source manifest hash mismatch")
    if reader.manifest.target_version != target_version:
        raise ValueError("Stage 4.5 source and transform targets differ")
    if record_ids is None:
        record_ids = tuple(value.record_id for value in reader.manifest.records)
    if not record_ids or len(set(record_ids)) != len(record_ids):
        raise ValueError("Stage 4.5 record IDs must be nonempty and unique")
    source_versions = {
        reader.manifest.record_map[value].source_version
        for value in record_ids
    }
    representations = {}
    for source_version in source_versions:
        expected = transforms[0].source_representations(source_version)
        if any(
            value.source_representations(source_version) != expected
            for value in transforms[1:]
        ):
            raise ValueError("Stage 4.5 representation contracts differ")
        representations[source_version] = expected
    extents = build_stage4_extents(
        reader.manifest,
        record_ids,
        representations,
        runtime_config.batch_size,
        runtime_config.length_bucket_width,
    )
    assignments = place_stage4_extents_lpt(extents, len(transforms))
    return Stage45ResidentPlan(
        source_manifest_path=Path(source_manifest_path).resolve(),
        source_manifest_sha256=reader.manifest_file_sha256,
        workload_content_sha256=reader.manifest.workload_content_sha256,
        method=method,
        target_version=target_version,
        source_tier=source_tier,
        runtime_config=runtime_config,
        record_ids=record_ids,
        num_layers=reader.manifest.num_layers,
        kv_width=reader.manifest.kv_width,
        extents=extents,
        assignments=assignments,
    )


def _dram_transient_source_bytes(
    assignment: tuple[Stage4ExtentSpec, ...],
    max_inflight: int,
) -> int:
    pending: deque[int] = deque()
    peak = 0
    for extent in assignment:
        pending.append(_padded_source_bytes(extent))
        peak = max(peak, sum(pending))
        if len(pending) >= max_inflight:
            pending.popleft()
    return peak


def stage45_resident_preflight(
    plan: Stage45ResidentPlan,
    transforms: tuple[Stage4Transform, ...],
    allocator_margin_bytes: int = 512 * 1024**2,
) -> dict[str, object]:
    method, target = _validate_transforms(
        transforms,
        plan.runtime_config,
    )
    if (
        method != plan.method
        or target != plan.target_version
        or len(transforms) != len(plan.assignments)
        or allocator_margin_bytes < 1
    ):
        raise ValueError("Stage 4.5 preflight inputs differ from its plan")
    host_standing = (
        plan.resident_source_bytes
        if plan.source_tier == "dram_resident"
        else 0
    )
    available_host = _available_host_bytes()
    per_gpu = []
    passed = available_host >= host_standing
    for index, (assignment, transform) in enumerate(
        zip(plan.assignments, transforms, strict=True)
    ):
        device = transform.device
        with torch.cuda.device(device):
            free, total = torch.cuda.mem_get_info(device)
            allocated = torch.cuda.memory_allocated(device)
        source = sum(_padded_source_bytes(value) for value in assignment)
        target_bytes = sum(_extent_output_bytes(value) for value in assignment)
        device_source = source if plan.source_tier == "hbm_resident" else 0
        movement = (
            0
            if plan.source_tier == "hbm_resident"
            else _dram_transient_source_bytes(
                assignment,
                plan.runtime_config.max_inflight,
            )
        )
        required = (
            allocated
            + device_source
            + target_bytes
            + movement
            + allocator_margin_bytes
        )
        device_passed = total >= required and free >= required - allocated
        passed = passed and device_passed
        per_gpu.append(
            {
                "index": index,
                "device": str(device),
                "observed_allocated_hbm_bytes": allocated,
                "observed_free_hbm_bytes": free,
                "total_hbm_bytes": total,
                "transform_resident_bytes": transform.resident_bytes,
                "source_standing_hbm_bytes": device_source,
                "source_standing_host_bytes": (
                    source if plan.source_tier == "dram_resident" else 0
                ),
                "assigned_target_bytes": target_bytes,
                "maximum_source_movement_bytes": movement,
                "allocator_margin_bytes": allocator_margin_bytes,
                "required_peak_hbm_bytes": required,
                "passed": device_passed,
            }
        )
    return {
        "protocol": STAGE45_RESIDENT_PROTOCOL,
        "method": plan.method,
        "source_tier": plan.source_tier,
        "record_count": plan.record_count,
        "prefix_tokens": plan.prefix_tokens,
        "observed_available_host_bytes": available_host,
        "required_standing_host_bytes": host_standing,
        "per_gpu": per_gpu,
        "passed": passed,
    }


@dataclass(frozen=True)
class Stage45PreloadDeviceMetrics:
    index: int
    extent_count: int
    record_count: int
    logical_source_bytes: int
    physical_source_bytes: int
    resident_source_bytes: int
    file_read_seconds: float
    integrity_seconds: float
    deserialize_validate_seconds: float
    pageable_allocation_seconds: float
    batch_pack_seconds: float
    pageable_to_pinned_seconds: float
    h2d_seconds: float
    elapsed_seconds: float

    @property
    def file_deserialize_decode_seconds(self) -> float:
        return (
            self.file_read_seconds
            + self.integrity_seconds
            + self.deserialize_validate_seconds
            + self.pageable_allocation_seconds
            + self.batch_pack_seconds
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "extent_count": self.extent_count,
            "record_count": self.record_count,
            "logical_source_bytes": self.logical_source_bytes,
            "physical_source_bytes": self.physical_source_bytes,
            "resident_source_bytes": self.resident_source_bytes,
            "file_read_seconds": self.file_read_seconds,
            "integrity_seconds": self.integrity_seconds,
            "deserialize_validate_seconds": (
                self.deserialize_validate_seconds
            ),
            "pageable_allocation_seconds": (
                self.pageable_allocation_seconds
            ),
            "batch_pack_seconds": self.batch_pack_seconds,
            "file_deserialize_decode_seconds": (
                self.file_deserialize_decode_seconds
            ),
            "pageable_to_pinned_seconds": self.pageable_to_pinned_seconds,
            "h2d_seconds": self.h2d_seconds,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class Stage45PreloadMetrics:
    elapsed_seconds: float
    logical_source_bytes: int
    physical_source_bytes: int
    resident_source_bytes: int
    standing_host_bytes: int
    standing_hbm_bytes: int
    devices: tuple[Stage45PreloadDeviceMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "logical_source_bytes": self.logical_source_bytes,
            "physical_source_bytes": self.physical_source_bytes,
            "resident_source_bytes": self.resident_source_bytes,
            "standing_host_bytes": self.standing_host_bytes,
            "standing_hbm_bytes": self.standing_hbm_bytes,
            "file_read_seconds": sum(
                value.file_read_seconds for value in self.devices
            ),
            "integrity_seconds": sum(
                value.integrity_seconds for value in self.devices
            ),
            "deserialize_validate_seconds": sum(
                value.deserialize_validate_seconds
                for value in self.devices
            ),
            "pageable_allocation_seconds": sum(
                value.pageable_allocation_seconds
                for value in self.devices
            ),
            "batch_pack_seconds": sum(
                value.batch_pack_seconds for value in self.devices
            ),
            "file_deserialize_decode_seconds": sum(
                value.file_deserialize_decode_seconds
                for value in self.devices
            ),
            "pageable_to_pinned_seconds": sum(
                value.pageable_to_pinned_seconds for value in self.devices
            ),
            "h2d_seconds": sum(value.h2d_seconds for value in self.devices),
            "devices": [value.to_dict() for value in self.devices],
        }


@dataclass(frozen=True)
class Stage45ResidentExtent:
    spec: Stage4ExtentSpec
    batch: Stage4SourceBatch


@dataclass(frozen=True)
class Stage45ResidentSource:
    plan: Stage45ResidentPlan
    assignments: tuple[tuple[Stage45ResidentExtent, ...], ...]
    preload: Stage45PreloadMetrics

    @property
    def extent_map(self) -> dict[str, Stage45ResidentExtent]:
        return {
            value.spec.extent_id: value
            for assignment in self.assignments
            for value in assignment
        }


def _materialize_assignment(
    index: int,
    reader: LazyStage4SourceReader,
    assignment: tuple[Stage4ExtentSpec, ...],
    transform: Stage4Transform,
    source_tier: str,
) -> tuple[tuple[Stage45ResidentExtent, ...], Stage45PreloadDeviceMetrics]:
    started = time.perf_counter()
    source_profile = _ProfiledSourceRead()
    pin_seconds = 0.0
    h2d_seconds = 0.0
    resident = []
    for spec in assignment:
        pageable, metrics, profile = _read_extent_profiled(reader, spec)
        source_profile.merge(profile)
        pin_started = time.perf_counter()
        pinned = _pin_batch(pageable)
        pin_seconds += time.perf_counter() - pin_started
        if source_tier == "hbm_resident":
            with torch.cuda.device(transform.device):
                stream = torch.cuda.Stream(device=transform.device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    start.record(stream)
                    batch = pinned.to(transform.device, non_blocking=True)
                    end.record(stream)
                end.synchronize()
                h2d_seconds += start.elapsed_time(end) / 1000.0
        else:
            batch = pinned
        if batch.nbytes != _padded_source_bytes(spec):
            raise RuntimeError("Stage 4.5 resident source byte count differs")
        if (
            metrics.physical_bytes != spec.physical_input_bytes
            or metrics.logical_bytes != spec.logical_input_bytes
        ):
            raise RuntimeError("Stage 4.5 preload source totals differ")
        resident.append(Stage45ResidentExtent(spec=spec, batch=batch))
    return tuple(resident), Stage45PreloadDeviceMetrics(
        index=index,
        extent_count=len(assignment),
        record_count=sum(len(value.records) for value in assignment),
        logical_source_bytes=sum(
            value.logical_input_bytes for value in assignment
        ),
        physical_source_bytes=sum(
            value.physical_input_bytes for value in assignment
        ),
        resident_source_bytes=sum(
            _padded_source_bytes(value) for value in assignment
        ),
        file_read_seconds=source_profile.file_read_seconds,
        integrity_seconds=source_profile.integrity_seconds,
        deserialize_validate_seconds=(
            source_profile.deserialize_validate_seconds
        ),
        pageable_allocation_seconds=(
            source_profile.pageable_allocation_seconds
        ),
        batch_pack_seconds=source_profile.batch_pack_seconds,
        pageable_to_pinned_seconds=pin_seconds,
        h2d_seconds=h2d_seconds,
        elapsed_seconds=time.perf_counter() - started,
    )


def materialize_stage45_resident_source(
    plan: Stage45ResidentPlan,
    transforms: tuple[Stage4Transform, ...],
    require_capacity: bool = True,
) -> Stage45ResidentSource:
    method, target = _validate_transforms(
        transforms,
        plan.runtime_config,
    )
    if (
        method != plan.method
        or target != plan.target_version
        or len(transforms) != len(plan.assignments)
    ):
        raise ValueError("Stage 4.5 materialization inputs differ")
    preflight = stage45_resident_preflight(plan, transforms)
    if require_capacity and not preflight["passed"]:
        raise MemoryError("Stage 4.5 resident source capacity preflight failed")
    reader = LazyStage4SourceReader(
        plan.source_manifest_path,
        plan.workload_content_sha256,
    )
    if reader.manifest_file_sha256 != plan.source_manifest_sha256:
        raise ValueError("Stage 4.5 source changed after planning")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(transforms)) as pool:
        futures = [
            pool.submit(
                _materialize_assignment,
                index,
                reader,
                assignment,
                transform,
                plan.source_tier,
            )
            for index, (assignment, transform) in enumerate(
                zip(plan.assignments, transforms, strict=True)
            )
        ]
        values = tuple(value.result() for value in futures)
    assignments = tuple(value[0] for value in values)
    devices = tuple(value[1] for value in values)
    resident_source_bytes = sum(
        value.resident_source_bytes for value in devices
    )
    preload = Stage45PreloadMetrics(
        elapsed_seconds=time.perf_counter() - started,
        logical_source_bytes=sum(
            value.logical_source_bytes for value in devices
        ),
        physical_source_bytes=sum(
            value.physical_source_bytes for value in devices
        ),
        resident_source_bytes=resident_source_bytes,
        standing_host_bytes=(
            resident_source_bytes
            if plan.source_tier == "dram_resident"
            else 0
        ),
        standing_hbm_bytes=(
            resident_source_bytes
            if plan.source_tier == "hbm_resident"
            else 0
        ),
        devices=devices,
    )
    return Stage45ResidentSource(
        plan=plan,
        assignments=assignments,
        preload=preload,
    )


@dataclass
class _Difference:
    elements: int = 0
    mismatched: int = 0
    max_abs_error: float = 0.0
    finite: bool = True

    def update(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        atol: float,
        rtol: float,
    ) -> None:
        if actual.shape != expected.shape:
            raise ValueError("Stage 4.5 correctness shapes differ")
        for start in range(0, actual.shape[1], 256):
            actual_chunk = actual[:, start : start + 256]
            expected_chunk = expected[:, start : start + 256]
            self.finite = self.finite and bool(
                torch.isfinite(actual_chunk).all()
                and torch.isfinite(expected_chunk).all()
            )
            self.mismatched += int(
                torch.count_nonzero(
                    ~torch.isclose(
                        actual_chunk,
                        expected_chunk,
                        atol=atol,
                        rtol=rtol,
                    )
                )
            )
            if actual_chunk.numel():
                delta = (
                    actual_chunk.float() - expected_chunk.float()
                ).abs()
                self.max_abs_error = max(
                    self.max_abs_error,
                    float(delta.max()),
                )
            self.elements += actual_chunk.numel()

    def merge(self, other: _Difference) -> None:
        self.elements += other.elements
        self.mismatched += other.mismatched
        self.max_abs_error = max(self.max_abs_error, other.max_abs_error)
        self.finite = self.finite and other.finite


@dataclass(frozen=True)
class Stage45Correctness:
    finite: bool
    allclose: bool
    max_abs_error: float
    valid_element_count: int
    record_order_valid: bool
    lengths_offsets_valid: bool
    validation_seconds: float
    atol: float = 0.02
    rtol: float = 0.02

    def to_dict(self) -> dict[str, object]:
        return {
            "finite": self.finite,
            "allclose": self.allclose,
            "max_abs_error": self.max_abs_error,
            "valid_element_count": self.valid_element_count,
            "record_order_valid": self.record_order_valid,
            "lengths_offsets_valid": self.lengths_offsets_valid,
            "validation_seconds": self.validation_seconds,
            "atol": self.atol,
            "rtol": self.rtol,
            "reference_kind": (
                "same resident source tensor, transform, numeric path, "
                "and HBM target layout"
            ),
        }


@dataclass(frozen=True)
class Stage45DeviceMetrics:
    index: int
    record_count: int
    prefix_tokens: int
    standing_source_hbm_bytes: int
    standing_source_host_bytes: int
    transform_resident_bytes: int
    h2d_traffic_bytes: int
    physical_output_bytes: int
    baseline_hbm_bytes: int
    peak_hbm_bytes: int
    peak_incremental_hbm_bytes: int
    elapsed_seconds: float
    h2d_seconds: float
    compute_seconds: float
    target_allocation_seconds: float
    stage_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "record_count": self.record_count,
            "prefix_tokens": self.prefix_tokens,
            "standing_source_hbm_bytes": self.standing_source_hbm_bytes,
            "standing_source_host_bytes": self.standing_source_host_bytes,
            "transform_resident_bytes": self.transform_resident_bytes,
            "h2d_traffic_bytes": self.h2d_traffic_bytes,
            "physical_output_bytes": self.physical_output_bytes,
            "baseline_hbm_bytes": self.baseline_hbm_bytes,
            "peak_hbm_bytes": self.peak_hbm_bytes,
            "peak_incremental_hbm_bytes": self.peak_incremental_hbm_bytes,
            "elapsed_seconds": self.elapsed_seconds,
            "h2d_seconds": self.h2d_seconds,
            "compute_seconds": self.compute_seconds,
            "target_allocation_seconds": self.target_allocation_seconds,
            "stage_seconds": self.stage_seconds,
        }


@dataclass(frozen=True)
class Stage45JobReport:
    method: str
    source_tier: str
    runtime_config: Stage4RuntimeConfig
    source_manifest_sha256: str
    workload_content_sha256: str
    record_count: int
    prefix_tokens: int
    logical_source_bytes: int
    physical_source_bytes: int
    resident_source_bytes: int
    logical_output_bytes: int
    physical_output_bytes: int
    elapsed_seconds: float
    begin_seconds: float
    commit_seconds: float
    coordinator_seconds: float
    devices: tuple[Stage45DeviceMetrics, ...]
    manifest: KVVersionManifest
    correctness: Stage45Correctness | None
    protocol: str = STAGE45_RESIDENT_PROTOCOL

    @property
    def load_imbalance_ratio(self) -> float:
        weights = [
            value.standing_source_hbm_bytes
            + value.standing_source_host_bytes
            + value.physical_output_bytes
            for value in self.devices
        ]
        return max(weights) / max(statistics.mean(weights), 1)

    def timing_breakdown(self) -> dict[str, float]:
        return {
            "begin": self.begin_seconds,
            "h2d": max(value.h2d_seconds for value in self.devices),
            "compute": max(value.compute_seconds for value in self.devices),
            "target_allocation": max(
                value.target_allocation_seconds for value in self.devices
            ),
            "stage": max(value.stage_seconds for value in self.devices),
            "commit": self.commit_seconds,
            "coordinator": self.coordinator_seconds,
            "elapsed": self.elapsed_seconds,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "method": self.method,
            "source_tier": self.source_tier,
            "runtime_config": self.runtime_config.to_dict(),
            "source_manifest_sha256": self.source_manifest_sha256,
            "workload_content_sha256": self.workload_content_sha256,
            "record_count": self.record_count,
            "prefix_tokens": self.prefix_tokens,
            "logical_source_bytes": self.logical_source_bytes,
            "physical_source_bytes": self.physical_source_bytes,
            "resident_source_bytes": self.resident_source_bytes,
            "logical_output_bytes": self.logical_output_bytes,
            "physical_output_bytes": self.physical_output_bytes,
            "elapsed_seconds": self.elapsed_seconds,
            "timing_breakdown": self.timing_breakdown(),
            "load_imbalance_ratio": self.load_imbalance_ratio,
            "devices": [value.to_dict() for value in self.devices],
            "manifest": self.manifest.to_dict(),
            "correctness": (
                None
                if self.correctness is None
                else self.correctness.to_dict()
            ),
        }


@dataclass(frozen=True)
class Stage45JobResult:
    report: Stage45JobReport
    destination: HBMKVUpdateDestination


@dataclass
class _Pending:
    extent: Stage45ResidentExtent
    device_batch: Stage4SourceBatch
    output: JaggedMigratedKVBatch
    h2d_start: torch.cuda.Event | None
    h2d_end: torch.cuda.Event | None
    layout_ready: torch.cuda.Event
    compute_start: torch.cuda.Event
    compute_end: torch.cuda.Event
    target_allocation_seconds: float


@dataclass(frozen=True)
class _WorkerResult:
    metrics: Stage45DeviceMetrics


class Stage45ResidentEngine:
    def __init__(
        self,
        source: Stage45ResidentSource,
        transforms: tuple[Stage4Transform, ...],
    ) -> None:
        method, target = _validate_transforms(
            transforms,
            source.plan.runtime_config,
        )
        if (
            method != source.plan.method
            or target != source.plan.target_version
            or len(transforms) != len(source.assignments)
        ):
            raise ValueError("Stage 4.5 engine inputs differ")
        for assignment, transform in zip(
            source.assignments,
            transforms,
            strict=True,
        ):
            expected = (
                transform.device
                if source.plan.source_tier == "hbm_resident"
                else torch.device("cpu")
            )
            if any(value.batch.device != expected for value in assignment):
                raise ValueError("Stage 4.5 resident source is misplaced")
            if (
                source.plan.source_tier == "dram_resident"
                and any(not value.batch.is_pinned for value in assignment)
            ):
                raise ValueError("Stage 4.5 DRAM source must be pinned")
        self.source = source
        self.transforms = transforms
        self._pools = tuple(
            ThreadPoolExecutor(max_workers=1) for _ in transforms
        )
        self._streams = []
        for transform in transforms:
            with torch.cuda.device(transform.device):
                self._streams.append(
                    (
                        torch.cuda.Stream(device=transform.device),
                        torch.cuda.Stream(device=transform.device),
                    )
                )
        self._streams = tuple(self._streams)
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            for pool in self._pools:
                pool.shutdown(wait=True)
            self._closed = True

    def _enqueue(
        self,
        extent: Stage45ResidentExtent,
        transform: Stage4Transform,
        h2d_stream: torch.cuda.Stream,
        compute_stream: torch.cuda.Stream,
    ) -> _Pending:
        allocation_started = time.perf_counter()
        with torch.cuda.device(transform.device):
            output = _make_device_output(
                extent.spec,
                self.source.plan.num_layers,
                self.source.plan.kv_width,
                transform.device,
            )
            layout_ready = torch.cuda.Event()
            layout_ready.record(torch.cuda.current_stream(transform.device))
            allocation_seconds = time.perf_counter() - allocation_started
            h2d_start = None
            h2d_end = None
            if self.source.plan.source_tier == "dram_resident":
                with torch.cuda.stream(h2d_stream):
                    h2d_start = torch.cuda.Event(enable_timing=True)
                    h2d_end = torch.cuda.Event(enable_timing=True)
                    h2d_start.record(h2d_stream)
                    device_batch = extent.batch.to(
                        transform.device,
                        non_blocking=True,
                    )
                    h2d_end.record(h2d_stream)
            else:
                device_batch = extent.batch
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(layout_ready)
                if h2d_end is not None:
                    compute_stream.wait_event(h2d_end)
                compute_start = torch.cuda.Event(enable_timing=True)
                compute_end = torch.cuda.Event(enable_timing=True)
                compute_start.record(compute_stream)
                transform.execute(device_batch, output)
                compute_end.record(compute_stream)
        return _Pending(
            extent=extent,
            device_batch=device_batch,
            output=output,
            h2d_start=h2d_start,
            h2d_end=h2d_end,
            layout_ready=layout_ready,
            compute_start=compute_start,
            compute_end=compute_end,
            target_allocation_seconds=allocation_seconds,
        )

    def _worker(
        self,
        index: int,
        assignment: tuple[Stage45ResidentExtent, ...],
        transform: Stage4Transform,
        transaction,
    ) -> _WorkerResult:
        started = time.perf_counter()
        pending: deque[_Pending] = deque()
        h2d_seconds = 0.0
        compute_seconds = 0.0
        allocation_seconds = 0.0
        stage_seconds = 0.0
        output_bytes = 0
        with torch.cuda.device(transform.device):
            torch.cuda.synchronize(transform.device)
            baseline = torch.cuda.memory_allocated(transform.device)
            torch.cuda.reset_peak_memory_stats(transform.device)
            h2d_stream, compute_stream = self._streams[index]

            def finalize(value: _Pending) -> None:
                nonlocal h2d_seconds
                nonlocal compute_seconds
                nonlocal allocation_seconds
                nonlocal stage_seconds
                nonlocal output_bytes
                value.compute_end.synchronize()
                if value.h2d_start is not None and value.h2d_end is not None:
                    h2d_seconds += (
                        value.h2d_start.elapsed_time(value.h2d_end) / 1000.0
                    )
                compute_seconds += (
                    value.compute_start.elapsed_time(value.compute_end)
                    / 1000.0
                )
                allocation_seconds += value.target_allocation_seconds
                stage_started = time.perf_counter()
                transaction.stage(value.extent.spec.extent_id, value.output)
                stage_seconds += time.perf_counter() - stage_started
                output_bytes += value.output.nbytes

            for extent in assignment:
                pending.append(
                    self._enqueue(
                        extent,
                        transform,
                        h2d_stream,
                        compute_stream,
                    )
                )
                if (
                    len(pending)
                    >= self.source.plan.runtime_config.max_inflight
                ):
                    finalize(pending.popleft())
            while pending:
                finalize(pending.popleft())
            torch.cuda.synchronize(transform.device)
            peak = torch.cuda.max_memory_allocated(transform.device)
        source_bytes = sum(value.batch.nbytes for value in assignment)
        return _WorkerResult(
            metrics=Stage45DeviceMetrics(
                index=index,
                record_count=sum(
                    len(value.spec.records) for value in assignment
                ),
                prefix_tokens=sum(
                    value.spec.token_count for value in assignment
                ),
                standing_source_hbm_bytes=(
                    source_bytes
                    if self.source.plan.source_tier == "hbm_resident"
                    else 0
                ),
                standing_source_host_bytes=(
                    source_bytes
                    if self.source.plan.source_tier == "dram_resident"
                    else 0
                ),
                transform_resident_bytes=transform.resident_bytes,
                h2d_traffic_bytes=(
                    source_bytes
                    if self.source.plan.source_tier == "dram_resident"
                    else 0
                ),
                physical_output_bytes=output_bytes,
                baseline_hbm_bytes=baseline,
                peak_hbm_bytes=peak,
                peak_incremental_hbm_bytes=max(peak - baseline, 0),
                elapsed_seconds=time.perf_counter() - started,
                h2d_seconds=h2d_seconds,
                compute_seconds=compute_seconds,
                target_allocation_seconds=allocation_seconds,
                stage_seconds=stage_seconds,
            )
        )

    def _validate(
        self,
        destination: HBMKVUpdateDestination,
        atol: float,
        rtol: float,
    ) -> Stage45Correctness:
        started = time.perf_counter()
        aggregate = _Difference()
        record_order_valid = True
        lengths_offsets_valid = True
        for assignment, transform in zip(
            self.source.assignments,
            self.transforms,
            strict=True,
        ):
            for extent in assignment:
                with torch.cuda.device(transform.device):
                    source_batch = (
                        extent.batch
                        if extent.batch.device.type == "cuda"
                        else extent.batch.to(transform.device)
                    )
                    oracle = _make_device_output(
                        extent.spec,
                        self.source.plan.num_layers,
                        self.source.plan.kv_width,
                        transform.device,
                    )
                    transform.execute(source_batch, oracle)
                    torch.cuda.synchronize(transform.device)
                    actual = destination.load_extent(
                        self.source.plan.target_version,
                        extent.spec.extent_id,
                    )
                    aggregate.update(actual.k, oracle.k, atol, rtol)
                    aggregate.update(actual.v, oracle.v, atol, rtol)
                    record_order_valid = (
                        record_order_valid
                        and actual.record_ids == extent.spec.record_ids
                    )
                    lengths = torch.tensor(
                        [
                            value.prefix_tokens
                            for value in extent.spec.records
                        ],
                        dtype=torch.long,
                        device=actual.lengths.device,
                    )
                    offsets = torch.cat(
                        (
                            torch.zeros(
                                1,
                                dtype=torch.long,
                                device=actual.offsets.device,
                            ),
                            lengths.cumsum(0),
                        )
                    )
                    lengths_offsets_valid = (
                        lengths_offsets_valid
                        and torch.equal(actual.lengths, lengths)
                        and torch.equal(actual.offsets, offsets)
                        and actual.token_count == extent.spec.token_count
                    )
        return Stage45Correctness(
            finite=aggregate.finite,
            allclose=aggregate.mismatched == 0,
            max_abs_error=aggregate.max_abs_error,
            valid_element_count=aggregate.elements,
            record_order_valid=record_order_valid,
            lengths_offsets_valid=lengths_offsets_valid,
            validation_seconds=time.perf_counter() - started,
            atol=atol,
            rtol=rtol,
        )

    def run(
        self,
        validate: bool = False,
        job_id: str | None = None,
        atol: float = 0.02,
        rtol: float = 0.02,
    ) -> Stage45JobResult:
        if self._closed:
            raise RuntimeError("Stage 4.5 resident engine is closed")
        if atol != 0.02 or rtol != 0.02:
            raise ValueError("Stage 4.5 transport tolerances are fixed")
        for transform in self.transforms:
            torch.cuda.synchronize(transform.device)
        started = time.perf_counter()
        destination = HBMKVUpdateDestination(
            tuple(value.device for value in self.transforms),
            destination_id=(
                f"stage45-hbm-{self.source.plan.method}-"
                f"{len(self.transforms)}gpu"
            ),
        )
        begin_started = time.perf_counter()
        transaction = destination.begin(
            job_id=job_id
            or (
                f"stage45-{self.source.plan.method}-"
                f"{self.source.plan.source_tier}-"
                f"{len(self.transforms)}gpu"
            ),
            target_version=self.source.plan.target_version,
            expected_record_ids=self.source.plan.record_ids,
        )
        begin_seconds = time.perf_counter() - begin_started
        try:
            futures = [
                pool.submit(
                    self._worker,
                    index,
                    assignment,
                    transform,
                    transaction,
                )
                for index, (pool, assignment, transform) in enumerate(
                    zip(
                        self._pools,
                        self.source.assignments,
                        self.transforms,
                        strict=True,
                    )
                )
            ]
            workers = tuple(value.result() for value in futures)
            commit_started = time.perf_counter()
            manifest = transaction.commit()
            commit_seconds = time.perf_counter() - commit_started
        except BaseException:
            transaction.abort()
            raise
        elapsed = time.perf_counter() - started
        devices = tuple(value.metrics for value in workers)
        covered = manifest.record_ids
        if (
            len(covered) != len(self.source.plan.record_ids)
            or len(set(covered)) != len(covered)
            or set(covered) != set(self.source.plan.record_ids)
        ):
            raise RuntimeError("Stage 4.5 committed coverage is incomplete")
        worker_elapsed = max(value.elapsed_seconds for value in devices)
        coordinator_seconds = max(
            elapsed
            - begin_seconds
            - worker_elapsed
            - commit_seconds,
            0.0,
        )
        correctness = (
            self._validate(destination, atol, rtol)
            if validate
            else None
        )
        expected_elements = (
            self.source.plan.logical_output_bytes
            // torch.tensor([], dtype=torch.float16).element_size()
        )
        if (
            correctness is not None
            and correctness.valid_element_count != expected_elements
        ):
            raise RuntimeError(
                "Stage 4.5 correctness did not cover every output element"
            )
        report = Stage45JobReport(
            method=self.source.plan.method,
            source_tier=self.source.plan.source_tier,
            runtime_config=self.source.plan.runtime_config,
            source_manifest_sha256=self.source.plan.source_manifest_sha256,
            workload_content_sha256=(
                self.source.plan.workload_content_sha256
            ),
            record_count=self.source.plan.record_count,
            prefix_tokens=self.source.plan.prefix_tokens,
            logical_source_bytes=self.source.plan.logical_source_bytes,
            physical_source_bytes=self.source.plan.physical_source_bytes,
            resident_source_bytes=self.source.plan.resident_source_bytes,
            logical_output_bytes=self.source.plan.logical_output_bytes,
            physical_output_bytes=sum(
                value.physical_output_bytes for value in devices
            ),
            elapsed_seconds=elapsed,
            begin_seconds=begin_seconds,
            commit_seconds=commit_seconds,
            coordinator_seconds=coordinator_seconds,
            devices=devices,
            manifest=manifest,
            correctness=correctness,
        )
        if report.physical_output_bytes != self.source.plan.physical_output_bytes:
            raise RuntimeError("Stage 4.5 target byte count differs from plan")
        return Stage45JobResult(report=report, destination=destination)
