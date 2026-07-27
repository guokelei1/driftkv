from __future__ import annotations

import math
import os
import statistics
import time
from collections import deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from ..models import HSTU, HSTUKVCache
from .capsule import MigrationCapsuleBatch
from .cohort_jagged import JaggedMigratedKVBatch
from .destination import (
    DRAMKVUpdateDestination,
    HBMKVUpdateDestination,
    KVPublicationTransaction,
    KVUpdateDestination,
    KVVersionManifest,
)
from .operator import MigrationOperator
from .program import MigrationProgram
from .selective import (
    ResidualHiddenSuffixState,
    SelectiveContiguousState,
    migrate_prefix_residual_from_hidden_suffix,
    migrate_selective_contiguous_cache,
)
from .stage4_source import (
    LazyStage4SourceReader,
    SourceReadMetrics,
    Stage4ExtentSpec,
    Stage4SourceBatch,
    build_stage4_extents,
    place_stage4_extents_lpt,
)

STAGE4_CORE_ENGINE_PROTOCOL = "cohortkv_stage4_core_engine_v1"
STAGE4_METHODS = (
    "compiled",
    "selective_contiguous",
    "residual_p",
    "exact",
    "no_transform",
)
STAGE4_DESTINATIONS = ("hbm", "dram")


def _model_device(model: HSTU) -> torch.device:
    devices = {
        value.device
        for value in (*model.parameters(), *model.buffers())
    }
    if len(devices) != 1:
        raise ValueError("Stage 4 model must reside on one device")
    return next(iter(devices))


def _model_nbytes(model: HSTU) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in (*model.parameters(), *model.buffers())
    )


def _cache_to_extent(
    cache: HSTUKVCache,
    lengths: torch.Tensor,
    destination: JaggedMigratedKVBatch,
) -> None:
    positions = torch.arange(cache.seq_len, device=lengths.device)
    valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
    destination.k.copy_(cache.k[:, valid].to(torch.float16))
    destination.v.copy_(cache.v[:, valid].to(torch.float16))


def _require_raw(batch: Stage4SourceBatch) -> None:
    if (
        batch.item_ids is None
        or batch.behaviors is None
        or batch.time_deltas is None
    ):
        raise ValueError("Stage 4 transform requires raw history")


class Stage4Transform(Protocol):
    method: str
    target_version: str
    device: torch.device

    @property
    def resident_bytes(self) -> int: ...

    def source_representations(self, source_version: str) -> tuple[str, ...]: ...

    def execute(
        self,
        batch: Stage4SourceBatch,
        destination: JaggedMigratedKVBatch,
    ) -> None: ...


class CompiledStage4Transform:
    method = "compiled"

    def __init__(
        self,
        programs: Mapping[str, MigrationProgram],
        operator: MigrationOperator,
        device: torch.device | str,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("compiled Stage 4 transform requires CUDA")
        if not programs:
            raise ValueError("compiled Stage 4 transform requires programs")
        self.operator = operator
        self.runtime_variant = (
            "fused_fp16"
            if operator.name.startswith("fused")
            else "packed_fp16"
        )
        self.programs = {
            source: operator.prepare_program(program, self.device)
            for source, program in programs.items()
        }
        targets = {value.target_version for value in self.programs.values()}
        if len(targets) != 1:
            raise ValueError("compiled Stage 4 program targets differ")
        self.target_version = next(iter(targets))

    @property
    def resident_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for program in self.programs.values()
            for value in (program.adapter.weights, program.adapter.biases)
        )

    def source_representations(self, source_version: str) -> tuple[str, ...]:
        if source_version not in self.programs:
            raise ValueError("compiled Stage 4 source has no program")
        return ("normalized_capsule_fp16",)

    @torch.no_grad()
    def execute(
        self,
        batch: Stage4SourceBatch,
        destination: JaggedMigratedKVBatch,
    ) -> None:
        if batch.normed is None:
            raise ValueError("compiled Stage 4 transform requires a capsule")
        program = self.programs[batch.migration_anchor_version]
        capsule = MigrationCapsuleBatch(
            record_ids=batch.record_ids,
            migration_anchor_version=batch.migration_anchor_version,
            normed=batch.normed,
            lengths=batch.lengths,
        )
        self.operator.execute_into(program, capsule, destination)


class ExactStage4Transform:
    method = "exact"

    def __init__(
        self,
        model: HSTU,
        target_version: str,
        execution_dtype: torch.dtype | None,
    ) -> None:
        if execution_dtype not in {None, torch.bfloat16}:
            raise ValueError("exact Stage 4 dtype must be BF16 or FP32")
        self.model = model.eval()
        self.device = _model_device(model)
        if self.device.type != "cuda":
            raise ValueError("exact Stage 4 transform requires CUDA")
        self.target_version = target_version
        self.execution_dtype = execution_dtype
        self.runtime_variant = (
            "bfloat16" if execution_dtype == torch.bfloat16 else "float32"
        )

    @property
    def resident_bytes(self) -> int:
        return _model_nbytes(self.model)

    def source_representations(self, source_version: str) -> tuple[str, ...]:
        return ("raw_history",)

    def _compute(self, batch: Stage4SourceBatch) -> HSTUKVCache:
        _require_raw(batch)
        if self.execution_dtype is None:
            return self.model.compute_kv(
                batch.item_ids,
                batch.behaviors,
                batch.time_deltas,
                lengths=batch.lengths,
            )
        with torch.autocast(device_type="cuda", dtype=self.execution_dtype):
            return self.model.compute_kv(
                batch.item_ids,
                batch.behaviors,
                batch.time_deltas,
                lengths=batch.lengths,
            )

    @torch.no_grad()
    def execute(
        self,
        batch: Stage4SourceBatch,
        destination: JaggedMigratedKVBatch,
    ) -> None:
        _cache_to_extent(self._compute(batch), batch.lengths, destination)


class SelectiveStage4Transform:
    method = "selective_contiguous"

    def __init__(
        self,
        model: HSTU,
        target_version: str,
        start_layer: int,
        end_layer: int,
    ) -> None:
        self.model = model.eval()
        self.device = _model_device(model)
        if self.device.type != "cuda":
            raise ValueError("selective Stage 4 transform requires CUDA")
        if not 0 <= start_layer <= end_layer < len(model.blocks):
            raise ValueError("selective Stage 4 interval is invalid")
        self.target_version = target_version
        self.start_layer = start_layer
        self.end_layer = end_layer

    @property
    def resident_bytes(self) -> int:
        return _model_nbytes(self.model)

    def source_representations(self, source_version: str) -> tuple[str, ...]:
        if self.start_layer != 0:
            raise ValueError("Stage 4 source contract lacks transition hidden state")
        return ("old_kv_fp16", "raw_history")

    @torch.no_grad()
    def execute(
        self,
        batch: Stage4SourceBatch,
        destination: JaggedMigratedKVBatch,
    ) -> None:
        _require_raw(batch)
        if batch.old_k is None or batch.old_v is None:
            raise ValueError("selective Stage 4 transform requires old K/V")
        state = SelectiveContiguousState(
            source_kv=HSTUKVCache(
                k=batch.old_k,
                v=batch.old_v,
                seq_len=batch.sequence_width,
            ),
            transition_hidden=None,
            lengths=batch.lengths,
            start_layer=self.start_layer,
        )
        cache = migrate_selective_contiguous_cache(
            self.model,
            state,
            batch.item_ids,
            batch.behaviors,
            batch.time_deltas,
            self.end_layer,
        )
        _cache_to_extent(cache, batch.lengths, destination)


class ResidualPStage4Transform:
    method = "residual_p"

    def __init__(
        self,
        model: HSTU,
        target_version: str,
        start_layer: int,
        residual_sources: tuple[str, ...],
    ) -> None:
        self.model = model.eval()
        self.device = _model_device(model)
        if self.device.type != "cuda":
            raise ValueError("residual Stage 4 transform requires CUDA")
        if not 1 <= start_layer < len(model.blocks):
            raise ValueError("residual Stage 4 start layer is invalid")
        if not residual_sources or len(set(residual_sources)) != len(residual_sources):
            raise ValueError("residual Stage 4 source scope is invalid")
        self.target_version = target_version
        self.start_layer = start_layer
        self.residual_sources = frozenset(residual_sources)

    @property
    def resident_bytes(self) -> int:
        return _model_nbytes(self.model)

    def source_representations(self, source_version: str) -> tuple[str, ...]:
        if source_version in self.residual_sources:
            return ("raw_history", "residual_hidden_suffix_bf16")
        return ("raw_history",)

    @torch.no_grad()
    def execute(
        self,
        batch: Stage4SourceBatch,
        destination: JaggedMigratedKVBatch,
    ) -> None:
        _require_raw(batch)
        if batch.migration_anchor_version not in self.residual_sources:
            cache = self.model.compute_kv(
                batch.item_ids,
                batch.behaviors,
                batch.time_deltas,
                lengths=batch.lengths,
            )
        else:
            if (
                batch.residual_hidden_states is None
                or batch.residual_start_layer != self.start_layer
            ):
                raise ValueError("residual Stage 4 hidden suffix is unavailable")
            state = ResidualHiddenSuffixState(
                hidden_states=tuple(batch.residual_hidden_states.unbind(0)),
                lengths=batch.lengths,
                start_layer=self.start_layer,
                num_layers=len(self.model.blocks),
            )
            cache = migrate_prefix_residual_from_hidden_suffix(
                self.model,
                state,
                batch.item_ids,
                batch.behaviors,
                batch.time_deltas,
            )
        _cache_to_extent(cache, batch.lengths, destination)


class NoTransformStage4Transform:
    method = "no_transform"

    def __init__(
        self,
        device: torch.device | str,
        target_version: str,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("no-transform Stage 4 path requires CUDA")
        self.target_version = target_version

    @property
    def resident_bytes(self) -> int:
        return 0

    def source_representations(self, source_version: str) -> tuple[str, ...]:
        return ("old_kv_fp16",)

    @torch.no_grad()
    def execute(
        self,
        batch: Stage4SourceBatch,
        destination: JaggedMigratedKVBatch,
    ) -> None:
        if batch.old_k is None or batch.old_v is None:
            raise ValueError("no-transform Stage 4 path requires old K/V")
        _cache_to_extent(
            HSTUKVCache(
                k=batch.old_k,
                v=batch.old_v,
                seq_len=batch.sequence_width,
            ),
            batch.lengths,
            destination,
        )


@dataclass(frozen=True)
class Stage4RuntimeConfig:
    batch_size: int
    length_bucket_width: int
    max_inflight: int
    compiled_operator: str | None = None
    exact_compute: str | None = None

    def __post_init__(self) -> None:
        if (
            self.batch_size not in {1, 2, 4}
            or self.length_bucket_width not in {16, 32, 64}
            or self.max_inflight not in {2, 3, 4}
        ):
            raise ValueError("Stage 4 runtime configuration is outside the frozen grid")
        if self.compiled_operator not in {None, "packed_fp16", "fused_fp16"}:
            raise ValueError("Stage 4 compiled operator is invalid")
        if self.exact_compute not in {None, "bfloat16", "float32"}:
            raise ValueError("Stage 4 exact compute dtype is invalid")

    def to_dict(self) -> dict[str, object]:
        value = {
            "batch_size": self.batch_size,
            "length_bucket_width": self.length_bucket_width,
            "max_inflight": self.max_inflight,
        }
        if self.compiled_operator is not None:
            value["compiled_operator"] = self.compiled_operator
        if self.exact_compute is not None:
            value["exact_compute"] = self.exact_compute
        return value


@dataclass
class _DifferenceAccumulator:
    elements: int = 0
    mismatched: int = 0
    max_abs_error: float = 0.0
    finite: bool = True

    def update(
        self,
        actual: torch.Tensor,
        reference: torch.Tensor,
        atol: float,
        rtol: float,
    ) -> None:
        if actual.shape != reference.shape:
            raise ValueError("correctness tensors have different shapes")
        for start in range(0, actual.shape[1], 256):
            actual_chunk = actual[:, start : start + 256]
            reference_chunk = reference[:, start : start + 256]
            self.finite = self.finite and bool(
                torch.isfinite(actual_chunk).all()
                and torch.isfinite(reference_chunk).all()
            )
            self.mismatched += int(
                torch.count_nonzero(
                    ~torch.isclose(
                        actual_chunk,
                        reference_chunk,
                        atol=atol,
                        rtol=rtol,
                    )
                )
            )
            if actual_chunk.numel():
                delta = (
                    actual_chunk.float() - reference_chunk.float()
                ).abs()
                self.max_abs_error = max(
                    self.max_abs_error,
                    float(delta.max()),
                )
            self.elements += actual_chunk.numel()

    def merge(self, other: _DifferenceAccumulator) -> None:
        self.elements += other.elements
        self.mismatched += other.mismatched
        self.max_abs_error = max(self.max_abs_error, other.max_abs_error)
        self.finite = self.finite and other.finite


@dataclass(frozen=True)
class Stage4Correctness:
    finite: bool
    allclose: bool
    max_abs_error: float
    valid_element_count: int
    record_order_valid: bool
    lengths_offsets_valid: bool
    atol: float = 0.02
    rtol: float = 0.02
    reference_kind: str = (
        "same selected method and numeric path resident on the same "
        "serialized source representation"
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "finite": self.finite,
            "allclose": self.allclose,
            "reference_kind": self.reference_kind,
            "atol": self.atol,
            "rtol": self.rtol,
            "max_abs_error": self.max_abs_error,
            "record_order_valid": self.record_order_valid,
            "lengths_offsets_valid": self.lengths_offsets_valid,
            "valid_element_count": self.valid_element_count,
        }


@dataclass(frozen=True)
class Stage4DeviceMetrics:
    index: int
    record_count: int
    prefix_tokens: int
    logical_input_bytes: int
    physical_input_bytes: int
    logical_output_bytes: int
    physical_output_bytes: int
    elapsed_seconds: float
    source_read_seconds: float
    h2d_seconds: float
    compute_seconds: float
    d2h_seconds: float
    stage_seconds: float
    target_allocation_seconds: float
    peak_hbm_bytes: int
    peak_source_resident_bytes: int
    peak_staging_bytes: int
    peak_publication_queue_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "record_count": self.record_count,
            "prefix_tokens": self.prefix_tokens,
            "logical_input_bytes": self.logical_input_bytes,
            "physical_input_bytes": self.physical_input_bytes,
            "logical_output_bytes": self.logical_output_bytes,
            "physical_output_bytes": self.physical_output_bytes,
            "elapsed_seconds": self.elapsed_seconds,
            "source_read_seconds": self.source_read_seconds,
            "h2d_seconds": self.h2d_seconds,
            "compute_seconds": self.compute_seconds,
            "d2h_seconds": self.d2h_seconds,
            "stage_seconds": self.stage_seconds,
            "target_allocation_seconds": self.target_allocation_seconds,
            "peak_hbm_bytes": self.peak_hbm_bytes,
            "peak_source_resident_bytes": self.peak_source_resident_bytes,
            "peak_staging_bytes": self.peak_staging_bytes,
            "peak_publication_queue_bytes": self.peak_publication_queue_bytes,
        }


@dataclass(frozen=True)
class Stage4JobReport:
    method: str
    destination: str
    runtime_config: Stage4RuntimeConfig
    source_manifest_sha256: str
    workload_content_sha256: str
    record_count: int
    prefix_tokens: int
    logical_input_bytes: int
    physical_input_bytes: int
    logical_output_bytes: int
    physical_output_bytes: int
    elapsed_seconds: float
    source_manifest_scan_seconds: float
    commit_seconds: float
    coordinator_seconds: float
    devices: tuple[Stage4DeviceMetrics, ...]
    manifest: KVVersionManifest
    correctness: Stage4Correctness | None
    protocol: str = STAGE4_CORE_ENGINE_PROTOCOL

    @property
    def load_imbalance_ratio(self) -> float:
        assigned = [
            value.logical_input_bytes + value.logical_output_bytes
            for value in self.devices
        ]
        return max(assigned) / max(statistics.mean(assigned), 1)

    @property
    def peak_hbm_bytes(self) -> int:
        return sum(value.peak_hbm_bytes for value in self.devices)

    @property
    def peak_source_resident_bytes(self) -> int:
        return sum(value.peak_source_resident_bytes for value in self.devices)

    @property
    def peak_staging_bytes(self) -> int:
        return sum(value.peak_staging_bytes for value in self.devices)

    @property
    def peak_publication_queue_bytes(self) -> int:
        return sum(value.peak_publication_queue_bytes for value in self.devices)

    @property
    def peak_host_bytes(self) -> int:
        retained = self.physical_output_bytes if self.destination == "dram" else 0
        return retained + self.peak_source_resident_bytes

    def timing_breakdown(self) -> dict[str, float]:
        return {
            "source_read": self.source_manifest_scan_seconds
            + max(value.source_read_seconds for value in self.devices),
            "h2d": max(value.h2d_seconds for value in self.devices),
            "compute": max(value.compute_seconds for value in self.devices),
            "d2h": max(value.d2h_seconds for value in self.devices),
            "stage": max(value.stage_seconds for value in self.devices),
            "commit": self.commit_seconds,
            "target_allocation": max(
                value.target_allocation_seconds for value in self.devices
            ),
            "coordinator": self.coordinator_seconds,
            "elapsed": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class Stage4JobResult:
    report: Stage4JobReport
    destination: KVUpdateDestination


@dataclass
class _PendingExtent:
    spec: Stage4ExtentSpec
    host_batch: Stage4SourceBatch
    device_batch: Stage4SourceBatch
    device_output: JaggedMigratedKVBatch
    published_output: JaggedMigratedKVBatch
    h2d_start: torch.cuda.Event
    h2d_end: torch.cuda.Event
    compute_start: torch.cuda.Event
    compute_end: torch.cuda.Event
    d2h_start: torch.cuda.Event | None
    d2h_end: torch.cuda.Event | None
    source_metrics: SourceReadMetrics
    source_read_seconds: float
    target_allocation_seconds: float


@dataclass
class _WorkerResult:
    metrics: Stage4DeviceMetrics
    correctness: _DifferenceAccumulator
    record_order_valid: bool
    lengths_offsets_valid: bool


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
            lengths.long().cumsum(0),
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


def _make_host_output(
    spec: Stage4ExtentSpec,
    num_layers: int,
    kv_width: int,
) -> JaggedMigratedKVBatch:
    lengths = torch.tensor(
        [value.prefix_tokens for value in spec.records],
        dtype=torch.long,
        pin_memory=True,
    )
    offsets = torch.empty(
        len(spec.records) + 1,
        dtype=torch.long,
        pin_memory=True,
    )
    offsets[0] = 0
    torch.cumsum(lengths, 0, out=offsets[1:])
    shape = (num_layers, spec.token_count, kv_width)
    return JaggedMigratedKVBatch(
        record_ids=spec.record_ids,
        migration_anchor_version=spec.migration_anchor_version,
        served_kv_target=spec.served_kv_target,
        k=torch.empty(shape, dtype=torch.float16, pin_memory=True),
        v=torch.empty(shape, dtype=torch.float16, pin_memory=True),
        lengths=lengths,
        offsets=offsets,
    )


def _validate_extent_metadata(
    spec: Stage4ExtentSpec,
    output: JaggedMigratedKVBatch,
) -> tuple[bool, bool]:
    record_order = output.record_ids == spec.record_ids
    expected_lengths = torch.tensor(
        [value.prefix_tokens for value in spec.records],
        dtype=torch.long,
        device=output.lengths.device,
    )
    expected_offsets = torch.cat(
        (
            torch.zeros(
                1,
                dtype=torch.long,
                device=output.offsets.device,
            ),
            expected_lengths.cumsum(0),
        )
    )
    metadata = (
        torch.equal(output.lengths, expected_lengths)
        and torch.equal(output.offsets, expected_offsets)
        and output.token_count == spec.token_count
    )
    return record_order, metadata


class Stage4CoreEngine:
    def __init__(
        self,
        source_manifest_path: Path | str,
        transforms: tuple[Stage4Transform, ...],
        destination: str,
        runtime_config: Stage4RuntimeConfig,
        expected_source_manifest_sha256: str | None = None,
        expected_workload_content_sha256: str | None = None,
    ) -> None:
        if not transforms:
            raise ValueError("Stage 4 core engine requires device transforms")
        if destination not in STAGE4_DESTINATIONS:
            raise ValueError("unsupported Stage 4 destination")
        methods = {value.method for value in transforms}
        targets = {value.target_version for value in transforms}
        devices = [value.device for value in transforms]
        if len(methods) != 1 or next(iter(methods)) not in STAGE4_METHODS:
            raise ValueError("Stage 4 transforms must share one supported method")
        if len(targets) != 1:
            raise ValueError("Stage 4 transforms must share one target")
        if (
            any(value.type != "cuda" for value in devices)
            or len(set(devices)) != len(devices)
        ):
            raise ValueError("Stage 4 transforms require unique CUDA devices")
        if len(transforms) not in {1, 2, 4}:
            raise ValueError("Stage 4 GPU count is outside the frozen matrix")
        method = next(iter(methods))
        if method == "compiled" and runtime_config.compiled_operator is None:
            raise ValueError("compiled Stage 4 config must name its operator")
        if method == "exact" and runtime_config.exact_compute is None:
            raise ValueError("exact Stage 4 config must name its compute dtype")
        if method == "compiled" and any(
            value.runtime_variant != runtime_config.compiled_operator
            for value in transforms
        ):
            raise ValueError("compiled Stage 4 runtime and transform variants differ")
        if method == "exact" and any(
            value.runtime_variant != runtime_config.exact_compute
            for value in transforms
        ):
            raise ValueError("exact Stage 4 runtime and transform variants differ")
        self.source_manifest_path = Path(source_manifest_path)
        self.transforms = transforms
        self.destination_kind = destination
        self.runtime_config = runtime_config
        self.expected_source_manifest_sha256 = expected_source_manifest_sha256
        self.expected_workload_content_sha256 = expected_workload_content_sha256
        self.method = method
        self.target_version = next(iter(targets))

    def _destination(self) -> KVUpdateDestination:
        devices = tuple(value.device for value in self.transforms)
        if self.destination_kind == "hbm":
            return HBMKVUpdateDestination(
                devices,
                destination_id=f"stage4-hbm-{len(devices)}gpu",
            )
        return DRAMKVUpdateDestination(
            destination_id=f"stage4-dram-{len(devices)}gpu",
            require_pinned=True,
        )

    def _representations(
        self,
        source_versions: set[str],
    ) -> dict[str, tuple[str, ...]]:
        values = {}
        for source in source_versions:
            expected = self.transforms[0].source_representations(source)
            if any(
                transform.source_representations(source) != expected
                for transform in self.transforms[1:]
            ):
                raise ValueError("Stage 4 device representation contracts differ")
            values[source] = expected
        return values

    def _enqueue(
        self,
        reader: LazyStage4SourceReader,
        spec: Stage4ExtentSpec,
        transform: Stage4Transform,
        h2d_stream: torch.cuda.Stream,
        compute_stream: torch.cuda.Stream,
        d2h_stream: torch.cuda.Stream,
    ) -> _PendingExtent:
        source_started = time.perf_counter()
        host_batch, source_metrics = reader.read_extent(spec, pin_memory=True)
        source_elapsed = time.perf_counter() - source_started
        source_metrics = SourceReadMetrics(
            physical_bytes=source_metrics.physical_bytes,
            logical_bytes=source_metrics.logical_bytes,
            peak_source_resident_bytes=source_metrics.peak_source_resident_bytes,
        )
        allocation_started = time.perf_counter()
        with torch.cuda.device(transform.device):
            with torch.cuda.stream(h2d_stream):
                h2d_start = torch.cuda.Event(enable_timing=True)
                h2d_end = torch.cuda.Event(enable_timing=True)
                h2d_start.record(h2d_stream)
                device_batch = host_batch.to(
                    transform.device,
                    non_blocking=True,
                )
                h2d_end.record(h2d_stream)
            device_output = _make_device_output(
                spec,
                reader.manifest.num_layers,
                reader.manifest.kv_width,
                transform.device,
            )
            if self.destination_kind == "dram":
                published_output = _make_host_output(
                    spec,
                    reader.manifest.num_layers,
                    reader.manifest.kv_width,
                )
            else:
                published_output = device_output
            allocation_elapsed = time.perf_counter() - allocation_started
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(h2d_end)
                compute_start = torch.cuda.Event(enable_timing=True)
                compute_end = torch.cuda.Event(enable_timing=True)
                compute_start.record(compute_stream)
                transform.execute(device_batch, device_output)
                compute_end.record(compute_stream)
            d2h_start = None
            d2h_end = None
            if self.destination_kind == "dram":
                with torch.cuda.stream(d2h_stream):
                    d2h_stream.wait_event(compute_end)
                    d2h_start = torch.cuda.Event(enable_timing=True)
                    d2h_end = torch.cuda.Event(enable_timing=True)
                    d2h_start.record(d2h_stream)
                    published_output.k.copy_(
                        device_output.k,
                        non_blocking=True,
                    )
                    published_output.v.copy_(
                        device_output.v,
                        non_blocking=True,
                    )
                    d2h_end.record(d2h_stream)
        return _PendingExtent(
            spec=spec,
            host_batch=host_batch,
            device_batch=device_batch,
            device_output=device_output,
            published_output=published_output,
            h2d_start=h2d_start,
            h2d_end=h2d_end,
            compute_start=compute_start,
            compute_end=compute_end,
            d2h_start=d2h_start,
            d2h_end=d2h_end,
            source_metrics=source_metrics,
            source_read_seconds=source_elapsed,
            target_allocation_seconds=allocation_elapsed,
        )

    def _validate_pending(
        self,
        pending: _PendingExtent,
        transform: Stage4Transform,
        atol: float,
        rtol: float,
    ) -> tuple[_DifferenceAccumulator, bool, bool]:
        with torch.cuda.device(transform.device):
            oracle = _make_device_output(
                pending.spec,
                pending.device_output.k.shape[0],
                pending.device_output.k.shape[2],
                transform.device,
            )
            transform.execute(pending.device_batch, oracle)
            torch.cuda.synchronize(transform.device)
            difference = _DifferenceAccumulator()
            if self.destination_kind == "hbm":
                actual_k = pending.published_output.k
                actual_v = pending.published_output.v
                reference_k = oracle.k
                reference_v = oracle.v
            else:
                actual_k = pending.published_output.k
                actual_v = pending.published_output.v
                reference_k = oracle.k.cpu()
                reference_v = oracle.v.cpu()
            difference.update(actual_k, reference_k, atol, rtol)
            difference.update(actual_v, reference_v, atol, rtol)
            record_order, metadata = _validate_extent_metadata(
                pending.spec,
                pending.published_output,
            )
            return difference, record_order, metadata

    def _worker(
        self,
        index: int,
        reader: LazyStage4SourceReader,
        assignment: tuple[Stage4ExtentSpec, ...],
        transform: Stage4Transform,
        transaction: KVPublicationTransaction,
        validate: bool,
        atol: float,
        rtol: float,
    ) -> _WorkerResult:
        started = time.perf_counter()
        device = transform.device
        pending: deque[_PendingExtent] = deque()
        source_seconds = 0.0
        h2d_seconds = 0.0
        compute_seconds = 0.0
        d2h_seconds = 0.0
        stage_seconds = 0.0
        allocation_seconds = 0.0
        physical_output = 0
        peak_source = 0
        peak_staging = 0
        difference = _DifferenceAccumulator()
        record_order_valid = True
        lengths_offsets_valid = True
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            h2d_stream = torch.cuda.Stream(device=device)
            compute_stream = torch.cuda.Stream(device=device)
            d2h_stream = torch.cuda.Stream(device=device)

            def finalize(value: _PendingExtent) -> None:
                nonlocal source_seconds
                nonlocal h2d_seconds
                nonlocal compute_seconds
                nonlocal d2h_seconds
                nonlocal stage_seconds
                nonlocal allocation_seconds
                nonlocal physical_output
                nonlocal peak_source
                nonlocal peak_staging
                nonlocal record_order_valid
                nonlocal lengths_offsets_valid
                done = value.d2h_end or value.compute_end
                done.synchronize()
                source_seconds += value.source_read_seconds
                h2d_seconds += (
                    value.h2d_start.elapsed_time(value.h2d_end) / 1000.0
                )
                compute_seconds += (
                    value.compute_start.elapsed_time(value.compute_end) / 1000.0
                )
                if value.d2h_start is not None and value.d2h_end is not None:
                    d2h_seconds += (
                        value.d2h_start.elapsed_time(value.d2h_end) / 1000.0
                    )
                allocation_seconds += value.target_allocation_seconds
                peak_source = max(
                    peak_source,
                    value.source_metrics.peak_source_resident_bytes,
                )
                if self.destination_kind == "dram":
                    peak_staging = max(
                        peak_staging,
                        value.device_output.nbytes,
                    )
                if validate:
                    current, order_valid, metadata_valid = self._validate_pending(
                        value,
                        transform,
                        atol,
                        rtol,
                    )
                    difference.merge(current)
                    record_order_valid = record_order_valid and order_valid
                    lengths_offsets_valid = (
                        lengths_offsets_valid and metadata_valid
                    )
                stage_started = time.perf_counter()
                transaction.stage(value.spec.extent_id, value.published_output)
                stage_seconds += time.perf_counter() - stage_started
                physical_output += value.published_output.nbytes

            for spec in assignment:
                prior_source = sum(
                    value.host_batch.nbytes for value in pending
                )
                value = self._enqueue(
                    reader,
                    spec,
                    transform,
                    h2d_stream,
                    compute_stream,
                    d2h_stream,
                )
                peak_source = max(
                    peak_source,
                    prior_source
                    + value.source_metrics.peak_source_resident_bytes,
                )
                pending.append(value)
                if self.destination_kind == "dram":
                    peak_staging = max(
                        peak_staging,
                        sum(
                            current.device_output.nbytes
                            for current in pending
                        ),
                    )
                if len(pending) >= self.runtime_config.max_inflight:
                    finalize(pending.popleft())
            while pending:
                finalize(pending.popleft())
            torch.cuda.synchronize(device)
            peak_hbm = torch.cuda.max_memory_allocated(device)
        records = sum(len(value.records) for value in assignment)
        tokens = sum(value.token_count for value in assignment)
        return _WorkerResult(
            metrics=Stage4DeviceMetrics(
                index=index,
                record_count=records,
                prefix_tokens=tokens,
                logical_input_bytes=sum(
                    value.logical_input_bytes for value in assignment
                ),
                physical_input_bytes=sum(
                    value.physical_input_bytes for value in assignment
                ),
                logical_output_bytes=sum(
                    value.logical_output_bytes for value in assignment
                ),
                physical_output_bytes=physical_output,
                elapsed_seconds=time.perf_counter() - started,
                source_read_seconds=source_seconds,
                h2d_seconds=h2d_seconds,
                compute_seconds=compute_seconds,
                d2h_seconds=d2h_seconds,
                stage_seconds=stage_seconds,
                target_allocation_seconds=allocation_seconds,
                peak_hbm_bytes=peak_hbm,
                peak_source_resident_bytes=peak_source,
                peak_staging_bytes=peak_staging,
                peak_publication_queue_bytes=0,
            ),
            correctness=difference,
            record_order_valid=record_order_valid,
            lengths_offsets_valid=lengths_offsets_valid,
        )

    def run(
        self,
        record_ids: tuple[int, ...] | None = None,
        validate: bool = False,
        job_id: str | None = None,
        atol: float = 0.02,
        rtol: float = 0.02,
    ) -> Stage4JobResult:
        if atol != 0.02 or rtol != 0.02:
            raise ValueError("Stage 4 transport tolerances are frozen")
        started = time.perf_counter()
        scan_started = time.perf_counter()
        reader = LazyStage4SourceReader(
            self.source_manifest_path,
            self.expected_workload_content_sha256,
        )
        scan_seconds = time.perf_counter() - scan_started
        if (
            self.expected_source_manifest_sha256 is not None
            and reader.manifest_file_sha256
            != self.expected_source_manifest_sha256
        ):
            raise ValueError("Stage 4 source manifest hash mismatch")
        if reader.manifest.target_version != self.target_version:
            raise ValueError("Stage 4 source and transform targets differ")
        if record_ids is None:
            record_ids = tuple(value.record_id for value in reader.manifest.records)
        source_versions = {
            reader.manifest.record_map[value].source_version
            for value in record_ids
        }
        extents = build_stage4_extents(
            reader.manifest,
            record_ids,
            self._representations(source_versions),
            self.runtime_config.batch_size,
            self.runtime_config.length_bucket_width,
        )
        assignments = place_stage4_extents_lpt(extents, len(self.transforms))
        destination = self._destination()
        transaction = destination.begin(
            job_id=job_id
            or (
                f"stage4-{self.method}-{self.destination_kind}-"
                f"{len(self.transforms)}gpu"
            ),
            target_version=self.target_version,
            expected_record_ids=record_ids,
        )
        try:
            with ThreadPoolExecutor(
                max_workers=len(self.transforms)
            ) as pool:
                futures = [
                    pool.submit(
                        self._worker,
                        index,
                        reader,
                        assignment,
                        transform,
                        transaction,
                        validate,
                        atol,
                        rtol,
                    )
                    for index, (assignment, transform) in enumerate(
                        zip(assignments, self.transforms, strict=True)
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
        coordinator_seconds = max(
            elapsed
            - scan_seconds
            - max(value.metrics.elapsed_seconds for value in workers)
            - commit_seconds,
            0.0,
        )
        devices = tuple(value.metrics for value in workers)
        covered = tuple(
            record_id
            for extent in manifest.extents
            for record_id in extent.record_ids
        )
        if (
            len(covered) != len(record_ids)
            or len(set(covered)) != len(covered)
            or set(covered) != set(record_ids)
        ):
            raise RuntimeError("Stage 4 committed coverage is incomplete")
        aggregate = _DifferenceAccumulator()
        record_order_valid = True
        lengths_offsets_valid = True
        for worker in workers:
            aggregate.merge(worker.correctness)
            record_order_valid = (
                record_order_valid and worker.record_order_valid
            )
            lengths_offsets_valid = (
                lengths_offsets_valid and worker.lengths_offsets_valid
            )
        correctness = None
        if validate:
            correctness = Stage4Correctness(
                finite=aggregate.finite,
                allclose=aggregate.mismatched == 0,
                max_abs_error=aggregate.max_abs_error,
                valid_element_count=aggregate.elements,
                record_order_valid=record_order_valid,
                lengths_offsets_valid=lengths_offsets_valid,
                atol=atol,
                rtol=rtol,
            )
        total_output = sum(value.logical_output_bytes for value in devices)
        expected_elements = total_output // torch.tensor(
            [],
            dtype=torch.float16,
        ).element_size()
        if validate and correctness.valid_element_count != expected_elements:
            raise RuntimeError("Stage 4 correctness did not cover all output elements")
        return Stage4JobResult(
            report=Stage4JobReport(
                method=self.method,
                destination=self.destination_kind,
                runtime_config=self.runtime_config,
                source_manifest_sha256=reader.manifest_file_sha256,
                workload_content_sha256=reader.manifest.workload_content_sha256,
                record_count=len(record_ids),
                prefix_tokens=sum(value.token_count for value in extents),
                logical_input_bytes=sum(
                    value.logical_input_bytes for value in devices
                ),
                physical_input_bytes=sum(
                    value.physical_input_bytes for value in devices
                ),
                logical_output_bytes=total_output,
                physical_output_bytes=sum(
                    value.physical_output_bytes for value in devices
                ),
                elapsed_seconds=elapsed,
                source_manifest_scan_seconds=scan_seconds,
                commit_seconds=commit_seconds,
                coordinator_seconds=max(coordinator_seconds, 0.0),
                devices=devices,
                manifest=manifest,
                correctness=correctness,
            ),
            destination=destination,
        )


def available_host_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_AVPHYS_PAGES")
    return page_size * pages


def _padded_source_bytes(extent: Stage4ExtentSpec) -> int:
    total = len(extent.records) * torch.tensor(
        [],
        dtype=torch.long,
    ).element_size()
    for record in extent.records:
        for representation in extent.representations:
            logical = record.shard_map[representation].logical_bytes
            if logical % record.prefix_tokens:
                raise ValueError("Stage 4 source bytes are not token-linear")
            total += (
                logical
                // record.prefix_tokens
                * extent.sequence_width
            )
    return total


def _source_wave_bytes(
    assignment: tuple[Stage4ExtentSpec, ...],
    max_inflight: int,
) -> int:
    if max_inflight < 1:
        raise ValueError("Stage 4 source wave requires positive in-flight depth")
    pending: deque[int] = deque()
    peak = 0
    for extent in assignment:
        batch_bytes = _padded_source_bytes(extent)
        transient = max(
            record.shard_map[representation].physical_bytes
            + record.shard_map[representation].logical_bytes
            for record in extent.records
            for representation in extent.representations
        )
        peak = max(
            peak,
            sum(pending) + batch_bytes + transient,
        )
        pending.append(batch_bytes)
        if len(pending) >= max_inflight:
            pending.popleft()
    return peak


def _retained_output_bytes(
    assignments: tuple[tuple[Stage4ExtentSpec, ...], ...],
) -> int:
    return sum(
        extent.logical_output_bytes
        + (2 * len(extent.records) + 1)
        * torch.tensor([], dtype=torch.long).element_size()
        for assignment in assignments
        for extent in assignment
    )


def _extent_output_bytes(extent: Stage4ExtentSpec) -> int:
    return (
        extent.logical_output_bytes
        + (2 * len(extent.records) + 1)
        * torch.tensor([], dtype=torch.long).element_size()
    )


def _device_transient_wave_bytes(
    assignment: tuple[Stage4ExtentSpec, ...],
    destination: str,
    max_inflight: int,
) -> int:
    if destination not in STAGE4_DESTINATIONS or max_inflight < 1:
        raise ValueError("Stage 4 device wave inputs are invalid")
    pending: deque[tuple[int, int]] = deque()
    peak = 0
    for extent in assignment:
        output = _extent_output_bytes(extent)
        resident = _padded_source_bytes(extent)
        if destination == "dram":
            resident += output
        pending.append((resident, output))
        peak = max(
            peak,
            sum(value[0] for value in pending) + pending[0][1],
        )
        if len(pending) >= max_inflight:
            pending.popleft()
    return peak


def stage4_capacity_preflight(
    assignments: tuple[tuple[Stage4ExtentSpec, ...], ...],
    transforms: tuple[Stage4Transform, ...],
    destination: str,
    maximum_transient_hbm_bytes: tuple[int, ...],
    max_inflight: int,
    calibration_assignments: (
        tuple[tuple[Stage4ExtentSpec, ...], ...] | None
    ) = None,
    allocator_margin_bytes: int = 512 * 1024**2,
) -> dict[str, object]:
    if (
        len(assignments) != len(transforms)
        or len(maximum_transient_hbm_bytes) != len(transforms)
        or destination not in STAGE4_DESTINATIONS
        or (
            calibration_assignments is not None
            and len(calibration_assignments) != len(transforms)
        )
    ):
        raise ValueError("Stage 4 capacity preflight inputs differ")
    available_host = available_host_bytes()
    source_peaks = [
        _source_wave_bytes(assignment, max_inflight)
        for assignment in assignments
    ]
    retained_target = _retained_output_bytes(assignments)
    required_host = sum(source_peaks)
    if destination == "dram":
        required_host += retained_target
    full_device_waves = tuple(
        _device_transient_wave_bytes(
            assignment,
            destination,
            max_inflight,
        )
        for assignment in assignments
    )
    calibration_device_waves = None
    calibrated_transients = maximum_transient_hbm_bytes
    compute_slack = None
    if calibration_assignments is not None:
        calibration_device_waves = tuple(
            _device_transient_wave_bytes(
                assignment,
                destination,
                max_inflight,
            )
            for assignment in calibration_assignments
        )
        compute_slack = max(
            max(measured - wave, 0)
            for measured, wave in zip(
                maximum_transient_hbm_bytes,
                calibration_device_waves,
                strict=True,
            )
        )
        measured_maximum = max(maximum_transient_hbm_bytes)
        calibrated_transients = tuple(
            max(measured_maximum, wave + compute_slack)
            for wave in full_device_waves
        )
    per_gpu = []
    passed = available_host >= required_host
    minimum_free = math.inf
    maximum_required = 0
    for index, (assignment, transform, transient) in enumerate(zip(
        assignments,
        transforms,
        calibrated_transients,
        strict=True,
    )):
        device = transform.device
        with torch.cuda.device(device):
            free, total = torch.cuda.mem_get_info(device)
            allocated = torch.cuda.memory_allocated(device)
        target = (
            sum(_extent_output_bytes(value) for value in assignment)
            if destination == "hbm"
            else 0
        )
        required = (
            allocated
            + target
            + transient
            + allocator_margin_bytes
        )
        device_passed = total >= required and free >= required - allocated
        passed = passed and device_passed
        minimum_free = min(minimum_free, free)
        maximum_required = max(maximum_required, required)
        per_gpu.append(
            {
                "index": device.index,
                "observed_free_hbm_bytes": free,
                "total_hbm_bytes": total,
                "resident_bytes": allocated,
                "assigned_target_bytes": target,
                "maximum_transient_bytes": transient,
                "measured_calibration_transient_bytes": (
                    maximum_transient_hbm_bytes[index]
                ),
                "full_device_wave_bytes": full_device_waves[index],
                "calibration_device_wave_bytes": (
                    None
                    if calibration_device_waves is None
                    else calibration_device_waves[index]
                ),
                "shared_compute_slack_bytes": compute_slack,
                "allocator_margin_bytes": allocator_margin_bytes,
                "required_peak_hbm_bytes": required,
                "passed": device_passed,
            }
        )
    return {
        "minimum_observed_free_hbm_bytes": int(minimum_free),
        "required_peak_hbm_bytes": maximum_required,
        "minimum_observed_available_host_bytes": available_host,
        "required_peak_host_bytes": required_host,
        "per_gpu": per_gpu,
        "passed": passed,
    }
