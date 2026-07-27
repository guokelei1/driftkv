from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import torch

from .cohort_jagged import JaggedMigratedKVBatch
from .stage4_engine import Stage4Transform
from .stage4_source import Stage4ExtentSpec
from .stage45_resident import (
    STAGE45_RESIDENT_PROTOCOL,
    Stage45DeviceMetrics,
    Stage45JobResult,
    Stage45ResidentEngine,
    Stage45ResidentExtent,
    Stage45ResidentPlan,
    Stage45ResidentSource,
    _available_host_bytes,
    _make_device_output,
    _padded_source_bytes,
    _Pending,
    _WorkerResult,
)

STAGE45_RECLAIM_PROTOCOL = "cohortkv_stage4_5_extent_reclaim_v1"


def _replacement_wave_bytes(
    assignment,
    max_inflight: int,
) -> int:
    pending: deque[int] = deque()
    peak = 0
    for extent in assignment:
        pending.append(
            extent.logical_output_bytes
            + (2 * len(extent.records) + 1)
            * torch.tensor([], dtype=torch.long).element_size()
        )
        peak = max(peak, sum(pending))
        if len(pending) >= max_inflight:
            pending.popleft()
    return peak


def _movement_wave_bytes(
    assignment,
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


def stage45_reclaim_preflight(
    plan: Stage45ResidentPlan,
    transforms: tuple[Stage4Transform, ...],
    allocator_margin_bytes: int = 2 * 1024**3,
) -> dict[str, object]:
    if (
        plan.source_tier not in {"dram_resident", "hbm_resident"}
        or len(transforms) != len(plan.assignments)
        or allocator_margin_bytes < 1
    ):
        raise ValueError("Stage 4.5 reclaim preflight inputs are invalid")
    available_host = _available_host_bytes()
    host_source = (
        plan.resident_source_bytes
        if plan.source_tier == "dram_resident"
        else 0
    )
    passed = available_host >= host_source
    per_gpu = []
    for index, (assignment, transform) in enumerate(
        zip(plan.assignments, transforms, strict=True)
    ):
        if transform.device.type != "cuda":
            raise ValueError("Stage 4.5 reclaim transform is not on CUDA")
        with torch.cuda.device(transform.device):
            free, total = torch.cuda.mem_get_info(transform.device)
            allocated = torch.cuda.memory_allocated(transform.device)
        old_kv = sum(
            extent.logical_output_bytes
            + (2 * len(extent.records) + 1)
            * torch.tensor([], dtype=torch.long).element_size()
            for extent in assignment
        )
        replacement_wave = _replacement_wave_bytes(
            assignment,
            plan.runtime_config.max_inflight,
        )
        movement_wave = _movement_wave_bytes(
            assignment,
            plan.runtime_config.max_inflight,
        )
        standing_hbm_source = (
            sum(_padded_source_bytes(value) for value in assignment)
            if plan.source_tier == "hbm_resident"
            else 0
        )
        timed_movement_wave = (
            movement_wave
            if plan.source_tier == "dram_resident"
            else 0
        )
        required = (
            allocated
            + old_kv
            + replacement_wave
            + standing_hbm_source
            + timed_movement_wave
            + allocator_margin_bytes
        )
        device_passed = total >= required and free >= required - allocated
        passed = passed and device_passed
        per_gpu.append(
            {
                "index": index,
                "device": str(transform.device),
                "observed_allocated_hbm_bytes": allocated,
                "observed_free_hbm_bytes": free,
                "total_hbm_bytes": total,
                "transform_resident_bytes": transform.resident_bytes,
                "standing_old_kv_bytes": old_kv,
                "standing_source_hbm_bytes": standing_hbm_source,
                "maximum_replacement_wave_bytes": replacement_wave,
                "maximum_source_movement_wave_bytes": timed_movement_wave,
                "allocator_margin_bytes": allocator_margin_bytes,
                "required_peak_hbm_bytes": required,
                "passed": device_passed,
            }
        )
    return {
        "protocol": STAGE45_RECLAIM_PROTOCOL,
        "parent_protocol": STAGE45_RESIDENT_PROTOCOL,
        "method": plan.method,
        "source_tier": plan.source_tier,
        "record_count": plan.record_count,
        "prefix_tokens": plan.prefix_tokens,
        "observed_available_host_bytes": available_host,
        "required_standing_host_source_bytes": host_source,
        "per_gpu": per_gpu,
        "passed": passed,
    }


@dataclass(frozen=True)
class Stage45ReclaimDeviceMetrics:
    index: int
    initial_old_kv_bytes: int
    retired_old_kv_bytes: int
    final_old_kv_bytes: int
    final_new_kv_bytes: int
    peak_old_plus_new_kv_bytes: int
    retired_extent_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "initial_old_kv_bytes": self.initial_old_kv_bytes,
            "retired_old_kv_bytes": self.retired_old_kv_bytes,
            "final_old_kv_bytes": self.final_old_kv_bytes,
            "final_new_kv_bytes": self.final_new_kv_bytes,
            "peak_old_plus_new_kv_bytes": (
                self.peak_old_plus_new_kv_bytes
            ),
            "retired_extent_count": self.retired_extent_count,
        }


@dataclass(frozen=True)
class Stage45ReclaimMetrics:
    initial_old_kv_bytes: int
    retired_old_kv_bytes: int
    final_old_kv_bytes: int
    final_new_kv_bytes: int
    peak_old_plus_new_kv_bytes: int
    retired_extent_count: int
    devices: tuple[Stage45ReclaimDeviceMetrics, ...]
    old_cache_kind: str = "shape_dtype_layout_equivalent_hbm_occupancy"
    retirement_boundary: str = (
        "replacement extent compute complete and transaction stage accepted"
    )
    protocol: str = STAGE45_RECLAIM_PROTOCOL

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "old_cache_kind": self.old_cache_kind,
            "retirement_boundary": self.retirement_boundary,
            "initial_old_kv_bytes": self.initial_old_kv_bytes,
            "retired_old_kv_bytes": self.retired_old_kv_bytes,
            "final_old_kv_bytes": self.final_old_kv_bytes,
            "final_new_kv_bytes": self.final_new_kv_bytes,
            "peak_old_plus_new_kv_bytes": (
                self.peak_old_plus_new_kv_bytes
            ),
            "retired_extent_count": self.retired_extent_count,
            "devices": [value.to_dict() for value in self.devices],
        }


class ReclaimableOldKV:
    def __init__(
        self,
        plan: Stage45ResidentPlan,
        batches: dict[str, JaggedMigratedKVBatch],
        extent_devices: dict[str, int],
    ) -> None:
        expected = {
            extent.extent_id
            for assignment in plan.assignments
            for extent in assignment
        }
        if set(batches) != expected or set(extent_devices) != expected:
            raise ValueError("Stage 4.5 old cache coverage is incomplete")
        self.plan = plan
        self._batches = batches
        self._extent_devices = extent_devices
        self._lock = threading.Lock()
        self._initial_by_device = [0] * len(plan.assignments)
        for extent_id, batch in batches.items():
            index = extent_devices[extent_id]
            if batch.k.device != torch.device("cuda", index):
                raise ValueError("Stage 4.5 old cache extent is misplaced")
            self._initial_by_device[index] += batch.nbytes
        self._old_by_device = list(self._initial_by_device)
        self._new_by_device = [0] * len(plan.assignments)
        self._peak_by_device = list(self._initial_by_device)
        self._retired_by_device = [0] * len(plan.assignments)
        self._retired_extents_by_device = [0] * len(plan.assignments)

    def batch_for(
        self,
        extent_id: str,
        device_index: int,
    ) -> JaggedMigratedKVBatch:
        with self._lock:
            if self._extent_devices.get(extent_id) != device_index:
                raise ValueError("Stage 4.5 old extent device differs")
            try:
                return self._batches[extent_id]
            except KeyError as exc:
                raise ValueError(
                    "Stage 4.5 old extent is already retired"
                ) from exc

    def register_replacement(
        self,
        extent: Stage45ResidentExtent | Stage4ExtentSpec,
        output: JaggedMigratedKVBatch,
        device_index: int,
    ) -> None:
        spec = (
            extent.spec
            if isinstance(extent, Stage45ResidentExtent)
            else extent
        )
        with self._lock:
            if (
                self._extent_devices.get(spec.extent_id)
                != device_index
                or spec.extent_id not in self._batches
                or output.k.device != torch.device("cuda", device_index)
            ):
                raise ValueError("Stage 4.5 replacement does not match old cache")
            old = self._batches[spec.extent_id]
            if (
                output.nbytes != old.nbytes
                or output.record_ids != old.record_ids
                or output.record_ids != spec.record_ids
            ):
                raise ValueError("Stage 4.5 replacement layout differs")
            self._new_by_device[device_index] += output.nbytes
            self._peak_by_device[device_index] = max(
                self._peak_by_device[device_index],
                self._old_by_device[device_index]
                + self._new_by_device[device_index],
            )

    def retire(
        self,
        extent_id: str,
        device_index: int,
    ) -> None:
        with self._lock:
            if self._extent_devices.get(extent_id) != device_index:
                raise ValueError("Stage 4.5 retirement device differs")
            try:
                batch = self._batches.pop(extent_id)
            except KeyError as exc:
                raise ValueError(
                    "Stage 4.5 old extent is already retired"
                ) from exc
            size = batch.nbytes
            self._old_by_device[device_index] -= size
            self._retired_by_device[device_index] += size
            self._retired_extents_by_device[device_index] += 1

    def metrics(self) -> Stage45ReclaimMetrics:
        with self._lock:
            devices = tuple(
                Stage45ReclaimDeviceMetrics(
                    index=index,
                    initial_old_kv_bytes=self._initial_by_device[index],
                    retired_old_kv_bytes=self._retired_by_device[index],
                    final_old_kv_bytes=self._old_by_device[index],
                    final_new_kv_bytes=self._new_by_device[index],
                    peak_old_plus_new_kv_bytes=self._peak_by_device[index],
                    retired_extent_count=(
                        self._retired_extents_by_device[index]
                    ),
                )
                for index in range(len(self._initial_by_device))
            )
        return Stage45ReclaimMetrics(
            initial_old_kv_bytes=sum(
                value.initial_old_kv_bytes for value in devices
            ),
            retired_old_kv_bytes=sum(
                value.retired_old_kv_bytes for value in devices
            ),
            final_old_kv_bytes=sum(
                value.final_old_kv_bytes for value in devices
            ),
            final_new_kv_bytes=sum(
                value.final_new_kv_bytes for value in devices
            ),
            peak_old_plus_new_kv_bytes=sum(
                value.peak_old_plus_new_kv_bytes for value in devices
            ),
            retired_extent_count=sum(
                value.retired_extent_count for value in devices
            ),
            devices=devices,
        )


def allocate_reclaimable_old_kv(
    plan: Stage45ResidentPlan,
    transforms: tuple[Stage4Transform, ...],
    zero: bool = False,
) -> ReclaimableOldKV:
    if len(transforms) != len(plan.assignments):
        raise ValueError("Stage 4.5 old cache device count differs")
    batches = {}
    extent_devices = {}
    for index, (assignment, transform) in enumerate(
        zip(plan.assignments, transforms, strict=True)
    ):
        with torch.cuda.device(transform.device):
            for extent in assignment:
                batch = _make_device_output(
                    extent,
                    plan.num_layers,
                    plan.kv_width,
                    transform.device,
                )
                if zero:
                    batch.k.zero_()
                    batch.v.zero_()
                batches[extent.extent_id] = batch
                extent_devices[extent.extent_id] = index
    return ReclaimableOldKV(plan, batches, extent_devices)


@dataclass(frozen=True)
class Stage45ReclaimJobResult:
    base: Stage45JobResult
    reclamation: Stage45ReclaimMetrics

    @property
    def report(self):
        return self.base.report

    @property
    def destination(self):
        return self.base.destination


class Stage45ReclaimingEngine(Stage45ResidentEngine):
    def __init__(
        self,
        source: Stage45ResidentSource,
        transforms: tuple[Stage4Transform, ...],
        old_cache: ReclaimableOldKV | None = None,
    ) -> None:
        super().__init__(source, transforms)
        if (
            source.plan.source_tier
            not in {"dram_resident", "hbm_resident"}
            or (
                old_cache is not None
                and old_cache.plan is not source.plan
            )
        ):
            raise ValueError("Stage 4.5 reclaim engine requires one shared plan")
        self.old_cache = old_cache

    def install_old_cache(self, old_cache: ReclaimableOldKV) -> None:
        if old_cache.plan is not self.source.plan:
            raise ValueError("Stage 4.5 replacement old cache plan differs")
        if (
            self.old_cache is not None
            and self.old_cache.metrics().final_old_kv_bytes != 0
        ):
            raise RuntimeError("Stage 4.5 prior old cache is not retired")
        self.old_cache = old_cache

    def _worker(
        self,
        index: int,
        assignment: tuple[Stage45ResidentExtent, ...],
        transform: Stage4Transform,
        transaction,
    ) -> _WorkerResult:
        if self.old_cache is None:
            raise RuntimeError("Stage 4.5 reclaim engine has no old cache")
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
                self.old_cache.retire(value.extent.spec.extent_id, index)
                stage_seconds += time.perf_counter() - stage_started
                output_bytes += value.output.nbytes

            for extent in assignment:
                value = self._enqueue(
                    extent,
                    transform,
                    h2d_stream,
                    compute_stream,
                )
                self.old_cache.register_replacement(
                    extent,
                    value.output,
                    index,
                )
                pending.append(value)
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

    def run(
        self,
        validate: bool = False,
        job_id: str | None = None,
        atol: float = 0.02,
        rtol: float = 0.02,
    ) -> Stage45ReclaimJobResult:
        if self.old_cache is None:
            raise RuntimeError("Stage 4.5 reclaim engine has no old cache")
        base = super().run(
            validate=validate,
            job_id=job_id,
            atol=atol,
            rtol=rtol,
        )
        reclamation = self.old_cache.metrics()
        if (
            reclamation.final_old_kv_bytes != 0
            or reclamation.retired_old_kv_bytes
            != reclamation.initial_old_kv_bytes
            or reclamation.final_new_kv_bytes
            != base.report.physical_output_bytes
            or reclamation.retired_extent_count
            != len(self.source.plan.extents)
        ):
            raise RuntimeError("Stage 4.5 old K/V reclamation is incomplete")
        return Stage45ReclaimJobResult(
            base=base,
            reclamation=reclamation,
        )
