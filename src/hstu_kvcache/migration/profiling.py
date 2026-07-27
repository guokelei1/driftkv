from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .capsule import (
    ContiguousKVOutputExtent,
    MigratedKVBatch,
    MigrationCapsuleBatch,
)
from .operator import MigrationOperator, validate_contiguous_output_extent
from .program import MigrationProgram


@dataclass(frozen=True)
class LatencySamples:
    values_ms: tuple[float, ...]

    @property
    def median_ms(self) -> float:
        return statistics.median(self.values_ms)

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.values_ms)


@dataclass(frozen=True)
class OperatorStageProfile:
    operator: str
    stages: dict[str, LatencySamples]
    total: LatencySamples


@dataclass(frozen=True)
class ExtentOperatorProfile:
    operator: str
    latency: LatencySamples
    temporary_peak_bytes: tuple[int, ...]


def _validate_cuda_profile(
    program: MigrationProgram,
    capsule: MigrationCapsuleBatch,
    repeats: int,
) -> None:
    if capsule.device.type != "cuda":
        raise ValueError("operator profiling requires a CUDA capsule")
    if program.device != capsule.device:
        raise ValueError("program and capsule must share a CUDA device")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    program.validate_capsule(capsule)


@torch.no_grad()
def benchmark_cuda_operator(
    operator: MigrationOperator,
    program: MigrationProgram,
    capsule: MigrationCapsuleBatch,
    warmup_repeats: int,
    timing_repeats: int,
) -> tuple[MigratedKVBatch, LatencySamples]:
    if warmup_repeats < 0:
        raise ValueError("warmup_repeats must be nonnegative")
    prepared = operator.prepare_program(program, capsule.device)
    _validate_cuda_profile(prepared, capsule, timing_repeats)
    result = None
    for _ in range(warmup_repeats):
        result = operator.execute(prepared, capsule)
    torch.cuda.synchronize(capsule.device)
    samples = []
    for _ in range(timing_repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operator.execute(prepared, capsule)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    if result is None:
        raise RuntimeError("operator benchmark produced no result")
    return result, LatencySamples(tuple(samples))


@torch.no_grad()
def benchmark_cuda_extent_operator(
    operator: MigrationOperator,
    program: MigrationProgram,
    capsule: MigrationCapsuleBatch,
    destination: ContiguousKVOutputExtent,
    warmup_repeats: int,
    timing_repeats: int,
) -> tuple[ContiguousKVOutputExtent, ExtentOperatorProfile]:
    if warmup_repeats < 0:
        raise ValueError("warmup_repeats must be nonnegative")
    prepared = operator.prepare_program(program, capsule.device)
    _validate_cuda_profile(prepared, capsule, timing_repeats)
    validate_contiguous_output_extent(
        prepared,
        capsule,
        destination,
        check_metadata_values=True,
    )
    result = destination
    for _ in range(warmup_repeats):
        result = operator.execute_into(prepared, capsule, destination)
    torch.cuda.synchronize(capsule.device)
    samples = []
    peaks = []
    for _ in range(timing_repeats):
        torch.cuda.reset_peak_memory_stats(capsule.device)
        baseline = torch.cuda.memory_allocated(capsule.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operator.execute_into(prepared, capsule, destination)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        peaks.append(
            max(
                0,
                torch.cuda.max_memory_allocated(capsule.device) - baseline,
            )
        )
    return result, ExtentOperatorProfile(
        operator=operator.name,
        latency=LatencySamples(tuple(samples)),
        temporary_peak_bytes=tuple(peaks),
    )


def _record_stages(
    operations: tuple[tuple[str, Callable[[], None]], ...],
    repeats: int,
) -> tuple[dict[str, LatencySamples], LatencySamples]:
    values = {name: [] for name, _ in operations}
    totals = []
    for _ in range(repeats):
        events = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(len(operations) + 1)
        ]
        events[0].record()
        for index, (_, operation) in enumerate(operations):
            operation()
            events[index + 1].record()
        events[-1].synchronize()
        for index, (name, _) in enumerate(operations):
            values[name].append(events[index].elapsed_time(events[index + 1]))
        totals.append(events[0].elapsed_time(events[-1]))
    return (
        {
            name: LatencySamples(tuple(samples))
            for name, samples in values.items()
        },
        LatencySamples(tuple(totals)),
    )


@torch.no_grad()
def profile_reference_operator_stages(
    program: MigrationProgram,
    capsule: MigrationCapsuleBatch,
    repeats: int,
) -> OperatorStageProfile:
    program = program.to(capsule.device, dtype=torch.float32)
    _validate_cuda_profile(program, capsule, repeats)
    state: dict[str, torch.Tensor] = {}

    def input_cast() -> None:
        state["normed"] = capsule.normed.float().flatten(1, 2)

    def projection() -> None:
        state["projected"] = torch.bmm(
            state["normed"],
            program.adapter.weights,
        ).unflatten(1, capsule.normed.shape[1:3])

    def bias() -> None:
        state["projected"] = (
            state["projected"] + program.adapter.biases[:, None, None, :]
        )

    def mask() -> None:
        positions = torch.arange(capsule.seq_len, device=capsule.device)
        valid = positions.unsqueeze(0) < capsule.lengths.unsqueeze(1)
        state["projected"] = (
            state["projected"] * valid.unsqueeze(0).unsqueeze(-1)
        )

    def output_cast() -> None:
        state["output"] = state["projected"].to(capsule.normed.dtype)

    operations = (
        ("input_cast", input_cast),
        ("bmm", projection),
        ("bias", bias),
        ("mask", mask),
        ("output_cast", output_cast),
    )
    for _, operation in operations:
        operation()
    torch.cuda.synchronize(capsule.device)
    stages, total = _record_stages(operations, repeats)
    return OperatorStageProfile(
        operator="reference_fp32",
        stages=stages,
        total=total,
    )


@torch.no_grad()
def profile_packed_operator_stages(
    program: MigrationProgram,
    capsule: MigrationCapsuleBatch,
    execution_dtype: torch.dtype,
    repeats: int,
) -> OperatorStageProfile:
    program = program.to(capsule.device, dtype=execution_dtype)
    _validate_cuda_profile(program, capsule, repeats)
    state: dict[str, torch.Tensor] = {}

    def input_cast() -> None:
        state["normed"] = capsule.normed.to(execution_dtype).flatten(1, 2)

    def projection_bias() -> None:
        bias = program.adapter.biases[:, None, :].expand(
            -1,
            state["normed"].shape[1],
            -1,
        )
        state["projected"] = torch.baddbmm(
            bias,
            state["normed"],
            program.adapter.weights,
        ).unflatten(1, capsule.normed.shape[1:3])

    def mask() -> None:
        positions = torch.arange(capsule.seq_len, device=capsule.device)
        invalid = positions.unsqueeze(0) >= capsule.lengths.unsqueeze(1)
        state["projected"].masked_fill_(
            invalid.unsqueeze(0).unsqueeze(-1),
            0,
        )

    def output_cast() -> None:
        state["output"] = state["projected"].to(capsule.normed.dtype)

    operations = (
        ("input_cast", input_cast),
        ("baddbmm_bias", projection_bias),
        ("mask_inplace", mask),
        ("output_cast", output_cast),
    )
    for _, operation in operations:
        operation()
    torch.cuda.synchronize(capsule.device)
    stages, total = _record_stages(operations, repeats)
    return OperatorStageProfile(
        operator=f"packed_{str(execution_dtype).removeprefix('torch.')}",
        stages=stages,
        total=total,
    )


def _write_projected_to_extent(
    projected: torch.Tensor,
    capsule: MigrationCapsuleBatch,
    destination: ContiguousKVOutputExtent,
) -> None:
    positions = torch.arange(capsule.seq_len, device=capsule.device)
    valid = positions.unsqueeze(0) < capsule.lengths.unsqueeze(1)
    width = projected.shape[-1] // 2
    destination.k.copy_(projected[..., :width][:, valid])
    destination.v.copy_(projected[..., width:][:, valid])


@torch.no_grad()
def profile_reference_extent_operator_stages(
    program: MigrationProgram,
    capsule: MigrationCapsuleBatch,
    destination: ContiguousKVOutputExtent,
    repeats: int,
) -> OperatorStageProfile:
    program = program.to(capsule.device, dtype=torch.float32)
    _validate_cuda_profile(program, capsule, repeats)
    validate_contiguous_output_extent(
        program,
        capsule,
        destination,
        check_metadata_values=True,
    )
    state: dict[str, torch.Tensor] = {}

    def input_cast() -> None:
        state["normed"] = capsule.normed.float().flatten(1, 2)

    def projection() -> None:
        state["projected"] = torch.bmm(
            state["normed"],
            program.adapter.weights,
        ).unflatten(1, capsule.normed.shape[1:3])

    def bias() -> None:
        state["projected"] = (
            state["projected"] + program.adapter.biases[:, None, None, :]
        )

    def mask() -> None:
        positions = torch.arange(capsule.seq_len, device=capsule.device)
        valid = positions.unsqueeze(0) < capsule.lengths.unsqueeze(1)
        state["projected"] = (
            state["projected"] * valid.unsqueeze(0).unsqueeze(-1)
        )

    def output_cast() -> None:
        state["projected"] = state["projected"].to(capsule.normed.dtype)

    def extent_write() -> None:
        _write_projected_to_extent(
            state["projected"],
            capsule,
            destination,
        )

    operations = (
        ("input_cast", input_cast),
        ("bmm", projection),
        ("bias", bias),
        ("length_mask", mask),
        ("output_cast", output_cast),
        ("split_compact_extent_write", extent_write),
    )
    for _, operation in operations:
        operation()
    torch.cuda.synchronize(capsule.device)
    stages, total = _record_stages(operations, repeats)
    return OperatorStageProfile(
        operator="reference_fp32_extent",
        stages=stages,
        total=total,
    )


@torch.no_grad()
def profile_packed_extent_operator_stages(
    program: MigrationProgram,
    capsule: MigrationCapsuleBatch,
    destination: ContiguousKVOutputExtent,
    execution_dtype: torch.dtype,
    repeats: int,
) -> OperatorStageProfile:
    program = program.to(capsule.device, dtype=execution_dtype)
    _validate_cuda_profile(program, capsule, repeats)
    validate_contiguous_output_extent(
        program,
        capsule,
        destination,
        check_metadata_values=True,
    )
    state: dict[str, torch.Tensor] = {}

    def input_cast() -> None:
        state["normed"] = capsule.normed.to(execution_dtype).flatten(1, 2)

    def projection_bias() -> None:
        bias = program.adapter.biases[:, None, :].expand(
            -1,
            state["normed"].shape[1],
            -1,
        )
        state["projected"] = torch.baddbmm(
            bias,
            state["normed"],
            program.adapter.weights,
        ).unflatten(1, capsule.normed.shape[1:3])

    def mask() -> None:
        positions = torch.arange(capsule.seq_len, device=capsule.device)
        invalid = positions.unsqueeze(0) >= capsule.lengths.unsqueeze(1)
        state["projected"].masked_fill_(
            invalid.unsqueeze(0).unsqueeze(-1),
            0,
        )

    def output_cast() -> None:
        state["projected"] = state["projected"].to(capsule.normed.dtype)

    def extent_write() -> None:
        _write_projected_to_extent(
            state["projected"],
            capsule,
            destination,
        )

    operations = (
        ("input_cast", input_cast),
        ("baddbmm_bias", projection_bias),
        ("length_mask_inplace", mask),
        ("output_cast", output_cast),
        ("split_compact_extent_write", extent_write),
    )
    for _, operation in operations:
        operation()
    torch.cuda.synchronize(capsule.device)
    stages, total = _record_stages(operations, repeats)
    dtype_name = str(execution_dtype).removeprefix("torch.")
    return OperatorStageProfile(
        operator=f"packed_{dtype_name}_extent",
        stages=stages,
        total=total,
    )


@torch.no_grad()
def profile_fused_extent_operator_stages(
    operator: MigrationOperator,
    program: MigrationProgram,
    capsule: MigrationCapsuleBatch,
    destination: ContiguousKVOutputExtent,
    repeats: int,
) -> OperatorStageProfile:
    prepared = operator.prepare_program(program, capsule.device)
    _validate_cuda_profile(prepared, capsule, repeats)
    validate_contiguous_output_extent(
        prepared,
        capsule,
        destination,
        check_metadata_values=True,
    )

    def fused() -> None:
        operator.execute_into(prepared, capsule, destination)

    operations = (
        ("affine_bias_length_split_direct_extent_write", fused),
    )
    fused()
    torch.cuda.synchronize(capsule.device)
    stages, total = _record_stages(operations, repeats)
    return OperatorStageProfile(
        operator=f"{operator.name}_extent",
        stages=stages,
        total=total,
    )
