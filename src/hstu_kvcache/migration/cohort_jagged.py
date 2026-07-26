from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from .executor import CohortExecutionMetrics
from .multigpu import DeviceExecutionMetrics, MultiGPUExecutionMetrics
from .program import MigrationProgram


@dataclass(frozen=True)
class JaggedMigrationCapsuleBatch:
    record_ids: tuple[int, ...]
    migration_anchor_version: str
    normed: torch.Tensor
    lengths: torch.Tensor
    offsets: torch.Tensor

    def __post_init__(self) -> None:
        if not self.migration_anchor_version:
            raise ValueError("migration_anchor_version must be nonempty")
        if self.normed.ndim != 3:
            raise ValueError("normed must have shape [layers, tokens, hidden]")
        if self.lengths.ndim != 1:
            raise ValueError("lengths must be one-dimensional")
        if self.offsets.ndim != 1:
            raise ValueError("offsets must be one-dimensional")
        if not self.normed.is_floating_point():
            raise ValueError("normed must have a floating-point dtype")
        integer_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if self.lengths.dtype not in integer_dtypes:
            raise ValueError("lengths must have an integer dtype")
        if self.offsets.dtype not in integer_dtypes:
            raise ValueError("offsets must have an integer dtype")
        if self.normed.shape[0] < 1:
            raise ValueError("capsule must contain at least one layer")
        if self.normed.shape[1] < 1 or self.normed.shape[2] < 1:
            raise ValueError("token and hidden dimensions must be positive")
        if len(self.record_ids) != self.lengths.shape[0]:
            raise ValueError("record_ids and lengths differ")
        if self.offsets.shape != (len(self.record_ids) + 1,):
            raise ValueError("offsets must have one boundary per record")
        if not self.record_ids:
            raise ValueError("capsule must contain at least one record")
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError("record_ids must be unique within a batch")
        if (
            self.normed.device != self.lengths.device
            or self.normed.device != self.offsets.device
        ):
            raise ValueError("normed, lengths, and offsets must share a device")
        if self.device.type == "cpu":
            if bool(torch.any(self.lengths <= 0)):
                raise ValueError("jagged record lengths must be positive")
            if int(self.offsets[0]) != 0:
                raise ValueError("jagged offsets must start at zero")
            if int(self.offsets[-1]) != self.normed.shape[1]:
                raise ValueError("final jagged offset must equal token count")
            if not torch.equal(self.offsets[1:] - self.offsets[:-1], self.lengths):
                raise ValueError("jagged offsets and lengths differ")

    @property
    def device(self) -> torch.device:
        return self.normed.device

    @property
    def num_layers(self) -> int:
        return self.normed.shape[0]

    @property
    def batch_size(self) -> int:
        return len(self.record_ids)

    @property
    def token_count(self) -> int:
        return self.normed.shape[1]

    @property
    def hidden_size(self) -> int:
        return self.normed.shape[2]

    @property
    def nbytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (self.normed, self.lengths, self.offsets)
        )

    @property
    def is_pinned(self) -> bool:
        return (
            self.device.type == "cpu"
            and self.normed.is_pinned()
            and self.lengths.is_pinned()
            and self.offsets.is_pinned()
        )

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ) -> JaggedMigrationCapsuleBatch:
        return JaggedMigrationCapsuleBatch(
            record_ids=self.record_ids,
            migration_anchor_version=self.migration_anchor_version,
            normed=self.normed.to(device, non_blocking=non_blocking),
            lengths=self.lengths.to(device, non_blocking=non_blocking),
            offsets=self.offsets.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> JaggedMigrationCapsuleBatch:
        if self.device.type != "cpu":
            raise ValueError("only CPU capsules can be pinned")
        if self.is_pinned:
            return self
        return JaggedMigrationCapsuleBatch(
            record_ids=self.record_ids,
            migration_anchor_version=self.migration_anchor_version,
            normed=self.normed.pin_memory(),
            lengths=self.lengths.pin_memory(),
            offsets=self.offsets.pin_memory(),
        )


@dataclass(frozen=True)
class JaggedMigratedKVBatch:
    record_ids: tuple[int, ...]
    migration_anchor_version: str
    served_kv_target: str
    k: torch.Tensor
    v: torch.Tensor
    lengths: torch.Tensor
    offsets: torch.Tensor

    def __post_init__(self) -> None:
        if not self.migration_anchor_version or not self.served_kv_target:
            raise ValueError("migration and serving versions must be nonempty")
        if self.k.ndim != 3 or self.v.ndim != 3:
            raise ValueError("K/V must have shape [layers, tokens, width]")
        if self.k.shape != self.v.shape:
            raise ValueError("K and V shapes differ")
        if len(self.record_ids) != self.lengths.shape[0]:
            raise ValueError("record_ids and lengths differ")
        if self.offsets.shape != (len(self.record_ids) + 1,):
            raise ValueError("offsets must have one boundary per record")
        if (
            self.k.device != self.v.device
            or self.k.device != self.lengths.device
            or self.k.device != self.offsets.device
        ):
            raise ValueError("K/V and jagged metadata must share a device")
        if self.k.device.type == "cpu":
            if int(self.offsets[0]) != 0:
                raise ValueError("jagged offsets must start at zero")
            if int(self.offsets[-1]) != self.k.shape[1]:
                raise ValueError("final jagged offset must equal token count")
            if not torch.equal(self.offsets[1:] - self.offsets[:-1], self.lengths):
                raise ValueError("jagged offsets and lengths differ")

    @property
    def batch_size(self) -> int:
        return len(self.record_ids)

    @property
    def token_count(self) -> int:
        return self.k.shape[1]

    @property
    def nbytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (self.k, self.v, self.lengths, self.offsets)
        )

    def record_index(self, record_id: int) -> int:
        try:
            return self.record_ids.index(record_id)
        except ValueError as exc:
            raise KeyError("record is absent from jagged K/V batch") from exc

    def record_kv(self, record_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.record_index(record_id)
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.k[:, start:end], self.v[:, start:end]

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ) -> JaggedMigratedKVBatch:
        return JaggedMigratedKVBatch(
            record_ids=self.record_ids,
            migration_anchor_version=self.migration_anchor_version,
            served_kv_target=self.served_kv_target,
            k=self.k.to(device, non_blocking=non_blocking),
            v=self.v.to(device, non_blocking=non_blocking),
            lengths=self.lengths.to(device, non_blocking=non_blocking),
            offsets=self.offsets.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class PinnedJaggedKVOutputPool:
    served_kv_target: str
    outputs: dict[tuple[int, ...], JaggedMigratedKVBatch]

    def __post_init__(self) -> None:
        if not self.served_kv_target:
            raise ValueError("served_kv_target must be nonempty")
        if not self.outputs:
            raise ValueError("output pool must contain at least one extent")
        seen: set[int] = set()
        for record_ids, output in self.outputs.items():
            if record_ids != output.record_ids:
                raise ValueError("output-pool key and record IDs differ")
            if output.served_kv_target != self.served_kv_target:
                raise ValueError("output-pool target versions differ")
            overlap = seen.intersection(record_ids)
            if overlap:
                raise ValueError("output-pool record IDs must be globally unique")
            seen.update(record_ids)
            if output.k.device.type != "cpu":
                raise ValueError("output-pool K/V must be CPU resident")
            if not output.k.is_pinned() or not output.v.is_pinned():
                raise ValueError("output-pool K/V must be pinned")

    @classmethod
    def allocate(
        cls,
        batches: Sequence[JaggedMigrationCapsuleBatch],
        served_kv_target: str,
        num_layers: int,
        kv_width: int,
        dtype: torch.dtype = torch.float16,
    ) -> PinnedJaggedKVOutputPool:
        if not batches:
            raise ValueError("cannot allocate an empty output pool")
        if num_layers < 1 or kv_width < 1:
            raise ValueError("output-pool dimensions must be positive")
        outputs = {}
        for batch in batches:
            if batch.device.type != "cpu":
                raise ValueError("output-pool source batches must be CPU resident")
            shape = (num_layers, batch.token_count, kv_width)
            k = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            v = torch.empty_like(k, pin_memory=True)
            lengths = (
                batch.lengths
                if batch.lengths.is_pinned()
                else batch.lengths.pin_memory()
            )
            offsets = (
                batch.offsets
                if batch.offsets.is_pinned()
                else batch.offsets.pin_memory()
            )
            outputs[batch.record_ids] = JaggedMigratedKVBatch(
                record_ids=batch.record_ids,
                migration_anchor_version=batch.migration_anchor_version,
                served_kv_target=served_kv_target,
                k=k,
                v=v,
                lengths=lengths,
                offsets=offsets,
            )
        return cls(served_kv_target=served_kv_target, outputs=outputs)

    @property
    def nbytes(self) -> int:
        return sum(output.nbytes for output in self.outputs.values())

    def acquire(
        self,
        capsule: JaggedMigrationCapsuleBatch,
    ) -> JaggedMigratedKVBatch:
        try:
            output = self.outputs[capsule.record_ids]
        except KeyError as exc:
            raise KeyError("capsule extent is absent from output pool") from exc
        if output.migration_anchor_version != capsule.migration_anchor_version:
            raise ValueError("output-pool migration anchor differs from capsule")
        if output.token_count != capsule.token_count:
            raise ValueError("output-pool token count differs from capsule")
        if not torch.equal(output.lengths, capsule.lengths):
            raise ValueError("output-pool lengths differ from capsule")
        if not torch.equal(output.offsets, capsule.offsets):
            raise ValueError("output-pool offsets differ from capsule")
        return output


@dataclass(frozen=True)
class DeviceJaggedKVOutputPool:
    served_kv_target: str
    device: torch.device
    outputs: dict[tuple[int, ...], JaggedMigratedKVBatch]

    def __post_init__(self) -> None:
        if not self.served_kv_target:
            raise ValueError("served_kv_target must be nonempty")
        if self.device.type != "cuda":
            raise ValueError("device output pool requires a CUDA device")
        if not self.outputs:
            raise ValueError("output pool must contain at least one extent")
        seen: set[int] = set()
        for record_ids, output in self.outputs.items():
            if record_ids != output.record_ids:
                raise ValueError("output-pool key and record IDs differ")
            if output.served_kv_target != self.served_kv_target:
                raise ValueError("output-pool target versions differ")
            if output.k.device != self.device:
                raise ValueError("output-pool K/V are on the wrong device")
            overlap = seen.intersection(record_ids)
            if overlap:
                raise ValueError("output-pool record IDs must be globally unique")
            seen.update(record_ids)

    @classmethod
    def allocate(
        cls,
        batches: Sequence[JaggedMigrationCapsuleBatch],
        served_kv_target: str,
        num_layers: int,
        kv_width: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float16,
    ) -> DeviceJaggedKVOutputPool:
        if not batches:
            raise ValueError("cannot allocate an empty output pool")
        if num_layers < 1 or kv_width < 1:
            raise ValueError("output-pool dimensions must be positive")
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("device output pool requires a CUDA device")
        if resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        outputs = {}
        with torch.cuda.device(resolved):
            for batch in batches:
                if batch.device.type != "cpu":
                    raise ValueError("output-pool source batches must be CPU resident")
                shape = (num_layers, batch.token_count, kv_width)
                k = torch.empty(shape, dtype=dtype, device=resolved)
                v = torch.empty_like(k)
                outputs[batch.record_ids] = JaggedMigratedKVBatch(
                    record_ids=batch.record_ids,
                    migration_anchor_version=batch.migration_anchor_version,
                    served_kv_target=served_kv_target,
                    k=k,
                    v=v,
                    lengths=batch.lengths.to(resolved),
                    offsets=batch.offsets.to(resolved),
                )
        return cls(
            served_kv_target=served_kv_target,
            device=resolved,
            outputs=outputs,
        )

    @property
    def nbytes(self) -> int:
        return sum(output.nbytes for output in self.outputs.values())

    def acquire(
        self,
        capsule: JaggedMigrationCapsuleBatch,
    ) -> JaggedMigratedKVBatch:
        try:
            output = self.outputs[capsule.record_ids]
        except KeyError as exc:
            raise KeyError("capsule extent is absent from output pool") from exc
        if output.migration_anchor_version != capsule.migration_anchor_version:
            raise ValueError("output-pool migration anchor differs from capsule")
        if output.token_count != capsule.token_count:
            raise ValueError("output-pool token count differs from capsule")
        return output


if triton is not None:

    @triton.jit
    def _cohort_jagged_affine_kv_kernel(
        normed,
        weights,
        biases,
        output_k,
        output_v,
        token_count,
        input_width: tl.constexpr,
        output_width: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        layer = tl.program_id(0)
        row_offsets = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        output_offsets = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
        reduction_offsets = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start in range(0, input_width, BLOCK_K):
            current_reduction = start + reduction_offsets
            normed_values = tl.load(
                normed
                + layer * token_count * input_width
                + row_offsets[:, None] * input_width
                + current_reduction[None, :],
                mask=(row_offsets[:, None] < token_count)
                & (current_reduction[None, :] < input_width),
                other=0.0,
            )
            weight_values = tl.load(
                weights
                + layer * input_width * output_width
                + current_reduction[:, None] * output_width
                + output_offsets[None, :],
                mask=(current_reduction[:, None] < input_width)
                & (output_offsets[None, :] < output_width),
                other=0.0,
            )
            accumulator += tl.dot(normed_values, weight_values)
        bias = tl.load(
            biases + layer * output_width + output_offsets,
            mask=output_offsets < output_width,
            other=0.0,
        )
        values = accumulator + bias[None, :]
        kv_width = output_width // 2
        tl.store(
            output_k
            + layer * token_count * kv_width
            + row_offsets[:, None] * kv_width
            + output_offsets[None, :],
            values,
            mask=(row_offsets[:, None] < token_count)
            & (output_offsets[None, :] < kv_width),
        )
        value_offsets = output_offsets - kv_width
        tl.store(
            output_v
            + layer * token_count * kv_width
            + row_offsets[:, None] * kv_width
            + value_offsets[None, :],
            values,
            mask=(row_offsets[:, None] < token_count)
            & (output_offsets[None, :] >= kv_width)
            & (output_offsets[None, :] < output_width),
        )

else:
    _cohort_jagged_affine_kv_kernel = None


class JaggedMigrationOperator(Protocol):
    @property
    def name(self) -> str: ...

    def prepare_program(
        self,
        program: MigrationProgram,
        device: torch.device | str,
    ) -> MigrationProgram: ...

    def execute(
        self,
        program: MigrationProgram,
        capsule: JaggedMigrationCapsuleBatch,
    ) -> JaggedMigratedKVBatch: ...

    def execute_into(
        self,
        program: MigrationProgram,
        capsule: JaggedMigrationCapsuleBatch,
        destination: JaggedMigratedKVBatch,
    ) -> JaggedMigratedKVBatch: ...


def validate_jagged_destination(
    program: MigrationProgram,
    capsule: JaggedMigrationCapsuleBatch,
    destination: JaggedMigratedKVBatch,
) -> None:
    if destination.record_ids != capsule.record_ids:
        raise ValueError("destination and capsule record IDs differ")
    if destination.migration_anchor_version != capsule.migration_anchor_version:
        raise ValueError("destination and capsule anchors differ")
    if destination.served_kv_target != program.target_version:
        raise ValueError("destination and program targets differ")
    expected_shape = (
        program.num_layers,
        capsule.token_count,
        program.kv_width,
    )
    if destination.k.shape != expected_shape or destination.v.shape != expected_shape:
        raise ValueError("destination K/V shape differs from migration output")
    if destination.k.device != capsule.device:
        raise ValueError("destination and capsule must share a device")
    if destination.k.dtype != capsule.normed.dtype:
        raise ValueError("destination and capsule dtypes differ")


class PackedJaggedMigrationOperator:
    def __init__(self, execution_dtype: torch.dtype = torch.float16) -> None:
        if execution_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("packed operator dtype must be float16, bfloat16, or float32")
        self.execution_dtype = execution_dtype

    @property
    def name(self) -> str:
        dtype = str(self.execution_dtype).removeprefix("torch.")
        return f"packed_jagged_{dtype}"

    def prepare_program(
        self,
        program: MigrationProgram,
        device: torch.device | str,
    ) -> MigrationProgram:
        return program.to(device, dtype=self.execution_dtype)

    @torch.no_grad()
    def execute(
        self,
        program: MigrationProgram,
        capsule: JaggedMigrationCapsuleBatch,
    ) -> JaggedMigratedKVBatch:
        program.validate_capsule(capsule)
        destination = JaggedMigratedKVBatch(
            record_ids=capsule.record_ids,
            migration_anchor_version=capsule.migration_anchor_version,
            served_kv_target=program.target_version,
            k=torch.empty(
                program.num_layers,
                capsule.token_count,
                program.kv_width,
                dtype=capsule.normed.dtype,
                device=capsule.device,
            ),
            v=torch.empty(
                program.num_layers,
                capsule.token_count,
                program.kv_width,
                dtype=capsule.normed.dtype,
                device=capsule.device,
            ),
            lengths=capsule.lengths,
            offsets=capsule.offsets,
        )
        return self.execute_into(program, capsule, destination)

    @torch.no_grad()
    def execute_into(
        self,
        program: MigrationProgram,
        capsule: JaggedMigrationCapsuleBatch,
        destination: JaggedMigratedKVBatch,
    ) -> JaggedMigratedKVBatch:
        program.validate_capsule(capsule)
        validate_jagged_destination(program, capsule, destination)
        normed = capsule.normed.to(program.adapter.weights.dtype)
        bias = program.adapter.biases[:, None, :].expand(
            -1,
            capsule.token_count,
            -1,
        )
        projected = torch.baddbmm(bias, normed, program.adapter.weights)
        projected = projected.to(capsule.normed.dtype)
        width = projected.shape[-1] // 2
        destination.k.copy_(projected[..., :width])
        destination.v.copy_(projected[..., width:])
        return destination


class FusedJaggedMigrationOperator:
    def __init__(
        self,
        block_m: int = 32,
        block_n: int = 128,
        block_k: int = 64,
        num_warps: int = 8,
        num_stages: int = 3,
    ) -> None:
        if triton is None:
            raise RuntimeError("Triton is unavailable")
        if min(block_m, block_n, block_k, num_warps, num_stages) < 1:
            raise ValueError("fused operator launch parameters must be positive")
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages

    @property
    def name(self) -> str:
        return (
            f"fused_cohort_jagged_m{self.block_m}_n{self.block_n}_"
            f"k{self.block_k}_w{self.num_warps}_s{self.num_stages}"
        )

    def prepare_program(
        self,
        program: MigrationProgram,
        device: torch.device | str,
    ) -> MigrationProgram:
        return program.to(device, dtype=torch.float16)

    @torch.no_grad()
    def execute(
        self,
        program: MigrationProgram,
        capsule: JaggedMigrationCapsuleBatch,
    ) -> JaggedMigratedKVBatch:
        program.validate_capsule(capsule)
        destination = JaggedMigratedKVBatch(
            record_ids=capsule.record_ids,
            migration_anchor_version=capsule.migration_anchor_version,
            served_kv_target=program.target_version,
            k=torch.empty(
                program.num_layers,
                capsule.token_count,
                program.kv_width,
                dtype=torch.float16,
                device=capsule.device,
            ),
            v=torch.empty(
                program.num_layers,
                capsule.token_count,
                program.kv_width,
                dtype=torch.float16,
                device=capsule.device,
            ),
            lengths=capsule.lengths,
            offsets=capsule.offsets,
        )
        return self.execute_into(program, capsule, destination)

    @torch.no_grad()
    def execute_into(
        self,
        program: MigrationProgram,
        capsule: JaggedMigrationCapsuleBatch,
        destination: JaggedMigratedKVBatch,
    ) -> JaggedMigratedKVBatch:
        program.validate_capsule(capsule)
        validate_jagged_destination(program, capsule, destination)
        if capsule.device.type != "cuda":
            raise ValueError("fused jagged migration requires a CUDA capsule")
        if capsule.normed.dtype != torch.float16:
            raise ValueError("fused jagged migration requires FP16 capsules")
        if program.adapter.weights.dtype != torch.float16:
            raise ValueError("fused jagged migration requires FP16 program weights")
        if not capsule.normed.is_contiguous():
            raise ValueError("fused jagged migration requires contiguous capsules")
        if (
            not program.adapter.weights.is_contiguous()
            or not program.adapter.biases.is_contiguous()
        ):
            raise ValueError("fused jagged migration requires a contiguous program")
        layers, token_count, hidden = capsule.normed.shape
        output_width = program.adapter.weights.shape[-1]
        grid = (
            layers,
            triton.cdiv(token_count, self.block_m),
            triton.cdiv(output_width, self.block_n),
        )
        _cohort_jagged_affine_kv_kernel[grid](
            capsule.normed,
            program.adapter.weights,
            program.adapter.biases,
            destination.k,
            destination.v,
            token_count,
            input_width=hidden,
            output_width=output_width,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )
        return destination


@dataclass(frozen=True)
class JaggedCohortExecutionReport:
    batches: tuple[JaggedMigratedKVBatch, ...]
    metrics: CohortExecutionMetrics


@dataclass
class _PendingJaggedCudaBatch:
    done: torch.cuda.Event
    host_capsule: JaggedMigrationCapsuleBatch
    device_capsule: JaggedMigrationCapsuleBatch
    device_result: JaggedMigratedKVBatch
    host_result: JaggedMigratedKVBatch


class JaggedCohortStreamingExecutor:
    def __init__(
        self,
        program: MigrationProgram | Sequence[MigrationProgram],
        device: torch.device | str,
        max_inflight_batches: int = 3,
        pin_inputs: bool = True,
        operator: JaggedMigrationOperator | None = None,
        output_pool: PinnedJaggedKVOutputPool | None = None,
    ) -> None:
        device = torch.device(device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("executor device must be CPU or CUDA")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA executor requested but CUDA is unavailable")
        if max_inflight_batches < 1:
            raise ValueError("max_inflight_batches must be positive")
        self.device = device
        self.max_inflight_batches = max_inflight_batches
        self.pin_inputs = pin_inputs
        self.operator = operator or PackedJaggedMigrationOperator(torch.float32)
        programs = (program,) if isinstance(program, MigrationProgram) else tuple(program)
        if not programs:
            raise ValueError("at least one migration program is required")
        sources = [value.source_version for value in programs]
        if len(set(sources)) != len(sources):
            raise ValueError("migration program sources must be unique")
        targets = {value.target_version for value in programs}
        shapes = {
            (value.num_layers, value.input_width, value.kv_width)
            for value in programs
        }
        if len(targets) != 1 or len(shapes) != 1:
            raise ValueError("migration programs must share target and tensor shape")
        self.target_version = next(iter(targets))
        self.programs = {
            value.source_version: self.operator.prepare_program(value, device)
            for value in programs
        }
        self.program = self.programs[programs[0].source_version]
        self.output_pool = output_pool
        if output_pool is not None and output_pool.served_kv_target != self.target_version:
            raise ValueError("output pool and program target versions differ")
        self._h2d_stream = None
        self._compute_stream = None
        self._d2h_stream = None
        if device.type == "cuda":
            with torch.cuda.device(device):
                self._h2d_stream = torch.cuda.Stream(device=device)
                self._compute_stream = torch.cuda.Stream(device=device)
                self._d2h_stream = torch.cuda.Stream(device=device)

    def _program_for(
        self,
        capsule: JaggedMigrationCapsuleBatch,
    ) -> MigrationProgram:
        try:
            return self.programs[capsule.migration_anchor_version]
        except KeyError as exc:
            raise ValueError("no migration program matches capsule anchor") from exc

    def _account_capsule(
        self,
        capsule: JaggedMigrationCapsuleBatch,
        seen_record_ids: set[int],
    ) -> tuple[int, int]:
        overlap = seen_record_ids.intersection(capsule.record_ids)
        if overlap:
            raise ValueError("record_ids must be unique across a cohort execution")
        seen_record_ids.update(capsule.record_ids)
        return capsule.batch_size, capsule.token_count

    @torch.no_grad()
    def run(
        self,
        batches: Iterable[JaggedMigrationCapsuleBatch],
    ) -> JaggedCohortExecutionReport:
        if self.device.type == "cuda":
            return self._run_cuda(batches)
        return self._run_cpu(batches)

    @torch.no_grad()
    def _run_cpu(
        self,
        batches: Iterable[JaggedMigrationCapsuleBatch],
    ) -> JaggedCohortExecutionReport:
        started = time.perf_counter()
        results = []
        seen_record_ids: set[int] = set()
        batch_count = 0
        record_count = 0
        token_count = 0
        input_bytes = 0
        output_bytes = 0
        for capsule in batches:
            records, tokens = self._account_capsule(capsule, seen_record_ids)
            device_capsule = capsule.to(self.device)
            program = self._program_for(capsule)
            result = self.operator.execute(program, device_capsule)
            results.append(result)
            batch_count += 1
            record_count += records
            token_count += tokens
            input_bytes += capsule.nbytes
            output_bytes += result.nbytes
        elapsed = time.perf_counter() - started
        return JaggedCohortExecutionReport(
            batches=tuple(results),
            metrics=CohortExecutionMetrics(
                batch_count=batch_count,
                record_count=record_count,
                token_count=token_count,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                auto_pinned_batches=0,
                preallocated_output_batches=0,
                elapsed_seconds=elapsed,
            ),
        )

    @torch.no_grad()
    def _run_cuda(
        self,
        batches: Iterable[JaggedMigrationCapsuleBatch],
    ) -> JaggedCohortExecutionReport:
        started = time.perf_counter()
        results = []
        pending: deque[_PendingJaggedCudaBatch] = deque()
        seen_record_ids: set[int] = set()
        batch_count = 0
        record_count = 0
        token_count = 0
        input_bytes = 0
        output_bytes = 0
        auto_pinned_batches = 0
        preallocated_output_batches = 0
        with torch.cuda.device(self.device):
            if (
                self._h2d_stream is None
                or self._compute_stream is None
                or self._d2h_stream is None
            ):
                raise RuntimeError("CUDA streams are not initialized")
            h2d_stream = self._h2d_stream
            compute_stream = self._compute_stream
            d2h_stream = self._d2h_stream
            for capsule in batches:
                if capsule.device.type != "cpu":
                    raise ValueError("CUDA executor requires CPU-resident capsules")
                records, tokens = self._account_capsule(capsule, seen_record_ids)
                host_capsule = capsule
                if self.pin_inputs and not capsule.is_pinned:
                    host_capsule = capsule.pin_memory()
                    auto_pinned_batches += 1
                h2d_done = torch.cuda.Event()
                with torch.cuda.stream(h2d_stream):
                    device_capsule = host_capsule.to(self.device, non_blocking=True)
                    h2d_done.record(h2d_stream)
                compute_done = torch.cuda.Event()
                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_event(h2d_done)
                    program = self._program_for(host_capsule)
                    device_result = self.operator.execute(program, device_capsule)
                    compute_done.record(compute_stream)
                if self.output_pool is None:
                    host_k = torch.empty(
                        device_result.k.shape,
                        dtype=device_result.k.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    host_v = torch.empty_like(host_k, pin_memory=True)
                    host_result = JaggedMigratedKVBatch(
                        record_ids=host_capsule.record_ids,
                        migration_anchor_version=host_capsule.migration_anchor_version,
                        served_kv_target=self.target_version,
                        k=host_k,
                        v=host_v,
                        lengths=host_capsule.lengths,
                        offsets=host_capsule.offsets,
                    )
                else:
                    host_result = self.output_pool.acquire(host_capsule)
                    host_k = host_result.k
                    host_v = host_result.v
                    if host_k.shape != device_result.k.shape:
                        raise ValueError("preallocated K shape differs from device result")
                    if host_v.shape != device_result.v.shape:
                        raise ValueError("preallocated V shape differs from device result")
                    if host_k.dtype != device_result.k.dtype:
                        raise ValueError("preallocated output dtype differs from device result")
                    preallocated_output_batches += 1
                d2h_done = torch.cuda.Event()
                with torch.cuda.stream(d2h_stream):
                    d2h_stream.wait_event(compute_done)
                    host_k.copy_(device_result.k, non_blocking=True)
                    host_v.copy_(device_result.v, non_blocking=True)
                    d2h_done.record(d2h_stream)
                pending.append(
                    _PendingJaggedCudaBatch(
                        done=d2h_done,
                        host_capsule=host_capsule,
                        device_capsule=device_capsule,
                        device_result=device_result,
                        host_result=host_result,
                    )
                )
                batch_count += 1
                record_count += records
                token_count += tokens
                input_bytes += capsule.nbytes
                output_bytes += host_result.nbytes
                if len(pending) >= self.max_inflight_batches:
                    completed = pending.popleft()
                    completed.done.synchronize()
                    results.append(completed.host_result)
            while pending:
                completed = pending.popleft()
                completed.done.synchronize()
                results.append(completed.host_result)
        elapsed = time.perf_counter() - started
        return JaggedCohortExecutionReport(
            batches=tuple(results),
            metrics=CohortExecutionMetrics(
                batch_count=batch_count,
                record_count=record_count,
                token_count=token_count,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                auto_pinned_batches=auto_pinned_batches,
                preallocated_output_batches=preallocated_output_batches,
                elapsed_seconds=elapsed,
            ),
        )


@dataclass
class _PendingHBMJaggedBatch:
    done: torch.cuda.Event
    host_capsule: JaggedMigrationCapsuleBatch
    device_capsule: JaggedMigrationCapsuleBatch
    device_result: JaggedMigratedKVBatch


class HBMJaggedCohortStreamingExecutor:
    def __init__(
        self,
        program: MigrationProgram | Sequence[MigrationProgram],
        device: torch.device | str,
        output_pool: DeviceJaggedKVOutputPool,
        max_inflight_batches: int = 3,
        pin_inputs: bool = True,
        operator: JaggedMigrationOperator | None = None,
    ) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("HBM executor requires a CUDA device")
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        if output_pool.device != device:
            raise ValueError("output pool and executor devices differ")
        if max_inflight_batches < 1:
            raise ValueError("max_inflight_batches must be positive")
        self.device = device
        self.output_pool = output_pool
        self.max_inflight_batches = max_inflight_batches
        self.pin_inputs = pin_inputs
        self.operator = operator or FusedJaggedMigrationOperator()
        programs = (program,) if isinstance(program, MigrationProgram) else tuple(program)
        if not programs:
            raise ValueError("at least one migration program is required")
        sources = [value.source_version for value in programs]
        if len(set(sources)) != len(sources):
            raise ValueError("migration program sources must be unique")
        targets = {value.target_version for value in programs}
        shapes = {
            (value.num_layers, value.input_width, value.kv_width)
            for value in programs
        }
        if len(targets) != 1 or len(shapes) != 1:
            raise ValueError("migration programs must share target and tensor shape")
        self.target_version = next(iter(targets))
        if output_pool.served_kv_target != self.target_version:
            raise ValueError("output pool and program target versions differ")
        self.programs = {
            value.source_version: self.operator.prepare_program(value, device)
            for value in programs
        }
        with torch.cuda.device(device):
            self._h2d_stream = torch.cuda.Stream(device=device)
            self._compute_stream = torch.cuda.Stream(device=device)

    def _program_for(
        self,
        capsule: JaggedMigrationCapsuleBatch,
    ) -> MigrationProgram:
        try:
            return self.programs[capsule.migration_anchor_version]
        except KeyError as exc:
            raise ValueError("no migration program matches capsule anchor") from exc

    @torch.no_grad()
    def run(
        self,
        batches: Iterable[JaggedMigrationCapsuleBatch],
    ) -> JaggedCohortExecutionReport:
        started = time.perf_counter()
        results = []
        pending: deque[_PendingHBMJaggedBatch] = deque()
        seen_record_ids: set[int] = set()
        batch_count = 0
        record_count = 0
        token_count = 0
        input_bytes = 0
        output_bytes = 0
        auto_pinned_batches = 0
        with torch.cuda.device(self.device):
            for capsule in batches:
                if capsule.device.type != "cpu":
                    raise ValueError("HBM executor requires CPU-resident capsules")
                overlap = seen_record_ids.intersection(capsule.record_ids)
                if overlap:
                    raise ValueError("record_ids must be unique across execution")
                seen_record_ids.update(capsule.record_ids)
                host_capsule = capsule
                if self.pin_inputs and not capsule.is_pinned:
                    host_capsule = capsule.pin_memory()
                    auto_pinned_batches += 1
                destination = self.output_pool.acquire(host_capsule)
                h2d_done = torch.cuda.Event()
                with torch.cuda.stream(self._h2d_stream):
                    device_capsule = JaggedMigrationCapsuleBatch(
                        record_ids=host_capsule.record_ids,
                        migration_anchor_version=host_capsule.migration_anchor_version,
                        normed=host_capsule.normed.to(
                            self.device,
                            non_blocking=True,
                        ),
                        lengths=destination.lengths,
                        offsets=destination.offsets,
                    )
                    h2d_done.record(self._h2d_stream)
                compute_done = torch.cuda.Event()
                with torch.cuda.stream(self._compute_stream):
                    self._compute_stream.wait_event(h2d_done)
                    program = self._program_for(host_capsule)
                    device_result = self.operator.execute_into(
                        program,
                        device_capsule,
                        destination,
                    )
                    compute_done.record(self._compute_stream)
                pending.append(
                    _PendingHBMJaggedBatch(
                        done=compute_done,
                        host_capsule=host_capsule,
                        device_capsule=device_capsule,
                        device_result=device_result,
                    )
                )
                batch_count += 1
                record_count += capsule.batch_size
                token_count += capsule.token_count
                input_bytes += (
                    capsule.normed.numel() * capsule.normed.element_size()
                )
                output_bytes += (
                    destination.k.numel() * destination.k.element_size()
                    + destination.v.numel() * destination.v.element_size()
                )
                if len(pending) >= self.max_inflight_batches:
                    completed = pending.popleft()
                    completed.done.synchronize()
                    results.append(completed.device_result)
            while pending:
                completed = pending.popleft()
                completed.done.synchronize()
                results.append(completed.device_result)
        elapsed = time.perf_counter() - started
        return JaggedCohortExecutionReport(
            batches=tuple(results),
            metrics=CohortExecutionMetrics(
                batch_count=batch_count,
                record_count=record_count,
                token_count=token_count,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                auto_pinned_batches=auto_pinned_batches,
                preallocated_output_batches=batch_count,
                elapsed_seconds=elapsed,
            ),
        )


@dataclass(frozen=True)
class MultiGPUJaggedExecutionReport:
    batches: tuple[JaggedMigratedKVBatch, ...]
    metrics: MultiGPUExecutionMetrics


class MultiGPUJaggedCohortExecutor:
    def __init__(
        self,
        program: MigrationProgram | Sequence[MigrationProgram],
        devices: Sequence[torch.device | str],
        max_inflight_batches: int = 3,
        pin_inputs: bool = True,
        operator: JaggedMigrationOperator | None = None,
        output_pool: PinnedJaggedKVOutputPool | None = None,
        partition_strategy: str = "greedy_lpt",
    ) -> None:
        resolved = tuple(torch.device(device) for device in devices)
        if not resolved:
            raise ValueError("at least one CUDA device is required")
        if any(device.type != "cuda" for device in resolved):
            raise ValueError("multi-GPU executor requires CUDA devices")
        if len(set(resolved)) != len(resolved):
            raise ValueError("multi-GPU executor devices must be unique")
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        if any(
            device.index is not None and device.index >= torch.cuda.device_count()
            for device in resolved
        ):
            raise ValueError("requested CUDA device is unavailable")
        if partition_strategy not in {
            "round_robin",
            "greedy_input_order",
            "greedy_lpt",
        }:
            raise ValueError("unsupported partition strategy")
        programs = (program,) if isinstance(program, MigrationProgram) else tuple(program)
        if not programs:
            raise ValueError("at least one migration program is required")
        sources = [value.source_version for value in programs]
        if len(set(sources)) != len(sources):
            raise ValueError("migration program sources must be unique")
        targets = {value.target_version for value in programs}
        shapes = {
            (value.num_layers, value.input_width, value.kv_width)
            for value in programs
        }
        if len(targets) != 1 or len(shapes) != 1:
            raise ValueError("migration programs must share target and tensor shape")
        shared_operator = operator or PackedJaggedMigrationOperator(torch.float32)
        self.programs = programs
        self.program = programs[0]
        self.devices = resolved
        self.partition_strategy = partition_strategy
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=len(resolved))
        self.executors = tuple(
            JaggedCohortStreamingExecutor(
                programs,
                device=device,
                max_inflight_batches=max_inflight_batches,
                pin_inputs=pin_inputs,
                operator=shared_operator,
                output_pool=output_pool,
            )
            for device in resolved
        )

    def close(self) -> None:
        if not self._closed:
            self._pool.shutdown(wait=True)
            self._closed = True

    def __enter__(self) -> MultiGPUJaggedCohortExecutor:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _work_bytes(self, capsule: JaggedMigrationCapsuleBatch) -> int:
        output_elements = (
            self.program.num_layers
            * capsule.token_count
            * 2
            * self.program.kv_width
        )
        output_bytes = output_elements * capsule.normed.element_size()
        return capsule.nbytes + output_bytes

    def _partition(
        self,
        batches: tuple[JaggedMigrationCapsuleBatch, ...],
    ) -> tuple[tuple[tuple[int, JaggedMigrationCapsuleBatch], ...], ...]:
        assignments: list[list[tuple[int, JaggedMigrationCapsuleBatch]]] = [
            [] for _ in self.devices
        ]
        if self.partition_strategy == "round_robin":
            for index, capsule in enumerate(batches):
                assignments[index % len(self.devices)].append((index, capsule))
            return tuple(tuple(assignment) for assignment in assignments)
        loads = [0 for _ in self.devices]
        indexed = list(enumerate(batches))
        if self.partition_strategy == "greedy_lpt":
            indexed.sort(key=lambda value: (-self._work_bytes(value[1]), value[0]))
        for index, capsule in indexed:
            worker = min(range(len(loads)), key=lambda value: (loads[value], value))
            assignments[worker].append((index, capsule))
            loads[worker] += self._work_bytes(capsule)
        return tuple(tuple(assignment) for assignment in assignments)

    @torch.no_grad()
    def run(
        self,
        batches: Iterable[JaggedMigrationCapsuleBatch],
    ) -> MultiGPUJaggedExecutionReport:
        if self._closed:
            raise RuntimeError("multi-GPU executor is closed")
        batches = tuple(batches)
        seen_record_ids: set[int] = set()
        for capsule in batches:
            overlap = seen_record_ids.intersection(capsule.record_ids)
            if overlap:
                raise ValueError("record_ids must be unique across a cohort execution")
            seen_record_ids.update(capsule.record_ids)
        assignments = self._partition(batches)
        started = time.perf_counter()

        def run_worker(worker: int):
            assignment = assignments[worker]
            report = self.executors[worker].run(
                capsule for _, capsule in assignment
            )
            return worker, assignment, report

        futures = [
            self._pool.submit(run_worker, worker)
            for worker in range(len(self.devices))
        ]
        worker_results = [future.result() for future in futures]
        elapsed = time.perf_counter() - started
        ordered: list[JaggedMigratedKVBatch | None] = [None] * len(batches)
        device_metrics = []
        assigned_loads = []
        for worker, assignment, report in sorted(worker_results):
            if len(assignment) != len(report.batches):
                raise RuntimeError("worker result count differs from its assignment")
            for (index, _), result in zip(assignment, report.batches, strict=True):
                ordered[index] = result
            assigned_work_bytes = sum(
                self._work_bytes(capsule) for _, capsule in assignment
            )
            assigned_loads.append(assigned_work_bytes)
            device_metrics.append(
                DeviceExecutionMetrics(
                    device=str(self.devices[worker]),
                    assigned_work_bytes=assigned_work_bytes,
                    execution=report.metrics,
                )
            )
        if any(result is None for result in ordered):
            raise RuntimeError("multi-GPU execution produced incomplete output")
        mean_load = sum(assigned_loads) / len(assigned_loads)
        load_imbalance = (
            max(assigned_loads) / mean_load - 1.0 if mean_load > 0 else 0.0
        )
        return MultiGPUJaggedExecutionReport(
            batches=tuple(result for result in ordered if result is not None),
            metrics=MultiGPUExecutionMetrics(
                device_count=len(self.devices),
                program_count=len(self.programs),
                batch_count=sum(
                    value.execution.batch_count for value in device_metrics
                ),
                record_count=sum(
                    value.execution.record_count for value in device_metrics
                ),
                token_count=sum(
                    value.execution.token_count for value in device_metrics
                ),
                input_bytes=sum(
                    value.execution.input_bytes for value in device_metrics
                ),
                output_bytes=sum(
                    value.execution.output_bytes for value in device_metrics
                ),
                program_replica_bytes=sum(
                    sum(program.nbytes for program in executor.programs.values())
                    for executor in self.executors
                ),
                elapsed_seconds=elapsed,
                load_imbalance=load_imbalance,
                partition_strategy=self.partition_strategy,
                cohort_batch_counts={
                    source: sum(
                        batch.migration_anchor_version == source
                        for batch in batches
                    )
                    for source in sorted(
                        {
                            batch.migration_anchor_version
                            for batch in batches
                        }
                    )
                },
                cohort_record_counts={
                    source: sum(
                        batch.batch_size
                        for batch in batches
                        if batch.migration_anchor_version == source
                    )
                    for source in sorted(
                        {
                            batch.migration_anchor_version
                            for batch in batches
                        }
                    )
                },
                devices=tuple(device_metrics),
            ),
        )


class MultiGPUHBMJaggedCohortExecutor:
    def __init__(
        self,
        program: MigrationProgram | Sequence[MigrationProgram],
        devices: Sequence[torch.device | str],
        batches: Sequence[JaggedMigrationCapsuleBatch],
        max_inflight_batches: int = 3,
        pin_inputs: bool = True,
        operator: JaggedMigrationOperator | None = None,
        partition_strategy: str = "greedy_lpt",
    ) -> None:
        resolved = tuple(torch.device(device) for device in devices)
        if not resolved:
            raise ValueError("at least one CUDA device is required")
        if any(device.type != "cuda" for device in resolved):
            raise ValueError("multi-GPU executor requires CUDA devices")
        if len(set(resolved)) != len(resolved):
            raise ValueError("multi-GPU executor devices must be unique")
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        if partition_strategy not in {
            "round_robin",
            "greedy_input_order",
            "greedy_lpt",
        }:
            raise ValueError("unsupported partition strategy")
        programs = (program,) if isinstance(program, MigrationProgram) else tuple(program)
        if not programs:
            raise ValueError("at least one migration program is required")
        sources = [value.source_version for value in programs]
        if len(set(sources)) != len(sources):
            raise ValueError("migration program sources must be unique")
        targets = {value.target_version for value in programs}
        shapes = {
            (value.num_layers, value.input_width, value.kv_width)
            for value in programs
        }
        if len(targets) != 1 or len(shapes) != 1:
            raise ValueError("migration programs must share target and tensor shape")
        batches = tuple(batches)
        if not batches:
            raise ValueError("HBM executor requires at least one batch")
        if len(batches) < len(resolved):
            resolved = resolved[: len(batches)]
        seen_record_ids: set[int] = set()
        for capsule in batches:
            if capsule.device.type != "cpu":
                raise ValueError("HBM executor requires CPU-resident capsules")
            overlap = seen_record_ids.intersection(capsule.record_ids)
            if overlap:
                raise ValueError("record_ids must be unique across execution")
            seen_record_ids.update(capsule.record_ids)
        self.programs = programs
        self.program = programs[0]
        self.devices = resolved
        self.batches = batches
        self.partition_strategy = partition_strategy
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=len(resolved))
        self.assignments = self._partition()
        selected_operator = operator or FusedJaggedMigrationOperator()
        self.output_pools = tuple(
            DeviceJaggedKVOutputPool.allocate(
                tuple(capsule for _, capsule in assignment),
                served_kv_target=self.program.target_version,
                num_layers=self.program.num_layers,
                kv_width=self.program.kv_width,
                device=device,
                dtype=torch.float16,
            )
            for device, assignment in zip(
                self.devices,
                self.assignments,
                strict=True,
            )
        )
        self.executors = tuple(
            HBMJaggedCohortStreamingExecutor(
                programs,
                device=device,
                output_pool=output_pool,
                max_inflight_batches=max_inflight_batches,
                pin_inputs=pin_inputs,
                operator=selected_operator,
            )
            for device, output_pool in zip(
                self.devices,
                self.output_pools,
                strict=True,
            )
        )

    @property
    def output_pool_nbytes(self) -> int:
        return sum(pool.nbytes for pool in self.output_pools)

    def close(self) -> None:
        if not self._closed:
            self._pool.shutdown(wait=True)
            self._closed = True

    def __enter__(self) -> MultiGPUHBMJaggedCohortExecutor:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _work_bytes(self, capsule: JaggedMigrationCapsuleBatch) -> int:
        output_elements = (
            self.program.num_layers
            * capsule.token_count
            * 2
            * self.program.kv_width
        )
        output_bytes = output_elements * capsule.normed.element_size()
        input_bytes = capsule.normed.numel() * capsule.normed.element_size()
        return input_bytes + output_bytes

    def _partition(
        self,
    ) -> tuple[tuple[tuple[int, JaggedMigrationCapsuleBatch], ...], ...]:
        assignments: list[list[tuple[int, JaggedMigrationCapsuleBatch]]] = [
            [] for _ in self.devices
        ]
        if self.partition_strategy == "round_robin":
            for index, capsule in enumerate(self.batches):
                assignments[index % len(self.devices)].append((index, capsule))
            return tuple(tuple(assignment) for assignment in assignments)
        loads = [0 for _ in self.devices]
        indexed = list(enumerate(self.batches))
        if self.partition_strategy == "greedy_lpt":
            indexed.sort(key=lambda value: (-self._work_bytes(value[1]), value[0]))
        for index, capsule in indexed:
            worker = min(range(len(loads)), key=lambda value: (loads[value], value))
            assignments[worker].append((index, capsule))
            loads[worker] += self._work_bytes(capsule)
        return tuple(tuple(assignment) for assignment in assignments)

    @torch.no_grad()
    def run(self) -> MultiGPUJaggedExecutionReport:
        if self._closed:
            raise RuntimeError("multi-GPU executor is closed")
        started = time.perf_counter()

        def run_worker(worker: int):
            assignment = self.assignments[worker]
            report = self.executors[worker].run(
                capsule for _, capsule in assignment
            )
            return worker, assignment, report

        futures = [
            self._pool.submit(run_worker, worker)
            for worker in range(len(self.devices))
        ]
        worker_results = [future.result() for future in futures]
        elapsed = time.perf_counter() - started
        ordered: list[JaggedMigratedKVBatch | None] = [None] * len(self.batches)
        device_metrics = []
        assigned_loads = []
        for worker, assignment, report in sorted(worker_results):
            if len(assignment) != len(report.batches):
                raise RuntimeError("worker result count differs from its assignment")
            for (index, _), result in zip(assignment, report.batches, strict=True):
                ordered[index] = result
            assigned_work_bytes = sum(
                self._work_bytes(capsule) for _, capsule in assignment
            )
            assigned_loads.append(assigned_work_bytes)
            device_metrics.append(
                DeviceExecutionMetrics(
                    device=str(self.devices[worker]),
                    assigned_work_bytes=assigned_work_bytes,
                    execution=report.metrics,
                )
            )
        if any(result is None for result in ordered):
            raise RuntimeError("multi-GPU execution produced incomplete output")
        mean_load = sum(assigned_loads) / len(assigned_loads)
        load_imbalance = (
            max(assigned_loads) / mean_load - 1.0 if mean_load > 0 else 0.0
        )
        sources = sorted(
            {batch.migration_anchor_version for batch in self.batches}
        )
        return MultiGPUJaggedExecutionReport(
            batches=tuple(result for result in ordered if result is not None),
            metrics=MultiGPUExecutionMetrics(
                device_count=len(self.devices),
                program_count=len(self.programs),
                batch_count=sum(
                    value.execution.batch_count for value in device_metrics
                ),
                record_count=sum(
                    value.execution.record_count for value in device_metrics
                ),
                token_count=sum(
                    value.execution.token_count for value in device_metrics
                ),
                input_bytes=sum(
                    value.execution.input_bytes for value in device_metrics
                ),
                output_bytes=sum(
                    value.execution.output_bytes for value in device_metrics
                ),
                program_replica_bytes=sum(
                    sum(program.nbytes for program in executor.programs.values())
                    for executor in self.executors
                ),
                elapsed_seconds=elapsed,
                load_imbalance=load_imbalance,
                partition_strategy=self.partition_strategy,
                cohort_batch_counts={
                    source: sum(
                        batch.migration_anchor_version == source
                        for batch in self.batches
                    )
                    for source in sources
                },
                cohort_record_counts={
                    source: sum(
                        batch.batch_size
                        for batch in self.batches
                        if batch.migration_anchor_version == source
                    )
                    for source in sources
                },
                devices=tuple(device_metrics),
            ),
        )
