from __future__ import annotations

from typing import Protocol

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from ..models import HSTUKVCache
from .capsule import (
    ContiguousKVOutputExtent,
    MigratedKVBatch,
    MigrationCapsuleBatch,
)
from .low_rank import CompiledCacheAdapter, migrate_compiled_cache_capsule
from .program import MigrationProgram, execute_migration_program

if triton is not None:

    @triton.jit
    def _fused_affine_kv_kernel(
        normed,
        weights,
        biases,
        lengths,
        output_k,
        output_v,
        rows,
        sequence_width,
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
                + layer * rows * input_width
                + row_offsets[:, None] * input_width
                + current_reduction[None, :],
                mask=(row_offsets[:, None] < rows)
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
        record_offsets = row_offsets // sequence_width
        token_offsets = row_offsets - record_offsets * sequence_width
        valid_lengths = tl.load(
            lengths + record_offsets,
            mask=row_offsets < rows,
            other=0,
        )
        valid = (row_offsets < rows) & (token_offsets < valid_lengths)
        values = tl.where(
            valid[:, None],
            accumulator + bias[None, :],
            0.0,
        )
        kv_width = output_width // 2
        tl.store(
            output_k
            + layer * rows * kv_width
            + row_offsets[:, None] * kv_width
            + output_offsets[None, :],
            values,
            mask=(row_offsets[:, None] < rows)
            & (output_offsets[None, :] < kv_width),
        )
        value_offsets = output_offsets - kv_width
        tl.store(
            output_v
            + layer * rows * kv_width
            + row_offsets[:, None] * kv_width
            + value_offsets[None, :],
            values,
            mask=(row_offsets[:, None] < rows)
            & (output_offsets[None, :] >= kv_width)
            & (output_offsets[None, :] < output_width),
        )

    @triton.jit
    def _fused_affine_extent_kernel(
        normed,
        weights,
        biases,
        lengths,
        offsets,
        output_k,
        output_v,
        rows,
        sequence_width,
        output_tokens,
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
        record_offsets = row_offsets // sequence_width
        token_offsets = row_offsets - record_offsets * sequence_width
        valid_lengths = tl.load(
            lengths + record_offsets,
            mask=row_offsets < rows,
            other=0,
        )
        valid = (row_offsets < rows) & (token_offsets < valid_lengths)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start in range(0, input_width, BLOCK_K):
            current_reduction = start + reduction_offsets
            normed_values = tl.load(
                normed
                + layer * rows * input_width
                + row_offsets[:, None] * input_width
                + current_reduction[None, :],
                mask=valid[:, None]
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
        compact_rows = tl.load(
            offsets + record_offsets,
            mask=row_offsets < rows,
            other=0,
        ) + token_offsets
        valid = valid & (compact_rows >= 0) & (compact_rows < output_tokens)
        values = accumulator + bias[None, :]
        kv_width = output_width // 2
        tl.store(
            output_k
            + layer * output_tokens * kv_width
            + compact_rows[:, None] * kv_width
            + output_offsets[None, :],
            values,
            mask=valid[:, None] & (output_offsets[None, :] < kv_width),
        )
        value_offsets = output_offsets - kv_width
        tl.store(
            output_v
            + layer * output_tokens * kv_width
            + compact_rows[:, None] * kv_width
            + value_offsets[None, :],
            values,
            mask=valid[:, None]
            & (output_offsets[None, :] >= kv_width)
            & (output_offsets[None, :] < output_width),
        )

else:
    _fused_affine_kv_kernel = None
    _fused_affine_extent_kernel = None


def validate_contiguous_output_extent(
    program: MigrationProgram,
    capsule: MigrationCapsuleBatch,
    destination: ContiguousKVOutputExtent,
    check_metadata_values: bool = False,
) -> None:
    if destination.record_ids != capsule.record_ids:
        raise ValueError("destination and capsule record IDs differ")
    if destination.migration_anchor_version != capsule.migration_anchor_version:
        raise ValueError("destination and capsule anchors differ")
    if destination.served_kv_target != program.target_version:
        raise ValueError("destination and program targets differ")
    if destination.k.ndim != 3 or destination.v.ndim != 3:
        raise ValueError("destination K/V must have shape [layers, tokens, width]")
    if destination.k.shape != destination.v.shape:
        raise ValueError("destination K and V shapes differ")
    if destination.k.shape[0] != program.num_layers:
        raise ValueError("destination and program depths differ")
    if destination.k.shape[2] != program.kv_width:
        raise ValueError("destination and program K/V widths differ")
    if destination.k.shape[1] < 1:
        raise ValueError("destination must contain at least one token")
    if destination.token_count != destination.k.shape[1]:
        raise ValueError("destination token count metadata is invalid")
    if destination.lengths.shape != capsule.lengths.shape:
        raise ValueError("destination and capsule length shapes differ")
    if destination.offsets.shape != (capsule.batch_size + 1,):
        raise ValueError("destination offsets have the wrong shape")
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if destination.lengths.dtype not in integer_dtypes:
        raise ValueError("destination lengths must have an integer dtype")
    if destination.offsets.dtype not in {torch.int32, torch.int64}:
        raise ValueError("destination offsets must be int32 or int64")
    tensors = (
        destination.k,
        destination.v,
        destination.lengths,
        destination.offsets,
    )
    if any(value.device != capsule.device for value in tensors):
        raise ValueError("destination and capsule must share a device")
    if destination.k.dtype != capsule.normed.dtype:
        raise ValueError("destination and capsule dtypes differ")
    if destination.v.dtype != destination.k.dtype:
        raise ValueError("destination K and V dtypes differ")
    if not destination.k.is_contiguous() or not destination.v.is_contiguous():
        raise ValueError("destination K and V must be contiguous")
    if (
        not destination.lengths.is_contiguous()
        or not destination.offsets.is_contiguous()
    ):
        raise ValueError("destination metadata must be contiguous")
    write_storages = {
        destination.k.untyped_storage().data_ptr(),
        destination.v.untyped_storage().data_ptr(),
    }
    if len(write_storages) != 2:
        raise ValueError("destination K and V must not share storage")
    read_tensors = (
        capsule.normed,
        capsule.lengths,
        program.adapter.weights,
        program.adapter.biases,
        destination.lengths,
        destination.offsets,
    )
    if any(
        value.untyped_storage().data_ptr() in write_storages
        for value in read_tensors
    ):
        raise ValueError("destination storage must not alias operator inputs")
    if check_metadata_values:
        if bool(torch.any(capsule.lengths < 0)) or bool(
            torch.any(capsule.lengths > capsule.seq_len)
        ):
            raise ValueError("capsule lengths exceed the dense sequence width")
        offsets = torch.cat(
            (
                torch.zeros(
                    1,
                    dtype=destination.offsets.dtype,
                    device=capsule.device,
                ),
                capsule.lengths.to(destination.offsets.dtype).cumsum(0),
            )
        )
        if not torch.equal(destination.lengths, capsule.lengths):
            raise ValueError("destination and capsule lengths differ")
        if not torch.equal(destination.offsets, offsets):
            raise ValueError("destination offsets do not match lengths")
        if destination.k.shape[1] != int(offsets[-1].item()):
            raise ValueError("destination token count does not match lengths")


def _copy_dense_cache_to_extent(
    cache: HSTUKVCache,
    lengths: torch.Tensor,
    destination: ContiguousKVOutputExtent,
) -> None:
    positions = torch.arange(cache.seq_len, device=lengths.device)
    valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
    destination.k.copy_(cache.k[:, valid])
    destination.v.copy_(cache.v[:, valid])


class MigrationOperator(Protocol):
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
        capsule: MigrationCapsuleBatch,
    ) -> MigratedKVBatch: ...

    def execute_into(
        self,
        program: MigrationProgram,
        capsule: MigrationCapsuleBatch,
        destination: ContiguousKVOutputExtent,
    ) -> ContiguousKVOutputExtent: ...


class ReferenceMigrationOperator:
    @property
    def name(self) -> str:
        return "reference_fp32"

    def prepare_program(
        self,
        program: MigrationProgram,
        device: torch.device | str,
    ) -> MigrationProgram:
        return program.to(device, dtype=torch.float32)

    @torch.no_grad()
    def execute(
        self,
        program: MigrationProgram,
        capsule: MigrationCapsuleBatch,
    ) -> MigratedKVBatch:
        return execute_migration_program(program, capsule)

    @torch.no_grad()
    def execute_into(
        self,
        program: MigrationProgram,
        capsule: MigrationCapsuleBatch,
        destination: ContiguousKVOutputExtent,
    ) -> ContiguousKVOutputExtent:
        program.validate_capsule(capsule)
        validate_contiguous_output_extent(program, capsule, destination)
        cache = migrate_compiled_cache_capsule(capsule, program.adapter)
        _copy_dense_cache_to_extent(cache, capsule.lengths, destination)
        return destination


class PackedMigrationOperator:
    def __init__(self, execution_dtype: torch.dtype = torch.float16) -> None:
        if execution_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("packed operator dtype must be float16, bfloat16, or float32")
        self.execution_dtype = execution_dtype

    @property
    def name(self) -> str:
        return f"packed_{str(self.execution_dtype).removeprefix('torch.')}"

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
        capsule: MigrationCapsuleBatch,
    ) -> MigratedKVBatch:
        program.validate_capsule(capsule)
        cache = migrate_packed_cache_capsule(capsule, program.adapter)
        return MigratedKVBatch(
            record_ids=capsule.record_ids,
            migration_anchor_version=capsule.migration_anchor_version,
            served_kv_target=program.target_version,
            cache=cache,
            lengths=capsule.lengths,
        )

    @torch.no_grad()
    def execute_into(
        self,
        program: MigrationProgram,
        capsule: MigrationCapsuleBatch,
        destination: ContiguousKVOutputExtent,
    ) -> ContiguousKVOutputExtent:
        program.validate_capsule(capsule)
        validate_contiguous_output_extent(program, capsule, destination)
        cache = migrate_packed_cache_capsule(capsule, program.adapter)
        _copy_dense_cache_to_extent(cache, capsule.lengths, destination)
        return destination


class FusedMigrationOperator:
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
            f"fused_triton_m{self.block_m}_n{self.block_n}_"
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
        capsule: MigrationCapsuleBatch,
    ) -> MigratedKVBatch:
        program.validate_capsule(capsule)
        if capsule.device.type != "cuda":
            raise ValueError("fused migration requires a CUDA capsule")
        if capsule.normed.dtype != torch.float16:
            raise ValueError("fused migration requires FP16 capsules")
        if program.adapter.weights.dtype != torch.float16:
            raise ValueError("fused migration requires FP16 program weights")
        if not capsule.normed.is_contiguous():
            raise ValueError("fused migration requires contiguous capsules")
        if (
            not program.adapter.weights.is_contiguous()
            or not program.adapter.biases.is_contiguous()
        ):
            raise ValueError("fused migration requires a contiguous program")
        layers, batch, sequence, hidden = capsule.normed.shape
        output_width = program.adapter.weights.shape[-1]
        width = output_width // 2
        rows = batch * sequence
        output_k = torch.empty(
            layers,
            batch,
            sequence,
            width,
            device=capsule.device,
            dtype=torch.float16,
        )
        output_v = torch.empty_like(output_k)
        grid = (
            layers,
            triton.cdiv(rows, self.block_m),
            triton.cdiv(output_width, self.block_n),
        )
        _fused_affine_kv_kernel[grid](
            capsule.normed,
            program.adapter.weights,
            program.adapter.biases,
            capsule.lengths,
            output_k,
            output_v,
            rows,
            sequence,
            input_width=hidden,
            output_width=output_width,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )
        return MigratedKVBatch(
            record_ids=capsule.record_ids,
            migration_anchor_version=capsule.migration_anchor_version,
            served_kv_target=program.target_version,
            cache=HSTUKVCache(
                k=output_k,
                v=output_v,
                seq_len=sequence,
            ),
            lengths=capsule.lengths,
        )

    @torch.no_grad()
    def execute_into(
        self,
        program: MigrationProgram,
        capsule: MigrationCapsuleBatch,
        destination: ContiguousKVOutputExtent,
    ) -> ContiguousKVOutputExtent:
        program.validate_capsule(capsule)
        validate_contiguous_output_extent(program, capsule, destination)
        if capsule.device.type != "cuda":
            raise ValueError("fused migration requires a CUDA capsule")
        if capsule.normed.dtype != torch.float16:
            raise ValueError("fused migration requires FP16 capsules")
        if program.adapter.weights.dtype != torch.float16:
            raise ValueError("fused migration requires FP16 program weights")
        if not capsule.normed.is_contiguous():
            raise ValueError("fused migration requires contiguous capsules")
        if (
            not program.adapter.weights.is_contiguous()
            or not program.adapter.biases.is_contiguous()
        ):
            raise ValueError("fused migration requires a contiguous program")
        layers, batch, sequence, hidden = capsule.normed.shape
        output_width = program.adapter.weights.shape[-1]
        rows = batch * sequence
        grid = (
            layers,
            triton.cdiv(rows, self.block_m),
            triton.cdiv(output_width, self.block_n),
        )
        _fused_affine_extent_kernel[grid](
            capsule.normed,
            program.adapter.weights,
            program.adapter.biases,
            capsule.lengths,
            destination.offsets,
            destination.k,
            destination.v,
            rows,
            sequence,
            destination.k.shape[1],
            input_width=hidden,
            output_width=output_width,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )
        return destination


@torch.no_grad()
def migrate_packed_cache_capsule(
    capsule: MigrationCapsuleBatch,
    adapter: CompiledCacheAdapter,
) -> HSTUKVCache:
    normed = capsule.normed
    if normed.shape[0] != adapter.weights.shape[0]:
        raise ValueError("capsule and packed adapter depths differ")
    if normed.shape[-1] != adapter.weights.shape[1]:
        raise ValueError("capsule and packed adapter widths differ")
    if normed.device != adapter.weights.device or normed.device != adapter.biases.device:
        raise ValueError("capsule and packed adapter must share a device")
    flattened = normed.to(adapter.weights.dtype).flatten(1, 2)
    bias = adapter.biases[:, None, :].expand(
        -1,
        flattened.shape[1],
        -1,
    )
    projected = torch.baddbmm(bias, flattened, adapter.weights)
    projected = projected.unflatten(1, normed.shape[1:3])
    positions = torch.arange(normed.shape[2], device=normed.device)
    invalid = positions.unsqueeze(0) >= capsule.lengths.unsqueeze(1)
    projected.masked_fill_(invalid.unsqueeze(0).unsqueeze(-1), 0)
    projected = projected.to(normed.dtype)
    width = projected.shape[-1] // 2
    return HSTUKVCache(
        k=projected[..., :width],
        v=projected[..., width:],
        seq_len=normed.shape[2],
    )
