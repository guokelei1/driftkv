from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections import deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from ..models import HSTU
from .artifacts import sha256_file
from .cohort_jagged import JaggedMigratedKVBatch
from .destination import HBMKVUpdateDestination, KVVersionManifest
from .program import MigrationProgram
from .stage4_engine import Stage4RuntimeConfig
from .stage4_source import (
    LazyStage4SourceReader,
    Stage4ExtentSpec,
    build_stage4_extents,
    place_stage4_extents_lpt,
)
from .stage45_reclaim import (
    ReclaimableOldKV,
    Stage45ReclaimMetrics,
)
from .stage45_resident import (
    Stage45DeviceMetrics,
    Stage45ResidentPlan,
    _make_device_output,
)

DIRECT_OLDKV_PROGRAM_PROTOCOL = "cohortkv_stage4_5_direct_oldkv_program_v1"
DIRECT_OLDKV_RUNTIME_PROTOCOL = "cohortkv_stage4_5_direct_oldkv_runtime_v1"
DIRECT_OLDKV_ENGINE_PROTOCOL = "cohortkv_stage4_5_direct_oldkv_engine_v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class DirectOldKVProgram:
    source_version: str
    target_version: str
    weights: torch.Tensor
    biases: torch.Tensor

    def __post_init__(self) -> None:
        if not self.source_version or not self.target_version:
            raise ValueError("direct old-K/V program versions are missing")
        if self.weights.ndim != 3 or self.biases.ndim != 2:
            raise ValueError("direct old-K/V program tensor ranks differ")
        if (
            self.weights.shape[0] < 1
            or self.weights.shape[1] < 2
            or self.weights.shape[1] % 2
            or self.weights.shape[1] != self.weights.shape[2]
            or self.biases.shape
            != (self.weights.shape[0], self.weights.shape[2])
        ):
            raise ValueError("direct old-K/V program shapes differ")
        if (
            self.weights.device != self.biases.device
            or self.weights.dtype != self.biases.dtype
            or not self.weights.is_floating_point()
            or not self.weights.is_contiguous()
            or not self.biases.is_contiguous()
        ):
            raise ValueError("direct old-K/V program layout differs")
        if not bool(torch.isfinite(self.weights).all()) or not bool(
            torch.isfinite(self.biases).all()
        ):
            raise ValueError("direct old-K/V program is nonfinite")

    @property
    def device(self) -> torch.device:
        return self.weights.device

    @property
    def num_layers(self) -> int:
        return self.weights.shape[0]

    @property
    def kv_width(self) -> int:
        return self.weights.shape[1] // 2

    @property
    def nbytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (self.weights, self.biases)
        )

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> DirectOldKVProgram:
        return DirectOldKVProgram(
            source_version=self.source_version,
            target_version=self.target_version,
            weights=self.weights.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            biases=self.biases.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
        )


@dataclass(frozen=True)
class DirectOldKVLayerCompileMetrics:
    layer: int
    condition_number: float
    float32_weight_residual_max: float
    float16_weight_residual_max: float
    float32_bias_residual_max: float
    float16_bias_residual_max: float

    def to_dict(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "condition_number": self.condition_number,
            "float32_weight_residual_max": (
                self.float32_weight_residual_max
            ),
            "float16_weight_residual_max": (
                self.float16_weight_residual_max
            ),
            "float32_bias_residual_max": (
                self.float32_bias_residual_max
            ),
            "float16_bias_residual_max": (
                self.float16_bias_residual_max
            ),
        }


@dataclass(frozen=True)
class DirectOldKVCompileMetrics:
    elapsed_seconds: float
    layers: tuple[DirectOldKVLayerCompileMetrics, ...]
    protocol: str = DIRECT_OLDKV_PROGRAM_PROTOCOL

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "elapsed_seconds": self.elapsed_seconds,
            "condition_number_min": min(
                value.condition_number for value in self.layers
            ),
            "condition_number_max": max(
                value.condition_number for value in self.layers
            ),
            "float32_weight_residual_max": max(
                value.float32_weight_residual_max for value in self.layers
            ),
            "float16_weight_residual_max": max(
                value.float16_weight_residual_max for value in self.layers
            ),
            "float32_bias_residual_max": max(
                value.float32_bias_residual_max for value in self.layers
            ),
            "float16_bias_residual_max": max(
                value.float16_bias_residual_max for value in self.layers
            ),
            "layers": [value.to_dict() for value in self.layers],
        }


def _projection_bias(
    value: torch.nn.Linear,
    device: torch.device,
) -> torch.Tensor:
    if value.bias is None:
        return torch.zeros(
            value.out_features,
            dtype=torch.float32,
            device=device,
        )
    return value.bias.detach().to(device=device, dtype=torch.float32)


@torch.no_grad()
def compile_direct_oldkv_program(
    source_model: HSTU,
    compiled_program: MigrationProgram,
) -> tuple[DirectOldKVProgram, DirectOldKVCompileMetrics]:
    if (
        len(source_model.blocks) != compiled_program.num_layers
        or compiled_program.source_version == compiled_program.target_version
    ):
        raise ValueError("direct old-K/V compiler inputs differ")
    model_device = next(source_model.parameters()).device
    started = time.perf_counter()
    weights = []
    biases = []
    metrics = []
    for layer, block in enumerate(source_model.blocks):
        k_weight = block.attn.k_proj.weight.detach().T.to(
            device=model_device,
            dtype=torch.float32,
        )
        v_weight = block.attn.v_proj.weight.detach().T.to(
            device=model_device,
            dtype=torch.float32,
        )
        projection = torch.cat((k_weight, v_weight), dim=1)
        target_weight = compiled_program.adapter.weights[layer].to(
            device=model_device,
            dtype=torch.float32,
        )
        target_bias = compiled_program.adapter.biases[layer].to(
            device=model_device,
            dtype=torch.float32,
        )
        if (
            projection.shape[0] != target_weight.shape[0]
            or projection.shape[1] != target_weight.shape[1]
        ):
            raise ValueError("direct old-K/V projection signature differs")
        direct_weight = projection.T @ torch.linalg.solve(
            projection @ projection.T,
            target_weight,
        )
        source_bias = torch.cat(
            (
                _projection_bias(block.attn.k_proj, model_device),
                _projection_bias(block.attn.v_proj, model_device),
            )
        )
        direct_bias = target_bias - source_bias @ direct_weight
        deployed_weight = direct_weight.to(torch.float16)
        deployed_bias = direct_bias.to(torch.float16)
        float_weight_residual = (
            projection @ direct_weight - target_weight
        ).abs()
        half_weight_residual = (
            projection @ deployed_weight.float() - target_weight
        ).abs()
        float_bias_residual = (
            source_bias @ direct_weight + direct_bias - target_bias
        ).abs()
        half_bias_residual = (
            source_bias @ deployed_weight.float()
            + deployed_bias.float()
            - target_bias
        ).abs()
        weights.append(deployed_weight.cpu().contiguous())
        biases.append(deployed_bias.cpu().contiguous())
        metrics.append(
            DirectOldKVLayerCompileMetrics(
                layer=layer,
                condition_number=float(torch.linalg.cond(projection)),
                float32_weight_residual_max=float(
                    float_weight_residual.max()
                ),
                float16_weight_residual_max=float(
                    half_weight_residual.max()
                ),
                float32_bias_residual_max=float(float_bias_residual.max()),
                float16_bias_residual_max=float(half_bias_residual.max()),
            )
        )
    program = DirectOldKVProgram(
        source_version=compiled_program.source_version,
        target_version=compiled_program.target_version,
        weights=torch.stack(weights).contiguous(),
        biases=torch.stack(biases).contiguous(),
    )
    return program, DirectOldKVCompileMetrics(
        elapsed_seconds=time.perf_counter() - started,
        layers=tuple(metrics),
    )


def write_direct_oldkv_program(
    program: DirectOldKVProgram,
    path: str | Path,
    provenance: dict[str, object],
    compile_metrics: DirectOldKVCompileMetrics,
) -> dict[str, object]:
    if not isinstance(provenance, dict):
        raise ValueError("direct old-K/V provenance must be a dictionary")
    prepared = program.to("cpu", dtype=torch.float16)
    payload = {
        "protocol": DIRECT_OLDKV_PROGRAM_PROTOCOL,
        "source_version": prepared.source_version,
        "target_version": prepared.target_version,
        "dtype": "float16",
        "source_representation": "existing_old_kv_fp16",
        "weights_shape": list(prepared.weights.shape),
        "biases_shape": list(prepared.biases.shape),
        "weights": prepared.weights,
        "biases": prepared.biases,
        "compile_metrics": compile_metrics.to_dict(),
        "provenance": provenance,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "protocol": DIRECT_OLDKV_PROGRAM_PROTOCOL,
        "source_version": prepared.source_version,
        "target_version": prepared.target_version,
        "dtype": "float16",
        "source_representation": "existing_old_kv_fp16",
        "weights_shape": list(prepared.weights.shape),
        "biases_shape": list(prepared.biases.shape),
        "compile_metrics": compile_metrics.to_dict(),
    }


def load_direct_oldkv_program(
    path: str | Path,
    expected_sha256: str | None = None,
    expected_source_version: str | None = None,
    expected_target_version: str | None = None,
    expected_num_layers: int | None = None,
    expected_kv_width: int | None = None,
) -> tuple[DirectOldKVProgram, dict[str, object]]:
    program_path = Path(path)
    digest = sha256_file(program_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("direct old-K/V program hash differs")
    payload = torch.load(
        program_path,
        map_location="cpu",
        weights_only=False,
    )
    if payload.get("protocol") != DIRECT_OLDKV_PROGRAM_PROTOCOL:
        raise ValueError("direct old-K/V program protocol differs")
    source = payload.get("source_version")
    target = payload.get("target_version")
    weights = payload.get("weights")
    biases = payload.get("biases")
    if (
        not isinstance(source, str)
        or not isinstance(target, str)
        or not isinstance(weights, torch.Tensor)
        or not isinstance(biases, torch.Tensor)
        or payload.get("dtype") != "float16"
        or payload.get("source_representation")
        != "existing_old_kv_fp16"
        or weights.dtype != torch.float16
        or biases.dtype != torch.float16
        or list(weights.shape) != payload.get("weights_shape")
        or list(biases.shape) != payload.get("biases_shape")
        or not isinstance(payload.get("provenance"), dict)
        or not isinstance(payload.get("compile_metrics"), dict)
    ):
        raise ValueError("direct old-K/V program payload differs")
    program = DirectOldKVProgram(
        source_version=source,
        target_version=target,
        weights=weights,
        biases=biases,
    )
    if (
        expected_source_version is not None
        and program.source_version != expected_source_version
    ):
        raise ValueError("direct old-K/V source version differs")
    if (
        expected_target_version is not None
        and program.target_version != expected_target_version
    ):
        raise ValueError("direct old-K/V target version differs")
    if (
        expected_num_layers is not None
        and program.num_layers != expected_num_layers
    ):
        raise ValueError("direct old-K/V layer count differs")
    if (
        expected_kv_width is not None
        and program.kv_width != expected_kv_width
    ):
        raise ValueError("direct old-K/V width differs")
    return program, {
        "path": str(program_path),
        "sha256": digest,
        "bytes": program_path.stat().st_size,
        "protocol": payload["protocol"],
        "source_version": source,
        "target_version": target,
        "dtype": payload["dtype"],
        "source_representation": payload["source_representation"],
        "weights_shape": payload["weights_shape"],
        "biases_shape": payload["biases_shape"],
        "compile_metrics": payload["compile_metrics"],
        "provenance": payload["provenance"],
    }


if triton is not None:

    @triton.jit
    def _direct_oldkv_affine_kernel(
        old_k,
        old_v,
        weights,
        biases,
        output_k,
        output_v,
        tokens,
        kv_width: tl.constexpr,
        output_width: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        layer = tl.program_id(0)
        token_offsets = (
            tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        )
        output_offsets = (
            tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
        )
        reduction_offsets = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start in range(0, kv_width, BLOCK_K):
            current_reduction = start + reduction_offsets
            source_mask = (
                (token_offsets[:, None] < tokens)
                & (current_reduction[None, :] < kv_width)
            )
            weight_mask = (
                (current_reduction[:, None] < kv_width)
                & (output_offsets[None, :] < output_width)
            )
            old_k_values = tl.load(
                old_k
                + layer * tokens * kv_width
                + token_offsets[:, None] * kv_width
                + current_reduction[None, :],
                mask=source_mask,
                other=0.0,
            )
            old_v_values = tl.load(
                old_v
                + layer * tokens * kv_width
                + token_offsets[:, None] * kv_width
                + current_reduction[None, :],
                mask=source_mask,
                other=0.0,
            )
            k_weights = tl.load(
                weights
                + layer * output_width * output_width
                + current_reduction[:, None] * output_width
                + output_offsets[None, :],
                mask=weight_mask,
                other=0.0,
            )
            v_weights = tl.load(
                weights
                + layer * output_width * output_width
                + (kv_width + current_reduction[:, None]) * output_width
                + output_offsets[None, :],
                mask=weight_mask,
                other=0.0,
            )
            accumulator += tl.dot(old_k_values, k_weights)
            accumulator += tl.dot(old_v_values, v_weights)
        bias = tl.load(
            biases + layer * output_width + output_offsets,
            mask=output_offsets < output_width,
            other=0.0,
        )
        values = accumulator + bias[None, :]
        tl.store(
            output_k
            + layer * tokens * kv_width
            + token_offsets[:, None] * kv_width
            + output_offsets[None, :],
            values,
            mask=(token_offsets[:, None] < tokens)
            & (output_offsets[None, :] < kv_width),
        )
        value_offsets = output_offsets - kv_width
        tl.store(
            output_v
            + layer * tokens * kv_width
            + token_offsets[:, None] * kv_width
            + value_offsets[None, :],
            values,
            mask=(token_offsets[:, None] < tokens)
            & (output_offsets[None, :] >= kv_width)
            & (output_offsets[None, :] < output_width),
        )

else:
    _direct_oldkv_affine_kernel = None


def validate_direct_oldkv_extent(
    program: DirectOldKVProgram,
    source: JaggedMigratedKVBatch,
    destination: JaggedMigratedKVBatch,
    check_metadata_values: bool = False,
) -> None:
    if (
        source.record_ids != destination.record_ids
        or source.record_ids == ()
        or source.migration_anchor_version != program.source_version
        or destination.migration_anchor_version != program.source_version
        or destination.served_kv_target != program.target_version
        or source.k.shape != destination.k.shape
        or source.v.shape != destination.v.shape
        or source.k.shape[0] != program.num_layers
        or source.k.shape[2] != program.kv_width
        or source.k.dtype != torch.float16
        or destination.k.dtype != torch.float16
        or source.k.device != program.device
        or destination.k.device != program.device
        or not source.k.is_contiguous()
        or not source.v.is_contiguous()
        or not destination.k.is_contiguous()
        or not destination.v.is_contiguous()
    ):
        raise ValueError("direct old-K/V extent signature differs")
    if (
        source.lengths.shape != destination.lengths.shape
        or source.offsets.shape != destination.offsets.shape
        or source.lengths.dtype != destination.lengths.dtype
        or source.offsets.dtype != destination.offsets.dtype
    ):
        raise ValueError("direct old-K/V extent metadata layout differs")
    write_storages = {
        destination.k.untyped_storage().data_ptr(),
        destination.v.untyped_storage().data_ptr(),
    }
    read_tensors = (
        source.k,
        source.v,
        source.lengths,
        source.offsets,
        program.weights,
        program.biases,
        destination.lengths,
        destination.offsets,
    )
    if len(write_storages) != 2 or any(
        value.untyped_storage().data_ptr() in write_storages
        for value in read_tensors
    ):
        raise ValueError("direct old-K/V output aliases an input")
    if check_metadata_values and (
        not torch.equal(source.lengths, destination.lengths)
        or not torch.equal(source.offsets, destination.offsets)
        or int(source.offsets[0]) != 0
        or int(source.offsets[-1]) != source.token_count
        or not torch.equal(
            source.offsets[1:] - source.offsets[:-1],
            source.lengths,
        )
    ):
        raise ValueError("direct old-K/V metadata values differ")


class DirectOldKVFusedOperator:
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
            raise ValueError("direct old-K/V launch parameters differ")
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages

    @property
    def name(self) -> str:
        return (
            f"direct_oldkv_triton_m{self.block_m}_n{self.block_n}_"
            f"k{self.block_k}_w{self.num_warps}_s{self.num_stages}"
        )

    def prepare_program(
        self,
        program: DirectOldKVProgram,
        device: torch.device | str,
    ) -> DirectOldKVProgram:
        return program.to(device, dtype=torch.float16)

    @torch.no_grad()
    def execute_into(
        self,
        program: DirectOldKVProgram,
        source: JaggedMigratedKVBatch,
        destination: JaggedMigratedKVBatch,
    ) -> JaggedMigratedKVBatch:
        validate_direct_oldkv_extent(program, source, destination)
        tokens = source.token_count
        output_width = 2 * program.kv_width
        grid = (
            program.num_layers,
            triton.cdiv(tokens, self.block_m),
            triton.cdiv(output_width, self.block_n),
        )
        _direct_oldkv_affine_kernel[grid](
            source.k,
            source.v,
            program.weights,
            program.biases,
            destination.k,
            destination.v,
            tokens,
            kv_width=program.kv_width,
            output_width=output_width,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )
        return destination


@torch.no_grad()
def execute_direct_oldkv_reference(
    program: DirectOldKVProgram,
    source: JaggedMigratedKVBatch,
    destination: JaggedMigratedKVBatch,
) -> JaggedMigratedKVBatch:
    validate_direct_oldkv_extent(
        program,
        source,
        destination,
        check_metadata_values=True,
    )
    joined = torch.cat((source.k, source.v), dim=-1)
    projected = torch.baddbmm(
        program.biases[:, None, :].expand(
            program.num_layers,
            source.token_count,
            2 * program.kv_width,
        ),
        joined,
        program.weights,
    )
    destination.k.copy_(projected[..., : program.kv_width])
    destination.v.copy_(projected[..., program.kv_width :])
    return destination


class DirectOldKVTransform:
    method = "compiled_old_kv"
    target_source_tier = "existing_old_kv_hbm"

    def __init__(
        self,
        programs: Mapping[str, DirectOldKVProgram],
        operator: DirectOldKVFusedOperator,
        device: torch.device | str,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda" or not programs:
            raise ValueError("direct old-K/V transform inputs differ")
        self.operator = operator
        self.runtime_variant = operator.name
        self.programs = {
            source: operator.prepare_program(program, self.device)
            for source, program in programs.items()
        }
        if set(self.programs) != {
            value.source_version for value in self.programs.values()
        }:
            raise ValueError("direct old-K/V program keys differ")
        targets = {value.target_version for value in self.programs.values()}
        layers = {value.num_layers for value in self.programs.values()}
        widths = {value.kv_width for value in self.programs.values()}
        if len(targets) != 1 or len(layers) != 1 or len(widths) != 1:
            raise ValueError("direct old-K/V programs differ")
        self.target_version = next(iter(targets))
        self.num_layers = next(iter(layers))
        self.kv_width = next(iter(widths))

    @property
    def resident_bytes(self) -> int:
        return sum(value.nbytes for value in self.programs.values())

    def source_representations(
        self,
        source_version: str,
    ) -> tuple[str, ...]:
        if source_version not in self.programs:
            raise ValueError("direct old-K/V source program is missing")
        return ("old_kv_fp16",)

    def execute(
        self,
        source: JaggedMigratedKVBatch,
        destination: JaggedMigratedKVBatch,
    ) -> None:
        self.operator.execute_into(
            self.programs[source.migration_anchor_version],
            source,
            destination,
        )


def _validate_direct_transforms(
    transforms: tuple[DirectOldKVTransform, ...],
) -> tuple[str, int, int]:
    if not transforms or len(transforms) not in {1, 2, 4}:
        raise ValueError("direct old-K/V transform count differs")
    devices = tuple(value.device for value in transforms)
    targets = {value.target_version for value in transforms}
    layers = {value.num_layers for value in transforms}
    widths = {value.kv_width for value in transforms}
    program_sources = tuple(set(value.programs) for value in transforms)
    if (
        len(set(devices)) != len(devices)
        or any(value.type != "cuda" for value in devices)
        or len(targets) != 1
        or len(layers) != 1
        or len(widths) != 1
        or any(value != program_sources[0] for value in program_sources[1:])
    ):
        raise ValueError("direct old-K/V transform signatures differ")
    return next(iter(targets)), next(iter(layers)), next(iter(widths))


def build_stage45_oldkv_plan(
    source_manifest_path: Path | str,
    transforms: tuple[DirectOldKVTransform, ...],
    runtime_config: Stage4RuntimeConfig,
    record_ids: tuple[int, ...] | None = None,
    expected_source_manifest_sha256: str | None = None,
    expected_workload_content_sha256: str | None = None,
) -> Stage45ResidentPlan:
    target, layers, width = _validate_direct_transforms(transforms)
    reader = LazyStage4SourceReader(
        source_manifest_path,
        expected_workload_content_sha256,
    )
    if (
        expected_source_manifest_sha256 is not None
        and reader.manifest_file_sha256
        != expected_source_manifest_sha256
    ):
        raise ValueError("direct old-K/V source manifest hash differs")
    if (
        reader.manifest.target_version != target
        or reader.manifest.num_layers != layers
        or reader.manifest.kv_width != width
    ):
        raise ValueError("direct old-K/V manifest signature differs")
    if record_ids is None:
        record_ids = tuple(value.record_id for value in reader.manifest.records)
    if not record_ids or len(set(record_ids)) != len(record_ids):
        raise ValueError("direct old-K/V record IDs differ")
    source_versions = {
        reader.manifest.record_map[value].source_version
        for value in record_ids
    }
    representations = {
        source: transforms[0].source_representations(source)
        for source in source_versions
    }
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
        method="compiled_old_kv",
        target_version=target,
        source_tier="hbm_resident",
        runtime_config=runtime_config,
        record_ids=record_ids,
        num_layers=layers,
        kv_width=width,
        extents=extents,
        assignments=assignments,
    )


def _replacement_wave_bytes(
    assignment: tuple[Stage4ExtentSpec, ...],
    max_inflight: int,
) -> int:
    pending: deque[int] = deque()
    peak = 0
    index_bytes = torch.tensor([], dtype=torch.long).element_size()
    for extent in assignment:
        pending.append(
            extent.logical_output_bytes
            + (2 * len(extent.records) + 1) * index_bytes
        )
        peak = max(peak, sum(pending))
        if len(pending) >= max_inflight:
            pending.popleft()
    return peak


def stage45_oldkv_preflight(
    plan: Stage45ResidentPlan,
    transforms: tuple[DirectOldKVTransform, ...],
    allocator_margin_bytes: int = 2 * 1024**3,
) -> dict[str, object]:
    target, layers, width = _validate_direct_transforms(transforms)
    if (
        plan.method != "compiled_old_kv"
        or plan.source_tier != "hbm_resident"
        or plan.target_version != target
        or plan.num_layers != layers
        or plan.kv_width != width
        or len(plan.assignments) != len(transforms)
        or allocator_margin_bytes < 1
    ):
        raise ValueError("direct old-K/V preflight inputs differ")
    passed = True
    per_gpu = []
    index_bytes = torch.tensor([], dtype=torch.long).element_size()
    for index, (assignment, transform) in enumerate(
        zip(plan.assignments, transforms, strict=True)
    ):
        with torch.cuda.device(transform.device):
            free, total = torch.cuda.mem_get_info(transform.device)
            allocated = torch.cuda.memory_allocated(transform.device)
        old_bytes = sum(
            value.logical_output_bytes
            + (2 * len(value.records) + 1) * index_bytes
            for value in assignment
        )
        wave = _replacement_wave_bytes(
            assignment,
            plan.runtime_config.max_inflight,
        )
        required = allocated + old_bytes + wave + allocator_margin_bytes
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
                "standing_existing_old_kv_bytes": old_bytes,
                "maximum_replacement_wave_bytes": wave,
                "allocator_margin_bytes": allocator_margin_bytes,
                "required_peak_hbm_bytes": required,
                "passed": device_passed,
            }
        )
    return {
        "protocol": DIRECT_OLDKV_ENGINE_PROTOCOL,
        "method": plan.method,
        "source_tier": "existing_old_kv_hbm",
        "record_count": plan.record_count,
        "prefix_tokens": plan.prefix_tokens,
        "additional_source_state_bytes": 0,
        "per_gpu": per_gpu,
        "passed": passed,
    }


@dataclass(frozen=True)
class Stage45OldKVCorrectness:
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
            "reference_kind": "analytic zero-old-K/V affine oracle",
        }


@dataclass(frozen=True)
class Stage45OldKVJobReport:
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
    correctness: Stage45OldKVCorrectness | None
    method: str = "compiled_old_kv"
    source_tier: str = "existing_old_kv_hbm"
    protocol: str = DIRECT_OLDKV_ENGINE_PROTOCOL

    @property
    def load_imbalance_ratio(self) -> float:
        values = [
            value.standing_source_hbm_bytes + value.physical_output_bytes
            for value in self.devices
        ]
        return max(values) / max(sum(values) / len(values), 1)

    def timing_breakdown(self) -> dict[str, float]:
        return {
            "begin": self.begin_seconds,
            "h2d": 0.0,
            "compute": max(
                value.compute_seconds for value in self.devices
            ),
            "target_allocation": max(
                value.target_allocation_seconds for value in self.devices
            ),
            "stage": max(value.stage_seconds for value in self.devices),
            "commit": self.commit_seconds,
            "coordinator": self.coordinator_seconds,
            "elapsed": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class Stage45OldKVJobResult:
    report: Stage45OldKVJobReport
    destination: HBMKVUpdateDestination
    reclamation: Stage45ReclaimMetrics


@dataclass
class _DirectPending:
    extent: Stage4ExtentSpec
    source: JaggedMigratedKVBatch
    output: JaggedMigratedKVBatch
    layout_ready: torch.cuda.Event
    compute_start: torch.cuda.Event
    compute_end: torch.cuda.Event
    target_allocation_seconds: float


@dataclass(frozen=True)
class _DirectWorkerResult:
    metrics: Stage45DeviceMetrics


class Stage45OldKVEngine:
    def __init__(
        self,
        plan: Stage45ResidentPlan,
        transforms: tuple[DirectOldKVTransform, ...],
    ) -> None:
        target, layers, width = _validate_direct_transforms(transforms)
        if (
            plan.method != "compiled_old_kv"
            or plan.source_tier != "hbm_resident"
            or plan.target_version != target
            or plan.num_layers != layers
            or plan.kv_width != width
            or len(plan.assignments) != len(transforms)
        ):
            raise ValueError("direct old-K/V engine inputs differ")
        self.plan = plan
        self.transforms = transforms
        self.old_cache: ReclaimableOldKV | None = None
        self._pools = tuple(
            ThreadPoolExecutor(max_workers=1) for _ in transforms
        )
        streams = []
        for transform in transforms:
            with torch.cuda.device(transform.device):
                streams.append(torch.cuda.Stream(device=transform.device))
        self._streams = tuple(streams)
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            for pool in self._pools:
                pool.shutdown(wait=True)
            self._closed = True

    def install_old_cache(self, old_cache: ReclaimableOldKV) -> None:
        if old_cache.plan is not self.plan:
            raise ValueError("direct old-K/V cache plan differs")
        if (
            self.old_cache is not None
            and self.old_cache.metrics().final_old_kv_bytes != 0
        ):
            raise RuntimeError("direct old-K/V prior cache is not retired")
        self.old_cache = old_cache

    def _enqueue(
        self,
        extent: Stage4ExtentSpec,
        transform: DirectOldKVTransform,
        stream: torch.cuda.Stream,
        device_index: int,
    ) -> _DirectPending:
        if self.old_cache is None:
            raise RuntimeError("direct old-K/V cache is absent")
        source = self.old_cache.batch_for(
            extent.extent_id,
            device_index,
        )
        allocation_started = time.perf_counter()
        with torch.cuda.device(transform.device):
            output = _make_device_output(
                extent,
                self.plan.num_layers,
                self.plan.kv_width,
                transform.device,
            )
            layout_ready = torch.cuda.Event()
            layout_ready.record(torch.cuda.current_stream(transform.device))
            allocation_seconds = time.perf_counter() - allocation_started
            with torch.cuda.stream(stream):
                stream.wait_event(layout_ready)
                compute_start = torch.cuda.Event(enable_timing=True)
                compute_end = torch.cuda.Event(enable_timing=True)
                compute_start.record(stream)
                transform.execute(source, output)
                compute_end.record(stream)
        return _DirectPending(
            extent=extent,
            source=source,
            output=output,
            layout_ready=layout_ready,
            compute_start=compute_start,
            compute_end=compute_end,
            target_allocation_seconds=allocation_seconds,
        )

    def _worker(
        self,
        index: int,
        assignment: tuple[Stage4ExtentSpec, ...],
        transform: DirectOldKVTransform,
        transaction,
    ) -> _DirectWorkerResult:
        if self.old_cache is None:
            raise RuntimeError("direct old-K/V cache is absent")
        started = time.perf_counter()
        pending: deque[_DirectPending] = deque()
        compute_seconds = 0.0
        allocation_seconds = 0.0
        stage_seconds = 0.0
        output_bytes = 0
        source_bytes = sum(
            self.old_cache.batch_for(value.extent_id, index).nbytes
            for value in assignment
        )
        with torch.cuda.device(transform.device):
            torch.cuda.synchronize(transform.device)
            baseline = torch.cuda.memory_allocated(transform.device)
            torch.cuda.reset_peak_memory_stats(transform.device)
            stream = self._streams[index]

            def finalize(value: _DirectPending) -> None:
                nonlocal compute_seconds
                nonlocal allocation_seconds
                nonlocal stage_seconds
                nonlocal output_bytes
                value.compute_end.synchronize()
                compute_seconds += (
                    value.compute_start.elapsed_time(value.compute_end)
                    / 1000.0
                )
                allocation_seconds += value.target_allocation_seconds
                stage_started = time.perf_counter()
                transaction.stage(value.extent.extent_id, value.output)
                self.old_cache.retire(value.extent.extent_id, index)
                stage_seconds += time.perf_counter() - stage_started
                output_bytes += value.output.nbytes

            for extent in assignment:
                value = self._enqueue(
                    extent,
                    transform,
                    stream,
                    index,
                )
                self.old_cache.register_replacement(
                    extent,
                    value.output,
                    index,
                )
                pending.append(value)
                if len(pending) >= self.plan.runtime_config.max_inflight:
                    finalize(pending.popleft())
            while pending:
                finalize(pending.popleft())
            torch.cuda.synchronize(transform.device)
            peak = torch.cuda.max_memory_allocated(transform.device)
        return _DirectWorkerResult(
            metrics=Stage45DeviceMetrics(
                index=index,
                record_count=sum(len(value.records) for value in assignment),
                prefix_tokens=sum(
                    value.token_count for value in assignment
                ),
                standing_source_hbm_bytes=source_bytes,
                standing_source_host_bytes=0,
                transform_resident_bytes=transform.resident_bytes,
                h2d_traffic_bytes=0,
                physical_output_bytes=output_bytes,
                baseline_hbm_bytes=baseline,
                peak_hbm_bytes=peak,
                peak_incremental_hbm_bytes=max(peak - baseline, 0),
                elapsed_seconds=time.perf_counter() - started,
                h2d_seconds=0.0,
                compute_seconds=compute_seconds,
                target_allocation_seconds=allocation_seconds,
                stage_seconds=stage_seconds,
            )
        )

    def _validate_zero_source(
        self,
        destination: HBMKVUpdateDestination,
        atol: float,
        rtol: float,
    ) -> Stage45OldKVCorrectness:
        started = time.perf_counter()
        finite = True
        mismatched = 0
        maximum = 0.0
        elements = 0
        order = True
        metadata = True
        for assignment, transform in zip(
            self.plan.assignments,
            self.transforms,
            strict=True,
        ):
            for extent in assignment:
                actual = destination.load_extent(
                    self.plan.target_version,
                    extent.extent_id,
                )
                program = transform.programs[
                    extent.migration_anchor_version
                ]
                order = order and actual.record_ids == extent.record_ids
                expected_lengths = torch.tensor(
                    [value.prefix_tokens for value in extent.records],
                    dtype=torch.long,
                    device=actual.lengths.device,
                )
                expected_offsets = torch.cat(
                    (
                        torch.zeros(
                            1,
                            dtype=torch.long,
                            device=actual.offsets.device,
                        ),
                        expected_lengths.cumsum(0),
                    )
                )
                metadata = (
                    metadata
                    and torch.equal(actual.lengths, expected_lengths)
                    and torch.equal(actual.offsets, expected_offsets)
                )
                for layer in range(program.num_layers):
                    expected_k = program.biases[layer, : program.kv_width]
                    expected_v = program.biases[layer, program.kv_width :]
                    for start in range(0, actual.token_count, 256):
                        end = min(start + 256, actual.token_count)
                        for value, expected in (
                            (actual.k[layer, start:end], expected_k),
                            (actual.v[layer, start:end], expected_v),
                        ):
                            finite = finite and bool(
                                torch.isfinite(value).all()
                            )
                            mismatched += int(
                                torch.count_nonzero(
                                    ~torch.isclose(
                                        value,
                                        expected,
                                        atol=atol,
                                        rtol=rtol,
                                    )
                                )
                            )
                            if value.numel():
                                maximum = max(
                                    maximum,
                                    float(
                                        (
                                            value.float()
                                            - expected.float()
                                        )
                                        .abs()
                                        .max()
                                    ),
                                )
                            elements += value.numel()
        return Stage45OldKVCorrectness(
            finite=finite,
            allclose=mismatched == 0,
            max_abs_error=maximum,
            valid_element_count=elements,
            record_order_valid=order,
            lengths_offsets_valid=metadata,
            validation_seconds=time.perf_counter() - started,
            atol=atol,
            rtol=rtol,
        )

    def run(
        self,
        validate_zero_source: bool = False,
        job_id: str | None = None,
        atol: float = 0.02,
        rtol: float = 0.02,
    ) -> Stage45OldKVJobResult:
        if self._closed or self.old_cache is None:
            raise RuntimeError("direct old-K/V engine is not ready")
        if atol != 0.02 or rtol != 0.02:
            raise ValueError("direct old-K/V tolerances are frozen")
        for transform in self.transforms:
            torch.cuda.synchronize(transform.device)
        started = time.perf_counter()
        destination = HBMKVUpdateDestination(
            tuple(value.device for value in self.transforms),
            destination_id=(
                f"stage45-oldkv-{len(self.transforms)}gpu"
            ),
        )
        begin_started = time.perf_counter()
        transaction = destination.begin(
            job_id=job_id
            or f"stage45-oldkv-{len(self.transforms)}gpu",
            target_version=self.plan.target_version,
            expected_record_ids=self.plan.record_ids,
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
                        self.plan.assignments,
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
        reclamation = self.old_cache.metrics()
        devices = tuple(value.metrics for value in workers)
        if (
            reclamation.final_old_kv_bytes != 0
            or reclamation.retired_old_kv_bytes
            != reclamation.initial_old_kv_bytes
            or reclamation.final_new_kv_bytes
            != sum(value.physical_output_bytes for value in devices)
            or reclamation.retired_extent_count != len(self.plan.extents)
            or set(manifest.record_ids) != set(self.plan.record_ids)
        ):
            raise RuntimeError("direct old-K/V publication is incomplete")
        correctness = (
            self._validate_zero_source(destination, atol, rtol)
            if validate_zero_source
            else None
        )
        expected_elements = (
            self.plan.logical_output_bytes
            // torch.tensor([], dtype=torch.float16).element_size()
        )
        if (
            correctness is not None
            and correctness.valid_element_count != expected_elements
        ):
            raise RuntimeError("direct old-K/V validation coverage differs")
        worker_elapsed = max(value.elapsed_seconds for value in devices)
        coordinator_seconds = max(
            elapsed
            - begin_seconds
            - worker_elapsed
            - commit_seconds,
            0.0,
        )
        report = Stage45OldKVJobReport(
            runtime_config=self.plan.runtime_config,
            source_manifest_sha256=self.plan.source_manifest_sha256,
            workload_content_sha256=self.plan.workload_content_sha256,
            record_count=self.plan.record_count,
            prefix_tokens=self.plan.prefix_tokens,
            logical_source_bytes=self.plan.logical_source_bytes,
            physical_source_bytes=reclamation.initial_old_kv_bytes,
            resident_source_bytes=reclamation.initial_old_kv_bytes,
            logical_output_bytes=self.plan.logical_output_bytes,
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
        return Stage45OldKVJobResult(
            report=report,
            destination=destination,
            reclamation=reclamation,
        )


@dataclass(frozen=True)
class Stage45SourcePolicyDecision:
    action: str
    reason: str
    compiled_preflight_passed: bool
    existing_old_kv_available: bool
    program_verified: bool
    fallback_action: str = "exact"
    protocol: str = DIRECT_OLDKV_RUNTIME_PROTOCOL

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "action": self.action,
            "reason": self.reason,
            "compiled_preflight_passed": self.compiled_preflight_passed,
            "existing_old_kv_available": (
                self.existing_old_kv_available
            ),
            "program_verified": self.program_verified,
            "fallback_action": self.fallback_action,
        }


def choose_stage45_source_action(
    compiled_preflight_passed: bool,
    existing_old_kv_available: bool,
    program_verified: bool,
) -> Stage45SourcePolicyDecision:
    if (
        compiled_preflight_passed
        and existing_old_kv_available
        and program_verified
    ):
        return Stage45SourcePolicyDecision(
            action="compiled_old_kv",
            reason="verified_existing_old_kv_hot_path",
            compiled_preflight_passed=True,
            existing_old_kv_available=True,
            program_verified=True,
        )
    failed = []
    if not compiled_preflight_passed:
        failed.append("capacity_preflight")
    if not existing_old_kv_available:
        failed.append("existing_old_kv")
    if not program_verified:
        failed.append("program_verification")
    return Stage45SourcePolicyDecision(
        action="exact",
        reason="fallback_" + "_".join(failed),
        compiled_preflight_passed=compiled_preflight_passed,
        existing_old_kv_available=existing_old_kv_available,
        program_verified=program_verified,
    )


def direct_oldkv_program_set_sha256(
    descriptors: list[dict[str, object]],
) -> str:
    values = [
        {
            "source_version": value["source_version"],
            "target_version": value["target_version"],
            "sha256": value["sha256"],
            "bytes": value["bytes"],
        }
        for value in descriptors
    ]
    return hashlib.sha256(_canonical_json(values)).hexdigest()
