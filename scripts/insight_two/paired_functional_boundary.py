"""Compile reduced release trajectories at the legacy S4 response boundary.

The module is deliberately narrow.  It does not propose another token-state
mapping: a fixed P8 history-query panel defines the positive ELU+1 activation
region and the complete reduced Parent/Current trajectories are aggregated
into signed response moments.  Serving retains the exact Parent cache and
adds the signed response before the native output projection, gate and
residual update.

For a factorized cache layer ``K=L Ck`` and ``V=L Cv``, the positive-region
moments are formed without materializing either cache::

    B_h = (sum_n mask_hn L_n) Cv_h
    M_h = Ck_h^T (L^T diag(mask_h) L) Cv_h

This identity is exact for the reduced trajectory.  It does not make that
trajectory exact, and the affine compiler is legacy-ELU+1-specific.  The
paired path receives no Current-Exact upper-layer state, candidate, label or
future event.  A single-arm Current replay and a full-Exact moment path are
provided only as controls.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hstu_kvcache.models import HSTUKVCache
from insight_two.attention_cone_moments import (
    AffineConeMoments,
    build_positive_affine_moments,
)
from insight_two.cone_response_memory import (
    ConeLayerResponseMoment,
    ConeResponseMemory,
)
from insight_two.mode_space_replay import (
    FactorizedCacheLayer,
    FactorizedReplay,
    PairedReleaseReplay,
)
from insight_two.paired_region_delta import majority_positive_mask

PRIMARY_PROBES = 8
MEDIUM_EXACT_ALL_FLOPS = 4_771_282_944


@dataclass(frozen=True)
class FunctionalBoundaryCost:
    """Static Medium cost ledger for one release-time compiler."""

    method: str
    trajectory_flops: int
    anchor_probe_flops: int
    mask_and_moment_flops: int
    signed_moment_subtraction_flops: int
    exact_all_flops: int
    sidecar_scalars: int
    transient_mask_bits: int
    mask_sign_comparisons: int
    initial_sin_cos_evaluations: int
    initial_gaussian_draws: int
    embedding_lookup_scalars: int
    raw_history_scalars: int
    incremental_reader_flops_per_query: int

    @property
    def total_constructor_flops(self) -> int:
        return (
            self.trajectory_flops
            + self.anchor_probe_flops
            + self.mask_and_moment_flops
            + self.signed_moment_subtraction_flops
        )

    @property
    def constructor_fraction(self) -> float:
        return self.total_constructor_flops / self.exact_all_flops

    @property
    def within_twenty_percent(self) -> bool:
        return self.constructor_fraction <= 0.20

    @property
    def sidecar_fp32_bytes(self) -> int:
        return 4 * self.sidecar_scalars


def _validate_probe_queries(
    model,
    probe_queries: tuple[torch.Tensor, ...],
) -> None:
    if len(probe_queries) != len(model.blocks):
        raise ValueError("one probe-query tensor is required per layer")
    for block, queries in zip(model.blocks, probe_queries, strict=True):
        expected = (
            PRIMARY_PROBES,
            block.attn.num_heads,
            1,
            block.attn.head_dim,
        )
        if queries.shape != expected:
            raise ValueError(f"P8 probe query shape must be {expected}")


def _validate_factorized_layer(
    attention,
    layer: FactorizedCacheLayer,
) -> tuple[int, int, int]:
    if layer.left.ndim != 3 or layer.key_core.ndim != 3:
        raise ValueError("factorized cache layer tensors must be rank three")
    batch, length, rank = layer.left.shape
    if layer.key_core.shape != (batch, rank, attention.inner):
        raise ValueError("factorized key core shape differs from attention")
    if layer.value_core.shape != layer.key_core.shape:
        raise ValueError("factorized K/V core shapes differ")
    if batch != 1:
        raise ValueError("functional preflight supports exactly one user")
    return batch, length, rank


def _factorized_heads(
    attention,
    core: torch.Tensor,
) -> torch.Tensor:
    batch, rank, width = core.shape
    if width != attention.inner:
        raise ValueError("factor core width differs from attention")
    return core.view(batch, rank, attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)


@torch.inference_mode()
def factorized_majority_positive_cone(
    attention,
    probe_q: torch.Tensor,
    layer: FactorizedCacheLayer,
) -> torch.Tensor:
    """Return the exact P8 majority mask for ``K=left @ key_core``."""

    _, _, _ = _validate_factorized_layer(attention, layer)
    expected = (
        PRIMARY_PROBES,
        attention.num_heads,
        1,
        attention.head_dim,
    )
    if probe_q.shape != expected:
        raise ValueError(f"probe_q must have shape {expected}")
    key_core = _factorized_heads(attention, layer.key_core)
    core_logits = torch.matmul(probe_q, key_core.transpose(-2, -1))
    logits = torch.matmul(
        core_logits,
        layer.left.transpose(1, 2).unsqueeze(1),
    )
    positive_votes = (logits * attention.scale >= 0).sum(dim=0).squeeze(1)
    return (2 * positive_votes >= PRIMARY_PROBES).unsqueeze(0)


@torch.inference_mode()
def build_factorized_positive_moments(
    attention,
    layer: FactorizedCacheLayer,
    positive_mask: torch.Tensor,
) -> AffineConeMoments:
    """Build exact positive-region ``B/M`` from token factors."""

    batch, length, _ = _validate_factorized_layer(attention, layer)
    expected_mask = (batch, attention.num_heads, length)
    if positive_mask.shape != expected_mask or positive_mask.dtype != torch.bool:
        raise ValueError(f"positive_mask must be boolean with shape {expected_mask}")
    if positive_mask.device != layer.left.device:
        raise ValueError("mask and factors must share a device")
    key_core = _factorized_heads(attention, layer.key_core)
    value_core = _factorized_heads(attention, layer.value_core)
    weight = positive_mask.to(dtype=layer.left.dtype)
    left_sum = torch.einsum("bhn,bnr->bhr", weight, layer.left)
    base = torch.einsum("bhr,bhrv->bhv", left_sum, value_core)
    gram = torch.einsum("bhn,bnr,bns->bhrs", weight, layer.left, layer.left)
    linear = torch.einsum("bhrk,bhrs,bhsv->bhkv", key_core, gram, value_core)
    return AffineConeMoments(
        base=base,
        linear=linear,
        positive_mask=positive_mask.detach(),
    )


def _dense_heads(attention, values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != attention.inner:
        raise ValueError("dense cache layer must have shape [1,N,attention.inner]")
    return values.view(1, values.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2)


@torch.inference_mode()
def build_dense_positive_moments(
    attention,
    key: torch.Tensor,
    value: torch.Tensor,
    positive_mask: torch.Tensor,
) -> AffineConeMoments:
    """Dense reference builder used by exact and single-arm controls."""

    keys = _dense_heads(attention, key)
    values = _dense_heads(attention, value)
    return build_positive_affine_moments(keys, values, positive_mask)


def _signed_layer(
    current: AffineConeMoments,
    parent: AffineConeMoments,
    *,
    source_length: int,
) -> ConeLayerResponseMoment:
    if current.base.shape != parent.base.shape:
        raise ValueError("Current and Parent base moments differ in shape")
    if current.linear.shape != parent.linear.shape:
        raise ValueError("Current and Parent linear moments differ in shape")
    positions = torch.arange(source_length, device=current.base.device, dtype=torch.long)
    weights = torch.ones(source_length, device=current.base.device, dtype=current.base.dtype)
    return ConeLayerResponseMoment(
        base=(current.base - parent.base).detach(),
        linear=(current.linear - parent.linear).detach(),
        current_positive_mask=current.positive_mask.detach(),
        parent_positive_mask=parent.positive_mask.detach(),
        current_sample_positions=positions,
        current_sample_weights=weights,
        source_length=source_length,
    )


def _memory(
    layers: list[ConeLayerResponseMoment],
    *,
    source_length: int,
    source_kv_scalars: int,
) -> ConeResponseMemory:
    return ConeResponseMemory(
        layers=tuple(layers),
        source_length=source_length,
        anchor_count=PRIMARY_PROBES,
        source_kv_scalars=source_kv_scalars,
    )


@torch.inference_mode()
def build_paired_factorized_response_memory(
    model,
    replay: PairedReleaseReplay,
    probe_queries: tuple[torch.Tensor, ...],
    *,
    source_kv_scalars: int,
) -> ConeResponseMemory:
    """Compile approximate Current-minus-Parent trajectories at S4."""

    _validate_probe_queries(model, probe_queries)
    if len(replay.parent.layers) != len(model.blocks):
        raise ValueError("paired replay and model layer counts differ")
    source_length = replay.parent.cache.seq_len
    if replay.current.cache.seq_len != source_length:
        raise ValueError("paired replay arm lengths differ")
    layers: list[ConeLayerResponseMoment] = []
    for block, queries, current_layer, parent_layer in zip(
        model.blocks,
        probe_queries,
        replay.current.layers,
        replay.parent.layers,
        strict=True,
    ):
        current_mask = factorized_majority_positive_cone(block.attn, queries, current_layer)
        parent_mask = factorized_majority_positive_cone(block.attn, queries, parent_layer)
        current = build_factorized_positive_moments(block.attn, current_layer, current_mask)
        parent = build_factorized_positive_moments(block.attn, parent_layer, parent_mask)
        layers.append(_signed_layer(current, parent, source_length=source_length))
    return _memory(
        layers,
        source_length=source_length,
        source_kv_scalars=source_kv_scalars,
    )


@torch.inference_mode()
def build_single_arm_factorized_response_memory(
    model,
    current_replay: FactorizedReplay,
    exact_parent: HSTUKVCache,
    probe_queries: tuple[torch.Tensor, ...],
) -> ConeResponseMemory:
    """Control: approximate Current moments minus complete exact Parent moments."""

    _validate_probe_queries(model, probe_queries)
    if len(current_replay.layers) != len(model.blocks):
        raise ValueError("Current replay and model layer counts differ")
    if exact_parent.k.shape != current_replay.cache.k.shape:
        raise ValueError("Current replay and exact Parent cache shapes differ")
    source_length = exact_parent.seq_len
    layers: list[ConeLayerResponseMoment] = []
    for layer_index, (block, queries, current_layer) in enumerate(
        zip(model.blocks, probe_queries, current_replay.layers, strict=True)
    ):
        current_mask = factorized_majority_positive_cone(block.attn, queries, current_layer)
        parent_mask = majority_positive_mask(block.attn, queries, exact_parent.k[layer_index])
        current = build_factorized_positive_moments(block.attn, current_layer, current_mask)
        parent = build_dense_positive_moments(
            block.attn,
            exact_parent.k[layer_index],
            exact_parent.v[layer_index],
            parent_mask,
        )
        layers.append(_signed_layer(current, parent, source_length=source_length))
    return _memory(
        layers,
        source_length=source_length,
        source_kv_scalars=exact_parent.k.numel() + exact_parent.v.numel(),
    )


@torch.inference_mode()
def build_full_exact_response_memory(
    model,
    exact_current: HSTUKVCache,
    exact_parent: HSTUKVCache,
    probe_queries: tuple[torch.Tensor, ...],
) -> ConeResponseMemory:
    """Evaluation-only ceiling: full Current-minus-Parent S4 moments."""

    _validate_probe_queries(model, probe_queries)
    if exact_current.k.shape != exact_parent.k.shape:
        raise ValueError("exact Current and Parent cache shapes differ")
    source_length = exact_parent.seq_len
    layers: list[ConeLayerResponseMoment] = []
    for layer_index, (block, queries) in enumerate(zip(model.blocks, probe_queries, strict=True)):
        current_mask = majority_positive_mask(block.attn, queries, exact_current.k[layer_index])
        parent_mask = majority_positive_mask(block.attn, queries, exact_parent.k[layer_index])
        current = build_dense_positive_moments(
            block.attn,
            exact_current.k[layer_index],
            exact_current.v[layer_index],
            current_mask,
        )
        parent = build_dense_positive_moments(
            block.attn,
            exact_parent.k[layer_index],
            exact_parent.v[layer_index],
            parent_mask,
        )
        layers.append(_signed_layer(current, parent, source_length=source_length))
    return _memory(
        layers,
        source_length=source_length,
        source_kv_scalars=exact_parent.k.numel() + exact_parent.v.numel(),
    )


def _factorized_side_flops(
    *,
    context: int,
    probes: int,
    heads: int,
    head_dim: int,
    rank: int,
) -> int:
    """Mask plus ``B/M`` build for one factorized arm and one layer."""

    mask = (
        2 * probes * heads * head_dim * rank
        + 2 * probes * heads * rank * context
        + probes * heads * context
        + (probes - 1) * heads * context
    )
    moments = (
        2 * context * heads * rank
        + 2 * heads * rank * head_dim
        + 2 * context * heads * rank * rank
        + 2 * heads * head_dim * rank * rank
        + 2 * heads * rank * head_dim * head_dim
    )
    return mask + moments


def _dense_side_flops(
    *,
    context: int,
    probes: int,
    heads: int,
    head_dim: int,
) -> int:
    """Mask plus ``B/M`` build for one complete dense cache side/layer."""

    mask = (
        2 * probes * heads * context * head_dim
        + probes * heads * context
        + (probes - 1) * heads * context
    )
    moments = 2 * context * heads * head_dim + 2 * context * heads * head_dim * head_dim
    return mask + moments


def medium_functional_boundary_cost(
    method: str,
    *,
    matrix_free_initial: bool = False,
) -> FunctionalBoundaryCost:
    """Return the fixed 6L/H192/N1024/P8 static cost ledger.

    The trajectory constants are the already-audited, optimistic
    factor-aware triangular executors with a KV-only final layer.  In
    particular, they include model-specific raw input formation, target-rank
    truncation at sketch width, native nonlinear attention and all five
    required post-block recompressions.  Moment compilation replaces (rather
    than receives for free) the old shared-U0/core compiler.
    """

    context = 1024
    layers = 6
    heads = 6
    head_dim = 32
    probes = PRIMARY_PROBES
    sidecar = layers * heads * (head_dim + head_dim * head_dim)
    subtraction = sidecar
    anchor_probe = 56_168_448
    reader = layers * (2 * heads * head_dim * head_dim + 3 * heads * head_dim)

    if method == "paired_r4_functional_moments":
        # Two r4/os4/p1 arms, final layer KV-only, excluding the superseded
        # U0 builder and six signed-core builds.
        trajectory = 1_039_053_832
        if matrix_free_initial:
            trajectory = trajectory - 2 * (88_489_984 + 12_951_382) + 2 * 18_033_494
        mask_and_moment = (
            2
            * layers
            * _factorized_side_flops(
                context=context,
                probes=probes,
                heads=heads,
                head_dim=head_dim,
                rank=4,
            )
        )
        transient_masks = 2 * layers * heads * context
        arm_count = 2
        sketch_width = 8
    elif method == "single_current_r8_functional_moments":
        # One r8/os4/p1 Current arm, final layer KV-only.  Exact Parent is
        # scanned again for its complete control moments.
        trajectory = 882_314_368
        if matrix_free_initial:
            trajectory = trajectory - (88_489_984 + 19_766_208) + 27_282_880
        mask_and_moment = layers * (
            _factorized_side_flops(
                context=context,
                probes=probes,
                heads=heads,
                head_dim=head_dim,
                rank=8,
            )
            + _dense_side_flops(
                context=context,
                probes=probes,
                heads=heads,
                head_dim=head_dim,
            )
        )
        transient_masks = 2 * layers * heads * context
        arm_count = 1
        sketch_width = 12
    elif method == "full_exact_functional_moment_oracle":
        if matrix_free_initial:
            raise ValueError("matrix-free reduced input does not apply to Exact oracle")
        trajectory = MEDIUM_EXACT_ALL_FLOPS
        mask_and_moment = (
            2
            * layers
            * _dense_side_flops(
                context=context,
                probes=probes,
                heads=heads,
                head_dim=head_dim,
            )
        )
        transient_masks = 2 * layers * heads * context
        arm_count = 1
        sketch_width = 0
    else:
        raise ValueError("unsupported functional-boundary method")
    return FunctionalBoundaryCost(
        method=method,
        trajectory_flops=trajectory,
        anchor_probe_flops=anchor_probe,
        mask_and_moment_flops=mask_and_moment,
        signed_moment_subtraction_flops=subtraction,
        exact_all_flops=MEDIUM_EXACT_ALL_FLOPS,
        sidecar_scalars=sidecar,
        transient_mask_bits=transient_masks,
        mask_sign_comparisons=2 * layers * probes * heads * context,
        initial_sin_cos_evaluations=arm_count * 2 * context * 16,
        initial_gaussian_draws=arm_count * 192 * sketch_width,
        embedding_lookup_scalars=arm_count * 2 * context * 192,
        raw_history_scalars=arm_count * 3 * context,
        incremental_reader_flops_per_query=reader,
    )
