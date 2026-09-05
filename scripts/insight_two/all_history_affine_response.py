"""Probe-free all-history affine response state for legacy HSTU.

For every layer and head this module stores

``B = sum_i V_i`` and ``M = sum_i K_i outer V_i``.

The resulting read ``B + scale * q @ M`` is exactly the response obtained by
replacing ELU+1 with its non-negative affine branch at every history position.
It is therefore a useful falsifier for whether a functional migration object
can be formed without candidate/query probes.  It is not a novel attention
state: algebraically it is the unnormalised linear-attention / fast-weight
memory for the feature map ``phi(x)=[1,x]``.

The legal single-arm compiler consumes a reduced Current trajectory and the
exact persistent Parent cache.  It never receives Current-Exact upper-layer
state, labels, candidates, probes or future events.  The full-Exact path is an
explicit evaluation-only oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hstu_kvcache.models import HSTUKVCache
from insight_two.attention_cone_moments import AffineConeMoments
from insight_two.cone_response_memory import (
    ConeLayerResponseMoment,
    ConeResponseMemory,
)
from insight_two.mode_space_replay import FactorizedCacheLayer, FactorizedReplay

MEDIUM_EXACT_ALL_FLOPS = 4_771_282_944


@dataclass(frozen=True)
class AllHistoryAffineCost:
    """Static Medium ledger under the repository multiply-add convention."""

    method: str
    trajectory_flops: int
    current_moment_flops: int
    parent_moment_flops: int
    signed_subtraction_flops: int
    exact_all_flops: int
    sidecar_scalars: int
    incremental_reader_flops_per_query: int
    initial_sin_cos_evaluations: int
    initial_gaussian_draws: int
    embedding_lookup_scalars: int
    raw_history_scalars: int

    @property
    def total_constructor_flops(self) -> int:
        return (
            self.trajectory_flops
            + self.current_moment_flops
            + self.parent_moment_flops
            + self.signed_subtraction_flops
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


def _validate_factorized_layer(
    attention,
    layer: FactorizedCacheLayer,
) -> tuple[int, int, int]:
    if layer.left.ndim != 3 or layer.key_core.ndim != 3:
        raise ValueError("factorized cache tensors must be rank three")
    batch, length, rank = layer.left.shape
    if batch != 1:
        raise ValueError("all-history affine preflight supports one user")
    if layer.key_core.shape != (batch, rank, attention.inner):
        raise ValueError("factorized key core shape differs from attention")
    if layer.value_core.shape != layer.key_core.shape:
        raise ValueError("factorized K/V core shapes differ")
    return batch, length, rank


def _factorized_heads(attention, core: torch.Tensor) -> torch.Tensor:
    batch, rank, width = core.shape
    if width != attention.inner:
        raise ValueError("factor core width differs from attention")
    return core.view(batch, rank, attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)


@torch.inference_mode()
def build_factorized_all_history_moments(
    attention,
    layer: FactorizedCacheLayer,
) -> AffineConeMoments:
    """Form exact all-position ``B/M`` from ``K=L Ck, V=L Cv``."""

    batch, length, _ = _validate_factorized_layer(attention, layer)
    key_core = _factorized_heads(attention, layer.key_core)
    value_core = _factorized_heads(attention, layer.value_core)
    left_sum = layer.left.sum(dim=1)
    base = torch.einsum("br,bhrv->bhv", left_sum, value_core)
    gram = layer.left.transpose(1, 2) @ layer.left
    linear = torch.einsum("bhrk,brs,bhsv->bhkv", key_core, gram, value_core)
    return AffineConeMoments(
        base=base,
        linear=linear,
        positive_mask=torch.ones(
            batch,
            attention.num_heads,
            length,
            device=layer.left.device,
            dtype=torch.bool,
        ),
    )


def _dense_heads(attention, values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != attention.inner:
        raise ValueError("dense cache layer must have shape [1,N,attention.inner]")
    return values.view(1, values.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2)


@torch.inference_mode()
def build_dense_all_history_moments(
    attention,
    key: torch.Tensor,
    value: torch.Tensor,
) -> AffineConeMoments:
    """Form all-position ``B/M`` from a complete dense cache layer."""

    if key.shape != value.shape:
        raise ValueError("dense K/V shapes differ")
    keys = _dense_heads(attention, key)
    values = _dense_heads(attention, value)
    return AffineConeMoments(
        base=values.sum(dim=2),
        linear=torch.einsum("bhnk,bhnv->bhkv", keys, values),
        positive_mask=torch.ones(keys.shape[:3], device=keys.device, dtype=torch.bool),
    )


def _signed_layer(
    current: AffineConeMoments,
    parent: AffineConeMoments,
    *,
    source_length: int,
) -> ConeLayerResponseMoment:
    if current.base.shape != parent.base.shape:
        raise ValueError("Current and Parent base moment shapes differ")
    if current.linear.shape != parent.linear.shape:
        raise ValueError("Current and Parent linear moment shapes differ")
    return ConeLayerResponseMoment(
        base=(current.base - parent.base).detach(),
        linear=(current.linear - parent.linear).detach(),
        current_positive_mask=current.positive_mask.detach(),
        parent_positive_mask=parent.positive_mask.detach(),
        current_sample_positions=torch.arange(
            source_length, device=current.base.device, dtype=torch.long
        ),
        current_sample_weights=torch.ones(
            source_length, device=current.base.device, dtype=current.base.dtype
        ),
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
        anchor_count=0,
        source_kv_scalars=source_kv_scalars,
    )


@torch.inference_mode()
def build_single_arm_all_history_response_memory(
    model,
    current_replay: FactorizedReplay,
    exact_parent: HSTUKVCache,
) -> ConeResponseMemory:
    """Compile probe-free approximate-Current minus exact-Parent moments."""

    if len(current_replay.layers) != len(model.blocks):
        raise ValueError("Current replay and model layer counts differ")
    if current_replay.cache.k.shape != exact_parent.k.shape:
        raise ValueError("Current replay and exact Parent cache shapes differ")
    layers: list[ConeLayerResponseMoment] = []
    for layer_index, (block, current_layer) in enumerate(
        zip(model.blocks, current_replay.layers, strict=True)
    ):
        current = build_factorized_all_history_moments(block.attn, current_layer)
        parent = build_dense_all_history_moments(
            block.attn,
            exact_parent.k[layer_index],
            exact_parent.v[layer_index],
        )
        layers.append(_signed_layer(current, parent, source_length=exact_parent.seq_len))
    return _memory(
        layers,
        source_length=exact_parent.seq_len,
        source_kv_scalars=exact_parent.k.numel() + exact_parent.v.numel(),
    )


@torch.inference_mode()
def build_full_exact_all_history_response_memory(
    model,
    exact_current: HSTUKVCache,
    exact_parent: HSTUKVCache,
) -> ConeResponseMemory:
    """Evaluation-only full-Exact all-position affine response oracle."""

    if exact_current.k.shape != exact_parent.k.shape:
        raise ValueError("exact Current and Parent cache shapes differ")
    if exact_current.seq_len != exact_parent.seq_len:
        raise ValueError("exact Current and Parent cache lengths differ")
    layers: list[ConeLayerResponseMoment] = []
    for layer_index, block in enumerate(model.blocks):
        current = build_dense_all_history_moments(
            block.attn,
            exact_current.k[layer_index],
            exact_current.v[layer_index],
        )
        parent = build_dense_all_history_moments(
            block.attn,
            exact_parent.k[layer_index],
            exact_parent.v[layer_index],
        )
        layers.append(_signed_layer(current, parent, source_length=exact_parent.seq_len))
    return _memory(
        layers,
        source_length=exact_parent.seq_len,
        source_kv_scalars=exact_parent.k.numel() + exact_parent.v.numel(),
    )


def medium_all_history_affine_cost(method: str) -> AllHistoryAffineCost:
    """Return the strict matrix-free-initial Medium cost for one method."""

    context = 1024
    layers = 6
    heads = 6
    head_dim = 32
    rank = 8
    sidecar = layers * heads * (head_dim + head_dim * head_dim)
    factorized_per_layer = (
        2 * context * heads * rank
        + 2 * heads * rank * head_dim
        + 2 * context * heads * rank * rank
        + 2 * heads * head_dim * rank * rank
        + 2 * heads * rank * head_dim * head_dim
    )
    dense_per_layer = 2 * context * heads * head_dim + 2 * context * heads * head_dim * head_dim
    reader = layers * (2 * heads * head_dim * head_dim + 3 * heads * head_dim)
    if method == "single_current_r8_all_history_affine":
        trajectory = 801_341_056
        current_moment = layers * factorized_per_layer
        parent_moment = layers * dense_per_layer
        gaussian = 192 * 12
    elif method == "full_exact_all_history_affine_oracle":
        trajectory = MEDIUM_EXACT_ALL_FLOPS
        current_moment = layers * dense_per_layer
        parent_moment = layers * dense_per_layer
        gaussian = 0
    else:
        raise ValueError("unsupported all-history affine method")
    return AllHistoryAffineCost(
        method=method,
        trajectory_flops=trajectory,
        current_moment_flops=current_moment,
        parent_moment_flops=parent_moment,
        signed_subtraction_flops=sidecar,
        exact_all_flops=MEDIUM_EXACT_ALL_FLOPS,
        sidecar_scalars=sidecar,
        incremental_reader_flops_per_query=reader,
        initial_sin_cos_evaluations=2 * context * 16,
        initial_gaussian_draws=gaussian,
        embedding_lookup_scalars=2 * context * 192,
        raw_history_scalars=3 * context,
    )


def medium_single_r8_kv_splice_cost() -> dict[str, int | float | bool]:
    """Matrix-free/final-KV-only cost of the existing single-r8 KV control."""

    trajectory = 801_341_056
    layer0_basis = 20_122_176
    upper_cores = 32_373_760
    total = trajectory + layer0_basis + upper_cores
    return {
        "trajectory_flops": trajectory,
        "layer0_basis_flops": layer0_basis,
        "upper_core_flops": upper_cores,
        "total_constructor_flops": total,
        "constructor_fraction": total / MEDIUM_EXACT_ALL_FLOPS,
        "within_twenty_percent": total / MEDIUM_EXACT_ALL_FLOPS <= 0.20,
    }
