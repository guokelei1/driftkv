"""Common-projection native-response defect for one model release.

The Current reduced trajectory supplies one dense history subspace per layer.
Both its approximate Current K/V and the exact Parent K/V are projected into
that *same* subspace.  A future query then reads

``R(Parent exact) + R(Current reduced) - R(Parent projected)``.

Subtraction happens after the model's native query--key activation and value
aggregation.  It therefore preserves finite K/V interaction and is not
equivalent to adding a low-rank tensor delta before a nonlinear reader.  The
constructor is legal when the Current trajectory is legal: it does not read
Current Exact upper-layer state, candidates, labels, or future events.

This module is a single-configuration preflight.  Common projection and
control variates are standard numerical components; only a stable advantage
over the strongest same-rank Current replay could justify studying the
response-level cancellation as a migration mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hstu_kvcache.models import HSTUKVCache
from insight_two.cone_response_memory import (
    _block_update,
    _native_prefix_heads,
    _native_self_heads,
)
from insight_two.mode_space_replay import (
    FactorizedCacheLayer,
    FactorizedReplay,
)

MEDIUM_EXACT_ALL_FLOPS = 4_771_282_944
SINGLE_R8_MATRIX_FREE_TERMINAL_KV_FLOPS = 853_836_992


@dataclass(frozen=True)
class CommonProjectionResponseMemory:
    """Per-layer same-basis Current/Parent reduced response states."""

    current: tuple[FactorizedCacheLayer, ...]
    parent: tuple[FactorizedCacheLayer, ...]
    source_length: int

    @property
    def stored_scalars(self) -> int:
        return sum(
            current.left.numel()
            + current.key_core.numel()
            + current.value_core.numel()
            + parent.key_core.numel()
            + parent.value_core.numel()
            for current, parent in zip(self.current, self.parent, strict=True)
        )


@dataclass(frozen=True)
class CommonProjectionIntervention:
    scores: torch.Tensor
    readout: torch.Tensor


@dataclass(frozen=True)
class CommonProjectionCost:
    base_single_r8_flops: int
    basis_qr_flops: int
    two_arm_projection_flops: int
    exact_all_flops: int

    @property
    def total_constructor_flops(self) -> int:
        return (
            self.base_single_r8_flops
            + self.basis_qr_flops
            + self.two_arm_projection_flops
        )

    @property
    def constructor_fraction(self) -> float:
        return self.total_constructor_flops / self.exact_all_flops

    @property
    def within_twenty_percent(self) -> bool:
        return self.constructor_fraction <= 0.20

    @property
    def incremental_reader_flops_per_query(self) -> int:
        # Two native rank-8 factor reads per layer, followed by two H-wide
        # signed additions.  The complete Parent read is the Reuse baseline.
        return 6 * (2 * 202_752 + 2 * 192)


def medium_common_projection_cost() -> CommonProjectionCost:
    """Conservative cost using the complete strong-control constructor.

    The base already includes the strong shared-layer0 splice compiler that
    this functional path does not need.  We nevertheless retain its full cost
    and additionally charge QR plus dense K/V projections for *both* reduced
    arms.  The resulting 19.49% is therefore an upper bound for this semantic
    path, not a hidden subtraction from the existing ledger.
    """

    length = 1024
    rank = 8
    width = 192
    layers = 6
    qr = layers * (2 * length * rank * rank - (2 * rank**3) // 3)
    # Per arm, per layer: U^T K and U^T V, each 2*N*r*H.
    projections = 2 * layers * 4 * length * rank * width
    return CommonProjectionCost(
        base_single_r8_flops=SINGLE_R8_MATRIX_FREE_TERMINAL_KV_FLOPS,
        basis_qr_flops=qr,
        two_arm_projection_flops=projections,
        exact_all_flops=MEDIUM_EXACT_ALL_FLOPS,
    )


@torch.inference_mode()
def build_common_projection_response_memory(
    current_replay: FactorizedReplay,
    exact_parent: HSTUKVCache,
) -> CommonProjectionResponseMemory:
    """Project approximate Current and exact Parent into Current replay spans."""

    if current_replay.cache.k.shape != exact_parent.k.shape:
        raise ValueError("Current replay and Parent cache shapes differ")
    if current_replay.cache.v.shape != exact_parent.v.shape:
        raise ValueError("Current replay and Parent cache V shapes differ")
    if len(current_replay.layers) != exact_parent.k.shape[0]:
        raise ValueError("Current replay and Parent cache layer counts differ")
    current_layers: list[FactorizedCacheLayer] = []
    parent_layers: list[FactorizedCacheLayer] = []
    for layer_index, layer in enumerate(current_replay.layers):
        basis, _ = torch.linalg.qr(layer.left.float(), mode="reduced")
        basis = basis[:, :, : layer.rank].to(dtype=layer.left.dtype)
        current_k = basis.transpose(1, 2) @ current_replay.cache.k[layer_index]
        current_v = basis.transpose(1, 2) @ current_replay.cache.v[layer_index]
        parent_k = basis.transpose(1, 2) @ exact_parent.k[layer_index]
        parent_v = basis.transpose(1, 2) @ exact_parent.v[layer_index]
        current_layers.append(FactorizedCacheLayer(basis, current_k, current_v))
        parent_layers.append(FactorizedCacheLayer(basis, parent_k, parent_v))
    return CommonProjectionResponseMemory(
        current=tuple(current_layers),
        parent=tuple(parent_layers),
        source_length=exact_parent.seq_len,
    )


@torch.inference_mode()
def factorized_prefix_heads(
    attention,
    q: torch.Tensor,
    layer: FactorizedCacheLayer,
) -> torch.Tensor:
    """Evaluate native pointwise attention on a factorized K/V cache."""

    if attention.position_bias is not None:
        raise ValueError("common-projection preflight requires no position bias")
    if layer.left.shape[0] != 1 or layer.key_core.shape[0] != 1:
        raise ValueError("single-user factorized prefix expected")
    key_core = layer.key_core.view(
        1, layer.rank, attention.num_heads, attention.head_dim
    ).permute(0, 2, 1, 3)
    value_core = layer.value_core.view(
        1, layer.rank, attention.num_heads, attention.head_dim
    ).permute(0, 2, 1, 3)
    core_logits = torch.matmul(q, key_core.transpose(-2, -1))
    logits = torch.matmul(
        core_logits, layer.left.transpose(1, 2).unsqueeze(1)
    ) * attention.scale
    weights = attention._activate(logits)
    if attention.block_variant == "hstu_reference":
        weights = weights / attention.cfg.max_seq_len
    weights = attention.attn_dropout(weights)
    reduced_weights = torch.matmul(weights, layer.left.unsqueeze(1))
    return torch.matmul(reduced_weights, value_core)


@torch.inference_mode()
def intervene_common_projection_response(
    model,
    exact_parent: HSTUKVCache,
    memory: CommonProjectionResponseMemory,
    candidate_ids: torch.Tensor,
    query_time_deltas: torch.Tensor,
) -> CommonProjectionIntervention:
    """Read exact Parent plus the same-projection native response defect."""

    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != 1:
        raise ValueError("single-user candidate_ids must have shape [1,C]")
    if exact_parent.seq_len != memory.source_length:
        raise ValueError("Parent and response-memory lengths differ")
    if len(memory.current) != len(model.blocks):
        raise ValueError("response-memory and model layer counts differ")
    candidates = candidate_ids.shape[1]
    x = model.embed_query_tokens(candidate_ids, query_time_deltas).reshape(
        candidates, 1, model.cfg.hidden_size
    )
    for layer_index, (block, current_layer, parent_layer) in enumerate(
        zip(model.blocks, memory.current, memory.parent, strict=True)
    ):
        residual = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        exact_parent_heads = _native_prefix_heads(
            block.attn,
            q,
            exact_parent.k[layer_index],
            exact_parent.v[layer_index],
        )
        current_heads = factorized_prefix_heads(block.attn, q, current_layer)
        projected_parent_heads = factorized_prefix_heads(
            block.attn, q, parent_layer
        )
        prefix = exact_parent_heads + current_heads - projected_parent_heads
        self_heads = _native_self_heads(block.attn, q, k_new, v_new)
        x = residual + _block_update(block, x_norm, prefix + self_heads)
    readout = model.final_norm(x).reshape(1, candidates, model.cfg.hidden_size)
    return CommonProjectionIntervention(
        scores=model.cc_score_head(readout).squeeze(-1),
        readout=readout,
    )


def materialize_common_projection_state_splice(
    exact_parent: HSTUKVCache,
    memory: CommonProjectionResponseMemory,
) -> HSTUKVCache:
    """Control: subtract the same defect in K/V space before native reading."""

    keys = []
    values = []
    for layer_index, (current, parent) in enumerate(
        zip(memory.current, memory.parent, strict=True)
    ):
        current_k, current_v = current.materialize()
        parent_k, parent_v = parent.materialize()
        keys.append(exact_parent.k[layer_index] + current_k - parent_k)
        values.append(exact_parent.v[layer_index] + current_v - parent_v)
    return HSTUKVCache(
        k=torch.stack(keys, dim=0),
        v=torch.stack(values, dim=0),
        seq_len=exact_parent.seq_len,
    )
