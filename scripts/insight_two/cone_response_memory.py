"""Signed affine-cone response memory for ELU+1 HSTU readers.

This is a mechanism diagnostic, not a fitted mapper.  Thirty-two label-free
anchor queries are run coherently by the Current model over the complete
Parent cache.  At each layer, their majority QK sign pattern defines one
Current cone and one Parent cone.  The positive branch of ELU+1 attention is
then represented by exact affine moments ``B=sum(v)`` and
``M=sum(k outer v)`` inside each cone.  Persistent state is their signed
Current-minus-Parent difference.

The coherent reader keeps the full Parent prefix response as a control path
and adds ``signed_B + scale * q @ signed_M`` before the native output
projection, gate, and residual update.  The Current half can be either an
all-position oracle moment or a fixed-position, positive-weighted estimate;
the Parent half is always constructed from the complete Parent cache.

Medium's supported path is legacy ELU+1 attention without relative-position
bias.  Training mode, another activation, or relative bias is rejected rather
than silently changing the algebra.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hstu_kvcache.models import HSTUKVCache
from insight_two.attention_cone_moments import (
    AffineConeMoments,
    build_positive_affine_moments,
    scaled_qk_logits,
)


REQUIRED_ANCHOR_COUNT = 32


@dataclass(frozen=True)
class ConeLayerResponseMoment:
    """One layer's signed positive-branch affine response state."""

    base: torch.Tensor
    linear: torch.Tensor
    current_positive_mask: torch.Tensor
    parent_positive_mask: torch.Tensor
    current_sample_positions: torch.Tensor
    current_sample_weights: torch.Tensor
    source_length: int

    @property
    def stored_scalars(self) -> int:
        """Persistent floating-point sidecar size; audit masks are excluded."""

        return self.base.numel() + self.linear.numel()

    @property
    def stored_bytes(self) -> int:
        return self.stored_scalars * self.base.element_size()

    @property
    def uses_full_current(self) -> bool:
        if self.current_sample_positions.numel() != self.source_length:
            return False
        expected = torch.arange(
            self.source_length,
            device=self.current_sample_positions.device,
            dtype=torch.long,
        )
        return bool(
            torch.equal(self.current_sample_positions, expected)
            and torch.equal(
                self.current_sample_weights,
                torch.ones_like(self.current_sample_weights),
            )
        )


@dataclass(frozen=True)
class ConeResponseMemory:
    """All-layer signed cone moments and auditable storage accounting."""

    layers: tuple[ConeLayerResponseMoment, ...]
    source_length: int
    anchor_count: int
    source_kv_scalars: int

    @property
    def stored_scalars(self) -> int:
        return sum(layer.stored_scalars for layer in self.layers)

    @property
    def stored_bytes(self) -> int:
        return sum(layer.stored_bytes for layer in self.layers)

    @property
    def storage_ratio_to_current_kv(self) -> float:
        return self.stored_scalars / self.source_kv_scalars


@dataclass(frozen=True)
class ConeResponseIntervention:
    """Outputs of the coherent signed-moment reader intervention."""

    scores: torch.Tensor
    readout: torch.Tensor
    layer_signed_heads: tuple[torch.Tensor, ...]


def _validate_attention(attention) -> None:
    if attention.training:
        raise ValueError("cone response memory requires attention eval mode")
    if attention.activation != "elu_plus1":
        raise ValueError("cone response memory requires elu_plus1 attention")
    if attention.position_bias is not None:
        raise ValueError("cone response memory does not support relative-position bias")


def _validate_model(model) -> None:
    if model.training:
        raise ValueError("cone response memory requires model.eval()")
    if not model.blocks:
        raise ValueError("model must contain at least one attention block")
    for block in model.blocks:
        _validate_attention(block.attn)


def _validate_cache(cache: HSTUKVCache, name: str) -> None:
    if cache.k.ndim != 4 or cache.k.shape != cache.v.shape:
        raise ValueError(f"{name} must contain matching [layers,B,N,width] K/V")
    if cache.k.shape[2] != cache.seq_len:
        raise ValueError(f"{name} seq_len differs from its tensor history width")
    if cache.k.shape[1] != 1:
        raise ValueError(f"{name} must contain exactly one user's cache")
    if not cache.k.is_floating_point() or not cache.v.is_floating_point():
        raise ValueError(f"{name} K/V must be floating point")


def _validate_cache_pair(exact_cache: HSTUKVCache, reuse_cache: HSTUKVCache) -> None:
    _validate_cache(exact_cache, "exact_cache")
    _validate_cache(reuse_cache, "reuse_cache")
    if exact_cache.k.shape != reuse_cache.k.shape:
        raise ValueError("Exact and Reuse cache shapes differ")
    if exact_cache.seq_len != reuse_cache.seq_len:
        raise ValueError("Exact and Reuse cache lengths differ")
    if exact_cache.k.device != reuse_cache.k.device:
        raise ValueError("Exact and Reuse caches must share a device")
    if exact_cache.k.dtype != reuse_cache.k.dtype:
        raise ValueError("Exact and Reuse caches must share a dtype")


def _layer_cache_heads(attention, values: torch.Tensor, name: str) -> torch.Tensor:
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != attention.inner:
        raise ValueError(f"{name} must have shape [1,N,attention.inner]")
    return values.view(
        1, values.shape[1], attention.num_heads, attention.head_dim
    ).transpose(1, 2)


def majority_positive_cone(
    attention,
    anchor_q: torch.Tensor,
    cache_k: torch.Tensor,
) -> torch.Tensor:
    """Return ``[1,H,N]`` positions receiving at least half positive votes."""

    _validate_attention(attention)
    if anchor_q.shape != (
        REQUIRED_ANCHOR_COUNT,
        attention.num_heads,
        1,
        attention.head_dim,
    ):
        raise ValueError(
            "anchor_q must have shape "
            f"[{REQUIRED_ANCHOR_COUNT},heads,1,head_dim]"
        )
    keys = _layer_cache_heads(attention, cache_k, "cache_k")
    keys = keys.expand(REQUIRED_ANCHOR_COUNT, -1, -1, -1)
    logits = scaled_qk_logits(anchor_q, keys, scale=attention.scale)
    votes = (logits >= 0).sum(dim=0).squeeze(1)
    return (2 * votes >= REQUIRED_ANCHOR_COUNT).unsqueeze(0)


def _full_positions_and_weights(
    source_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.arange(source_length, device=device, dtype=torch.long),
        torch.ones(source_length, device=device, dtype=dtype),
    )


def _validate_samples(
    source_length: int,
    positions: torch.Tensor,
    weights: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = positions.to(device=device, dtype=torch.long)
    weights = weights.to(device=device, dtype=dtype)
    if positions.ndim != 1 or weights.shape != positions.shape:
        raise ValueError("sample positions and weights must be matching vectors")
    if positions.numel() < 1:
        raise ValueError("at least one Current sample is required")
    if bool((positions < 0).any()) or bool((positions >= source_length).any()):
        raise ValueError("Current sample position is outside the cache")
    if torch.unique(positions).numel() != positions.numel():
        raise ValueError("Current sample positions must be unique")
    if not torch.isfinite(weights).all() or bool((weights <= 0).any()):
        raise ValueError("Current sample weights must be finite and positive")
    return positions, weights


def _moments_for_cache(
    attention,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    positive_mask: torch.Tensor,
    positions: torch.Tensor,
    weights: torch.Tensor,
) -> AffineConeMoments:
    keys = _layer_cache_heads(attention, cache_k, "cache_k")
    values = _layer_cache_heads(attention, cache_v, "cache_v")
    keys = keys.index_select(2, positions)
    values = values.index_select(2, positions)
    selected_mask = positive_mask.index_select(2, positions)
    weighted_values = values * weights.view(1, 1, -1, 1)
    return build_positive_affine_moments(keys, weighted_values, selected_mask)


@torch.inference_mode()
def build_layer_signed_cone_moment(
    attention,
    anchor_q: torch.Tensor,
    exact_k: torch.Tensor,
    exact_v: torch.Tensor,
    reuse_k: torch.Tensor,
    reuse_v: torch.Tensor,
    *,
    current_sample_positions: torch.Tensor | None = None,
    current_sample_weights: torch.Tensor | None = None,
) -> ConeLayerResponseMoment:
    """Build one signed layer moment; Parent always uses every position."""

    _validate_attention(attention)
    exact_keys = _layer_cache_heads(attention, exact_k, "exact_k")
    reuse_keys = _layer_cache_heads(attention, reuse_k, "reuse_k")
    if exact_k.shape != exact_v.shape or reuse_k.shape != reuse_v.shape:
        raise ValueError("each layer's K/V shapes must match")
    if exact_k.shape != reuse_k.shape:
        raise ValueError("Exact and Reuse layer cache shapes differ")
    source_length = exact_k.shape[1]

    current_mask = majority_positive_cone(attention, anchor_q, exact_k)
    parent_mask = majority_positive_cone(attention, anchor_q, reuse_k)
    if current_sample_positions is None and current_sample_weights is None:
        positions, weights = _full_positions_and_weights(
            source_length, device=exact_k.device, dtype=exact_v.dtype
        )
    elif current_sample_positions is None or current_sample_weights is None:
        raise ValueError("Current sample positions and weights must be supplied together")
    else:
        positions, weights = _validate_samples(
            source_length,
            current_sample_positions,
            current_sample_weights,
            device=exact_k.device,
            dtype=exact_v.dtype,
        )

    current = _moments_for_cache(
        attention, exact_k, exact_v, current_mask, positions, weights
    )
    parent_positions, parent_weights = _full_positions_and_weights(
        source_length, device=reuse_k.device, dtype=reuse_v.dtype
    )
    parent = _moments_for_cache(
        attention,
        reuse_k,
        reuse_v,
        parent_mask,
        parent_positions,
        parent_weights,
    )
    return ConeLayerResponseMoment(
        base=(current.base - parent.base).detach(),
        linear=(current.linear - parent.linear).detach(),
        current_positive_mask=current_mask.detach(),
        parent_positive_mask=parent_mask.detach(),
        current_sample_positions=positions.detach(),
        current_sample_weights=weights.detach(),
        source_length=source_length,
    )


def _native_prefix_heads(
    attention,
    q: torch.Tensor,
    layer_k: torch.Tensor,
    layer_v: torch.Tensor,
) -> torch.Tensor:
    keys = _layer_cache_heads(attention, layer_k, "layer_k")
    values = _layer_cache_heads(attention, layer_v, "layer_v")
    keys = keys.expand(q.shape[0], -1, -1, -1)
    values = values.expand(q.shape[0], -1, -1, -1)
    weights = scaled_qk_logits(q, keys, scale=attention.scale)
    weights = attention._activate(weights)
    if attention.block_variant == "hstu_reference":
        weights = weights / attention.cfg.max_seq_len
    return torch.matmul(attention.attn_dropout(weights), values)


def _native_self_heads(
    attention,
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
) -> torch.Tensor:
    if attention.causal_diagonal != "inclusive":
        return torch.zeros_like(v_new)
    weights = (q * k_new).sum(dim=-1, keepdim=True) * attention.scale
    weights = attention._activate(weights)
    if attention.block_variant == "hstu_reference":
        weights = weights / attention.cfg.max_seq_len
    return attention.attn_dropout(weights) * v_new


def _read_signed_moment(
    attention,
    q: torch.Tensor,
    moment: ConeLayerResponseMoment,
) -> torch.Tensor:
    base = moment.base.to(device=q.device, dtype=q.dtype).expand(q.shape[0], -1, -1)
    linear = moment.linear.to(device=q.device, dtype=q.dtype).expand(
        q.shape[0], -1, -1, -1
    )
    out = base.unsqueeze(2) + attention.scale * torch.einsum(
        "bhqk,bhkv->bhqv", q, linear
    )
    if attention.block_variant == "hstu_reference":
        out = out / attention.cfg.max_seq_len
    return out


def _block_update(block, x_norm: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
    attention_out = block.attn._finish(heads)
    if block.block_variant == "hstu_reference":
        assert block.attn_output_norm is not None
        return block.attn.out_proj(
            block.attn_output_norm(attention_out) * F.silu(block.gate_proj(x_norm))
        )
    if block.gating == "silu_gate":
        return attention_out * F.silu(block.gate_proj(x_norm))
    if block.gating == "glu":
        return attention_out * torch.sigmoid(block.gate_proj(x_norm))
    if block.gating == "ffn":
        return block.fc2(F.silu(block.fc1(x_norm)) * block.fc3(x_norm))
    return attention_out


def _anchor_query_input(
    model,
    anchor_candidate_ids: torch.Tensor,
    anchor_query_time_deltas: torch.Tensor,
    *,
    query_type_ids: torch.Tensor | None,
    query_action_ids: torch.Tensor | None,
    candidate_item_vectors: torch.Tensor | None,
) -> torch.Tensor:
    if anchor_candidate_ids.shape != (1, REQUIRED_ANCHOR_COUNT):
        raise ValueError(
            f"anchor_candidate_ids must have shape [1,{REQUIRED_ANCHOR_COUNT}]"
        )
    return model.embed_query_tokens(
        anchor_candidate_ids,
        anchor_query_time_deltas,
        query_type_ids=query_type_ids,
        query_action_ids=query_action_ids,
        item_vectors=candidate_item_vectors,
    ).reshape(REQUIRED_ANCHOR_COUNT, 1, model.cfg.hidden_size)


@torch.inference_mode()
def build_cone_response_memory(
    model,
    exact_cache: HSTUKVCache,
    reuse_cache: HSTUKVCache,
    anchor_candidate_ids: torch.Tensor,
    anchor_query_time_deltas: torch.Tensor,
    *,
    current_sample_positions: torch.Tensor | None = None,
    current_sample_weights: torch.Tensor | None = None,
    query_type_ids: torch.Tensor | None = None,
    query_action_ids: torch.Tensor | None = None,
    candidate_item_vectors: torch.Tensor | None = None,
) -> ConeResponseMemory:
    """Build all layer moments from Current-reader-over-Parent anchor queries."""

    _validate_model(model)
    _validate_cache_pair(exact_cache, reuse_cache)
    if exact_cache.k.shape[0] != len(model.blocks):
        raise ValueError("cache and model layer counts differ")
    x = _anchor_query_input(
        model,
        anchor_candidate_ids,
        anchor_query_time_deltas,
        query_type_ids=query_type_ids,
        query_action_ids=query_action_ids,
        candidate_item_vectors=candidate_item_vectors,
    )
    layer_moments: list[ConeLayerResponseMoment] = []
    for layer, block in enumerate(model.blocks):
        residual = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        layer_moments.append(
            build_layer_signed_cone_moment(
                block.attn,
                q,
                exact_cache.k[layer],
                exact_cache.v[layer],
                reuse_cache.k[layer],
                reuse_cache.v[layer],
                current_sample_positions=current_sample_positions,
                current_sample_weights=current_sample_weights,
            )
        )
        # Anchors intentionally follow the uncorrected coherent Reuse reader.
        reuse_heads = _native_prefix_heads(
            block.attn, q, reuse_cache.k[layer], reuse_cache.v[layer]
        )
        self_heads = _native_self_heads(block.attn, q, k_new, v_new)
        x = residual + _block_update(block, x_norm, reuse_heads + self_heads)

    return ConeResponseMemory(
        layers=tuple(layer_moments),
        source_length=exact_cache.seq_len,
        anchor_count=REQUIRED_ANCHOR_COUNT,
        source_kv_scalars=exact_cache.k.numel() + exact_cache.v.numel(),
    )


@torch.inference_mode()
def intervene_cone_response_memory(
    model,
    reuse_cache: HSTUKVCache,
    memory: ConeResponseMemory,
    candidate_ids: torch.Tensor,
    query_time_deltas: torch.Tensor,
    *,
    query_type_ids: torch.Tensor | None = None,
    query_action_ids: torch.Tensor | None = None,
    candidate_item_vectors: torch.Tensor | None = None,
) -> ConeResponseIntervention:
    """Read Reuse plus signed moments coherently through every Current layer."""

    _validate_model(model)
    _validate_cache(reuse_cache, "reuse_cache")
    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != 1 or candidate_ids.shape[1] < 1:
        raise ValueError("candidate_ids must have shape [1,C] with C >= 1")
    if len(memory.layers) != len(model.blocks):
        raise ValueError("memory and model layer counts differ")
    if reuse_cache.seq_len != memory.source_length:
        raise ValueError("Reuse cache and memory source lengths differ")
    if reuse_cache.k.shape[0] != len(model.blocks):
        raise ValueError("Reuse cache and model layer counts differ")

    candidates = candidate_ids.shape[1]
    x = model.embed_query_tokens(
        candidate_ids,
        query_time_deltas,
        query_type_ids=query_type_ids,
        query_action_ids=query_action_ids,
        item_vectors=candidate_item_vectors,
    ).reshape(candidates, 1, model.cfg.hidden_size)
    corrections: list[torch.Tensor] = []
    for layer, (block, moment) in enumerate(zip(model.blocks, memory.layers)):
        residual = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        reuse_heads = _native_prefix_heads(
            block.attn, q, reuse_cache.k[layer], reuse_cache.v[layer]
        )
        signed_heads = _read_signed_moment(block.attn, q, moment)
        self_heads = _native_self_heads(block.attn, q, k_new, v_new)
        corrections.append(signed_heads.reshape(
            1, candidates, block.attn.num_heads, block.attn.head_dim
        ))
        x = residual + _block_update(
            block, x_norm, reuse_heads + signed_heads + self_heads
        )

    readout = model.final_norm(x).reshape(1, candidates, model.cfg.hidden_size)
    return ConeResponseIntervention(
        scores=model.cc_score_head(readout).squeeze(-1),
        readout=readout,
        layer_signed_heads=tuple(corrections),
    )
