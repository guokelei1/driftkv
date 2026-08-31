"""Lightweight Per-user Reader Offset (PRO) primitives.

The serving artifact is a layerwise AV sidecar.  This module never builds or
returns a version-translated prefix cache.  A fixed joint Parent->Current map
is instead pushed into the one-probe HSTU read:

    K' = [K; V] A_K, V' = [K; V] A_V
    activated(q K'^T) V'
      = (activated((q A_K^T) [K; V]^T) [K; V]) A_V.

Recent Current carriers are dependency-closed against the unmodified Parent
prefix.  This keeps their contextual replay at ordinary HSTU attention cost;
applying the fused map to every carrier query would repeat the wide joint-state
scan and defeat the <=20%-of-Full budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

if __package__:  # Package import from the rolling evaluator.
    from .candidate_shared_causal import _block_update, _cached_prefix_heads
    from .reader_compatibility_correction import _self_heads, intervene_reader_correction
else:  # Direct script/test import with scripts/insight on sys.path.
    from candidate_shared_causal import _block_update, _cached_prefix_heads
    from reader_compatibility_correction import _self_heads, intervene_reader_correction
from hstu_kvcache.models import HSTUKVCache


def pro_path(carrier_count: int) -> str:
    if carrier_count < 1:
        raise ValueError("carrier count must be positive")
    return f"evokv_pro_lazy_reader_c{carrier_count}_rolling"


PRO_PATH = pro_path(32)


@dataclass(frozen=True)
class PROCarrierLayout:
    nominal_positions: int
    old_positions: int
    repair_evidence: int
    carriers: int
    represented_mass: int


@dataclass(frozen=True)
class PROSidecar:
    corrections: tuple[torch.Tensor, ...]
    replay_max_abs_error: float
    source_scores: torch.Tensor


@dataclass(frozen=True)
class PROProbeComponents:
    """One probe split into old-coordinate and recent-context corrections."""

    corrections: tuple[torch.Tensor, ...]
    old_corrections: tuple[torch.Tensor, ...]
    recent_corrections: tuple[torch.Tensor, ...]
    replay_max_abs_error: float
    source_scores: torch.Tensor


def _validate_joint_map(attention, mapping: torch.Tensor) -> None:
    inner = attention.inner
    if mapping.shape != (2 * inner, 2 * inner):
        raise ValueError("joint version map must have shape [2*inner, 2*inner]")
    if attention.position_bias is not None:
        raise ValueError("fused PRO aggregation requires no relative-position bias")


def _cache_slice(cache: HSTUKVCache, start: int, stop: int) -> HSTUKVCache:
    if not 0 <= start <= stop <= cache.seq_len:
        raise ValueError("cache slice is outside the persistent state")
    return HSTUKVCache(
        k=cache.k[:, :, start:stop],
        v=cache.v[:, :, start:stop],
        seq_len=stop - start,
    )


def fused_joint_map_prefix_heads(
    attention,
    q: torch.Tensor,
    parent_cache: HSTUKVCache,
    mapping: torch.Tensor,
    *,
    layer: int,
    length: int,
) -> torch.Tensor:
    """Read a mapped Parent prefix without materialising mapped K/V.

    ``q`` has the native attention layout ``[B, heads, queries, head_dim]``.
    The result has the same layout.  The operation is algebraically identical
    to first applying the joint map to every Parent K/V row and then reading
    the mapped cache, up to floating-point reassociation.
    """
    _validate_joint_map(attention, mapping)
    if q.ndim != 4:
        raise ValueError("query must have shape [B, heads, queries, head_dim]")
    if q.shape[0] != parent_cache.k.shape[1]:
        raise ValueError("query and Parent cache batch dimensions differ")
    if q.shape[1] != attention.num_heads or q.shape[3] != attention.head_dim:
        raise ValueError("query head layout differs from attention")
    if not 0 <= length <= parent_cache.seq_len:
        raise ValueError("mapped prefix length is outside the Parent cache")
    if not 0 <= layer < parent_cache.k.shape[0]:
        raise ValueError("mapped prefix layer is outside the Parent cache")

    inner = attention.inner
    source = torch.cat(
        [
            parent_cache.k[layer, :, :length].float(),
            parent_cache.v[layer, :, :length].float(),
        ],
        dim=-1,
    )
    version_map = mapping.to(device=q.device, dtype=torch.float32)
    key_map = version_map[:, :inner].reshape(
        2 * inner, attention.num_heads, attention.head_dim
    )
    value_map = version_map[:, inner:].reshape(
        2 * inner, attention.num_heads, attention.head_dim
    )

    # Move the key-side map to the (single/few) probe queries, then stream the
    # original Parent joint state.  The value-side map is applied once after
    # the weighted history sum.  No [B, length, 2*inner] mapped cache is built.
    source_query = torch.einsum("bhqd,zhd->bhqz", q.float(), key_map)
    weights = torch.einsum("bhqz,bnz->bhqn", source_query, source)
    weights = attention._activate(weights * attention.scale)
    if attention.block_variant == "hstu_reference":
        weights = weights / attention.cfg.max_seq_len
    weights = attention.attn_dropout(weights)
    source_measure = torch.einsum("bhqn,bnz->bhqz", weights, source)
    heads = torch.einsum("bhqz,zhd->bhqd", source_measure, value_map)
    return heads.to(q.dtype)


@torch.inference_mode()
def build_parent_conditioned_carriers(
    *,
    parent_cache: HSTUKVCache,
    current,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    repair_width: int = 128,
    carrier_count: int = 32,
) -> tuple[HSTUKVCache, PROCarrierLayout]:
    """Replay fixed recent-history carriers against the unmodified Parent prefix.

    Only the newly produced carrier K/V is returned.  The Parent prefix remains
    the persistent serving state and no translated prefix state is allocated or
    written back.
    """
    if item_ids.shape != behaviors.shape or item_ids.shape != time_deltas.shape:
        raise ValueError("raw prefix tensors differ in shape")
    if item_ids.ndim != 2 or item_ids.shape[0] != parent_cache.k.shape[1]:
        raise ValueError("raw prefix and Parent cache batch dimensions differ")
    if item_ids.shape[1] != parent_cache.seq_len:
        raise ValueError("raw prefix width differs from Parent cache")
    nominal = parent_cache.seq_len
    if not 1 <= repair_width <= nominal:
        raise ValueError("repair width must be within the Parent cache")
    if carrier_count < 1 or repair_width % carrier_count:
        raise ValueError("carrier count must divide the repair width")

    old_positions = nominal - repair_width
    represented_mass = repair_width // carrier_count
    endpoints = torch.arange(
        represented_mass - 1,
        repair_width,
        represented_mass,
        dtype=torch.long,
        device=item_ids.device,
    )
    recent_items = item_ids[:, -repair_width:].index_select(1, endpoints)
    recent_behaviors = behaviors[:, -repair_width:].index_select(1, endpoints)
    recent_deltas = time_deltas[:, -repair_width:].index_select(1, endpoints)
    embedded = current.embed_inputs(recent_items, recent_behaviors, recent_deltas)
    parent_prefix = HSTUKVCache(
        k=parent_cache.k[:, :, :old_positions],
        v=parent_cache.v[:, :, :old_positions],
        seq_len=old_positions,
    )
    _, carriers = current.forward_with_cache_embedded_new_kv(parent_prefix, embedded)
    scaled_values = carriers.v.clone()
    scaled_values *= float(represented_mass)
    carriers = HSTUKVCache(k=carriers.k, v=scaled_values, seq_len=carrier_count)
    return carriers, PROCarrierLayout(
        nominal_positions=nominal,
        old_positions=old_positions,
        repair_evidence=repair_width,
        carriers=carrier_count,
        represented_mass=represented_mass,
    )


@torch.inference_mode()
def generate_lazy_pro_probe_components(
    model,
    parent_reuse_cache: HSTUKVCache,
    carrier_cache: HSTUKVCache,
    joint_maps: tuple[torch.Tensor, ...],
    probe_item_ids: torch.Tensor,
    *,
    old_positions: int,
) -> PROProbeComponents:
    """Generate and decompose one fused-probe AV correction.

    The split is exact for the executed probe because HSTU's activated history
    read is an unnormalised sum over positions.  It exposes two scalar-amplitude
    sources for progressive PRO without materialising either prefix segment.
    """
    if probe_item_ids.ndim != 1:
        raise ValueError("probe item IDs must have shape [B]")
    if probe_item_ids.shape[0] != parent_reuse_cache.k.shape[1]:
        raise ValueError("probe and Parent cache batch dimensions differ")
    if carrier_cache.k.shape[1] != probe_item_ids.shape[0]:
        raise ValueError("probe and carrier cache batch dimensions differ")
    if len(joint_maps) != len(model.blocks):
        raise ValueError("joint version map count differs from model layers")
    if not 0 <= old_positions <= parent_reuse_cache.seq_len:
        raise ValueError("old-position boundary differs from Parent cache")
    if model.cfg.relative_position_bias:
        raise ValueError("lazy PRO requires no relative-position bias")

    candidates = probe_item_ids[:, None]
    deltas = torch.zeros(
        probe_item_ids.shape[0], dtype=torch.float32, device=probe_item_ids.device
    )
    batch = probe_item_ids.shape[0]
    x = model.embed_query_tokens(candidates, deltas).reshape(
        batch, 1, model.cfg.hidden_size
    )
    corrections: list[torch.Tensor] = []
    old_corrections: list[torch.Tensor] = []
    recent_corrections: list[torch.Tensor] = []
    old_reuse_cache = _cache_slice(parent_reuse_cache, 0, old_positions)
    recent_reuse_cache = _cache_slice(
        parent_reuse_cache, old_positions, parent_reuse_cache.seq_len
    )

    for layer, block in enumerate(model.blocks):
        residual_x = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        mapped_old_heads = fused_joint_map_prefix_heads(
            block.attn,
            q,
            parent_reuse_cache,
            joint_maps[layer],
            layer=layer,
            length=old_positions,
        )
        carrier_heads = _cached_prefix_heads(
            block.attn, q, carrier_cache, layer, 1
        ).reshape(batch, block.attn.num_heads, 1, block.attn.head_dim)
        reuse_old_heads = _cached_prefix_heads(
            block.attn, q, old_reuse_cache, layer, 1
        ).reshape(batch, block.attn.num_heads, 1, block.attn.head_dim)
        reuse_recent_heads = _cached_prefix_heads(
            block.attn, q, recent_reuse_cache, layer, 1
        ).reshape(batch, block.attn.num_heads, 1, block.attn.head_dim)
        source_prefix_heads = mapped_old_heads + carrier_heads
        old_correction = (mapped_old_heads - reuse_old_heads).squeeze(2)
        recent_correction = (carrier_heads - reuse_recent_heads).squeeze(2)
        correction = old_correction + recent_correction
        old_corrections.append(old_correction.detach())
        recent_corrections.append(recent_correction.detach())
        corrections.append(correction.detach())

        source_av = source_prefix_heads + _self_heads(block, q, k_new, v_new)
        update = _block_update(block, x_norm, source_av)
        x = residual_x + update

    readout = model.final_norm(x).reshape(batch, 1, model.cfg.hidden_size)
    source_scores = model.cc_score_head(readout).squeeze(-1)
    replay_scores, _ = intervene_reader_correction(
        model,
        parent_reuse_cache,
        candidates,
        deltas,
        stage="av_aggregation",
        corrections=tuple(corrections),
    )
    return PROProbeComponents(
        corrections=tuple(corrections),
        old_corrections=tuple(old_corrections),
        recent_corrections=tuple(recent_corrections),
        replay_max_abs_error=float(torch.max(torch.abs(replay_scores - source_scores))),
        source_scores=source_scores.detach(),
    )


@torch.inference_mode()
def generate_lazy_pro_sidecar(
    model,
    parent_reuse_cache: HSTUKVCache,
    carrier_cache: HSTUKVCache,
    joint_maps: tuple[torch.Tensor, ...],
    probe_item_ids: torch.Tensor,
    *,
    old_positions: int,
) -> PROSidecar:
    """Generate the frozen v1 sidecar while retaining its original API."""
    components = generate_lazy_pro_probe_components(
        model,
        parent_reuse_cache,
        carrier_cache,
        joint_maps,
        probe_item_ids,
        old_positions=old_positions,
    )
    return PROSidecar(
        corrections=components.corrections,
        replay_max_abs_error=components.replay_max_abs_error,
        source_scores=components.source_scores,
    )
