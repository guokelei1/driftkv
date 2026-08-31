"""Fixed one-release EvoKV refinement used by the rolling AUC experiment.

This module implements the single frozen plan only.  It does not contain a
scheduler or any multi-release policy:

    CAST(old prefix) -> GROUP(recent evidence by pairs)
    -> PATCH(Current carriers) -> SCALE(represented mass)

The zero prefix is an execution-only padding representation.  For the frozen
HSTU checkpoints, which have no relative-position bias, zero K/V positions
contribute exactly zero to the unnormalised read.  Padding therefore keeps all
rolling paths at the same nominal cache width while the compact state contains
fewer materialised positions; padding is evicted before real state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hstu_kvcache.models import HSTUKVCache


REPAIR_WIDTH = 128
GROUP_SIZE = 2
OUR_PATH = "evokv_cast_group_patch_scale_r128_c64_rolling"
EVIDENCE_MEASURE_PATH = "evokv_cast_measure_current_residual_r128_c64_rolling"
BROADCAST_RESIDUAL_PATH = "compact_probe_AV_broadcast_residual_rolling"


@dataclass(frozen=True)
class RefinementLayout:
    nominal_positions: int
    cast_positions: int
    repair_evidence: int
    carriers: int
    padding_positions: int


def parameter_cast_maps(parent, current) -> tuple[torch.Tensor, ...]:
    """Construct one parameter-only joint K/V CAST map per layer.

    The maps use Parent and Current parameters only.  No request label or
    Current target K/V is fitted.
    """
    if len(parent.blocks) != len(current.blocks):
        raise ValueError("Parent and Current layer counts differ")
    maps = []
    for parent_block, current_block in zip(parent.blocks, current.blocks, strict=True):
        parent_projection = torch.cat(
            [parent_block.attn.k_proj.weight.T.float(), parent_block.attn.v_proj.weight.T.float()],
            dim=1,
        )
        current_projection = torch.cat(
            [
                current_block.attn.k_proj.weight.T.float(),
                current_block.attn.v_proj.weight.T.float(),
            ],
            dim=1,
        )
        if parent_projection.shape != current_projection.shape:
            raise ValueError("Parent and Current K/V projection shapes differ")
        norm_scale = current_block.norm.weight.float() / parent_block.norm.weight.float().clamp_min(
            1e-8
        )
        maps.append(
            torch.linalg.pinv(parent_projection)
            @ torch.diag(norm_scale)
            @ current_projection
        )
    return tuple(maps)


@torch.inference_mode()
def cast_prefix(
    cache: HSTUKVCache,
    maps: tuple[torch.Tensor, ...],
    length: int,
) -> HSTUKVCache:
    """CAST only the prefix that will not be overwritten by PATCH."""
    if not 0 <= length <= cache.seq_len:
        raise ValueError("CAST prefix length is outside the cache")
    if len(maps) != cache.k.shape[0]:
        raise ValueError("CAST map count differs from cache layer count")
    width = cache.k.shape[-1]
    translated_k, translated_v = [], []
    for layer, mapping in enumerate(maps):
        source = torch.cat(
            [cache.k[layer, :, :length].float(), cache.v[layer, :, :length].float()],
            dim=-1,
        )
        target = source @ mapping
        translated_k.append(target[..., :width].to(cache.k.dtype))
        translated_v.append(target[..., width:].to(cache.v.dtype))
    return HSTUKVCache(
        k=torch.stack(translated_k),
        v=torch.stack(translated_v),
        seq_len=length,
    )


@torch.inference_mode()
def _cast_slice(
    cache: HSTUKVCache,
    maps: tuple[torch.Tensor, ...],
    start: int,
    end: int,
) -> HSTUKVCache:
    """Apply the frozen joint K/V CAST map to one contiguous cache slice."""
    if not 0 <= start <= end <= cache.seq_len:
        raise ValueError("CAST slice is outside the cache")
    if len(maps) != cache.k.shape[0]:
        raise ValueError("CAST map count differs from cache layer count")
    width = cache.k.shape[-1]
    translated_k, translated_v = [], []
    for layer, mapping in enumerate(maps):
        source = torch.cat(
            [cache.k[layer, :, start:end].float(), cache.v[layer, :, start:end].float()],
            dim=-1,
        )
        target = source @ mapping
        translated_k.append(target[..., :width].to(cache.k.dtype))
        translated_v.append(target[..., width:].to(cache.v.dtype))
    return HSTUKVCache(
        k=torch.stack(translated_k),
        v=torch.stack(translated_v),
        seq_len=end - start,
    )


@torch.inference_mode()
def _cast_values_at(
    cache: HSTUKVCache,
    maps: tuple[torch.Tensor, ...],
    positions: torch.Tensor,
) -> torch.Tensor:
    """CAST only V at selected positions, using half of a joint-map matmul."""
    if positions.ndim != 1:
        raise ValueError("CAST value positions must be one-dimensional")
    if positions.numel() and (
        int(positions.min()) < 0 or int(positions.max()) >= cache.seq_len
    ):
        raise ValueError("CAST value positions are outside the cache")
    if len(maps) != cache.k.shape[0]:
        raise ValueError("CAST map count differs from cache layer count")
    width = cache.k.shape[-1]
    translated = []
    for layer, mapping in enumerate(maps):
        source = torch.cat(
            [
                cache.k[layer].index_select(1, positions).float(),
                cache.v[layer].index_select(1, positions).float(),
            ],
            dim=-1,
        )
        translated.append((source @ mapping[:, width:]).to(cache.v.dtype))
    return torch.stack(translated)


def _pair_endpoints_and_mass(width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if width < 1:
        raise ValueError("repair width must be positive")
    endpoints = torch.arange(1, width, GROUP_SIZE, dtype=torch.long, device=device)
    masses = torch.full(
        (width // GROUP_SIZE,), float(GROUP_SIZE), dtype=torch.float32, device=device
    )
    if width % GROUP_SIZE:
        endpoints = torch.cat(
            [endpoints, torch.tensor([width - 1], dtype=torch.long, device=device)]
        )
        masses = torch.cat([masses, torch.ones(1, dtype=torch.float32, device=device)])
    return endpoints, masses


@torch.inference_mode()
def build_fixed_refinement_cache(
    *,
    parent_cache: HSTUKVCache,
    current,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    cast_maps: tuple[torch.Tensor, ...],
    repair_width: int = REPAIR_WIDTH,
) -> tuple[HSTUKVCache, RefinementLayout]:
    """Build the fixed one-hop CAST + pair-GROUP/PATCH/SCALE state.

    Histories of at least 128 positions use exactly ``r=128,c=64``.  For the
    small fraction of shorter formal histories, the same label-free rule is
    applied to the available prefix: repair all evidence in adjacent pairs and
    retain one carrier per pair.
    """
    if item_ids.shape != behaviors.shape or item_ids.shape != time_deltas.shape:
        raise ValueError("raw prefix tensors differ in shape")
    if item_ids.ndim != 2 or item_ids.shape[0] != parent_cache.k.shape[1]:
        raise ValueError("raw prefix and Parent cache batch dimensions differ")
    if item_ids.shape[1] != parent_cache.seq_len:
        raise ValueError("raw prefix width differs from Parent cache")
    if repair_width < 1:
        raise ValueError("repair width must be positive")
    if current.cfg.relative_position_bias:
        raise ValueError("zero-padded compact refinement requires no relative-position bias")

    nominal = parent_cache.seq_len
    repair = min(repair_width, nominal)
    cast_length = nominal - repair
    endpoints, masses = _pair_endpoints_and_mass(repair, item_ids.device)
    carriers = int(endpoints.numel())

    cast = cast_prefix(parent_cache, cast_maps, cast_length)
    repair_items = item_ids[:, -repair:].index_select(1, endpoints)
    repair_behaviors = behaviors[:, -repair:].index_select(1, endpoints)
    repair_deltas = time_deltas[:, -repair:].index_select(1, endpoints)
    embedded = current.embed_inputs(repair_items, repair_behaviors, repair_deltas)
    _, compact = current.forward_with_cache_embedded(cast, embedded)

    values = compact.v.clone()
    values[:, :, cast_length:] *= masses.to(values.dtype).view(1, 1, carriers, 1)
    compact = HSTUKVCache(k=compact.k, v=values, seq_len=cast_length + carriers)

    padding = nominal - compact.seq_len
    if padding:
        zero_k = compact.k.new_zeros(
            compact.k.shape[0], compact.k.shape[1], padding, compact.k.shape[-1]
        )
        zero_v = compact.v.new_zeros(
            compact.v.shape[0], compact.v.shape[1], padding, compact.v.shape[-1]
        )
        padded = HSTUKVCache(
            k=torch.cat([zero_k, compact.k], dim=2),
            v=torch.cat([zero_v, compact.v], dim=2),
            seq_len=nominal,
        )
    else:
        padded = compact
    return padded, RefinementLayout(
        nominal_positions=nominal,
        cast_positions=cast_length,
        repair_evidence=repair,
        carriers=carriers,
        padding_positions=padding,
    )


@torch.inference_mode()
def build_broadcast_probe_source_cache(
    *,
    parent_cache: HSTUKVCache,
    current,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    cast_maps: tuple[torch.Tensor, ...],
    repair_width: int = REPAIR_WIDTH,
) -> tuple[HSTUKVCache, RefinementLayout]:
    """Build the frozen disposable 32-carrier Current probe source.

    This is not a serving cache.  It spends half of Design 0's Current
    carriers on fixed consecutive groups of four, preserving their represented
    mass, so that two one-token reader probes still remain below the frozen
    Design-0 Current-compute budget.
    """
    if item_ids.shape != behaviors.shape or item_ids.shape != time_deltas.shape:
        raise ValueError("raw prefix tensors differ in shape")
    if item_ids.ndim != 2 or item_ids.shape[0] != parent_cache.k.shape[1]:
        raise ValueError("raw prefix and Parent cache batch dimensions differ")
    if item_ids.shape[1] != parent_cache.seq_len:
        raise ValueError("raw prefix width differs from Parent cache")
    nominal = parent_cache.seq_len
    repair = min(repair_width, nominal)
    cast_length = nominal - repair
    group_size = 4
    endpoints = torch.arange(
        group_size - 1, repair, group_size, dtype=torch.long, device=item_ids.device
    )
    masses = torch.full(
        (repair // group_size,), float(group_size), dtype=torch.float32, device=item_ids.device
    )
    remainder = repair % group_size
    if remainder:
        endpoints = torch.cat(
            [endpoints, torch.tensor([repair - 1], dtype=torch.long, device=item_ids.device)]
        )
        masses = torch.cat(
            [masses, torch.tensor([float(remainder)], device=item_ids.device)]
        )
    carriers = int(endpoints.numel())
    cast = cast_prefix(parent_cache, cast_maps, cast_length)
    repair_items = item_ids[:, -repair:].index_select(1, endpoints)
    repair_behaviors = behaviors[:, -repair:].index_select(1, endpoints)
    repair_deltas = time_deltas[:, -repair:].index_select(1, endpoints)
    embedded = current.embed_inputs(repair_items, repair_behaviors, repair_deltas)
    _, compact = current.forward_with_cache_embedded(cast, embedded)
    values = compact.v.clone()
    values[:, :, cast_length:] *= masses.to(values.dtype).view(1, 1, carriers, 1)
    compact = HSTUKVCache(k=compact.k, v=values, seq_len=cast_length + carriers)

    padding = nominal - compact.seq_len
    if padding:
        zero_k = compact.k.new_zeros(
            compact.k.shape[0], compact.k.shape[1], padding, compact.k.shape[-1]
        )
        zero_v = compact.v.new_zeros(
            compact.v.shape[0], compact.v.shape[1], padding, compact.v.shape[-1]
        )
        compact = HSTUKVCache(
            k=torch.cat([zero_k, compact.k], dim=2),
            v=torch.cat([zero_v, compact.v], dim=2),
            seq_len=nominal,
        )
    return compact, RefinementLayout(
        nominal_positions=nominal,
        cast_positions=cast_length,
        repair_evidence=repair,
        carriers=carriers,
        padding_positions=padding,
    )


@torch.inference_mode()
def build_evidence_measure_basis_cache(
    *,
    parent_cache: HSTUKVCache,
    current,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    cast_maps: tuple[torch.Tensor, ...],
    repair_width: int = REPAIR_WIDTH,
) -> tuple[HSTUKVCache, RefinementLayout]:
    """Build the frozen signed evidence-measure basis at Design-0 cost.

    The Current carrier key is the later event of each adjacent pair.  Its
    signed value is the parameter-CAST value of the earlier event plus the
    dependency-closed Current value of the anchor.  Equivalently, this is the
    pair's CAST value measure plus the anchor's Current-minus-CAST contextual
    residual; the cancelled anchor CAST is not executed.

    A value-only CAST costs half of a joint K/V CAST.  For a full 512-position
    state, 32 oldest joint CASTs are therefore reallocated to the 64 earlier
    pair values.  Current replay, carrier count, raw repair region, state
    layout and total parameter-map arithmetic remain matched to Design 0.
    """
    if item_ids.shape != behaviors.shape or item_ids.shape != time_deltas.shape:
        raise ValueError("raw prefix tensors differ in shape")
    if item_ids.ndim != 2 or item_ids.shape[0] != parent_cache.k.shape[1]:
        raise ValueError("raw prefix and Parent cache batch dimensions differ")
    if item_ids.shape[1] != parent_cache.seq_len:
        raise ValueError("raw prefix width differs from Parent cache")
    if repair_width < 1:
        raise ValueError("repair width must be positive")
    if current.cfg.relative_position_bias:
        raise ValueError("zero-padded compact refinement requires no relative-position bias")

    nominal = parent_cache.seq_len
    repair = min(repair_width, nominal)
    old_prefix = nominal - repair
    endpoints, masses = _pair_endpoints_and_mass(repair, item_ids.device)
    carriers = int(endpoints.numel())
    pair_starts = torch.arange(0, repair, GROUP_SIZE, dtype=torch.long, device=item_ids.device)
    if pair_starts.numel() != carriers:
        raise RuntimeError("pair starts and endpoints differ")

    # Two value-only maps equal one joint K/V map.  Reallocate only whole
    # joint-map equivalents, and use the most recent pair starts if a short
    # history cannot fund every shared value transform.
    pair_starts_with_mass = pair_starts[masses > 1]
    reallocated_joint = min(old_prefix, int(pair_starts_with_mass.numel()) // 2)
    value_cast_count = 2 * reallocated_joint
    untranslated_old = reallocated_joint
    joint_cast_start = untranslated_old
    if joint_cast_start:
        prefix_k = [parent_cache.k[:, :, :joint_cast_start]]
        prefix_v = [parent_cache.v[:, :, :joint_cast_start]]
    else:
        prefix_k, prefix_v = [], []
    if joint_cast_start < old_prefix:
        translated_old = _cast_slice(parent_cache, cast_maps, joint_cast_start, old_prefix)
        prefix_k.append(translated_old.k)
        prefix_v.append(translated_old.v)
    prefix = HSTUKVCache(
        k=torch.cat(prefix_k, dim=2) if prefix_k else parent_cache.k[:, :, :0],
        v=torch.cat(prefix_v, dim=2) if prefix_v else parent_cache.v[:, :, :0],
        seq_len=old_prefix,
    )

    repair_items = item_ids[:, -repair:].index_select(1, endpoints)
    repair_behaviors = behaviors[:, -repair:].index_select(1, endpoints)
    repair_deltas = time_deltas[:, -repair:].index_select(1, endpoints)
    embedded = current.embed_inputs(repair_items, repair_behaviors, repair_deltas)
    _, compact = current.forward_with_cache_embedded(prefix, embedded)

    source_positions = old_prefix + pair_starts
    shared_values = parent_cache.v.index_select(2, source_positions).clone()
    if value_cast_count:
        selected_pair_indices = torch.nonzero(masses > 1, as_tuple=False).flatten()[
            -value_cast_count:
        ]
        selected = source_positions.index_select(0, selected_pair_indices)
        shared_values.index_copy_(
            2,
            selected_pair_indices,
            _cast_values_at(parent_cache, cast_maps, selected),
        )
    carrier_values = compact.v[:, :, old_prefix:].clone()
    pair_mask = (masses > 1).to(carrier_values.dtype).view(1, 1, carriers, 1)
    carrier_values = carrier_values + pair_mask * shared_values
    values = compact.v.clone()
    values[:, :, old_prefix:] = carrier_values
    compact = HSTUKVCache(k=compact.k, v=values, seq_len=old_prefix + carriers)

    padding = nominal - compact.seq_len
    if padding:
        zero_k = compact.k.new_zeros(
            compact.k.shape[0], compact.k.shape[1], padding, compact.k.shape[-1]
        )
        zero_v = compact.v.new_zeros(
            compact.v.shape[0], compact.v.shape[1], padding, compact.v.shape[-1]
        )
        padded = HSTUKVCache(
            k=torch.cat([zero_k, compact.k], dim=2),
            v=torch.cat([zero_v, compact.v], dim=2),
            seq_len=nominal,
        )
    else:
        padded = compact
    return padded, RefinementLayout(
        nominal_positions=nominal,
        cast_positions=old_prefix,
        repair_evidence=repair,
        carriers=carriers,
        padding_positions=padding,
    )
