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
