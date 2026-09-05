"""Parameter maps retained for the sealed Large resource-canary baseline.

These are not the translator used by the prospective EvoKV design.
"""
from __future__ import annotations

import torch

from hstu_kvcache.models import HSTUKVCache


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


def cast_prefix(
    cache: HSTUKVCache,
    maps: tuple[torch.Tensor, ...],
    length: int,
) -> HSTUKVCache:
    """Materialize a mapped prefix as a reference for lazy-reader tests."""
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
