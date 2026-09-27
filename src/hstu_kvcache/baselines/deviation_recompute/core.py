"""CacheBlend-inspired early-layer deviation selection for HSTU.

The early layer is fixed prospectively to layer 0. Its current K/V projections
depend only on the raw token, so all-history scoring needs no hidden teacher
or quadratic attention. Selected positions then replay all current layers,
conditioned on the untouched inherited K/V at unselected positions.
"""

import torch
import torch.nn.functional as F

from hstu_kvcache.baselines.layer_recompute.core import _evaluation
from hstu_kvcache.baselines.sparse_recompute import recompute_positions, select_top_positions


@torch.no_grad()
def layer0_deviation_scores(current, cache, item_ids, behaviors, time_deltas):
    """Mean squared K plus V projection deviation, one value per history row.

    Only K/V are projected: no Q, attention output, or exact upper-layer state
    is computed. Arithmetic is promoted for the reduction, not for projections.

    Treat row RMS differences below 32 projection-dtype machine epsilons as
    zero, relative to the current row's K/V RMS magnitude. Already-current
    layer-0 rows are mathematically identical; scalar versus banded GEMM can
    otherwise rank their roundoff instead of model drift. Exact zero ties use
    the shared chronological tie break. This threshold never uses labels.
    """
    if item_ids.shape != cache.k.shape[1:3]:
        raise ValueError("raw history must align with cache")
    with _evaluation(current):
        block = current.blocks[0]
        x = block.norm(current.embed_inputs(item_ids, behaviors, time_deltas))
        k, v = block.attn.k_proj(x), block.attn.v_proj(x)
        if block.block_variant == "hstu_reference":
            k, v = F.silu(k), F.silu(v)
        score = ((k.float() - cache.k[0].float()).square().mean(dim=-1)
                 + (v.float() - cache.v[0].float()).square().mean(dim=-1))
        magnitude = k.float().square().mean(dim=-1) + v.float().square().mean(dim=-1)
        floor = (32 * torch.finfo(k.dtype).eps) ** 2 * magnitude
        return score.masked_fill(score <= floor, 0)


@torch.no_grad()
def recompute_deviation(current, cache, item_ids, behaviors, time_deltas, n, *, query_chunk_size=64):
    """Select and replay n history rows; bypass selection at both endpoints."""
    if n < 0:
        raise ValueError("selected count must be nonnegative")
    count = min(n, cache.seq_len)
    if count == 0:
        return cache
    if count == cache.seq_len:
        positions = torch.arange(count, device=item_ids.device)[None].expand(item_ids.shape[0], -1)
    else:
        scores = layer0_deviation_scores(current, cache, item_ids, behaviors, time_deltas)
        positions = select_top_positions(scores, count)
    return recompute_positions(current, cache, item_ids, behaviors, time_deltas, positions,
                               query_chunk_size=query_chunk_size)
