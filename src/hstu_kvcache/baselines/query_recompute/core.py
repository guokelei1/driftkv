"""ProphetKV-inspired current-query attention selection for HSTU.

Use current layer-0 Q against inherited layer-0 K; no teacher K or label enters
the selector. HSTU attention is not softmax, so scores use the magnitude of its
actual activated weights, averaged across heads and candidates. In legacy
ELU+1 attention these weights are already nonnegative.
"""

import torch
import torch.nn.functional as F

from hstu_kvcache.baselines.layer_recompute.core import _evaluation
from hstu_kvcache.baselines.sparse_recompute import recompute_positions, select_top_positions


@torch.no_grad()
def query_attention_scores(current, cache, candidate_ids, query_time_deltas):
    """Return [B,N] attention importance using first-layer request queries.

    Candidates are transient independent requests at position N, never history
    events. Selection performs Q projection and QK only, with no query AV or
    deeper blocks. The caller counts this extra work before repaired scoring.
    """
    with _evaluation(current):
        block = current.blocks[0]
        attn = block.attn
        x = block.norm(current.embed_query_tokens(candidate_ids, query_time_deltas))
        batch, candidates, _ = x.shape
        q = attn.q_proj(x).reshape(batch, candidates, attn.num_heads, attn.head_dim)
        if block.block_variant == "hstu_reference":
            q = F.silu(q)
        q = q.transpose(1, 2)
        k = cache.k[0].reshape(batch, cache.seq_len, attn.num_heads, attn.head_dim)
        weights = torch.matmul(q, k.permute(0, 2, 3, 1)) * attn.scale
        positions = torch.full((candidates,), cache.seq_len, device=q.device)
        keys = torch.arange(cache.seq_len, device=q.device)
        bias = attn._relative_position_bias(positions, keys, weights.dtype)
        if bias is not None:
            weights = weights + bias
        weights = attn._activate(weights)
        if block.block_variant == "hstu_reference":
            weights = weights / attn.cfg.max_seq_len
        return weights.float().abs().mean(dim=(1, 2))


@torch.no_grad()
def recompute_query(current, cache, item_ids, behaviors, time_deltas,
                    candidate_ids, query_time_deltas, n, *, query_chunk_size=64):
    """Select and replay n history rows; bypass selection at both endpoints."""
    if n < 0:
        raise ValueError("selected count must be nonnegative")
    count = min(n, cache.seq_len)
    if count == 0:
        return cache
    if count == cache.seq_len:
        positions = torch.arange(count, device=item_ids.device)[None].expand(item_ids.shape[0], -1)
    else:
        positions = select_top_positions(
            query_attention_scores(current, cache, candidate_ids, query_time_deltas), count
        )
    return recompute_positions(current, cache, item_ids, behaviors, time_deltas, positions,
                               query_chunk_size=query_chunk_size)
