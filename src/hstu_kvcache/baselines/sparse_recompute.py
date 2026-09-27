"""Replay selected history rows through current HSTU blocks.

Each selected row receives the output of its own current-model lower block.
At attention, selected K/V are replaced and unselected K/V remain inherited.
No fully recomputed teacher state is used. Repairs are functional: the caller
can discard the returned cache after a request and continue the same Reuse
trajectory. Inputs are unpadded equal-length cohorts, as in the evaluator.
"""

import torch
import torch.nn.functional as F

from hstu_kvcache.baselines.layer_recompute.core import _evaluation
from hstu_kvcache.models import HSTU, HSTUKVCache


def select_top_positions(scores: torch.Tensor, count: int) -> torch.Tensor:
    """Stable score ranking, with history order as the deterministic tie break."""
    if scores.ndim != 2 or not 0 <= count <= scores.shape[1]:
        raise ValueError("scores must be [batch, history] and count within history")
    ranked = torch.argsort(scores, dim=1, descending=True, stable=True)
    return ranked[:, :count].sort(dim=1).values


def _selected_attention(attn, q, k, v, positions, *, query_chunk_size):
    """Dense QK/AV for bounded chunks of arbitrary causal query positions.

The dense products include future positions that are subsequently masked.
The cost ledger must count M*N pairs, not only the retained causal pairs.
"""
    batch, heads, count, _ = q.shape
    length = k.shape[2]
    key_positions = torch.arange(length, device=q.device)
    diagonal = -1 if attn.causal_diagonal == "exclusive" else 0
    outputs = []
    for start in range(0, count, query_chunk_size):
        end = min(start + query_chunk_size, count)
        query_positions = positions[:, start:end]
        weights = torch.matmul(q[:, :, start:end], k.transpose(-2, -1)) * attn.scale
        if attn.position_bias is not None:
            indices = (key_positions[None, None, :] - query_positions[:, :, None]
                       + attn.cfg.max_seq_len - 1)
            bias = attn.position_bias(indices).permute(0, 3, 1, 2)
            weights = weights + bias.to(weights.dtype)
        weights = attn._activate(weights)
        if attn.block_variant == "hstu_reference":
            weights = weights / attn.cfg.max_seq_len
        keep = key_positions[None, None, :] <= query_positions[:, :, None] + diagonal
        weights = attn.attn_dropout(weights * keep[:, None].to(weights.dtype))
        outputs.append(torch.matmul(weights, v))
    return torch.cat(outputs, dim=2).reshape(batch, heads, count, attn.head_dim)


def _block_output(block, residual, normalized, attention):
    """Use the same gate/residual equations as HSTUBlock.forward."""
    if block.block_variant == "hstu_reference":
        out = block.attn.out_proj(
            block.attn_output_norm(attention) * F.silu(block.gate_proj(normalized))
        )
    elif block.gating == "silu_gate":
        out = attention * F.silu(block.gate_proj(normalized))
    elif block.gating == "glu":
        out = attention * torch.sigmoid(block.gate_proj(normalized))
    elif block.gating == "ffn":
        out = block.fc2(F.silu(block.fc1(normalized)) * block.fc3(normalized))
    else:
        out = attention
    return residual + out


@torch.no_grad()
def recompute_positions(
    current: HSTU,
    cache: HSTUKVCache,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    positions: torch.Tensor,
    *,
    query_chunk_size: int = 64,
) -> HSTUKVCache:
    """Replay [B,M] sorted unique selected positions through every layer.

    Empty selection returns ``cache``; selecting the whole history uses the
    native full rebuild. In between, unselected rows remain bitwise unchanged.
    Original timestamps/deltas and absolute positions within retained history
    are preserved. The largest attention buffer has B*heads*chunk*N elements.
    """
    if (item_ids.ndim != 2 or item_ids.shape != behaviors.shape
            or item_ids.shape != time_deltas.shape
            or item_ids.shape != cache.k.shape[1:3]):
        raise ValueError("raw events must match the unpadded retained cache")
    if positions.ndim != 2 or positions.shape[0] != item_ids.shape[0]:
        raise ValueError("positions must have shape [batch, selected rows]")
    if positions.dtype != torch.long or query_chunk_size < 1:
        raise ValueError("positions must be int64 and query_chunk_size positive")
    count = positions.shape[1]
    if count == 0:
        return cache
    if (bool((positions < 0).any()) or bool((positions >= cache.seq_len).any())
            or bool((positions[:, 1:] <= positions[:, :-1]).any())):
        raise ValueError("positions must be sorted, unique and within the cache")
    with _evaluation(current):
        if count == cache.seq_len:
            return current.compute_kv(item_ids, behaviors, time_deltas)
        selected = tuple(values.gather(1, positions)
                         for values in (item_ids, behaviors, time_deltas))
        x = current.embed_inputs(*selected)
        keys, values = [], []
        batch = item_ids.shape[0]
        for layer, block in enumerate(current.blocks):
            residual, normalized = x, block.norm(x)
            attn = block.attn
            q, selected_k, selected_v = attn._project(normalized)
            selected_k = selected_k.transpose(1, 2).reshape(batch, count, attn.inner)
            selected_v = selected_v.transpose(1, 2).reshape(batch, count, attn.inner)
            indices = positions[:, :, None].expand(-1, -1, attn.inner)
            layer_k = cache.k[layer].scatter(1, indices, selected_k)
            layer_v = cache.v[layer].scatter(1, indices, selected_v)
            k = layer_k.reshape(batch, cache.seq_len, attn.num_heads, attn.head_dim)
            v = layer_v.reshape(batch, cache.seq_len, attn.num_heads, attn.head_dim)
            out = _selected_attention(
                attn, q, k.transpose(1, 2), v.transpose(1, 2), positions,
                query_chunk_size=query_chunk_size,
            )
            x = _block_output(block, residual, normalized, attn._finish(out))
            keys.append(layer_k)
            values.append(layer_v)
    return HSTUKVCache(torch.stack(keys), torch.stack(values), cache.seq_len)
