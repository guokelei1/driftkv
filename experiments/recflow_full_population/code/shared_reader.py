"""Eager correctness prototype: candidate queries share one HSTU prefix.

No [batch*candidates, history, width] cache replication. Supports the native
HSTU-reference SiLU block and relative position bias. Optional non-reentrant
query checkpointing bounds saved query activations across chunks. This is not
yet a profiled training engine; DDP/input loading come later.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from hstu_kvcache.models.kv_cache import HSTUKVCache


def score_shared_cache(model, cache, candidates, query_deltas, lengths, *, item_vectors=None,
                       validate=True):
    if model.cfg.block_variant != "hstu_reference":
        raise ValueError("This new protocol uses the HSTU-reference block")
    if model.cfg.causal_diagonal != "inclusive" or model.cfg.attn_dropout != 0:
        raise ValueError("Prototype requires inclusive attention with zero attention dropout")
    batch, count = candidates.shape
    width = cache.k.shape[2]
    if cache.k.shape != cache.v.shape or cache.k.shape[:2] != (model.cfg.num_layers, batch):
        raise ValueError("Cache and model/batch dimensions differ")
    if lengths.shape != (batch,) or (validate and ((lengths < 0).any() or (lengths > width).any())):
        raise ValueError("Invalid true prefix lengths")
    if width >= model.cfg.max_seq_len:
        raise ValueError("max_seq_len must accommodate the history and one transient query")
    query = model.embed_query_tokens(candidates, query_deltas, item_vectors=item_vectors)
    positions = torch.arange(width, device=query.device)
    valid = positions[None, :] < lengths[:, None]
    for layer, block in enumerate(model.blocks):
        normalized = block.norm(query)
        attn = block.attn
        q, self_k, self_v = attn._project(normalized)
        k = cache.k[layer].view(batch, width, attn.num_heads, attn.head_dim).transpose(1, 2)
        v = cache.v[layer].view(batch, width, attn.num_heads, attn.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * attn.scale
        self_scores = (q * self_k).sum(dim=-1, keepdim=True) * attn.scale
        if attn.position_bias is not None:
            offset = attn.cfg.max_seq_len - 1
            indices = positions[None, :] - lengths[:, None] + offset
            bias = attn.position_bias(indices).permute(0, 2, 1).unsqueeze(2)
            scores = scores + bias.to(scores.dtype)
            self_scores = self_scores + attn.position_bias.weight[offset][None, :, None, None].to(scores.dtype)
        weights = attn._activate(scores) / attn.cfg.max_seq_len
        weights = weights * valid[:, None, None, :]
        read = weights @ v
        read = read + attn._activate(self_scores) / attn.cfg.max_seq_len * self_v
        read = read.transpose(1, 2).reshape(batch, count, attn.inner)
        output = attn.out_proj(block.attn_output_norm(read) * F.silu(block.gate_proj(normalized)))
        query = query + output
    return model.cc_score_head(model.final_norm(query)).squeeze(-1)


def score_full(model, items, behaviors, deltas, lengths, candidates, query_deltas, *, chunk_size=128,
               checkpoint_queries=False, prefix_item_vectors=None, candidate_item_vectors=None,
               validate=True):
    """One differentiable history; chunk only queries, not the ranking problem.

    validate=False requires a checked CPU batch. It avoids per-chunk scalar GPU
    synchronizations. External item vectors allow the planned factorized item
    and metadata encoder without allocating a second full-width catalog table.
    """
    if chunk_size < 1 or candidates.shape[1] < 1:
        raise ValueError("Nonempty candidates and a positive chunk size are required")
    if lengths.shape != (items.shape[0],) or (validate and ((lengths < 0).any() or (lengths > items.shape[1]).any())):
        raise ValueError("Invalid true prefix lengths")
    if prefix_item_vectors is None:
        _, cache = model(items, behaviors, deltas, return_kv=True, return_hidden=False, lengths=lengths)
    else:
        embedded = model.combine_input_features(prefix_item_vectors, behaviors, deltas)
        _, cache = model.forward_embedded(embedded, return_kv=True, return_hidden=False, lengths=lengths)

    def read_chunk(ids, times, keys, values, vectors):
        prefix = HSTUKVCache(keys, values, seq_len=cache.seq_len)
        return score_shared_cache(model, prefix, ids, times, lengths, item_vectors=vectors, validate=False)

    chunks = []
    for start in range(0, candidates.shape[1], chunk_size):
        end = start + chunk_size
        times = query_deltas if query_deltas.ndim == 1 else query_deltas[:, start:end]
        vectors = None if candidate_item_vectors is None else candidate_item_vectors[:, start:end]
        args = (candidates[:, start:end], times, cache.k, cache.v, vectors)
        if checkpoint_queries and torch.is_grad_enabled():
            chunks.append(checkpoint(read_chunk, *args, use_reentrant=False))
        else:
            chunks.append(read_chunk(*args))
    return torch.cat(chunks, dim=1)
