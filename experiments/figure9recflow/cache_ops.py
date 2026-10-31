"""RecFlow cache repairs for the external-feature HSTU-reference backbone.

Histories are unpadded, equal-length batches. A batch may also be a method
dimension: event features of batch size one are broadcast over resident caches.
All repairs are functional. Queries are transient; append evicts BEFORE each
behavior, preserving the relative-position semantics of a true bounded cache.
These operations require eval mode and never change a model's mode or weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from experiments.recflow_full_population.code.shared_reader import score_shared_cache
from hstu_kvcache.models.kv_cache import HSTUKVCache


@dataclass(frozen=True)
class CapturedCache:
    cache: HSTUKVCache
    # Actual input to each block, before its LayerNorm; not teacher boundaries.
    layer_inputs: tuple[torch.Tensor, ...]


def _backbone(model):
    backbone = model.backbone
    cfg = backbone.cfg
    if (cfg.block_variant != "hstu_reference" or cfg.activation != "silu"
            or cfg.causal_diagonal != "inclusive" or cfg.attn_dropout != 0):
        raise ValueError("RecFlow operations require inclusive SiLU HSTU-reference with zero attention dropout")
    if model.training or backbone.training:
        raise ValueError("Cache operations require model.eval()")
    return backbone


def _check_cache(backbone, cache):
    if cache.k.ndim != 4 or cache.k.shape != cache.v.shape:
        raise ValueError("Cache/model dimensions differ")
    expected = (backbone.cfg.num_layers, cache.k.shape[1], cache.seq_len,
                backbone.blocks[0].attn.inner)
    if tuple(cache.k.shape) != expected:
        raise ValueError("Cache/model dimensions differ")
    if not 0 <= cache.seq_len < backbone.cfg.max_seq_len:
        raise ValueError("History must leave room for a transient query")


def _events(features, behaviors, deltas, *, batch=None, length=None):
    if (features.ndim != 3 or features.shape[-1] != 5 or behaviors.shape != features.shape[:2]
            or deltas.shape != behaviors.shape):
        raise ValueError("Expected features [B,N,5], behaviors/deltas [B,N]")
    if length is not None and features.shape[1] != length:
        raise ValueError("Raw history and cache lengths differ")
    if batch is not None and features.shape[0] not in (1, batch):
        raise ValueError("Input batch must equal cache batch or be one")


def _expand(tensor, batch):
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError("Cannot broadcast input batch")
    return tensor.expand(batch, *tensor.shape[1:])


def _embed(model, features, behaviors, deltas, batch):
    # Encode shared raw inputs once, before expanding over methods.
    hidden = model.backbone.combine_input_features(model.features(features), behaviors, deltas)
    return _expand(hidden, batch)


def _kv(block, hidden):
    normalized = block.norm(hidden)
    # project_kv() currently calls _project(), which unnecessarily computes Q.
    return F.silu(block.attn.k_proj(normalized)), F.silu(block.attn.v_proj(normalized))


@torch.no_grad()
def encode_cache(model, features, behaviors, deltas, *, capture=False):
    """Native full cache; optionally retain actual inputs to every layer."""
    backbone = _backbone(model)
    _events(features, behaviors, deltas)
    n = features.shape[1]
    if n >= backbone.cfg.max_seq_len:
        raise ValueError("History must leave room for a transient query")
    hidden = _embed(model, features, behaviors, deltas, features.shape[0])
    pairs, boundaries = [], []
    for index, block in enumerate(backbone.blocks):
        if capture:
            boundaries.append(hidden.detach().clone())
        # The final output is dead: only its K/V is used by candidate reads.
        if n == 0 or index == len(backbone.blocks) - 1:
            pair = _kv(block, hidden)
        else:
            hidden, pair = block(hidden, return_kv=True)
        pairs.append(pair)
    cache = HSTUKVCache.from_layer_list(pairs, n)
    return CapturedCache(cache, tuple(boundaries)) if capture else cache


@torch.no_grad()
def recompute_interval(model, cache, features, behaviors, deltas, interval,
                       *, layer_inputs: Sequence[torch.Tensor] | Mapping[int, torch.Tensor] | None = None):
    """Replay one 0-based inclusive layer interval; leave other KV untouched.

    A non-prefix interval MUST start from that layer's saved old-producer input.
    It must not substitute a current-teacher hidden state. ``layer_inputs`` can
    be CapturedCache.layer_inputs, or a mapping retaining only needed starts.
    """
    backbone = _backbone(model)
    _check_cache(backbone, cache)
    batch, n = cache.k.shape[1:3]
    _events(features, behaviors, deltas, batch=batch, length=n)
    start, end = interval
    if not 0 <= start <= end < len(backbone.blocks):
        raise ValueError("Invalid inclusive layer interval")
    if start == 0:
        hidden = _embed(model, features, behaviors, deltas, batch)
    else:
        if layer_inputs is None:
            raise ValueError("Non-prefix Layer requires actual old input boundary")
        try:
            hidden = _expand(layer_inputs[start], batch)
        except (KeyError, IndexError) as error:
            raise ValueError("Missing actual old input boundary") from error
        if hidden.shape != (batch, n, backbone.cfg.hidden_size):
            raise ValueError("Old input boundary dimensions differ")
    keys, values = cache.k.clone(), cache.v.clone()
    for index in range(start, end + 1):
        block = backbone.blocks[index]
        if n == 0 or index == end:
            pair = _kv(block, hidden)
        else:
            hidden, pair = block(hidden, return_kv=True)
        keys[index], values[index] = pair
    return HSTUKVCache(keys, values, n)


def _positions(positions, batch, n, device):
    positions = torch.as_tensor(positions, device=device, dtype=torch.long)
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    if positions.ndim != 2:
        raise ValueError("Positions must be [M] or [B,M]")
    positions = _expand(positions, batch)
    if positions.shape[1] and (bool(((positions < 0) | (positions >= n)).any())
                               or bool((positions[:, 1:] <= positions[:, :-1]).any())):
        raise ValueError("Positions must be unique, chronological, and inside history")
    return positions


def _gather(tensor, positions):
    tensor = _expand(tensor, positions.shape[0])
    if tensor.ndim == 2:
        return tensor.gather(1, positions)
    return tensor.gather(1, positions[..., None].expand(-1, -1, tensor.shape[-1]))


def _sparse_read(attn, q, keys, values, positions, query_chunk_size):
    """Bounded dense oracle-compatible read with real per-batch positions.

    A legacy fused ELU kernel cannot implement RecFlow's SiLU/divisor/relative
    bias. Chunking only query rows preserves the complete causal key set.
    """
    batch, n, _ = keys.shape
    kh = keys.reshape(batch, n, attn.num_heads, attn.head_dim).transpose(1, 2)
    vh = values.reshape(batch, n, attn.num_heads, attn.head_dim).transpose(1, 2)
    key_pos = torch.arange(n, device=keys.device)
    outputs = []
    for start in range(0, positions.shape[1], query_chunk_size):
        row_pos = positions[:, start:start + query_chunk_size]
        scores = q[:, :, start:start + query_chunk_size] @ kh.transpose(-2, -1)
        scores = scores * attn.scale
        if attn.position_bias is not None:
            indices = key_pos[None, None, :] - row_pos[:, :, None] + attn.cfg.max_seq_len - 1
            bias = attn.position_bias(indices).permute(0, 3, 1, 2)
            scores = scores + bias.to(scores.dtype)
        keep = key_pos[None, None, :] <= row_pos[:, :, None]
        weights = F.silu(scores) / attn.cfg.max_seq_len
        outputs.append((weights * keep[:, None]) @ vh)
    return torch.cat(outputs, dim=2).transpose(1, 2).reshape(batch, positions.shape[1], attn.inner)


@torch.no_grad()
def recompute_positions(model, cache, features, behaviors, deltas, positions, *, query_chunk_size=64):
    """Recompute selected raw rows at every layer, reading mixed inherited KV."""
    backbone = _backbone(model)
    _check_cache(backbone, cache)
    batch, n = cache.k.shape[1:3]
    _events(features, behaviors, deltas, batch=batch, length=n)
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    positions = _positions(positions, batch, n, cache.k.device)
    m = positions.shape[1]
    if m == 0:
        return cache
    if m == n:
        return encode_cache(model, _expand(features, batch), _expand(behaviors, batch), _expand(deltas, batch))
    hidden = _embed(model, _gather(features, positions), _gather(behaviors, positions),
                    _gather(deltas, positions), batch)
    keys, values = [], []
    for layer, block in enumerate(backbone.blocks):
        normalized = block.norm(hidden)
        attn = block.attn
        if layer == len(backbone.blocks) - 1:
            sk, sv = F.silu(attn.k_proj(normalized)), F.silu(attn.v_proj(normalized))
        else:
            q, sk, sv = attn._project(normalized)
            sk = sk.transpose(1, 2).reshape(batch, m, attn.inner)
            sv = sv.transpose(1, 2).reshape(batch, m, attn.inner)
        scatter = positions[..., None].expand(-1, -1, attn.inner)
        k, v = cache.k[layer].scatter(1, scatter, sk), cache.v[layer].scatter(1, scatter, sv)
        keys.append(k)
        values.append(v)
        if layer < len(backbone.blocks) - 1:
            read = _sparse_read(attn, q, k, v, positions, query_chunk_size)
            hidden = hidden + attn.out_proj(block.attn_output_norm(read) * F.silu(block.gate_proj(normalized)))
    return HSTUKVCache(torch.stack(keys), torch.stack(values), n)


def _suffix(backbone, cache, hidden):
    """Native suffix replay, retaining its original relative positions."""
    pairs = []
    for layer, block in enumerate(backbone.blocks):
        if layer == len(backbone.blocks) - 1:
            k, v = _kv(block, hidden)
            pair = (torch.cat((cache.k[layer], k), dim=1), torch.cat((cache.v[layer], v), dim=1))
        else:
            hidden, pair = block.forward_with_cache(hidden, cache.k[layer], cache.v[layer])
        pairs.append(pair)
    return HSTUKVCache.from_layer_list(pairs, cache.seq_len + hidden.shape[1])


@torch.no_grad()
def recompute_tail(model, cache, features, behaviors, deltas, count):
    """Replay min(count,N) newest rows; preserve their recorded time deltas."""
    backbone = _backbone(model)
    _check_cache(backbone, cache)
    batch, n = cache.k.shape[1:3]
    _events(features, behaviors, deltas, batch=batch, length=n)
    if count < 0:
        raise ValueError("Tail count must be nonnegative")
    count = min(int(count), n)
    if count == 0:
        return cache
    start = n - count
    prefix = HSTUKVCache(cache.k[:, :, :start], cache.v[:, :, :start], start)
    hidden = _embed(model, features[:, start:], behaviors[:, start:], deltas[:, start:], batch)
    return _suffix(backbone, prefix, hidden)


@torch.no_grad()
def deviation_scores(model, cache, features, behaviors, deltas):
    """First-layer K/V squared deviation, averaged over KV width; no Q."""
    backbone = _backbone(model)
    _check_cache(backbone, cache)
    batch, n = cache.k.shape[1:3]
    _events(features, behaviors, deltas, batch=batch, length=n)
    hidden = _embed(model, features, behaviors, deltas, batch)
    k, v = _kv(backbone.blocks[0], hidden)
    return (k.float() - cache.k[0].float()).square().mean(-1) + (v.float() - cache.v[0].float()).square().mean(-1)


@torch.no_grad()
def select_positions(scores, count):
    """Stable score-descending selection, then restore chronological order."""
    if scores.ndim != 2 or count < 0:
        raise ValueError("Expected scores [B,N] and a nonnegative count")
    count = min(int(count), scores.shape[1])
    return torch.argsort(scores, dim=1, descending=True, stable=True)[:, :count].sort(dim=1).values


@torch.no_grad()
def append_behaviors(model, cache, features, behaviors, deltas, *, window_size=1024):
    """Persist current-model writes, with exact single-behavior FIFO semantics.

    Relative bias forbids reusing the legacy rolling-band shortcut. Features
    can be encoded in a batch, but every new row sees at most window_size-1
    resident rows, after that row's own eviction. No caller input is modified.
    """
    backbone = _backbone(model)
    _check_cache(backbone, cache)
    batch = cache.k.shape[1]
    _events(features, behaviors, deltas, batch=batch)
    if not 1 <= window_size < backbone.cfg.max_seq_len or cache.seq_len > window_size:
        raise ValueError("Invalid rolling history capacity")
    hidden = _embed(model, features, behaviors, deltas, batch)
    state = cache
    for index in range(hidden.shape[1]):
        if state.seq_len == window_size:
            state = HSTUKVCache(state.k[:, :, 1:], state.v[:, :, 1:], window_size - 1)
        state = _suffix(backbone, state, hidden[:, index:index + 1])
    return state


@torch.no_grad()
def score_candidates(model, cache, candidate_features, query_deltas, *, lengths=None, chunk_size=None,
                     candidate_vectors=None):
    """Transient shared-reader scores; candidate rows never enter the cache."""
    backbone = _backbone(model)
    _check_cache(backbone, cache)
    batch = cache.k.shape[1]
    if candidate_features.ndim != 3 or candidate_features.shape[-1] != 5:
        raise ValueError("Expected candidate features [B,Q,5]")
    features = candidate_features
    ids = _expand(features[..., 0].long().clamp_min(0), batch)
    vectors = _expand(model.features(features) if candidate_vectors is None else candidate_vectors, batch)
    if vectors.shape != (batch, ids.shape[1], backbone.cfg.hidden_size):
        raise ValueError("Candidate vector dimensions differ")
    if query_deltas.ndim not in (1, 2):
        raise ValueError("Query deltas must be [B] or [B,Q]")
    query_deltas = _expand(query_deltas, batch)
    if query_deltas.ndim == 2 and query_deltas.shape[1] != ids.shape[1]:
        raise ValueError("Query deltas and candidate counts differ")
    if lengths is None:
        lengths = torch.full((batch,), cache.seq_len, device=cache.k.device, dtype=torch.long)
    else:
        lengths = _expand(lengths, batch)
        if lengths.shape != (batch,) or bool(((lengths < 0) | (lengths > cache.seq_len)).any()):
            raise ValueError("Invalid candidate prefix lengths")
    size = int(chunk_size if chunk_size is not None else getattr(model, "chunk_size", 128))
    if size < 1:
        raise ValueError("Candidate chunk size must be positive")
    chunks = []
    for start in range(0, ids.shape[1], size):
        times = query_deltas if query_deltas.ndim == 1 else query_deltas[:, start:start + size]
        chunks.append(score_shared_cache(backbone, cache, ids[:, start:start + size], times, lengths,
                                         item_vectors=vectors[:, start:start + size], validate=False))
    return torch.cat(chunks, dim=1) if chunks else vectors.new_empty((batch, 0))
