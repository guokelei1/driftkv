from __future__ import annotations

from dataclasses import dataclass

import torch

from .hstu import HSTU
from .kv_cache import HSTUKVCache


@dataclass(frozen=True)
class TransitionWork:
    projection_tokens: int
    recomputed_token_layers: int
    attention_pair_work: int
    old_kv_read_bytes: int
    new_kv_write_bytes: int
    raw_history_read_bytes: int


def frozen_segment(name: str, length: int) -> slice:
    if length < 1:
        raise ValueError("persistent prefix must contain at least one token")
    if name == "full":
        return slice(0, length)
    if name == "middle":
        return slice(length // 4, max(length // 4 + 1, (3 * length + 1) // 4))
    if name.startswith("recent_"):
        width = int(name.removeprefix("recent_"))
        if width < 1:
            raise ValueError("recent width must be positive")
        return slice(max(0, length - width), length)
    raise ValueError(f"unknown executable segment: {name}")


def truncate_cache(cache: HSTUKVCache, length: int) -> HSTUKVCache:
    if not 0 <= length <= cache.seq_len or length > cache.k.shape[2]:
        raise ValueError("truncation length outside cache")
    return HSTUKVCache(
        k=cache.k[:, :, :length, :],
        v=cache.v[:, :, :length, :],
        seq_len=length,
    )


def retain_latest_cache(cache: HSTUKVCache, length: int) -> HSTUKVCache:
    """Retain the newest ``length`` positions from a rolling persistent cache."""
    if not 0 <= length <= cache.seq_len or length > cache.k.shape[2]:
        raise ValueError("retained length outside cache")
    start = cache.seq_len - length
    return HSTUKVCache(
        k=cache.k[:, :, start : cache.seq_len, :],
        v=cache.v[:, :, start : cache.seq_len, :],
        seq_len=length,
    )


@torch.no_grad()
def append_with_rolling_cap(
    current: HSTU,
    cache: HSTUKVCache,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    max_length: int,
) -> HSTUKVCache:
    """Append events in chronological order while enforcing a true cache cap.

    Eviction happens before each append, so each new token attends to at most
    ``max_length - 1`` cached positions.  Processing the whole suffix and
    cropping afterward would give later tokens access to already-evicted K/V
    and is therefore not equivalent to a bounded online persistent state.
    """
    if max_length < 1:
        raise ValueError("rolling cap must be positive")
    if item_ids.shape != behaviors.shape or item_ids.shape != time_deltas.shape:
        raise ValueError("appended event tensors differ in shape")
    if item_ids.ndim != 2 or item_ids.shape[0] != cache.k.shape[1]:
        raise ValueError("appended events and cache batch dimensions differ")
    state = cache
    for position in range(item_ids.shape[1]):
        if state.seq_len >= max_length:
            state = retain_latest_cache(state, max_length - 1)
        _, state = current.forward_with_cache(
            state,
            item_ids[:, position : position + 1],
            behaviors[:, position : position + 1],
            time_deltas[:, position : position + 1],
        )
    return state


@torch.no_grad()
def project_exact_layer0_segment(
    current: HSTU,
    parent_cache: HSTUKVCache,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    segment: str,
) -> HSTUKVCache:
    """Refresh selected layer-0 K/V exactly under the current model.

    Inputs must be the original parent-prefix tokens with their original time
    deltas. In particular, callers may not reset the first selected token's
    temporal delta when the segment begins after position zero.
    """
    if item_ids.shape != behaviors.shape or item_ids.shape != time_deltas.shape:
        raise ValueError("raw prefix tensors differ in shape")
    if item_ids.ndim != 2 or item_ids.shape[1] != parent_cache.seq_len:
        raise ValueError("raw prefix width must equal cache seq_len")
    if parent_cache.k.shape[1] != item_ids.shape[0]:
        raise ValueError("raw prefix and cache batch dimensions differ")
    selected = frozen_segment(segment, parent_cache.seq_len)
    was_training = current.training
    current.eval()
    try:
        embedded = current.embed_inputs(
            item_ids[:, selected], behaviors[:, selected], time_deltas[:, selected]
        )
        normalized = current.blocks[0].norm(embedded)
        k_new, v_new = current.blocks[0].attn.project_kv(normalized)
    finally:
        if was_training:
            current.train()
    k, v = parent_cache.k.clone(), parent_cache.v.clone()
    k[0, :, selected, :].copy_(k_new)
    v[0, :, selected, :].copy_(v_new)
    return HSTUKVCache(k=k, v=v, seq_len=parent_cache.seq_len)


@torch.no_grad()
def hybrid_tail_refresh(
    current: HSTU,
    parent_cache: HSTUKVCache,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    width: int,
) -> HSTUKVCache:
    """Replay a raw tail with current weights, conditioned on parent prefix K/V."""
    if width < 1:
        raise ValueError("tail width must be positive")
    if item_ids.shape != behaviors.shape or item_ids.shape != time_deltas.shape:
        raise ValueError("raw prefix tensors differ in shape")
    if item_ids.ndim != 2 or item_ids.shape[1] != parent_cache.seq_len:
        raise ValueError("raw prefix width must equal cache seq_len")
    start = max(0, parent_cache.seq_len - width)
    prefix = truncate_cache(parent_cache, start)
    _, refreshed = current.forward_with_cache(
        prefix, item_ids[:, start:], behaviors[:, start:], time_deltas[:, start:]
    )
    return refreshed


def transition_work(
    action: str,
    cache: HSTUKVCache,
    raw_item_ids: torch.Tensor,
    raw_behaviors: torch.Tensor,
    raw_time_deltas: torch.Tensor,
) -> TransitionWork:
    layers, batch, length, width = cache.k.shape
    element_bytes = cache.k.element_size()
    kv_token_bytes = 2 * width * element_bytes
    raw_token_bytes = (
        raw_item_ids.element_size() + raw_behaviors.element_size() + raw_time_deltas.element_size()
    )
    if action == "noop":
        tokens = token_layers = pairs = reads = writes = raw = 0
    elif action.startswith("layer0_"):
        segment = action.removeprefix("layer0_")
        segment = "recent_128" if segment == "recent128" else segment
        selected = frozen_segment(segment, length)
        tokens = selected.stop - selected.start
        token_layers, pairs = tokens, 0
        reads = writes = batch * tokens * kv_token_bytes
        raw = batch * tokens * raw_token_bytes
    elif action.startswith("hybrid_tail"):
        width_tokens = int(action.removeprefix("hybrid_tail"))
        tokens = min(length, width_tokens)
        prefix = length - tokens
        token_layers = tokens * layers
        pairs = layers * (tokens * prefix + tokens * (tokens + 1) // 2)
        reads = batch * prefix * layers * kv_token_bytes
        writes = batch * tokens * layers * kv_token_bytes
        raw = batch * tokens * raw_token_bytes
    elif action == "exact_all":
        tokens = length
        token_layers = length * layers
        pairs = layers * length * (length + 1) // 2
        reads = 0
        writes = batch * length * layers * kv_token_bytes
        raw = batch * length * raw_token_bytes
    else:
        raise ValueError(f"unknown transition action: {action}")
    return TransitionWork(
        projection_tokens=tokens,
        recomputed_token_layers=token_layers,
        attention_pair_work=pairs,
        old_kv_read_bytes=reads,
        new_kv_write_bytes=writes,
        raw_history_read_bytes=raw,
    )
