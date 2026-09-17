"""Tail recomputation conditioned on the retained, possibly stale prefix."""

import torch

from hstu_kvcache.models import HSTU, HSTUKVCache
from hstu_kvcache.models.state_transition import hybrid_tail_refresh


@torch.no_grad()
def recompute_tail(
    current: HSTU,
    cache: HSTUKVCache,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    n: int,
) -> HSTUKVCache:
    """Replay the last min(n, N) events through every current-model layer.

    Inputs are the original, unpadded [batch, N] events aligned with cache.
    Keep their original time deltas, including the first replayed event's delta.
    n=0 returns the input cache; n>=N rebuilds the complete retained history.
    Other n values inherit the unchanged prefix's errors. This function never
    mutates the input, and uses the model's deterministic native append path.
    """
    if n < 0:
        raise ValueError("tail length must be nonnegative")
    if n == 0 or cache.seq_len == 0:
        return cache
    return hybrid_tail_refresh(current, cache, item_ids, behaviors, time_deltas, width=n)
