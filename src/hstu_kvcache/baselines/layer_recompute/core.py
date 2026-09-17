"""In-memory partial-layer replay for unpadded, equal-length HSTU batches.

Boundary inputs describe the actual cache-producing execution. They need not
equal a fresh encoding under either the current or the original model.
"""

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from hstu_kvcache.models import HSTU, HSTUKVCache
from hstu_kvcache.models.state_transition import retain_latest_cache

Interval = tuple[int, int] | None


@dataclass
class LayerRecomputeState:
    cache: HSTUKVCache
    # E_l before block l's norm, indexed by actual layer number 1..L-1.
    boundary_inputs: dict[int, torch.Tensor]


@contextmanager
def _evaluation(model: HSTU):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


@contextmanager
def _capture_boundaries(model: HSTU):
    captured = {}
    handles = []

    def capture(layer):
        def hook(_module, inputs):
            captured[layer] = inputs[0].detach().clone()
        return hook

    try:
        for layer in range(1, len(model.blocks)):
            # Incremental execution calls block.forward_with_cache directly;
            # the norm module still goes through __call__ in both paths.
            handles.append(model.blocks[layer].norm.register_forward_pre_hook(capture(layer)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def capture_state(model: HSTU, item_ids, behaviors, time_deltas) -> LayerRecomputeState:
    """Build ordinary K/V and E_1..E_{L-1} from an unpadded initial history."""
    with _evaluation(model), _capture_boundaries(model) as boundaries:
        cache = model.compute_kv(item_ids, behaviors, time_deltas)
    return LayerRecomputeState(cache, boundaries)


@torch.no_grad()
def recompute_interval(
    model: HSTU,
    state: LayerRecomputeState,
    item_ids,
    behaviors,
    time_deltas,
    interval: Interval,
) -> LayerRecomputeState:
    """Replay inclusive [a,b]; None retains the same state.

    Raw inputs must match the complete retained history, without padding, and
    keep their original time deltas. Only a=0 consumes them for embedding.
    All calls return state without mutating the input; unchanged E tensors may
    share storage. No teacher hidden or K/V is used to fill missing boundaries.
    """
    if interval is None:
        return state
    start, end = interval
    if not 0 <= start <= end < len(model.blocks):
        raise ValueError("interval must lie within the model layers")
    if (item_ids.shape != behaviors.shape or item_ids.shape != time_deltas.shape
            or item_ids.shape != state.cache.k.shape[1:3]):
        raise ValueError("raw events must match the unpadded retained cache")
    if start and start not in state.boundary_inputs:
        raise ValueError(f"missing actual boundary input E_{start}")

    k, v = state.cache.k.clone(), state.cache.v.clone()
    boundaries = dict(state.boundary_inputs)
    with _evaluation(model):
        x = (model.embed_inputs(item_ids, behaviors, time_deltas)
             if start == 0 else state.boundary_inputs[start])
        for layer in range(start, end + 1):
            if layer:
                boundaries[layer] = x.detach().clone()
            x, (layer_k, layer_v) = model.blocks[layer](x, return_kv=True)
            k[layer].copy_(layer_k)
            v[layer].copy_(layer_v)
    # In particular, E_{end+1} still belongs to that untouched layer's K/V.
    return LayerRecomputeState(HSTUKVCache(k, v, state.cache.seq_len), boundaries)


@torch.no_grad()
def append(model: HSTU, state: LayerRecomputeState, item_ids, behaviors, time_deltas):
    """Append an unpadded suffix and capture its actual per-layer inputs.

    The caller retains space before each event for a rolling window. Appending
    a whole suffix and cropping afterward is not the bounded online process.
    """
    with _evaluation(model), _capture_boundaries(model) as added:
        _, cache = model.forward_with_cache(state.cache, item_ids, behaviors, time_deltas)
    boundaries = {
        layer: torch.cat((state.boundary_inputs[layer], added[layer]), dim=1)
        for layer in range(1, len(model.blocks))
    }
    return LayerRecomputeState(cache, boundaries)


def retain_latest(state: LayerRecomputeState, length: int) -> LayerRecomputeState:
    """Evict the same leading events from K/V and every captured E tensor."""
    cache = retain_latest_cache(state.cache, length)
    start = state.cache.seq_len - length
    return LayerRecomputeState(cache, {
        layer: values[:, start:state.cache.seq_len]
        for layer, values in state.boundary_inputs.items()
    })


def enumerate_intervals(num_layers: int, *, include_reuse: bool = True) -> list[Interval]:
    intervals = [(a, b) for a in range(num_layers) for b in range(a, num_layers)]
    return ([None] if include_reuse else []) + intervals


@torch.no_grad()
def profile_intervals(
    model: HSTU,
    state: LayerRecomputeState,
    item_ids,
    behaviors,
    time_deltas,
    *,
    score: Callable[[LayerRecomputeState], float],
    intervals: Iterable[Interval] | None = None,
) -> list[dict]:
    """Score independent candidate replays; the caller owns data and selection.

    Scores may use the current model's ordinary query path. Layer counts are
    structural work counts, not measured latency or a complete cost estimate.
    """
    if intervals is None:
        intervals = enumerate_intervals(len(model.blocks))
    rows = []
    with _evaluation(model):
        for interval in intervals:
            candidate = recompute_interval(
                model, state, item_ids, behaviors, time_deltas, interval
            )
            layers = 0 if interval is None else interval[1] - interval[0] + 1
            rows.append(dict(interval=interval, score=float(score(candidate)),
                             recomputed_layers=layers))
    return rows
