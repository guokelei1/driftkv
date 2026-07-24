from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models import HSTU, HSTUKVCache


@dataclass
class LayerwiseCacheState:
    kv: HSTUKVCache
    hidden_states: tuple[torch.Tensor, ...]
    normed_states: tuple[torch.Tensor, ...]
    lengths: torch.Tensor


@torch.no_grad()
def capture_layerwise_state(
    model: HSTU,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    lengths: torch.Tensor,
) -> LayerwiseCacheState:
    was_training = model.training
    model.eval()
    try:
        lengths = lengths.to(item_ids.device)
        valid = torch.arange(item_ids.shape[1], device=item_ids.device).unsqueeze(0)
        valid = valid < lengths.unsqueeze(1)
        x = model.embed_inputs(item_ids, behaviors, time_deltas) * valid.unsqueeze(-1)
        hidden_states = []
        normed_states = []
        kvs = []
        for block in model.blocks:
            hidden_states.append(x.detach())
            normed_states.append(block.norm(x).detach())
            x, (k, v) = block(x, return_kv=True)
            x = x * valid.unsqueeze(-1)
            kvs.append((k, v))
        kv = HSTUKVCache.from_layer_list(kvs, seq_len=item_ids.shape[1])
        return LayerwiseCacheState(
            kv=kv,
            hidden_states=tuple(hidden_states),
            normed_states=tuple(normed_states),
            lengths=lengths.detach(),
        )
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def migrate_legacy_suffix_cache(
    model: HSTU,
    state: LayerwiseCacheState,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    top_n_full: int,
) -> HSTUKVCache:
    """Reference suffix operator that executes the terminal block in full."""
    num_layers = len(model.blocks)
    if not 0 <= top_n_full <= num_layers:
        raise ValueError(f"top_n_full must be in [0, {num_layers}]")
    if len(state.hidden_states) != num_layers or len(state.normed_states) != num_layers:
        raise ValueError("layerwise state and model depth differ")

    was_training = model.training
    model.eval()
    try:
        split = num_layers - top_n_full
        kvs: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers
        for layer in range(split):
            normed = state.normed_states[layer]
            block = model.blocks[layer]
            kvs[layer] = (block.attn.k_proj(normed), block.attn.v_proj(normed))

        if top_n_full:
            lengths = state.lengths.to(item_ids.device)
            valid = torch.arange(item_ids.shape[1], device=item_ids.device).unsqueeze(0)
            valid = valid < lengths.unsqueeze(1)
            if split == 0:
                x = model.embed_inputs(item_ids, behaviors, time_deltas)
                x = x * valid.unsqueeze(-1)
            else:
                x = state.hidden_states[split]
            for layer in range(split, num_layers):
                x, (k, v) = model.blocks[layer](x, return_kv=True)
                x = x * valid.unsqueeze(-1)
                kvs[layer] = (k, v)

        complete = [kv for kv in kvs if kv is not None]
        if len(complete) != num_layers:
            raise RuntimeError("incomplete migrated cache")
        return HSTUKVCache.from_layer_list(complete, seq_len=item_ids.shape[1])
    finally:
        if was_training:
            model.train()


def contiguous_intervals(num_layers: int) -> list[tuple[int, int]]:
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    return [
        (start, end)
        for start in range(num_layers)
        for end in range(start, num_layers)
    ]


def _validate_interval(
    num_layers: int,
    start_layer: int | None,
    end_layer: int | None,
) -> None:
    if (start_layer is None) != (end_layer is None):
        raise ValueError("start_layer and end_layer must both be set or both be None")
    if start_layer is None:
        return
    if not 0 <= start_layer <= end_layer < num_layers:
        raise ValueError(
            f"interval must satisfy 0 <= start_layer <= end_layer < {num_layers}"
        )


@torch.no_grad()
def migrate_contiguous_cache(
    model: HSTU,
    state: LayerwiseCacheState,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    start_layer: int | None,
    end_layer: int | None,
) -> HSTUKVCache:
    num_layers = len(model.blocks)
    _validate_interval(num_layers, start_layer, end_layer)
    if len(state.hidden_states) != num_layers or len(state.normed_states) != num_layers:
        raise ValueError("layerwise state and model depth differ")

    was_training = model.training
    model.eval()
    try:
        kvs: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers
        for layer, block in enumerate(model.blocks):
            if start_layer is not None and start_layer <= layer <= end_layer:
                continue
            normed = state.normed_states[layer]
            kvs[layer] = (block.attn.k_proj(normed), block.attn.v_proj(normed))

        if start_layer is not None:
            lengths = state.lengths.to(item_ids.device)
            valid = torch.arange(item_ids.shape[1], device=item_ids.device).unsqueeze(0)
            valid = valid < lengths.unsqueeze(1)
            if start_layer == 0:
                x = model.embed_inputs(item_ids, behaviors, time_deltas)
                x = x * valid.unsqueeze(-1)
            else:
                x = state.hidden_states[start_layer]
            for layer in range(start_layer, end_layer):
                x, kv = model.blocks[layer](x, return_kv=True)
                x = x * valid.unsqueeze(-1)
                kvs[layer] = kv
            terminal = model.blocks[end_layer]
            normed = terminal.norm(x)
            kvs[end_layer] = (
                terminal.attn.k_proj(normed),
                terminal.attn.v_proj(normed),
            )

        complete = [kv for kv in kvs if kv is not None]
        if len(complete) != num_layers:
            raise RuntimeError("incomplete migrated cache")
        return HSTUKVCache.from_layer_list(complete, seq_len=item_ids.shape[1])
    finally:
        if was_training:
            model.train()


def migrate_suffix_cache(
    model: HSTU,
    state: LayerwiseCacheState,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    top_n_full: int,
) -> HSTUKVCache:
    """Current suffix operator with projection-only terminal execution."""
    num_layers = len(model.blocks)
    if not 0 <= top_n_full <= num_layers:
        raise ValueError(f"top_n_full must be in [0, {num_layers}]")
    if top_n_full == 0:
        interval = (None, None)
    else:
        interval = (num_layers - top_n_full, num_layers - 1)
    return migrate_contiguous_cache(
        model,
        state,
        item_ids,
        behaviors,
        time_deltas,
        *interval,
    )


def _validate_dense_prefix(
    state: LayerwiseCacheState,
    item_ids: torch.Tensor,
) -> int:
    seq_len = item_ids.shape[1]
    if state.kv.seq_len != seq_len:
        raise ValueError("state and input sequence widths differ")
    if not bool(torch.all(state.lengths == seq_len)):
        raise ValueError("recent replay requires length-bucketed dense prefixes")
    return seq_len


@torch.no_grad()
def migrate_recent_suffix_cache(
    model: HSTU,
    state: LayerwiseCacheState,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    top_n_full: int,
    replay_tokens: int,
) -> HSTUKVCache:
    num_layers = len(model.blocks)
    if not 0 <= top_n_full <= num_layers:
        raise ValueError(f"top_n_full must be in [0, {num_layers}]")
    seq_len = _validate_dense_prefix(state, item_ids)
    if not 0 <= replay_tokens <= seq_len:
        raise ValueError(f"replay_tokens must be in [0, {seq_len}]")
    if len(state.hidden_states) != num_layers or len(state.normed_states) != num_layers:
        raise ValueError("layerwise state and model depth differ")
    if top_n_full == 0 or replay_tokens == 0:
        return migrate_contiguous_cache(
            model,
            state,
            item_ids,
            behaviors,
            time_deltas,
            None,
            None,
        )

    was_training = model.training
    model.eval()
    try:
        split = num_layers - top_n_full
        stable_tokens = seq_len - replay_tokens
        kvs: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers
        for layer in range(split):
            normed = state.normed_states[layer]
            block = model.blocks[layer]
            kvs[layer] = (block.attn.k_proj(normed), block.attn.v_proj(normed))

        if split == 0:
            x = model.embed_inputs(
                item_ids[:, stable_tokens:],
                behaviors[:, stable_tokens:],
                time_deltas[:, stable_tokens:],
            )
        else:
            x = state.hidden_states[split][:, stable_tokens:]

        for layer in range(split, num_layers):
            block = model.blocks[layer]
            stable_normed = state.normed_states[layer][:, :stable_tokens]
            stable_k = block.attn.k_proj(stable_normed)
            stable_v = block.attn.v_proj(stable_normed)
            if layer < num_layers - 1:
                x, kv = block.forward_with_cache(x, stable_k, stable_v)
                kvs[layer] = kv
            else:
                normed = block.norm(x)
                tail_k = block.attn.k_proj(normed)
                tail_v = block.attn.v_proj(normed)
                kvs[layer] = (
                    torch.cat((stable_k, tail_k), dim=1),
                    torch.cat((stable_v, tail_v), dim=1),
                )

        complete = [kv for kv in kvs if kv is not None]
        if len(complete) != num_layers:
            raise RuntimeError("incomplete migrated cache")
        return HSTUKVCache.from_layer_list(complete, seq_len=seq_len)
    finally:
        if was_training:
            model.train()


def recent_suffix_extra_state_numel(
    state: LayerwiseCacheState,
    top_n_full: int,
    replay_tokens: int,
) -> int:
    num_layers = len(state.normed_states)
    if not 0 <= top_n_full <= num_layers:
        raise ValueError(f"top_n_full must be in [0, {num_layers}]")
    seq_len = state.kv.seq_len
    if not 0 <= replay_tokens <= seq_len:
        raise ValueError(f"replay_tokens must be in [0, {seq_len}]")
    if top_n_full == 0 or replay_tokens == 0:
        return sum(value.numel() for value in state.normed_states)
    split = num_layers - top_n_full
    stable_tokens = seq_len - replay_tokens
    batch = state.normed_states[0].shape[0]
    hidden = state.normed_states[0].shape[-1]
    total = split * batch * seq_len * hidden
    total += top_n_full * batch * stable_tokens * hidden
    if split > 0:
        total += batch * replay_tokens * hidden
    return total


@torch.no_grad()
def migrate_prefix_residual_cache(
    model: HSTU,
    state: LayerwiseCacheState,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    prefix_depth: int,
) -> HSTUKVCache:
    num_layers = len(model.blocks)
    if not 0 <= prefix_depth <= num_layers:
        raise ValueError(f"prefix_depth must be in [0, {num_layers}]")
    if len(state.hidden_states) != num_layers or len(state.normed_states) != num_layers:
        raise ValueError("layerwise state and model depth differ")
    if prefix_depth == 0:
        return migrate_contiguous_cache(
            model,
            state,
            item_ids,
            behaviors,
            time_deltas,
            None,
            None,
        )
    if prefix_depth == num_layers:
        return migrate_contiguous_cache(
            model,
            state,
            item_ids,
            behaviors,
            time_deltas,
            0,
            num_layers - 1,
        )

    was_training = model.training
    model.eval()
    try:
        lengths = state.lengths.to(item_ids.device)
        positions = torch.arange(item_ids.shape[1], device=item_ids.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        x = model.embed_inputs(item_ids, behaviors, time_deltas)
        x = x * valid.unsqueeze(-1)
        kvs = []
        for layer in range(prefix_depth):
            x, kv = model.blocks[layer](x, return_kv=True)
            x = x * valid.unsqueeze(-1)
            kvs.append(kv)
        boundary_delta = x - state.hidden_states[prefix_depth]
        boundary_delta = boundary_delta * valid.unsqueeze(-1)
        for layer in range(prefix_depth, num_layers):
            hidden = state.hidden_states[layer] + boundary_delta
            hidden = hidden * valid.unsqueeze(-1)
            block = model.blocks[layer]
            normed = block.norm(hidden)
            kvs.append(
                (
                    block.attn.k_proj(normed),
                    block.attn.v_proj(normed),
                )
            )
        return HSTUKVCache.from_layer_list(kvs, seq_len=item_ids.shape[1])
    finally:
        if was_training:
            model.train()


def prefix_residual_extra_state_numel(
    state: LayerwiseCacheState,
    prefix_depth: int,
) -> int:
    num_layers = len(state.hidden_states)
    if not 0 <= prefix_depth <= num_layers:
        raise ValueError(f"prefix_depth must be in [0, {num_layers}]")
    if prefix_depth == 0:
        return sum(value.numel() for value in state.normed_states)
    return sum(
        state.hidden_states[layer].numel()
        for layer in range(prefix_depth, num_layers)
    )


@torch.no_grad()
def migrate_current_norm_cache(
    model: HSTU,
    state: LayerwiseCacheState,
) -> HSTUKVCache:
    num_layers = len(model.blocks)
    if len(state.hidden_states) != num_layers:
        raise ValueError("layerwise state and model depth differ")
    was_training = model.training
    model.eval()
    try:
        kvs = []
        for layer, block in enumerate(model.blocks):
            normed = block.norm(state.hidden_states[layer])
            kvs.append((block.attn.k_proj(normed), block.attn.v_proj(normed)))
        return HSTUKVCache.from_layer_list(kvs, seq_len=state.kv.seq_len)
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def migrate_embedding_delta_cache(
    model: HSTU,
    state: LayerwiseCacheState,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    layer_scales: torch.Tensor | None = None,
) -> HSTUKVCache:
    num_layers = len(model.blocks)
    if len(state.hidden_states) != num_layers:
        raise ValueError("layerwise state and model depth differ")
    if layer_scales is None:
        layer_scales = torch.ones(
            num_layers,
            device=item_ids.device,
            dtype=state.hidden_states[0].dtype,
        )
    if layer_scales.numel() != num_layers:
        raise ValueError(f"layer_scales must contain {num_layers} values")
    was_training = model.training
    model.eval()
    try:
        lengths = state.lengths.to(item_ids.device)
        valid = torch.arange(item_ids.shape[1], device=item_ids.device).unsqueeze(0)
        valid = valid < lengths.unsqueeze(1)
        current_input = model.embed_inputs(item_ids, behaviors, time_deltas)
        current_input = current_input * valid.unsqueeze(-1)
        input_delta = current_input - state.hidden_states[0]
        kvs = []
        for layer, block in enumerate(model.blocks):
            hidden = state.hidden_states[layer] + layer_scales[layer] * input_delta
            normed = block.norm(hidden)
            kvs.append((block.attn.k_proj(normed), block.attn.v_proj(normed)))
        return HSTUKVCache.from_layer_list(kvs, seq_len=state.kv.seq_len)
    finally:
        if was_training:
            model.train()


def extra_state_numel(state: LayerwiseCacheState, top_n_full: int) -> int:
    num_layers = len(state.normed_states)
    if not 0 <= top_n_full <= num_layers:
        raise ValueError(f"top_n_full must be in [0, {num_layers}]")
    split = num_layers - top_n_full
    total = sum(state.normed_states[layer].numel() for layer in range(split))
    if 0 < top_n_full < num_layers:
        total += state.hidden_states[split].numel()
    return total


def interval_extra_state_numel(
    state: LayerwiseCacheState,
    start_layer: int | None,
    end_layer: int | None,
) -> int:
    num_layers = len(state.normed_states)
    _validate_interval(num_layers, start_layer, end_layer)
    if start_layer is None:
        return sum(value.numel() for value in state.normed_states)
    total = sum(
        value.numel()
        for layer, value in enumerate(state.normed_states)
        if not start_layer <= layer <= end_layer
    )
    if start_layer > 0:
        total += state.hidden_states[start_layer].numel()
    return total


def sample_relative_cache_error(
    cache: HSTUKVCache,
    fresh: HSTUKVCache,
) -> torch.Tensor:
    delta = (cache.k.float() - fresh.k.float()).square().sum(dim=(0, 2, 3))
    delta += (cache.v.float() - fresh.v.float()).square().sum(dim=(0, 2, 3))
    scale = fresh.k.float().square().sum(dim=(0, 2, 3))
    scale += fresh.v.float().square().sum(dim=(0, 2, 3))
    return delta.sqrt() / scale.sqrt().clamp_min(1e-12)
