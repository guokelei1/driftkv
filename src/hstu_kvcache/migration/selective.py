from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models import HSTU, HSTUKVCache
from .layerwise import LayerwiseCacheState


@dataclass(frozen=True)
class SelectiveContiguousState:
    source_kv: HSTUKVCache
    transition_hidden: torch.Tensor | None
    lengths: torch.Tensor
    start_layer: int

    @property
    def batch_size(self) -> int:
        return self.source_kv.k.shape[1]

    @property
    def seq_len(self) -> int:
        return self.source_kv.seq_len

    @property
    def nbytes(self) -> int:
        tensors = [self.source_kv.k, self.source_kv.v, self.lengths]
        if self.transition_hidden is not None:
            tensors.append(self.transition_hidden)
        return sum(value.numel() * value.element_size() for value in tensors)


@dataclass(frozen=True)
class ResidualHiddenSuffixState:
    hidden_states: tuple[torch.Tensor, ...]
    lengths: torch.Tensor
    start_layer: int
    num_layers: int

    @property
    def nbytes(self) -> int:
        tensors = (*self.hidden_states, self.lengths)
        return sum(value.numel() * value.element_size() for value in tensors)


def selective_contiguous_intervals(
    num_layers: int,
    widths: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    if not widths or len(set(widths)) != len(widths):
        raise ValueError("widths must be unique and nonempty")
    if any(width < 1 or width > num_layers for width in widths):
        raise ValueError("widths must be within the model depth")
    return tuple(
        (start, start + width - 1)
        for width in widths
        for start in range(num_layers - width + 1)
    )


def capture_selective_contiguous_state(
    state: LayerwiseCacheState,
    start_layer: int,
    storage_dtype: torch.dtype | None = None,
) -> SelectiveContiguousState:
    num_layers = len(state.hidden_states)
    if len(state.normed_states) != num_layers:
        raise ValueError("layerwise state is incomplete")
    if not 0 <= start_layer < num_layers:
        raise ValueError("start_layer must be within the model depth")
    dtype = storage_dtype or state.kv.k.dtype
    source_kv = HSTUKVCache(
        k=state.kv.k.to(dtype=dtype),
        v=state.kv.v.to(dtype=dtype),
        seq_len=state.kv.seq_len,
    )
    transition = None
    if start_layer > 0:
        transition = state.hidden_states[start_layer].to(dtype=dtype)
    return SelectiveContiguousState(
        source_kv=source_kv,
        transition_hidden=transition,
        lengths=state.lengths,
        start_layer=start_layer,
    )


def capture_residual_hidden_suffix(
    state: LayerwiseCacheState,
    start_layer: int,
    storage_dtype: torch.dtype | None = None,
) -> ResidualHiddenSuffixState:
    num_layers = len(state.hidden_states)
    if len(state.normed_states) != num_layers:
        raise ValueError("layerwise state is incomplete")
    if not 1 <= start_layer <= num_layers:
        raise ValueError("start_layer must be in [1, num_layers]")
    dtype = storage_dtype or state.hidden_states[0].dtype
    hidden_states = tuple(
        state.hidden_states[layer].to(dtype=dtype)
        for layer in range(start_layer, num_layers)
    )
    return ResidualHiddenSuffixState(
        hidden_states=hidden_states,
        lengths=state.lengths,
        start_layer=start_layer,
        num_layers=num_layers,
    )


def _validate_history(
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    lengths: torch.Tensor,
) -> None:
    if item_ids.ndim != 2:
        raise ValueError("item_ids must have shape [batch, sequence]")
    if behaviors.shape != item_ids.shape or time_deltas.shape != item_ids.shape:
        raise ValueError("history tensor shapes differ")
    if lengths.shape != (item_ids.shape[0],):
        raise ValueError("lengths and history batch dimension differ")
    devices = {
        item_ids.device,
        behaviors.device,
        time_deltas.device,
        lengths.device,
    }
    if len(devices) != 1:
        raise ValueError("history tensors must share one device")
    if bool(torch.any(lengths < 0)) or bool(torch.any(lengths > item_ids.shape[1])):
        raise ValueError("lengths must be within the padded sequence width")


def _validate_source_kv(
    model: HSTU,
    source_kv: HSTUKVCache,
    item_ids: torch.Tensor,
) -> None:
    expected = (
        len(model.blocks),
        item_ids.shape[0],
        item_ids.shape[1],
        model.blocks[0].attn.inner,
    )
    if source_kv.k.shape != expected or source_kv.v.shape != expected:
        raise ValueError("source K/V shape differs from model and history")
    if source_kv.seq_len != item_ids.shape[1]:
        raise ValueError("source K/V and history sequence widths differ")
    if source_kv.k.device != item_ids.device or source_kv.v.device != item_ids.device:
        raise ValueError("source K/V and history must share one device")
    if source_kv.k.dtype != source_kv.v.dtype:
        raise ValueError("source K/V dtypes differ")


@torch.no_grad()
def migrate_selective_contiguous_cache(
    model: HSTU,
    state: SelectiveContiguousState,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    end_layer: int,
) -> HSTUKVCache:
    num_layers = len(model.blocks)
    start_layer = state.start_layer
    if not start_layer <= end_layer < num_layers:
        raise ValueError("end_layer must be at or after start_layer")
    _validate_history(
        item_ids,
        behaviors,
        time_deltas,
        state.lengths,
    )
    _validate_source_kv(model, state.source_kv, item_ids)
    if state.start_layer == 0:
        if state.transition_hidden is not None:
            raise ValueError("layer-zero replay must not carry transition hidden state")
    else:
        expected = (
            item_ids.shape[0],
            item_ids.shape[1],
            model.cfg.hidden_size,
        )
        if (
            state.transition_hidden is None
            or state.transition_hidden.shape != expected
        ):
            raise ValueError("transition hidden state shape mismatch")
        if state.transition_hidden.device != item_ids.device:
            raise ValueError("transition hidden state and history must share one device")

    was_training = model.training
    model.eval()
    try:
        lengths = state.lengths.to(item_ids.device)
        positions = torch.arange(item_ids.shape[1], device=item_ids.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        if start_layer == 0:
            x = model.embed_inputs(item_ids, behaviors, time_deltas)
            x = x * valid.unsqueeze(-1)
        else:
            assert state.transition_hidden is not None
            x = state.transition_hidden * valid.unsqueeze(-1)
        kvs = [
            (state.source_kv.k[layer], state.source_kv.v[layer])
            for layer in range(num_layers)
        ]
        output_dtype = state.source_kv.k.dtype
        for layer in range(start_layer, end_layer):
            x, (k, v) = model.blocks[layer](x, return_kv=True)
            x = x * valid.unsqueeze(-1)
            kvs[layer] = (
                k.to(dtype=output_dtype),
                v.to(dtype=output_dtype),
            )
        terminal = model.blocks[end_layer]
        normed = terminal.norm(x)
        terminal_k = terminal.attn.k_proj(normed)
        terminal_v = terminal.attn.v_proj(normed)
        kvs[end_layer] = (
            terminal_k.to(dtype=output_dtype),
            terminal_v.to(dtype=output_dtype),
        )
        return HSTUKVCache.from_layer_list(kvs, seq_len=item_ids.shape[1])
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def migrate_prefix_residual_from_hidden_suffix(
    model: HSTU,
    state: ResidualHiddenSuffixState,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
) -> HSTUKVCache:
    num_layers = len(model.blocks)
    if state.num_layers != num_layers:
        raise ValueError("hidden suffix and model depth differ")
    if len(state.hidden_states) != num_layers - state.start_layer:
        raise ValueError("hidden suffix is incomplete")
    _validate_history(
        item_ids,
        behaviors,
        time_deltas,
        state.lengths,
    )
    expected = (
        item_ids.shape[0],
        item_ids.shape[1],
        model.cfg.hidden_size,
    )
    if any(value.shape != expected for value in state.hidden_states):
        raise ValueError("hidden suffix tensor shape mismatch")
    if any(value.device != item_ids.device for value in state.hidden_states):
        raise ValueError("hidden suffix and history must share one device")

    was_training = model.training
    model.eval()
    try:
        positions = torch.arange(item_ids.shape[1], device=item_ids.device)
        valid = positions.unsqueeze(0) < state.lengths.unsqueeze(1)
        x = model.embed_inputs(item_ids, behaviors, time_deltas)
        x = x * valid.unsqueeze(-1)
        kvs = []
        for layer in range(state.start_layer):
            x, kv = model.blocks[layer](x, return_kv=True)
            x = x * valid.unsqueeze(-1)
            kvs.append(kv)
        if state.start_layer == num_layers:
            return HSTUKVCache.from_layer_list(
                kvs,
                seq_len=item_ids.shape[1],
            )
        boundary_delta = x - state.hidden_states[0]
        boundary_delta = boundary_delta * valid.unsqueeze(-1)
        for offset, layer in enumerate(range(state.start_layer, num_layers)):
            hidden = state.hidden_states[offset] + boundary_delta
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
