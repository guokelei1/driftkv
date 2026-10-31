"""Regression tests for temporal continuity across a persistent KV boundary."""

from __future__ import annotations

import torch

from hstu_kvcache.data import event_time_deltas
from hstu_kvcache.models import HSTU, HSTUConfig


def compact_history_tensors(history, item_map, device, *, previous_timestamp=None):
    items = torch.tensor([[item_map[item] for item, _, _ in history]], dtype=torch.long, device=device)
    behaviors = torch.tensor([[behavior for _, _, behavior in history]], dtype=torch.long, device=device)
    deltas = torch.from_numpy(
        event_time_deltas(history, previous_timestamp=previous_timestamp)
    ).to(device)
    lengths = torch.tensor([len(history)], dtype=torch.long, device=device)
    return items, behaviors, deltas[None, :], lengths


def test_first_incremental_token_keeps_delta_from_cached_prefix() -> None:
    prefix = [(10, 100, 1), (11, 130, 2)]
    suffix = [(12, 185, 1), (13, 210, 2)]

    assert event_time_deltas(prefix).tolist() == [0.0, 30.0]
    assert event_time_deltas(suffix).tolist() == [0.0, 25.0]
    assert event_time_deltas(suffix, previous_timestamp=prefix[-1][1]).tolist() == [55.0, 25.0]


def test_same_model_full_and_append_match_with_continuous_temporal_inputs() -> None:
    torch.manual_seed(7)
    model = HSTU(
        HSTUConfig(
            num_items=32,
            num_behaviors=3,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            input_dropout=0.0,
        )
    ).eval()
    item_map = {raw: raw for raw in range(1, 32)}
    prefix = [(10, 100, 1), (11, 130, 2)]
    suffix = [(12, 185, 1), (13, 210, 2)]

    full_items, full_behaviors, full_deltas, full_lengths = compact_history_tensors(
        prefix + suffix, item_map, torch.device("cpu")
    )
    prefix_items, prefix_behaviors, prefix_deltas, prefix_lengths = compact_history_tensors(
        prefix, item_map, torch.device("cpu")
    )
    suffix_items, suffix_behaviors, suffix_deltas, _ = compact_history_tensors(
        suffix,
        item_map,
        torch.device("cpu"),
        previous_timestamp=prefix[-1][1],
    )

    with torch.inference_mode():
        full_hidden, _ = model(full_items, full_behaviors, full_deltas, lengths=full_lengths)
        cache = model.compute_kv(prefix_items, prefix_behaviors, prefix_deltas, prefix_lengths)
        append_hidden, _ = model.forward_with_cache(cache, suffix_items, suffix_behaviors, suffix_deltas)

    assert torch.allclose(full_hidden[:, -len(suffix) :, :], append_hidden, atol=1e-6, rtol=1e-6)


def test_readout_ablation_scales_preserve_default_execution_and_are_finite() -> None:
    torch.manual_seed(11)
    model = HSTU(
        HSTUConfig(
            num_items=16,
            num_behaviors=3,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            input_dropout=0.0,
        )
    ).eval()
    items = torch.tensor([[1, 2, 3]], dtype=torch.long)
    behaviors = torch.tensor([[1, 2, 1]], dtype=torch.long)
    deltas = torch.tensor([[0.0, 30.0, 45.0]])
    lengths = torch.tensor([3])
    with torch.inference_mode():
        default, _ = model(items, behaviors, deltas, lengths=lengths)
        explicit, _ = model(
            items, behaviors, deltas, lengths=lengths,
            residual_scale=1.0, attention_scale=1.0,
        )
        residual_only, _ = model(
            items, behaviors, deltas, lengths=lengths,
            residual_scale=1.0, attention_scale=0.0,
        )
        attention_only, _ = model(
            items, behaviors, deltas, lengths=lengths,
            residual_scale=0.0, attention_scale=1.0,
        )
    assert torch.allclose(default, explicit, atol=0.0, rtol=0.0)
    assert torch.isfinite(residual_only).all()
    assert torch.isfinite(attention_only).all()
