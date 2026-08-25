from __future__ import annotations

import copy

import torch

from hstu_kvcache.models import (
    HSTU,
    HSTUConfig,
    append_with_rolling_cap,
    hybrid_tail_refresh,
    project_exact_layer0_segment,
    retain_latest_cache,
    transition_work,
)


def model() -> HSTU:
    torch.manual_seed(7)
    return HSTU(HSTUConfig(
        num_items=64, num_prediction_items=64, num_behaviors=4,
        hidden_size=16, num_layers=4, num_heads=2, max_seq_len=32,
        input_dropout=0.0, attn_dropout=0.0,
    )).eval()


def raw() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long),
        torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2]], dtype=torch.long),
        torch.tensor([[0, 5, 9, 3, 7, 2, 11, 4]], dtype=torch.float32),
    )


def test_layer0_segment_projection_matches_current_exact_and_preserves_other_state() -> None:
    current = model()
    parent = copy.deepcopy(current)
    with torch.no_grad():
        parent.blocks[0].attn.k_proj.weight.add_(0.2)
        parent.blocks[0].attn.v_proj.weight.sub_(0.1)
    items, behaviors, deltas = raw()
    parent_cache = parent.compute_kv(items, behaviors, deltas)
    exact = current.compute_kv(items, behaviors, deltas)
    migrated = project_exact_layer0_segment(current, parent_cache, items, behaviors, deltas, "middle")
    selected = slice(2, 6)
    assert torch.allclose(migrated.k[0, :, selected], exact.k[0, :, selected], atol=1e-7, rtol=0)
    assert torch.allclose(migrated.v[0, :, selected], exact.v[0, :, selected], atol=1e-7, rtol=0)
    assert torch.equal(migrated.k[0, :, :2], parent_cache.k[0, :, :2])
    assert torch.equal(migrated.k[0, :, 6:], parent_cache.k[0, :, 6:])


def test_hybrid_tail_is_exact_when_parent_prefix_is_current_model() -> None:
    current = model()
    items, behaviors, deltas = raw()
    exact = current.compute_kv(items, behaviors, deltas)
    replayed = hybrid_tail_refresh(current, exact, items, behaviors, deltas, width=3)
    assert torch.allclose(replayed.k, exact.k, atol=1e-6, rtol=0)
    assert torch.allclose(replayed.v, exact.v, atol=1e-6, rtol=0)


def test_retain_latest_cache_keeps_tail_not_prefix() -> None:
    current = model()
    items, behaviors, deltas = raw()
    cache = current.compute_kv(items, behaviors, deltas)
    latest = retain_latest_cache(cache, 3)
    assert latest.seq_len == 3
    assert torch.equal(latest.k, cache.k[:, :, -3:])
    assert torch.equal(latest.v, cache.v[:, :, -3:])


def test_rolling_append_evicts_before_each_event() -> None:
    current = model()
    items, behaviors, deltas = raw()
    initial = current.compute_kv(items[:, :4], behaviors[:, :4], deltas[:, :4])
    rolled = append_with_rolling_cap(
        current, initial, items[:, 4:6], behaviors[:, 4:6], deltas[:, 4:6], max_length=4
    )
    manual = initial
    for position in range(4, 6):
        manual = retain_latest_cache(manual, 3)
        _, manual = current.forward_with_cache(
            manual, items[:, position:position + 1], behaviors[:, position:position + 1],
            deltas[:, position:position + 1],
        )
    assert rolled.seq_len == 4
    assert torch.equal(rolled.k, manual.k)
    assert torch.equal(rolled.v, manual.v)


def test_transition_work_keeps_compute_and_io_dimensions_separate() -> None:
    current = model()
    items, behaviors, deltas = raw()
    cache = current.compute_kv(items, behaviors, deltas)
    layer0 = transition_work("layer0_middle", cache, items, behaviors, deltas)
    hybrid = transition_work("hybrid_tail32", cache, items, behaviors, deltas)
    exact = transition_work("exact_all", cache, items, behaviors, deltas)
    assert layer0.recomputed_token_layers == 4
    assert layer0.attention_pair_work == 0
    assert hybrid.recomputed_token_layers == exact.recomputed_token_layers == 32
    assert hybrid.attention_pair_work == exact.attention_pair_work
    assert layer0.raw_history_read_bytes < exact.raw_history_read_bytes
