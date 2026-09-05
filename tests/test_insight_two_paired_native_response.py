from __future__ import annotations

import torch
from insight_two.mode_space_replay import paired_release_replay
from insight_two.paired_native_response import (
    build_paired_native_response_memory,
    intervene_paired_native_response,
    medium_paired_native_response_cost,
    select_paired_native_response_rows,
)

from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache


def _model(seed: int, *, length: int = 8) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=64,
            num_behaviors=3,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            max_seq_len=length,
            temporal_num_freqs=2,
            input_dropout=0.0,
            activation="elu_plus1",
            block_variant="legacy",
            relative_position_bias=False,
        )
    ).eval()


def _history(length: int = 6) -> tuple[torch.Tensor, ...]:
    items = (torch.arange(length).reshape(1, length) % 61 + 1).long()
    actions = (torch.arange(length).reshape(1, length) % 2 + 1).long()
    times = torch.arange(1, length + 1).reshape(1, length).float()
    return items, actions, times


def _select_cache(cache: HSTUKVCache, positions: torch.Tensor) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k.index_select(2, positions),
        v=cache.v.index_select(2, positions),
        seq_len=int(positions.numel()),
    )


def test_full_token_rank_recovers_exact_current_reader() -> None:
    parent = _model(301)
    current = _model(303)
    items, actions, times = _history()
    exact_parent = parent.compute_kv(items, actions, times)
    exact_current = current.compute_kv(items, actions, times)
    replay = paired_release_replay(
        parent,
        current,
        parent.embed_inputs(items, actions, times),
        current.embed_inputs(items, actions, times),
        rank=items.shape[1],
        compression="exact_svd",
    )
    memory = build_paired_native_response_memory(
        replay,
        source_kv_scalars=exact_parent.k.numel() + exact_parent.v.numel(),
    )
    candidates = torch.tensor([[21, 22, 23]])
    query_time = torch.tensor([9.0])
    observed = intervene_paired_native_response(
        current, exact_parent, memory, candidates, query_time
    )
    expected = current.score_cc_reuse(
        exact_current, candidates, query_time
    )
    assert torch.allclose(observed.scores, expected, atol=2e-4, rtol=2e-4)


def test_identical_releases_reduce_to_exact_parent_reuse_at_low_rank() -> None:
    model = _model(311)
    items, actions, times = _history()
    exact_parent = model.compute_kv(items, actions, times)
    replay = paired_release_replay(
        model,
        model,
        model.embed_inputs(items, actions, times),
        model.embed_inputs(items, actions, times),
        rank=2,
        compression="fixed_range_finder",
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=17,
    )
    memory = build_paired_native_response_memory(
        replay,
        source_kv_scalars=exact_parent.k.numel() + exact_parent.v.numel(),
    )
    candidates = torch.tensor([[31, 32]])
    query_time = torch.tensor([8.0])
    observed = intervene_paired_native_response(
        model, exact_parent, memory, candidates, query_time
    )
    expected = model.score_cc_reuse(exact_parent, candidates, query_time)
    assert torch.allclose(observed.scores, expected, atol=1e-6, rtol=1e-6)
    for signed in observed.layer_signed_heads:
        assert torch.count_nonzero(signed) == 0


def test_factor_rows_are_eviction_closed_at_full_rank() -> None:
    parent = _model(321)
    current = _model(323)
    items, actions, times = _history()
    exact_parent = parent.compute_kv(items, actions, times)
    exact_current = current.compute_kv(items, actions, times)
    replay = paired_release_replay(
        parent,
        current,
        parent.embed_inputs(items, actions, times),
        current.embed_inputs(items, actions, times),
        rank=items.shape[1],
        compression="exact_svd",
    )
    memory = build_paired_native_response_memory(
        replay,
        source_kv_scalars=exact_parent.k.numel() + exact_parent.v.numel(),
    )
    keep = torch.tensor([1, 3, 4, 5])
    selected_memory = select_paired_native_response_rows(memory, keep)
    selected_parent = _select_cache(exact_parent, keep)
    selected_current = _select_cache(exact_current, keep)
    candidates = torch.tensor([[41, 42]])
    query_time = torch.tensor([9.0])
    observed = intervene_paired_native_response(
        current,
        selected_parent,
        selected_memory,
        candidates,
        query_time,
    )
    expected = current.score_cc_reuse(
        selected_current, candidates, query_time
    )
    assert selected_memory.source_length == 4
    assert torch.allclose(observed.scores, expected, atol=2e-4, rtol=2e-4)


def test_exact_current_suffix_adds_without_rewriting_cutover_sidecar() -> None:
    parent = _model(331)
    current = _model(333)
    items, actions, times = _history()
    cutover = 4
    old = tuple(values[:, :cutover] for values in (items, actions, times))
    exact_parent = parent.compute_kv(*old)
    replay = paired_release_replay(
        parent,
        current,
        parent.embed_inputs(*old),
        current.embed_inputs(*old),
        rank=cutover,
        compression="exact_svd",
    )
    memory = build_paired_native_response_memory(
        replay,
        source_kv_scalars=exact_parent.k.numel() + exact_parent.v.numel(),
    )
    exact_current = current.compute_kv(items, actions, times)
    suffix_positions = torch.arange(cutover, items.shape[1])
    current_suffix = _select_cache(exact_current, suffix_positions)
    candidates = torch.tensor([[51, 52]])
    query_time = torch.tensor([9.0])
    observed = intervene_paired_native_response(
        current,
        exact_parent,
        memory,
        candidates,
        query_time,
        current_suffix=current_suffix,
    )
    expected = current.score_cc_reuse(
        exact_current, candidates, query_time
    )
    assert memory.source_length == cutover
    assert torch.allclose(observed.scores, expected, atol=2e-4, rtol=2e-4)


def test_medium_cost_and_two_arm_sidecar_are_explicit() -> None:
    cost = medium_paired_native_response_cost()
    assert cost.total_constructor_flops == 872_238_088
    assert 0.182 < cost.constructor_fraction < 0.184
    assert cost.within_twenty_percent
    assert cost.sidecar_scalars == 67_584
    assert cost.sidecar_fp32_bytes == 270_336
    assert cost.factor_reads_per_layer_per_query == 2
    assert cost.factor_reads_per_query == 12
    assert cost.factorized_read_flops_per_arm_per_layer_per_query == 101_376
    assert cost.incremental_reader_flops_per_query == 1_218_816
    assert cost.native_activation_evaluations_per_query == 73_728
