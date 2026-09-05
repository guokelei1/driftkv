from __future__ import annotations

import pytest
import torch
from insight_two.activation_boundary_replay import (
    build_no_target_boundary_replay_cache,
    intervene_serving_boundary_delta,
    medium_activation_boundary_cost_audit,
    trace_exact_endpoint_graphs,
)

from hstu_kvcache.models import HSTU, HSTUConfig


def _model(seed: int) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=41,
            num_behaviors=2,
            hidden_size=8,
            num_layers=2,
            num_heads=2,
            max_seq_len=8,
            input_dropout=0.0,
        )
    ).eval()


def _history() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[1, 2, 3, 4, 5]]),
        torch.tensor([[0, 1, 0, 1, 0]]),
        torch.tensor([[1.0, 2.0, 4.0, 7.0, 11.0]]),
    )


def test_exact_endpoint_trace_matches_native_caches_and_partitions_delta() -> None:
    parent = _model(7)
    current = _model(11)
    items, actions, times = _history()
    trace = trace_exact_endpoint_graphs(parent, current, items, actions, times)
    native_parent = parent.compute_kv(items, actions, times)
    native_current = current.compute_kv(items, actions, times)
    torch.testing.assert_close(trace.parent_cache.k, native_parent.k)
    torch.testing.assert_close(trace.parent_cache.v, native_parent.v)
    torch.testing.assert_close(trace.current_cache.k, native_current.k)
    torch.testing.assert_close(trace.current_cache.v, native_current.v)
    assert len(trace.layer_metrics) == 2
    for layer in trace.layer_metrics:
        assert (
            layer.activation_region_agreement + layer.activation_region_crossing_fraction
            == pytest.approx(1.0)
        )
        assert layer.decomposition_relative_l2_error < 1e-6


def test_current_and_parent_serving_modes_match_native_reader() -> None:
    parent = _model(13)
    current = _model(17)
    items, actions, times = _history()
    trace = trace_exact_endpoint_graphs(parent, current, items, actions, times)
    candidates = torch.tensor([[6, 7, 8]])
    query_times = torch.tensor([12.0])
    current_result = intervene_serving_boundary_delta(
        current,
        trace.current_cache,
        trace.parent_cache,
        candidates,
        query_times,
        mode="current",
    )
    parent_result = intervene_serving_boundary_delta(
        current,
        trace.current_cache,
        trace.parent_cache,
        candidates,
        query_times,
        mode="parent",
    )
    torch.testing.assert_close(
        current_result.scores,
        current.score_cc_reuse(trace.current_cache, candidates, query_times),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        parent_result.scores,
        current.score_cc_reuse(trace.parent_cache, candidates, query_times),
        atol=2e-5,
        rtol=2e-5,
    )


def test_no_target_boundary_replay_has_exact_identity_endpoint() -> None:
    model = _model(19)
    items, actions, times = _history()
    exact = model.compute_kv(items, actions, times)
    replay = build_no_target_boundary_replay_cache(model, exact, items, actions, times)
    torch.testing.assert_close(replay.cache.k, exact.k, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(replay.cache.v, exact.v, atol=2e-5, rtol=2e-5)
    assert replay.crossing_fraction_by_active_layer == (0.0,)


def test_exact_boundary_discovery_exceeds_twenty_percent_before_response() -> None:
    audit = medium_activation_boundary_cost_audit()
    assert audit.causal_pairs_per_layer_per_head == 524_800
    assert audit.current_graph_qk_floor_flops == 1_007_616_000
    assert audit.current_graph_qk_floor_over_exact == pytest.approx(0.21118345145871106)
    assert audit.no_target_two_graph_one_value_floor_flops == 3_022_848_000
    assert audit.no_target_two_graph_one_value_floor_over_exact == pytest.approx(0.6335503543761332)
    assert audit.parent_graph_bits_if_persisted == 15_744_000
    assert not audit.within_twenty_percent_before_response_or_projection
    assert audit.verdict.startswith("NO_GO")
