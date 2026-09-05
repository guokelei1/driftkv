from __future__ import annotations

import pytest
import torch
from insight_two.producer_reader_commutator import (
    commuted_endpoint,
    finite_difference_diagnostic,
    trace_reader_producer,
)

from hstu_kvcache.models import HSTU, HSTUConfig


def _model(seed: int) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=31,
            num_behaviors=2,
            hidden_size=8,
            num_layers=2,
            num_heads=2,
            max_seq_len=8,
            input_dropout=0.0,
        )
    ).eval()


def _inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([[1, 2, 3, 4]]),
        torch.tensor([[0, 1, 0, 1]]),
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        torch.tensor([[5, 6, 7]]),
        torch.tensor([5.0]),
    )


def test_four_traces_match_each_readers_native_cross_cache_scores() -> None:
    parent = _model(7)
    current = _model(11)
    items, actions, times, candidates, query_times = _inputs()
    parent_cache = parent.compute_kv(items, actions, times)
    current_cache = current.compute_kv(items, actions, times)

    for reader, cache in (
        (parent, parent_cache),
        (parent, current_cache),
        (current, parent_cache),
        (current, current_cache),
    ):
        trace = trace_reader_producer(
            reader,
            cache,
            candidates,
            query_times,
        )
        expected_scores, expected_readout = reader.observe_cc_reuse(
            cache,
            candidates,
            query_times,
        )
        assert torch.allclose(trace.scores, expected_scores, atol=2e-5, rtol=2e-5)
        assert torch.allclose(
            trace.readout,
            expected_readout,
            atol=2e-5,
            rtol=2e-5,
        )
        assert len(trace.layer_s4) == len(reader.blocks)
        assert trace.layer_s4[0].shape == (1, candidates.shape[1], 8)


def test_commuted_endpoint_error_is_exactly_mixed_finite_difference() -> None:
    current_current = torch.tensor([[3.0, 5.0, 8.0]])
    current_parent = torch.tensor([[2.0, 4.0, 7.0]])
    parent_current = torch.tensor([[1.5, 2.5, 4.0]])
    parent_parent = torch.tensor([[1.0, 2.0, 3.0]])
    estimate = commuted_endpoint(current_parent, parent_current, parent_parent)
    diagnostic = finite_difference_diagnostic(
        current_current,
        current_parent,
        parent_current,
        parent_parent,
    )
    assert torch.equal(current_current - estimate, diagnostic.mixed)
    assert diagnostic.mixed_over_current_state_l2 >= 0.0
    assert diagnostic.l2_recovery == pytest.approx(
        1.0 - diagnostic.mixed_over_current_state_l2
    )


def test_zero_mixed_difference_recovers_current_endpoint() -> None:
    parent_parent = torch.randn(1, 4, 8)
    reader_shift = torch.randn_like(parent_parent)
    producer_shift = torch.randn_like(parent_parent)
    current_parent = parent_parent + reader_shift
    parent_current = parent_parent + producer_shift
    current_current = parent_parent + reader_shift + producer_shift
    estimate = commuted_endpoint(current_parent, parent_current, parent_parent)
    diagnostic = finite_difference_diagnostic(
        current_current,
        current_parent,
        parent_current,
        parent_parent,
    )
    assert torch.allclose(estimate, current_current)
    assert torch.allclose(
        diagnostic.mixed,
        torch.zeros_like(current_current),
        atol=1e-6,
    )
    assert diagnostic.l2_recovery == pytest.approx(1.0, abs=1e-6)


def test_trace_rejects_training_reader_and_shape_mismatch() -> None:
    model = _model(7)
    items, actions, times, candidates, query_times = _inputs()
    cache = model.compute_kv(items, actions, times)
    model.train()
    with pytest.raises(ValueError, match="model.eval"):
        trace_reader_producer(model, cache, candidates, query_times)
    with pytest.raises(ValueError, match="endpoint shapes"):
        finite_difference_diagnostic(
            torch.zeros(1, 2),
            torch.zeros(1, 3),
            torch.zeros(1, 2),
            torch.zeros(1, 2),
        )
