from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from insight_two.historical_prefix_moment_closure import (  # noqa: E402
    activation_region_statistics,
    build_prefix_affine_moments,
    combine_affine_summaries,
    dense_prefix_moment_cost,
    diagnose_model_pair,
    exact_history_response_heads,
    exact_to_rollout_layer_diagnostics,
    fixed_historical_query_positions,
    historical_probe_region,
    paired_teacher_forced_diagnostics,
    read_prefix_affine_moments,
    rollout_dense_prefix_moments,
    summarize_affine_segment,
    trace_exact_history_layers,
)

from hstu_kvcache.models import HSTU, HSTUConfig  # noqa: E402


def _model(seed: int, *, length: int = 16) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=128,
            num_behaviors=3,
            hidden_size=16,
            num_layers=3,
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


def _history(length: int = 16) -> tuple[torch.Tensor, ...]:
    items = (torch.arange(length).reshape(1, length) + 1).long()
    behaviors = (torch.arange(length).reshape(1, length) % 2 + 1).long()
    deltas = torch.arange(length).reshape(1, length).float()
    return items, behaviors, deltas


def test_fixed_historical_probes_are_causal_endpoints_with_tail_coverage() -> None:
    assert fixed_historical_query_positions(16, 4).tolist() == [3, 7, 11, 15]
    assert fixed_historical_query_positions(16, 4, layout="recent_tail").tolist() == [
        12,
        13,
        14,
        15,
    ]


def test_positive_region_prefix_moments_reconstruct_native_elu_response() -> None:
    torch.manual_seed(901)
    q = torch.rand(1, 2, 12, 4)
    k = torch.rand(1, 2, 12, 4)
    v = torch.randn(1, 2, 12, 4)
    region = historical_probe_region(q, k, scale=0.5, probe_count=4)
    assert region.positive_mask.all()
    moments = build_prefix_affine_moments(k, v, region.positive_mask)
    observed = read_prefix_affine_moments(q, moments, scale=0.5)

    model = _model(907, length=12)
    attention = model.blocks[0].attn
    attention.scale = 0.5
    exact = exact_history_response_heads(attention, q, k, v)
    assert torch.allclose(observed, exact, atol=2e-5, rtol=2e-5)

    stats = activation_region_statistics(q, k, region, scale=0.5)
    assert torch.equal(stats["shared_region_pair_agreement"], torch.ones(2))
    assert torch.equal(stats["unanimous_key_fraction"], torch.ones(2))


def test_affine_segment_summary_combination_is_associative_and_complete() -> None:
    torch.manual_seed(911)
    k = torch.randn(1, 2, 10, 4)
    v = torch.randn(1, 2, 10, 4)
    mask = torch.rand(1, 2, 10) > 0.4
    left = summarize_affine_segment(k[:, :, :4], v[:, :, :4], mask[:, :, :4])
    middle = summarize_affine_segment(k[:, :, 4:7], v[:, :, 4:7], mask[:, :, 4:7])
    right = summarize_affine_segment(k[:, :, 7:], v[:, :, 7:], mask[:, :, 7:])
    full = summarize_affine_segment(k, v, mask)
    left_first = combine_affine_summaries(combine_affine_summaries(left, middle), right)
    right_first = combine_affine_summaries(left, combine_affine_summaries(middle, right))
    assert torch.allclose(left_first.base, full.base)
    assert torch.allclose(left_first.linear, full.linear)
    assert torch.allclose(left_first.base, right_first.base)
    assert torch.allclose(left_first.linear, right_first.linear)


def test_exact_trace_teacher_forcing_and_closed_rollout_are_finite() -> None:
    current = _model(919)
    parent = _model(929)
    items, behaviors, deltas = _history()
    current_trace = trace_exact_history_layers(current, items, behaviors, deltas)
    parent_trace = trace_exact_history_layers(parent, items, behaviors, deltas)
    teacher = paired_teacher_forced_diagnostics(current, current_trace, parent_trace, probe_count=4)
    rollout = rollout_dense_prefix_moments(current, items, behaviors, deltas, probe_count=4)
    closed = exact_to_rollout_layer_diagnostics(current_trace, rollout)
    assert len(teacher) == 3 * 2
    assert len(closed) == 3 * 2
    assert all(
        torch.isfinite(torch.tensor(value))
        for row in teacher + closed
        for value in row.values()
        if isinstance(value, float)
    )
    assert rollout.final_hidden.shape == (1, 16, 16)
    combined = diagnose_model_pair(parent, current, items, behaviors, deltas, probe_count=4)
    assert len(combined.layer_head_records) == 3 * 2
    assert combined.cost["support_semantics"] == ("all_history_dense_reduction_no_token_subset")


def test_identical_versions_have_exact_zero_paired_delta() -> None:
    model = _model(937)
    trace = trace_exact_history_layers(model, *_history())
    rows = paired_teacher_forced_diagnostics(model, trace, trace, probe_count=4)
    assert all(row["teacher_forced_paired_response_recovery"] == 1.0 for row in rows)


def test_medium_dense_prefix_cost_exposes_transform_floor_and_full_cost() -> None:
    p8 = dense_prefix_moment_cost(
        layers=6,
        hidden=192,
        heads=6,
        context=1024,
        probes=8,
    )
    p32 = dense_prefix_moment_cost(
        layers=6,
        hidden=192,
        heads=6,
        context=1024,
        probes=32,
    )
    assert p8["dense_transform_floor_over_exact"] > 0.20
    assert p8["total_over_exact"] > p8["dense_transform_floor_over_exact"]
    assert p32["total_over_exact"] > p8["total_over_exact"]
    assert p8["persistent_delta_moment_scalars"] == 38_016
    assert p8["persistent_delta_moment_ratio_to_full_Current_KV"] == 0.01611328125
    assert p8["support_semantics"] == "all_history_dense_reduction_no_token_subset"
