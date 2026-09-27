"""Focused arithmetic/reference checks; no trained models or datasets."""

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from selective_recompute_2026_09.cost import CostModel, normalized_point
from hstu_kvcache.models import HSTU, HSTUConfig


def test_full_cache_matrix_cost_matches_real_operator_profile(monkeypatch):
    """Count CPU mm/bmm FLOPs independently of the hand-written ledger."""
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    d, layers, heads, n, batch = 64, 2, 2, 9, 2
    model = HSTU(HSTUConfig(num_items=30, num_behaviors=4, hidden_size=d,
                           num_layers=layers, num_heads=heads, input_dropout=0)).eval()
    items = torch.ones((batch, n), dtype=torch.long)
    times = torch.arange(n, dtype=torch.float32).expand(batch, -1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profile:
        model.compute_kv(items, items, times)
    actual = sum(event.flops for event in profile.key_averages()
                 if event.key in ("aten::mm", "aten::bmm", "aten::addmm"))
    cost = CostModel(d, layers, heads, attention_backend="torch")
    rows, pairs = batch * n, batch * n * n
    scalar = (48 + rows * (48 + 2 * d)
              + layers * (cost.rms_norm(rows) + 5 * rows * d + 4 * heads * pairs)
              + cost.rms_norm(rows))
    assert cost.full_cache(n, batch) - scalar == actual


@pytest.mark.parametrize("scale", ("medium", "large", "max"))
def test_full_tail_has_full_execution_cost_and_dense_sparse_counts_masked_pairs(scale):
    cost = CostModel.for_scale(scale)
    assert cost.tail_rebuild(1024, 1024) == cost.full_cache(1024)
    assert cost.sparse_rebuild(1024, 1024) > cost.full_cache(1024)
    # Triton visits32*64 tiles. These hand-checkable boundary cases must not
    # silently use the useful triangle or drop padded lanes.
    assert cost.attention_pairs(1, 33, 32) == 32 * 64
    assert cost.attention_pairs(65, 65) == (1 + 1 + 2) * 32 * 64


def test_normalization_charges_rolling_appends_and_preserves_unfavorable_points():
    cost = CostModel.for_scale("max")
    denominator = cost.workload_denominator({100: 3}, {99: 2}, append_new_kv_only=False)
    assert denominator["full_minus_reuse_flops"] == 3 * cost.full_cache(100) - 2 * cost.append(99, new_kv_only=False)
    point = normalized_point(full_auc=.7, reuse_auc=.6, baseline_auc=.55,
                             extra_flops=12, full_minus_reuse_flops=10)
    assert point["recovery_percent"] == pytest.approx(-50)
    assert point["relative_flops_percent"] == 120


def test_profile_and_selector_costs_cannot_disappear_from_a_budget():
    cost = CostModel.for_scale("max")
    profile = cost.layer_profile([(100, 4)])
    assert profile["profile_intervals"] == 136
    assert profile["components"]["profile_parent_cache"] == cost.full_cache(100)
    assert profile["components"]["profile_teacher_cache"] == cost.full_cache(100)
    assert profile["calibration_flops"] > profile["components"]["profile_interval_rebuilds"]
    for method in ("deviation", "query"):
        one = cost.operation_cost(method, 100, selected_tokens=1)
        many = cost.operation_cost(method, 100, selected_tokens=50)
        assert many["selection_flops"] == one["selection_flops"] > 0
        assert many["recompute_flops"] > one["recompute_flops"] > 0
        assert cost.operation_cost(method, 100, selected_tokens=0)["total_flops"] == 0
        full = cost.operation_cost(method, 100, selected_tokens=100)
        assert full["selection_flops"] == 0
        assert full["recompute_flops"] == cost.full_cache(100)


def test_torch_short_cohort_work_replaces_native_counts_even_after_growing_to_cap():
    native = CostModel.for_scale("medium", "triton")
    eager = CostModel.for_scale("medium", "torch")
    actual = native.workload_denominator(
        {61: 2, 1024: 3}, {60: 1, 1023: 4}, append_new_kv_only=False,
        torch_full_history_hist={"61": 2, "1024": 1},
        torch_append_prefix_hist={"60": 1, "1023": 1},
    )
    full = 2 * eager.full_cache(61) + eager.full_cache(1024) + 2 * native.full_cache(1024)
    reuse = (eager.append(60, new_kv_only=False) + eager.append(1023, new_kv_only=False)
             + 3 * native.append(1023, new_kv_only=False))
    assert actual == {"full_history_flops": full, "reuse_append_flops": reuse,
                      "full_minus_reuse_flops": full - reuse}
    with pytest.raises(ValueError, match="subsets"):
        native.workload_denominator({61: 2}, {}, torch_full_history_hist={61: 3})


def test_band_append_counts_executed_masked_tiles_and_mixed_backends_once():
    native = CostModel.for_scale("medium", "triton")
    eager = CostModel.for_scale("medium", "torch")
    # A one-row band keeps1024 old keys plus the new key; the first old key is
    # masked, but a whole extra64-key tile is executed. No final norm is run.
    scalar = native.append(1023, new_kv_only=False)
    extra_tile = native.num_layers * (4 * native.hidden_size + 4 * native.num_heads) * (32 * 64)
    assert native.band_append(1024, 1) == scalar + extra_tile - native.rms_norm(1)
    assert native.attention_pairs(32, 1056, 1024, window_size=1024) == 32 * 1088
    assert native.band_append(1024, 32) < 32 * scalar
    assert eager.band_append(61, 8) == (eager.embedding(8)
                                      + eager.num_layers * eager.block(8, 8 * 69))
    actual = native.workload_denominator(
        {1024: 4}, {1023: 1}, append_new_kv_only=False,
        band_append_hist={"1024:32": 3, "61:8": 2},
        torch_band_append_hist={"1024:32": 1, "61:8": 2},
    )
    reuse = (scalar + 2 * native.band_append(1024, 32) + eager.band_append(1024, 32)
             + 2 * eager.band_append(61, 8))
    assert actual["reuse_append_flops"] == reuse
    assert actual["full_minus_reuse_flops"] == 4 * native.full_cache(1024) - reuse


def test_deviation_cost_includes_the_fixed_row_magnitude_floor():
    cost = CostModel.for_scale("medium")
    d, rows = cost.hidden_size, 7
    # Per K or V: difference+square+sum+mean =3D; magnitude square+sum+mean=2D.
    difference = rows * (2 * 3 * d + 1)
    floor = rows * (2 * 2 * d + 2)
    projections_and_input = cost.embedding(rows) + cost.rms_norm(rows) + 4 * rows * d * d
    assert cost.deviation_selector(rows) == projections_and_input + difference + floor
