from __future__ import annotations

import sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from hstu_kvcache.models import HSTU, HSTUConfig
from insight.pro_lazy_reader import build_parent_conditioned_carriers
from insight.reader_compatibility_correction import (
    STAGES,
    intervene_reader_correction,
    trace_reader_correction,
)
from insight_two.common import (
    ANCHOR_INDICES,
    HELDOUT_INDICES,
    STAGE_PRESENTATION,
    score_metrics,
)
from insight_two.low_rank_correction import (
    fit_predict_low_rank,
    low_rank_final_representation,
    low_rank_layered_correction,
)
from insight_two.functional_probe_estimator import (
    PROBE_COUNTS,
    PROBE_HISTORY_OFFSETS,
    estimate_functional_probe_means,
    fixed_history_probe_items,
    functional_probe_cost,
    medium_cost_grid,
)
from insight.one_release_refinement import parameter_cast_maps
from insight_two.time_aligned_functional_probe import generate_time_aligned_probe
from insight_two.release_functional_basis import (
    fit_oracle_release_basis,
    rank_at_energy,
)
from insight_two.temporal_persistence import (
    append_bucket,
    correction_drift,
    correction_sha256,
    remaining_parent_fraction,
    scale_correction,
    time_bucket,
)
from insight_two.tail_functional_estimator import (
    PROBE_COUNTS as TAIL_PROBE_COUNTS,
    estimate_tail_functional_sidecars,
    medium_tail_functional_costs,
)
from insight_two.temporal_coefficient import (
    project_global_coefficient,
    project_layerwise_coefficients,
)
from insight.pro_lazy_reader import generate_lazy_pro_probe_components


def _model(seed: int) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=96,
            num_behaviors=3,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            max_seq_len=16,
            temporal_num_freqs=2,
            input_dropout=0.0,
        )
    ).eval()


def test_frozen_candidate_split_is_disjoint_and_complete() -> None:
    assert len(ANCHOR_INDICES) == len(HELDOUT_INDICES) == 32
    assert set(ANCHOR_INDICES).isdisjoint(HELDOUT_INDICES)
    assert sorted(ANCHOR_INDICES + HELDOUT_INDICES) == list(range(64))
    assert set(STAGE_PRESENTATION) == set(STAGES)


def test_score_metrics_has_exact_and_reuse_endpoints() -> None:
    exact = torch.tensor([[2.0, -1.0, 0.5]])
    reuse = torch.tensor([[0.0, 1.0, -0.5]])
    exact_metrics = score_metrics(exact, reuse, exact)
    reuse_metrics = score_metrics(exact, reuse, reuse)
    assert torch.allclose(exact_metrics["probability_gap_recovery"], torch.ones(1))
    assert torch.allclose(exact_metrics["logit_gap_recovery"], torch.ones(1))
    assert torch.allclose(reuse_metrics["probability_gap_recovery"], torch.zeros(1))
    assert torch.allclose(reuse_metrics["logit_gap_recovery"], torch.zeros(1))


def test_anchor_correction_replays_on_heldout_without_mutating_cache() -> None:
    parent, current = _model(7), _model(11)
    items = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    behaviors = torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2]])
    deltas = torch.arange(8).float().unsqueeze(0)
    exact_cache = current.compute_kv(items, behaviors, deltas)
    reuse_cache = parent.compute_kv(items, behaviors, deltas)
    original_k = reuse_cache.k.clone()
    original_v = reuse_cache.v.clone()
    anchors = torch.tensor([[17, 19, 21, 23]])
    heldout = torch.tensor([[18, 20, 22, 24]])
    query_delta = torch.tensor([10.0])
    trace = trace_reader_correction(
        current, exact_cache, reuse_cache, anchors, query_delta
    )
    exact_scores = current.score_cc_reuse(exact_cache, heldout, query_delta)
    reuse_scores = current.score_cc_reuse(reuse_cache, heldout, query_delta)
    for stage in STAGES:
        scores, _ = intervene_reader_correction(
            current,
            reuse_cache,
            heldout,
            query_delta,
            stage=stage,
            corrections=trace.corrections[stage],
        )
        metrics = score_metrics(exact_scores, reuse_scores, scores)
        assert all(torch.isfinite(value).all() for value in metrics.values())
    assert torch.equal(reuse_cache.k, original_k)
    assert torch.equal(reuse_cache.v, original_v)


def test_rank_zero_reduced_rank_fit_is_anchor_mean() -> None:
    features = torch.randn(2, 4, 6)
    targets = torch.randn(2, 4, 8)
    heldout = torch.randn(2, 3, 6)
    anchor_prediction, heldout_prediction, diagnostics, storage = fit_predict_low_rank(
        features, targets, heldout, rank=0
    )
    expected = targets.mean(dim=1, keepdim=True)
    assert torch.allclose(anchor_prediction, expected.expand_as(targets))
    assert torch.allclose(heldout_prediction, expected.expand(2, 3, 8))
    assert len(diagnostics) == 2
    assert storage == 8


def test_rank_zero_new_adapter_matches_frozen_stage_intervention() -> None:
    parent, current = _model(29), _model(31)
    items = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    behaviors = torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2]])
    deltas = torch.arange(8).float().unsqueeze(0)
    exact_cache = current.compute_kv(items, behaviors, deltas)
    reuse_cache = parent.compute_kv(items, behaviors, deltas)
    anchors = torch.tensor([[17, 19, 21, 23]])
    heldout = torch.tensor([[18, 20, 22, 24]])
    query_delta = torch.tensor([10.0])
    trace = trace_reader_correction(current, exact_cache, reuse_cache, anchors, query_delta)
    for stage in ("av_aggregation", "u_gated_update", "layer_hidden"):
        expected, _ = intervene_reader_correction(
            current,
            reuse_cache,
            heldout,
            query_delta,
            stage=stage,
            corrections=trace.corrections[stage],
        )
        observed = low_rank_layered_correction(
            current,
            exact_cache,
            reuse_cache,
            anchors,
            heldout,
            query_delta,
            stage=stage,
            rank=0,
        )
        assert torch.allclose(observed.scores, expected, atol=1e-6)
    expected, _ = intervene_reader_correction(
        current,
        reuse_cache,
        heldout,
        query_delta,
        stage="final_readout",
        corrections=trace.corrections["final_readout"],
    )
    observed = low_rank_final_representation(
        current,
        exact_cache,
        reuse_cache,
        anchors,
        heldout,
        query_delta,
        rank=0,
    )
    assert torch.allclose(observed.scores, expected, atol=1e-6)


def test_fixed_functional_probes_are_history_only_and_nested() -> None:
    items = torch.arange(1, 257).reshape(2, 128)
    probes = fixed_history_probe_items(items)
    assert probes.shape == (2, 8)
    assert probes[0].tolist() == [
        int(items[0, 128 + offset]) for offset in PROBE_HISTORY_OFFSETS
    ]
    assert PROBE_COUNTS == (1, 2, 4, 8)


def test_eight_probe_estimator_contains_the_one_probe_result() -> None:
    parent, current = _model(43), _model(47)
    items = torch.arange(1, 17).reshape(1, 16)
    behaviors = (torch.arange(16).reshape(1, 16) % 2 + 1).long()
    deltas = torch.arange(16).float().reshape(1, 16)
    parent_cache = parent.compute_kv(items, behaviors, deltas)
    carriers, layout = build_parent_conditioned_carriers(
        parent_cache=parent_cache,
        current=current,
        item_ids=items,
        behaviors=behaviors,
        time_deltas=deltas,
        repair_width=8,
        carrier_count=2,
    )
    # Repeat the synthetic 16-position history to meet the production probe
    # selector's 128-event input contract without changing probe item values.
    probe_history = items.repeat(1, 8)
    estimate = estimate_functional_probe_means(
        current,
        parent_cache,
        carriers,
        parameter_cast_maps(parent, current),
        fixed_history_probe_items(probe_history),
        old_positions=layout.old_positions,
    )
    assert set(estimate.corrections_by_probe_count) == set(PROBE_COUNTS)
    for layer, individual in enumerate(estimate.individual_probe_corrections):
        assert torch.allclose(
            estimate.corrections_by_probe_count[1][layer], individual[:, 7]
        )
    assert estimate.replay_max_abs_error < 1e-5


def test_medium_probe_cost_grid_is_ordered_and_below_twenty_percent() -> None:
    grid = medium_cost_grid()
    assert len(grid) == 16
    assert all(row["materialized_version_translated_prefix_scalars"] == 0 for row in grid)
    assert all(float(row["over_full_fraction"]) <= 0.20 for row in grid)
    assert min(float(row["over_full_fraction"]) for row in grid) > 0.02
    assert max(float(row["over_full_fraction"]) for row in grid) < 0.19
    fixed_carriers = [
        functional_probe_cost(
            layers=6,
            hidden=192,
            heads=6,
            context=1024,
            repair_evidence=128,
            carriers=32,
            probes=probes,
        )["over_full_fraction"]
        for probes in PROBE_COUNTS
    ]
    assert fixed_carriers == sorted(fixed_carriers)


def test_time_aligned_probe_matches_legacy_at_zero_and_replays_nonzero() -> None:
    parent, current = _model(61), _model(67)
    items = torch.arange(1, 17).reshape(1, 16)
    behaviors = (torch.arange(16).reshape(1, 16) % 2 + 1).long()
    deltas = torch.arange(16).float().reshape(1, 16)
    parent_cache = parent.compute_kv(items, behaviors, deltas)
    carriers, layout = build_parent_conditioned_carriers(
        parent_cache=parent_cache,
        current=current,
        item_ids=items,
        behaviors=behaviors,
        time_deltas=deltas,
        repair_width=8,
        carrier_count=2,
    )
    maps = parameter_cast_maps(parent, current)
    probe = items[:, -1]
    legacy = generate_lazy_pro_probe_components(
        current,
        parent_cache,
        carriers,
        maps,
        probe,
        old_positions=layout.old_positions,
    )
    aligned_zero = generate_time_aligned_probe(
        current,
        parent_cache,
        carriers,
        maps,
        probe,
        torch.zeros(1),
        old_positions=layout.old_positions,
    )
    for actual, expected in zip(
        aligned_zero.corrections, legacy.corrections, strict=True
    ):
        assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)
    aligned_nonzero = generate_time_aligned_probe(
        current,
        parent_cache,
        carriers,
        maps,
        probe,
        torch.tensor([19.0]),
        old_positions=layout.old_positions,
    )
    assert aligned_nonzero.replay_max_abs_error < 1e-5


def test_release_basis_rank_zero_is_mean_and_full_rank_recovers() -> None:
    rng = np.random.default_rng(71)
    targets = rng.normal(size=(8, 3, 6)).astype(np.float32)
    fitted = fit_oracle_release_basis(targets)
    rank_zero = fitted.project(targets[4:], rank=0)
    assert np.allclose(rank_zero, fitted.mean[None])
    full = fitted.project(targets, rank=8)
    assert np.allclose(full, targets, atol=1e-10)
    assert all(rank_at_energy(values, 0.90) <= 7 for values in fitted.layer_singular_values)


def test_temporal_persistence_buckets_and_coverage_are_frozen() -> None:
    assert time_bucket(0) == "[0d,1d)"
    assert time_bucket(86_400) == "[1d,3d)"
    assert time_bucket(7 * 86_400) == "[7d,14d)"
    assert append_bucket(0) == "0"
    assert append_bucket(8) == "[1,8]"
    assert append_bucket(513) == ">512"
    assert remaining_parent_fraction(0, 1024) == 1.0
    assert remaining_parent_fraction(128, 1024) == 0.875
    assert remaining_parent_fraction(2048, 1024) == 0.0


def test_temporal_correction_drift_and_hash() -> None:
    frozen = (torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.0, 1.0]]]))
    current = scale_correction(frozen, 2.0)
    metrics = correction_drift(current, frozen)
    assert torch.allclose(metrics["direction_cosine"], torch.ones(1))
    assert torch.allclose(
        metrics["current_to_frozen_norm_ratio"], torch.full((1,), 2.0)
    )
    original_hash = correction_sha256(frozen)
    assert original_hash == correction_sha256(tuple(value.clone() for value in frozen))
    assert original_hash != correction_sha256(current)


def test_tail_functional_estimator_is_dependency_closed_and_under_budget() -> None:
    parent, current = _model(73), _model(79)
    items = torch.arange(1, 17).reshape(1, 16)
    behaviors = (torch.arange(16).reshape(1, 16) % 2 + 1).long()
    deltas = torch.arange(16).float().reshape(1, 16)
    parent_cache = parent.compute_kv(items, behaviors, deltas)
    original_k = parent_cache.k.clone()
    original_v = parent_cache.v.clone()
    probe_items = items[:, [0, 2, 4, 6, 8, 10, 12, 15]]
    estimate = estimate_tail_functional_sidecars(
        current,
        parent_cache,
        items,
        behaviors,
        deltas,
        probe_items,
        torch.tensor([17.0]),
        tail_width=4,
    )
    assert set(estimate.corrections_by_probe_count) == set(TAIL_PROBE_COUNTS)
    assert estimate.single_probe_replay_max_abs_error < 2e-5
    assert estimate.parent_prefix_max_abs_change == 0.0
    assert torch.equal(parent_cache.k, original_k)
    assert torch.equal(parent_cache.v, original_v)
    costs = medium_tail_functional_costs()
    assert [row["probes"] for row in costs] == [1, 2, 4]
    assert all(float(row["over_full_fraction"]) < 0.20 for row in costs)
    assert [row["over_full_fraction"] for row in costs] == sorted(
        row["over_full_fraction"] for row in costs
    )


def test_temporal_coefficient_projection_separates_global_and_layerwise_scale() -> None:
    frozen = (
        torch.tensor([[[1.0, 2.0]]]),
        torch.tensor([[[3.0, 4.0]]]),
    )
    current = (2.0 * frozen[0], 0.5 * frozen[1])
    global_projection = project_global_coefficient(current, frozen)
    layer_projection = project_layerwise_coefficients(current, frozen)
    assert global_projection.coefficients.shape == (1, 1)
    assert global_projection.relative_l2.item() > 0
    assert torch.allclose(
        layer_projection.coefficients, torch.tensor([[2.0, 0.5]])
    )
    assert torch.allclose(layer_projection.relative_l2, torch.zeros(1), atol=1e-7)
    assert all(
        torch.allclose(actual, expected)
        for actual, expected in zip(layer_projection.correction, current, strict=True)
    )
