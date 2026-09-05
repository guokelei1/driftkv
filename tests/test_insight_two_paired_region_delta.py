from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from insight_two.cone_response_memory import build_cone_response_memory
from insight_two.paired_region_delta import (
    build_paired_region_delta_memory,
    causal_delta_closure_cost,
    certify_nested_moment_disagreement,
    exact_cache_samples,
    fixed_history_probe_positions,
    paired_region_delta_cost,
    project_full_current_layer0,
    replay_parent_conditioned_current_carriers,
    select_legal_layer0_address_landmarks,
    trace_history_item_region_queries,
)


def _model(seed: int, *, layers: int = 2, length: int = 32) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=256,
            num_behaviors=3,
            hidden_size=16,
            num_layers=layers,
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


def _history(length: int = 32) -> tuple[torch.Tensor, ...]:
    items = (torch.arange(length).reshape(1, length) + 1).long()
    behaviors = (torch.arange(length).reshape(1, length) % 2 + 1).long()
    deltas = torch.arange(length).reshape(1, length).float()
    query = torch.tensor([float(length + 3)])
    return items, behaviors, deltas, query


def _pair(length: int = 32) -> tuple[HSTU, HSTU, HSTUKVCache, HSTUKVCache]:
    parent, current = _model(701, length=length), _model(703, length=length)
    items, behaviors, deltas, _ = _history(length)
    return (
        parent,
        current,
        current.compute_kv(items, behaviors, deltas),
        parent.compute_kv(items, behaviors, deltas),
    )


def test_fixed_history_probe_positions_are_equal_width_lower_midpoints() -> None:
    assert fixed_history_probe_positions(32, 8).tolist() == [1, 5, 9, 13, 17, 21, 25, 29]
    assert fixed_history_probe_positions(32, 32).tolist() == list(range(32))
    with pytest.raises(ValueError):
        fixed_history_probe_positions(31, 8)


def test_full_paired_positions_equal_existing_full_signed_moments() -> None:
    parent, current, exact, reuse = _pair()
    items, _, _, query = _history()
    probes = items
    traced = trace_history_item_region_queries(
        current, reuse, items, query, probe_count=32
    )
    positions = torch.arange(32)
    paired = build_paired_region_delta_memory(
        current,
        exact_cache_samples(exact, positions),
        reuse,
        traced,
        positions,
        torch.ones(32),
    )
    existing = build_cone_response_memory(
        current, exact, reuse, probes, query
    )
    for actual, expected in zip(paired.layers, existing.layers, strict=True):
        assert torch.allclose(actual.base, expected.base)
        assert torch.allclose(actual.linear, expected.linear)


def test_identical_versions_have_zero_paired_delta() -> None:
    model = _model(709)
    items, behaviors, deltas, query = _history()
    cache = model.compute_kv(items, behaviors, deltas)
    traced = trace_history_item_region_queries(
        model, cache, items, query, probe_count=8
    )
    positions = torch.arange(0, 32, 4)
    samples = exact_cache_samples(cache, positions)
    memory = build_paired_region_delta_memory(
        model, samples, cache, traced, positions, torch.ones(8)
    )
    for layer in memory.layers:
        assert torch.count_nonzero(layer.base) == 0
        assert torch.count_nonzero(layer.linear) == 0


def test_legal_layer0_projection_matches_exact_cache_without_compute_kv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, current, exact, reuse = _pair()
    items, behaviors, deltas, _ = _history()

    def forbidden(*args, **kwargs):
        raise AssertionError("legal projection must not call compute_kv")

    monkeypatch.setattr(current, "compute_kv", forbidden)
    projected = project_full_current_layer0(
        current, reuse, items, behaviors, deltas
    )
    assert torch.allclose(projected.k, exact.k[0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(projected.v, exact.v[0], atol=1e-6, rtol=1e-6)


def test_parent_conditioned_carrier_layer0_is_exact_and_suffix_causal() -> None:
    _, current, exact, reuse = _pair()
    items, behaviors, deltas, _ = _history()
    position = torch.tensor([11])
    original = replay_parent_conditioned_current_carriers(
        current, reuse, items, behaviors, deltas, position
    )
    assert torch.allclose(original.k[0, 0, 0], exact.k[0, 0, 11], atol=1e-6)
    assert torch.allclose(original.v[0, 0, 0], exact.v[0, 0, 11], atol=1e-6)

    changed = HSTUKVCache(k=reuse.k.clone(), v=reuse.v.clone(), seq_len=reuse.seq_len)
    changed.k[:, :, 11:].add_(123.0)
    changed.v[:, :, 11:].sub_(91.0)
    causal = replay_parent_conditioned_current_carriers(
        current, changed, items, behaviors, deltas, position
    )
    assert torch.equal(original.k, causal.k)
    assert torch.equal(original.v, causal.v)


def test_legal_selector_and_replay_preserve_parent_cache() -> None:
    _, current, _, reuse = _pair(length=128)
    items, behaviors, deltas, query = _history(length=128)
    before = (reuse.k.clone(), reuse.v.clone())
    projected = project_full_current_layer0(
        current, reuse, items, behaviors, deltas
    )
    selection = select_legal_layer0_address_landmarks(
        projected, reuse, sample_count=64
    )
    carriers = replay_parent_conditioned_current_carriers(
        current,
        reuse,
        items,
        behaviors,
        deltas,
        selection.selected_positions,
    )
    probes = trace_history_item_region_queries(
        current, reuse, items, query, probe_count=8
    )
    memory = build_paired_region_delta_memory(
        current,
        carriers,
        reuse,
        probes,
        selection.selected_positions,
        selection.cluster_masses.float(),
    )
    assert carriers.seq_len == 64
    assert memory.anchor_count == 8
    assert memory.source_length == 128
    assert torch.equal(reuse.k, before[0])
    assert torch.equal(reuse.v, before[1])


def test_nested_moment_certificate_uses_no_exact_target() -> None:
    parent, current, exact, reuse = _pair(length=128)
    del parent
    items, behaviors, deltas, query = _history(length=128)
    probes = trace_history_item_region_queries(
        current, reuse, items, query, probe_count=8
    )
    projected = project_full_current_layer0(
        current, reuse, items, behaviors, deltas
    )
    coarse_selection = select_legal_layer0_address_landmarks(
        projected, reuse, sample_count=64
    )
    fine_selection = select_legal_layer0_address_landmarks(
        projected, reuse, sample_count=128
    )
    coarse = build_paired_region_delta_memory(
        current,
        exact_cache_samples(exact, coarse_selection.selected_positions),
        reuse,
        probes,
        coarse_selection.selected_positions,
        coarse_selection.cluster_masses.float(),
    )
    fine = build_paired_region_delta_memory(
        current,
        exact_cache_samples(exact, fine_selection.selected_positions),
        reuse,
        probes,
        fine_selection.selected_positions,
        fine_selection.cluster_masses.float(),
    )
    certificate = certify_nested_moment_disagreement(
        current, probes, coarse, fine
    )
    assert certificate.relative_l2 >= 0
    assert -1 <= certificate.cosine <= 1
    assert certificate.maximum_absolute_difference >= 0
    assert certificate.reference_l2 > 0


def test_cost_separates_neural_selection_and_total_flops() -> None:
    r64 = paired_region_delta_cost(
        layers=6,
        hidden=192,
        heads=6,
        context=1024,
        carriers=64,
        probes=8,
    )
    r128 = paired_region_delta_cost(
        layers=6,
        hidden=192,
        heads=6,
        context=1024,
        carriers=128,
        probes=8,
    )
    p32 = paired_region_delta_cost(
        layers=6,
        hidden=192,
        heads=6,
        context=1024,
        carriers=64,
        probes=32,
    )
    assert r64["neural_generation_flops_per_user"] == 686_961_792
    assert r64["address_selection_flops_per_user"] == 153_615_360
    assert r64["total_generation_flops_per_user"] == 840_577_152
    assert r64["total_generation_flops_per_user"] == (
        r64["neural_generation_flops_per_user"]
        + r64["address_selection_flops_per_user"]
    )
    assert r64["total_over_full_fraction"] == r64["over_full_fraction"]
    assert r64["neural_over_full_fraction"] == pytest.approx(
        0.143978422588388
    )
    assert r64["selection_over_full_fraction"] == pytest.approx(
        0.03219581856766112
    )
    assert r64["total_over_full_fraction"] == pytest.approx(
        0.17617424115604913
    )

    # The honest total, rather than neural FLOPs alone, defines admission.
    assert r64["total_over_full_fraction"] < 0.20
    assert r64["maximum_total_over_full_fraction"] > 0.20
    assert r64["within_20_percent_for_all_unique_position_sets"] is False
    assert r128["neural_over_full_fraction"] > 0.20
    assert r128["total_over_full_fraction"] > 0.27
    assert r128["optimized_kv_only_total_over_full_fraction"] > 0.25
    assert p32["total_over_full_fraction"] > 0.20
    assert r64["over_full_fraction"] < r128["over_full_fraction"]
    assert r64["over_full_fraction"] < p32["over_full_fraction"]


def test_cost_accounts_full_layer0_kv_and_discarded_q_projection() -> None:
    cost = paired_region_delta_cost(
        layers=6,
        hidden=192,
        heads=6,
        context=1024,
        carriers=64,
        probes=8,
    )
    assert cost["full_layer0_kv_projection_flops_per_user"] == 150_994_944
    assert cost["discarded_full_layer0_q_projection_flops_per_user"] == 75_497_472
    assert cost["full_layer0_projection_flops_per_user"] == (
        cost["full_layer0_input_projection_flops_per_user"]
        + cost["full_layer0_kv_projection_flops_per_user"]
        + cost["discarded_full_layer0_q_projection_flops_per_user"]
        + cost["full_layer0_pointwise_flops_per_user"]
    )
    assert cost["optimized_kv_only_total_flops_per_user"] == (
        cost["total_generation_flops_per_user"]
        - cost["discarded_full_layer0_q_projection_flops_per_user"]
    )


def test_cost_reports_serving_and_optional_carrier_storage_separately() -> None:
    r64 = paired_region_delta_cost(
        layers=6,
        hidden=192,
        heads=6,
        context=1024,
        carriers=64,
        probes=8,
    )
    r128 = paired_region_delta_cost(
        layers=6,
        hidden=192,
        heads=6,
        context=1024,
        carriers=128,
        probes=8,
    )
    assert r64["persistent_moment_scalars"] == 38_016
    assert r64["persistent_moment_bytes_fp32"] == 152_064
    assert r64["persistent_moment_bytes_fp16"] == 76_032
    assert r64["persistent_moment_ratio_to_full_current_kv"] == pytest.approx(
        0.01611328125
    )
    assert r64["current_carrier_kv_scalars_if_retained"] == 147_456
    assert r128["current_carrier_kv_scalars_if_retained"] == 294_912
    assert r64["persistent_moment_plus_carrier_ratio_if_retained"] == pytest.approx(
        0.07861328125
    )
    assert r128["persistent_moment_plus_carrier_ratio_if_retained"] == pytest.approx(
        0.14111328125
    )
    assert r64["transient_full_current_layer0_kv_scalars"] == 393_216
    assert r64["transient_address_feature_scalars_fp32"] == 393_216
    assert r64["transient_voronoi_distance_scalars_fp32"] == 65_536
    assert "not_final_closure" in str(r64["cost_semantics"])


def test_cost_requires_observed_unique_carrier_position_sum_for_a_certificate() -> None:
    estimated = paired_region_delta_cost(
        layers=6,
        hidden=192,
        heads=6,
        context=1024,
        carriers=64,
        probes=8,
    )
    observed = paired_region_delta_cost(
        layers=6,
        hidden=192,
        heads=6,
        context=1024,
        carriers=64,
        probes=8,
        carrier_position_sum=31_000,
    )
    assert estimated["carrier_position_sum"] == 32_736
    assert estimated["carrier_position_sum_is_observed"] is False
    assert observed["carrier_position_sum_is_observed"] is True
    assert observed["total_generation_flops_per_user"] < estimated[
        "total_generation_flops_per_user"
    ]

    with pytest.raises(ValueError, match="unique source positions"):
        paired_region_delta_cost(
            layers=6,
            hidden=192,
            heads=6,
            context=1024,
            carriers=64,
            probes=8,
            carrier_position_sum=1_000,
        )


def test_native_closure_cost_excludes_probe_moment_diagnostic() -> None:
    independent = causal_delta_closure_cost(
        layers=6,
        hidden=192,
        heads=6,
        context=1024,
        carriers=64,
        recursive_delta=False,
    )
    recursive = causal_delta_closure_cost(
        layers=6,
        hidden=192,
        heads=6,
        context=1024,
        carriers=64,
        recursive_delta=True,
    )
    assert independent["neural_generation_flops_per_user"] == 613_982_208
    assert independent["selection_flops_per_user"] == 153_615_360
    assert independent["total_generation_flops_per_user"] == 767_597_568
    assert independent["over_full_fraction"] == pytest.approx(
        0.16087865192846548
    )
    assert independent["recursive_delta_flops_per_user"] == 0
    assert independent["native_pair_materialization_flops_per_user"] == 221_184

    assert recursive["recursive_earlier_pair_count_per_layer"] == 2_016
    assert recursive["recursive_delta_flops_per_user"] == 18_943_488
    assert recursive["total_generation_flops_per_user"] == 786_541_056
    assert recursive["over_full_fraction"] == pytest.approx(
        0.16484896520108785
    )
    assert recursive["within_20_percent_at_reported_position_sum"] is True
    assert recursive["maximum_total_over_full_fraction"] == pytest.approx(
        0.19451766472309215
    )
    assert recursive["within_20_percent_for_all_unique_position_sets"] is True
    assert recursive["selection_flops_per_user"] == independent[
        "selection_flops_per_user"
    ]


def test_native_closure_r128_is_diagnostic_and_storage_is_not_hidden() -> None:
    independent = causal_delta_closure_cost(
        layers=6,
        hidden=192,
        heads=6,
        context=1024,
        carriers=128,
        recursive_delta=False,
    )
    recursive = causal_delta_closure_cost(
        layers=6,
        hidden=192,
        heads=6,
        context=1024,
        carriers=128,
        recursive_delta=True,
    )
    assert independent["over_full_fraction"] == pytest.approx(
        0.2550441510768639
    )
    assert recursive["over_full_fraction"] == pytest.approx(
        0.27102005040931815
    )
    assert independent["within_20_percent_at_reported_position_sum"] is False
    assert recursive["within_20_percent_at_reported_position_sum"] is False
    assert independent["within_20_percent_for_all_unique_position_sets"] is False
    assert recursive["within_20_percent_for_all_unique_position_sets"] is False

    # The serving minimum reuses Parent K/V. The intervention object's Parent
    # copies are exposed separately and cannot masquerade as free persistence.
    assert independent["irreducible_incremental_carrier_scalars"] == 294_912
    assert independent["irreducible_incremental_sidecar_bytes_fp32"] == 1_181_696
    assert independent[
        "irreducible_incremental_sidecar_ratio_to_full_current_kv_fp32"
    ] == pytest.approx(0.1252170138888889)
    assert independent["materialized_intervention_pair_scalars"] == 589_824
    assert independent["materialized_intervention_pair_bytes_fp32"] == 2_361_344
    assert independent[
        "materialized_intervention_ratio_to_full_current_kv_fp32"
    ] == pytest.approx(0.2502170138888889)
    assert independent["transient_full_current_layer0_kv_scalars"] == 393_216
    assert independent["transient_address_feature_scalars_fp32"] == 393_216
    assert independent["transient_voronoi_distance_scalars_fp32"] == 131_072
