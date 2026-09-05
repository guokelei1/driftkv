from __future__ import annotations

import pytest
import torch
from insight_two.coupling_depth_replay import (
    coupling_depth_layer0_basis,
    factorized_current_rank_handoff_replay,
    factorized_parent_prefix_replay,
    medium_coupling_depth_cost,
    medium_rank_handoff_cost,
    splice_with_coupling_depth,
)
from insight_two.mode_space_replay import factorized_reduced_current_replay

from hstu_kvcache.models import HSTU, HSTUConfig


def _model(*, layers: int = 4) -> HSTU:
    return HSTU(
        HSTUConfig(
            num_items=37,
            num_behaviors=4,
            hidden_size=8,
            num_layers=layers,
            num_heads=2,
            head_dim=4,
            max_seq_len=12,
            temporal_num_freqs=2,
            input_dropout=0.0,
            block_variant="legacy",
        )
    ).eval()


def _history(length: int = 7) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    items = torch.randint(0, 37, (1, length))
    behaviors = torch.randint(0, 4, (1, length))
    deltas = torch.arange(length, dtype=torch.float32)[None]
    return items, behaviors, deltas


def test_parent_prefix_forms_only_requested_layers_and_matches_full_replay() -> None:
    torch.manual_seed(51)
    parent = _model()
    items, behaviors, deltas = _history()
    embedded = parent.embed_inputs(items, behaviors, deltas)
    prefix = factorized_parent_prefix_replay(
        parent,
        embedded,
        formation_depth=3,
        rank=3,
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=71,
    )
    full = factorized_reduced_current_replay(
        parent,
        embedded,
        rank=3,
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=71,
    )
    assert prefix.formation_depth == 3
    assert len(prefix.layers) == 3
    assert len(prefix.block_input_factors) == 3
    for layer in range(3):
        prefix_key, prefix_value = prefix.layers[layer].materialize()
        full_key, full_value = full.layers[layer].materialize()
        assert torch.allclose(prefix_key, full_key, atol=6e-5, rtol=6e-5)
        assert torch.allclose(prefix_value, full_value, atol=6e-5, rtol=6e-5)


def test_depth_splice_switches_from_matched_to_exact_parent_subtraction() -> None:
    torch.manual_seed(52)
    parent_model = _model()
    current_model = _model()
    items, behaviors, deltas = _history()
    exact_parent = parent_model.compute_kv(items, behaviors, deltas)
    current = factorized_reduced_current_replay(
        current_model,
        current_model.embed_inputs(items, behaviors, deltas),
        rank=3,
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=81,
    )
    prefix = factorized_parent_prefix_replay(
        parent_model,
        parent_model.embed_inputs(items, behaviors, deltas),
        formation_depth=2,
        rank=3,
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=81,
    )
    basis = coupling_depth_layer0_basis(
        prefix,
        current,
        rank=3,
        oversample=2,
        power_iterations=0,
        seed=1081,
    )
    splice = splice_with_coupling_depth(exact_parent, current, prefix, basis)
    transpose = basis.transpose(1, 2)
    for layer in range(4):
        current_key, current_value = current.layers[layer].materialize()
        if layer < 2:
            parent_key, parent_value = prefix.layers[layer].materialize()
        else:
            parent_key, parent_value = exact_parent.k[layer], exact_parent.v[layer]
        assert torch.allclose(
            transpose @ (splice.cache.k[layer] - exact_parent.k[layer]),
            transpose @ (current_key - parent_key),
            atol=6e-5,
            rtol=6e-5,
        )
        assert torch.allclose(
            transpose @ (splice.cache.v[layer] - exact_parent.v[layer]),
            transpose @ (current_value - parent_value),
            atol=6e-5,
            rtol=6e-5,
        )


@pytest.mark.parametrize("formation_depth", [1, 2, 4])
def test_full_token_rank_has_current_exact_limit(formation_depth: int) -> None:
    torch.manual_seed(53)
    parent_model = _model()
    current_model = _model()
    items, behaviors, deltas = _history(length=6)
    exact_parent = parent_model.compute_kv(items, behaviors, deltas)
    exact_current = current_model.compute_kv(items, behaviors, deltas)
    current = factorized_reduced_current_replay(
        current_model,
        current_model.embed_inputs(items, behaviors, deltas),
        rank=6,
        compression="exact_svd",
    )
    prefix = factorized_parent_prefix_replay(
        parent_model,
        parent_model.embed_inputs(items, behaviors, deltas),
        formation_depth=formation_depth,
        rank=6,
        compression="exact_svd",
    )
    basis = coupling_depth_layer0_basis(
        prefix,
        current,
        rank=6,
        oversample=0,
        power_iterations=0,
        seed=1091,
    )
    splice = splice_with_coupling_depth(exact_parent, current, prefix, basis)
    assert torch.allclose(splice.cache.k, exact_current.k, atol=3e-4, rtol=3e-4)
    assert torch.allclose(splice.cache.v, exact_current.v, atol=3e-4, rtol=3e-4)


def test_medium_cost_profile_is_monotone_and_reproduces_full_depth_ledger() -> None:
    costs = {depth: medium_coupling_depth_cost(depth) for depth in (1, 3, 5, 6)}
    assert [costs[d]["migration_sufficient_total_flops_per_user"] for d in costs] == [
        654_783_130,
        809_357_126,
        963_931_122,
        1_041_218_120,
    ]
    assert costs[6]["migration_sufficient_over_exact_all"] == pytest.approx(0.21822602688221543)
    assert costs[1]["within_twenty_percent"] is True
    assert costs[3]["within_twenty_percent"] is True
    assert costs[5]["within_twenty_percent"] is False
    assert costs[6]["within_twenty_percent"] is False
    assert all(costs[depth]["sidecar_scalars"] == 26_624 for depth in costs)


def test_rank_handoff_preserves_early_current_arm_and_changes_only_upper_rank() -> None:
    torch.manual_seed(54)
    current_model = _model()
    items, behaviors, deltas = _history()
    embedded = current_model.embed_inputs(items, behaviors, deltas)
    rank4 = factorized_reduced_current_replay(
        current_model,
        embedded,
        rank=3,
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=101,
    )
    handoff = factorized_current_rank_handoff_replay(
        current_model,
        embedded,
        handoff_depth=2,
        early_rank=3,
        upper_rank=5,
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=101,
    )
    assert [factors.rank for factors in handoff.block_input_factors] == [3, 3, 5, 5]
    for layer in range(2):
        assert torch.equal(handoff.cache.k[layer], rank4.cache.k[layer])
        assert torch.equal(handoff.cache.v[layer], rank4.cache.v[layer])


def test_rank_handoff_keeps_u0_and_early_signed_cores_unchanged() -> None:
    torch.manual_seed(55)
    parent_model = _model()
    current_model = _model()
    items, behaviors, deltas = _history()
    exact_parent = parent_model.compute_kv(items, behaviors, deltas)
    parent_prefix = factorized_parent_prefix_replay(
        parent_model,
        parent_model.embed_inputs(items, behaviors, deltas),
        formation_depth=2,
        rank=3,
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=111,
    )
    ordinary = factorized_reduced_current_replay(
        current_model,
        current_model.embed_inputs(items, behaviors, deltas),
        rank=3,
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=111,
    )
    handoff = factorized_current_rank_handoff_replay(
        current_model,
        current_model.embed_inputs(items, behaviors, deltas),
        handoff_depth=2,
        early_rank=3,
        upper_rank=5,
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=111,
    )
    ordinary_basis = coupling_depth_layer0_basis(
        parent_prefix,
        ordinary,
        rank=3,
        oversample=2,
        power_iterations=0,
        seed=1111,
    )
    handoff_basis = coupling_depth_layer0_basis(
        parent_prefix,
        handoff,
        rank=3,
        oversample=2,
        power_iterations=0,
        seed=1111,
    )
    assert torch.equal(ordinary_basis, handoff_basis)
    ordinary_splice = splice_with_coupling_depth(
        exact_parent, ordinary, parent_prefix, ordinary_basis
    )
    handoff_splice = splice_with_coupling_depth(exact_parent, handoff, parent_prefix, handoff_basis)
    for layer in range(2):
        assert torch.equal(
            ordinary_splice.delta_k_cores[layer],
            handoff_splice.delta_k_cores[layer],
        )
        assert torch.equal(
            ordinary_splice.delta_v_cores[layer],
            handoff_splice.delta_v_cores[layer],
        )


def test_medium_rank_handoff_cost_charges_boundary_and_slightly_exceeds_cap() -> None:
    cost = medium_rank_handoff_cost()
    assert cost["current_handoff_arm_flops_per_user"] == 669_327_938
    assert cost["migration_sufficient_total_flops_per_user"] == 959_428_484
    assert cost["migration_sufficient_over_exact_all"] == pytest.approx(0.20108396321507274)
    assert cost["within_twenty_percent"] is False
    assert cost["single_arm_rank8_kv_terminal_flops_per_user"] == 934_810_304
    assert cost["single_arm_rank8_kv_terminal_over_exact_all"] < 0.20
    assert cost["matrix_free_initial_factor_flops_per_arm"] == 18_033_494
    assert cost["dense_input_plus_initial_factor_flops_per_arm"] == 101_441_366
    assert cost["matrix_free_two_arm_saving_flops_per_user"] == 166_815_744
    assert cost["matrix_free_combined_total_flops_per_user"] == 792_612_740
    assert cost["matrix_free_combined_over_exact_all"] == pytest.approx(0.16612151266290529)
    assert cost["matrix_free_combined_within_twenty_percent"] is True


def test_depth_validation_rejects_empty_or_out_of_model_prefix() -> None:
    with pytest.raises(ValueError, match=r"\[1,6\]"):
        medium_coupling_depth_cost(0)
    with pytest.raises(ValueError, match=r"\[1,6\]"):
        medium_coupling_depth_cost(7)
