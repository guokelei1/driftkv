from __future__ import annotations

import inspect

import torch
from insight_two.all_history_affine_response import (
    build_dense_all_history_moments,
    build_factorized_all_history_moments,
    build_full_exact_all_history_response_memory,
    build_single_arm_all_history_response_memory,
    medium_all_history_affine_cost,
    medium_single_r8_kv_splice_cost,
)
from insight_two.mode_space_replay import (
    FactorizedCacheLayer,
    paired_release_replay,
)

from hstu_kvcache.models import HSTU, HSTUConfig


def _model(seed: int, *, length: int = 16) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=128,
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


def _history(length: int = 16) -> tuple[torch.Tensor, ...]:
    items = (torch.arange(length).reshape(1, length) % 127 + 1).long()
    behaviors = (torch.arange(length).reshape(1, length) % 2 + 1).long()
    deltas = torch.arange(length).reshape(1, length).float()
    return items, behaviors, deltas


def test_factorized_all_history_moments_equal_materialized_dense_state() -> None:
    torch.manual_seed(1301)
    model = _model(1303)
    layer = FactorizedCacheLayer(
        left=torch.randn(1, 16, 5),
        key_core=torch.randn(1, 5, 16),
        value_core=torch.randn(1, 5, 16),
    )
    key, value = layer.materialize()
    observed = build_factorized_all_history_moments(model.blocks[0].attn, layer)
    expected = build_dense_all_history_moments(model.blocks[0].attn, key, value)
    assert torch.allclose(observed.base, expected.base, atol=3e-5, rtol=3e-5)
    assert torch.allclose(observed.linear, expected.linear, atol=1e-4, rtol=1e-4)
    assert bool(observed.positive_mask.all())


def test_full_rank_single_arm_compiler_matches_full_exact_affine_oracle() -> None:
    parent = _model(1311)
    current = _model(1313)
    items, behaviors, deltas = _history()
    exact_parent = parent.compute_kv(items, behaviors, deltas)
    exact_current = current.compute_kv(items, behaviors, deltas)
    paired = paired_release_replay(
        parent,
        current,
        parent.embed_inputs(items, behaviors, deltas),
        current.embed_inputs(items, behaviors, deltas),
        rank=16,
        compression="exact_svd",
    )
    observed = build_single_arm_all_history_response_memory(current, paired.current, exact_parent)
    expected = build_full_exact_all_history_response_memory(current, exact_current, exact_parent)
    for actual_layer, expected_layer in zip(observed.layers, expected.layers, strict=True):
        assert torch.allclose(actual_layer.base, expected_layer.base, atol=2e-4, rtol=2e-4)
        assert torch.allclose(
            actual_layer.linear,
            expected_layer.linear,
            atol=5e-4,
            rtol=5e-4,
        )


def test_legal_compiler_api_has_no_probe_candidate_or_exact_current_input() -> None:
    parameters = inspect.signature(build_single_arm_all_history_response_memory).parameters
    assert set(parameters) == {"model", "current_replay", "exact_parent"}


def test_matrix_free_cost_is_below_twenty_percent_without_probe_cost() -> None:
    legal = medium_all_history_affine_cost("single_current_r8_all_history_affine")
    oracle = medium_all_history_affine_cost("full_exact_all_history_affine_oracle")
    kv = medium_single_r8_kv_splice_cost()
    assert legal.total_constructor_flops == 885_299_968
    assert legal.within_twenty_percent
    assert legal.sidecar_scalars == 38_016
    assert legal.incremental_reader_flops_per_query == 77_184
    assert legal.initial_gaussian_draws == 2_304
    assert oracle.total_constructor_flops == 4_927_034_496
    assert oracle.constructor_fraction > 1.0
    assert kv["total_constructor_flops"] == 853_836_992
    assert kv["within_twenty_percent"] is True
