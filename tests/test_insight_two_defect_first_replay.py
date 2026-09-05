from __future__ import annotations

import pytest
import torch
from insight_two.defect_first_replay import (
    MatrixFreeReleaseDefectInput,
    absolute_replay_from_initial_factors,
    add_token_factors,
    defect_first_release_replay,
    forward_one_with_native_response_defect,
    matrix_free_defect_first_initial_factors,
    medium_defect_first_costs,
    native_response_sidecar_scalars,
)
from insight_two.matrix_free_input_range import (
    hstu_input_operator,
    matrix_free_randomized_token_factors,
)
from insight_two.mode_space_replay import TokenModeFactors, truncated_token_factors

from hstu_kvcache.models import HSTU, HSTUConfig


def _model() -> HSTU:
    return HSTU(
        HSTUConfig(
            num_items=43,
            num_behaviors=5,
            hidden_size=8,
            num_layers=3,
            num_heads=2,
            head_dim=4,
            max_seq_len=16,
            temporal_num_freqs=2,
            input_dropout=0.0,
            block_variant="legacy",
        )
    ).eval()


def _history(length: int = 6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randint(1, 43, (1, length)),
        torch.randint(0, 6, (1, length)),
        torch.rand(1, length) * 200.0,
    )


def test_factor_sum_preserves_base_and_defect_exactly() -> None:
    torch.manual_seed(501)
    base = TokenModeFactors(torch.randn(2, 7, 2), torch.randn(2, 2, 8))
    defect = TokenModeFactors(torch.randn(2, 7, 4), torch.randn(2, 4, 8))
    observed = add_token_factors(base, defect)
    assert observed.rank == 6
    assert torch.allclose(
        observed.materialize(), base.materialize() + defect.materialize(), atol=2e-6
    )


def test_matrix_free_release_defect_matches_dense_products_and_factors() -> None:
    torch.manual_seed(502)
    parent = _model()
    current = _model()
    items, behaviors, deltas = _history(9)
    parent_operator = hstu_input_operator(parent, items, behaviors, deltas)
    current_operator = hstu_input_operator(current, items, behaviors, deltas)
    defect_operator = MatrixFreeReleaseDefectInput(parent_operator, current_operator)
    dense_defect = current.embed_inputs(items, behaviors, deltas) - parent.embed_inputs(
        items, behaviors, deltas
    )
    right = torch.randn(8, 5)
    left = torch.randn(1, 9, 5)
    assert torch.allclose(
        defect_operator.right_multiply(right),
        dense_defect @ right,
        atol=3e-5,
        rtol=3e-5,
    )
    assert torch.allclose(
        defect_operator.transpose_multiply(left),
        dense_defect.transpose(1, 2) @ left,
        atol=3e-5,
        rtol=3e-5,
    )
    expected = matrix_free_randomized_token_factors(
        defect_operator,
        rank=4,
        oversample=3,
        power_iterations=1,
        seed=117,
    )
    _, observed = matrix_free_defect_first_initial_factors(
        parent_operator,
        current_operator,
        base_rank=2,
        defect_rank=4,
        sketch_oversample=3,
        sketch_power_iterations=1,
        sketch_seed=117,
    )
    assert torch.allclose(
        observed.materialize(), expected.materialize(), atol=5e-5, rtol=5e-5
    )


def test_matrix_free_initial_path_never_calls_dense_input_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(503)
    parent = _model()
    current = _model()
    items, behaviors, deltas = _history(9)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense temporal/in_proj path was called")

    monkeypatch.setattr(parent.temporal_enc, "forward", forbidden)
    monkeypatch.setattr(parent.in_proj, "forward", forbidden)
    monkeypatch.setattr(current.temporal_enc, "forward", forbidden)
    monkeypatch.setattr(current.in_proj, "forward", forbidden)
    base, defect = matrix_free_defect_first_initial_factors(
        hstu_input_operator(parent, items, behaviors, deltas),
        hstu_input_operator(current, items, behaviors, deltas),
        base_rank=2,
        defect_rank=4,
        sketch_oversample=3,
        sketch_power_iterations=1,
        sketch_seed=118,
    )
    assert base.left.shape == (1, 9, 2)
    assert defect.left.shape == (1, 9, 4)


def test_defect_first_recurrence_has_finite_release_exact_limit() -> None:
    torch.manual_seed(504)
    parent = _model()
    current = _model()
    length = 6
    items, behaviors, deltas = _history(length)
    parent_x = parent.embed_inputs(items, behaviors, deltas)
    current_x = current.embed_inputs(items, behaviors, deltas)
    replay = defect_first_release_replay(
        parent,
        current,
        parent_x,
        current_x,
        base_rank=length,
        defect_rank=length,
        compression="exact_svd",
    )
    exact_parent = parent.compute_kv(items, behaviors, deltas)
    exact_current = current.compute_kv(items, behaviors, deltas)
    assert torch.allclose(replay.parent.cache.k, exact_parent.k, atol=9e-5, rtol=9e-5)
    assert torch.allclose(replay.parent.cache.v, exact_parent.v, atol=9e-5, rtol=9e-5)
    assert torch.allclose(replay.current.cache.k, exact_current.k, atol=1e-4, rtol=1e-4)
    assert torch.allclose(replay.current.cache.v, exact_current.v, atol=1e-4, rtol=1e-4)

    parent_state = parent_x
    current_state = current_x
    for layer in range(len(parent.blocks) - 1):
        parent_state, _ = parent.blocks[layer](parent_state, return_kv=True)
        current_state, _ = current.blocks[layer](current_state, return_kv=True)
        assert torch.allclose(
            replay.post_block_defects[layer],
            current_state - parent_state,
            atol=1e-4,
            rtol=1e-4,
        )


def test_native_response_difference_has_current_exact_full_rank_limit() -> None:
    torch.manual_seed(505)
    parent = _model()
    current = _model()
    length = 6
    items, behaviors, deltas = _history(length)
    replay = defect_first_release_replay(
        parent,
        current,
        parent.embed_inputs(items, behaviors, deltas),
        current.embed_inputs(items, behaviors, deltas),
        base_rank=length,
        defect_rank=length,
        compression="exact_svd",
    )
    exact_parent = parent.compute_kv(items, behaviors, deltas)
    exact_current = current.compute_kv(items, behaviors, deltas)
    candidate = torch.randint(1, 43, (1, 1))
    query = current.embed_query_tokens(candidate, torch.tensor([250.0]))
    expected, _ = current.forward_with_cache_embedded(exact_current, query)
    observed = forward_one_with_native_response_defect(
        current,
        exact_parent,
        replay.parent,
        replay.current,
        query,
    )
    assert torch.allclose(observed, expected, atol=2e-4, rtol=2e-4)


def test_absolute_terminal_control_has_full_rank_exact_limit() -> None:
    torch.manual_seed(506)
    model = _model()
    length = 6
    items, behaviors, deltas = _history(length)
    embedded = model.embed_inputs(items, behaviors, deltas)
    replay = absolute_replay_from_initial_factors(
        model,
        truncated_token_factors(embedded, rank=length),
        rank=length,
        compression="exact_svd",
    )
    exact = model.compute_kv(items, behaviors, deltas)
    assert torch.allclose(replay.cache.k, exact.k, atol=8e-5, rtol=8e-5)
    assert torch.allclose(replay.cache.v, exact.v, atol=8e-5, rtol=8e-5)


def test_response_sidecar_counts_both_factorized_trajectories() -> None:
    torch.manual_seed(507)
    parent = _model()
    current = _model()
    items, behaviors, deltas = _history(7)
    replay = defect_first_release_replay(
        parent,
        current,
        parent.embed_inputs(items, behaviors, deltas),
        current.embed_inputs(items, behaviors, deltas),
        base_rank=2,
        defect_rank=4,
        compression="fixed_range_finder",
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=119,
    )
    # Per layer: (N + 2H) * (parent rank 2 + Current rank 6).
    assert native_response_sidecar_scalars(replay.parent, replay.current) == (
        3 * (7 + 2 * 8) * 8
    )


def test_medium_costs_are_matched_and_below_twenty_percent() -> None:
    costs = medium_defect_first_costs()
    assert costs["defect_first_b2_d4"]["initial_factor_flops"] == 48_999_118
    assert costs["defect_first_b2_d4"]["per_nonterminal_transition_flops"] == 166_063_054
    assert costs["defect_first_b2_d4"]["total_flops_per_user"] == 880_621_524
    assert costs["ordinary_asymmetric_p2_c6"]["total_flops_per_user"] == 887_589_512
    assert costs["paired_absolute_p4_c4"]["total_flops_per_user"] == 872_238_088
    assert costs["single_absolute_c8"]["total_flops_per_user"] == 853_836_992
    assert all(bool(row["within_twenty_percent"]) for row in costs.values())
    assert all(row["persistent_sidecar_scalars_fp32"] == 67_584 for row in costs.values())
    assert costs["defect_first_b2_d4"]["per_request_extra_two_factor_response_flops"] == 1_218_816
    assert costs["defect_first_b2_d4"]["native_activation_evaluations_per_request"] == 73_728
