from __future__ import annotations

import pytest
import torch
from insight_two.matrix_free_input_range import (
    hstu_input_operator,
    matrix_free_input_cost,
    matrix_free_randomized_token_factors,
    medium_paired_kv_only_cost,
)
from insight_two.mode_space_replay import randomized_token_factors

from hstu_kvcache.models import HSTU, HSTUConfig


def _model() -> HSTU:
    return HSTU(
        HSTUConfig(
            num_items=41,
            num_behaviors=5,
            hidden_size=12,
            num_layers=2,
            num_heads=3,
            head_dim=4,
            max_seq_len=20,
            temporal_num_freqs=3,
            input_dropout=0.0,
            block_variant="legacy",
        )
    ).eval()


def _history() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    items = torch.randint(1, 41, (2, 13))
    behaviors = torch.randint(0, 6, (2, 13))
    deltas = torch.rand(2, 13) * 300.0
    return items, behaviors, deltas


def test_matrix_free_operator_matches_dense_input_products() -> None:
    torch.manual_seed(401)
    model = _model()
    items, behaviors, deltas = _history()
    dense = model.embed_inputs(items, behaviors, deltas).float()
    operator = hstu_input_operator(model, items, behaviors, deltas)

    shared_right = torch.randn(12, 5)
    batched_right = torch.randn(2, 12, 5)
    left = torch.randn(2, 13, 5)
    assert torch.allclose(
        operator.right_multiply(shared_right),
        dense @ shared_right,
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.allclose(
        operator.right_multiply(batched_right),
        dense @ batched_right,
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.allclose(
        operator.transpose_multiply(left),
        dense.transpose(1, 2) @ left,
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize("power_iterations", [0, 1])
@pytest.mark.parametrize("truncation", ["svd", "gram_eigh"])
def test_matrix_free_range_finder_matches_dense_semantics(
    power_iterations: int,
    truncation: str,
) -> None:
    torch.manual_seed(402)
    model = _model()
    items, behaviors, deltas = _history()
    dense = model.embed_inputs(items, behaviors, deltas)
    expected = randomized_token_factors(
        dense,
        rank=4,
        oversample=3,
        power_iterations=power_iterations,
        seed=117,
    )
    observed = matrix_free_randomized_token_factors(
        hstu_input_operator(model, items, behaviors, deltas),
        rank=4,
        oversample=3,
        power_iterations=power_iterations,
        seed=117,
        truncation=truncation,
    )
    # QR/SVD signs are implementation choices, so compare the represented
    # approximation rather than requiring bit-identical individual factors.
    assert torch.allclose(
        observed.materialize(),
        expected.materialize(),
        atol=8e-5,
        rtol=8e-5,
    )


def test_matrix_free_path_does_not_call_dense_temporal_or_input_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(403)
    model = _model()
    items, behaviors, deltas = _history()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense temporal/in_proj path was called")

    monkeypatch.setattr(model.temporal_enc, "forward", forbidden)
    monkeypatch.setattr(model.in_proj, "forward", forbidden)
    factors = matrix_free_randomized_token_factors(
        hstu_input_operator(model, items, behaviors, deltas),
        rank=4,
        oversample=3,
        power_iterations=1,
        seed=118,
    )
    assert factors.left.shape == (2, 13, 4)
    assert factors.right.shape == (2, 4, 12)


def test_medium_cost_counts_all_four_operator_applications() -> None:
    cost = matrix_free_input_cost(
        history_length=1024,
        hidden_size=192,
        temporal_num_freqs=16,
        rank=4,
        oversample=4,
        power_iterations=1,
    )
    assert cost.base_lookup_additions == 196_608
    assert cost.temporal_phase_multiplications == 16_384
    assert cost.right_operator_applications == 8_732_672
    assert cost.transpose_operator_applications == 8_719_360
    assert cost.thin_qr == 261_462
    assert cost.small_gram_eigh_rotation == 107_008
    assert cost.flops == 18_033_494
    assert cost.gaussian_draws == 1_536
    assert cost.sin_cos_evaluations == 32_768
    assert cost.frequency_exponentials == 16
    assert cost.embedding_lookup_scalars == 393_216
    assert cost.raw_history_scalars == 3_072


def test_matrix_free_input_moves_paired_kv_only_point_below_twenty_percent() -> None:
    cost = medium_paired_kv_only_cost()
    assert cost.old_two_raw_and_initial_flops == 202_882_732
    assert cost.new_two_matrix_free_initial_flops == 36_066_988
    assert cost.total_flops == 874_402_376
    assert cost.exact_all_fraction == pytest.approx(0.18326357633004797)
    assert cost.exact_all_fraction <= 0.20


def test_matrix_free_input_rejects_training_mode() -> None:
    model = _model().train()
    items, behaviors, deltas = _history()
    with pytest.raises(ValueError, match=r"model\.eval"):
        hstu_input_operator(model, items, behaviors, deltas)
