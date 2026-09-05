from __future__ import annotations

import torch
from insight_two.defect_first_replay import (
    forward_one_with_native_response_defect,
)
from insight_two.mode_space_replay import truncated_token_factors
from insight_two.source_residual_closure import (
    deim_interpolation_rows,
    exact_parent_block_update_rows,
    interpolate_sampled_residual,
    medium_source_defect_closure_cost,
    medium_source_residual_closure_cost,
    source_defect_closed_replay_from_initial_factors,
    source_residual_closed_replay_from_initial_factors,
)

from hstu_kvcache.models import HSTU, HSTUConfig


def _model() -> HSTU:
    return HSTU(
        HSTUConfig(
            num_items=47,
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
        torch.randint(1, 47, (1, length)),
        torch.randint(0, 6, (1, length)),
        torch.rand(1, length) * 200.0,
    )


def test_deim_lift_reproduces_all_source_test_functionals() -> None:
    torch.manual_seed(701)
    trial = torch.randn(1, 11, 4)
    basis, positions = deim_interpolation_rows(trial)
    samples = torch.randn(4, 1, 8)
    lifted, condition = interpolate_sampled_residual(basis, positions, samples)
    assert positions.unique().numel() == 4
    assert condition >= 1.0
    torch.testing.assert_close(
        lifted[0].index_select(0, positions),
        samples[:, 0],
        atol=3e-5,
        rtol=3e-5,
    )


def test_exact_parent_row_response_matches_native_parent_block() -> None:
    torch.manual_seed(702)
    model = _model()
    items, actions, deltas = _history(7)
    exact = model.compute_kv(items, actions, deltas)
    x = model.embed_inputs(items, actions, deltas)
    positions = torch.tensor([0, 3, 6])
    for layer, block in enumerate(model.blocks):
        next_x, _ = block(x, return_kv=True)
        expected_update = (next_x - x)[0].index_select(0, positions)[:, None]
        observed = exact_parent_block_update_rows(
            block,
            exact.k[layer],
            exact.v[layer],
            positions,
        )
        torch.testing.assert_close(observed, expected_update, atol=2e-4, rtol=2e-4)
        x = next_x


def test_source_closed_replay_has_full_rank_current_exact_limit() -> None:
    torch.manual_seed(703)
    parent = _model()
    current = _model()
    length = 6
    items, actions, deltas = _history(length)
    parent_x = parent.embed_inputs(items, actions, deltas)
    current_x = current.embed_inputs(items, actions, deltas)
    exact_parent = parent.compute_kv(items, actions, deltas)
    exact_current = current.compute_kv(items, actions, deltas)
    replay = source_residual_closed_replay_from_initial_factors(
        parent,
        current,
        exact_parent,
        truncated_token_factors(parent_x, rank=length),
        truncated_token_factors(current_x, rank=length),
        rank=length,
        compression="exact_svd",
    )
    torch.testing.assert_close(
        replay.paired.parent.cache.k, exact_parent.k, atol=2e-4, rtol=2e-4
    )
    torch.testing.assert_close(
        replay.paired.parent.cache.v, exact_parent.v, atol=2e-4, rtol=2e-4
    )
    torch.testing.assert_close(
        replay.paired.current.cache.k, exact_current.k, atol=3e-4, rtol=3e-4
    )
    torch.testing.assert_close(
        replay.paired.current.cache.v, exact_current.v, atol=3e-4, rtol=3e-4
    )
    assert max(
        certificate.interpolation_max_abs_error
        for certificate in replay.certificates
    ) < 2e-4

    candidate = torch.randint(1, 47, (1, 1))
    query = current.embed_query_tokens(candidate, torch.tensor([250.0]))
    expected, _ = current.forward_with_cache_embedded(exact_current, query)
    observed = forward_one_with_native_response_defect(
        current,
        exact_parent,
        replay.paired.parent,
        replay.paired.current,
        query,
    )
    torch.testing.assert_close(observed, expected, atol=4e-4, rtol=4e-4)


def test_zero_release_keeps_the_two_closed_arms_identical() -> None:
    torch.manual_seed(704)
    model = _model()
    items, actions, deltas = _history(8)
    embedded = model.embed_inputs(items, actions, deltas)
    initial = truncated_token_factors(embedded, rank=4)
    exact = model.compute_kv(items, actions, deltas)
    replay = source_residual_closed_replay_from_initial_factors(
        model,
        model,
        exact,
        initial,
        initial,
        rank=4,
        compression="exact_svd",
    )
    torch.testing.assert_close(
        replay.paired.current.cache.k, replay.paired.parent.cache.k
    )
    torch.testing.assert_close(
        replay.paired.current.cache.v, replay.paired.parent.cache.v
    )


def test_source_defect_closure_has_full_rank_current_exact_limit() -> None:
    torch.manual_seed(705)
    parent = _model()
    current = _model()
    length = 6
    items, actions, deltas = _history(length)
    parent_x = parent.embed_inputs(items, actions, deltas)
    current_x = current.embed_inputs(items, actions, deltas)
    exact_parent = parent.compute_kv(items, actions, deltas)
    exact_current = current.compute_kv(items, actions, deltas)
    replay = source_defect_closed_replay_from_initial_factors(
        parent,
        current,
        exact_parent,
        truncated_token_factors(parent_x, rank=length),
        truncated_token_factors(current_x, rank=length),
        rank=length,
        compression="exact_svd",
    )
    torch.testing.assert_close(
        replay.paired.parent.cache.k, exact_parent.k, atol=2e-4, rtol=2e-4
    )
    torch.testing.assert_close(
        replay.paired.current.cache.k, exact_current.k, atol=4e-4, rtol=4e-4
    )
    torch.testing.assert_close(
        replay.paired.current.cache.v, exact_current.v, atol=4e-4, rtol=4e-4
    )
    assert max(
        torch.linalg.vector_norm(certificate.sampled_residual.float()).item()
        for certificate in replay.certificates
    ) < 4e-3


def test_source_defect_closure_is_identically_zero_for_zero_release() -> None:
    torch.manual_seed(706)
    model = _model()
    items, actions, deltas = _history(8)
    embedded = model.embed_inputs(items, actions, deltas)
    initial = truncated_token_factors(embedded, rank=4)
    exact = model.compute_kv(items, actions, deltas)
    replay = source_defect_closed_replay_from_initial_factors(
        model,
        model,
        exact,
        initial,
        initial,
        rank=4,
        compression="exact_svd",
    )
    torch.testing.assert_close(
        replay.paired.current.cache.k, replay.paired.parent.cache.k
    )
    assert max(
        torch.linalg.vector_norm(certificate.sampled_residual.float()).item()
        for certificate in replay.certificates
    ) < 2e-4


def test_medium_source_certificate_fits_strict_twenty_percent_ledger() -> None:
    cost = medium_source_residual_closure_cost()
    assert cost.base_paired_native_flops == 872_238_088
    assert cost.certificate_flops == 33_428_910
    assert cost.total_constructor_flops == 905_666_998
    assert cost.constructor_fraction < 0.20
    assert cost.within_twenty_percent
    defect = medium_source_defect_closure_cost()
    assert defect.certificate_flops == 56_852_910
    assert defect.total_constructor_flops == 929_090_998
    assert defect.constructor_fraction < 0.20
    assert defect.within_twenty_percent
