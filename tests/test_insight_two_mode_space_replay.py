from __future__ import annotations

import pytest
import torch
from insight_two.mode_space_replay import (
    TokenModeFactors,
    approximate_layer0_defect_basis,
    approximate_paired_layer0_defect_basis,
    dependency_free_layer0_defect_basis,
    factorized_legacy_attention,
    factorized_reduced_current_replay,
    factorized_rmsnorm,
    forward_one_with_shared_mode_splice,
    orthonormal_token_basis,
    paired_release_replay,
    project_onto_token_basis,
    randomized_token_basis,
    randomized_token_factors,
    reduced_current_replay,
    splice_current_modes_into_parent,
    splice_shared_modes_from_factorized_replay,
    splice_shared_modes_from_paired_replay,
    truncated_token_factors,
)

from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache


def _model() -> HSTU:
    model = HSTU(
        HSTUConfig(
            num_items=31,
            num_behaviors=4,
            hidden_size=8,
            num_layers=2,
            num_heads=2,
            head_dim=4,
            max_seq_len=12,
            temporal_num_freqs=2,
            input_dropout=0.0,
            block_variant="legacy",
        )
    )
    return model.eval()


def test_truncated_factors_and_projection_exact_at_full_rank() -> None:
    torch.manual_seed(3)
    matrix = torch.randn(2, 5, 4)
    factors = truncated_token_factors(matrix, rank=4)
    assert factors.rank == 4
    assert torch.allclose(factors.materialize(), matrix, atol=2e-5, rtol=2e-5)
    basis = orthonormal_token_basis(matrix, rank=4)
    assert torch.allclose(
        project_onto_token_basis(matrix, basis), matrix, atol=2e-5, rtol=2e-5
    )


def test_projection_rejects_nonorthonormal_basis() -> None:
    matrix = torch.randn(1, 5, 3)
    with pytest.raises(ValueError, match="orthonormal"):
        project_onto_token_basis(matrix, torch.ones(1, 5, 2))


def test_fixed_range_finder_is_deterministic_and_exact_on_low_rank_input() -> None:
    torch.manual_seed(31)
    matrix = torch.randn(2, 11, 3) @ torch.randn(2, 3, 9)
    first = randomized_token_factors(
        matrix, rank=3, oversample=2, power_iterations=1, seed=101
    )
    second = randomized_token_factors(
        matrix, rank=3, oversample=2, power_iterations=1, seed=101
    )
    assert torch.equal(first.left, second.left)
    assert torch.equal(first.right, second.right)
    assert torch.allclose(first.materialize(), matrix, atol=4e-5, rtol=4e-5)
    basis = randomized_token_basis(
        matrix, rank=3, oversample=2, power_iterations=1, seed=101
    )
    assert torch.allclose(
        project_onto_token_basis(matrix, basis), matrix, atol=4e-5, rtol=4e-5
    )


def test_fixed_range_finder_validates_iteration_controls() -> None:
    matrix = torch.randn(1, 6, 4)
    with pytest.raises(ValueError, match="nonnegative"):
        randomized_token_factors(matrix, rank=2, oversample=-1)
    with pytest.raises(ValueError, match="nonnegative"):
        randomized_token_factors(matrix, rank=2, power_iterations=-1)


def test_reduced_replay_full_rank_matches_native_cache() -> None:
    torch.manual_seed(4)
    model = _model()
    items = torch.randint(0, 31, (1, 6))
    behaviors = torch.randint(0, 4, (1, 6))
    deltas = torch.arange(6, dtype=torch.float32)[None]
    embedded = model.embed_inputs(items, behaviors, deltas)
    replay = reduced_current_replay(model, embedded, rank=6)
    exact = model.compute_kv(items, behaviors, deltas)
    assert torch.allclose(replay.cache.k, exact.k, atol=3e-5, rtol=3e-5)
    assert torch.allclose(replay.cache.v, exact.v, atol=3e-5, rtol=3e-5)


def test_mode_splice_replaces_basis_and_preserves_complement() -> None:
    torch.manual_seed(5)
    layers, batch, length, width, rank = 2, 1, 7, 4, 2
    parent = HSTUKVCache(
        k=torch.randn(layers, batch, length, width),
        v=torch.randn(layers, batch, length, width),
        seq_len=length,
    )
    reduced = HSTUKVCache(
        k=torch.randn(layers, batch, length, width),
        v=torch.randn(layers, batch, length, width),
        seq_len=length,
    )
    bases = tuple(
        orthonormal_token_basis(torch.randn(batch, length, rank), rank=rank)
        for _ in range(layers)
    )
    splice = splice_current_modes_into_parent(parent, reduced, bases)
    for layer, basis in enumerate(bases):
        projector = basis @ basis.transpose(1, 2)
        complement = torch.eye(length)[None] - projector
        assert torch.allclose(
            basis.transpose(1, 2) @ splice.cache.k[layer],
            basis.transpose(1, 2) @ reduced.k[layer],
            atol=2e-5,
            rtol=2e-5,
        )
        assert torch.allclose(
            complement @ splice.cache.k[layer],
            complement @ parent.k[layer],
            atol=2e-5,
            rtol=2e-5,
        )
        assert torch.allclose(
            complement @ splice.cache.v[layer],
            complement @ parent.v[layer],
            atol=2e-5,
            rtol=2e-5,
        )
    assert splice.sidecar_scalars == layers * batch * rank * (length + 2 * width)


def test_layer0_defect_basis_reads_no_upper_layer_state() -> None:
    torch.manual_seed(6)
    parent = _model()
    current = _model()
    items = torch.randint(0, 31, (1, 7))
    behaviors = torch.randint(0, 4, (1, 7))
    deltas = torch.arange(7, dtype=torch.float32)[None]
    parent_x = parent.embed_inputs(items, behaviors, deltas)
    current_x = current.embed_inputs(items, behaviors, deltas)
    basis = dependency_free_layer0_defect_basis(
        parent, current, parent_x, current_x, rank=3
    )
    assert basis.shape == (1, 7, 3)
    gram = basis.transpose(1, 2) @ basis
    assert torch.allclose(gram, torch.eye(3)[None], atol=2e-5, rtol=2e-5)


def test_factorized_legacy_attention_matches_dense_native_attention() -> None:
    torch.manual_seed(7)
    model = _model()
    block = model.blocks[0]
    factors = TokenModeFactors(
        left=torch.randn(1, 6, 3),
        right=torch.randn(1, 3, 8),
    )
    dense = factors.materialize()
    expected, (expected_k, expected_v) = block.attn(dense, return_kv=True)
    observed, observed_k, observed_v = factorized_legacy_attention(block, factors)
    assert torch.allclose(observed, expected, atol=3e-5, rtol=3e-5)
    assert torch.allclose(observed_k, expected_k, atol=2e-6, rtol=2e-6)
    assert torch.allclose(observed_v, expected_v, atol=2e-6, rtol=2e-6)


def test_factorized_rmsnorm_matches_dense_norm() -> None:
    torch.manual_seed(8)
    model = _model()
    factors = TokenModeFactors(
        left=torch.randn(2, 7, 3),
        right=torch.randn(2, 3, 8),
    )
    expected = model.blocks[0].norm(factors.materialize())
    observed = factorized_rmsnorm(model.blocks[0].norm, factors).materialize()
    assert torch.allclose(observed, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("compression", ["exact_svd", "fixed_range_finder"])
def test_factorized_replay_matches_dense_semantic_replay(compression: str) -> None:
    torch.manual_seed(9)
    model = _model()
    items = torch.randint(0, 31, (1, 7))
    behaviors = torch.randint(0, 4, (1, 7))
    deltas = torch.arange(7, dtype=torch.float32)[None]
    embedded = model.embed_inputs(items, behaviors, deltas)
    dense = reduced_current_replay(
        model,
        embedded,
        rank=3,
        compression=compression,
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=71,
    )
    factorized = factorized_reduced_current_replay(
        model,
        embedded,
        rank=3,
        compression=compression,
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=71,
    )
    assert torch.allclose(factorized.cache.k, dense.cache.k, atol=8e-5, rtol=8e-5)
    assert torch.allclose(factorized.cache.v, dense.cache.v, atol=8e-5, rtol=8e-5)


def test_shared_factorized_splice_matches_dense_splice_and_reader() -> None:
    torch.manual_seed(10)
    parent_model = _model()
    current_model = _model()
    items = torch.randint(0, 31, (1, 7))
    behaviors = torch.randint(0, 4, (1, 7))
    deltas = torch.arange(7, dtype=torch.float32)[None]
    parent_cache = parent_model.compute_kv(items, behaviors, deltas)
    embedded = current_model.embed_inputs(items, behaviors, deltas)
    replay = factorized_reduced_current_replay(
        current_model,
        embedded,
        rank=3,
        compression="fixed_range_finder",
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=81,
    )
    basis = approximate_layer0_defect_basis(
        parent_cache,
        replay,
        rank=3,
        oversample=2,
        power_iterations=1,
        seed=1081,
    )
    shared = splice_shared_modes_from_factorized_replay(
        parent_cache, replay, basis
    )
    dense = splice_current_modes_into_parent(
        parent_cache, replay.cache, tuple(basis for _ in range(2))
    )
    assert torch.allclose(shared.cache.k, dense.cache.k, atol=3e-5, rtol=3e-5)
    assert torch.allclose(shared.cache.v, dense.cache.v, atol=3e-5, rtol=3e-5)
    assert shared.sidecar_scalars == basis.numel() + 2 * 2 * 3 * 8

    candidate = torch.randint(0, 31, (1, 1))
    query_delta = torch.tensor([8.0])
    query = current_model.embed_query_tokens(candidate, query_delta)
    expected, _ = current_model.forward_with_cache_embedded(shared.cache, query)
    observed = forward_one_with_shared_mode_splice(
        current_model, parent_cache, shared, query
    )
    assert torch.allclose(observed, expected, atol=4e-5, rtol=4e-5)


@pytest.mark.parametrize("compression", ["exact_svd", "fixed_range_finder"])
def test_paired_recurrence_matches_equal_resolution_independent_arms(
    compression: str,
) -> None:
    torch.manual_seed(11)
    parent_model = _model()
    current_model = _model()
    items = torch.randint(0, 31, (1, 7))
    behaviors = torch.randint(0, 4, (1, 7))
    deltas = torch.arange(7, dtype=torch.float32)[None]
    parent_x = parent_model.embed_inputs(items, behaviors, deltas)
    current_x = current_model.embed_inputs(items, behaviors, deltas)
    arguments = {
        "rank": 3,
        "compression": compression,
        "sketch_oversample": 2,
        "sketch_power_iterations": 1,
        "sketch_seed": 91,
    }
    paired = paired_release_replay(
        parent_model,
        current_model,
        parent_x,
        current_x,
        **arguments,
    )
    parent_only = factorized_reduced_current_replay(
        parent_model, parent_x, **arguments
    )
    current_only = factorized_reduced_current_replay(
        current_model, current_x, **arguments
    )
    assert torch.allclose(
        paired.parent.cache.k, parent_only.cache.k, atol=8e-5, rtol=8e-5
    )
    assert torch.allclose(
        paired.current.cache.k, current_only.cache.k, atol=8e-5, rtol=8e-5
    )
    assert torch.allclose(
        paired.current.cache.v, current_only.cache.v, atol=8e-5, rtol=8e-5
    )


def test_paired_splice_has_current_exact_full_rank_limit() -> None:
    torch.manual_seed(12)
    parent_model = _model()
    current_model = _model()
    length = 6
    items = torch.randint(0, 31, (1, length))
    behaviors = torch.randint(0, 4, (1, length))
    deltas = torch.arange(length, dtype=torch.float32)[None]
    parent_x = parent_model.embed_inputs(items, behaviors, deltas)
    current_x = current_model.embed_inputs(items, behaviors, deltas)
    exact_parent = parent_model.compute_kv(items, behaviors, deltas)
    exact_current = current_model.compute_kv(items, behaviors, deltas)
    paired = paired_release_replay(
        parent_model,
        current_model,
        parent_x,
        current_x,
        rank=length,
        compression="exact_svd",
    )
    full_token_basis = torch.eye(length)[None]
    splice = splice_shared_modes_from_paired_replay(
        exact_parent, paired, full_token_basis
    )
    assert torch.allclose(splice.cache.k, exact_current.k, atol=7e-5, rtol=7e-5)
    assert torch.allclose(splice.cache.v, exact_current.v, atol=7e-5, rtol=7e-5)


def test_paired_layer0_basis_uses_approximate_release_difference() -> None:
    torch.manual_seed(13)
    parent_model = _model()
    current_model = _model()
    items = torch.randint(0, 31, (1, 7))
    behaviors = torch.randint(0, 4, (1, 7))
    deltas = torch.arange(7, dtype=torch.float32)[None]
    paired = paired_release_replay(
        parent_model,
        current_model,
        parent_model.embed_inputs(items, behaviors, deltas),
        current_model.embed_inputs(items, behaviors, deltas),
        rank=3,
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=111,
    )
    basis = approximate_paired_layer0_defect_basis(
        paired,
        rank=4,
        oversample=2,
        power_iterations=0,
        seed=1111,
    )
    assert basis.shape == (1, 7, 4)
    assert torch.allclose(
        basis.transpose(1, 2) @ basis,
        torch.eye(4)[None],
        atol=3e-5,
        rtol=3e-5,
    )
