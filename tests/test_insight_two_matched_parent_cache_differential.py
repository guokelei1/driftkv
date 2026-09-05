from __future__ import annotations

import torch
from insight_two.matched_parent_cache_differential import (
    approximate_exact_parent_cache,
    matched_layer0_defect_basis,
    paired_replay_layer0_defect_basis,
    splice_matched_parent_cache_differential,
    splice_paired_replay_differential,
)
from insight_two.mode_space_replay import factorized_reduced_current_replay

from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache


def _model() -> HSTU:
    return HSTU(
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
    ).eval()


def test_joint_parent_cache_operator_is_deterministic_and_exact_at_full_rank() -> None:
    torch.manual_seed(41)
    parent = HSTUKVCache(
        k=torch.randn(2, 1, 6, 4),
        v=torch.randn(2, 1, 6, 4),
        seq_len=6,
    )
    first = approximate_exact_parent_cache(
        parent, rank=6, oversample=2, power_iterations=1, seed=71
    )
    second = approximate_exact_parent_cache(
        parent, rank=6, oversample=2, power_iterations=1, seed=71
    )
    for layer, (left, right) in enumerate(zip(first.layers, second.layers, strict=True)):
        first_k, first_v = left.materialize()
        second_k, second_v = right.materialize()
        assert torch.equal(first_k, second_k)
        assert torch.equal(first_v, second_v)
        assert torch.allclose(first_k, parent.k[layer], atol=5e-5, rtol=5e-5)
        assert torch.allclose(first_v, parent.v[layer], atol=5e-5, rtol=5e-5)


def test_matched_splice_full_history_rank_recovers_current_cache() -> None:
    torch.manual_seed(42)
    parent_model = _model()
    current_model = _model()
    items = torch.randint(0, 31, (1, 6))
    behaviors = torch.randint(0, 4, (1, 6))
    deltas = torch.arange(6, dtype=torch.float32)[None]
    parent_cache = parent_model.compute_kv(items, behaviors, deltas)
    current_cache = current_model.compute_kv(items, behaviors, deltas)
    current_replay = factorized_reduced_current_replay(
        current_model,
        current_model.embed_inputs(items, behaviors, deltas),
        rank=6,
        compression="fixed_range_finder",
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=81,
    )
    parent_approximation = approximate_exact_parent_cache(
        parent_cache,
        rank=6,
        oversample=2,
        power_iterations=1,
        seed=81,
    )
    basis = matched_layer0_defect_basis(
        parent_cache,
        current_replay,
        parent_approximation,
        rank=6,
        oversample=2,
        power_iterations=0,
        seed=1081,
    )
    splice = splice_matched_parent_cache_differential(
        parent_cache, current_replay, parent_approximation, basis
    )
    assert torch.allclose(splice.cache.k, current_cache.k, atol=2e-4, rtol=2e-4)
    assert torch.allclose(splice.cache.v, current_cache.v, atol=2e-4, rtol=2e-4)


def test_paired_replay_splice_uses_approximate_parent_not_exact_parent() -> None:
    torch.manual_seed(43)
    parent_model = _model()
    current_model = _model()
    items = torch.randint(0, 31, (1, 7))
    behaviors = torch.randint(0, 4, (1, 7))
    deltas = torch.arange(7, dtype=torch.float32)[None]
    parent_cache = parent_model.compute_kv(items, behaviors, deltas)
    parent_replay = factorized_reduced_current_replay(
        parent_model,
        parent_model.embed_inputs(items, behaviors, deltas),
        rank=3,
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=91,
    )
    current_replay = factorized_reduced_current_replay(
        current_model,
        current_model.embed_inputs(items, behaviors, deltas),
        rank=3,
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=91,
    )
    basis = paired_replay_layer0_defect_basis(
        parent_cache,
        parent_replay,
        current_replay,
        rank=3,
        oversample=2,
        power_iterations=0,
        seed=1091,
    )
    splice = splice_paired_replay_differential(
        parent_cache, parent_replay, current_replay, basis
    )
    transpose = basis.transpose(1, 2)
    for layer in range(2):
        current_key, current_value = current_replay.layers[layer].materialize()
        parent_key, parent_value = parent_replay.layers[layer].materialize()
        assert torch.allclose(
            transpose @ (splice.cache.k[layer] - parent_cache.k[layer]),
            transpose @ (current_key - parent_key),
            atol=5e-5,
            rtol=5e-5,
        )
        assert torch.allclose(
            transpose @ (splice.cache.v[layer] - parent_cache.v[layer]),
            transpose @ (current_value - parent_value),
            atol=5e-5,
            rtol=5e-5,
        )


def test_matched_sidecar_has_one_basis_and_per_layer_signed_cores() -> None:
    torch.manual_seed(44)
    parent_model = _model()
    current_model = _model()
    items = torch.randint(0, 31, (1, 7))
    behaviors = torch.randint(0, 4, (1, 7))
    deltas = torch.arange(7, dtype=torch.float32)[None]
    parent_cache = parent_model.compute_kv(items, behaviors, deltas)
    current_replay = factorized_reduced_current_replay(
        current_model,
        current_model.embed_inputs(items, behaviors, deltas),
        rank=3,
        sketch_oversample=2,
        sketch_power_iterations=1,
        sketch_seed=101,
    )
    approximation = approximate_exact_parent_cache(
        parent_cache,
        rank=3,
        oversample=2,
        power_iterations=1,
        seed=101,
    )
    basis = matched_layer0_defect_basis(
        parent_cache,
        current_replay,
        approximation,
        rank=3,
        oversample=2,
        power_iterations=0,
        seed=1101,
    )
    splice = splice_matched_parent_cache_differential(
        parent_cache, current_replay, approximation, basis
    )
    assert splice.sidecar_scalars == 7 * 3 + 2 * 2 * 3 * 8
