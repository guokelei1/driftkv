"""Numerical oracle for batched heads and compact retained SVD storage."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.analysis import decompose_interaction
from experiments.ch3_motivation.code.lowrank_batch import BatchedReconstruction, decompose_interaction_batch
from experiments.ch3_motivation.code.lowrank_rank_batch import reconstruct_rank_batch


@pytest.mark.parametrize("method", ["exact", "randomized"])
@pytest.mark.parametrize("only,normalize", [("both", False), ("both", True), ("key", True), ("value", True)])
def test_batch_matches_independent_scalar_heads(method, only, normalize):
    generator = torch.Generator().manual_seed(17)
    h = torch.randn(4, 12, 21, 8, dtype=torch.float64, generator=generator)
    h[..., 4:] *= 12
    options = dict(method=method, only=only, normalize_kv=normalize,
                   max_rank=6, oversample=3, power_iterations=1, seed=93)
    batch = decompose_interaction_batch(h, **options)
    packed = BatchedReconstruction(h, batch)
    candidates, rows = torch.tensor([2, 8]), slice(3, 11)
    for rank in (0, 2, 6, None):
        expected = torch.stack([obj.reconstruct(rank, candidates=candidates, rows=rows) for obj in batch])
        torch.testing.assert_close(packed.reconstruct(rank, candidates=candidates, rows=rows), expected)
    ranks = (0, 1, 2, 6, 16)
    expected = torch.stack([packed.reconstruct(min(rank, packed.max_rank), candidates=candidates, rows=rows) for rank in ranks])
    torch.testing.assert_close(reconstruct_rank_batch(packed, ranks, candidates=candidates, rows=rows), expected)
    for head, actual in enumerate(batch):
        expected = decompose_interaction(h[head], **options)
        for rank in (0, 1, 2, 6):
            torch.testing.assert_close(actual.reconstruct(rank), expected.reconstruct(rank), atol=1e-10, rtol=1e-10)
        assert torch.equal(actual.reconstruct(None), h[head])
        for factor in (actual.left, actual.singular_values, actual.right):
            assert factor.untyped_storage().nbytes() == factor.numel()*factor.element_size()


def test_batch_zero_interaction_and_zero_rank_are_finite():
    h = torch.arange(4*9*6, dtype=torch.float64).reshape(4, 9, 1, 6).expand(4, 9, 8, 6).clone()
    for method in ("exact", "randomized"):
        result = decompose_interaction_batch(h, max_rank=0, normalize_kv=True, method=method)
        for head, obj in enumerate(result):
            assert obj.max_rank == 0
            assert torch.equal(obj.reconstruct(0), h[head])


def test_energy_screen_preserves_full_small_spectrum_and_only_failed_matrix_uses_exact(monkeypatch):
    generator = torch.Generator().manual_seed(123)
    low = torch.randn(24, 4, dtype=torch.float64, generator=generator)[:, None]*torch.randn(
        1, 30, 1, dtype=torch.float64, generator=generator)
    flat = torch.randn(24, 30, 4, dtype=torch.float64, generator=generator)
    h = torch.stack((low, flat))
    calls, original = [], torch.linalg.svd
    def counted(matrix, *args, **kwargs):
        calls.append(tuple(matrix.shape))
        return original(matrix, *args, **kwargs)
    monkeypatch.setattr(torch.linalg, "svd", counted)
    result = decompose_interaction_batch(h, method="randomized", max_rank=6,
        oversample=2, power_iterations=0, energy_guard=True)
    first, second = [obj.metadata["energy_guard"] for obj in result]
    assert not first["exact_fallback"] and second["exact_fallback"]
    assert len(first["all_sketch_singular_values"]) == 8 > result[0].max_rank
    assert min(first["captured_energy_ratio_lower_bound"]) >= .98
    assert sum(len(shape) == 2 for shape in calls) == 1
    assert result[0].metadata["exact_fallback_batch_matrices"] == 1
    exact = decompose_interaction(h[1], method="exact", max_rank=6)
    torch.testing.assert_close(result[1].reconstruct(6), exact.reconstruct(6), atol=1e-11, rtol=1e-11)
    assert result[0].metadata["energy_guard_batch_seconds"] >= 0
    assert result[0].metadata["exact_fallback_batch_seconds"] > 0


@pytest.mark.parametrize("batch_heads", [1, 4, 7])
def test_cross_layer_batches_preserve_head_order_and_per_layer_sketch(batch_heads):
    from experiments.ch3_motivation.code.release_lowrank import decompose_release_heads
    generator = torch.Generator().manual_seed(22)
    h = torch.randn(3, 2, 12, 21, 8, dtype=torch.float64, generator=generator)
    options = dict(method="randomized", max_rank=6, oversample=3, power_iterations=1)
    actual = decompose_release_heads(h, batch_heads=batch_heads, seed=93, **options)
    assert len(actual) == 6
    for index, obj in enumerate(actual):
        layer, head = divmod(index, 2)
        expected = decompose_interaction(h[layer, head], seed=93+layer, **options)
        torch.testing.assert_close(obj.reconstruct(6), expected.reconstruct(6), atol=1e-10, rtol=1e-10)
        assert obj.metadata["seed"] == 93+layer
