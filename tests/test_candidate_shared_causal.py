from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from candidate_shared_causal import (  # noqa: E402
    nested_width_indices,
    signed_candidate_decomposition,
    signed_head_intervention,
)
from hstu_kvcache.models import HSTU, HSTUConfig  # noqa: E402


def test_nested_width_indices_are_deterministic_subsets() -> None:
    width8 = nested_width_indices(64, 8)
    width16 = nested_width_indices(64, 16)
    width32 = nested_width_indices(64, 32)
    assert np.array_equal(width8, np.arange(0, 64, 8))
    assert set(width8).issubset(set(width16))
    assert set(width16).issubset(set(width32))


def test_signed_decomposition_is_exact_and_orthogonal() -> None:
    generator = torch.Generator().manual_seed(17)
    exact = torch.randn(3, 8, 2, 1, 4, generator=generator)
    reuse = torch.randn(3, 8, 2, 1, 4, generator=generator)
    shared, residual, metrics = signed_candidate_decomposition(exact, reuse)
    assert torch.allclose(reuse + shared + residual, exact, atol=1e-6)
    assert torch.max(metrics["orthogonality_error"]) < 1e-6
    assert torch.allclose(
        metrics["shared_energy"] + metrics["residual_energy"],
        metrics["total_energy"],
        rtol=1e-5,
        atol=1e-5,
    )


def _model(seed: int) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=64,
            num_behaviors=3,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            max_seq_len=32,
            temporal_num_freqs=2,
            input_dropout=0.0,
        )
    ).eval()


def test_full_signed_delta_reconstructs_native_current_reader() -> None:
    parent, current = _model(3), _model(5)
    items = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]
    )
    behaviors = torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2], [2, 1, 2, 1, 2, 1, 2, 1]])
    deltas = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 4, 6, 8, 10, 12, 14]]).float()
    candidates = torch.tensor([[17, 18, 19, 20], [21, 22, 23, 24]])
    query_deltas = torch.tensor([10.0, 20.0])
    parent_cache = parent.compute_kv(items, behaviors, deltas)
    current_cache = current.compute_kv(items, behaviors, deltas)

    native_exact, _ = current.observe_cc_reuse(current_cache, candidates, query_deltas)
    native_reuse, _ = current.observe_cc_reuse(parent_cache, candidates, query_deltas)
    exact = signed_head_intervention(
        current, current_cache, parent_cache, candidates, query_deltas, mode="exact"
    )
    reuse = signed_head_intervention(
        current, current_cache, parent_cache, candidates, query_deltas, mode="reuse"
    )
    full = signed_head_intervention(
        current, current_cache, parent_cache, candidates, query_deltas, mode="full_delta"
    )
    assert torch.allclose(exact.scores, native_exact, atol=1e-6)
    assert torch.allclose(reuse.scores, native_reuse, atol=1e-6)
    assert torch.allclose(full.scores, native_exact, atol=1e-6)
