from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from insight_two.address_response_memory import (
    build_oracle_address_response_memory,
    select_address_landmarks,
)
from insight_two.signed_response_memory import (
    intervene_oracle_signed_response_memory,
)


def _model(seed: int, *, relative_position_bias: bool = False) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=96,
            num_behaviors=3,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            max_seq_len=16,
            temporal_num_freqs=2,
            input_dropout=0.0,
            relative_position_bias=relative_position_bias,
        )
    ).eval()


def _random_cache_pair(seed: int, *, history_length: int = 12) -> tuple[HSTUKVCache, HSTUKVCache]:
    generator = torch.Generator().manual_seed(seed)
    shape = (2, 1, history_length, 16)
    exact = HSTUKVCache(
        k=torch.randn(shape, generator=generator),
        v=torch.randn(shape, generator=generator),
        seq_len=history_length,
    )
    reuse = HSTUKVCache(
        k=torch.randn(shape, generator=generator),
        v=torch.randn(shape, generator=generator),
        seq_len=history_length,
    )
    return exact, reuse


def test_address_selection_is_deterministic_and_nested() -> None:
    exact, reuse = _random_cache_pair(211)

    first = select_address_landmarks(exact, reuse, sample_count=5)
    repeated = select_address_landmarks(exact, reuse, sample_count=5)
    shorter = select_address_landmarks(exact, reuse, sample_count=3)

    assert torch.equal(first.selected_positions, repeated.selected_positions)
    assert torch.equal(first.cluster_masses, repeated.cluster_masses)
    assert torch.equal(first.assignments, repeated.assignments)
    assert torch.equal(shorter.selected_positions, first.selected_positions[:3])

    # An all-tie address geometry resolves every farthest-first choice to the
    # smallest still-unselected token index.
    tied = HSTUKVCache(
        k=torch.zeros(2, 1, 6, 16),
        v=torch.zeros(2, 1, 6, 16),
        seq_len=6,
    )
    tied_selection = select_address_landmarks(tied, tied, sample_count=4)
    assert torch.equal(tied_selection.selected_positions, torch.arange(4))


def test_voronoi_cluster_masses_are_positive_and_cover_history() -> None:
    exact, reuse = _random_cache_pair(223, history_length=13)
    selection = select_address_landmarks(exact, reuse, sample_count=5)

    assert selection.cluster_masses.dtype == torch.long
    assert torch.all(selection.cluster_masses > 0)
    assert int(selection.cluster_masses.sum()) == exact.seq_len
    assert selection.assignments.shape == (exact.seq_len,)
    assert int(selection.assignments.min()) == 0
    assert int(selection.assignments.max()) < selection.sample_count


def test_full_address_memory_reconstructs_exact_reader_and_preserves_inputs() -> None:
    parent = _model(227, relative_position_bias=True)
    current = _model(229, relative_position_bias=True)
    items = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    behaviors = torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2]])
    deltas = torch.arange(8).float().reshape(1, 8)
    exact_cache = current.compute_kv(items, behaviors, deltas)
    reuse_cache = parent.compute_kv(items, behaviors, deltas)
    exact_k_before = exact_cache.k.clone()
    exact_v_before = exact_cache.v.clone()
    reuse_k_before = reuse_cache.k.clone()
    reuse_v_before = reuse_cache.v.clone()
    candidates = torch.tensor([[21, 23, 25]])
    query_deltas = torch.tensor([[11.0, 12.0, 13.0]])
    expected_scores, expected_readout = current.observe_cc_reuse(
        exact_cache, candidates, query_deltas
    )

    memory = build_oracle_address_response_memory(
        exact_cache, reuse_cache, sample_count=exact_cache.seq_len
    )
    observed = intervene_oracle_signed_response_memory(
        current, reuse_cache, memory, candidates, query_deltas
    )

    assert torch.equal(memory.inverse_inclusion_probabilities, torch.ones(8))
    assert torch.allclose(observed.scores, expected_scores, rtol=2e-5, atol=3e-6)
    assert torch.allclose(observed.readout, expected_readout, rtol=2e-5, atol=3e-6)
    assert torch.equal(exact_cache.k, exact_k_before)
    assert torch.equal(exact_cache.v, exact_v_before)
    assert torch.equal(reuse_cache.k, reuse_k_before)
    assert torch.equal(reuse_cache.v, reuse_v_before)
