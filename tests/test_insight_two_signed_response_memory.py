from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from insight_two.signed_response_memory import (
    SUPPORTED_SAMPLE_COUNTS,
    build_oracle_signed_response_memory,
    fixed_midpoint_strata,
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


def test_fixed_midpoint_strata_supports_frozen_sample_grid() -> None:
    expected_width = {8: 16, 16: 8, 32: 4, 64: 2, 128: 1}
    for sample_count in SUPPORTED_SAMPLE_COUNTS:
        strata = fixed_midpoint_strata(128, sample_count)
        width = expected_width[sample_count]
        assert strata.sample_count == sample_count
        assert torch.equal(strata.stops - strata.starts, torch.full_like(strata.starts, width))
        assert torch.equal(
            strata.midpoints,
            strata.starts + (width - 1) // 2,
        )
        assert torch.equal(
            strata.inverse_inclusion_probabilities,
            torch.full((sample_count,), float(width)),
        )
    with pytest.raises(ValueError, match="one of"):
        fixed_midpoint_strata(128, 4)
    with pytest.raises(ValueError, match="equal-width"):
        fixed_midpoint_strata(130, 8)
    full_instrumentation = fixed_midpoint_strata(1024, 1024)
    assert torch.equal(full_instrumentation.midpoints, torch.arange(1024))
    assert torch.equal(
        full_instrumentation.inverse_inclusion_probabilities, torch.ones(1024)
    )


def test_oracle_memory_is_paired_signed_ipw_native_atoms() -> None:
    parent, current = _model(101), _model(103)
    items = torch.arange(1, 17).reshape(1, 16)
    behaviors = (torch.arange(16).reshape(1, 16) % 2 + 1).long()
    deltas = torch.arange(16).float().reshape(1, 16)
    exact_cache = current.compute_kv(items, behaviors, deltas)
    reuse_cache = parent.compute_kv(items, behaviors, deltas)
    exact_k_before = exact_cache.k.clone()
    reuse_v_before = reuse_cache.v.clone()

    memory = build_oracle_signed_response_memory(
        exact_cache, reuse_cache, sample_count=8
    )
    positions = torch.arange(0, 16, 2)
    assert memory.sample_count == 8
    assert memory.atom_count == 16
    assert torch.equal(memory.sample_positions, positions)
    assert torch.equal(memory.source_positions, torch.cat((positions, positions)))
    assert torch.allclose(memory.keys[:, :, :8], exact_cache.k[:, :, positions])
    assert torch.allclose(memory.keys[:, :, 8:], reuse_cache.k[:, :, positions])
    assert torch.allclose(
        memory.signed_values[:, :, :8], 2.0 * exact_cache.v[:, :, positions]
    )
    assert torch.allclose(
        memory.signed_values[:, :, 8:], -2.0 * reuse_cache.v[:, :, positions]
    )
    assert torch.equal(exact_cache.k, exact_k_before)
    assert torch.equal(reuse_cache.v, reuse_v_before)


@pytest.mark.parametrize("relative_position_bias", [False, True])
def test_full_sampling_signed_memory_reconstructs_exact_reader_path(
    relative_position_bias: bool,
) -> None:
    parent = _model(107, relative_position_bias=relative_position_bias)
    current = _model(109, relative_position_bias=relative_position_bias)
    items = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16],
        ]
    )
    behaviors = torch.tensor(
        [
            [1, 2, 1, 2, 1, 2, 1, 2],
            [2, 1, 2, 1, 2, 1, 2, 1],
        ]
    )
    deltas = torch.arange(8).float().expand(2, 8)
    exact_cache = current.compute_kv(items, behaviors, deltas)
    reuse_cache = parent.compute_kv(items, behaviors, deltas)
    reuse_k_before = reuse_cache.k.clone()
    reuse_v_before = reuse_cache.v.clone()
    candidates = torch.tensor([[21, 23, 25], [22, 24, 26]])
    query_deltas = torch.tensor([[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]])
    expected_scores, expected_readout = current.observe_cc_reuse(
        exact_cache, candidates, query_deltas
    )

    memory = build_oracle_signed_response_memory(
        exact_cache, reuse_cache, sample_count=8
    )
    observed = intervene_oracle_signed_response_memory(
        current, reuse_cache, memory, candidates, query_deltas
    )

    assert torch.allclose(observed.scores, expected_scores, rtol=2e-5, atol=2e-6)
    assert torch.allclose(observed.readout, expected_readout, rtol=2e-5, atol=2e-6)
    assert len(observed.layer_residual_heads) == len(current.blocks)
    assert observed.layer_residual_heads[0].shape == (2, 3, 2, 8)
    assert torch.equal(reuse_cache.k, reuse_k_before)
    assert torch.equal(reuse_cache.v, reuse_v_before)


def test_midpoint_ipw_is_exact_when_native_atoms_are_constant_within_strata() -> None:
    current = _model(113)
    torch.manual_seed(127)
    layers = len(current.blocks)
    batch = 1
    strata = 8
    width = current.blocks[0].attn.inner

    def _constant_pairs() -> torch.Tensor:
        return torch.randn(layers, batch, strata, width).repeat_interleave(2, dim=2)

    exact_cache = HSTUKVCache(
        k=_constant_pairs(),
        v=_constant_pairs(),
        seq_len=16,
    )
    reuse_cache = HSTUKVCache(
        k=_constant_pairs(),
        v=_constant_pairs(),
        seq_len=16,
    )
    candidates = torch.tensor([[31, 33, 35, 37]])
    query_delta = torch.tensor([19.0])
    expected_scores, expected_readout = current.observe_cc_reuse(
        exact_cache, candidates, query_delta
    )
    memory = build_oracle_signed_response_memory(
        exact_cache, reuse_cache, sample_count=8
    )
    observed = intervene_oracle_signed_response_memory(
        current, reuse_cache, memory, candidates, query_delta
    )

    assert torch.allclose(observed.scores, expected_scores, rtol=2e-5, atol=3e-6)
    assert torch.allclose(observed.readout, expected_readout, rtol=2e-5, atol=3e-6)
