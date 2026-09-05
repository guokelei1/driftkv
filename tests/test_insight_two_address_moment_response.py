from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hstu_kvcache.models import (
    HSTU,
    HSTUConfig,
    HSTUKVCache,
    PointwiseAttention,
    PointwiseAttentionConfig,
)
from insight_two.address_moment_response import (
    build_oracle_address_moment_memory,
    intervene_oracle_address_moment_memory,
    native_activation_and_derivative,
    read_address_moment_residual,
)


def _cache(k: torch.Tensor, v: torch.Tensor) -> HSTUKVCache:
    return HSTUKVCache(k=k, v=v, seq_len=k.shape[2])


def _model(seed: int) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=96,
            num_behaviors=3,
            hidden_size=8,
            num_layers=2,
            num_heads=2,
            head_dim=4,
            max_seq_len=8,
            temporal_num_freqs=2,
            input_dropout=0.0,
        )
    ).eval()


def test_single_cell_moments_match_manual_outer_product_formula() -> None:
    current_k = torch.tensor([[[[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]]]])
    parent_k = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]])
    current_v = torch.tensor([[[[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]]]])
    parent_v = torch.tensor([[[[0.5, 1.0], [1.0, 2.0], [3.0, 0.5]]]])
    exact = _cache(current_k, current_v)
    reuse = _cache(parent_k, parent_v)

    memory = build_oracle_address_moment_memory(
        exact, reuse, sample_count=1, num_heads=1
    )
    center = torch.cat((current_k[0, 0], parent_k[0, 0]), dim=0).mean(dim=0)
    expected_s = (current_v[0, 0] - parent_v[0, 0]).sum(dim=0)
    expected_m = sum(
        torch.outer(current_k[0, 0, i] - center, current_v[0, 0, i])
        - torch.outer(parent_k[0, 0, i] - center, parent_v[0, 0, i])
        for i in range(3)
    )

    assert memory.centers.shape == (1, 1, 1, 1, 2)
    assert memory.signed_zeroth.shape == (1, 1, 1, 1, 2)
    assert memory.signed_first.shape == (1, 1, 1, 1, 2, 2)
    assert torch.allclose(memory.centers[0, 0, 0, 0], center)
    assert torch.allclose(memory.signed_zeroth[0, 0, 0, 0], expected_s)
    assert torch.allclose(memory.signed_first[0, 0, 0, 0], expected_m)


@pytest.mark.parametrize("activation", ["elu_plus1", "relu", "silu"])
def test_current_equals_parent_has_zero_moment_residual(activation: str) -> None:
    torch.manual_seed(331)
    k = torch.randn(2, 1, 6, 8)
    v = torch.randn(2, 1, 6, 8)
    cache = _cache(k, v)
    memory = build_oracle_address_moment_memory(
        cache, cache, sample_count=3, num_heads=2
    )
    attention = PointwiseAttention(
        PointwiseAttentionConfig(
            hidden_size=8,
            num_heads=2,
            head_dim=4,
            activation=activation,
        )
    ).eval()
    q = torch.randn(5, 2, 1, 4)
    residual = read_address_moment_residual(
        attention, q, memory, layer=1, candidate_count=5
    )

    assert torch.equal(memory.signed_zeroth, torch.zeros_like(memory.signed_zeroth))
    assert torch.equal(memory.signed_first, torch.zeros_like(memory.signed_first))
    assert torch.equal(residual, torch.zeros_like(residual))


def test_coherent_intervention_preserves_input_caches() -> None:
    parent, current = _model(337), _model(347)
    items = torch.tensor([[1, 2, 3, 4, 5, 6]])
    behaviors = torch.tensor([[1, 2, 1, 2, 1, 2]])
    deltas = torch.arange(6).float().reshape(1, 6)
    exact_cache = current.compute_kv(items, behaviors, deltas)
    reuse_cache = parent.compute_kv(items, behaviors, deltas)
    exact_before = (exact_cache.k.clone(), exact_cache.v.clone())
    reuse_before = (reuse_cache.k.clone(), reuse_cache.v.clone())
    memory = build_oracle_address_moment_memory(
        exact_cache, reuse_cache, sample_count=3, num_heads=2
    )
    observed = intervene_oracle_address_moment_memory(
        current,
        reuse_cache,
        memory,
        torch.tensor([[21, 23, 25]]),
        torch.tensor([[7.0, 8.0, 9.0]]),
    )

    assert torch.isfinite(observed.scores).all()
    assert torch.isfinite(observed.readout).all()
    assert len(observed.layer_residual_heads) == 2
    assert torch.equal(exact_cache.k, exact_before[0])
    assert torch.equal(exact_cache.v, exact_before[1])
    assert torch.equal(reuse_cache.k, reuse_before[0])
    assert torch.equal(reuse_cache.v, reuse_before[1])


def test_moment_delta_is_exact_when_all_logits_share_positive_linear_region() -> None:
    current_k = torch.tensor([[[[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]]])
    parent_k = torch.tensor([[[[1.0, 0.0], [2.5, 0.0], [5.0, 0.0]]]])
    current_v = torch.tensor([[[[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]]]])
    parent_v = torch.tensor([[[[0.5, 1.0], [2.0, 3.0], [-1.0, 2.0]]]])
    exact = _cache(current_k, current_v)
    reuse = _cache(parent_k, parent_v)
    memory = build_oracle_address_moment_memory(
        exact, reuse, sample_count=1, num_heads=1
    )
    attention = PointwiseAttention(
        PointwiseAttentionConfig(
            hidden_size=2,
            num_heads=1,
            head_dim=2,
            qk_scale=2.0,
            activation="elu_plus1",
        )
    ).eval()
    q = torch.tensor([[[[1.0, 0.0]]]])
    current_logits = torch.matmul(q, current_k[0, 0].T) * attention.scale
    parent_logits = torch.matmul(q, parent_k[0, 0].T) * attention.scale
    assert torch.all(current_logits > 0) and torch.all(parent_logits > 0)
    exact_delta = (
        torch.matmul(attention._activate(current_logits), current_v[0, 0])
        - torch.matmul(attention._activate(parent_logits), parent_v[0, 0])
    )
    moment_delta = read_address_moment_residual(
        attention, q, memory, layer=0, candidate_count=1
    )

    assert torch.allclose(moment_delta, exact_delta, rtol=1e-6, atol=1e-6)


def test_native_activation_derivatives_cover_supported_kernels() -> None:
    logits = torch.tensor([-1.0, 0.0, 2.0])
    elu, elu_prime = native_activation_and_derivative("elu_plus1", logits)
    relu, relu_prime = native_activation_and_derivative("relu", logits)
    silu, silu_prime = native_activation_and_derivative("silu", logits)

    assert torch.allclose(elu, F.elu(logits) + 1.0)
    assert torch.allclose(elu_prime, torch.tensor([torch.exp(torch.tensor(-1.0)), 1.0, 1.0]))
    assert torch.equal(relu, torch.tensor([0.0, 0.0, 2.0]))
    assert torch.equal(relu_prime, torch.tensor([0.0, 0.0, 1.0]))
    sigmoid = torch.sigmoid(logits)
    assert torch.allclose(silu, F.silu(logits))
    assert torch.allclose(silu_prime, sigmoid * (1.0 + logits * (1.0 - sigmoid)))
