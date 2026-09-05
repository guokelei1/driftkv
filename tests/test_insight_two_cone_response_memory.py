from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from insight_two.cone_response_memory import (
    REQUIRED_ANCHOR_COUNT,
    build_cone_response_memory,
    build_layer_signed_cone_moment,
    intervene_cone_response_memory,
)


def _model(
    seed: int,
    *,
    activation: str = "elu_plus1",
    relative_bias: bool = False,
    num_layers: int = 2,
) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=128,
            num_behaviors=3,
            hidden_size=16,
            num_layers=num_layers,
            num_heads=2,
            head_dim=8,
            max_seq_len=16,
            temporal_num_freqs=2,
            input_dropout=0.0,
            activation=activation,
            relative_position_bias=relative_bias,
        )
    ).eval()


def _anchors() -> tuple[torch.Tensor, torch.Tensor]:
    candidates = torch.arange(32, 32 + REQUIRED_ANCHOR_COUNT).reshape(1, -1)
    times = torch.linspace(20.0, 21.0, REQUIRED_ANCHOR_COUNT).reshape(1, -1)
    return candidates, times


def _cache_pair(parent: HSTU, current: HSTU) -> tuple[HSTUKVCache, HSTUKVCache]:
    items = torch.arange(1, 9).reshape(1, 8)
    behaviors = (torch.arange(8).reshape(1, 8) % 2 + 1).long()
    deltas = torch.arange(8).float().reshape(1, 8)
    return current.compute_kv(items, behaviors, deltas), parent.compute_kv(
        items, behaviors, deltas
    )


def test_full_layer_moment_is_exact_for_shared_positive_cone() -> None:
    model = _model(401)
    attention = model.blocks[0].attn
    history = 7
    # Every anchor and both versions occupy the all-positive cone.
    anchor_q = torch.ones(REQUIRED_ANCHOR_COUNT, 2, 1, 8)
    exact_k = torch.rand(1, history, 16) + 0.2
    reuse_k = torch.rand(1, history, 16) + 0.2
    exact_v = torch.randn(1, history, 16)
    reuse_v = torch.randn(1, history, 16)
    moment = build_layer_signed_cone_moment(
        attention, anchor_q, exact_k, exact_v, reuse_k, reuse_v
    )

    q = torch.rand(5, 2, 1, 8) + 0.2
    exact_kh = exact_k.view(1, history, 2, 8).transpose(1, 2).expand(5, -1, -1, -1)
    reuse_kh = reuse_k.view(1, history, 2, 8).transpose(1, 2).expand(5, -1, -1, -1)
    exact_vh = exact_v.view(1, history, 2, 8).transpose(1, 2).expand(5, -1, -1, -1)
    reuse_vh = reuse_v.view(1, history, 2, 8).transpose(1, 2).expand(5, -1, -1, -1)
    exact_response = torch.matmul(
        torch.nn.functional.elu(torch.matmul(q, exact_kh.transpose(-2, -1)) * attention.scale) + 1,
        exact_vh,
    )
    reuse_response = torch.matmul(
        torch.nn.functional.elu(torch.matmul(q, reuse_kh.transpose(-2, -1)) * attention.scale) + 1,
        reuse_vh,
    )
    affine = moment.base.expand(5, -1, -1).unsqueeze(2) + attention.scale * torch.einsum(
        "bhqk,bhkv->bhqv", q, moment.linear.expand(5, -1, -1, -1)
    )

    assert moment.current_positive_mask.all()
    assert moment.parent_positive_mask.all()
    assert torch.allclose(affine, exact_response - reuse_response, atol=3e-5, rtol=3e-5)


def test_sampling_every_position_with_unit_weight_equals_full_moment() -> None:
    model = _model(403)
    attention = model.blocks[0].attn
    generator = torch.Generator().manual_seed(405)
    history = 9
    anchor_q = torch.randn(REQUIRED_ANCHOR_COUNT, 2, 1, 8, generator=generator)
    exact_k = torch.randn(1, history, 16, generator=generator)
    exact_v = torch.randn(1, history, 16, generator=generator)
    reuse_k = torch.randn(1, history, 16, generator=generator)
    reuse_v = torch.randn(1, history, 16, generator=generator)
    full = build_layer_signed_cone_moment(
        attention, anchor_q, exact_k, exact_v, reuse_k, reuse_v
    )
    sampled = build_layer_signed_cone_moment(
        attention,
        anchor_q,
        exact_k,
        exact_v,
        reuse_k,
        reuse_v,
        current_sample_positions=torch.arange(history),
        current_sample_weights=torch.ones(history),
    )

    assert full.uses_full_current
    assert sampled.uses_full_current
    assert torch.equal(sampled.base, full.base)
    assert torch.equal(sampled.linear, full.linear)


def test_identical_current_parent_produce_zero_full_signed_moment() -> None:
    model = _model(407)
    attention = model.blocks[0].attn
    generator = torch.Generator().manual_seed(409)
    history = 6
    anchor_q = torch.randn(REQUIRED_ANCHOR_COUNT, 2, 1, 8, generator=generator)
    k = torch.randn(1, history, 16, generator=generator)
    v = torch.randn(1, history, 16, generator=generator)
    moment = build_layer_signed_cone_moment(attention, anchor_q, k, v, k, v)

    assert torch.count_nonzero(moment.base) == 0
    assert torch.count_nonzero(moment.linear) == 0
    assert torch.equal(moment.current_positive_mask, moment.parent_positive_mask)


def test_model_build_and_coherent_intervention_preserve_input_caches() -> None:
    parent, current = _model(411, num_layers=6), _model(413, num_layers=6)
    exact, reuse = _cache_pair(parent, current)
    exact_before = (exact.k.clone(), exact.v.clone())
    reuse_before = (reuse.k.clone(), reuse.v.clone())
    anchors, anchor_times = _anchors()
    memory = build_cone_response_memory(
        current, exact, reuse, anchors, anchor_times
    )
    candidates = torch.tensor([[81, 83, 85]])
    result = intervene_cone_response_memory(
        current, reuse, memory, candidates, torch.tensor([25.0])
    )

    assert len(memory.layers) == len(current.blocks) == 6
    assert memory.anchor_count == REQUIRED_ANCHOR_COUNT
    assert memory.stored_scalars == sum(layer.stored_scalars for layer in memory.layers)
    assert 0 < memory.storage_ratio_to_current_kv < 1
    assert result.scores.shape == (1, 3)
    assert result.readout.shape == (1, 3, current.cfg.hidden_size)
    assert len(result.layer_signed_heads) == len(current.blocks)
    assert torch.equal(exact.k, exact_before[0])
    assert torch.equal(exact.v, exact_before[1])
    assert torch.equal(reuse.k, reuse_before[0])
    assert torch.equal(reuse.v, reuse_before[1])


def test_rejects_unsupported_attention_semantics() -> None:
    anchors, times = _anchors()
    for model in (_model(419, activation="relu"), _model(421, relative_bias=True)):
        exact, reuse = _cache_pair(model, model)
        with pytest.raises(ValueError):
            build_cone_response_memory(model, exact, reuse, anchors, times)
    training_model = _model(423).train()
    exact, reuse = _cache_pair(training_model.eval(), training_model)
    training_model.train()
    with pytest.raises(ValueError, match="eval"):
        build_cone_response_memory(training_model, exact, reuse, anchors, times)
