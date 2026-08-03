import torch

from hstu_kvcache.migration.low_rank import compile_low_rank_cache_adapter
from hstu_kvcache.migration.xp_residual_fit import (
    capture_embedded_normed_states,
    fit_distributed_low_rank_cache_adapter,
)
from hstu_kvcache.models import HSTU, HSTUConfig


def model() -> HSTU:
    return HSTU(
        HSTUConfig(
            num_items=32,
            num_behaviors=4,
            hidden_size=8,
            num_layers=2,
            num_heads=1,
            head_dim=8,
            max_seq_len=8,
            input_dropout=0.0,
        )
    )


def test_embedded_normed_capture_matches_manual_blocks() -> None:
    current = model().eval()
    vectors = torch.randn(3, 6, 8)
    behaviors = torch.randint(0, 4, (3, 6))
    deltas = torch.rand(3, 6)
    lengths = torch.tensor([6, 4, 2])
    captured = capture_embedded_normed_states(
        current,
        vectors,
        behaviors,
        deltas,
        lengths,
    )
    valid = torch.arange(6)[None, :] < lengths[:, None]
    hidden = current.combine_input_features(vectors, behaviors, deltas)
    hidden = hidden * valid.unsqueeze(-1)
    expected = []
    for block in current.blocks:
        expected.append(block.norm(hidden))
        hidden = block(hidden, return_kv=False) * valid.unsqueeze(-1)
    assert all(
        torch.allclose(value, reference)
        for value, reference in zip(captured, expected, strict=True)
    )


def test_residual_adapter_recovers_shared_low_rank_map() -> None:
    current = model().eval()
    generator = torch.Generator().manual_seed(13)
    features = []
    residuals = []
    for _ in current.blocks:
        x = torch.randn(128, 8, generator=generator)
        left = torch.randn(8, 2, generator=generator) * 0.1
        right = torch.randn(2, 16, generator=generator) * 0.1
        bias = torch.randn(16, generator=generator) * 0.01
        features.append(x)
        residuals.append(x @ left @ right + bias)
    adapter, metrics = fit_distributed_low_rank_cache_adapter(
        features,
        residuals,
        rank=2,
        ridge=1e-6,
        maximum_tokens_per_rank=128,
        seed=17,
        device=torch.device("cpu"),
    )
    compiled = compile_low_rank_cache_adapter(current, adapter)
    for layer, x in enumerate(features):
        baseline = torch.cat(
            (
                current.blocks[layer].attn.k_proj(x),
                current.blocks[layer].attn.v_proj(x),
            ),
            dim=-1,
        )
        predicted = x @ compiled.weights[layer] + compiled.biases[layer]
        assert torch.allclose(
            predicted,
            baseline + residuals[layer],
            atol=2e-4,
            rtol=2e-4,
        )
    assert metrics["global_sampled_tokens_per_layer"] == [128, 128]
    assert metrics["labels_used"] is False
