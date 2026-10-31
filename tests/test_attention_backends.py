"""Numerical safeguards for the optional fused HSTU backend.

These compare the original PyTorch equations, including learned bias gradients,
and cache causality; CPU-only installations retain the reference implementation.
"""

import importlib.util

import pytest
import torch

from hstu_kvcache.models.attention import PointwiseAttention, PointwiseAttentionConfig

CUDA_TRITON = torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


def assert_relative(actual, expected, tolerance=1e-5):
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).norm()
    scale = expected.float().norm()
    assert error <= tolerance * scale + 1e-7, (float(error), float(scale))


def test_backend_is_execution_only_and_cpu_falls_back(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    module = PointwiseAttention(PointwiseAttentionConfig(hidden_size=64, num_heads=2))
    x = torch.randn(2, 11, 64)
    expected = module(x)
    original_state = module.state_dict()
    module.backend = "auto"
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    clone = PointwiseAttention(module.cfg)
    clone.load_state_dict(original_state, strict=True)
    clone.backend = "triton"
    torch.testing.assert_close(clone(x), expected, rtol=0, atol=0)
    assert original_state.keys() == clone.state_dict().keys()


@pytest.mark.skipif(not CUDA_TRITON, reason="CUDA and Triton required")
@pytest.mark.parametrize("variant", ["legacy", "hstu_reference"])
@pytest.mark.parametrize("diagonal", ["inclusive", "exclusive"])
def test_fused_attention_outputs_kv_and_gradients(variant, diagonal):
    torch.manual_seed(17)
    module = PointwiseAttention(PointwiseAttentionConfig(
        hidden_size=64, num_heads=2, max_seq_len=128,
        block_variant=variant, activation="silu" if variant == "hstu_reference" else "elu_plus1",
        relative_position_bias=True, causal_diagonal=diagonal,
        qk_scale=0.7,
    )).cuda()
    x = torch.randn(2, 67, 64, device="cuda", requires_grad=True)
    cotangent = torch.randn_like(x)
    snapshots = []
    for name in ("torch", "triton"):
        module.backend = name
        module.zero_grad(set_to_none=True)
        x.grad = None
        output, (k, v) = module(x, return_kv=True)
        (output * cotangent).sum().backward()
        snapshots.append((output.detach(), k.detach(), v.detach(), x.grad.clone(),
                          {key: parameter.grad.clone() for key, parameter in module.named_parameters()
                           if parameter.grad is not None}))
    for expected, actual in zip(snapshots[0][:4], snapshots[1][:4], strict=True):
        assert_relative(actual, expected)
    assert snapshots[0][4].keys() == snapshots[1][4].keys()
    for key, expected in snapshots[0][4].items():
        assert_relative(snapshots[1][4][key], expected)


@pytest.mark.skipif(not CUDA_TRITON, reason="CUDA and Triton required")
@pytest.mark.parametrize("diagonal", ["inclusive", "exclusive"])
def test_fused_stale_and_windowed_cache_attention(diagonal):
    torch.manual_seed(21)
    module = PointwiseAttention(PointwiseAttentionConfig(
        hidden_size=64, num_heads=2, max_seq_len=128,
        block_variant="hstu_reference", activation="silu",
        relative_position_bias=True, causal_diagonal=diagonal,
    )).cuda().eval()
    prefix = torch.randn(2, 31, 64, device="cuda")
    suffix = torch.randn(2, 7, 64, device="cuda")
    with torch.no_grad():
        module.backend = "torch"
        _, (k, v) = module(prefix, return_kv=True)
        # Freeze arbitrary old producer K/V once for both implementations.
        k, v = k * 1.03, v * 0.91
        expected_window, expected_kv = module.forward_with_cache(suffix, k, v, window_size=16)
        expected_stale = module.forward_stale_kv(prefix, k, v)
        module.backend = "triton"
        actual_window, actual_kv = module.forward_with_cache(suffix, k, v, window_size=16)
        actual_stale = module.forward_stale_kv(prefix, k, v)
    assert_relative(actual_window, expected_window)
    assert_relative(actual_stale, expected_stale)
    for actual, expected in zip(actual_kv, expected_kv, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not CUDA_TRITON, reason="CUDA and Triton required")
def test_custom_attention_mask_retains_reference_path():
    module = PointwiseAttention(PointwiseAttentionConfig(hidden_size=64, num_heads=2)).cuda().eval()
    x = torch.randn(2, 17, 64, device="cuda")
    mask = torch.full((17, 17), -torch.inf, device="cuda")
    mask[:, :3] = 0
    with torch.no_grad():
        module.backend = "torch"
        expected = module(x, attn_mask=mask)
        module.backend = "triton"
        actual = module(x, attn_mask=mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
