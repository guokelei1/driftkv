from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]

import torch

from hstu_kvcache.models import PointwiseAttention, PointwiseAttentionConfig
from hstu_kvcache.methods.cachetide.adapter import UnifiedRelayAdapter
from hstu_kvcache.methods.cachetide.calibration import freeze_query_gates


def test_constant_gate_weights_stay_zero_while_reader_and_update_learn():
    torch.manual_seed(17)
    adapter = UnifiedRelayAdapter(1, mode="A", xi_width=1,
        read_shared=3, read_private=3, gate_normalization="rms", gate_source="layer")
    layer = adapter.layers[0]
    with torch.no_grad():
        layer.key_gate_weight.normal_()
        layer.value_gate_weight.normal_()
        layer.key_decoder.normal_(std=.02)
        layer.decoder.normal_(std=.02)
        for j in range(len(layer.branches)):
            getattr(layer, f"native_basis_{j}").normal_(std=.02)
    frozen_before = {name: value.clone() for name, value in adapter.state_dict().items()
                     if "gate_weight" not in name}
    freeze_query_gates(adapter)
    for name, value in frozen_before.items():
        torch.testing.assert_close(adapter.state_dict()[name], value, rtol=0, atol=0)
    assert layer.key_decoder.requires_grad and layer.decoder.requires_grad
    assert all(p.requires_grad for p in layer.native_update)
    assert layer.item_update_weight.requires_grad
    assert not layer.key_gate_weight.requires_grad and not layer.value_gate_weight.requires_grad
    q = torch.randn(1, 6, 16, 32)
    for key, value in (layer.query_gates(q), layer.query_gates(q * 7 + 3)):
        torch.testing.assert_close(key, torch.ones_like(key), rtol=0, atol=0)
        torch.testing.assert_close(value, torch.ones_like(value), rtol=0, atol=0)

    # Exercise both independently trainable components through a real code
    # update and score/value projection, then take an optimizer step.
    k, v = torch.randn(1, 5, 192), torch.randn(1, 5, 192)
    previous, item = torch.randn(1, 5, 6, 8), torch.randn(1, 5, 192)
    code = layer.advance(k, v, previous, item)
    accessible = layer.accessible_codes(code.reshape(1, 5, 6, 8))
    gate_k, gate_v = layer.query_gates(q)
    dk, dv = layer.key_decoder[:, layer.read_slots], layer.decoder[:, layer.read_slots]
    projected = (q @ dk.transpose(-2, -1)[None]) * gate_k
    score = projected @ accessible.transpose(-2, -1)
    decoded = (accessible.mean(2)[:, :, None] * gate_v) @ dv[None]
    loss = score.square().mean() + decoded.square().mean()
    optimizer = torch.optim.Adam([p for p in adapter.parameters() if p.requires_grad], lr=.001)
    optimizer.zero_grad(set_to_none=True); loss.backward()
    assert layer.key_decoder.grad is not None and layer.key_decoder.grad.abs().sum() > 0
    assert layer.decoder.grad is not None and layer.decoder.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.native_update)
    assert layer.item_update_weight.grad is not None and layer.item_update_weight.grad.abs().sum() > 0
    optimizer.step()
    assert layer.key_gate_weight.grad is None and layer.value_gate_weight.grad is None
    assert layer.key_gate_weight.count_nonzero() == 0 and layer.value_gate_weight.count_nonzero() == 0

    # A single fixed effective K/V pair must represent ALL of these queries.
    attention = PointwiseAttention(PointwiseAttentionConfig(192, 6, head_dim=32)).eval()
    codes = code.detach().reshape(1, 5, 6, 8)
    z = layer.accessible_codes(codes)
    dk, dv = layer.key_decoder[:, layer.read_slots], layer.decoder[:, layer.read_slots]
    keys = k.reshape(1, 5, 6, 32).transpose(1, 2) + z @ dk[None]
    values = v.reshape(1, 5, 6, 32).transpose(1, 2) + z @ dv[None]
    for queries in (q * .1, (q + 2) * .1):
        actual = layer.read(attention, queries, k, v, codes)
        expected = attention._activate((queries @ keys.transpose(-2, -1)) * attention.scale) @ values
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
