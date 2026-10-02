"""The new read maps preserve identity, physical units and native storage."""

from io import BytesIO

import torch
from torch.nn import functional as F

from hstu_kvcache.adaptation.reader import history_read
from hstu_kvcache.design_one.nonlinear import (
    NonlinearResponseAdapter, KVReadViewAdapter, ContextKVReadViewAdapter,
    append_context, append_context_flops,
)
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache


def _restore(adapter):
    saved = BytesIO()
    torch.save(adapter.export_state(), saved)
    saved.seek(0)
    return type(adapter).from_state_dict(torch.load(saved, weights_only=True))


def test_response_identity_cross_head_formula_and_checkpoint():
    torch.manual_seed(1821)
    adapter = NonlinearResponseAdapter(2, 2, 3, hidden_width=5).double()
    query, native = torch.randn(2, 2, 4, 3, dtype=torch.double), torch.randn(2, 2, 4, 3, dtype=torch.double)
    counts = torch.tensor([2, 7])
    torch.testing.assert_close(adapter.make_history_override(counts)(0, query, native),
                               native, atol=0, rtol=0)
    layer = adapter.layers[0]
    layer.set_normalization(torch.randn(12, dtype=torch.double),
                            torch.rand(12, dtype=torch.double) + .5,
                            torch.rand(6, dtype=torch.double) + .2)
    with torch.no_grad():
        layer.output.weight.normal_(std=.03)
        layer.output.bias.normal_(std=.03)
    inputs = torch.cat((query.transpose(1, 2).flatten(2),
                        native.transpose(1, 2).flatten(2) / counts[:, None, None]), -1)
    hidden = F.silu(F.linear((inputs - layer.input_center) / layer.input_scale,
                             layer.input.weight, layer.input.bias))
    rate = F.linear(hidden, layer.output.weight, layer.output.bias) * layer.output_scale
    expected = native + (rate * counts[:, None, None]).reshape(2, 4, 2, 3).transpose(1, 2)
    callback = adapter.make_history_override(counts, fitted_layers=1)
    torch.testing.assert_close(callback(0, query, native), expected, atol=1e-14, rtol=1e-14)
    assert callback(1, query, native) is native
    restored = _restore(adapter)
    torch.testing.assert_close(restored.make_history_override(counts)(0, query, native), expected)
    # Both trainable matrices receive gradients through the normalized actual
    # query/response, with source tensors treated as frozen calibration inputs.
    loss = (layer.delta_rate(query, native, counts) - .1).square().mean()
    loss.backward()
    for parameter in layer.parameters():
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_kv_view_is_separate_row_local_and_differentiable(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(1822)
    adapter = KVReadViewAdapter(2, 2, 3, hidden_width=4, max_length=5).double()
    cache = HSTUKVCache(torch.randn(2, 2, 5, 6, dtype=torch.double),
                        torch.randn(2, 2, 5, 6, dtype=torch.double), 5)
    original_k, original_v = cache.k.clone(), cache.v.clone()
    identity = adapter.map_cache(cache)
    torch.testing.assert_close(identity.k, cache.k, atol=0, rtol=0)
    torch.testing.assert_close(identity.v, cache.v, atol=0, rtol=0)
    assert identity.k.data_ptr() != cache.k.data_ptr()
    with torch.no_grad():
        for layer in adapter.layers:
            layer.output.weight.normal_(std=.03)
            layer.output.bias.normal_(std=.03)
    mapped = adapter.map_cache(cache)
    # An eviction and a suffix publication can reuse every retained mapped row.
    tail = HSTUKVCache(cache.k[:, :, 3:], cache.v[:, :, 3:], 2)
    mapped_tail = adapter.map_cache(tail)
    torch.testing.assert_close(mapped_tail.k, mapped.k[:, :, 3:])
    torch.testing.assert_close(mapped_tail.v, mapped.v[:, :, 3:])
    restored = _restore(adapter).map_cache(cache)
    torch.testing.assert_close(restored.k, mapped.k)
    torch.testing.assert_close(restored.v, mapped.v)
    torch.testing.assert_close(cache.k, original_k, atol=0, rtol=0)
    torch.testing.assert_close(cache.v, original_v, atol=0, rtol=0)

    model = HSTU(HSTUConfig(num_items=12, num_behaviors=2, hidden_size=6,
                            num_layers=2, num_heads=2, max_seq_len=5,
                            input_dropout=0., attn_dropout=0.)).double().eval()
    query = torch.randn(2, 2, 3, 3, dtype=torch.double) * .2
    native = history_read(model.blocks[0].attn, query, cache.k[0], cache.v[0])
    prefix = adapter.map_cache(cache, fitted_layers=1)
    override = adapter.make_history_override(model, prefix)
    expected = history_read(model.blocks[0].attn, query, mapped.k[0], mapped.v[0])
    torch.testing.assert_close(override(0, query, native), expected)
    assert override(1, query, native) is native
    assert adapter.make_history_override(model, adapter.map_cache(cache, fitted_layers=0))(
        0, query, native) is native

    # A finite difference crosses both the nonlinear K/V map and the actual
    # ELU+1 history read, which is the KV candidate's fitting objective.
    def objective():
        k, v = adapter.layers[0].map_tokens(cache.k[0], cache.v[0])
        return (history_read(model.blocks[0].attn, query, k, v) - native).square().mean()

    objective().backward()
    parameter = adapter.layers[0].input.weight
    analytic = parameter.grad[0, 0].item()
    epsilon = 1e-5
    with torch.no_grad():
        parameter[0, 0] += epsilon
        above = objective().item()
        parameter[0, 0] -= 2 * epsilon
        below = objective().item()
        parameter[0, 0] += epsilon
    assert abs(analytic) > 1e-9
    torch.testing.assert_close(torch.tensor(analytic), torch.tensor((above - below) / (2 * epsilon)),
                               atol=1e-8, rtol=1e-4)


def test_write_context_matches_sequential_inclusive_eviction_and_keeps_birth_values():
    context = torch.tensor([[[1., 1.], [1., 1.], [1., 1.]],
                            [[1., 1.], [0., .5], [0., 1/3]]])
    before = context.clone()
    batched = append_context(context, old_length=3, width=2, max_length=4)
    reference = context
    for _ in range(2):
        reference = append_context(reference, reference.shape[1], 1, 4)
    torch.testing.assert_close(batched, reference, atol=0, rtol=0)
    torch.testing.assert_close(batched[:, :2], context[:, 1:], atol=0, rtol=0)
    torch.testing.assert_close(batched[:, -2:, 1], torch.tensor([[.75, .5], [.25, 0.]]))
    torch.testing.assert_close(context, before, atol=0, rtol=0)
    renewed = append_context(batched, old_length=4, width=3, max_length=4)
    assert torch.count_nonzero(renewed[:, :, 0]) == 0
    # New Current rows can retain old-state influence after every Parent row
    # is gone. They keep the fraction from their own birth, not today's zero.
    torch.testing.assert_close(renewed[0, :, 1], torch.tensor([.5, .25, 0., 0.]))
    assert append_context(renewed, 4, 0, 4) is renewed
    assert append_context_flops(batch=2, old_length=4, width=0, max_length=4) == 0


def test_context_map_uses_both_fields_and_preserves_zero_identity_and_checkpoint():
    torch.manual_seed(1823)
    adapter = ContextKVReadViewAdapter(1, 1, 2, hidden_width=3, max_length=4).double()
    layer = adapter.layers[0]
    cache = HSTUKVCache(torch.randn(1, 1, 3, 2, dtype=torch.double),
                        torch.randn(1, 1, 3, 2, dtype=torch.double), 3)
    context = torch.tensor([[[1., 1.], [0., .75], [0., .25]]], dtype=torch.double)
    identity = adapter.map_cache(cache, context)
    torch.testing.assert_close(identity.k, cache.k, atol=0, rtol=0)
    torch.testing.assert_close(identity.v, cache.v, atol=0, rtol=0)
    layer.set_normalization(torch.randn(6, dtype=torch.double),
                            torch.rand(6, dtype=torch.double) + .5,
                            torch.rand(4, dtype=torch.double) + .5)
    with torch.no_grad():
        layer.output.weight.normal_(std=.1)
        layer.output.bias.normal_(std=.03)
    source = torch.cat((cache.k[0], cache.v[0]), -1)
    features = torch.cat((source, context), -1)
    hidden = F.silu(F.linear((features-layer.input_center)/layer.input_scale,
                             layer.input.weight, layer.input.bias))
    expected = source + F.linear(hidden, layer.output.weight, layer.output.bias)*layer.output_scale
    actual = adapter.map_cache(cache, context)
    torch.testing.assert_close(torch.cat((actual.k[0], actual.v[0]), -1), expected)
    loaded = _restore(adapter).map_cache(cache, context)
    torch.testing.assert_close(loaded.k, actual.k)
    torch.testing.assert_close(loaded.v, actual.v)
    default = adapter.map_cache(cache)
    explicit_parent = adapter.map_cache(cache, torch.ones_like(context))
    torch.testing.assert_close(default.k, explicit_parent.k)
    torch.cat((actual.k[0], actual.v[0]), -1).square().mean().backward()
    assert torch.all(layer.input.weight.grad[:, -2:].abs().sum(0) > 0)
    assert adapter.estimate_flops(tokens=3)["token_transform"] > KVReadViewAdapter(
        1, 1, 2, hidden_width=3).estimate_flops(tokens=3)["token_transform"]
