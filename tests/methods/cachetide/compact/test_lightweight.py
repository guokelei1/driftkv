"""One-attention Value codes: identity, explicit reference, fit and row alignment."""

import torch

from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.methods.cachetide.compact.lightweight import LightweightValueAdapter, LightweightValueState
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache


def _case(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(4104)
    model = HSTU(HSTUConfig(num_items=20, num_behaviors=2, hidden_size=8,
                            num_layers=2, num_heads=2, max_seq_len=5,
                            input_dropout=0., attn_dropout=0.)).eval().requires_grad_(False)
    items = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    cache = model.compute_kv(items, torch.zeros_like(items), torch.ones_like(items).float())
    features = model.lookup_item_embeddings(items)
    adapter = LightweightValueAdapter(2, 2, 4, rank=2, response_hidden_width=8, max_length=5)
    return model, adapter, cache, features


def test_single_attention_identity_reference_and_gradients(monkeypatch):
    model, adapter, cache, features = _case(monkeypatch)
    candidates, delta = torch.tensor([[11, 12], [13, 14]]), torch.tensor([1., 2.])
    original = (cache.k.clone(), cache.v.clone(), features.clone())
    native = score(model, cache, candidates, delta)[0]
    state = adapter.encode_cache(cache, features)
    actual = score(model, cache, candidates, delta, history_read_fn=adapter.make_history_read(state))[0]
    torch.testing.assert_close(actual, native, atol=0, rtol=0)

    with torch.no_grad():
        for layer in adapter.layers:
            layer.input_center.normal_(std=.2)
            layer.input_scale.uniform_(.5, 1.5)
            layer.decoder.normal_(std=.03)
        for layer in adapter.response:
            layer.output.weight.normal_(std=.01)
            layer.output.bias.normal_(std=.01)
    state = adapter.encode_cache(cache, features)
    mapped_values = []
    for index, layer in enumerate(adapter.layers):
        decoded = state.codes[index].transpose(1, 2) @ layer.decoder[None]
        mapped_values.append(cache.v[index] + decoded.transpose(1, 2).flatten(2))

    def reference(layer, q, _):
        r = history_read(model.blocks[layer].attn, q, cache.k[layer], mapped_values[layer])
        return adapter.response[layer](q, r, torch.full((q.shape[0],), cache.seq_len))

    actual, _ = score(model, cache, candidates, delta, history_read_fn=adapter.make_history_read(state))
    expected, _ = score(model, cache, candidates, delta, history_override=reference)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    variables = tuple(adapter.parameters())
    grad_actual = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
    grad_expected = torch.autograd.grad(expected.square().sum(), variables)
    for value, target in zip(grad_actual, grad_expected, strict=True):
        torch.testing.assert_close(value, target, atol=2e-6, rtol=3e-5)
        assert torch.isfinite(value).all()
    assert all(value.abs().sum() > 0 for value in grad_actual)
    assert all(parameter.grad is None for parameter in model.parameters())
    for value, target in zip((cache.k, cache.v, features), original, strict=True):
        torch.testing.assert_close(value, target, atol=0, rtol=0)


def test_codes_follow_append_eviction_without_dense_kv_copy(monkeypatch):
    model, adapter, cache, features = _case(monkeypatch)
    adapter.response = None
    state = adapter.encode_cache(cache, features)
    new_cache = HSTUKVCache(torch.randn(2, 2, 2, 8), torch.randn(2, 2, 2, 8), 2)
    new_features = torch.randn(2, 2, 8)
    new_state = adapter.encode_cache(new_cache, new_features)
    advanced = LightweightValueState(torch.cat((state.codes[:, :, 2:], new_state.codes), dim=2), 5)
    full_cache = HSTUKVCache(torch.cat((cache.k[:, :, 2:], new_cache.k), dim=2),
                             torch.cat((cache.v[:, :, 2:], new_cache.v), dim=2), 5)
    reference = adapter.encode_cache(full_cache, torch.cat((features[:, 2:], new_features), dim=1))
    torch.testing.assert_close(advanced.codes, reference.codes, atol=1e-7, rtol=2e-6)
    query = torch.randn(2, 2, 3, 4)
    read = adapter.make_history_read(advanced)(0, model.blocks[0].attn, query, full_cache.k[0], full_cache.v[0])
    native = history_read(model.blocks[0].attn, query, full_cache.k[0], full_cache.v[0])
    torch.testing.assert_close(read, native, atol=0, rtol=0)
    assert set(vars(advanced)) == {"codes", "seq_len"}
    assert advanced.storage_bytes() == advanced.codes.numel() * advanced.codes.element_size()


def test_offline_strength_gradients_and_decoder_fold(monkeypatch):
    """Six release-fixed strengths can train through queries and disappear at serve."""
    import copy

    model, adapter, cache, features = _case(monkeypatch)
    adapter.response = None
    with torch.no_grad():
        for layer in adapter.layers:
            layer.decoder.normal_(std=.07)
    adapter.eval().requires_grad_(False)
    state = adapter.encode_cache(cache, features)
    parameters = torch.nn.Parameter(torch.tensor([.3, -.2]))
    strengths = 1+torch.tanh(parameters)
    candidates, delta = torch.tensor([[11, 12], [13, 14]]), torch.tensor([1., 2.])
    actual = score(model, cache, candidates, delta,
        history_read_fn=adapter.make_history_read(state, strengths=strengths))[0]

    def explicit(layer, query, _):
        decoded = state.codes[layer].transpose(1, 2) @ adapter.layers[layer].decoder[None]
        values = cache.v[layer] + strengths[layer]*decoded.transpose(1, 2).flatten(2)
        return history_read(model.blocks[layer].attn, query, cache.k[layer], values)

    reference = score(model, cache, candidates, delta, history_override=explicit)[0]
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
    gradient = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference.square().sum(), parameters)[0]
    torch.testing.assert_close(gradient, reference_gradient, atol=2e-6, rtol=3e-5)
    assert torch.isfinite(gradient).all() and (gradient.abs() > 0).all()
    assert all(p.grad is None for module in (model, adapter) for p in module.parameters())

    folded = copy.deepcopy(adapter)
    with torch.no_grad():
        for index, layer in enumerate(folded.layers):
            layer.decoder.mul_(strengths[index])
        deployed = score(model, cache, candidates, delta,
            history_read_fn=folded.make_history_read(state))[0]
    torch.testing.assert_close(deployed, actual, atol=0, rtol=0)
