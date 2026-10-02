import copy

import torch
import torch.nn.functional as F

from hstu_kvcache.read_correction.history_conditioned.core import HistoryCorrection
from hstu_kvcache.read_correction.history_conditioned.v2 import HistoryCorrectionV2


def inputs():
    torch.manual_seed(37)
    return (torch.randn(3, 2, 3, 4, dtype=torch.float64),
            torch.randn(3, 7, 8, dtype=torch.float64),
            torch.randn(3, 7, 8, dtype=torch.float64), torch.tensor([7, 4, 0]))


def test_zero_residual_preserves_fitted_base_and_cache():
    q, k, v, counts = inputs()
    base = HistoryCorrection(2, 4, width=5).double()
    with torch.no_grad():
        base.weight.normal_()
        base.bias.normal_()
        base.history_weight.normal_()
    base.set_history_normalization(k, v, counts)
    saved = {name: value.clone() for name, value in base.state_dict().items()}
    original_k, original_v = k.clone(), v.clone()
    module = HistoryCorrectionV2(2, 4, base_width=5).double().initialize_history(base)
    module.set_output_scale(torch.tensor([1e-5, 1e4]))
    torch.testing.assert_close(module(q, k, v, counts), base(q, k, v, counts), rtol=0., atol=0.)
    assert torch.equal(k, original_k) and torch.equal(v, original_v)
    for name, value in base.state_dict().items():
        assert torch.equal(value, saved[name])
    assert all(not p.requires_grad for p in module.base.parameters())
    restored = HistoryCorrectionV2(**module.get_config()).double()
    restored.load_state_dict(module.state_dict())
    torch.testing.assert_close(restored(q, k, v, counts), module(q, k, v, counts))


def test_query_chunks_masks_gradients_and_history_order():
    q, k, v, counts = inputs()
    small = HistoryCorrectionV2(2, 4, encoder_width=8, query_chunk=1).double()
    small.set_history_normalization(k, v, counts)
    with torch.no_grad():
        small.output.weight.normal_(std=.2)
        small.output.bias.normal_(std=.1)
    large = copy.deepcopy(small)
    large.query_chunk = 3
    actual = small(q, k, v, counts)
    expected = large(q, k, v, counts)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for (name, left), (_, right) in zip(small.named_parameters(), large.named_parameters()):
        if left.requires_grad:
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-9, atol=1e-9, msg=name)
    # Padding cannot influence retained-token encoding, even as attention keys.
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[1, 4:] = 1e8
    changed_v[1, 4:] = -1e8
    changed_k[2] = 1e8
    changed_v[2] = 1e8
    torch.testing.assert_close(small(q, changed_k, changed_v, counts), actual, rtol=0., atol=0.)
    assert torch.equal(actual[2], torch.zeros_like(actual[2]))
    residual = small.residual_rate(q, k, v, counts)
    assert not torch.allclose(residual, small.residual_rate(q + .5, k, v, counts))
    assert not torch.allclose(residual, small.residual_rate(q, k + .5, v, counts))
    assert not torch.allclose(residual[0], small.residual_rate(q, k.flip(1), v.flip(1), counts)[0])


def test_history_attention_matches_dense_masked_reference():
    q, _, _, counts = inputs()
    module = HistoryCorrectionV2(2, 4, encoder_width=8).double()
    x = torch.randn(3, 7, 8, dtype=q.dtype)
    valid = torch.arange(7)[None] < counts[:, None]
    projected = module.qkv_projection(module.attention_norm(x))
    a, b, c = projected.reshape(3, 7, 3, 4, 2).permute(2, 0, 3, 1, 4).unbind(0)
    logits = a @ b.transpose(-1, -2) / (2 ** .5)
    masked = logits.masked_fill(~valid[:, None, None], float("-inf"))
    weights = torch.softmax(masked, dim=-1).nan_to_num(0.)
    attended = (weights @ c).transpose(1, 2).reshape_as(x)
    expected = x + module.attention_output(attended)
    expected = expected + module.feedforward_out(F.gelu(module.feedforward_in(module.feedforward_norm(expected))))
    torch.testing.assert_close(module._encode(x, valid), expected, rtol=1e-10, atol=1e-10)


def test_extra_encoder_learns_query_conditioned_user_signal():
    torch.manual_seed(41)
    q = torch.randn(10, 2, 2, 2)
    k, v = torch.randn(10, 5, 4), torch.randn(10, 5, 4)
    counts = torch.full((10,), 5)
    module = HistoryCorrectionV2(2, 2, encoder_width=8, query_chunk=2)
    module.set_history_normalization(k, v, counts)
    # Cross-head, ordered user information plus query dependence.
    user = (v[:, -1] - .3 * k[:, 0]).reshape(10, 2, 2)
    target = torch.tanh(user[:, :, None] + .2 * q)
    scale = torch.tensor([1e-4, 1e3])
    module.set_output_scale(scale)
    physical_target = target * scale[None, :, None, None]
    optimizer = torch.optim.Adam([p for p in module.parameters() if p.requires_grad], lr=.02)
    initial = float(target.square().mean())
    for step in range(80):
        optimizer.zero_grad(set_to_none=True)
        error = (module.rate(q, k, v, counts) - physical_target) / scale[None, :, None, None]
        loss = error.square().mean()
        loss.backward()
        if step == 1:
            assert module.key_projection.weight.grad.abs().sum() > 0
            assert module.qkv_projection.weight.grad.abs().sum() > 0
            assert module.query_projection.weight.grad.abs().sum() > 0
        optimizer.step()
    with torch.no_grad():
        final = float(((module.rate(q, k, v, counts) - physical_target) / scale[None, :, None, None]).square().mean())
    assert final < initial * .1
