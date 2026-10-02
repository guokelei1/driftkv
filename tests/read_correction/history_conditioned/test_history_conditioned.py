import copy

import torch
import torch.nn.functional as F

from hstu_kvcache.read_correction import HistoryCorrection, QueryCorrection


def _inputs():
    torch.manual_seed(17)
    return (torch.randn(3, 2, 5, 3, dtype=torch.float64),
            torch.randn(3, 9, 6, dtype=torch.float64),
            torch.randn(3, 9, 6, dtype=torch.float64), torch.tensor([9, 6, 0]))


def _dense(module, q, k, v, counts):
    qn = module.normalize_query(q)
    kn = (module._heads(k) - module.k_mean[None, :, None]) / module.k_scale[None, :, None]
    vn = (module._heads(v) - module.v_mean[None, :, None]) / module.v_scale[None, :, None]
    q_part = torch.einsum("bhqd,hdw->bhqw", qn, module.query_projection)
    kv_part = (torch.einsum("bhnd,hdw->bhnw", kn, module.key_projection)
               + torch.einsum("bhnd,hdw->bhnw", vn, module.value_projection))
    phi = F.silu(q_part.unsqueeze(-2) + kv_part.unsqueeze(-3) + module.encoder_bias[None, :, None, None])
    valid = torch.arange(k.shape[1])[None] < counts[:, None]
    mean = (phi * valid[:, None, None, :, None]).sum(-2) / counts.clamp_min(1)[:, None, None, None]
    base = QueryCorrection.rate(module, q)
    return (base + torch.einsum("bhqw,hwd->bhqd", mean, module.history_weight)) * counts[:, None, None, None]


def test_chunked_forward_and_grad_match_dense_and_ignore_padding():
    q, k, v, counts = _inputs()
    module = HistoryCorrection(2, 3, width=7, query_chunk=2, token_chunk=4).double()
    module.set_history_normalization(k, v, counts)
    with torch.no_grad():
        module.history_weight.normal_()
    dense = copy.deepcopy(module)
    actual = module(q, k, v, counts)
    expected = _dense(dense, q, k, v, counts)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for (_, left), (_, right) in zip(module.named_parameters(), dense.named_parameters()):
        torch.testing.assert_close(left.grad, right.grad)
    k_changed, v_changed = k.clone(), v.clone()
    k_changed[1, 6:] = 100
    v_changed[1, 6:] = -100
    torch.testing.assert_close(module(q, k_changed, v_changed, counts), actual)
    assert torch.equal(actual[2], torch.zeros_like(actual[2]))


def test_history_branch_depends_on_both_query_and_user_history():
    q, k, v, counts = _inputs()
    module = HistoryCorrection(2, 3, width=7).double()
    with torch.no_grad():
        module.history_weight.normal_()
    initial = module.rate(q, k, v, counts)
    assert not torch.allclose(initial, module.rate(q + 1., k, v, counts))
    assert not torch.allclose(initial, module.rate(q, k + 1., v, counts))
    assert not torch.allclose(initial, module.rate(q, k, v + 1., counts))
    original_k, original_v = k.clone(), v.clone()
    module(q, k, v, counts)
    assert torch.equal(k, original_k) and torch.equal(v, original_v)


def test_history_learns_after_zero_output_initialization():
    torch.manual_seed(19)
    q = torch.randn(12, 1, 4, 2)
    k, v = torch.randn(12, 6, 2), torch.randn(12, 6, 2)
    counts = torch.full((12,), 6)
    # User-dependent target cannot be obtained solely from the random q.
    target = (v.mean(1)[:, None, None] + .1 * q).detach()
    base = QueryCorrection(1, 2)
    module = HistoryCorrection(1, 2, width=8, query_chunk=4, token_chunk=6)
    module.initialize_query(base).set_history_normalization(k, v, counts)
    torch.testing.assert_close(module.rate(q, k, v, counts), base.rate(q))
    optimizer = torch.optim.Adam(module.parameters(), lr=.03)
    initial = float((module.rate(q, k, v, counts) - target).square().mean().detach())
    for step in range(60):
        optimizer.zero_grad()
        loss = (module.rate(q, k, v, counts) - target).square().mean()
        loss.backward()
        if step == 1:
            assert module.key_projection.grad.abs().sum() > 0
            assert module.query_projection.grad.abs().sum() > 0
        optimizer.step()
    final = float((module.rate(q, k, v, counts) - target).square().mean().detach())
    assert final < initial * .1
