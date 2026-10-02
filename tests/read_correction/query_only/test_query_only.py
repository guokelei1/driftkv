import pytest
import torch

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.read_correction import QueryCorrection, build_correction, fit_query, score_corrected


def test_ridge_matches_independent_augmented_reference():
    torch.manual_seed(17)
    q = torch.randn(7, 2, 5, 3, dtype=torch.float64)
    target = torch.randn_like(q)
    module, stats = fit_query(q, target, torch.full((7,), 11), ridge=0.03)
    expected = []
    for head in range(2):
        x = q[:, head].reshape(-1, 3)
        y = target[:, head].reshape(-1, 3)
        x = (x - x.mean(0)) / x.std(0, correction=0)
        x = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], dim=1)
        regularizer = torch.diag(torch.tensor([0.03, 0.03, 0.03, 0.], dtype=x.dtype))
        weights = torch.linalg.solve(x.T @ x / len(x) + regularizer, x.T @ y / len(x))
        expected.append((x @ weights).reshape(7, 5, 3))
    torch.testing.assert_close(module.rate(q), torch.stack(expected, dim=1))
    rebuilt = build_correction(module.kind, module.get_config(), module.state_dict())
    torch.testing.assert_close(rebuilt.rate(q), module.rate(q))
    assert stats["solve_dtype"] == "float64"


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("diagonal", ["inclusive", "exclusive"])
@torch.no_grad()
def test_zero_correction_matches_native_and_does_not_change_cache(reference, diagonal):
    def model(seed):
        torch.manual_seed(seed)
        return HSTU(HSTUConfig(
            num_items=32, num_behaviors=3, hidden_size=12, num_heads=3,
            num_layers=2, max_seq_len=32, input_dropout=0.,
            block_variant="hstu_reference" if reference else "legacy",
            activation="silu" if reference else "elu_plus1", causal_diagonal=diagonal,
        )).eval()
    parent, current = model(3), model(7)
    items = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    cache = parent.compute_kv(items, torch.ones_like(items), torch.zeros_like(items).float())
    before_k, before_v = cache.k.clone(), cache.v.clone()
    candidates = torch.tensor([[5, 6, 7], [8, 9, 10]])
    delta = torch.tensor([10., 12.])
    modules = [QueryCorrection(3, 4) for _ in range(2)]
    actual, trace = score_corrected(current, cache, candidates, delta, modules, torch.tensor([4, 4]), trace=True)
    expected = current.score_cc_reuse(cache, candidates, delta)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert torch.equal(cache.k, before_k) and torch.equal(cache.v, before_v)
    assert len(trace.queries) == len(trace.corrections) == 2
    assert not any(bool(value.count_nonzero()) for value in trace.corrections)


def test_same_query_target_affine_fit_recovers_known_correction():
    torch.manual_seed(23)
    q = torch.randn(24, 2, 8, 3, dtype=torch.float64)
    weight = torch.randn(2, 3, 3, dtype=q.dtype)
    bias = torch.randn(2, 3, dtype=q.dtype)
    target = torch.einsum("bhqd,hde->bhqe", q, weight) + bias[None, :, None]
    counts = torch.arange(1, 25)
    module, _ = fit_query(q, target, counts, ridge=0.)
    torch.testing.assert_close(module(q, None, None, counts), target * counts[:, None, None, None])
