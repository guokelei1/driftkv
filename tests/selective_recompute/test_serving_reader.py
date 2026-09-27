import pytest
import torch

from hstu_kvcache.baselines.serving import score_one_query


@pytest.mark.parametrize("reference,diagonal", [(False, "inclusive"), (False, "exclusive"),
                                                (True, "inclusive"), (True, "exclusive")])
def test_single_query_append_only_matches_native_reader(model_factory, raw_events, reference, diagonal):
    parent = model_factory(17, reference=reference, diagonal=diagonal)
    current = model_factory(29, reference=reference, diagonal=diagonal)
    cache = parent.compute_kv(*raw_events)
    old_k, old_v = cache.k.clone(), cache.v.clone()
    candidates = torch.tensor([[9], [11]])
    delta = torch.tensor([2., 5.])
    expected = current.score_cc_reuse(cache, candidates, delta)
    current.train()
    actual = score_one_query(current, cache, candidates, delta)
    assert current.training
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert torch.equal(cache.k, old_k) and torch.equal(cache.v, old_v)
