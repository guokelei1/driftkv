"""Compare batched arbitrary-position replay to native one-event replay."""

import pytest
import torch

from hstu_kvcache.baselines.sparse_recompute import recompute_positions, select_top_positions
from hstu_kvcache.baselines.tail_recompute import recompute_tail
from hstu_kvcache.models import HSTUKVCache


def scalar_reference(current, cache, raw, positions):
    k, v = cache.k.clone(), cache.v.clone()
    for batch, selected in enumerate(positions.tolist()):
        for position in selected:
            prefix = HSTUKVCache(k[:, batch:batch + 1, :position],
                                v[:, batch:batch + 1, :position], position)
            event = tuple(x[batch:batch + 1, position:position + 1] for x in raw)
            _, updated = current.forward_with_cache(prefix, *event)
            k[:, batch:batch + 1, position:position + 1] = updated.k[:, :, -1:]
            v[:, batch:batch + 1, position:position + 1] = updated.v[:, :, -1:]
    return HSTUKVCache(k, v, cache.seq_len)


@pytest.mark.parametrize("reference,diagonal", [(False, "inclusive"), (False, "exclusive"),
                                                (True, "inclusive"), (True, "exclusive")])
def test_sparse_matches_native_event_replay_and_preserves_unselected(
        model_factory, raw_events, reference, diagonal):
    parent = model_factory(17, reference=reference, diagonal=diagonal)
    current = model_factory(29, reference=reference, diagonal=diagonal)
    cache = parent.compute_kv(*raw_events)
    original_k, original_v = cache.k.clone(), cache.v.clone()
    positions = torch.tensor([[0, 2, 5], [1, 3, 4]])
    current.train()
    actual = recompute_positions(current, cache, *raw_events, positions, query_chunk_size=2)
    assert current.training
    expected = scalar_reference(current, cache, raw_events, positions)
    torch.testing.assert_close(actual.k, expected.k, atol=2e-6, rtol=3e-5)
    torch.testing.assert_close(actual.v, expected.v, atol=2e-6, rtol=3e-5)
    for batch in range(2):
        untouched = [p for p in range(cache.seq_len) if p not in positions[batch]]
        assert torch.equal(actual.k[:, batch, untouched], original_k[:, batch, untouched])
        assert torch.equal(actual.v[:, batch, untouched], original_v[:, batch, untouched])
    assert torch.equal(cache.k, original_k) and torch.equal(cache.v, original_v)


def test_sparse_endpoints_and_tail_equivalence(model_factory, raw_events):
    parent, current = model_factory(17), model_factory(29)
    cache = parent.compute_kv(*raw_events)
    empty = torch.empty((2, 0), dtype=torch.long)
    assert recompute_positions(current, cache, *raw_events, empty) is cache
    positions = torch.arange(cache.seq_len)[None].expand(2, -1)
    actual = recompute_positions(current, cache, *raw_events, positions)
    full = current.compute_kv(*raw_events)
    torch.testing.assert_close(actual.k, full.k)
    torch.testing.assert_close(actual.v, full.v)
    tail = recompute_tail(current, cache, *raw_events, n=2)
    selected_tail = recompute_positions(current, cache, *raw_events, positions[:, -2:])
    torch.testing.assert_close(selected_tail.k, tail.k)
    torch.testing.assert_close(selected_tail.v, tail.v)


def test_score_ties_are_deterministic_and_replay_order_is_chronological():
    scores = torch.tensor([[1., 1., 2., 2., 0.], [0., 3., 0., 1., 4.]])
    assert select_top_positions(scores, 3).tolist() == [[0, 2, 3], [1, 3, 4]]
