"""CPU references for the executable tail, not oracle target-KV splices."""

import torch

from hstu_kvcache.baselines.tail_recompute import recompute_tail
from hstu_kvcache.models import HSTU, HSTUConfig, retain_latest_cache, truncate_cache


def model(seed):
    torch.manual_seed(seed)
    return HSTU(HSTUConfig(
        num_items=32, num_behaviors=3, hidden_size=12, num_layers=6,
        num_heads=3, max_seq_len=32, input_dropout=0.2,
    )).eval()


def events():
    return (
        torch.tensor([[1, 4, 2, 8, 3, 6, 7, 9]]),
        torch.tensor([[1, 2, 1, 2, 2, 1, 1, 2]]),
        torch.tensor([[0., 7., 11., 2., 19., 3., 5., 13.]]),
    )


def assert_cache_close(actual, expected):
    assert actual.seq_len == expected.seq_len
    torch.testing.assert_close(actual.k, expected.k, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual.v, expected.v, atol=2e-6, rtol=2e-5)


def test_tail_matches_sequential_replay_with_original_temporal_boundary():
    parent, current = model(17), model(23)
    raw = events()
    old = parent.compute_kv(*raw)
    saved_k, saved_v = old.k.clone(), old.v.clone()
    current.train()  # The public migration must still be deterministic.
    migrated = recompute_tail(current, old, *raw, n=3)
    reference = truncate_cache(old, 5)
    for i in range(5, 8):
        _, reference = current.forward_with_cache(reference, *(x[:, i:i+1] for x in raw))
    assert_cache_close(migrated, reference)
    assert current.training
    torch.testing.assert_close(migrated.k[:, :, :5], saved_k[:, :, :5], rtol=0, atol=0)
    torch.testing.assert_close(migrated.v[:, :, :5], saved_v[:, :, :5], rtol=0, atol=0)
    assert torch.equal(old.k, saved_k) and torch.equal(old.v, saved_v)


def test_tail_endpoints_and_same_model_static_reference():
    parent, current = model(17), model(23)
    raw = events()
    old = parent.compute_kv(*raw)
    assert recompute_tail(current, old, *raw, n=0) is old
    assert_cache_close(recompute_tail(current, old, *raw, n=20), current.compute_kv(*raw))
    assert_cache_close(recompute_tail(parent, old, *raw, n=3), old)


def test_second_release_uses_state_after_actual_append_and_eviction():
    parent, first, second = model(17), model(23), model(31)
    raw = events()
    state = parent.compute_kv(*(x[:, :6] for x in raw))
    state = recompute_tail(first, state, *(x[:, :6] for x in raw), n=2)
    state = retain_latest_cache(state, 4)
    _, state = first.forward_with_cache(state, *(x[:, 6:] for x in raw))
    # Retained events are now original positions 2..7; do not reset their deltas.
    migrated = recompute_tail(second, state, *(x[:, 2:] for x in raw), n=2)
    reference = truncate_cache(state, 4)
    for i in range(6, 8):
        _, reference = second.forward_with_cache(reference, *(x[:, i:i+1] for x in raw))
    assert_cache_close(migrated, reference)
