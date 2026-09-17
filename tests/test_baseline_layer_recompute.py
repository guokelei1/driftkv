import torch

from hstu_kvcache.baselines.layer_recompute import (
    append,
    capture_state,
    profile_intervals,
    recompute_interval,
    retain_latest,
)
from hstu_kvcache.models import HSTU, HSTUConfig


def model(seed=11):
    torch.manual_seed(seed)
    return HSTU(HSTUConfig(
        num_items=32, num_behaviors=3, hidden_size=16, num_layers=6,
        num_heads=2, max_seq_len=32, input_dropout=0.2, attn_dropout=0.1,
        block_variant="legacy", activation="elu_plus1", causal_diagonal="inclusive",
    )).eval()


def events():
    return (
        torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]),
        torch.tensor([[1, 2, 1, 2, 1, 2], [2, 1, 2, 1, 2, 1]]),
        torch.tensor([[0., 2., 3., 1., 4., 2.], [0., 1., 5., 2., 1., 3.]]),
    )


def assert_cache(actual, expected):
    torch.testing.assert_close(actual.k, expected.k)
    torch.testing.assert_close(actual.v, expected.v)
    assert actual.seq_len == expected.seq_len


def test_capture_and_same_model_replay_match_full_and_restore_mode():
    current = model()
    raw = events()
    full = current.compute_kv(*raw)
    with torch.no_grad():
        x = current.embed_inputs(*raw)
        expected_inputs = {}
        for layer, block in enumerate(current.blocks):
            if layer:
                expected_inputs[layer] = x.clone()
            x = block(x)
    current.train()
    state = capture_state(current, *raw)
    assert current.training
    assert_cache(state.cache, full)
    for layer, expected in expected_inputs.items():
        torch.testing.assert_close(state.boundary_inputs[layer], expected)
        assert not state.boundary_inputs[layer].requires_grad
        assert not current.blocks[layer].norm._forward_pre_hooks

    assert recompute_interval(current, state, *raw, None) is state
    for interval in ((2, 4), (0, 5)):
        refreshed = recompute_interval(current, state, *raw, interval)
        assert_cache(refreshed.cache, full)
        assert current.training


def test_cross_version_replay_uses_saved_entry_and_preserves_unselected_state():
    parent, current = model(11), model(19)
    raw = events()
    state = capture_state(parent, *raw)
    old_k, old_v = state.cache.k.clone(), state.cache.v.clone()
    old_inputs = {layer: value.clone() for layer, value in state.boundary_inputs.items()}
    refreshed = recompute_interval(current, state, *raw, (2, 3))

    with torch.no_grad():
        entry = old_inputs[2]
        after_two, (k_two, v_two) = current.blocks[2](entry, return_kv=True)
        _, (k_three, v_three) = current.blocks[3](after_two, return_kv=True)
    for layer, k, v in ((2, k_two, v_two), (3, k_three, v_three)):
        torch.testing.assert_close(refreshed.cache.k[layer], k)
        torch.testing.assert_close(refreshed.cache.v[layer], v)
    torch.testing.assert_close(refreshed.boundary_inputs[2], entry)
    torch.testing.assert_close(refreshed.boundary_inputs[3], after_two)
    for layer in (0, 1, 4, 5):
        assert torch.equal(refreshed.cache.k[layer], old_k[layer])
        assert torch.equal(refreshed.cache.v[layer], old_v[layer])
    for layer in (1, 4, 5):
        assert torch.equal(refreshed.boundary_inputs[layer], old_inputs[layer])
    assert torch.equal(state.cache.k, old_k)
    assert torch.equal(state.cache.v, old_v)

    # Replaying every layer starts from current embeddings, not parent E_0.
    assert_cache(recompute_interval(current, state, *raw, (0, 5)).cache,
                 current.compute_kv(*raw))


def test_append_eviction_and_second_release_use_actual_boundary_history():
    parent, current, next_model = model(11), model(19), model(23)
    raw = events()
    state = capture_state(parent, *raw)
    state = recompute_interval(current, state, *raw, (2, 3))
    state = retain_latest(state, 3)  # Evict before the new event, cap=4.
    retained_inputs = {layer: value.clone() for layer, value in state.boundary_inputs.items()}
    new = (torch.tensor([[7], [8]]), torch.tensor([[2], [1]]),
           torch.tensor([[9.], [6.]]))
    with torch.no_grad():
        x = current.embed_inputs(*new)
        expected_inputs, expected_k, expected_v = {}, [], []
        for layer, block in enumerate(current.blocks):
            if layer:
                expected_inputs[layer] = x.clone()
            x, (k, v) = block.forward_with_cache(x, state.cache.k[layer], state.cache.v[layer])
            expected_k.append(k)
            expected_v.append(v)

    current.train()
    appended = append(current, state, *new)
    assert current.training
    torch.testing.assert_close(appended.cache.k, torch.stack(expected_k))
    torch.testing.assert_close(appended.cache.v, torch.stack(expected_v))
    for layer, added in expected_inputs.items():
        assert torch.equal(appended.boundary_inputs[layer][:, :-1], retained_inputs[layer])
        torch.testing.assert_close(appended.boundary_inputs[layer][:, -1:], added)
        assert not current.blocks[layer].norm._forward_pre_hooks

    retained_raw = tuple(torch.cat((old[:, -3:], added), 1)
                         for old, added in zip(raw, new, strict=True))
    migrated = recompute_interval(next_model, appended, *retained_raw, (4, 5))
    with torch.no_grad():
        x = appended.boundary_inputs[4]
        for layer in (4, 5):
            torch.testing.assert_close(migrated.boundary_inputs[layer], x)
            x, (k, v) = next_model.blocks[layer](x, return_kv=True)
            torch.testing.assert_close(migrated.cache.k[layer], k)
            torch.testing.assert_close(migrated.cache.v[layer], v)
    assert torch.equal(migrated.cache.k[:4], appended.cache.k[:4])
    assert torch.equal(migrated.cache.v[:4], appended.cache.v[:4])


def test_profiling_scores_independent_candidates_from_the_same_state():
    parent, current = model(11), model(19)
    raw = events()
    state = capture_state(parent, *raw)
    old = state.cache.k.clone()
    intervals = [None, (0, 0), (2, 3)]
    candidates = torch.tensor([[9, 10], [10, 9]])
    query_delta = torch.tensor([2., 3.])

    def score(candidate):
        return current.score_cc_reuse(candidate.cache, candidates, query_delta).mean().item()

    expected = [score(recompute_interval(current, state, *raw, interval))
                for interval in intervals]
    current.train()
    rows = profile_intervals(current, state, *raw, score=score, intervals=intervals)
    assert [row["interval"] for row in rows] == intervals
    assert [row["score"] for row in rows] == expected
    assert [row["recomputed_layers"] for row in rows] == [0, 1, 2]
    assert current.training
    assert torch.equal(state.cache.k, old)
