"""CPU mathematical checks; no real checkpoints, datasets, or GPUs needed."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from experiments.figure9recflow.cache_ops import (
    append_behaviors, deviation_scores, encode_cache, recompute_interval,
    recompute_positions, recompute_tail, score_candidates, select_positions,
)
from experiments.recflow_full_population.code.model import RequestRanker
from experiments.recflow_full_population.code.shared_reader import score_shared_cache
from hstu_kvcache.models.hstu import HSTU, HSTUConfig
from hstu_kvcache.models.kv_cache import HSTUKVCache


class TinyFeatures(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.projection = nn.Linear(5, hidden)

    def forward(self, features):
        return self.projection(features.float() / 10)


class TinyRanker(nn.Module):
    # Exercise the real production encoder/reader with a tiny external encoder.
    encode_history = RequestRanker.encode_history
    read = RequestRanker.read

    def __init__(self):
        super().__init__()
        self.backbone = HSTU(HSTUConfig(
            num_items=1, num_behaviors=4, hidden_size=8, num_layers=3,
            num_heads=2, head_dim=4, max_seq_len=17, temporal_num_freqs=2,
            block_variant="hstu_reference", activation="silu",
            relative_position_bias=True, attn_dropout=0, input_dropout=0,
        ))
        for block in self.backbone.blocks:
            block.attn.backend = "torch"
        self.features = TinyFeatures(8)
        self.chunk_size = 2
        self.checkpoint_queries = False


@pytest.fixture
def models():
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    old = TinyRanker().eval()
    current = copy.deepcopy(old)
    with torch.no_grad():
        for parameter in current.parameters():
            parameter.add_(torch.randn_like(parameter) * .08)
    return old, current


def raw(n=7, batch=1):
    generator = torch.Generator().manual_seed(93 + n + batch)
    return (torch.randint(0, 20, (batch, n, 5), generator=generator),
            torch.randint(0, 4, (batch, n), generator=generator),
            torch.rand((batch, n), generator=generator) * 123 + 1)


def close_cache(actual, expected, *, atol=2e-6):
    assert actual.seq_len == expected.seq_len
    torch.testing.assert_close(actual.k, expected.k, rtol=2e-5, atol=atol)
    torch.testing.assert_close(actual.v, expected.v, rtol=2e-5, atol=atol)


@torch.no_grad()
def dense_sparse_oracle(model, cache, events, positions):
    """Full N-by-N native stale-KV forward at every layer, then select rows.

    This independently obtains query positions and relative bias from the full
    native forward, rather than sharing sparse attention/indexing code.
    Unselected temporary hidden rows cannot affect selected query outputs.
    """
    features, behaviors, deltas = events
    b, n = features.shape[:2]
    hidden = model.backbone.combine_input_features(model.features(features), behaviors, deltas)
    keys, values = [], []
    for layer, block in enumerate(model.backbone.blocks):
        _, (full_k, full_v) = block(hidden, return_kv=True)
        k, v = cache.k[layer].clone(), cache.v[layer].clone()
        for user in range(b):
            k[user, positions[user]] = full_k[user, positions[user]]
            v[user, positions[user]] = full_v[user, positions[user]]
        hidden = block.forward_stale_kv(hidden, k, v)
        keys.append(k)
        values.append(v)
    return HSTUKVCache.from_layer_list(list(zip(keys, values)), n)


def test_encode_matches_production_and_captures_real_inputs(models):
    old, _ = models
    events = raw()
    captured = encode_cache(old, *events, capture=True)
    batch = dict(history_features=events[0], behaviors=events[1], deltas=events[2], lengths=torch.tensor([7]))
    close_cache(captured.cache, old.encode_history(batch))
    close_cache(captured.cache, old.encode_history(batch, reference=True))
    recorded = []
    hooks = [block.norm.register_forward_pre_hook(lambda module, args: recorded.append(args[0].detach().clone()))
             for block in old.backbone.blocks]
    try:
        old.encode_history(batch, reference=True)
    finally:
        for hook in hooks:
            hook.remove()
    for actual, expected in zip(captured.layer_inputs, recorded, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("n", [0, 1, 7])
def test_zero_and_full_repair_endpoints(models, n):
    old, current = models
    events = raw(n)
    cache, rebuilt = encode_cache(old, *events), encode_cache(current, *events)
    assert recompute_tail(current, cache, *events, 0) is cache
    assert recompute_positions(current, cache, *events, []) is cache
    close_cache(recompute_tail(current, cache, *events, 100), rebuilt)
    close_cache(recompute_interval(current, cache, *events, (0, 2)), rebuilt)
    close_cache(recompute_positions(current, cache, *events, torch.arange(n)), rebuilt)


def test_sparse_matches_independent_dense_reference_and_preserves_unselected(models):
    old, current = models
    events = raw(batch=2)
    cache = encode_cache(old, *events)
    old_k, old_v = cache.k.clone(), cache.v.clone()
    positions = torch.tensor([[0, 2, 6], [1, 3, 5]])
    expected = dense_sparse_oracle(current, cache, events, positions)
    for chunk in [1, 2, 64]:
        repaired = recompute_positions(current, cache, *events, positions, query_chunk_size=chunk)
        close_cache(repaired, expected)
        for user in range(2):
            unused = torch.tensor([x for x in range(7) if x not in positions[user]])
            assert torch.equal(repaired.k[:, user, unused], cache.k[:, user, unused])
            assert torch.equal(repaired.v[:, user, unused], cache.v[:, user, unused])
    assert torch.equal(old_k, cache.k) and torch.equal(old_v, cache.v)


def test_sparse_causality_even_with_changed_selected_future(models):
    old, current = models
    events = raw()
    cache = encode_cache(old, *events)
    positions = torch.tensor([[1, 5]])
    first = recompute_positions(current, cache, *events, positions)
    modified = tuple(x.clone() for x in events)
    modified[0][:, 5:] += 200
    altered_cache = HSTUKVCache(cache.k.clone(), cache.v.clone(), cache.seq_len)
    altered_cache.k[:, :, 2:] += 500
    altered_cache.v[:, :, 2:] -= 300
    second = recompute_positions(current, altered_cache, *modified, positions)
    torch.testing.assert_close(first.k[:, :, 1], second.k[:, :, 1], rtol=0, atol=0)
    torch.testing.assert_close(first.v[:, :, 1], second.v[:, :, 1], rtol=0, atol=0)


def test_tail_matches_sparse_and_native_suffix_preserving_original_deltas(models):
    old, current = models
    events = raw()
    cache = encode_cache(old, *events)
    actual = recompute_tail(current, cache, *events, 3)
    sparse = recompute_positions(current, cache, *events, torch.arange(4, 7))
    close_cache(actual, sparse)
    prefix = HSTUKVCache(cache.k[:, :, :4], cache.v[:, :, :4], 4)
    vectors = current.features(events[0][:, 4:])
    _, expected = current.backbone.forward_with_cache_from_item_embeddings(prefix, vectors, events[1][:, 4:], events[2][:, 4:])
    close_cache(actual, expected)
    assert torch.equal(actual.k[:, :, :4], cache.k[:, :, :4])


def test_layer_nonprefix_uses_actual_old_boundary_and_only_changes_interval(models):
    old, current = models
    events = raw()
    captured = encode_cache(old, *events, capture=True)
    with pytest.raises(ValueError, match="actual old"):
        recompute_interval(current, captured.cache, *events, (1, 2))
    actual = recompute_interval(current, captured.cache, *events, (1, 2), layer_inputs=captured.layer_inputs)
    hidden = captured.layer_inputs[1]
    expected_k, expected_v = captured.cache.k.clone(), captured.cache.v.clone()
    for layer in [1, 2]:
        hidden, (expected_k[layer], expected_v[layer]) = current.backbone.blocks[layer](hidden, return_kv=True)
    close_cache(actual, HSTUKVCache(expected_k, expected_v, 7))
    assert torch.equal(actual.k[0], captured.cache.k[0])
    # This is genuinely a mixed-boundary computation, not teacher replacement.
    assert not torch.allclose(actual.k[2], encode_cache(current, *events).k[2])


def test_deviation_no_q_and_stable_ties(models):
    old, current = models
    events = raw()
    cache = encode_cache(old, *events)

    def reject_q(*args):
        raise AssertionError("Deviation selector must not compute Q")

    hook = current.backbone.blocks[0].attn.q_proj.register_forward_pre_hook(reject_q)
    try:
        actual = deviation_scores(current, cache, *events)
    finally:
        hook.remove()
    fresh = encode_cache(current, *events)
    expected = (fresh.k[0] - cache.k[0]).square().mean(-1) + (fresh.v[0] - cache.v[0]).square().mean(-1)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(select_positions(torch.tensor([[1., 3., 3., 0.]]), 2), torch.tensor([[1, 2]]))
    assert torch.equal(select_positions(torch.ones(1, 7), 3), torch.tensor([[0, 1, 2]]))
    assert torch.equal(deviation_scores(old, cache, *events), torch.zeros_like(actual))


@torch.no_grad()
def native_sequential_append(model, cache, events, window):
    state = cache
    for index in range(events[0].shape[1]):
        keep = min(state.seq_len, window - 1)
        start = state.seq_len - keep
        state = HSTUKVCache(state.k[:, :, start:], state.v[:, :, start:], keep)
        _, state = model.backbone.forward_with_cache_from_item_embeddings(
            state, model.features(events[0][:, index:index + 1]),
            events[1][:, index:index + 1], events[2][:, index:index + 1])
    return state


def test_append_fifo_matches_native_single_event_and_differs_from_crop_after(models):
    old, current = models
    cache = encode_cache(old, *raw(4))
    events = raw(3)
    original_k = cache.k.clone()
    actual = append_behaviors(current, cache, *events, window_size=4)
    close_cache(actual, native_sequential_append(current, cache, events, 4))
    # Appending all then cropping permits expired keys to influence writes.
    _, unbounded = current.backbone.forward_with_cache_from_item_embeddings(
        cache, current.features(events[0]), events[1], events[2])
    assert not torch.allclose(actual.k[-1], unbounded.k[-1, :, -4:], atol=1e-6)
    assert torch.equal(cache.k, original_k)
    assert actual.seq_len == 4


def test_method_batch_shared_inputs_matches_individual_append_and_sparse(models):
    old, current = models
    history = raw(4)
    one, two = encode_cache(old, *history), encode_cache(current, *history)
    batch = HSTUKVCache(torch.cat([one.k, two.k], dim=1), torch.cat([one.v, two.v], dim=1), 4)
    events = raw(3)
    actual = append_behaviors(current, batch, *events, window_size=4)
    repaired = recompute_positions(current, batch, *history, torch.tensor([[0, 2], [1, 3]]))
    for index, cache in enumerate([one, two]):
        expected = append_behaviors(current, cache, *events, window_size=4)
        close_cache(HSTUKVCache(actual.k[:, index:index + 1], actual.v[:, index:index + 1], 4), expected)
        sparse = recompute_positions(current, cache, *history, [[0, 2], [1, 3]][index])
        close_cache(HSTUKVCache(repaired.k[:, index:index + 1], repaired.v[:, index:index + 1], 4), sparse)


@pytest.mark.parametrize("history_length", [0, 7])
def test_candidates_native_shared_reader_chunking_no_writes(models, history_length):
    _, current = models
    events = raw(history_length)
    cache = encode_cache(current, *events)
    candidates = raw(5)[0]
    times = torch.arange(5).float()[None] + 2
    before = cache.k.clone(), cache.v.clone()
    expected = score_shared_cache(current.backbone, cache, candidates[..., 0], times,
                                  torch.tensor([history_length]), item_vectors=current.features(candidates))
    # Independent dense oracle: append each transient candidate to the complete
    # native history in a separate forward, so candidates cannot see each other.
    prefix = current.backbone.combine_input_features(current.features(events[0]), events[1], events[2])
    dense_scores = []
    for index in range(candidates.shape[1]):
        query = current.backbone.embed_query_tokens(
            candidates[:, index:index + 1, 0], times[:, index:index + 1],
            item_vectors=current.features(candidates[:, index:index + 1]))
        hidden, _ = current.backbone.forward_embedded(torch.cat((prefix, query), dim=1))
        dense_scores.append(current.backbone.cc_score_head(hidden[:, -1]).squeeze(-1))
    torch.testing.assert_close(expected, torch.stack(dense_scores, dim=1), rtol=2e-5, atol=2e-6)
    for size in [1, 3, 20]:
        actual = score_candidates(current, cache, candidates, times, chunk_size=size)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert torch.equal(cache.k, before[0]) and torch.equal(cache.v, before[1])
    doubled = HSTUKVCache(cache.k.expand(-1, 2, -1, -1), cache.v.expand(-1, 2, -1, -1), cache.seq_len)
    torch.testing.assert_close(score_candidates(current, doubled, candidates, times), expected.expand(2, -1), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(score_candidates(current, cache, candidates, times,
                                               candidate_vectors=current.features(candidates)), expected,
                               rtol=2e-5, atol=2e-6)


def test_invalid_positions_and_training_mode_rejected(models):
    old, current = models
    events = raw()
    cache = encode_cache(old, *events)
    for positions in [[1, 1], [4, 2], [-1], [7]]:
        with pytest.raises(ValueError, match="Positions"):
            recompute_positions(current, cache, *events, positions)
    current.train()
    with pytest.raises(ValueError, match="eval"):
        encode_cache(current, *events)
