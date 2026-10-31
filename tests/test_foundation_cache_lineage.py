from __future__ import annotations

import copy

import torch

from hstu_kvcache.evaluation import (
    OneHopRollingBundle, VersionedCacheState, append_timestamp_group, timestamp_groups,
)
from hstu_kvcache.models import HSTU, HSTUConfig


def _model() -> HSTU:
    torch.manual_seed(31)
    return HSTU(HSTUConfig(
        num_items=32, num_behaviors=4, hidden_size=16, num_layers=2,
        num_heads=2, max_seq_len=8, input_dropout=0.0, attn_dropout=0.0,
    )).eval()


def test_timestamp_groups_do_not_depend_on_input_row_order() -> None:
    events = [(11, 4, 1), (10, 3, 2), (11, 2, 1), (10, 1, 1)]
    forward = list(timestamp_groups(events))
    reverse = list(timestamp_groups(reversed(events)))
    assert forward == reverse
    assert [timestamp for timestamp, _ in forward] == [10, 11]


def test_timestamp_group_append_tracks_producers_and_rolling_eviction() -> None:
    model = _model()
    items = torch.tensor([[1, 2, 3]], dtype=torch.long)
    behaviors = torch.tensor([[1, 1, 2]], dtype=torch.long)
    deltas = torch.tensor([[0.0, 1.0, 1.0]])
    cache = model.compute_kv(items, behaviors, deltas)
    state = VersionedCacheState(cache, 9, ("v0", "v0", "v0"))
    updated = append_timestamp_group(
        model, state, [(10, 5, 1), (10, 4, 2)], producer_version="v1", max_length=4
    )
    assert updated.cache.seq_len == 4
    assert updated.last_timestamp == 10
    assert updated.producer_versions == ("v0", "v0", "v1", "v1")
    assert updated.producer_counts() == {"v0": 2, "v1": 2}


def test_readout_observation_preserves_scores_and_cache() -> None:
    current = _model()
    parent = copy.deepcopy(current)
    with torch.no_grad():
        parent.blocks[0].attn.k_proj.weight.add_(0.05)
    items = torch.tensor([[1, 2, 3]], dtype=torch.long)
    behaviors = torch.tensor([[1, 1, 2]], dtype=torch.long)
    deltas = torch.tensor([[0.0, 1.0, 1.0]])
    candidates = torch.tensor([[4, 5]], dtype=torch.long)
    query_deltas = torch.tensor([2.0])
    cache = parent.compute_kv(items, behaviors, deltas)
    before_k, before_v = cache.k.clone(), cache.v.clone()
    with torch.inference_mode():
        original = current.score_cc_reuse(cache, candidates, query_deltas)
        observed, readout = current.observe_cc_reuse(cache, candidates, query_deltas)
        full_score = current.score_cc_full(
            items, behaviors, deltas, candidates, query_deltas
        )
        full_observed, full_readout = current.observe_cc_full(
            items, behaviors, deltas, candidates, query_deltas
        )
    assert torch.equal(original, observed)
    assert torch.equal(full_score, full_observed)
    assert readout.shape == full_readout.shape == (1, 2, 16)
    assert torch.equal(cache.k, before_k) and torch.equal(cache.v, before_v)


def test_execution_matched_bundle_has_four_aligned_rolling_paths() -> None:
    parent = _model()
    current = copy.deepcopy(parent)
    bundle = OneHopRollingBundle.at_cutover(
        parent, current, [(7, 1, 1), (8, 2, 1)], parent_version="v0",
        current_version="v1", max_length=8,
    )
    observed = bundle.observe(parent, current, candidate_id=3, query_timestamp=9)
    assert set(observed) == {
        "parent_exact_rolling", "current_exact_rolling",
        "one_hop_reuse_rolling", "recursive_reuse_rolling",
    }
    assert len({round(value[0], 7) for value in observed.values()}) == 1
    bundle.append_group(
        parent, current, [(9, 4, 1), (9, 5, 2)],
        parent_version="v0", current_version="v1", max_length=8,
    )
    assert bundle.parent_exact.producer_counts() == {"v0": 4}
    assert bundle.current_exact.producer_counts() == {"v1": 4}
    assert bundle.one_hop_reuse.producer_counts() == {"v0": 2, "v1": 2}


def test_r0_same_producer_reuse_is_at_numeric_floor() -> None:
    parent = _model()
    r0 = copy.deepcopy(parent)
    with torch.no_grad():
        r0.query_encoder.type_embedding.weight.add_(0.1)
        r0.cc_score_head.bias.add_(0.1)
    items = torch.tensor([[1, 2, 3]], dtype=torch.long)
    behaviors = torch.tensor([[1, 1, 2]], dtype=torch.long)
    deltas = torch.tensor([[0.0, 1.0, 1.0]])
    candidate = torch.tensor([[4]], dtype=torch.long)
    query_delta = torch.tensor([2.0])
    old_cache = parent.compute_kv(items, behaviors, deltas)
    new_cache = r0.compute_kv(items, behaviors, deltas)
    with torch.inference_mode():
        reused = r0.score_cc_reuse(old_cache, candidate, query_delta)
        exact = r0.score_cc_reuse(new_cache, candidate, query_delta)
    assert torch.equal(old_cache.k, new_cache.k)
    assert torch.equal(old_cache.v, new_cache.v)
    assert torch.equal(reused, exact)
