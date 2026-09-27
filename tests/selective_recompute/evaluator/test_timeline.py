"""Independent scalar comparisons for the new batched repair evaluator."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from hstu_kvcache.baselines import layer_recompute as lr
from hstu_kvcache.baselines.deviation_recompute import layer0_deviation_scores
from hstu_kvcache.baselines.query_recompute import query_attention_scores
from hstu_kvcache.baselines.sparse_recompute import recompute_positions, select_top_positions
from hstu_kvcache.baselines.tail_recompute import recompute_tail
from hstu_kvcache.evaluation import append_timestamp_group, materialize_state, timestamp_groups
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.training import FoundationHistoryIndex, collate_foundation_batch
from selective_recompute_2026_09 import evaluate
from selective_recompute_2026_09.cost import CostModel


@pytest.fixture
def tiny_models(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    models = []
    for seed in (17, 19):
        torch.manual_seed(seed)
        model = HSTU(HSTUConfig(num_items=40, num_behaviors=3, hidden_size=32,
                               num_heads=1, num_layers=2, max_seq_len=4,
                               input_dropout=0)).eval()
        models.append(model)
    return tuple(models)


def _history_and_requests(short=False):
    if short:
        events = {3: [(5, 1, 1), (8, 2, 2), (20, 7, 1), (22, 8, 2),
                      (22, 4, 1), (700000, 9, 2)]}
        queries = {3: [(20, 5), (21, 6), (22, 7), (23, 8), (700001, 9)]}
    else:
        # Same-timestamp raw order deliberately differs from canonical mapped
        # item order; this happens after compact mapping in the real loader.
        events = {
            1: [(1, 1, 1), (5, 6, 1), (10, 9, 2), (10, 2, 1), (12, 4, 1),
                (20, 8, 1), (20, 3, 2), (22, 7, 1), (25, 5, 2)],
            2: [(4, 12, 1), (7, 13, 2), (7, 5, 1), (15, 6, 1),
                (20, 11, 1), (22, 10, 1), (22, 1, 2), (25, 2, 2)],
        }
        queries = {1: [(20, 5), (20, 6), (21, 7), (22, 8), (26, 9)],
                   2: [(20, 6), (23, 7), (23, 8), (26, 9)]}
    history = FoundationHistoryIndex({uid: tuple(np.asarray(column, dtype=np.int64)
                                                  for column in zip(*rows))
                                      for uid, rows in events.items()})
    by_user = {uid: [dict(request_id=f"{uid}:{i}", uid=uid, query_timestamp=t,
                           item_idx=item, label=float(i % 2), weight=1.0)
                     for i, (t, item) in enumerate(rows)] for uid, rows in queries.items()}
    return events, history, by_user


@torch.inference_mode()
def _scalar_oracle(events, history, by_user, parent, current):
    """Use the existing scalar lifecycle, independent of the new scheduler."""
    oracle = {}
    for uid, requests in by_user.items():
        state = materialize_state(parent, [row for row in events[uid] if row[0] < 20],
                                  producer_version="parent", max_length=4)
        groups = list(timestamp_groups(row for row in events[uid] if row[0] >= 20))
        at_time = defaultdict(list)
        for request in requests:
            at_time[request["query_timestamp"]].append(request)
        cursor = 0
        for timestamp, simultaneous in sorted(at_time.items()):
            while cursor < len(groups) and groups[cursor][0] < timestamp:
                state = append_timestamp_group(current, state, groups[cursor][1],
                                               producer_version="current", max_length=4)
                cursor += 1
            for request in simultaneous:
                candidate = torch.tensor([[request["item_idx"]]])
                delta = torch.tensor([float(timestamp - state.last_timestamp)])
                reuse = current.observe_cc_reuse(state.cache, candidate, delta)[0][0, 0]
                full = collate_foundation_batch([request], history, device=torch.device("cpu"), max_history=4)
                full_logit = current.observe_cc_full(full.item_ids, full.behaviors, full.time_deltas,
                    full.candidate_ids, full.query_time_deltas, lengths=full.lengths)[0][0, 0]
                request["reuse_logit"], request["full_logit"] = float(reuse), float(full_logit)
                oracle[request["request_id"]] = (state.cache.k.clone(), state.cache.v.clone(), float(delta[0]))
    return oracle


def _stats():
    return {name: Counter() for name in ("full_history_hist", "append_prefix_hist", "initial_history_hist")}


@pytest.mark.parametrize("short", (False, True))
@torch.inference_mode()
def test_snapshots_match_scalar_ties_append_and_eviction(tiny_models, short):
    parent, current = tiny_models
    events, history, requests = _history_and_requests(short)
    expected = _scalar_oracle(events, history, requests, parent, current)
    stats, seen, lengths = _stats(), set(), []
    for snap in evaluate.snapshots(list(requests), requests, history, parent, current, 20,
                                   max_length=4, query_batch=2, stats=stats):
        for index, request in enumerate(snap.requests):
            key = request["request_id"]
            k, v, delta = expected[key]
            torch.testing.assert_close(snap.state.cache.k[:, index:index + 1], k, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(snap.state.cache.v[:, index:index + 1], v, atol=1e-6, rtol=1e-5)
            assert float(snap.query_deltas[index]) == delta
            uid, query = request["uid"], request["query_timestamp"]
            prior = sorted(row for row in events[uid] if row[0] < query)[-4:]
            assert snap.events[0][index].tolist() == [row[1] for row in prior]
            assert snap.events[1][index].tolist() == [row[2] for row in prior]
            lengths.append(len(prior))
            assert key not in seen
            seen.add(key)
    assert seen == set(expected)
    assert stats["full_history_hist"] == Counter(lengths)
    if short:
        assert stats["full_history_hist"] == {2: 1, 3: 2, 4: 2}
        assert sum(int(key.split(":")[1]) * count for key, count in stats["band_append_hist"].items()) == 4
    else:
        assert sum(int(key.split(":")[1]) * count for key, count in stats["band_append_hist"].items()) == 8


@torch.inference_mode()
def test_ephemeral_repairs_do_not_change_future_reuse(tiny_models):
    parent, current = tiny_models
    events, history, requests = _history_and_requests()
    expected = _scalar_oracle(events, history, requests, parent, current)
    for snap in evaluate.snapshots(list(requests), requests, history, parent, current, 20,
                                   max_length=4, query_batch=3, stats=_stats()):
        before_k, before_v = snap.state.cache.k.clone(), snap.state.cache.v.clone()
        before_boundaries = {layer: values.clone() for layer, values in snap.state.boundary_inputs.items()}
        caches = [lr.recompute_interval(current, snap.state, *snap.events, (1, 1)).cache,
                  recompute_tail(current, snap.state.cache, *snap.events, 2)]
        for scores in (layer0_deviation_scores(current, snap.state.cache, *snap.events),
                       query_attention_scores(current, snap.state.cache, snap.candidates, snap.query_deltas)):
            caches.append(recompute_positions(current, snap.state.cache, *snap.events,
                                               select_top_positions(scores, 2)))
        for cache in caches:
            current.observe_cc_reuse(cache, snap.candidates, snap.query_deltas)
        assert torch.equal(snap.state.cache.k, before_k)
        assert torch.equal(snap.state.cache.v, before_v)
        for layer, values in before_boundaries.items():
            assert torch.equal(snap.state.boundary_inputs[layer], values)
        for index, request in enumerate(snap.requests):
            k, v, _ = expected[request["request_id"]]
            torch.testing.assert_close(snap.state.cache.k[:, index:index + 1], k, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(snap.state.cache.v[:, index:index + 1], v, atol=1e-6, rtol=1e-5)


@torch.inference_mode()
def test_zero_repair_control_uses_saved_reuse_and_original_full_collator(tiny_models):
    parent, current = tiny_models
    events, history, requests = _history_and_requests(short=True)
    _scalar_oracle(events, history, requests, parent, current)
    # Frozen Full/Reuse Parquet panels carry labels/logits but no training
    # weight column. Verification must supply the collator-only default.
    for user_requests in requests.values():
        for request in user_requests:
            request.pop("weight")
    rows, stats, controls = evaluate.score_unit(list(requests), requests, history, parent, current, 20,
        intervals={"1": [0, 0]}, cost_model=CostModel(32, 2, 1, "torch"),
        cohort_size=2, query_batch=2, verify=True, methods=())
    assert rows == {}
    assert controls["requests"] == 5
    assert controls["reuse_max_abs_logit_error"] < 2e-5
    assert controls["full_max_abs_logit_error"] < 2e-5
    assert controls["repair_full_max_abs_logit_error"] < 2e-5
    assert sum(int(key.split(":")[1]) * count for key, count in stats["band_append_hist"].items()) == 4


@torch.inference_mode()
def test_short_history_full_endpoint_matches_charged_native_fastpath(tiny_models, monkeypatch):
    parent, current = tiny_models
    history = FoundationHistoryIndex({1: (np.array([5]), np.array([2]), np.array([1]))})
    requests = {1: [dict(request_id="one", uid=1, query_timestamp=20, item_idx=3, label=1.0, weight=1.0)]}
    calls = {"deviation": 0, "query": 0, "sparse": 0}
    for name, key in (("layer0_deviation_scores", "deviation"),
                      ("query_attention_scores", "query"), ("recompute_positions", "sparse")):
        original = getattr(evaluate, name)
        def count(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(evaluate, name, count)
    cost = CostModel(32, 2, 1, "torch")
    rows, _, _ = evaluate.score_unit([1], requests, history, parent, current, 20,
        intervals={"1": [0, 0]}, cost_model=cost, cohort_size=1, query_batch=1,
        methods=("deviation", "query"))
    for method in rows:
        assert rows[method]
        for row in rows[method]:
            assert row["selected_tokens"] == 1
            assert row["selection_flops"] == 0
            assert row["recompute_flops"] == cost.full_cache(1)
    assert calls == {"deviation": 0, "query": 0, "sparse": 0}
