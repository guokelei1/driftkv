"""CPU integration against the real feature encoder and RequestRanker.

The item vocabulary and transformer width are tiny, but production input,
block, reader, per-method repair and chronological engine code all execute.
"""
from __future__ import annotations

import copy
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from experiments.figure9recflow.alignment import fit_cache_pairs
from experiments.figure9recflow.cache_ops import append_behaviors, encode_cache, score_candidates
from experiments.figure9recflow.common import policies
from experiments.figure9recflow.engine import ReleaseEngine, tensors
from experiments.figure9recflow.inputs import RequestPayload, UserPayload, event_deltas
from experiments.recflow_full_population.code.configuration import load_proposal
from experiments.recflow_full_population.code.model import RequestRanker
from hstu_kvcache.models.kv_cache import HSTUKVCache


EVENT = np.dtype([("uid", "<i4"), ("ts", "<i8"), ("rid", "<i8"),
                  ("features", "<i4", (5,)), ("behavior", "u1"), ("label", "i1")])


@pytest.fixture
def make_models(tmp_path):
    torch.set_num_threads(1)
    vocab = tmp_path / "vocab"
    vocab.mkdir()
    (vocab / "manifest.json").write_text(json.dumps(dict(explicit_ids=32, hash_buckets=8)))
    np.save(vocab / "item_map.npy", np.arange(64, dtype=np.int32) % 32 + 1)

    def create(layers=2):
        torch.manual_seed(981 + layers)
        cfg = load_proposal()
        cfg["architecture"]["pilot"] = dict(layers=layers, hidden_size=8, heads=2,
                                             head_dim=4, item_embedding_dimension=8)
        cfg["execution"]["candidate_chunk_size"] = 3
        cfg["execution"]["checkpoint_candidate_chunks"] = False
        old = RequestRanker(vocab, "pilot", cfg).eval().requires_grad_(False)
        current = copy.deepcopy(old)
        with torch.no_grad():
            for parameter in current.backbone.parameters():
                parameter.add_(torch.randn_like(parameter) * .04)
            current.features.projection.weight.add_(torch.randn_like(current.features.projection.weight) * .02)
        for model in (old, current):
            for block in model.backbone.blocks:
                block.attn.backend = "torch"
        return old, current

    return create


def events(times, uid=3):
    result = np.zeros(len(times), dtype=EVENT)
    rng = np.random.default_rng(200 + len(times))
    result["uid"] = uid
    result["ts"] = times
    result["rid"] = np.arange(len(times))
    result["features"] = rng.integers(0, 30, size=(len(times), 5))
    result["behavior"] = np.arange(len(times)) % 4
    return result


def payload(n=1024, *, with_requests=True, with_events=True):
    history = events(np.arange(n) * 1000)
    base = (n + 1) * 1000
    future = events(np.array([base + 1000, base + 1000, base + 3000]) if with_events else [])
    features = np.array([[11, 1, 2, 3, 300], [12, 2, 3, 4, 500],
                         [13, 3, 4, 5, 800], [14, 4, 5, 6, 400]], np.int32)
    requests = ()
    if with_requests:
        first_delta = (base + 1000 - int(history["ts"][-1])) / 1000 if n else 0.
        requests = (
            RequestPayload(3, 101, base + 1000, features, np.array([1, 0, -1, 0], np.int8), n, first_delta),
            RequestPayload(3, 102, base + 1000, features.copy(), np.zeros(4, np.int8), n, first_delta),
            RequestPayload(3, 103, base + 2000, features.copy(), np.array([0, 0, 1, -1], np.int8),
                           min(1024, n + (2 if with_events else 0)), 1. if with_events else first_delta + 1),
        )
    return UserPayload(3, 29, history, event_deltas(history), future,
        event_deltas(future, int(history["ts"][-1]) if n else None), requests)


def fitted_maps(old, current, rows):
    points = [dict(p, fit_users=2) for p in rows if p["method"] == "kv_map"]
    pairs = []
    for n in (4, 5):
        p = payload(n=n, with_requests=False, with_events=False)
        raw = tensors(p.initial_history["features"], p.initial_history["behavior"].astype(np.int64) + 1,
                      p.initial_deltas, "cpu")
        pairs.append((encode_cache(old, *raw), encode_cache(current, *raw)))
    fit = fit_cache_pairs(pairs, points, num_heads=2, settings=dict(epochs=2, batch_rows=4),
                          teacher_flops=[100, 200], device="cpu")
    return fit["maps"]


def custom_policies():
    return [dict(tag="reuse", method="reuse"), dict(tag="rebuild", method="rebuild"),
            dict(tag="layer_prefix", method="layer", interval=[0, 0]),
            dict(tag="layer_upper", method="layer", interval=[1, 1]),
            dict(tag="tail", method="tail", tokens=2),
            dict(tag="deviation", method="deviation", tokens=2),
            dict(tag="map", method="kv_map", fit_users=2, width=5)]


def test_all_29_policies_initialize_and_degenerate_full_repairs_match(make_models):
    old, current = make_models(layers=16)
    rows = policies()
    assert len(rows) == 29
    engine = ReleaseEngine(old, current, rows, fitted_maps(old, current, rows))
    p = payload(n=5, with_requests=False, with_events=False)
    actual, ledger = engine.initialize(p)
    assert actual.k.shape == (16, 29, 5, 8)
    assert len(ledger["repair_flops"]) == 29
    rebuilt = actual.k[:, 1:2]
    for index, row in enumerate(rows):
        if row["method"] in ("tail", "deviation"):
            torch.testing.assert_close(actual.k[:, index:index + 1], rebuilt, rtol=2e-5, atol=2e-6)
            assert ledger["repair_flops"][index] == ledger["once_rebuild_flops"]
        elif row["method"] == "layer" and row["interval"][0] == 0:
            end = row["interval"][1] + 1
            torch.testing.assert_close(actual.k[:end, index:index + 1], rebuilt[:end], rtol=0, atol=0)
    assert ledger["repair_flops"][0] == 0
    assert ledger["old_boundary_union_bytes"] > 0


def test_method_batch_matches_independent_engines_with_real_fifo_and_ties(make_models):
    old, current = make_models()
    rows = custom_policies()
    maps = fitted_maps(old, current, rows)
    engine = ReleaseEngine(old, current, rows, maps, candidate_chunk=3)
    p = payload()
    old_parameters = [x.clone() for x in old.parameters()]
    original_history, original_events = p.initial_history.copy(), p.events.copy()
    observed_appends = []

    def spy(model, state, f, b, d, **kwargs):
        result = append_behaviors(model, state, f, b, d, **kwargs)
        observed_appends.append((f.shape[1], state.seq_len, result.k.clone(), result.v.clone()))
        return result

    with patch("experiments.figure9recflow.engine.append_behaviors", side_effect=spy):
        combined = engine.run_user(p, save_scores=True)
    assert [(x[0], x[1]) for x in observed_appends] == [(2, 1024), (1, 1024)]
    assert [r["rid"] for r in combined["records"]] == [101, 102, 103]
    np.testing.assert_array_equal(combined["full_scores"][0]["scores"], combined["full_scores"][1]["scores"])
    assert np.isnan(combined["records"][1]["metrics"]).all()
    for index, row in enumerate(rows):
        independent = ReleaseEngine(old, current, [row], maps, candidate_chunk=2).run_user(p, save_scores=True)
        for grouped, separate in zip(combined["full_scores"], independent["full_scores"], strict=True):
            assert grouped["rid"] == separate["rid"]
            np.testing.assert_array_equal(grouped["ids"], separate["ids"])
            np.testing.assert_allclose(grouped["scores"][index:index + 1], separate["scores"],
                                       rtol=3e-5, atol=3e-6)
        assert combined["ledger"]["repair_flops"][index] == independent["ledger"]["repair_flops"][0]
    ledger = combined["ledger"]
    assert (ledger["requests"], ledger["candidates"], ledger["append_events"]) == (3, 12, 3)
    assert (ledger["initial_rows"], ledger["final_rows"]) == (1024, 1024)
    assert ledger["post_last_query_append_events"] == 1
    assert ledger["no_query_append_events"] == 0
    assert ledger["ordinary_append_flops_per_policy"] == (
        engine.append_cost.rolling_append(1024, 2) + engine.append_cost.rolling_append(1024, 1))
    assert ledger["ordinary_read_flops_per_policy"] == 3 * engine.cost.read(1024, 4, chunk_size=3)
    # Read the pre-append state independently; both same-timestamp requests
    # must equal this, even though two observed behaviors have the same time.
    initialized, _ = engine.initialize(p)
    features = torch.tensor(p.requests[0].features).unsqueeze(0)
    before = score_candidates(current, initialized, features,
                              torch.tensor([p.requests[0].query_delta]), chunk_size=3).numpy()
    np.testing.assert_array_equal(before, combined["full_scores"][0]["scores"])
    for actual, expected in zip(old.parameters(), old_parameters, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(p.initial_history, original_history)
    np.testing.assert_array_equal(p.events, original_events)


def test_behavior_only_user_gets_full_maintenance(make_models):
    old, current = make_models()
    rows = custom_policies()
    engine = ReleaseEngine(old, current, rows, fitted_maps(old, current, rows))
    p = payload(n=3, with_requests=False)
    appended = []
    def spy(model, state, f, b, d, **kwargs):
        result = append_behaviors(model, state, f, b, d, **kwargs)
        appended.append(result.seq_len)
        return result
    with patch("experiments.figure9recflow.engine.append_behaviors", side_effect=spy):
        result = engine.run_user(p, save_scores=True)
    assert appended == [5, 6]
    assert result["records"] == result["full_scores"] == []
    assert result["ledger"]["requests"] == 0
    assert result["ledger"]["ordinary_read_flops_per_policy"] == 0
    assert result["ledger"]["append_events"] == 3
    assert result["ledger"]["final_rows"] == 6
    assert result["ledger"]["ordinary_append_flops_per_policy"] > 0
    assert result["ledger"]["no_query_append_events"] == 3
    assert result["ledger"]["post_last_query_append_events"] == 0


def test_completely_inactive_user_is_retained_without_repair(make_models):
    old, current = make_models()
    rows = custom_policies()
    # No map is needed: this user has no consumer, so none may be applied.
    engine = ReleaseEngine(old, current, rows, {})
    p = payload(n=7, with_requests=False, with_events=False)
    with patch.object(engine, "initialize", side_effect=AssertionError("No consumer must not trigger repair")):
        result = engine.run_user(p)
    assert result["uid"] == p.uid
    ledger = result["ledger"]
    assert ledger["has_history_consumer"] is False
    assert ledger["repair_flops"] == [0] * len(rows)
    assert ledger["once_rebuild_flops"] == 0
    assert ledger["source_cache_flops"] > 0  # Sunk serving reference is separate.
    assert ledger["append_events"] == ledger["requests"] == 0
    assert ledger["ordinary_append_flops_per_policy"] == ledger["ordinary_read_flops_per_policy"] == 0
    assert ledger["initial_rows"] == ledger["final_rows"] == 7


def test_false_causal_history_length_is_rejected(make_models):
    old, current = make_models()
    p = payload(n=3)
    wrong = p.requests[0]
    wrong = RequestPayload(wrong.uid, wrong.rid, wrong.ts, wrong.features, wrong.labels,
                           wrong.history_length + 1, wrong.query_delta)
    p = UserPayload(p.uid, p.day, p.initial_history, p.initial_deltas, p.events,
                    p.event_deltas, (wrong, *p.requests[1:]))
    engine = ReleaseEngine(old, current, [dict(tag="reuse", method="reuse")], {})
    with pytest.raises(RuntimeError, match="causal input ledger"):
        engine.run_user(p)
