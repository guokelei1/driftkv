"""Causal teacher forcing keeps mapped old rows and replaces only new writes."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from experiments.ch3_motivation.code.kv_evolution import (
    checkpoint_metadata, clone_native_trajectory, teacher_forced_cache,
)
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from hstu_kvcache.models.state_transition import append_with_rolling_cap


@pytest.fixture
def state(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.set_num_threads(1)
    torch.manual_seed(29)
    model = HSTU(HSTUConfig(num_items=300, num_behaviors=2, hidden_size=8,
        num_heads=2, num_layers=2, max_seq_len=4,
        input_dropout=0., attn_dropout=0.)).eval().requires_grad_(False)
    old = copy.deepcopy(model)
    events = (np.array([10, 20, 30, 60, 70, 80, 90, 100, 110]),
              np.arange(1, 10), np.zeros(9, dtype=np.int64))
    native = NativeTrajectory(old, 0, events, 50, max_length=4, append_band_size=1)
    return model, native


@torch.no_grad()
def test_clone_owns_native_counters_and_advances_without_mutating_original(state):
    model, native = state
    clone = clone_native_trajectory(native)
    assert clone.cache is native.cache and clone.events is native.events
    assert clone.ledger is not native.ledger and clone.producers is not native.producers
    original = native.cache.k.clone(), native.producers.copy(), dict(native.ledger)
    clone.switch(model, 1, 50)
    clone.advance(71)
    assert native.cursor == 3 and clone.cursor == 5
    assert native.appended_tokens == 0 and clone.appended_tokens == 2
    assert dict(native.ledger) == original[2]
    assert np.array_equal(native.producers, original[1])
    torch.testing.assert_close(native.cache.k, original[0], atol=0, rtol=0)
    assert checkpoint_metadata(clone, 3) == dict(time=71, event_cursor=5,
        appended_events=2, evicted_events=1, history_rows=4,
        surviving_inherited_rows=2, inherited_fraction=.5)


@torch.no_grad()
def test_snapshot_matches_causal_new_row_forcing_through_growth_and_fifo(state):
    model, native = state
    mapped = HSTUKVCache(native.cache.k + .07, native.cache.v - .09, native.cache.seq_len)
    original = mapped.k.clone(), mapped.v.clone()
    teacher = NativeTrajectory(model, 1, native.events, 50, max_length=4, append_band_size=1)
    # Same initial mapped prefix; the manual path writes with the student and
    # immediately substitutes ONLY the just-written row from the teacher.
    manual = HSTUKVCache(mapped.k.clone(), mapped.v.clone(), mapped.seq_len)
    previous = 30
    for position in range(3, len(native.events[0])):
        timestamp = int(native.events[0][position])
        teacher.advance(timestamp + 1)
        teacher_before = teacher.cache.k.clone(), teacher.cache.v.clone()
        item = torch.tensor([[native.events[1][position]]])
        behavior = torch.tensor([[native.events[2][position]]])
        delta = torch.tensor([[timestamp - previous]], dtype=torch.float32)
        manual = append_with_rolling_cap(model, manual, item, behavior, delta, 4)
        manual.k[:, :, -1:].copy_(teacher.cache.k[:, :, -1:])
        manual.v[:, :, -1:].copy_(teacher.cache.v[:, :, -1:])
        actual = teacher_forced_cache(mapped, teacher.cache, teacher.evicted_tokens)
        torch.testing.assert_close(actual.k, manual.k)
        torch.testing.assert_close(actual.v, manual.v)
        torch.testing.assert_close(teacher.cache.k, teacher_before[0], atol=0, rtol=0)
        torch.testing.assert_close(teacher.cache.v, teacher_before[1], atol=0, rtol=0)
        assert actual.k.data_ptr() != teacher.cache.k.data_ptr()
        previous = timestamp
    # Once the initial rows are gone the diagnostic is exactly the teacher.
    torch.testing.assert_close(actual.k, teacher.cache.k, atol=0, rtol=0)
    torch.testing.assert_close(actual.v, teacher.cache.v, atol=0, rtol=0)
    torch.testing.assert_close(mapped.k, original[0], atol=0, rtol=0)
    torch.testing.assert_close(mapped.v, original[1], atol=0, rtol=0)


def test_teacher_forcing_rejects_impossible_prefix_and_mismatched_geometry():
    old = HSTUKVCache(torch.zeros(2, 1, 3, 8), torch.zeros(2, 1, 3, 8), 3)
    short = HSTUKVCache(torch.zeros(2, 1, 2, 8), torch.zeros(2, 1, 2, 8), 2)
    with pytest.raises(ValueError, match="surviving inherited"):
        teacher_forced_cache(old, short, 0)
    different = HSTUKVCache(torch.zeros(2, 1, 3, 4), torch.zeros(2, 1, 3, 4), 3)
    with pytest.raises(ValueError, match="geometry"):
        teacher_forced_cache(old, different, 0)


@torch.no_grad()
def test_population_user_ties_synthetic_no_labels_and_first_last_roles(state):
    from experiments.ch3_motivation.code.run_kv_evolution import replay_user
    from experiments.ch3_motivation.code.probes import ProbeConfig, ProbeLadder
    model, native = state
    parent = copy.deepcopy(model)
    next(model.parameters()).add_(.03)
    probe = ProbeLadder(ProbeConfig.from_model(model, hidden_width=4, depth=1), "f").eval()
    times = np.array([86370, 86380, 86390, 86400, 86400, 86402, 86403, 86404])
    events = times, np.arange(1, len(times) + 1), np.zeros(len(times), dtype=np.int64)
    cfg = dict(release_day=1, end_day=2, edge=1, history_length=4, append_band_size=2,
               query_batch=16, seed=17)
    requests = {86400: [dict(raw_item_id=12, item_idx=12, label=1)],
                86403: [dict(raw_item_id=13, item_idx=13, label=0)]}
    record, rows = replay_user(parent, model, probe, events, requests, 7, cfg, thresholds=(1, 3))
    assert [(s["checkpoint"], s["appended_events"]) for s in record["snapshots"]] == [
        ("cutover", 0), ("append_1", 2), ("append_3", 3)]
    assert len(rows) == 8 and {r["phase"] for r in rows} == {"first", "last"}
    for timestamp in requests:
        scores = {r["method"]: r["score"] for r in rows if r["query_timestamp"] == timestamp}
        assert scores["f"] == pytest.approx(scores["reuse"], abs=1e-6)
    assert all("label" not in s for s in record["snapshots"])
    only = {86400: requests[86400]}
    _, once = replay_user(parent, model, probe, events, only, 7, cfg, thresholds=())
    assert len(once) == 4 and {r["phase"] for r in once} == {"first_last"}
