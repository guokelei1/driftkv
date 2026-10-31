"""Real-quality paths preserve independent native writes and causal reads."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from experiments.ch3_motivation.code.kv_evolution import clone_native_trajectory
from experiments.ch3_motivation.code.probe_variants import build_variant
from experiments.ch3_motivation.code.real_quality_paths import (
    BatchedQualityPaths, ORACLE_BETAS, PERSISTENT_METHODS, _blend, _member, replay_user,
)
from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache


class CountMap:
    def __init__(self, probe):
        self.probe, self.calls = probe, 0

    def prepare_cache(self, *args):
        self.calls += 1
        return self.probe.prepare_cache(*args)


@pytest.fixture
def scenario(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.set_num_threads(1)
    torch.manual_seed(83)
    old = HSTU(HSTUConfig(num_items=300, num_behaviors=2, hidden_size=8,
        num_heads=2, num_layers=2, max_seq_len=4,
        input_dropout=0., attn_dropout=0.)).eval().requires_grad_(False)
    current = copy.deepcopy(old)
    with torch.no_grad():
        for parameter in current.parameters():
            parameter.add_(torch.randn_like(parameter)*.03)
    probe = build_variant("full_item", current, stage="f", hidden_width=4,
        depth=1, query_chunk=16, row_chunk=4).eval()
    with torch.no_grad():
        for module in probe.f:
            module.weights[-1].normal_(std=.2)
    times = np.array([86370, 86380, 86390, 86400, 86401, 86401,
                      86402, 86403, 86404, 86405, 86407, 9*86400])
    events = times, np.arange(1, len(times)+1), np.arange(len(times)) % 2
    cfg = dict(edge=2, release_day=1, end_day=12, append_band_size=2, query_batch=2)
    return old, current, probe, events, cfg


@torch.no_grad()
def test_batched_nine_equal_independent_native_trajectories_at_writes_and_reads(scenario):
    old, current, probe, events, cfg = scenario
    tracked = CountMap(probe)
    paths = BatchedQualityPaths(old, current, tracked, events, cfg,
        history_cap=4, check_batch=True)
    assert tracked.calls == 1
    assert paths.methods == list(PERSISTENT_METHODS)
    base = NativeTrajectory(old, 1, events, 86400, max_length=4, append_band_size=2)
    initial = paths.cache
    ids = torch.as_tensor(base.retained_events[1], dtype=torch.long)[None]
    mapped = probe.prepare_cache(current, base.cache, ids)
    for expected, name in ((base.cache,"reuse"), (mapped,"f1"),
                           (base.fresh_cache(current),"rebuild")):
        actual = _member(initial,paths.index[name])
        torch.testing.assert_close(actual.k,expected.k,atol=0,rtol=0)
        torch.testing.assert_close(actual.v,expected.v,atol=0,rtol=0)
    references = []
    for i in range(len(paths.methods)):
        native = clone_native_trajectory(base)
        native.switch(current,2,86400)
        native.cache = _member(initial,i)
        references.append(native)
    candidates = torch.tensor([[35,48,19]])
    saw_different_new_row = False
    for timestamp in (86400,86401,86402,86403,86404,86406,86409,9*86400+1):
        paths.advance(timestamp)
        for i, native in enumerate(references):
            native.advance(timestamp)
            actual = _member(paths.cache,i)
            torch.testing.assert_close(actual.k,native.cache.k,atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(actual.v,native.cache.v,atol=1e-6,rtol=1e-5)
            assert paths.cursor == native.cursor
            assert paths.appended_tokens == native.appended_tokens
            assert paths.evicted_tokens == native.evicted_tokens
        gap = 0 if paths.last_timestamp is None else timestamp-paths.last_timestamp
        delta = torch.tensor([gap],dtype=torch.float32)
        names, scores, errors = paths.score(candidates,delta)
        for i, native in enumerate(references):
            expected = native_score(current,native.cache,candidates,delta)[0]
            torch.testing.assert_close(scores[i:i+1],expected,atol=1e-6,rtol=1e-5)
        f_error = errors[names.index("f1")]
        for name,beta in ORACLE_BETAS:
            for key in ("k_mse","v_mse","kv_mse"):
                assert errors[names.index(name)][key] == pytest.approx(
                    (1-beta)**2*f_error[key],rel=2e-5,abs=1e-10)
        before = paths.cache.k.clone(),paths.cache.v.clone()
        paths.snapshot()
        torch.testing.assert_close(paths.cache.k,before[0],atol=0,rtol=0)
        torch.testing.assert_close(paths.cache.v,before[1],atol=0,rtol=0)
        if paths.appended_tokens:
            reuse, f2 = paths.index["reuse"],paths.index["f2"]
            saw_different_new_row |= bool((paths.cache.k[1,reuse,-1] != paths.cache.k[1,f2,-1]).any())
    assert saw_different_new_row, "upper-layer new rows must use each path's own history"
    assert paths.initial_length == 3 and paths.cache.seq_len == 4
    assert paths.evicted_tokens == paths.initial_length+paths.appended_tokens-4
    assert paths.max_append_abs_error < 1e-6 and paths.max_score_abs_error < 1e-6
    assert paths.cursor == len(events[0])
    json.dumps(paths.ledger())


@torch.no_grad()
def test_real_event_rows_ties_readonly_oracle_extras_and_small_controls(scenario):
    old,current,probe,events,cfg = scenario
    requests = {86400:[dict(item_idx=41,raw_item_id=141,label=0)],
        86401:[dict(item_idx=42+i,raw_item_id=142+i,label=i%2) for i in range(5)],
        86402:[dict(item_idx=51,raw_item_id=151,label=1)]}
    extra_calls = []
    def shift(cache):
        extra_calls.append(1)
        return HSTUKVCache(cache.k+.05,cache.v-.04,cache.seq_len)
    tracked = CountMap(probe)
    rows,stats,ledger = replay_user(old,current,tracked,events,requests,17,cfg,
        history_cap=4,check_batch=True,extra_maps={"shift":shift})
    assert tracked.calls == len(extra_calls) == 1
    assert stats["feedback_events"] == 7 and stats["request_timestamps"] == 3
    assert stats["initial_history_length"] == 3 and stats["appended_events"] == 3
    assert stats["max_batch_abs_error"] < 1e-6
    assert len(rows) == 7*(9+1+3)
    assert ledger["extra_map_materializations"] == {"shift":1}
    assert ledger["event_input_execution"]["per_append_scalar_tensor_constructions"] == 0
    for row in rows:
        assert row["uid"] == 17
        assert row["appended_events"] == {86400:0,86401:1,86402:3}[row["query_timestamp"]]
        assert row["history_rows"] == min(4,3+row["appended_events"])
        assert row["query_gap"] == {86400:10,86401:1,86402:1}[row["query_timestamp"]]
        assert row["initial_kv_mse"] == stats["initial_mse"][row["method"]]["kv_mse"]
        if row["method"] == "rebuild":
            assert row["kv_mse"] == row["initial_kv_mse"] == 0
    for timestamp in requests:
        selected = [r for r in rows if r["query_timestamp"]==timestamp and r["method"]=="f1"]
        assert [r["label"] for r in selected] == [r["label"] for r in requests[timestamp]]
        assert [r["item_idx"] for r in selected] == [r["item_idx"] for r in requests[timestamp]]
    json.dumps((stats,ledger))
    small_rows,small_stats,_ = replay_user(old,current,probe,events,requests,17,cfg,
        history_cap=4,structural_controls=False,check_batch=True)
    assert small_stats["persistent_methods"] == ["reuse","f1","rebuild"]
    by_key = {(r["method"],r["query_timestamp"],r["item_idx"]):r for r in rows}
    for row in small_rows:
        expected = by_key[row["method"],row["query_timestamp"],row["item_idx"]]
        assert row["score"] == pytest.approx(expected["score"],abs=1e-6)
        assert row["kv_mse"] == pytest.approx(expected["kv_mse"],rel=1e-6,abs=1e-8)


@torch.no_grad()
def test_oracle_exact_endpoints_and_reject_mutating_extra_map(scenario):
    old,current,probe,events,cfg = scenario
    paths = BatchedQualityPaths(old,current,probe,events,cfg,history_cap=4)
    f = _member(paths.cache,paths.index["f1"])
    teacher = _member(paths.cache,paths.index["rebuild"])
    assert _blend(f,teacher,0) is f and _blend(f,teacher,1) is teacher
    def bad_map(cache):
        cache.k.add_(1)
        return cache
    with pytest.raises(ValueError,match="must not modify"):
        BatchedQualityPaths(old,current,probe,events,cfg,history_cap=4,
                            extra_maps={"bad":bad_map})
