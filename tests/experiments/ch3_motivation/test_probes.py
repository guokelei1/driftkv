"""Numerical contracts for the new Chapter 3 probes; CPU only."""
import copy
from pathlib import Path
import sys

import pytest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.probes import (
    ProbeConfig, ProbeLadder, fit_probe, bind_fit_scales, prepare_training_samples, probe_loss,
)
from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from experiments.ch3_motivation.code.probe_replay import ProbeTrajectory, replay_probe_user
from hstu_kvcache.evaluation.trajectory import NativeTrajectory


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(variant="legacy", diagonal="inclusive", length=5):
    torch.manual_seed(711)
    cfg = HSTUConfig(num_items=30, num_behaviors=2, hidden_size=8, num_heads=2,
                     num_layers=2, max_seq_len=8, input_dropout=0., attn_dropout=0.,
                     block_variant=variant, activation="silu" if variant == "hstu_reference" else "elu_plus1",
                     causal_diagonal=diagonal, qk_scale=.7)
    old = HSTU(cfg).eval().requires_grad_(False)
    current = copy.deepcopy(old)
    with torch.no_grad():
        current.item_emb.weight[1:].add_(torch.randn_like(current.item_emb.weight[1:]) * .02)
        current.blocks[0].attn.k_proj.weight.add_(torch.randn_like(current.blocks[0].attn.k_proj.weight)*.03)
    ids = torch.arange(1, length+1)[None]
    behaviors = torch.ones_like(ids)
    deltas = torch.ones_like(ids, dtype=torch.float32)
    sample = dict(parent_cache=old.compute_kv(ids, behaviors, deltas),
                  teacher_cache=current.compute_kv(ids, behaviors, deltas), item_ids=ids,
                  candidates=torch.tensor([[12, 17, 20]]), query_delta=torch.ones(1, 3))
    return current, sample


@pytest.mark.parametrize("variant", ["legacy", "hstu_reference"])
@pytest.mark.parametrize("diagonal", ["inclusive", "exclusive"])
@pytest.mark.parametrize("stage", ["f", "fg", "fgh"])
def test_zero_probe_is_native_including_candidate_self(variant, diagonal, stage):
    model, sample = fixture(variant, diagonal)
    cfg = ProbeConfig.from_model(model, hidden_width=7, query_chunk=2, row_chunk=2)
    probe = ProbeLadder(cfg, stage)
    actual, trace = probe.score(model, sample["parent_cache"], sample["item_ids"],
                                sample["candidates"], sample["query_delta"], trace=True)
    wanted, native = score(model, sample["parent_cache"], sample["candidates"], sample["query_delta"], trace=True)
    torch.testing.assert_close(actual, wanted, atol=5e-7, rtol=3e-6)
    for a, b in zip(trace.new_kv.k, native.new_kv.k, strict=True):
        torch.testing.assert_close(a, b, atol=4e-7, rtol=3e-6)


@pytest.mark.parametrize("stage", ["fg", "fgh"])
@pytest.mark.parametrize("variant", ["legacy", "hstu_reference"])
def test_chunked_history_matches_explicit_effective_kv(stage, variant):
    model, sample = fixture(variant)
    probe = ProbeLadder(ProbeConfig.from_model(model, hidden_width=7, query_chunk=2, row_chunk=2), stage)
    probe.fit_input_statistics(model, [sample])
    with torch.no_grad():
        for modules in (probe.f, probe.g, probe.h):
            if modules is not None:
                for layer in modules:
                    layer.weights[-1].normal_(std=.03)
    source = sample["parent_cache"]
    prepared = probe.prepare_cache(model, source, sample["item_ids"])

    def oracle(layer, attention, q, k, v):
        k = probe._heads(k)[:, :, None]
        v = probe._heads(v)[:, :, None]
        dk, dv = probe.query_correction(layer, q).chunk(2, -1)
        dk, dv = dk[:, :, :, None], dv[:, :, :, None]
        if probe.h is not None:
            rows = probe.row_features(model, source, sample["item_ids"], layer)
            hk, hv = probe.interaction_correction(layer, q, rows).chunk(2, -1)
            dk, dv = dk+hk, dv+hv
        weights = attention._activate((q[:, :, :, None]*(k+dk)).sum(-1)*attention.scale)
        if variant == "hstu_reference":
            weights = weights/attention.cfg.max_seq_len
        return (weights[..., None]*(v+dv)).sum(-2)

    expected, _ = score(model, prepared, sample["candidates"], sample["query_delta"], history_read_fn=oracle)
    actual, _ = probe.score(model, source, sample["item_ids"], sample["candidates"], sample["query_delta"])
    torch.testing.assert_close(actual, expected, atol=6e-7, rtol=4e-6)
    probe.cfg.row_chunk = 1024
    features = probe.prepare_read_features(model, source, sample["item_ids"])
    full_rows, _ = probe.score(model, source, sample["item_ids"], sample["candidates"], sample["query_delta"],
                               read_features=features)
    torch.testing.assert_close(actual, full_rows, atol=6e-7, rtol=4e-6)


def test_nested_initialization_and_once_prepared_f():
    model, sample = fixture()
    cfg = ProbeConfig.from_model(model, hidden_width=6)
    f = ProbeLadder(cfg)
    with torch.no_grad():
        for layer in f.f:
            layer.weights[-1].normal_(std=.02)
    fg, fgh = f.expanded("fg"), f.expanded("fg").expanded("fgh")
    parent = sample["parent_cache"]
    before = parent.k.clone()
    predictions = [p.score(model, parent, sample["item_ids"], sample["candidates"], sample["query_delta"])[0]
                   for p in (f, fg, fgh)]
    torch.testing.assert_close(predictions[0], predictions[1])
    torch.testing.assert_close(predictions[0], predictions[2])
    torch.testing.assert_close(parent.k, before, atol=0, rtol=0)
    prepared = f.prepare_cache(model, parent, sample["item_ids"])
    # Scoring the persisted f cache never calls the row mapper again.
    def forbidden(*_):
        raise AssertionError("f was re-executed on a read")
    for layer in f.f:
        layer.forward = forbidden
    actual, _ = f.score(model, parent, sample["item_ids"], sample["candidates"],
                        sample["query_delta"], prepared_cache=prepared)
    torch.testing.assert_close(actual, predictions[0])


def test_h_is_nonlinear_joint_mapping_not_a_low_rank_product():
    model, sample = fixture()
    probe = ProbeLadder(ProbeConfig.from_model(model, hidden_width=12, normalize=False), "fgh")
    with torch.no_grad():
        probe.h[0].weights[-1].normal_(std=.1)
    rows = probe.row_features(model, sample["parent_cache"], sample["item_ids"], 0)
    q = torch.randn(1, 2, 3, 4)
    value = probe.interaction_correction(0, q, rows)
    cross = value[:, :, 0, 0] - value[:, :, 0, 1] - value[:, :, 1, 0] + value[:, :, 1, 1]
    assert cross.abs().max() > 1e-6
    assert value.shape == (1, 2, 3, 5, 8)


def test_append_uses_own_corrected_reads_and_fifo_without_refitting_f():
    model, sample = fixture(length=5)
    probe = ProbeLadder(ProbeConfig.from_model(model, hidden_width=8, row_chunk=2), "fgh")
    with torch.no_grad():
        for modules in (probe.f, probe.g, probe.h):
            for layer in modules:
                layer.weights[-1].normal_(std=.02)
    source, ids = sample["parent_cache"], sample["item_ids"]
    cache = probe.prepare_cache(model, source, ids)
    for layer in probe.f:
        layer.forward = lambda *_: (_ for _ in ()).throw(AssertionError("f re-executed at append"))
    for item in range(21, 27):
        before = cache
        cache, source, ids, result = probe.append(model, cache, source, ids, torch.tensor([[item]]),
                                                 torch.ones(1, 1, dtype=torch.long), torch.ones(1, 1),
                                                 max_length=5, trace=True)
        assert cache.seq_len == 5
        assert ids.shape[1] == source.seq_len
        torch.testing.assert_close(cache.k[:, :, -1:], result.new_kv.k)
        # A new row is the actual corrected-path write, not teacher copied KV.
        assert not torch.equal(cache.k[:, :, -1:], before.k[:, :, -1:])
    assert source.seq_len == 0
    # Once the inherited prefix has left, only the native history read remains.
    _, qtrace = score(model, cache, sample["candidates"], sample["query_delta"], trace=True)
    callback = probe.make_reader(model, source, ids, cache)
    actual = callback(0, model.blocks[0].attn, qtrace.queries[0], cache.k[0], cache.v[0])
    expected = history_read(model.blocks[0].attn, qtrace.queries[0], cache.k[0], cache.v[0])
    torch.testing.assert_close(actual, expected)


def test_fitting_caches_targets_and_updates_only_probe_with_accounting():
    model, sample = fixture()
    second = dict(sample, candidates=torch.tensor([[13, 18, 22]]))
    probe = ProbeLadder(ProbeConfig.from_model(model, hidden_width=8), "fgh")
    targets = prepare_training_samples(model, [sample])
    assert "teacher_scores" in targets[0] and "read_scales" in targets[0]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    report = fit_probe(model, probe, [sample], [second], steps=4, validation_every=2,
                       patience=0, learning_rate=1e-3)
    assert report["completed_steps"] == 4
    assert report["best_validation_loss"] <= report["history"][0]["validation_loss"]
    assert report["operation_counts"]["training_backward_calls"] == 4
    assert report["total_calibration_flops_estimate"] > 0
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in probe.h.parameters())
    with torch.no_grad():
        value, metrics = probe_loss(model, probe, targets[0])
    assert torch.isfinite(value)
    assert set(metrics) == {"read_mse", "score_mse", "raw_score_mse"}
    assert "score_scale" in targets[0]


@pytest.mark.parametrize("stage", ["f", "fg", "fgh"])
def test_zero_probe_replay_matches_native_ties_eviction_and_candidate_isolation(stage):
    current, _ = fixture()
    parent = copy.deepcopy(current)
    cfg = dict(release_day=1, end_day=2, edge=1, history_length=5, append_band_size=2, query_batch=2)
    times = np.array([86390,86391,86392,86393,86394,86400,86401,86402,86403,86404,86405])
    events = times, np.arange(1,len(times)+1), np.ones(len(times),dtype=np.int64)
    probe = ProbeLadder(ProbeConfig.from_model(current, hidden_width=6, row_chunk=2), stage)
    trajectory = ProbeTrajectory(parent,current,probe,events,cfg)
    native = NativeTrajectory(parent,0,events,86400,max_length=5,append_band_size=2)
    native.switch(current,1,86400)
    candidates = torch.tensor([[17,19]])
    for timestamp in (86400,86401,86403,86406):
        trajectory.advance(timestamp)
        native.advance(timestamp)
        gap = torch.tensor([timestamp-native.last_timestamp],dtype=torch.float32)
        old_k, old_cursor = trajectory.cache.k.clone(), trajectory.native.cursor
        actual = trajectory.score(candidates,gap)
        expected,_ = score(current,native.cache,candidates,gap)
        torch.testing.assert_close(actual,expected,atol=5e-7,rtol=5e-6)
        torch.testing.assert_close(trajectory.cache.k,old_k,atol=0,rtol=0)
        assert trajectory.native.cursor == old_cursor
        torch.testing.assert_close(trajectory.cache.k,native.cache.k,atol=5e-7,rtol=5e-6)
    assert trajectory.source.seq_len == 0
    assert trajectory.report()["counts"]["prepare_calls"] == 1
    assert trajectory.report()["counts"]["appended_rows"] == 6


def test_probe_replay_outputs_real_feedback_and_processes_event_only_tail():
    current,_ = fixture()
    cfg = dict(release_day=1,end_day=2,edge=1,history_length=5,append_band_size=2,query_batch=2)
    times = np.array([86390,86391,86392,86400,86401,86402])
    events = times,np.arange(1,7),np.ones(6,dtype=np.int64)
    probe = ProbeLadder(ProbeConfig.from_model(current,hidden_width=5),"fgh")
    requests = {86400:[dict(item_idx=17,label=1,query_timestamp=86400),
                      dict(item_idx=19,label=0,query_timestamp=86400)]}
    rows,report = replay_probe_user(current,current,probe,events,requests,99,cfg)
    assert [row["label"] for row in rows] == [1,0]
    assert all(row["uid"] == 99 and row["method"] == "fgh" for row in rows)
    assert report["counts"]["appended_rows"] == 3
    assert report["counts"]["candidate_reads"] == 2


def test_fit_pool_scales_are_shared_across_lengths_and_zero_difference_validation():
    model,short=fixture(length=2)
    _,long=fixture(length=8)
    # This validation teacher has exactly the source cache: its zero drift
    # must not supply a tiny per-user denominator for a nonzero correction.
    validation=dict(short,teacher_cache=short["parent_cache"],
                    read_scales=torch.full((2,1,2,1,1),1e-12),score_scale=torch.tensor(1e-12))
    train,val,metadata=bind_fit_scales(model,[short,long],[validation])
    assert metadata["users"] == 2
    assert metadata["source"] == "fitting_users_only"
    assert metadata["validation_uses_fitting_scales"]
    for sample in train+val:
        torch.testing.assert_close(sample["read_scales"],train[0]["read_scales"],atol=0,rtol=0)
        torch.testing.assert_close(sample["score_scale"],train[0]["score_scale"],atol=0,rtol=0)
    assert val[0]["read_moment_sum"].abs().max() == 0
    assert val[0]["score_moment_sum"] == 0
    expected=((train[0]["read_moment_sum"].double()+train[1]["read_moment_sum"].double())/2).sqrt().clamp_min(1e-4)
    torch.testing.assert_close(train[0]["read_scales"][:,0,:,0,0].double(),expected)
    probe=ProbeLadder(ProbeConfig.from_model(model,hidden_width=8),"fgh")
    with torch.no_grad():
        for modules in (probe.f,probe.g,probe.h):
            for layer in modules:
                layer.weights[-1].normal_(std=.001)
        loss,parts=probe_loss(model,probe,val[0])
    assert torch.isfinite(loss)
    assert float(parts["read_mse"])<1e4
    assert float(parts["score_mse"])<1e4


def test_fit_probe_overwrites_prepared_validation_pool_scales():
    model,short=fixture(length=2)
    _,long=fixture(length=8)
    validation=dict(short,teacher_cache=short["parent_cache"])
    train=prepare_training_samples(model,[short,long])
    val=prepare_training_samples(model,[validation])
    assert float(val[0]["score_scale"]) <= 1.0001e-4
    expected=float(train[0]["score_scale"])
    probe=ProbeLadder(ProbeConfig.from_model(model,hidden_width=7),"f")
    report=fit_probe(model,probe,train,val,steps=1,validation_every=1,patience=0)
    assert report["fit_pool_loss_scales"]["score_scale"] == expected
    assert report["fit_pool_loss_scales"]["users"] == 2
    assert report["history"][0]["validation_loss"] == 0
    assert report["history"][1]["validation_loss"] < 1e4
