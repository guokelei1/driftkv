"""Nonzero masked full-window probe reads/writes against the eager oracle."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.probe_fast import (
    FastProbeTrajectory, GraphProbePool, PaddedProbeGraph, pad_features, replay_fast_probe_user,
)
from experiments.ch3_motivation.code.probe_replay import ProbeTrajectory, replay_probe_user
from experiments.ch3_motivation.code.probes import ProbeConfig, ProbeLadder
from hstu_kvcache.models import HSTU,HSTUConfig,HSTUKVCache


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(stage="fgh",variant="legacy"):
    torch.manual_seed(744)
    cfg=HSTUConfig(num_items=30,num_behaviors=2,hidden_size=8,num_heads=2,num_layers=2,
        max_seq_len=5,input_dropout=0.,attn_dropout=0.,block_variant=variant,
        activation="silu" if variant=="hstu_reference" else "elu_plus1",qk_scale=.7)
    model=HSTU(cfg).eval().requires_grad_(False)
    ids=torch.arange(1,6)[None]
    cache=model.compute_kv(ids,torch.ones_like(ids),torch.ones_like(ids,dtype=torch.float32))
    probe=ProbeLadder(ProbeConfig.from_model(model,hidden_width=7,depth=1,row_chunk=2),stage)
    with torch.no_grad():
        for modules in (probe.f,probe.g,probe.h):
            if modules is not None:
                for layer in modules:
                    layer.weights[-1].normal_(std=.02)
    return model,probe,cache,ids


@pytest.mark.parametrize("stage",["f","fg","fgh"])
@pytest.mark.parametrize("variant",["legacy","hstu_reference"])
def test_fixed_mask_score_matches_nonzero_reference_for_all_inherited_counts(stage,variant):
    model,probe,original,ids=inputs(stage,variant)
    prepared=probe.prepare_cache(model,original,ids)
    graph=PaddedProbeGraph(model,probe,operation="score",window=5,queries=3,capture=False)
    candidates=torch.tensor([[13,17,19]])
    delta=torch.ones(1,3)
    with torch.no_grad():
        for inherited in (5,3,1,0):
            source=HSTUKVCache(original.k[:,:,:inherited],original.v[:,:,:inherited],inherited)
            source_ids=ids[:,:inherited]
            features=probe.prepare_read_features(model,source,source_ids)
            actual,=graph.run(prepared,pad_features(features,5),inherited,candidates,delta)
            expected,_=probe.score(model,source,source_ids,candidates,delta,
                                    prepared_cache=prepared,read_features=features)
            torch.testing.assert_close(actual,expected,atol=7e-7,rtol=5e-6)
    assert graph.replays==4
    assert graph.report()["physical_read_rows"]==5
    if stage=="fgh":
        assert graph.report()["actual_h_mlp_row_pairs_per_replay"]==2*2*3*5


@pytest.mark.parametrize("stage",["f","fg","fgh"])
@pytest.mark.parametrize("variant",["legacy","hstu_reference"])
def test_three_nonzero_fifo_appends_and_candidate_reads_match_reference(stage,variant):
    model,probe,source,ids=inputs(stage,variant)
    reference=probe.prepare_cache(model,source,ids)
    actual=HSTUKVCache(reference.k.clone(),reference.v.clone(),5)
    features=probe.prepare_read_features(model,source,ids)
    padded=pad_features(features,5)
    pool=GraphProbePool(model,probe,window=5,capture=False)
    inherited=5
    with torch.no_grad():
        for item in (21,22,23):
            token=torch.tensor([[item]])
            behavior=torch.ones(1,1,dtype=torch.long)
            delta=torch.ones(1,1)
            k,v,padded=pool.get("append",1).run(actual,padded,inherited,token,delta,behavior)
            actual=HSTUKVCache(k,v,5)
            reference,source,ids,_=probe.append(model,reference,source,ids,token,behavior,delta,
                                                max_length=5,read_features=features)
            features=tuple(rows[:,:,1:] for rows in features)
            inherited-=1
            torch.testing.assert_close(actual.k,reference.k,atol=7e-7,rtol=5e-6)
            torch.testing.assert_close(actual.v,reference.v,atol=7e-7,rtol=5e-6)
            candidates=torch.tensor([[13,17,19]])
            previous=actual.k.clone()
            values,=pool.get("score",3).run(actual,padded,inherited,candidates,delta)
            wanted,_=probe.score(model,source,ids,candidates,delta[:,0],
                                  prepared_cache=reference,read_features=features)
            torch.testing.assert_close(values,wanted,atol=7e-7,rtol=5e-6)
            torch.testing.assert_close(actual.k,previous,atol=0,rtol=0)
    # The varying inherited count has no effect on graph keys.
    assert len(pool.graphs)==2


@pytest.mark.parametrize("stage",["f","fg","fgh"])
def test_fast_trajectory_short_window_transition_and_event_ties(stage):
    model,probe,_,_=inputs(stage)
    parent=copy.deepcopy(model)
    cfg=dict(edge=1,release_day=1,end_day=2,history_length=5,append_band_size=2)
    times=np.array([86390,86391,86392,86400,86401,86402,86403,86404,86405,86406])
    events=times,np.arange(1,len(times)+1),np.ones(len(times),dtype=np.int64)
    pool=GraphProbePool(model,probe,window=5,capture=False)
    fast=FastProbeTrajectory(parent,model,probe,events,cfg,pool=pool)
    reference=ProbeTrajectory(parent,model,probe,events,cfg)
    candidates=torch.tensor([[17,19]])
    with torch.no_grad():
        for timestamp in (86400,86401,86403,86404,86406,86408):
            fast.advance(timestamp)
            reference.advance(timestamp)
            delta=torch.tensor([timestamp-reference.native.last_timestamp],dtype=torch.float32)
            wanted=reference.score(candidates,delta)
            actual=fast.score(candidates,delta)
            torch.testing.assert_close(actual,wanted,atol=9e-7,rtol=8e-6)
            torch.testing.assert_close(fast.cache.k,reference.cache.k,atol=9e-7,rtol=8e-6)
            assert fast.source.seq_len==reference.source.seq_len
    if stage!="f":
        report=fast.report()
        assert report["counts"]["fixed_graph_append_calls"]>0
        assert report["counts"]["eager_corrected_append_calls"]>0
        assert report["padded_read_shapes"]
    assert fast.source.seq_len==0


def test_quality_shortcut_retains_scores_and_skips_unscored_tail():
    model,probe,_,_=inputs("fgh")
    cfg=dict(edge=1,release_day=1,end_day=2,history_length=5,append_band_size=2,query_batch=2)
    times=np.array([86390,86391,86392,86400,86401,86402])
    events=times,np.arange(1,7),np.ones(6,dtype=np.int64)
    rows={86400:[dict(item_idx=17,label=1,query_timestamp=86400),
                 dict(item_idx=19,label=0,query_timestamp=86400)]}
    pool=GraphProbePool(model,probe,window=5,capture=False)
    actual,report=replay_fast_probe_user(model,model,probe,events,rows,77,cfg,pool=pool)
    expected,full=replay_probe_user(model,model,probe,events,rows,77,cfg)
    torch.testing.assert_close(torch.tensor([r["score"] for r in actual]),
                               torch.tensor([r["score"] for r in expected]))
    assert full["counts"]["appended_rows"]==3
    assert report["counts"].get("appended_rows",0)==0
    assert report["quality_shortcut"]["stop_after_last_query"]
    complete,without_shortcut=replay_fast_probe_user(model,model,probe,events,rows,77,cfg,
        pool=pool,stop_after_last_query=False)
    assert without_shortcut["counts"]["appended_rows"]==3
    assert complete==actual


def test_no_feedback_quality_shortcut_never_prefills(monkeypatch):
    model,probe,_,_=inputs("fgh")
    cfg=dict(edge=1,release_day=1,end_day=2,history_length=5,append_band_size=2)
    events=np.array([86390,86400]),np.array([1,2]),np.array([1,1])
    pool=GraphProbePool(model,probe,window=5,capture=False)
    def forbidden(*args,**kwargs):
        raise AssertionError("no-feedback quality user performed a GPU prefill")
    monkeypatch.setattr(model,"compute_kv",forbidden)
    rows,report=replay_fast_probe_user(model,model,probe,events,{},77,cfg,pool=pool)
    assert rows==[]
    assert report["counts"]["quality_skipped_no_feedback_users"]==1
    assert report["initial_shapes"]==[]
