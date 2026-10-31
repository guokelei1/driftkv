"""First-affine caching preserves ordinary h, native self and causal writes."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.probe_variants import build_variant
from experiments.ch3_motivation.code.probe_fast import PaddedProbeGraph, pad_features
from experiments.ch3_motivation.code.probe_replay import ProbeTrajectory
from experiments.ch3_motivation.code.probe_projected_fast import (
    project_row_features, projected_interaction_correction, projected_row_preparation_flops,
    projected_read_extra_flops, ProjectedPaddedGraph, ProjectedGraphPool, ProjectedFastProbeTrajectory,
    evaluate_projected_probe,
)
from experiments.ch3_motivation.code.probe_cost import padded_read_extra_flops
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTU,HSTUConfig,HSTUKVCache


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    threads=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fixture(variant="full_item",stage="fgh",depth=1,normalize=True,initial=5):
    torch.manual_seed(145)
    cfg=HSTUConfig(num_items=32,num_behaviors=2,hidden_size=8,num_heads=2,
        num_layers=2,max_seq_len=5,input_dropout=0.,attn_dropout=0.)
    parent=HSTU(cfg).eval().requires_grad_(False)
    model=copy.deepcopy(parent)
    with torch.no_grad():
        model.item_emb.weight[1:].add_(torch.randn_like(model.item_emb.weight[1:])*.02)
    ids=torch.arange(1,initial+1)[None]
    delta=torch.ones_like(ids,dtype=torch.float32);delta[:,0]=0
    sample=dict(parent_cache=parent.compute_kv(ids,torch.ones_like(ids),delta),
        item_ids=ids,candidates=torch.tensor([[15,17,19]]),query_delta=torch.ones(1))
    probe=build_variant(variant,model,stage=stage,hidden_width=7,depth=depth,normalize=normalize,
                        query_chunk=1,row_chunk=2)
    probe.fit_input_statistics(model,[sample])
    return parent,model,sample,probe


def nonzero(probe):
    with torch.no_grad():
        for terms in (probe.f,probe.g,probe.h):
            if terms is not None:
                for layer in terms:
                    if hasattr(layer,"weights"):
                        layer.weights[-1].normal_(std=.01)
                        # Exercise nonzero first and later biases too.
                        for bias in layer.biases:
                            bias.normal_(std=.02)
                    else:
                        layer.network[-1].weight.normal_(std=.01)


@pytest.mark.parametrize("variant",["head","full_item","layerwise_kv"])
@pytest.mark.parametrize("depth",[1,2,3])
@pytest.mark.parametrize("normalize",[False,True])
def test_projected_h_matches_complete_mlp_and_query_gradient(variant,depth,normalize):
    _,model,sample,probe=fixture(variant,depth=depth,normalize=normalize)
    probe=probe.double()
    nonzero(probe)
    # Arbitrary already-normalized fixed rows and query scales verify the
    # affine identity independently of the transformer reader's outputs.
    rows=tuple(torch.randn(2,2,5,probe.cfg.row_dim,dtype=torch.float64) for _ in range(2))
    with torch.no_grad():
        probe.query_scale.uniform_(.3,2.)
        probe.row_scale.uniform_(.4,3.)
        if hasattr(probe,"kv_output_scale"):
            probe.kv_output_scale.uniform_(.5,4.)
    projected=project_row_features(probe,rows)
    for layer in range(2):
        q=torch.randn(2,2,3,4,dtype=torch.float64,requires_grad=True)
        expected=probe.interaction_correction(layer,q,rows[layer])
        actual=projected_interaction_correction(probe,layer,q,projected[layer])
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        a=torch.autograd.grad(actual.square().sum(),q,retain_graph=True)[0]
        b=torch.autograd.grad(expected.square().sum(),q)[0]
        torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)


@pytest.mark.parametrize("variant",["head","full_item","layerwise_kv"])
@pytest.mark.parametrize("depth",[1,2])
def test_zero_and_nonzero_fixed_graph_all_counts_reads_logits_and_append(variant,depth):
    _,model,sample,probe=fixture(variant,depth=depth)
    features=probe.prepare_read_features(model,sample["parent_cache"],sample["item_ids"])
    raw=PaddedProbeGraph(model,probe,operation="score",window=5,queries=3,capture=False)
    projected=ProjectedPaddedGraph(model,probe,operation="score",window=5,queries=3,capture=False)
    append=ProjectedPaddedGraph(model,probe,operation="append",window=5,queries=1,capture=False)
    item=torch.tensor([[9]]);behavior=torch.ones_like(item);delta=torch.ones(1,1)
    with torch.no_grad():
        for active in (False,True):
            if active:
                nonzero(probe)
            prepared=probe.prepare_cache(model,sample["parent_cache"],sample["item_ids"],read_features=features)
            rows=pad_features(project_row_features(probe,features),5)
            for count in range(6):
                native,=raw.run(prepared,pad_features(features,5),count,sample["candidates"],sample["query_delta"])
                actual,=projected.run(prepared,rows,count,sample["candidates"],sample["query_delta"])
                assert float((actual-native).abs().max())<3e-5
                # Check historical reads and propagated queries, not logits alone.
                native_trace=score(model,prepared,sample["candidates"],sample["query_delta"],trace=True,history_read_fn=raw._read)[1]
                actual_trace=score(model,prepared,sample["candidates"],sample["query_delta"],trace=True,history_read_fn=projected._read)[1]
                for a,b in zip(actual_trace.history_heads,native_trace.history_heads,strict=True):
                    assert float((a-b).norm()/b.norm().clamp_min(1e-6))<2e-5
                source=HSTUKVCache(sample["parent_cache"].k[:,:,:count],sample["parent_cache"].v[:,:,:count],count)
                fs=tuple(value[:,:,:count] for value in features)
                k,v,next_rows=append.run(prepared,rows,count,item,delta,behavior)
                expected,_,_,_=probe.append(model,prepared,source,sample["item_ids"][:,:count],item,behavior,delta,
                    max_length=5,read_features=fs)
                torch.testing.assert_close(k,expected.k,atol=3e-5,rtol=2e-5)
                torch.testing.assert_close(v,expected.v,atol=3e-5,rtol=2e-5)
                torch.testing.assert_close(next_rows[...,:4,:],rows[...,1:,:],atol=0,rtol=0)
            if not active:
                wanted=score(model,sample["parent_cache"],sample["candidates"],sample["query_delta"])[0]
                torch.testing.assert_close(actual,wanted,atol=3e-5,rtol=2e-5)
    assert projected.report()["row_projection_input_bytes_per_replay"]==2*1*2*5*7*4


@pytest.mark.parametrize("variant",["head","full_item","layerwise_kv"])
@pytest.mark.parametrize("stage",["f","fg","fgh"])
@pytest.mark.parametrize("initial",[2,5])
def test_projected_trajectory_full_fifo_short_fallback_and_source_retirement(variant,stage,initial):
    parent,model,sample,probe=fixture(variant,stage=stage,depth=2,initial=initial)
    nonzero(probe)
    times=np.r_[np.arange(86400-initial,86400),np.arange(86400,86409)]
    events=times,np.arange(1,len(times)+1),np.ones(len(times),dtype=np.int64)
    cfg=dict(edge=1,release_day=1,end_day=2,history_length=5,append_band_size=2,query_batch=3)
    pool=ProjectedGraphPool(model,probe,window=5,capture=False)
    actual=ProjectedFastProbeTrajectory(parent,model,probe,events,cfg,pool=pool)
    reference=ProbeTrajectory(parent,model,probe,events,cfg)
    with torch.no_grad():
        # f's one-time lazy materialization is allowed to publish cache state;
        # subsequent candidate reads must leave that published state intact.
        actual._prepare();reference._prepare()
        for timestamp in range(86400,86409):
            actual.advance(timestamp);reference.advance(timestamp)
            previous=actual.cache.k.clone()
            a=actual.score(sample["candidates"],sample["query_delta"])
            b=reference.score(sample["candidates"],sample["query_delta"])
            assert float((a-b).abs().max())<3e-5
            torch.testing.assert_close(actual.cache.k,previous,atol=0,rtol=0)
            torch.testing.assert_close(actual.cache.k,reference.cache.k,atol=3e-5,rtol=2e-5)
            torch.testing.assert_close(actual.cache.v,reference.cache.v,atol=3e-5,rtol=2e-5)
    assert actual.source.seq_len==reference.source.seq_len==0
    report=actual.report()
    if stage=="fgh":
        assert report["counts"]["projected_row_prepare_calls"]==1
        assert report["projected_execution"]["extra_row_preparation_flops"]==projected_row_preparation_flops(probe,initial)
    else:
        assert report["projected_execution"]["extra_row_preparation_flops"]==0
    assert report["projected_execution"]["retained_tensor_storage_bytes"]>0
    assert len(pool.graphs)<=2


def test_projected_cost_is_full_hidden_table_and_keeps_dense_tail():
    _,model,_,probe=fixture("layerwise_kv",depth=2)
    c=probe.cfg
    expected_prep=2*1024*c.num_layers*c.num_heads*c.row_dim*c.hidden_width
    assert projected_row_preparation_flops(probe,1024)==expected_prep
    # At one row, one query, amortization need not be a speedup. At full
    # shape it removes repeated row projection while retaining the whole MLP.
    assert projected_read_extra_flops(probe,1024,16)<padded_read_extra_flops(probe,1024,16)
    graph=ProjectedPaddedGraph(model,probe,operation="append",window=5,queries=1,capture=False)
    report=graph.report()
    assert report["projected_feature_width"]==c.hidden_width
    assert report["actual_h_mlp_row_pairs_per_replay"]==c.num_layers*c.num_heads*4


def test_projected_quality_shard_skips_idle_user_without_allocating_graph():
    parent,model,_,probe=fixture()
    cfg=dict(edge=1,release_day=1,end_day=2,history_length=5,append_band_size=2,query_batch=3)
    pool=ProjectedGraphPool(model,probe,window=5,capture=False)
    histories=type("Histories",(),{"rows":{11:None}})()
    scores,report=evaluate_projected_probe(parent,model,probe,histories,{},[11],cfg,pool=pool)
    assert scores==[] and pool.graphs=={}
    assert report["projected_execution"]["extra_row_preparation_flops"]==0
    assert report["quality_shortcut"]["skipped_no_feedback_users"]==1
