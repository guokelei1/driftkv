"""Current item gates on real causal writer bands and transient candidate reads."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[4]

from tests.methods.cachetide.relay.test_session import fixtures
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.methods.cachetide.adapter import item_gate_query
from hstu_kvcache.methods.cachetide.session import RelaySession, append_corrected_event


def observe_gate_inputs(monkeypatch,adapter):
    calls=[[] for _ in adapter.layers]
    for index,layer in enumerate(adapter.layers):
        original=layer.query_gates
        def record(query,*,function=original,index=index):
            calls[index].append(query.detach().clone())
            return function(query)
        monkeypatch.setattr(layer,'query_gates',record)
    return calls


@torch.no_grad()
def test_writer_band_uses_each_current_event_item_and_preserves_fifo(monkeypatch):
    models,adapters,events,make=fixtures(monkeypatch,mode='A')
    adapter=adapters[0]
    adapter.set_gate_source('item')
    adapter.set_gate_normalization('rms')
    calls=observe_gate_inputs(monkeypatch,adapter)
    session=RelaySession(make())
    session.publish(models[1],3,40,adapter)
    session.prepare_codes()
    before,state=session.cache,session.state
    session.advance(51)
    wanted=item_gate_query(models[1],torch.tensor([[5,6]]))
    for layer_calls in calls:
        assert len(layer_calls)==1
        torch.testing.assert_close(layer_calls[0],wanted,atol=0,rtol=0)
    assert not torch.equal(wanted,item_gate_query(models[0],torch.tensor([[5,6]])))
    assert not torch.equal(wanted[:,:,0],wanted[:,:,1])
    assert session.costs['writer_gate_item_lookup_bytes']==2*12*4
    torch.testing.assert_close(session.cache.k[:,:,:2],before.k[:,:,-2:],atol=0,rtol=0)
    torch.testing.assert_close(session.cache.v[:,:,:2],before.v[:,:,-2:],atol=0,rtol=0)
    torch.testing.assert_close(session.state.codes[:,:,:2],state.codes[:,:,-2:],atol=0,rtol=0)
    assert session.coded_rows==2
    assert not torch.count_nonzero(session.state.codes[:,:,2:])
    oracle,oracle_codes=before,state
    count=4
    for item in (5,6):
        oracle,oracle_codes,_=append_corrected_event(models[1],oracle,oracle_codes,adapter,
            torch.tensor([[item]]),torch.tensor([[0]]),torch.tensor([[10.]]),4,code_count=count)
        count-=1
    torch.testing.assert_close(session.cache.k,oracle.k,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(session.cache.v,oracle.v,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(session.state.codes,oracle_codes.codes,atol=0,rtol=0)


@torch.no_grad()
def test_future_band_item_cannot_change_previous_event_native_state(monkeypatch):
    models,adapters,events,make=fixtures(monkeypatch,mode='A')
    adapter=adapters[0]
    adapter.set_gate_source('item')
    first=RelaySession(make())
    changed_items=events[1].copy()
    changed_items[5]=29
    altered=NativeTrajectory(models[0],2,(events[0],changed_items,events[2]),40,
                               max_length=4,append_band_size=4)
    second=RelaySession(altered)
    for session in (first,second):
        session.publish(models[1],3,40,adapter)
        session.advance(51)
    # Both bands append item5 then another item. Item5 lands at position2
    # after FIFO eviction and must not depend on the later band's item.
    torch.testing.assert_close(first.cache.k[:,:,2],second.cache.k[:,:,2],atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(first.cache.v[:,:,2],second.cache.v[:,:,2],atol=2e-6,rtol=2e-5)
    assert not torch.equal(first.cache.k[0,:,3],second.cache.k[0,:,3])


@torch.no_grad()
def test_candidate_read_uses_each_current_item_and_writes_no_cache(monkeypatch):
    models,adapters,_,make=fixtures(monkeypatch,mode='A')
    adapter=adapters[0]
    adapter.set_gate_source('item')
    adapter.set_gate_normalization('rms')
    calls=observe_gate_inputs(monkeypatch,adapter)
    session=RelaySession(make())
    session.publish(models[1],3,40,adapter)
    native=session.cache.k.clone(),session.cache.v.clone()
    candidates=torch.tensor([[30,31,32]])
    logits=session.score(40,candidates)
    wanted=item_gate_query(models[1],candidates)
    for layer_calls in calls:
        assert len(layer_calls)==1
        torch.testing.assert_close(layer_calls[0],wanted,atol=0,rtol=0)
    assert session.costs['ranking_gate_item_lookup_bytes']==3*12*4
    direct=score(models[1],session.cache,candidates,torch.tensor([10.]),
        history_read_fn=adapter.make_history_read(session.state,code_count=session.coded_rows,
                                                   gate_query=wanted))[0]
    torch.testing.assert_close(logits,direct,atol=0,rtol=0)
    torch.testing.assert_close(session.cache.k,native[0],atol=0,rtol=0)
    torch.testing.assert_close(session.cache.v,native[1],atol=0,rtol=0)


@pytest.mark.parametrize('source',['layer','first'])
@torch.no_grad()
def test_nonitem_session_sources_keep_direct_read_behavior(monkeypatch,source):
    models,adapters,_,make=fixtures(monkeypatch,mode='A')
    adapter=adapters[0]
    adapter.set_gate_source(source)
    session=RelaySession(make())
    session.publish(models[1],3,40,adapter)
    session.prepare_codes()
    candidates=torch.tensor([[30,31]])
    expected=score(models[1],session.cache,candidates,torch.tensor([10.]),
                   history_read_fn=adapter.make_history_read(session.state,
                                                              code_count=session.coded_rows))[0]
    got=session.score(40,candidates)
    torch.testing.assert_close(got,expected,atol=0,rtol=0)
    assert not session.costs['ranking_gate_item_lookup_bytes']
    assert not session.costs['writer_gate_item_lookup_bytes']
