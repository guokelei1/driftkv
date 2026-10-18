"""Operation-local native first-query gates, with unchanged layer projections."""
import copy
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))

from code_structure_2026_10.structure import StructuredKVAdapter
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTU,HSTUConfig,HSTUKVCache,PointwiseAttention,PointwiseAttentionConfig
from query_relay_2026_10.adapter import UnifiedRelayAdapter
from query_relay_2026_10.calibrate import gate_statistics
from query_relay_2026_10.evaluate import GateStatistics
from query_relay_2026_10.session import append_corrected_event


def actor(layers=2,source='first'):
    torch.manual_seed(20261015)
    origin=StructuredKVAdapter(layers,6,4,layout='shared12_private6')
    with torch.no_grad():
        for layer in origin.layers:
            layer.input_weight.normal_(std=.07)
            layer.input_bias.normal_(std=.02)
            layer.decoder.normal_(std=.03)
            layer.key_decoder.normal_(std=.03)
    result=UnifiedRelayAdapter.from_structured(origin,mode='A',gate_source=source,
                                               gate_normalization='rms').double()
    with torch.no_grad():
        for layer in result.layers:
            layer.key_gate_weight.normal_(std=.3)
            layer.value_gate_weight.normal_(std=.3)
    return result


def test_first_gate_keeps_layer_query_scores_gradients_and_independent_rows():
    adapter=actor()
    cache=HSTUKVCache(torch.randn(2,2,4,24,dtype=torch.float64),
                       torch.randn(2,2,4,24,dtype=torch.float64),4)
    features=torch.randn(2,4,24,dtype=torch.float64)
    state=adapter.encode_initial(cache,features)
    attention=PointwiseAttention(PointwiseAttentionConfig(24,6,head_dim=4)).double().eval()
    q0=torch.randn(2,6,3,4,dtype=torch.float64,requires_grad=True)
    q1=torch.randn_like(q0,requires_grad=True)
    read=adapter.make_history_read(state)
    read(0,attention,q0,cache.k[0],cache.v[0])
    actual=read(1,attention,q1,cache.k[1],cache.v[1])
    expected=adapter.layers[1].read(attention,q1,cache.k[1],cache.v[1],
                                    state.codes[1],gate_q=q0)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    dq0,dq1=torch.autograd.grad(actual.square().mean(),(q0,q1),retain_graph=True)
    assert dq0.abs().sum()>0 and dq1.abs().sum()>0
    changed=q0.detach().clone()
    changed[0,:,1]+=1
    read(0,attention,changed,cache.k[0],cache.v[0])
    following=read(1,attention,q1,cache.k[1],cache.v[1])
    torch.testing.assert_close(following[1],actual[1],atol=0,rtol=0)
    torch.testing.assert_close(following[0,:,[0,2]],actual[0,:,[0,2]],atol=0,rtol=0)
    assert (following[0,:,1]-actual[0,:,1]).abs().max()>1e-8


def test_old_payload_default_and_single_layer_first_are_exact():
    original=actor(layers=1,source='layer')
    config=original.get_config()
    config.pop('gate_source')
    restored=UnifiedRelayAdapter(**config).double()
    restored.load_state_dict(original.state_dict())
    first=copy.deepcopy(original)
    first.set_gate_source('first')
    cache=HSTUKVCache(torch.randn(1,1,4,24,dtype=torch.float64),
                       torch.randn(1,1,4,24,dtype=torch.float64),4)
    features=torch.randn(1,4,24,dtype=torch.float64)
    state=original.encode_initial(cache,features)
    attention=PointwiseAttention(PointwiseAttentionConfig(24,6,head_dim=4)).double().eval()
    q=torch.randn(1,6,3,4,dtype=torch.float64)
    baseline=original.make_history_read(state)(0,attention,q,cache.k[0],cache.v[0])
    for candidate in (restored,first):
        got=candidate.make_history_read(state)(0,attention,q,cache.k[0],cache.v[0])
        torch.testing.assert_close(got,baseline,atol=0,rtol=0)
        torch.testing.assert_close(candidate.encode_initial(cache,features).codes,state.codes,atol=0,rtol=0)
    assert restored.gate_source=='layer'
    assert first.get_config()['gate_source']=='first'


def test_real_query_and_event_q0_predates_history_and_statistics_match(monkeypatch):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND','torch')
    torch.manual_seed(20261015)
    model=HSTU(HSTUConfig(num_items=24,num_behaviors=2,hidden_size=24,num_heads=6,
                          num_layers=2,max_seq_len=6,input_dropout=0)).double().eval().requires_grad_(False)
    adapter=actor()
    candidates=torch.tensor([[15,16,17]])
    delta=torch.tensor([7.],dtype=torch.float64)
    histories=[]
    for items in (torch.tensor([[1,2,3,4]]),torch.tensor([[4,5,6,7]])):
        cache=model.compute_kv(items,torch.ones_like(items),torch.ones_like(items,dtype=torch.float64))
        features=model.lookup_item_embeddings(items)
        state=adapter.encode_initial(cache,features)
        logits,trace=score(model,cache,candidates,delta,trace=True,
                            history_read_fn=adapter.make_history_read(state))
        histories.append((cache,state,trace,features))
    torch.testing.assert_close(histories[0][2].queries[0],histories[1][2].queries[0],atol=0,rtol=0)
    assert not torch.equal(histories[0][2].queries[1],histories[1][2].queries[1])
    for index,layer in enumerate(adapter.layers):
        a=layer.query_gates(histories[0][2].queries[0])
        b=layer.query_gates(histories[1][2].queries[0])
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    event_traces=[]
    for cache,state,_,_ in histories:
        _,_,trace=append_corrected_event(model,cache,state,adapter,torch.tensor([[9]]),
            torch.tensor([[1]]),torch.tensor([[2.]],dtype=torch.float64),6,trace=True)
        event_traces.append(trace)
    torch.testing.assert_close(event_traces[0].queries[0],event_traces[1].queries[0],atol=0,rtol=0)
    assert not torch.equal(event_traces[0].queries[1],event_traces[1].queries[1])
    cache,state,trace,features=histories[0]
    rows={0:dict(cache=cache,previous=torch.zeros_like(state.codes),item=features,
                 ids=candidates,delta=delta)}
    means,_=gate_statistics(model,adapter,rows,[0])
    statistics=GateStatistics()
    statistics.add('v3',adapter,trace.queries,cache.seq_len)
    for index,layer in enumerate(adapter.layers):
        gk,gv=layer.query_gates(trace.queries[0])
        wanted=torch.stack((gk.mean((0,2)),gv.mean((0,2))))
        torch.testing.assert_close(means[index],wanted,atol=1e-12,rtol=1e-12)
        for side,wanted_side in zip(('key','value'),(gk,gv),strict=True):
            entry=statistics.units['v3',index,side]
            torch.testing.assert_close(entry['sum']/entry['count'],wanted_side.mean((0,2)),
                                       atol=1e-12,rtol=1e-12)
