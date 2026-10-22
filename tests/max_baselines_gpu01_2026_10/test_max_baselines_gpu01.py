"""Causality and ledger checks for the independent-edge baseline runner."""
from collections import Counter
import copy
import gc
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]

from max_baselines_gpu01_2026_10.baselines import evaluate,event_shapes
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.design_two.replay import NativeTrajectory
from hstu_kvcache.models import HSTU,HSTUConfig


def test_paired_uid_bootstrap_weighted_auc_and_ties():
    from sklearn.metrics import roc_auc_score
    from max_baselines_gpu01_2026_10.statistics import weighted_auc,paired_user_bootstrap
    labels=np.array([0,1,0,1,1,0],dtype=float)
    uids=np.array([0,0,1,1,2,2])
    scores=np.array([.1,.4,.4,.6,.9,.5])
    multiplicities=np.array([[1,1,1],[2,0,1],[0,3,2]],dtype=np.uint16)
    expected=[roc_auc_score(labels,scores,sample_weight=w[uids]) for w in multiplicities]
    np.testing.assert_allclose(weighted_auc(scores,labels,uids,multiplicities),expected,rtol=0,atol=1e-15)
    def rows(values):
        return [dict(uid=int(uid),query_timestamp=i,raw_item_id=i,label=int(label),score=float(value))
            for i,(uid,label,value) in enumerate(zip(uids,labels,values,strict=True))]
    groups=dict(reuse=rows(scores),rebuild=rows(labels*.8+.1),same_as_rebuild=rows(labels*.8+.1))
    intervals,metadata=paired_user_bootstrap(groups,[0,1,2,3],replicates=100,seed=17)
    assert intervals["reuse"]["auc_gain_over_reuse_ci95"]==[0.,0.]
    assert intervals["rebuild"]==intervals["same_as_rebuild"]
    assert metadata["eligible_users"]==4 and metadata["users_with_feedback"]==3


@torch.no_grad()
def test_score_batch_does_not_change_query_repair_selection(monkeypatch,tmp_path):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    torch.manual_seed(17)
    parent=HSTU(HSTUConfig(num_items=32,num_behaviors=2,hidden_size=192,num_heads=6,
        num_layers=2,max_seq_len=4,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
    current=copy.deepcopy(parent)
    current.item_emb.weight.add_(torch.randn_like(current.item_emb.weight)*.1)
    events=(np.arange(8)*10,np.arange(1,9),np.zeros(8,dtype=np.int64))
    histories=SimpleNamespace(rows={7:events})
    requests=[dict(uid=7,query_timestamp=40,item_idx=i,label=i%2,raw_item_id=i) for i in (18,21,25)]
    by_user={7:{40:requests}}
    policies=[dict(tag="reuse",method="reuse"),dict(tag="query",method="query",fraction=.5)]
    base=dict(edge=1,release_day=40/86400,end_day=80/86400,history_length=4,
        append_band_size=2,repair_query_batch=16)
    records=[]
    for tile in (1,16):
        cfg=dict(base,query_batch=tile)
        actual,_=evaluate(parent,current,histories,by_user,[7],policies,cfg,torch.device("cpu"),tmp_path)
        assert event_shapes(histories,by_user,[7],cfg)["first"]==Counter({(4,"query",3):1})
        records.append([r["score"] for r in actual])
    np.testing.assert_allclose(*records,atol=1e-6,rtol=1e-6)


def test_user_blocks_survive_interruption_and_validate_resume(monkeypatch,tmp_path):
    from max_baselines_gpu01_2026_10 import fast_baselines as fast
    calls=[]
    def fake_evaluate(*args,**kwargs):
        uids=args[4]
        calls.append(uids)
        if uids==[3,4] and calls.count([3,4])==1:raise RuntimeError("interruption")
        rows=[dict(uid=uid,query_timestamp=40,raw_item_id=18,item_idx=18,label=1,method="reuse",score=.1)
            for uid in uids if uid<=2]
        return rows,dict(stats=dict(feedback_events=len(rows)),seconds=.01)
    monkeypatch.setattr(fast,"evaluate_batched",fake_evaluate)
    cfg=dict(evaluation_checkpoint_users=2,stop_after_last_query=True)
    args=(None,None,None,None,[1,2,3,4],[],cfg,None,tmp_path)
    with pytest.raises(RuntimeError,match="interruption"):
        fast.evaluate_checkpointed(*args,identity=dict(protocol="frozen"))
    assert (tmp_path/"quality_shards/part_000000.json").exists()
    assert not (tmp_path/"quality_shards/part_000002.json").exists()
    scores,stats=fast.evaluate_checkpointed(*args,identity=dict(protocol="frozen"))
    assert calls==[[1,2],[3,4],[3,4]]
    assert len(scores)==2 and stats["stats"]["feedback_events"]==2
    assert [block["resumed"] for block in stats["blocks"]]==[True,False]
    with pytest.raises(AssertionError,match="settings/users differ"):
        fast.evaluate_checkpointed(*args,identity=dict(protocol="changed"))


def test_two_workers_commit_disjoint_global_blocks_and_resume_empty_users(monkeypatch,tmp_path):
    from max_baselines_gpu01_2026_10 import fast_baselines as fast
    import json
    calls=[]
    def fake(*args,**kwargs):
        uids=args[4]
        calls.extend(uids)
        rows=[dict(uid=u,query_timestamp=1,raw_item_id=1,item_idx=1,label=u%2,
                   method="reuse",score=.1) for u in uids if u!=5]
        return rows,dict(stats=dict(feedback_events=len(rows)),seconds=.01)
    monkeypatch.setattr(fast,"evaluate_batched",fake)
    args=(None,None,None,None,[1,2,3,4,5],[],dict(evaluation_checkpoint_users=2,
          stop_after_last_query=True),None,tmp_path)
    seen=[]
    for offsets in ({0,4},{2}):
        scores,record=fast.evaluate_checkpointed(*args,identity=dict(edge="max"),
                          assigned_offsets=offsets,retain_scores=False,progress=seen.append)
        assert not scores
    assert calls==[1,2,5,3,4]
    scores,record=fast.evaluate_checkpointed(*args,identity=dict(edge="max"))
    assert sorted(r["uid"] for r in scores)==[1,2,3,4]
    assert all(r["resumed"] for r in record["blocks"])
    assert len(calls)==5
    marker=json.loads((tmp_path/"quality_shards/part_000004.json").read_text())
    assert marker["users"]==1 and marker["scores"]==0
    assert [r["scores"] for r in seen]==[2,2,2]
    assert [(r["users"],r["total"]) for r in seen]==[(2,3),(3,3),(2,2)]


@torch.no_grad()
def test_common_first_history_consumer_and_independent_repair_states(monkeypatch,tmp_path):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    torch.manual_seed(17)
    config=HSTUConfig(num_items=32,num_behaviors=2,hidden_size=192,num_heads=6,
        num_layers=2,max_seq_len=4,input_dropout=0.,attn_dropout=0.)
    parent=HSTU(config).eval().requires_grad_(False)
    current=copy.deepcopy(parent)
    current.item_emb.weight.add_(torch.randn_like(current.item_emb.weight)*.02)
    events=(np.arange(8)*10,np.arange(1,9),np.zeros(8,dtype=np.int64))
    histories=SimpleNamespace(rows={7:events})
    by_user={7:{45:[dict(uid=7,query_timestamp=45,item_idx=21,label=0,raw_item_id=21)],
                60:[dict(uid=7,query_timestamp=60,item_idx=22,label=1,raw_item_id=22)],
                75:[dict(uid=7,query_timestamp=75,item_idx=23,label=1,raw_item_id=23)]}}
    cfg=dict(edge=1,release_day=40/86400,end_day=80/86400,history_length=4,
        append_band_size=2,query_batch=16)
    policies=[dict(tag="reuse",method="reuse"),dict(tag="rebuild",method="rebuild"),
        dict(tag="layer",method="layer",interval=(1,1)),
        *[dict(tag=m,method=m,fraction=.5) for m in ("tail","deviation","query")]]
    actual,observed=evaluate(parent,current,histories,by_user,[7],policies,cfg,torch.device("cpu"),tmp_path)
    reference=NativeTrajectory(parent,0,events,40,max_length=4,append_band_size=2)
    # First post-release event is at 40, before the first scored query at 45.
    # Prepare on its pre-event source, then append under the current model.
    reference.switch(current,1,40,rebuild=True)
    for timestamp,requests in sorted(by_user[7].items()):
        reference.advance(timestamp)
        expected=score(current,reference.cache,torch.tensor([[requests[0]["item_idx"]]]),
            torch.tensor([timestamp-reference.last_timestamp],dtype=torch.float32))[0].item()
        matching=[r for r in actual if r["method"]=="rebuild" and r["query_timestamp"]==timestamp]
        assert len(matching)==1
        torch.testing.assert_close(torch.tensor(matching[0]["score"]),torch.tensor(expected),atol=0,rtol=0)
    assert len(actual)==len(policies)*3
    assert all(v["candidate_scores"]==3 and v["prepare_rows"]==4 and v["appended_rows"]==4
               for v in observed.values())
    # CPU shape accounting must reproduce actual query-separated append bands.
    shapes=event_shapes(histories,by_user,[7],cfg)
    assert shapes["first"]==Counter({(4,"behavior",1):1})
    reference.advance(80)
    counted=Counter((before,width,True) for action,version,before,width in reference.ledger.elements()
                    if action=="append")
    assert shapes["appends"]==counted


@pytest.mark.skipif(not torch.cuda.is_available(),reason="requires allocated GPU")
@torch.no_grad()
def test_graph_retains_packed_query_weights_across_user_blocks(monkeypatch):
    from max_baselines_gpu01_2026_10.fast_baselines import NativeGraphs,PackedQuery
    from hstu_kvcache.models import HSTUKVCache
    from hstu_kvcache.read_correction import score_corrected
    from hstu_kvcache.read_correction_v5.query_only.core import QueryFeatureCorrection
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","triton")
    torch.manual_seed(17)
    cfg=HSTUConfig(num_items=32,num_behaviors=2,hidden_size=320,num_heads=10,
        num_layers=2,max_seq_len=1024,input_dropout=0.,attn_dropout=0.)
    model=HSTU(cfg).cuda().eval().requires_grad_(False)
    items=torch.randint(1,32,(1,1024),device="cuda")
    source=model.compute_kv(items,torch.zeros_like(items),torch.ones_like(items,dtype=torch.float32))
    policies=[]
    for i in range(3):
        modules=[QueryFeatureCorrection(10,32,"cross_phi").cuda().eval().requires_grad_(False) for _ in range(2)]
        for module in modules:module.weight.normal_(std=.001)
        policies.append(dict(modules=modules,tag=str(i)))
    cache=HSTUKVCache(source.k.expand(-1,3,-1,-1),source.v.expand(-1,3,-1,-1),1024)
    graph=NativeGraphs()
    for item in (18,23):
        packed=PackedQuery(policies)
        ids=torch.full((3,1),item,device="cuda")
        delta=torch.ones(3,device="cuda")
        actual=graph.run("score",model,cache,ids,delta,reader=packed.reader(1024),reader_key="packed_query").clone()
        expected=torch.cat([score_corrected(model,source,ids[i:i+1],delta[i:i+1],p["modules"],
            torch.tensor([1024],device="cuda"))[0] for i,p in enumerate(policies)],0)
        torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-5)
        del packed
        gc.collect()
        # A later block must not replay graph pointers into freed packed weights.
        scratch=torch.randn(3,640,320,device="cuda")
        del scratch
    assert graph.counts["captures"]==1
    assert graph.counts["graph_score"]==2
