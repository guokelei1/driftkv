"""Small checks for irrecoverable metric/checkpoint mistakes."""
import tempfile
from pathlib import Path
import sys
import numpy as np
import pytest
from experiments.figure9max.common import Rolling, configuration, policies

def test_exact_matrix():
    rows=policies(configuration())
    assert len(rows)==29
    intervals=[tuple(p["interval"]) for p in rows if p["method"]=="layer"]
    assert intervals==[(0, 1), (5, 6), (10, 11), (0, 2), (5, 7), (10, 12), (0, 4), (8, 12), (0, 5), (0, 9), (0, 12)]
    assert [p["tokens"] for p in rows if p["method"]=="tail"]==[64,128,256,512,768]

def test_rollover_preserves_exact_scalar_inputs_and_cursor():
    with tempfile.TemporaryDirectory() as directory:
        s=Rolling(directory,"bound-input",1800)
        s.append(7,[0,1],[[.1,.2],[.8,.9]])
        assert not s.save(1,2)
        s.last-=1801
        assert s.save(1,2)
        s.append(9,[0,1],[[.3,.1],[.7,.8]])
        assert s.save(2,2,True)
        restored=Rolling(directory,"bound-input")
        assert restored.cursor==2
        y,u,x=restored.arrays(2)
        assert y.tolist()==[0,1,0,1] and u.tolist()==[7,7,9,9]
        np.testing.assert_allclose(x,[[.1,.2],[.8,.9],[.3,.1],[.7,.8]])
        assert len(list(Path(directory).glob("*.npz")))==2
        with pytest.raises(RuntimeError,match="identity changed"):
            Rolling(directory,"different-model-or-code")

def test_weighted_auc_ties_match_exact_expanded_population():
    from sklearn.metrics import roc_auc_score
    from hstu_kvcache.evaluation.statistics import weighted_auc
    scores=np.array([.2,.2,.8,.7,.7])
    labels=np.array([0,1,1,0,1])
    clusters=np.array([0,0,1,2,2])
    weights=np.array([[1,1,1],[2,1,0],[0,2,3]])
    actual=weighted_auc(scores,labels,clusters,weights)
    for i,w in enumerate(weights):
        repeated=np.repeat(np.arange(len(scores)),w[clusters])
        assert actual[i]==pytest.approx(roc_auc_score(labels[repeated],scores[repeated]),abs=1e-14)

def test_complete_cost_includes_unscored_users_and_suffix():
    from types import SimpleNamespace
    from experiments.figure9max.common import ledger
    from hstu_kvcache.evaluation.cost import CostModel
    at=86400
    history=SimpleNamespace(rows={u:(np.array([at-3,at-1,at+5,at+20]),np.ones(4,np.int64),np.ones(4,np.int64)) for u in (1,2)})
    inputs=dict(histories=history,quality=[1,2],by_user={1:{at+10:[{}]}})
    cfg=dict(release_day=1,end_day=2,history_length=1024,append_band_size=128,query_batch=16)
    rows=[dict(tag="reuse",method="reuse"),dict(tag="rebuild",method="rebuild"),dict(tag="middle",method="layer",interval=[3,3])]
    result=ledger(inputs,cfg,rows,{})
    assert result["once_rebuild_flops"]==2*CostModel(320,16,10,"torch").full_cache(2)
    assert result["old_boundary_bytes"]["middle"]==4*320*4
    assert result["native_append_flops"]>0

def test_final_statistics_use_pooled_auc_and_paired_uid_draws():
    from experiments.figure9max.run import summarize_arrays
    from sklearn.metrics import roc_auc_score
    y=np.tile([0,1],12).astype(np.uint8)
    u=np.repeat(np.arange(12),2)
    a=np.tile([.3,.7],12)
    a[:6]=np.tile([.8,.2],3)
    s=np.stack((a,y.astype(float),.5*a+.5*y),1)
    rows=[dict(tag=k,method=k) for k in ("reuse","rebuild","test")]
    results,confidence=summarize_arrays(y,u,s,rows,list(range(12)),40,17)
    assert results[0]["auc"]==roc_auc_score(y,a)
    assert results[1]["recovery_percent"]==100
    assert confidence["replicates"]==40
    assert results[2]["auc_ci95"] is not None

def test_formal_launch_requires_explicit_launch_instruction_flag():
    from types import SimpleNamespace
    from experiments.figure9max.run import launch
    with pytest.raises(RuntimeError,match="Awaiting user instruction"):
        launch(SimpleNamespace(mode="formal",authorize=False))
