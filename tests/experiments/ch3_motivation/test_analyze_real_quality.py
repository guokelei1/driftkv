from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.ch3_motivation.code.analyze_real_quality import (
    binary_point_draws, event_gain_harm, fit_affine, summarize, uid_weights,
)
from experiments.ch3_motivation.code import analyze_real_quality as analysis


def test_auc_ties_and_uid_multiplicity_match_explicit_pair_probability():
    y=np.array([0,1,0,1]);z=np.array([1.,1.,2.,3.]);idx=np.array([0,0,1,1])
    weights=np.array([[1,1],[2,1]],dtype=np.uint16)
    point,draws=binary_point_draws(y,z,idx,weights)
    assert point['AUC']['value']==.625
    for row,actual in zip(weights,draws['AUC']):
        w=row[idx];numerator=0.
        for i in np.flatnonzero(y==1):
            for j in np.flatnonzero(y==0):
                numerator+=w[i]*w[j]*(float(z[i]>z[j])+.5*float(z[i]==z[j]))
        expected=numerator/(w[y==1].sum()*w[y==0].sum())
        assert np.isclose(actual,expected)


def test_logloss_is_stable_and_majority_accuracy_is_uninformative():
    y=np.array([0,1]);idx=np.array([0,1]);w=np.array([[1,1]],dtype=np.uint16)
    correct,_=binary_point_draws(y,np.array([-1000.,1000.]),idx,w)
    wrong,_=binary_point_draws(y,np.array([1000.,-1000.]),idx,w)
    assert correct['log_loss']['value']==0.
    assert wrong['log_loss']['value']==1000.
    majority,_=binary_point_draws(np.array([1,1,1,0]),np.ones(4),np.array([0,0,1,1]),w)
    assert majority['accuracy_at_05']['value']==.75
    assert majority['negative_recall']['value']==0.
    assert majority['balanced_accuracy_at_05']['value']==.5


def test_affine_fitting_has_positive_slope_and_preserves_auc():
    x=np.array([0.,2.,1.,3.]);y=2*x+3
    positive,bias=fit_affine(x,y)
    assert positive=={'slope':2.,'bias':3.}
    assert bias['slope']==1.
    labels=np.array([0,1,0,1]);idx=np.arange(4);w=np.ones((1,4),dtype=np.uint16)
    base,_=binary_point_draws(labels,x,idx,w)
    changed,_=binary_point_draws(labels,positive['slope']*x+positive['bias'],idx,w)
    assert base['AUC']['value']==changed['AUC']['value']
    assert base['log_loss']['value']!=changed['log_loss']['value']


def test_teacher_score_closeness_does_not_imply_auc_monotonicity():
    labels=np.array([0,1,0,1]);f=np.array([0.,2.,1.,3.]);t=np.array([2.,3.,0.,1.])
    idx=np.arange(4);w=np.ones((1,4),dtype=np.uint16)
    initial,_=binary_point_draws(labels,f,idx,w)
    teacher,_=binary_point_draws(labels,t,idx,w)
    assert np.mean((t-t)**2)<np.mean((f-t)**2)
    assert teacher['AUC']['value']<initial['AUC']['value']


def test_uid_resampling_keeps_all_events_in_the_cluster():
    index,w=uid_weights([11,12,13],np.array([11,11,13],dtype=np.uint64),replicates=8,seed=1)
    assert index.tolist()==[0,0,2]
    assert np.array_equal(w[:,index[0]],w[:,index[1]])
    assert np.all(w.sum(1)==3)


def test_single_label_auc_is_explicitly_undefined():
    values=[dict(uid=11,query_timestamp=1,item_idx=4,label=1,score=1.),
            dict(uid=12,query_timestamp=1,item_idx=4,label=1,score=2.)]
    groups={m:[dict(r,method=m) for r in values] for m in ['reuse','f1','rebuild']}
    result,_=summarize(groups,[11,12])
    assert result['methods']['f1']['AUC']['value'] is None
    assert result['methods']['f1']['AUC_minus_reuse']['value'] is None
    assert result['methods']['f1']['AUC']['invalid_bootstrap_draws']==500
    assert not result['mechanism_use_permitted']


def test_event_harm_retains_opposite_individual_effects():
    values=[dict(uid=11,query_timestamp=1,item_idx=4,label=1,score=0.),
            dict(uid=11,query_timestamp=2,item_idx=4,label=0,score=0.),
            dict(uid=12,query_timestamp=1,item_idx=4,label=1,score=0.)]
    groups={'reuse':values,'f1':[dict(r,score=s) for r,s in zip(values,[1.,1.,0.])]}
    result=event_gain_harm(groups,[11,12])['reuse']['f1']
    assert result['improved_fraction']['value']==1/3
    assert result['harmed_fraction']['value']==1/3
    assert result['unchanged_fraction']['value']==1/3
    assert result['benefit_log_loss_mass']['value']>0
    assert result['harm_log_loss_mass']['value']>result['benefit_log_loss_mass']['value']
    assert np.isclose(result['mean_log_loss_gain']['value'],
        (result['benefit_log_loss_mass']['value']-result['harm_log_loss_mass']['value'])/3)


def test_full_reader_checks_binding_and_keeps_sparse_groups_diagnostic(tmp_path,monkeypatch):
    population=dict(users=list(range(1,4001)),edges={'2':{'user_metadata':
        {str(u):dict(pre_gap_seconds=100,prerelease14activity=1) for u in range(1,4001)}}})
    population_path=tmp_path/'population.json';population_path.write_text(json.dumps(population))
    folder=tmp_path/'run';folder.mkdir()
    config=dict(edge=2,diagnostic=False,users=population['users'],feedback_users=[1,2],
        population_sha256=analysis.sha(population_path),checkpoint_sha256='fixture',
        settings=dict(release_day=1,history_length=1024))
    (folder/'configuration.json').write_text(json.dumps(config))
    rows=[]
    for method,shift in [('reuse',0),('f1',.1),('rebuild',.2)]:
        for uid,label,day in [(1,1,1),(1,1,8),(2,0,1),(2,0,8)]:
            rows.append(dict(uid=uid,label=label,query_timestamp=(1+day)*86400,
                item_idx=4,method=method,score=.5+shift,appended_events=0 if day==1 else 2048))
    block=folder/'part.parquet';pq.write_table(pa.Table.from_pylist(rows),block)
    frozen=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    manifest=folder/'part.json';manifest.write_text(json.dumps(dict(configuration_sha256=frozen,
        score_sha256=analysis.sha(block),score_rows=len(rows))))
    summary=dict(status='complete',quality_use_permitted=True,score_rows=len(rows),cost_reference={},artifacts=[dict(
        path=block.name,sha256=analysis.sha(block),manifest=manifest.name,manifest_sha256=analysis.sha(manifest))])
    (folder/'summary.json').write_text(json.dumps(summary))
    controls={name:dict(slope=1.,bias=0.) for name in ['affine_teacher_fit','bias_teacher_fit']}
    monkeypatch.setattr(analysis,'affine_controls',lambda edge:(controls,dict(checkpoint_sha256='fixture')))
    result=analysis.full_requests(folder,population_path,tmp_path/'analysis')
    assert result['groups']['all']['feedback_users']==2
    assert not result['groups']['all']['mechanism_use_permitted']
    assert result['groups']['days_3_7']['events']==0
    assert result['paired_time']['feedback_users']==2
    assert result['paired_time']['methods']=={}
    assert result['paired_fifo']['feedback_users']==2
    assert result['paired_fifo']['methods']=={}
    assert (tmp_path/'analysis/event_gain_harm.csv').is_file()
    assert json.loads((tmp_path/'analysis/summary.json').read_text())['groups']['all']['events']==4
    config['settings']['history_length']=128
    (folder/'configuration.json').write_text(json.dumps(config))
    import pytest
    with pytest.raises(ValueError,match='different frozen configuration'):
        analysis.full_requests(folder,population_path,tmp_path/'invalid')
