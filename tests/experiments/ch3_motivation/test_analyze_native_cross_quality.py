from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.ch3_motivation.code.analyze_native_cross_quality import METHODS, align, contrasts, summarize
from experiments.ch3_motivation.code import analyze_native_cross_quality as analysis


def observed_rows():
    return [dict(uid=uid,query_timestamp=t,item_idx=4,label=label,score=score,method=method)
        for method in METHODS for uid,t,label,score in [(11,1,0,-1.),(11,2,1,1.),(12,1,0,0.),(12,2,1,2.)]]


def test_factorial_gain_sign_is_consistent_for_auc_and_loss():
    auc=[.5,.6,.55,.8];loss=[.7,.6,.65,.4]
    points={m:dict(AUC={'value':a},log_loss={'value':l}) for m,a,l in zip(METHODS,auc,loss)}
    draws={m:dict(AUC=np.array([a,a+.1]),log_loss=np.array([l,l+.1])) for m,a,l in zip(METHODS,auc,loss)}
    result=contrasts(points,draws)
    for metric in ['AUC','log_loss']:
        assert np.isclose(result['key_effect_with_old_value'][metric]['value'],.1)
        assert np.isclose(result['key_effect_with_teacher_value'][metric]['value'],.25)
        assert np.isclose(result['value_effect_with_old_key'][metric]['value'],.05)
        assert np.isclose(result['joint_teacher_effect'][metric]['value'],.3)
        assert np.isclose(result['joint_minus_sum_of_single_effects'][metric]['value'],.15)


def test_identical_counterfactuals_have_zero_paired_effects():
    result=summarize(align(observed_rows()),[11,12])
    for effects in result['effects'].values():
        for value in effects.values():
            assert value['value']==0
            assert value['ci95']==[0.,0.]
    assert not result['mechanism_use_permitted']


def test_alignment_rejects_same_key_with_conflicting_labels_and_method_mismatch():
    rows=observed_rows();rows.append(dict(rows[0],label=1))
    with pytest.raises(ValueError,match='duplicate or conflicting'):
        align(rows)
    rows=observed_rows();rows[-1]['label']=0
    with pytest.raises(ValueError,match='keys or labels'):
        align(rows)
    with pytest.raises(ValueError,match='missing native cross method'):
        align([r for r in observed_rows() if r['method']!=METHODS[-1]])


def test_completed_reader_preserves_support_limits_and_rejects_configuration_changes(tmp_path):
    population=dict(users=list(range(1,4001)),edges={'2':{'user_metadata':
        {str(u):dict(pre_gap_seconds=10,prerelease14activity=10) for u in range(1,4001)}}})
    population_path=tmp_path/'population.json';population_path.write_text(json.dumps(population))
    folder=tmp_path/'run';folder.mkdir()
    config=dict(format='native_cross_quality_v1',method_names=list(METHODS),sources={},
        edge=2,diagnostic=False,users=population['users'],feedback_users=list(range(1,513)),
        population_sha256=analysis.sha(population_path),settings=dict(release_day=1,history_length=1024))
    (folder/'configuration.json').write_text(json.dumps(config))
    rows=[dict(uid=uid,query_timestamp=(1+day)*86400,item_idx=4,label=label,method=method,score=float(label))
        for method in METHODS for uid in range(1,513) for day,label in [(1,0),(8,1)]]
    block=folder/'part.parquet';pq.write_table(pa.Table.from_pylist(rows),block)
    frozen=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    manifest=folder/'part.json';manifest.write_text(json.dumps(dict(configuration_sha256=frozen,
        score_sha256=analysis.sha(block),score_rows=len(rows))))
    summary=dict(status='complete',quality_use_permitted=True,feedback_users=512,score_rows=len(rows),artifacts=[dict(
        path=block.name,sha256=analysis.sha(block),manifest=manifest.name,manifest_sha256=analysis.sha(manifest))])
    (folder/'summary.json').write_text(json.dumps(summary))
    result=analysis.run(folder,population_path,tmp_path/'analysis')
    assert result['groups']['all']['mechanism_use_permitted']
    assert result['groups']['all']['feedback_users']==512
    assert not result['groups']['days_0_3']['mechanism_use_permitted']  # one label only
    assert result['groups']['days_3_7']['events']==0
    assert (tmp_path/'analysis/factorial_effects.csv').is_file()
    assert json.loads((tmp_path/'analysis/summary.json').read_text())['groups']['all']['events']==1024
    config['settings']['history_length']=128
    (folder/'configuration.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='different frozen configuration'):
        analysis.run(folder,population_path,tmp_path/'invalid')
