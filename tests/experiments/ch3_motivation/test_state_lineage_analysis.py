"""Independent pair-count and loss checks for the bounded CPU lineage analysis."""
from pathlib import Path
import sys
import numpy as np

from experiments.ch3_motivation.code.analyze_state_lineage import summarize


def brute_auc(scores,labels,multiplicity):
    positive=np.flatnonzero(labels==1);negative=np.flatnonzero(labels==0)
    denominator=sum(multiplicity[i]*multiplicity[j] for i in positive for j in negative)
    if not denominator:return np.nan
    return sum(multiplicity[i]*multiplicity[j]*((scores[i]>scores[j])+.5*(scores[i]==scores[j]))
        for i in positive for j in negative)/denominator


def test_auc_loss_paired_uid_and_unclipped_recovery():
    residents=[11,22,33,44]
    uids=np.asarray([11]*6+[22]*2+[33]*2+[44]*2)
    labels=np.tile([0,1],6)
    scores={'reuse':np.zeros(12),'direct':labels*4.-2.,
        'chain':np.asarray([2,-2,-2,2,2,-2,-2,2,2,-2,-2,2],dtype=float),
        'rebuild':np.asarray([-1,2,-1,2,-1,2,-1,2,-1,2,3,-2],dtype=float)}
    mse={11:10.,22:1.,33:2.,44:3.}
    groups={method:[dict(uid=int(u),label=int(y),score=float(s),
        kv_mse=0. if method=='rebuild' else mse[int(u)],
        k_mse=0. if method=='rebuild' else mse[int(u)],
        v_mse=0. if method=='rebuild' else mse[int(u)]) for u,y,s in zip(uids,labels,z,strict=True)]
        for method,z in scores.items()}
    result=summarize(groups,residents)
    for method,z in scores.items():
        point=brute_auc(z,labels,np.ones(len(uids)))
        np.testing.assert_allclose(result['methods'][method]['AUC']['value'],point,atol=1e-15)
        expected_loss=np.mean(np.logaddexp(0,z)-labels*z)
        np.testing.assert_allclose(result['methods'][method]['log_loss']['value'],expected_loss,atol=1e-15)
    rng=np.random.default_rng(17);differences=[]
    for _ in range(500):
        weights=np.bincount(rng.integers(4,size=4),minlength=4)
        event_weights=weights[np.searchsorted(residents,uids)]
        differences.append(brute_auc(scores['direct'],labels,event_weights)-brute_auc(scores['chain'],labels,event_weights))
    np.testing.assert_allclose(result['paired_direct_minus_chain']['AUC']['ci95'],np.percentile(differences,[2.5,97.5]),atol=1e-15)
    assert result['methods']['direct']['recovery_percent']['value']>100
    assert result['methods']['direct']['recovery_percent']['ci95'] is None
    assert result['methods']['direct']['feedback_uid_mean_snapshot_MSE']['kv_mse']==4.
    assert result['feedback_users']==4 and not result['mechanism_use_permitted']
