from pathlib import Path
import sys

import numpy as np

from experiments.ch3_motivation.code.analyze_candidate_class_auc import (
    METHODS, align, effect_difference, recovery, summarize, uid_weights,
)


def test_recovery_retains_unclipped_point_and_refuses_nonpositive_denominator_draws():
    x=recovery(.02,.01,np.array([.01,.02,.03]),np.array([.01,-.001,.02]))
    assert x['value']==200.
    assert x['ci95'] is None and not x['ratio_ci_permitted']
    assert x['nonpositive_or_invalid_denominator_draws']==1
    negative=recovery(.01,-.02,np.ones(3)*.01,np.ones(3)*-.02)
    assert negative['value']==-50.
    assert negative['ci95'] is None
    assert recovery(.01,0.,np.ones(3)*.01,np.ones(3)*.02)['value'] is None


def test_positive_denominator_allows_exact_constant_recovery_interval():
    x=recovery(.01,.02,np.array([.005,.01,.015]),np.array([.01,.02,.03]))
    assert x['value']==50.
    assert x['ci95']==[50.,50.]
    assert x['ratio_ci_permitted']


def test_class_effect_difference_reuses_the_entire_uid_cluster_draw():
    rows=[]
    for method in METHODS:
        for uid in [11,12]:
            for category in [0,1]:
                for label in [0,1]:
                    score=float(label) if uid==11 else 2.-3.*label
                    if method!='reuse':score=float(label)
                    rows.append(dict(uid=uid,query_timestamp=1+category*2+label,item_idx=4,
                        label=label,method=method,score=score))
    groups=align(rows);base=groups['reuse']
    index,weights=uid_weights([11,12],np.array([r['uid'] for r in base],dtype=np.uint64))
    first=np.array([r['query_timestamp']<=2 for r in base])
    a,ad=summarize(groups,first,index,weights);b,bd=summarize(groups,~first,index,weights)
    differences=effect_difference(a,b,ad,bd)
    assert np.ptp(ad['reuse'])>0  # UID composition changes pooled AUC.
    for x in differences.values():
        assert x['value']==0.
        assert x['ci95']==[0.,0.]
