from pathlib import Path
import sys
import numpy as np
import pytest

from experiments.ch3_motivation.code.analyze_candidate_robustness import (
    align, auc_summary, class_differences, uid_weights,
)


def test_higher_recovery_can_leave_larger_absolute_remaining_auc_gap():
    def point(native,gain):
        return dict(teacher_minus_reuse_AUC={'value':native},methods={
            'map':dict(teacher_minus_map_AUC={'value':native-gain},map_minus_reuse_AUC={'value':gain})})
    a,b=point(.01,.004),point(.04,.03)
    ad={m:np.array([x,x]) for m,x in [('reuse',.6),('map',.604),('rebuild',.61)]}
    bd={m:np.array([x,x]) for m,x in [('reuse',.6),('map',.63),('rebuild',.64)]}
    result=class_differences(a,b,ad,bd)
    assert .03/.04>.004/.01
    assert np.isclose(result['methods']['map']['remaining_gap_difference']['value'],-.004)
    assert np.isclose(result['native_gap_difference']['value'],-.03)
    assert np.isclose(result['methods']['map']['map_gain_difference']['value'],-.026)


def test_dynamic_static_method_names_work_without_f1_or_oracles():
    methods=['reuse','rebuild','linear_fit256_userbalanced']
    rows=[dict(uid=u,query_timestamp=t,item_idx=4,label=y,method=m,score=float(y))
        for m in methods for u in [11,12] for t,y in [(1,0),(2,1)]]
    groups=align(rows,methods)
    index,weights=uid_weights([11,12],np.array([r['uid'] for r in groups['reuse']],dtype=np.uint64))
    result,_=auc_summary(groups,np.ones(4,dtype=bool),index,weights)
    assert set(result['methods'])==set(methods)
    for x in result['methods'].values():
        assert x['teacher_minus_map_AUC']['value']==0.
        assert x['recovery_percent']['value'] is None
    with pytest.raises(ValueError,match='static maps only'):
        align(rows,['reuse','rebuild','kv_oracle05'])


def test_paired_remaining_gap_identity_holds_with_different_group_variances():
    # Each effect uses one common array of UID-resampled AUCs. The identity is
    # algebraic even when class-specific native gaps have different variances.
    def point(native,gain):
        return dict(teacher_minus_reuse_AUC={'value':native},methods={
            'map':dict(teacher_minus_map_AUC={'value':native-gain},map_minus_reuse_AUC={'value':gain})})
    ad=dict(reuse=np.array([.5,.55]),map=np.array([.6,.61]),rebuild=np.array([.7,.73]))
    bd=dict(reuse=np.array([.6,.52]),map=np.array([.65,.63]),rebuild=np.array([.71,.74]))
    result=class_differences(point(.2,.1),point(.11,.05),ad,bd)
    assert np.allclose(result['methods']['map']['remaining_gap_difference']['ci95'],[.01075,.03925])
