"""Flat/tie handling and a hand-computable shared UID bootstrap loss curve."""
from pathlib import Path
import sys
import numpy as np
from experiments.ch3_motivation.code.analyze_readonly_candidate_strength import (
    GRID, preference, analyze_scope,
)
from experiments.ch3_motivation.code.analyze_real_quality import uid_weights


def test_flat_and_cross_group_tie_do_not_manufacture_preference_one():
    flat=preference(np.ones(9))
    assert flat['flat'] and flat['representative'] is None and flat['group']=='flat'
    losses=np.ones(9);losses[0]=0.;losses[4]=.5e-6
    tied=preference(losses)
    assert not tied['flat'] and tied['representative']==1.
    assert tied['group']=='ambiguous' and tied['minimum_set_span']==[0.,1.]
    invalid=preference(np.full(9,np.nan))
    assert not invalid['valid'] and invalid['group']=='invalid'


def test_bootstrap_matches_weighted_quadratic_minimum():
    # EachUID contributes two events. Half favor0, half favor2; the full point favors1.
    users=list(range(10,20));uids=np.repeat(np.asarray(users,dtype=np.uint64),2)
    labels=np.tile([0,1],10);target=np.repeat(np.asarray([0.]*5+[2.]*5),2)
    loss=.2+.01*(GRID[None,:]-target[:,None])**2
    base=[dict(uid=int(u),query_timestamp=100+i,item_idx=55) for i,u in enumerate(uids)]
    index,weights=uid_weights(users,uids,replicates=200,seed=17)
    records,curves,summary,draws=analyze_scope(np.ones(20,dtype=bool),'test',base,loss,uids,labels,np.ones(20),index,weights,{55})
    assert len(records)==1 and records[0]['preference']['representative']==1.
    assert summary['items']==1 and len(curves)==9
    expected=[]
    for multiplicities in weights:
        center=2*sum(multiplicities[5:])/sum(multiplicities)
        expected.append(float(GRID[np.argmin((GRID-center)**2)]))
    np.testing.assert_array_equal(draws[0],expected)
    fraction=np.mean(np.abs(np.asarray(expected)-1.)<=.25)
    assert records[0]['bootstrap']['point_within_one_grid_step_probability']==fraction
    assert records[0]['bootstrap']['valid_draws']==200
    assert records[0]['bootstrap']['stable_descriptive']==(fraction>=.70)
