from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.feedback_candidate_pool import (
    fitting_feedback_catalog, pooled_candidates,
)
from experiments.ch3_motivation.code.prepare_candidates import candidates_for


def test_catalog_does_not_include_validation_or_evaluation_histories():
    recent={"1":[1,2,8],"2":[3,4,8],"3":[999],"4":[1000]}
    assert fitting_feedback_catalog(recent,[1,2],1000)==[1,2,3,4,8]


def test_empty_own_feedback_still_gets_reproducible_observed_quota():
    pool=list(range(1,41))
    values,counts=pooled_candidates(71,[],pool,100,count=16,recent_fraction=.5)
    assert counts==dict(own_recent=0,other_fit_recent=8,uniform=8,
        requested_feedback_quota=8,actual_feedback_quota=8)
    assert len(values)==len(set(values))==16
    assert set(values[:8])<=set(pool)
    assert (values,counts)==pooled_candidates(71,[],pool,100,count=16,recent_fraction=.5)


def test_full_own_quota_and_uniform_control_match_existing_rule():
    for fraction in (0.,.5,1.):
        own=list(range(1,17))
        actual,counts=pooled_candidates(89,own,[],100,recent_fraction=fraction)
        assert actual==candidates_for(89,own,100,recent_fraction=fraction)
        assert counts["other_fit_recent"]==0
    values,counts=pooled_candidates(91,[],[1],4,count=4,recent_fraction=1.)
    assert set(values)=={1,2,3,4}  # final known ID is included
    assert counts["actual_feedback_quota"]==1 and counts["uniform"]==3
