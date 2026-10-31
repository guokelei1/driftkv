import sys
from pathlib import Path

import numpy as np

from experiments.ch3_motivation.code.analyze_candidate_residual_probe import (
    centered, derangements, features, fit_ridge, ratio_stat, bootstrap_weights, pair_geometry,
)


def test_uid_derangement_preserves_feature_distribution_and_is_deterministic():
    uids = np.array([10, 11, 12], dtype=np.uint64)
    p = derangements(uids)
    np.testing.assert_array_equal(p, derangements(uids))
    assert np.all(p != np.arange(16))
    assert not np.array_equal(p[0], p[1])
    x = np.random.default_rng(1).normal(size=(3, 16, 16))
    real, scale = features(x)
    shuffled, _ = features(x, scale, permutation=p)
    np.testing.assert_allclose(real.mean(1), 0, atol=1e-15)
    np.testing.assert_allclose((real**2).sum((0,1)), (shuffled**2).sum((0,1)))


def test_closed_form_true_and_permuted_conditions_receive_same_capacity():
    rng = np.random.default_rng(2)
    x, vx = rng.normal(size=(40,16,16)), rng.normal(size=(20,16,16))
    x, scale = features(x); vx, _ = features(vx,scale)
    truth = rng.normal(size=(16,5))
    y, vy = x @ truth, vx @ truth
    true, a = fit_ridge(x,y,vx,vy)
    px = x[np.arange(40)[:,None],derangements(range(40))]
    pvx = vx[np.arange(20)[:,None],derangements(range(100,120))]
    wrong,b = fit_ridge(px,y,pvx,vy)
    assert a['coefficients']==b['coefficients']==16*5
    assert np.mean((vx @ true-vy)**2) < .01*np.mean((pvx @ wrong-vy)**2)
    assert len(a['validation_grid'])==len(b['validation_grid'])==4


def test_centered_metric_separates_tiny_candidate_component_from_total():
    w = bootstrap_weights(8,replicates=20,seed=17)
    mean = np.full(8,100.)
    gain = np.full(8,.01)
    result=ratio_stat(gain,mean,w)
    np.testing.assert_allclose(result['mean'],.01)
    np.testing.assert_allclose(result['ci95'],[.01,.01])
    y=np.arange(32).reshape(2,16)+np.array([[100.],[200.]])
    np.testing.assert_allclose(centered(y).mean(1),0)


def test_pair_geometry_separates_student_ties_from_strict_flips():
    teacher=np.arange(16,dtype=float)[None]
    tied=np.zeros((1,16))
    g=pair_geometry(teacher,tied)
    assert g['teacher_nonzero_pair_fraction'][0]==1
    assert g['student_tie_pair_fraction'][0]==1
    assert g['strict_opposite_pair_fraction'][0]==0
    assert g['teacher_valid_disagreement_fraction_all_pairs'][0]==1
    same=pair_geometry(teacher,teacher)
    assert same['teacher_valid_disagreement_fraction_all_pairs'][0]==0
    reverse=pair_geometry(teacher,-teacher)
    assert reverse['strict_opposite_pair_fraction'][0]==1
    none=pair_geometry(tied,tied)
    assert none['teacher_nonzero_pair_fraction'][0]==0
    result=ratio_stat(np.zeros(1),np.zeros(1),np.ones((3,1)),percent=False)
    assert result['mean'] is None and result['invalid_bootstrap_draws']==3
