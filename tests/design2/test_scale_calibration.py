import numpy as np
import pandas as pd

from design2.scale_calibration import assign_groups, fit_affine, make_groups, uid_folds


def test_nonnegative_fit_and_matched_intercept():
    u = np.array([1., 2., 4., 7.]) * 1e5
    w = np.array([1., 2., 1., 3.])
    y = .04 + 2e-6*u
    f = fit_affine(u, y, w)
    np.testing.assert_allclose([f["b"], f["a"]], [.04, 2e-6], rtol=1e-10)
    np.testing.assert_allclose(f["background"], np.average(y, weights=w))
    # Negative slope is disallowed; the optimum lies on the intercept-only boundary.
    f = fit_affine(u, 2.-1e-6*u, w)
    assert f["a"] == 0
    np.testing.assert_allclose(f["b"], f["background"])
    # A negative unconstrained intercept is disallowed too.
    f = fit_affine(np.array([1., 2., 3.]), np.array([0., 1., 2.]), np.ones(3))
    assert f["b"] == 0 and f["a"] > 0


def test_uid_folds_sparse_merge_and_no_label_access():
    ids = np.arange(20)
    folds = uid_folds(ids, 17, 5)
    f = pd.DataFrame([dict(uid=int(u), target=1, count=n, fold=folds[u])
                      for u in ids for n in ([4, 64] if u < 2 else [64])])
    cfg = dict(base_length_upper_bounds=[32, 1024], targets=[1], folds=5,
               minimum_train_uids_per_group=8)
    # Only metadata is supplied; two short-history UIDs cannot form a supported group.
    g = make_groups(f, cfg)
    assert len(g) == 1 and (g[0]["lower"], g[0]["upper"]) == (1, 1024)
    assert min(g[0]["training_fold_uids"]) == 16
    assert assign_groups(f, g).group.nunique() == 1
    assert f.groupby("uid").fold.nunique().max() == 1
