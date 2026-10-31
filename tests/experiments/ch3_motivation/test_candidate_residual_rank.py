import importlib.util
from pathlib import Path
import sys
import numpy as np

scripts=Path(__file__).resolve().parents[3]/'experiments/ch3_motivation/code'
spec=importlib.util.spec_from_file_location('candidate_rank',scripts/'analyze_candidate_residual_rank.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_known_rank2_centered_candidate_matrix():
    rng=np.random.default_rng(17)
    left=rng.normal(size=(16,2));left-=left.mean(0)
    left=np.linalg.qr(left)[0]
    right=np.linalg.qr(rng.normal(size=(32,2)))[0]
    matrix=(left*np.array([5.,3.]))@right.T
    total,retained=m.svd_energies(matrix)
    np.testing.assert_allclose(total,34.,rtol=1e-12)
    np.testing.assert_allclose(retained,[25.,34.,34.,34.],rtol=1e-12)
    # A candidate-constant mean changes the raw residual, not its centered rank.
    shifted=matrix+np.arange(32)[None,:]
    _,again=m.svd_energies(shifted)
    np.testing.assert_allclose(again,retained,rtol=1e-12)


def test_exact_zero_residual_retention_is_undefined():
    total,retained=m.svd_energies(np.zeros((16,32)))
    assert total==0 and np.all(retained==0)
    assert np.isnan(m.divide(retained,total)).all()
