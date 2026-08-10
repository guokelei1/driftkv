import numpy as np

from hstu_kvcache.streaming.kuairand_path_attribution import _strata


def test_position_strata_are_disjoint_and_complete():
    values = np.array([0, 3, 4, 15, 16, 63, 64, 255, 256, 900])

    strata = _strata(values, [0, 4, 16, 64, 256])

    assert list(strata) == ["0_4", "4_16", "16_64", "64_256", "256_plus"]
    membership = np.stack(list(strata.values()))
    assert np.all(membership.sum(axis=0) == 1)
    assert np.flatnonzero(strata["0_4"]).tolist() == [0, 1]
    assert np.flatnonzero(strata["256_plus"]).tolist() == [8, 9]
