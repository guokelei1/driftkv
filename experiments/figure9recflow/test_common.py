import numpy as np

from experiments.figure9recflow.common import configuration, policies, ranking_metrics
from experiments.recflow_full_population.code.protocol import request_metrics


def test_method_ranking_matches_existing_request_oracle_and_ties():
    rng = np.random.default_rng(97)
    ids = rng.permutation(np.arange(59)*7+1)
    labels = rng.choice([-1, 0, 1], len(ids))
    scores = rng.integers(-2, 3, (29, len(ids))).astype(np.float32)
    actual, top = ranking_metrics(ids, labels, scores)
    for i, score in enumerate(scores):
        wanted = request_metrics(ids, score, labels)
        for j, k in enumerate((5, 10, 20)):
            np.testing.assert_allclose(actual[i, 2*j:2*j+2],
                                       [wanted[f"ndcg@{k}"], wanted[f"recall@{k}"]], atol=1e-15)
        expected = np.lexsort((ids, -score))[:20]
        np.testing.assert_array_equal(top[i], expected)


def test_zero_positive_retained_as_undefined_not_synthetic_negative():
    result, top = ranking_metrics(np.array([9, 2, 4]), np.array([-1, 0, -1]), np.zeros((2, 3)))
    assert np.isnan(result).all()
    np.testing.assert_array_equal(top, [[1, 2, 0], [1, 2, 0]])


def test_config_declares_all_29_natural_points_without_launch():
    rows = policies()
    assert len(rows) == len({x['tag'] for x in rows}) == 29
    assert configuration()['automatic_successors'] is False
    assert configuration()['hash_protection'] is False
