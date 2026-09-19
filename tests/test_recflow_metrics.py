from itertools import permutations

import numpy as np

from hstu_kvcache.recflow.metrics import (
    evaluate_rankings,
    random_expected_metrics,
    request_metrics,
    sample_candidates,
)


def test_multiple_positives_oov_and_all_oov_requests_remain_in_denominator():
    summary, rows = evaluate_rankings([[1, 1, 2], [2, 1], [1]], [[1, 99], [98], []], [1, 2], [7, 7, 8], ks=(2,))
    ideal = 1 + 1 / np.log2(3)
    assert np.isclose(rows[0]["ndcg@2"], 1 / ideal)
    assert rows[0]["recall@2"] == 0.5
    assert rows[0]["warm_ndcg@2"] == 1.0
    assert rows[1]["ndcg@2"] == 0.0
    assert summary["positive_requests"] == 2
    assert summary["all_oov_positive_requests"] == 1
    assert summary["recall@2"] == 0.25
    assert summary["target_coverage"] == 1 / 3


def test_candidate_panel_is_shared_and_does_not_inject_oov_target():
    catalog = np.arange(1, 101)
    a = sample_candidates(catalog, [3, 5, 101], 10, 17, 200)
    b = sample_candidates(catalog, [3, 5, 101], 10, 17, 200)
    assert np.array_equal(a, b)
    assert len(set(a)) == 12
    assert 3 in a and 5 in a and 101 not in a
    assert request_metrics(a, [3, 5, 101], set(catalog), ks=(20,))["recall@20"] == 2 / 3


def test_random_expectation_matches_every_small_pool_permutation_with_oov():
    catalog = {1, 2, 3, 4}
    ks = (1, 2, 5)
    # Check multi-positive/OOV denominators, all-OOV zeros, and no-positive NaNs.
    for positives in ([1, 2, 99], [99], []):
        enumerated = [request_metrics(order, positives, catalog, ks) for order in permutations(catalog)]
        expected = random_expected_metrics(len(positives), len(set(positives) & catalog), len(catalog), ks)
        assert expected.keys() == enumerated[0].keys()
        for key, value in expected.items():
            np.testing.assert_allclose(value, np.mean([row[key] for row in enumerated]), equal_nan=True)
