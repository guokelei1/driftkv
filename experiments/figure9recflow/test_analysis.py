"""CPU checks for paired population bootstrap and honest cross-edge summaries."""
import json

import numpy as np
import pytest

from experiments.figure9recflow.analysis import aggregate_edges, summarize
from experiments.figure9recflow.common import METRIC_NAMES


POLICIES = [{"tag": tag, "method": tag} for tag in ["reuse", "rebuild", "tail_64"]]


def record(uid, rid, values, positives=1):
    return dict(uid=uid, rid=rid, ts=100 + rid, candidates=10, positives=positives,
                metrics=np.repeat(np.array(values, np.float64)[:, None], len(METRIC_NAMES), axis=1))


def summary(values, edge, *, bootstrap=80):
    result = summarize([record(1, 1, values)], [1], POLICIES, bootstrap=bootstrap)
    result["edge"] = edge
    return result


def test_request_macro_inactive_uid_bootstrap_and_zero_positive_coverage():
    rows = [record(10, 1, [.1, .5, .3]), record(10, 2, [.1, .5, .3]),
            record(20, 3, [.1, .3, .3]), record(20, 4, [np.nan] * 3, positives=0)]
    result = summarize(rows, [30, 10, 20], POLICIES, bootstrap=200, seed=11)
    metric = result["results"][2]["metrics"]["ndcg@10"]
    assert metric["value"] == pytest.approx(.3)
    assert metric["recovery_percent"] == pytest.approx(60)
    assert result["requests"] == 4 and result["positive_requests"] == 3
    assert result["zero_positive_requests"] == 1 and result["candidates"] == 40
    assert result["request_inactive_users"] == 1
    # Independently recreate the full population draws (including inactive UID30).
    rng = np.random.default_rng(11)
    counts = np.array([0, 2, 1])
    sums = np.array([0, 1, .3])
    means, empty = [], 0
    for _ in range(200):
        chosen = rng.integers(0, 3, size=3)
        n = counts[chosen].sum()
        if n:
            means.append(sums[chosen].sum() / n)
        else:
            empty += 1
    assert result["bootstrap"]["empty_request_draws"] == empty > 0
    np.testing.assert_allclose(result["results"][1]["metrics"]["ndcg@10"]["ci95"],
                               np.quantile(means, [.025, .975]))
    assert metric["recovery_percent_ci95"] is None
    assert metric["recovery_ci_status"] == "unstable_empty_request_draws"
    json.dumps(result, allow_nan=False)


def test_negative_and_above_100_recovery_are_not_clipped():
    positive = summary([.2, .4, .5], "edge1")
    negative = summary([.2, .4, .1], "edge2")
    a = positive["results"][2]["metrics"]["recall@10"]
    b = negative["results"][2]["metrics"]["recall@10"]
    assert a["recovery_percent"] == pytest.approx(150)
    assert b["recovery_percent"] == pytest.approx(-50)
    np.testing.assert_allclose(a["recovery_percent_ci95"], [150, 150])
    np.testing.assert_allclose(b["recovery_percent_ci95"], [-50, -50])


def test_reference_gap_crossing_zero_withholds_ratio_ci():
    result = summarize([record(1, 1, [.4, .8, .6]), record(2, 2, [.4, .2, .5])],
                       [1, 2], POLICIES, bootstrap=500)
    reference = result["reference_gaps"]["ndcg@10"]
    assert reference["ci95"][0] < 0 < reference["ci95"][1]
    metric = result["results"][2]["metrics"]["ndcg@10"]
    assert metric["recovery_percent"] == pytest.approx(150)
    assert metric["recovery_ci_status"] == "unstable_denominator"
    assert metric["recovery_percent_ci95"] is None
    assert metric["ci95"] is not None


def test_zero_gap_and_no_positive_requests_are_explicit_missing_values():
    zero_gap = summary([.4, .4, .6], "edge1")
    assert zero_gap["results"][2]["metrics"]["ndcg@10"]["recovery_percent"] is None
    empty = summarize([record(1, 1, [np.nan] * 3, positives=0)], [1, 2], POLICIES, bootstrap=20)
    assert not empty["quality_defined"]
    assert empty["requests"] == 1
    assert empty["results"][0]["metrics"]["ndcg@10"]["value"] is None
    assert empty["bootstrap"]["empty_request_draws"] == 20
    json.dumps(empty, allow_nan=False)


def test_equal_edge_recovery_mean_not_ratio_of_metric_means():
    a = summary([.1, .2, .2], "a")  # 100% of .1
    b = summary([.1, .5, .1], "b")  # 0% of .4
    result = aggregate_edges([a, b], expected_edges=["a", "b"])
    value = result["frontier"][2]["metrics"]["ndcg@10"]
    assert value["mean_recovery_percent"] == pytest.approx(50)  # ratio of means would be 20%
    assert value["recovery_percent_by_edge"] == pytest.approx([100, 0])
    assert value["mean_recovery_ci95"] is None
    assert len(result["edges"]) == 2
    undefined = summary([.1, .1, .2], "c")
    aggregate = aggregate_edges([a, undefined])
    metric = aggregate["frontier"][2]["metrics"]["ndcg@10"]
    assert metric["mean_recovery_percent"] is None and metric["undefined_recovery_edges"] == ["c"]


def test_negative_reference_gap_preserved_with_interpretation_flag():
    result = summary([.6, .4, .5], "negative")
    assert result["reference_gaps"]["ndcg@10"]["negative_point_gap"]
    metric = result["results"][2]["metrics"]["ndcg@10"]
    assert metric["recovery_percent"] == pytest.approx(50)
    assert metric["recovery_ci_status"] == "stable_negative_gap"


def test_mismatched_inputs_are_not_silently_filtered():
    row = record(1, 1, [.1, .5, .3])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([row, row], [1], POLICIES)
    with pytest.raises(ValueError, match="outside"):
        summarize([row], [2], POLICIES)
    with pytest.raises(ValueError, match="undefined"):
        summarize([record(1, 1, [0, 0, 0], positives=0)], [1], POLICIES)
    with pytest.raises(ValueError, match="entire declared"):
        aggregate_edges([summary([.1, .2, .15], "a")], expected_edges=["a", "b"])


def test_six_metrics_keep_the_declared_order_and_json_null_input():
    row = record(1, 1, [.1, .2, .15])
    row["metrics"][2] = [.11, .12, .13, .14, .15, .16]
    empty = record(2, 2, [np.nan] * 3, positives=0)
    empty["metrics"] = [[None] * 6 for _ in POLICIES]
    result = summarize([row, empty], [1, 2], POLICIES, bootstrap=0)
    assert result["metric_names"] == METRIC_NAMES
    assert [result["results"][2]["metrics"][name]["value"] for name in METRIC_NAMES] == [.11, .12, .13, .14, .15, .16]
    assert result["results"][2]["metrics"]["ndcg@10"]["recovery_ci_status"] == "not_estimated"
    json.dumps(result, allow_nan=False)
