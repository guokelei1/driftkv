from __future__ import annotations

import numpy as np
import torch

from hstu_kvcache.streaming.qk_protocol_sweep_runner import (
    METRICS,
    _stable_gate,
    candidate_score_sums,
    nested_popular_candidate_ids,
    nested_uniform_candidate_ids,
    summarize_candidate_matrix,
)


def test_nested_uniform_candidates_are_unique_and_nested() -> None:
    positives = torch.tensor([1, 17, 100])
    first = nested_uniform_candidate_ids(
        positives,
        num_prediction_items=100,
        maximum_negative_count=49,
        seed=31,
    )
    second = nested_uniform_candidate_ids(
        positives,
        num_prediction_items=100,
        maximum_negative_count=99,
        seed=31,
    )
    assert torch.equal(first, second[:, :50])
    assert torch.equal(first[:, 0], positives)
    for row in first:
        assert len(torch.unique(row)) == len(row)


def test_uniform_candidate_seeds_change_the_pool() -> None:
    positives = torch.tensor([11, 29])
    first = nested_uniform_candidate_ids(
        positives,
        num_prediction_items=250_000,
        maximum_negative_count=99,
        seed=41,
    )
    second = nested_uniform_candidate_ids(
        positives,
        num_prediction_items=250_000,
        maximum_negative_count=99,
        seed=43,
    )
    assert not torch.equal(first[:, 1:], second[:, 1:])


def test_popular_candidates_remove_the_positive_without_duplicates() -> None:
    values = nested_popular_candidate_ids(
        torch.tensor([1, 8]),
        torch.arange(1, 60),
        maximum_negative_count=49,
    )
    assert values[0, :10].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert values[1, :10].tolist() == [8, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    for row in values:
        assert len(torch.unique(row)) == len(row)


def test_candidate_score_sums_reports_ranking_improvement() -> None:
    scores = torch.tensor(
        [
            [[0.0, 1.0, -1.0], [0.0, 2.0, 1.0]],
            [[2.0, 1.0, -1.0], [3.0, 2.0, 1.0]],
        ]
    )
    sums = candidate_score_sums(scores)
    assert sums.shape == (2, len(METRICS))
    assert sums[1, METRICS.index("cross_entropy")] < sums[
        0, METRICS.index("cross_entropy")
    ]
    assert sums[1, METRICS.index("mrr")] > sums[0, METRICS.index("mrr")]
    assert sums[1, METRICS.index("hit_rate_at_1")] == 2.0


def test_candidate_matrix_uses_record_cluster_bootstrap() -> None:
    sums = np.zeros((1, 1, 2, len(METRICS)), dtype=np.float64)
    sums[0, 0, 0, :] = [4.0, 0.2, 0.3, 0.4, 0.0, 0.2, 0.3]
    sums[0, 0, 1, :] = [3.0, 0.4, 0.5, 0.6, 0.2, 0.4, 0.5]
    payloads = [
        {"targets": 1, "sums": sums.copy()},
        {"targets": 1, "sums": sums.copy()},
    ]
    result = summarize_candidate_matrix(
        payloads,
        variant_names=["uniform_unique_seed_0"],
        negative_counts=[9],
        bootstrap_samples=20,
        bootstrap_seed=7,
    )
    metrics = result["protocols"]["uniform_unique_seed_0"][
        "negative_counts"
    ]["9"]["metrics"]
    assert metrics["cross_entropy"]["absolute_gap"] == 1.0
    assert metrics["ndcg_at_10"]["relative_to_reuse_percent"] > 5.0
    assert metrics["ndcg_at_10"]["positive_direction_with_ci"] is True


def test_stable_gate_prefers_99_and_does_not_reward_overshoot() -> None:
    def cell(percent: float) -> dict[str, object]:
        return {
            "metrics": {
                "ndcg_at_10": {
                    "relative_to_reuse_percent": percent,
                    "positive_direction_with_ci": True,
                },
                "mrr": {"positive_direction_with_ci": True},
            }
        }

    summary = {
        "protocols": {
            "uniform": {
                "negative_counts": {
                    "49": cell(6.0),
                    "99": cell(25.0),
                }
            }
        }
    }
    result = _stable_gate(
        summary,
        uniform_names=["uniform"],
        popularity_names=[],
        negative_counts=[49, 99],
        minimum_negative_count=49,
        preferred_negative_count=99,
        minimum_gap_percent=5.0,
        maximum_gap_percent=10.0,
    )
    assert result["status"] == "admitted_protocol_found"
    assert result["selected"]["negative_count"] == 49
    assert result["all_checked"][1]["stable_in_preferred_gap_range"] is False
