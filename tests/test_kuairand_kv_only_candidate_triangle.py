from __future__ import annotations

import torch

from hstu_kvcache.streaming.kuairand_kv_only_candidate_triangle import (
    METRICS,
    _metric_sums_from_scores,
)


def test_candidate_metric_sums_use_positive_at_index_zero() -> None:
    scores = torch.tensor(
        [
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 3.0, 2.0],
        ]
    )
    values = dict(zip(METRICS, _metric_sums_from_scores(scores).tolist(), strict=True))
    assert values["mrr"] == 1.25
    assert values["hit_rate_at_1"] == 1.0
    assert values["hit_rate_at_5"] == 2.0
    assert values["pairwise_win_rate"] == 1.0


def test_candidate_pairwise_win_rate_counts_ties_as_half() -> None:
    scores = torch.tensor([[2.0, 1.0, 2.0, 3.0]])
    values = dict(zip(METRICS, _metric_sums_from_scores(scores).tolist(), strict=True))
    assert values["pairwise_win_rate"] == 0.5
