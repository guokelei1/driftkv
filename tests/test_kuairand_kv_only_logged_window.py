from __future__ import annotations

import torch

from hstu_kvcache.streaming.kuairand_kv_only_logged_window import (
    METRICS,
    _window_metric_values,
)


def test_logged_window_metrics_rank_positives_first() -> None:
    values = dict(
        zip(
            METRICS,
            _window_metric_values(
                torch.tensor([4.0, 3.0, 2.0, 1.0]),
                torch.tensor([True, True, False, False]),
            ).tolist(),
            strict=True,
        )
    )
    assert values["roc_auc"] == 1.0
    assert values["average_precision"] == 1.0
    assert values["ndcg_at_10"] == 1.0
    assert values["recall_at_10"] == 1.0


def test_logged_window_auc_counts_ties_as_half() -> None:
    values = dict(
        zip(
            METRICS,
            _window_metric_values(
                torch.tensor([2.0, 2.0]),
                torch.tensor([True, False]),
            ).tolist(),
            strict=True,
        )
    )
    assert values["roc_auc"] == 0.5
