from __future__ import annotations

import pytest

from hstu_kvcache.data.identifiability import (
    IDENTIFIABILITY_FEATURES,
    grouped_folds,
    history_item_stats,
    identifiability_vector,
    nearest_within_caliper,
    request_conditional_metrics,
    select_caliper,
    uid_fold,
)


def test_identifiability_vector_ignores_panel_slot() -> None:
    history = [(7, 100, 1), (8, 200, 1), (9, 300, 1)]
    artist = [ -1] * 20
    artist[7] = artist[8] = artist[9] = 3
    stats = history_item_stats(history, artist, recent_window=2, max_history=8)
    popularity = [0] * 20
    popularity[8] = 11
    ranks = {8: 17, 7: 4}
    first = identifiability_vector(
        8, query_ts=400, stats=stats, popularity=popularity, qmain_ranks=ranks, artist_by_item=artist
    )
    # A different assembled order must not change the item's features.
    second = identifiability_vector(
        8, query_ts=400, stats=stats, popularity=popularity, qmain_ranks=ranks, artist_by_item=artist
    )
    assert first == second
    assert IDENTIFIABILITY_FEATURES[8] == "log_proposal_rank"
    assert first[8] != pytest.approx(0.0)  # rank 17, not slot 0
    assert "slot" not in "".join(IDENTIFIABILITY_FEATURES)


def test_proposal_rank_rejects_non_positive_map_entries() -> None:
    stats = history_item_stats([(3, 1, 1)], [-1, -1, -1, -1])
    with pytest.raises(ValueError):
        identifiability_vector(
            3,
            query_ts=2,
            stats=stats,
            popularity=[0, 0, 0, 1],
            qmain_ranks={3: 0},
            artist_by_item=[-1, -1, -1, -1],
        )


def test_grouped_folds_never_split_the_same_uid() -> None:
    uids = [10, 10, 11, 12, 12, 13, 14, 15, 16, 17]
    for train, test in grouped_folds(uids, folds=3, seed=1):
        assert {uids[i] for i in train}.isdisjoint({uids[i] for i in test})
    assert uid_fold(99, folds=5, seed=1) == uid_fold(99, folds=5, seed=1)


def test_request_conditional_metrics_are_within_request() -> None:
    # Target always has the global-lowest score, but is best inside request 1
    # if we only looked globally we would call it a failure everywhere.
    metrics = request_conditional_metrics([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    assert metrics["cauc"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)


def test_caliper_selection_ignores_auc_fields() -> None:
    summaries = [
        {
            "caliper": 0.5,
            "complete": 500,
            "stratum_complete": {"a": 100, "b": 100, "c": 100},
            "max_abs_smd": 0.10,
            "mean_abs_smd": 0.08,
            "holdout_auc_if_leaked": 0.99,
        },
        {
            "caliper": 1.0,
            "complete": 800,
            "stratum_complete": {"a": 200, "b": 200, "c": 200},
            "max_abs_smd": 0.20,
            "mean_abs_smd": 0.12,
            "holdout_auc_if_leaked": 0.51,
        },
    ]
    chosen = select_caliper(summaries)
    assert chosen["caliper"] == 0.5
    assert chosen["status"] == "balanced"


def test_nearest_within_caliper_keeps_order_and_limit() -> None:
    chosen = nearest_within_caliper([0.0, 0.0], [[0.1, 0.0], [3.0, 0.0], [0.2, 0.0]], k=2, caliper=1.0)
    assert chosen == [0, 2]
