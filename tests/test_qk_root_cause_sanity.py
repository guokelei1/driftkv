from __future__ import annotations

from hstu_kvcache.streaming.qk_root_cause_sanity import _aggregate


def test_root_cause_aggregate_orients_fresh_advantages() -> None:
    methods = [
        "fresh_full_a",
        "fresh_full_b",
        "stale_theta1",
        "zero_prefix",
        "no_prefix",
        "wrong_user_fresh",
        "shuffled_prefix",
        "recent_4",
        "recent_16",
        "recent_64",
    ]
    records = []
    for record in range(8):
        metric_sums = {}
        hidden = {}
        for method in methods:
            degraded = method not in ("fresh_full_a", "fresh_full_b")
            metric_sums[method] = {
                "cross_entropy": 12.0 if degraded else 10.0,
                "ndcg_at_10": 1.0 if degraded else 2.0,
                "mrr": 0.5 if degraded else 1.0,
                "hit_rate_at_10": 1.0 if degraded else 2.0,
                "hit_rate_at_50": 2.0 if degraded else 3.0,
                "hit_rate_at_200": 3.0 if degraded else 4.0,
            }
            hidden[method] = 0.1 if degraded else 0.0
        records.append(
            {
                "record": record,
                "targets": 5,
                "metric_sums": metric_sums,
                "hidden_relative_error": hidden,
                "fresh_duplicate": {
                    "cache_maximum_absolute_error": 0.0,
                    "hidden_maximum_absolute_error": 0.0,
                },
                "canonical_equivalence": {
                    "maximum_nll_absolute_error": 0.0,
                    "ranks_equal": True,
                },
            }
        )
    result = _aggregate(
        records,
        methods,
        bootstrap_samples=64,
        bootstrap_seed=7,
    )
    assert result["sanity"]["implementation_passed"] is True
    assert (
        result["fresh_reference_comparisons"]["stale_theta1"]["cross_entropy"][
            "fresh_advantage_absolute"
        ]
        > 0
    )
    assert (
        result["fresh_reference_comparisons"]["stale_theta1"]["ndcg_at_10"][
            "fresh_advantage_absolute"
        ]
        > 0
    )
