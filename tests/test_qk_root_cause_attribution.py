from __future__ import annotations

from hstu_kvcache.streaming.qk_root_cause_attribution import (
    _is_kv_parameter,
    _metric_comparison,
)


def test_parameter_group_partition_is_disjoint() -> None:
    assert _is_kv_parameter("core.blocks.3.attn.k_proj.weight")
    assert _is_kv_parameter("core.blocks.3.attn.v_proj.weight")
    assert not _is_kv_parameter("core.blocks.3.attn.q_proj.weight")
    assert not _is_kv_parameter("core.blocks.3.gate_proj.weight")


def test_cross_model_comparison_orients_new_model_advantage() -> None:
    records = [
        {
            "targets": 4,
            "old": {
                "cross_entropy": 8.0,
                "ndcg_at_10": 1.0,
                "mrr": 0.5,
                "hit_rate_at_10": 1.0,
                "hit_rate_at_50": 2.0,
                "hit_rate_at_200": 3.0,
            },
            "new": {
                "cross_entropy": 6.0,
                "ndcg_at_10": 2.0,
                "mrr": 1.0,
                "hit_rate_at_10": 2.0,
                "hit_rate_at_50": 3.0,
                "hit_rate_at_200": 4.0,
            },
        }
        for _ in range(8)
    ]
    result = _metric_comparison(
        records,
        "old",
        "new",
        lambda value: value["old"],
        lambda value: value["new"],
        bootstrap_samples=64,
        bootstrap_seed=5,
    )
    assert result["cross_entropy"]["new_advantage_absolute"] > 0
    assert result["ndcg_at_10"]["new_advantage_absolute"] > 0
    assert result["mrr"]["new_advantage_positive_with_ci"] is True
