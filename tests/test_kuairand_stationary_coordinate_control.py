import pytest

from hstu_kvcache.streaming.kuairand_stationary_coordinate_control import (
    _fidelity_summary,
    _select_transform,
    load_stationary_coordinate_control_config,
)

CONFIG = (
    "configs/evokv_root_cause/"
    "kuairand_stationary_coordinate_control_theta5_v1_v8_20260810_v0.json"
)


def test_stationary_coordinate_control_binds_large_anchor_and_eight_versions():
    document = load_stationary_coordinate_control_config(CONFIG)
    assert document["source"]["anchor_version"] == 5
    assert document["evaluation"]["virtual_versions"] == 8
    assert len(document["transform_candidates"]) == 6


def test_stationary_coordinate_selection_uses_label_free_tuning_band():
    values = [
        {
            "transform": {
                "name": name,
                "key_log_step": step,
                "value_log_step": step,
            },
            "maximum_fresh_metric_difference": 0.0,
            "tuning_fidelity": {"top10_changed_fraction": changed},
        }
        for name, step, changed in (
            ("small", 0.01, 0.04),
            ("middle", 0.02, 0.09),
            ("large", 0.03, 0.14),
        )
    ]
    selected = _select_transform(
        values,
        {
            "metric": "top10_changed_fraction",
            "minimum": 0.05,
            "maximum": 0.15,
            "target": 0.1,
        },
    )
    assert selected["name"] == "middle"
    assert selected["inside_predeclared_band"]


def test_stationary_coordinate_fidelity_aggregates_users_before_records():
    records = [
        {
            "user_id": user,
            "fidelity": {
                "hidden_cosine": value,
                "score_cosine": value,
                "score_relative_error": 1.0 - value,
                "score_kl_from_fresh": 1.0 - value,
                "top10_overlap_with_fresh": value,
            },
        }
        for user, value in ((1, 0.8), (1, 1.0), (2, 0.5))
    ]
    summary = _fidelity_summary(records)
    assert summary["users"] == 2
    assert summary["user_mean"]["top10_overlap_with_fresh"]["mean"] == pytest.approx(0.7)
    assert summary["top10_changed_fraction"] == pytest.approx(0.3)
