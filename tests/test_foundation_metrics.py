from __future__ import annotations

import numpy as np

from hstu_kvcache.evaluation import (
    bernoulli_js,
    fixed_query_curve,
    paired_harm,
    release_debt,
    rolling_erase_fraction,
    stable_log_loss,
)


def test_paired_harm_sign_and_positive_negative_mass() -> None:
    labels = np.asarray([1, 0, 1, 0])
    current = np.asarray([2.0, -2.0, 0.5, -0.5])
    reuse = np.asarray([1.0, -1.0, 1.0, -1.0])
    delta = stable_log_loss(reuse, labels) - stable_log_loss(current, labels)
    result = paired_harm(
        uids=np.asarray([10, 10, 20, 20]),
        labels=labels,
        reuse_logits=reuse,
        current_logits=current,
        bootstrap_repetitions=50,
    )
    assert np.isclose(result["event_weighted_mean"], delta.mean())
    assert np.isclose(result["D_positive"], np.maximum(delta, 0).sum())
    assert np.isclose(result["D_negative"], np.maximum(-delta, 0).sum())
    assert result["D_positive"] > 0 and result["D_negative"] > 0


def test_js_is_symmetric_and_zero_only_for_equal_logits() -> None:
    left = np.asarray([-2.0, 0.0, 3.0])
    right = np.asarray([-1.0, 0.0, 1.0])
    assert np.allclose(bernoulli_js(left, right), bernoulli_js(right, left))
    assert np.all(bernoulli_js(left, left) == 0)
    assert bernoulli_js(left, right)[[0, 2]].min() > 0


def test_rolling_erase_fraction_is_na_for_nonpositive_parent_gain() -> None:
    assert rolling_erase_fraction(
        reuse_loss=0.9, current_loss=0.8, parent_exact_rolling_loss=1.0
    ) == 0.5
    assert rolling_erase_fraction(
        reuse_loss=0.9, current_loss=1.0, parent_exact_rolling_loss=0.9
    ) is None


def test_release_debt_names_natural_curve_observational() -> None:
    result = release_debt(
        delta_loss=np.asarray([0.1, 0.2, -0.1]),
        uids=np.asarray([1, 1, 2]),
        append_counts=np.asarray([0, 1, 1]),
        horizon_days=7,
    )
    assert np.isclose(result["D_release"], 0.2)
    assert result["observational_not_causal"] is True
    assert result["traffic_weighted_debt_persistence_curve"][1]["append_count"] == 1


def test_fixed_query_curve_reports_causal_append_half_life() -> None:
    result = fixed_query_curve(
        np.asarray([0, 1, 2, 4, 8]), np.asarray([0.4, 0.35, 0.19, 0.1, 0.02])
    )
    assert result["append_half_life"] == 2
    assert result["causal_only_if_uid_query_target_model_readout_and_cutover_prefix_are_fixed"]
