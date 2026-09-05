from __future__ import annotations

import numpy as np

from hstu_kvcache.evaluation import (
    bernoulli_js,
    paired_harm,
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
