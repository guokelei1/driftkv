"""Numerical and isolation checks for outcome-conditioned diagnostic panels."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.selective_recompute_2026_09.cohort import (
    class_balanced_advantage, request_concordance, select_users,
)


def brute_auc(labels, scores):
    pairs = scores[labels == 1, None] - scores[labels == 0][None, :]
    return np.mean((pairs > 0) + 0.5 * (pairs == 0))


def test_contribution_mean_matches_auc_difference_with_ties_and_imbalance():
    labels = np.array([1, 1, 0, 0, 0, 0, 0])
    full = np.array([0.9, 0.5, 0.5, 0.2, 0.8, 0.2, 0.1])
    reuse = np.array([0.5, 0.2, 0.5, 0.3, 0.8, 0.1, 0.9])
    observed = class_balanced_advantage(labels, full, reuse).mean()
    np.testing.assert_allclose(observed, brute_auc(labels, full) - brute_auc(labels, reuse), atol=1e-15)


def test_concordance_counts_half_ties_for_both_classes():
    labels, scores = np.array([1, 1, 0, 0]), np.array([1.0, 2.0, 1.0, 3.0])
    np.testing.assert_allclose(request_concordance(labels, scores), [0.25, 0.5, 0.75, 0.0])


def test_selection_supports_single_label_users_and_is_order_independent():
    rng = np.random.default_rng(91)
    uids = np.repeat(np.arange(50), 2)
    labels = (uids % 3 == 0).astype(int)  # Every individual user has one label only.
    full, reuse = rng.normal(size=len(uids)), rng.normal(size=len(uids))
    kwargs = dict(evaluation_users=20, calibration_users=5, canary_users=4)
    panels, scores = select_users(uids, labels, full, reuse, **kwargs)
    order = rng.permutation(len(uids))
    shuffled, shuffled_scores = select_users(uids[order], labels[order], full[order], reuse[order], **kwargs)
    assert panels == shuffled
    assert scores == shuffled_scores
    assert len(set(sum(panels.values(), []))) == 29
    assert min(scores[uid] for uid in panels["evaluation"]) >= max(
        scores[uid] for uid in scores if uid not in panels["evaluation"]
    )
    changed, _ = select_users(uids, labels, reuse, full, **kwargs)
    assert changed["calibration"] == panels["calibration"]
    assert changed["canary"] == panels["canary"]
