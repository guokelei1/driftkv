"""Deterministic, outcome-conditioned user panels for local-recomputation diagnostics.

This is deliberately not a population sample: existing Full/Reuse outcomes select
users, before any of the four new baselines are evaluated.
"""

from __future__ import annotations

import hashlib

import numpy as np


RULE = "mean_class_balanced_pairwise_concordance_advantage_v1"
HASH_NAMESPACE = "selective_recompute_2026_09/user_reservation_v1"


def uid_key(uid: int) -> str:
    return hashlib.sha256(f"{HASH_NAMESPACE}/{int(uid)}".encode()).hexdigest()


def request_concordance(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Fraction of opposite-label requests ranked correctly, including half ties."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("concordance requires binary labels and finite scores")
    positive = labels == 1
    if not positive.any() or positive.all():
        raise ValueError("the reference population must contain both labels")
    positives, negatives = np.sort(scores[positive]), np.sort(scores[~positive])
    result = np.empty(len(scores), dtype=np.float64)
    lower = np.searchsorted(negatives, scores[positive], side="left")
    upper = np.searchsorted(negatives, scores[positive], side="right")
    result[positive] = (lower + 0.5 * (upper - lower)) / len(negatives)
    lower = np.searchsorted(positives, scores[~positive], side="left")
    upper = np.searchsorted(positives, scores[~positive], side="right")
    result[~positive] = (len(positives) - upper + 0.5 * (upper - lower)) / len(positives)
    return result


def class_balanced_advantage(
    labels: np.ndarray, full_scores: np.ndarray, reuse_scores: np.ndarray
) -> np.ndarray:
    """Return per-request values whose population mean equals AUC Full minus Reuse.

Each value compares that request with the entire opposite-label reference pool.
It is not an individual user's AUC, and supports single-label user histories.
"""
    labels = np.asarray(labels, dtype=np.int64)
    delta = request_concordance(labels, full_scores) - request_concordance(labels, reuse_scores)
    counts = np.bincount(labels, minlength=2)
    return delta * (len(labels) / (2.0 * counts[labels]))


def select_users(
    uids: np.ndarray,
    labels: np.ndarray,
    full_scores: np.ndarray,
    reuse_scores: np.ndarray,
    *,
    evaluation_users: int = 3000,
    calibration_users: int = 32,
    canary_users: int = 16,
) -> tuple[dict[str, list[int]], dict[int, float]]:
    """Reserve controls by UID hash, then take the highest mean-advantage users.

    There is one selection pass. No population/subset AUC threshold changes the
    count or triggers a new draw. Every selected user's requests are retained.
    """
    uids = np.asarray(uids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    full_scores, reuse_scores = np.asarray(full_scores), np.asarray(reuse_scores)
    if not (uids.ndim == 1 and uids.shape == labels.shape == full_scores.shape == reuse_scores.shape):
        raise ValueError("request arrays must align")
    unique = sorted((int(uid) for uid in np.unique(uids)), key=lambda uid: (uid_key(uid), uid))
    needed = evaluation_users + calibration_users + canary_users
    if min(evaluation_users, calibration_users, canary_users) < 1 or needed > len(unique):
        raise ValueError("insufficient users for disjoint evaluation, calibration and canary panels")
    calibration = unique[:calibration_users]
    canary = unique[calibration_users:calibration_users + canary_users]
    reserved = calibration + canary
    eligible = ~np.isin(uids, reserved)
    advantage = class_balanced_advantage(labels[eligible], full_scores[eligible], reuse_scores[eligible])
    users, inverse, counts = np.unique(uids[eligible], return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=advantage) / counts
    user_scores = dict(zip((int(uid) for uid in users), (float(value) for value in means), strict=True))
    ranked = sorted(user_scores, key=lambda uid: (-user_scores[uid], uid_key(uid), uid))
    return {
        "evaluation": ranked[:evaluation_users], "calibration": calibration, "canary": canary,
    }, user_scores
