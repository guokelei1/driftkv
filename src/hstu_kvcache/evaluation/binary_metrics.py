"""Pure NumPy metrics for sealed binary foundation observations."""

from __future__ import annotations

import hashlib

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp = np.exp(values[~positive])
    output[~positive] = exp / (1.0 + exp)
    return output


def stable_log_loss(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if values.shape != targets.shape or values.ndim != 1:
        raise ValueError("logits and labels must be aligned rank-one arrays")
    if not np.isin(targets, (0.0, 1.0)).all():
        raise ValueError("labels must be binary")
    return np.maximum(values, 0.0) - targets * values + np.log1p(np.exp(-np.abs(values)))


def bernoulli_js(left_logits: np.ndarray, right_logits: np.ndarray) -> np.ndarray:
    left, right = sigmoid(left_logits), sigmoid(right_logits)
    if left.shape != right.shape:
        raise ValueError("JS inputs must align")
    midpoint = 0.5 * (left + right)
    epsilon = 1e-15
    left, right, midpoint = (
        np.clip(value, epsilon, 1.0 - epsilon) for value in (left, right, midpoint)
    )
    kl_left = left * np.log(left / midpoint) + (1 - left) * np.log((1 - left) / (1 - midpoint))
    kl_right = right * np.log(right / midpoint) + (1 - right) * np.log((1 - right) / (1 - midpoint))
    return 0.5 * (kl_left + kl_right)


def _equal_user_weights(uids: np.ndarray) -> np.ndarray:
    values = np.asarray(uids, dtype=np.int64)
    if values.ndim != 1 or not len(values):
        raise ValueError("uids must be a nonempty rank-one array")
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.sum()


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = labels == 1
    positives, negatives = int(positive.sum()), int((~positive).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(scores)
    return float((ranks[positive].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * sorted_labels) / positives)


def binary_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float | None]:
    targets = np.asarray(labels, dtype=np.int64)
    values = np.asarray(logits, dtype=np.float64)
    losses = stable_log_loss(values, targets)
    probabilities = sigmoid(values)
    dislike = targets == 0
    return {
        "log_loss": float(losses.mean()),
        "Brier": float(np.mean((probabilities - targets) ** 2)),
        "ROC_AUC": _roc_auc(targets, probabilities),
        "dislike_PR_AUC": _average_precision(1 - targets, 1.0 - probabilities),
        "dislike_only_log_loss": float(losses[dislike].mean()) if dislike.any() else None,
    }


def _cluster_bootstrap(
    uids: np.ndarray,
    values: np.ndarray,
    *,
    repetitions: int,
    namespace: str,
) -> dict[str, float | int]:
    unique, inverse = np.unique(uids, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    user_means = sums / counts
    seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.integers(0, len(unique), size=len(unique))
        estimates[index] = user_means[sampled].mean()
    return {
        "repetitions": repetitions,
        "p2_5": float(np.quantile(estimates, 0.025)),
        "p97_5": float(np.quantile(estimates, 0.975)),
    }


def paired_harm(
    *,
    uids: np.ndarray,
    labels: np.ndarray,
    reuse_logits: np.ndarray,
    current_logits: np.ndarray,
    bootstrap_repetitions: int = 2000,
    namespace: str = "foundation_paired_harm",
) -> dict:
    uids = np.asarray(uids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    reuse = np.asarray(reuse_logits, dtype=np.float64)
    current = np.asarray(current_logits, dtype=np.float64)
    if not (uids.shape == labels.shape == reuse.shape == current.shape) or uids.ndim != 1:
        raise ValueError("paired harm arrays must align")
    delta = stable_log_loss(reuse, labels) - stable_log_loss(current, labels)
    weights = _equal_user_weights(uids)
    unique, inverse = np.unique(uids, return_inverse=True)
    user_sums = np.bincount(inverse, weights=delta)
    user_counts = np.bincount(inverse)
    user_delta = user_sums / user_counts
    positive_user_mass = np.maximum(user_sums, 0.0)
    sorted_mass = np.sort(positive_user_mass)[::-1]
    total_positive_user_mass = float(sorted_mass.sum())
    concentration = {}
    for fraction in (0.01, 0.05, 0.10):
        count = max(1, int(np.ceil(len(unique) * fraction)))
        concentration[f"top_{int(fraction * 100)}pct_users"] = (
            float(sorted_mass[:count].sum() / total_positive_user_mass)
            if total_positive_user_mass > 0 else None
        )
    return {
        "requests": int(len(delta)),
        "users": int(len(unique)),
        "event_weighted_mean": float(delta.mean()),
        "user_weighted_mean": float(np.dot(weights, delta)),
        "D_positive": float(np.maximum(delta, 0.0).sum()),
        "D_negative": float(np.maximum(-delta, 0.0).sum()),
        "positive_request_fraction": float(np.mean(delta > 0)),
        "positive_user_fraction": float(np.mean(user_delta > 0)),
        "positive_harm_user_concentration": concentration,
        "user_quantiles": {
            name: float(np.quantile(user_delta, value))
            for name, value in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99))
        },
        "user_cluster_bootstrap_95CI": _cluster_bootstrap(
            uids, delta, repetitions=bootstrap_repetitions, namespace=namespace
        ),
    }
