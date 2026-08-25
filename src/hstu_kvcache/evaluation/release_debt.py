"""Release-level aggregation for paired cache harm."""

from __future__ import annotations

import numpy as np


def rolling_erase_fraction(
    *, reuse_loss: float, current_loss: float, parent_exact_rolling_loss: float
) -> float | None:
    denominator = float(parent_exact_rolling_loss) - float(current_loss)
    if denominator <= 0:
        return None
    return (float(reuse_loss) - float(current_loss)) / denominator


def release_debt(
    *,
    delta_loss: np.ndarray,
    uids: np.ndarray,
    append_counts: np.ndarray,
    horizon_days: float,
) -> dict:
    delta = np.asarray(delta_loss, dtype=np.float64)
    users = np.asarray(uids, dtype=np.int64)
    appends = np.asarray(append_counts, dtype=np.int64)
    if not (delta.shape == users.shape == appends.shape) or delta.ndim != 1 or not len(delta):
        raise ValueError("release debt arrays must be nonempty and aligned")
    if horizon_days <= 0 or np.any(appends < 0):
        raise ValueError("release horizon and append counts must be valid")
    unique_users = len(np.unique(users))
    total = float(delta.sum())
    by_append = []
    for value in np.unique(appends):
        selected = appends == value
        by_append.append({
            "append_count": int(value),
            "requests": int(selected.sum()),
            "traffic_weight": float(selected.mean()),
            "mean_delta_loss": float(delta[selected].mean()),
            "debt": float(delta[selected].sum()),
        })
    return {
        "D_release": total,
        "per_million_requests": total / len(delta) * 1_000_000,
        "per_thousand_active_users": total / unique_users * 1_000,
        "per_user_day": total / (unique_users * horizon_days),
        "traffic_weighted_debt_persistence_curve": by_append,
        "observational_not_causal": True,
    }


def fixed_query_curve(k: np.ndarray, delta_loss: np.ndarray) -> dict:
    appends = np.asarray(k, dtype=np.int64)
    delta = np.asarray(delta_loss, dtype=np.float64)
    if appends.shape != delta.shape or appends.ndim != 1 or not len(appends):
        raise ValueError("fixed-query arrays must align")
    if np.any(np.diff(appends) <= 0) or appends[0] != 0:
        raise ValueError("fixed-query k must increase strictly from zero")
    baseline = abs(float(delta[0]))
    half_life = None
    if baseline > 0:
        candidates = appends[np.abs(delta) <= 0.5 * baseline]
        if len(candidates):
            half_life = int(candidates[0])
    return {
        "k": appends.tolist(),
        "delta_loss": delta.tolist(),
        "append_half_life": half_life,
        "causal_only_if_uid_query_target_model_readout_and_cutover_prefix_are_fixed": True,
    }
