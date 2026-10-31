"""Observed-positive retrieval metrics; an unobserved item is not a true negative."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import numpy as np


def request_metrics(
    ranked_ids: Sequence[int],
    positive_ids: Sequence[int],
    catalog: set[int],
    ks: Sequence[int] = (20, 50, 100),
) -> dict[str, float]:
    """Keep OOV positives in the main denominator, including all-OOV requests."""
    positives = set(map(int, positive_ids))
    known = positives & catalog
    ranked = list(dict.fromkeys(int(i) for i in ranked_ids if int(i) in catalog))
    result = {"positives": float(len(positives)), "known_positives": float(len(known))}
    for k in ks:
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        hits = np.array([i in positives for i in ranked[:k]], dtype=float)
        dcg = float(np.dot(hits, discounts[:len(hits)]))
        result[f"recall@{k}"] = float(hits.sum() / len(positives)) if positives else float("nan")
        result[f"ndcg@{k}"] = dcg / float(discounts[:min(k, len(positives))].sum()) if positives else float("nan")
        result[f"warm_recall@{k}"] = float(hits.sum() / len(known)) if known else float("nan")
        result[f"warm_ndcg@{k}"] = dcg / float(discounts[:min(k, len(known))].sum()) if known else float("nan")
    return result


def random_expected_metrics(
    positive_count: int,
    known_positive_count: int,
    candidate_count: int,
    ks: Sequence[int] = (20, 50, 100),
) -> dict[str, float]:
    """Expected metrics for a uniform permutation of the same allowed items.

    Counts refer to distinct positives and unique candidates. For a sampled
    panel, ``candidate_count`` includes injected known positives as well as
    distractors. This random baseline is over videos, not category branches.
    OOV positives remain in the main denominators, as in ``request_metrics``.
    """
    if min(positive_count, known_positive_count, candidate_count) < 0:
        raise ValueError("random-baseline counts must be nonnegative")
    if known_positive_count > min(positive_count, candidate_count):
        raise ValueError("known positives must be a subset of positives and candidates")
    result = {"positives": float(positive_count), "known_positives": float(known_positive_count)}
    relevance_probability = known_positive_count / candidate_count if candidate_count else 0.0
    for k in ks:
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        ranked_count = min(k, candidate_count)
        expected_hits = relevance_probability * ranked_count
        expected_dcg = relevance_probability * float(discounts[:ranked_count].sum())
        result[f"recall@{k}"] = expected_hits / positive_count if positive_count else float("nan")
        result[f"ndcg@{k}"] = expected_dcg / float(discounts[:min(k, positive_count)].sum()) if positive_count else float("nan")
        result[f"warm_recall@{k}"] = expected_hits / known_positive_count if known_positive_count else float("nan")
        result[f"warm_ndcg@{k}"] = expected_dcg / float(discounts[:min(k, known_positive_count)].sum()) if known_positive_count else float("nan")
    return result


def aggregate_metrics(rows: Sequence[dict[str, float]], uids: Sequence[int] | None = None) -> dict:
    if not rows:
        return {"requests": 0, "positive_requests": 0}
    positive = np.array([r["positives"] > 0 for r in rows])
    warm = np.array([r["known_positives"] > 0 for r in rows])
    total = sum(r["positives"] for r in rows)
    result = {
        "requests": len(rows),
        "positive_requests": int(positive.sum()),
        "zero_positive_requests": int((~positive).sum()),
        "all_oov_positive_requests": int((positive & ~warm).sum()),
        "positive_targets": int(total),
        "known_positive_targets": int(sum(r["known_positives"] for r in rows)),
        "target_coverage": sum(r["known_positives"] for r in rows) / total if total else None,
    }
    metric_keys = [key for key in rows[0] if "@" in key]
    for key in metric_keys:
        values = np.array([r[key] for r in rows])
        valid = np.isfinite(values)
        result[key] = float(values[valid].mean()) if valid.any() else None
    if uids is not None:
        uid_array = np.asarray(uids)
        by_user = {}
        for key in metric_keys:
            values = np.array([r[key] for r in rows])
            user_values = []
            for uid in np.unique(uid_array):
                selected = values[(uid_array == uid) & np.isfinite(values)]
                if len(selected):
                    user_values.append(float(selected.mean()))
            by_user[key] = float(np.mean(user_values)) if user_values else None
        result["user_mean"] = by_user
        result["users"] = int(len(np.unique(uid_array)))
    return result


def evaluate_rankings(
    rankings: Sequence[Sequence[int]],
    positives: Sequence[Sequence[int]],
    catalog: Iterable[int],
    uids: Sequence[int] | None = None,
    ks: Sequence[int] = (20, 50, 100),
) -> tuple[dict, list[dict[str, float]]]:
    known = set(map(int, catalog))
    rows = [request_metrics(r, p, known, ks) for r, p in zip(rankings, positives, strict=True)]
    return aggregate_metrics(rows, uids), rows


def sample_candidates(
    catalog_ids: np.ndarray,
    positive_ids: Sequence[int],
    distractors: int,
    seed: int,
    request_id: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """A diagnostic candidate panel; all known positives plus sampled distractors.

    Missing/OOV positives remain in the metric's original positive set. Samples
    are shared across model versions by (seed, request_id), not model scores.
    """
    catalog_ids = np.asarray(catalog_ids)
    known_positive = np.isin(catalog_ids, np.asarray(positive_ids))
    pool = np.flatnonzero(~known_positive)
    digest = hashlib.sha256(f"recflow:candidates:{seed}:{request_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    count = min(distractors, len(pool))
    probabilities = None
    if weights is not None and count:
        probabilities = np.asarray(weights, dtype=float)[pool]
        # Small uniform smoothing gives every catalog item nonzero support.
        probabilities = probabilities + 1.0
        probabilities /= probabilities.sum()
    chosen = rng.choice(pool, size=count, replace=False, p=probabilities)
    return np.concatenate([catalog_ids[known_positive], catalog_ids[chosen]])
