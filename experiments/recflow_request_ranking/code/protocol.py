"""Temporal protocol and observed-positive metrics; no training or data mutation."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np

FIRST_DATE = date(2024, 1, 13)
FIRST_COMPLETE_DAY = 24
FOUNDATION_END = 26
LAST_UPDATE_DAY = 35
CONFIRMATION_DAY = 37
DATASET_TIMEZONE = ZoneInfo("Asia/Shanghai")


def day_date(day: int) -> str:
    if not 1 <= day <= 37:
        raise ValueError("RecFlow day must be in 1..37")
    return (FIRST_DATE + timedelta(days=day - 1)).isoformat()


def download_files() -> list[str]:
    return [f"{day_date(day)}.feather" for day in range(FIRST_COMPLETE_DAY, 38)]


def day_bounds_ms(day: int) -> tuple[int, int]:
    """Half-open [start, end) matching the actual Beijing-date daily files."""
    calendar_day = date.fromisoformat(day_date(day))
    start = datetime.combine(calendar_day, time.min, tzinfo=DATASET_TIMEZONE)
    end = start + timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def version_schedule() -> list[dict]:
    rows = [{
        "version": "v0", "parent": None,
        "training_days": [12, FOUNDATION_END],
        "exposure_warmup_days": [12, 23],
        "candidate_alignment_days": [24, FOUNDATION_END],
        "available_from_day": 27, "evaluation_days": [27, 27],
    }]
    for index, day in enumerate(range(FOUNDATION_END + 1, LAST_UPDATE_DAY + 1), 1):
        rows.append({
            "version": f"v{index}", "parent": f"v{index - 1}",
            "training_days": [day, day], "candidate_alignment_days": [day, day],
            "available_from_day": day + 1, "evaluation_days": [day + 1, day + 1],
        })
    for row in rows:
        row["training_dates"] = [day_date(d) for d in row["training_days"]]
        row["evaluation_dates"] = [day_date(d) for d in row["evaluation_days"]]
        row["training_interval_ms_half_open"] = [day_bounds_ms(row["training_days"][0])[0],
                                                  day_bounds_ms(row["training_days"][1])[1]]
        row["evaluation_interval_ms_half_open"] = [day_bounds_ms(row["evaluation_days"][0])[0],
                                                    day_bounds_ms(row["evaluation_days"][1])[1]]
    return rows


def observed_labels(candidate_ids, exposed_ids, effective_ids) -> np.ndarray:
    """-1 = unexposed/unknown, 0 = observed ineffective, 1 = observed effective.

    Complete-stage membership must be established upstream. Never inject a
    missing positive or collapse unknown items into observed negatives here.
    """
    candidates = np.asarray(candidate_ids)
    if candidates.ndim != 1 or not len(candidates):
        raise ValueError("Expected a nonempty one-dimensional candidate set")
    if len(np.unique(candidates)) != len(candidates):
        raise ValueError("Candidate videos must be deduplicated before labeling")
    exposed, positive = set(map(int, exposed_ids)), set(map(int, effective_ids))
    if not positive <= exposed:
        raise ValueError("Effective positives must have observed exposure")
    if not exposed <= set(map(int, candidates)):
        raise ValueError("Exposure missing from stage candidates: audit the join; do not inject")
    labels = np.full(len(candidates), -1, dtype=np.int8)
    labels[np.isin(candidates, list(exposed))] = 0
    labels[np.isin(candidates, list(positive))] = 1
    return labels


def request_metrics(candidate_ids, scores, labels, ks=(5, 10, 20)) -> dict:
    """Standard DCG/IDCG per complete request, binary *observed* relevance.

    Unknown items receive zero observed gain; this is not a negative-feedback
    assertion. Zero-positive requests are counted, but their NDCG is undefined.
    Raw video ID breaks score ties deterministically, independently of labels.
    """
    candidates, scores, labels = map(np.asarray, (candidate_ids, scores, labels))
    if candidates.ndim != 1 or candidates.shape != scores.shape or labels.shape != scores.shape:
        raise ValueError("Candidate IDs, scores, labels must be equal-length vectors")
    if not len(candidates) or len(np.unique(candidates)) != len(candidates):
        raise ValueError("Expected nonempty distinct raw candidate IDs")
    if not np.isfinite(scores).all() or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("Nonfinite scores or invalid feedback states")
    if not ks or any(not isinstance(k, (int, np.integer)) or k < 1 for k in ks):
        raise ValueError("Metric cutoffs must be positive integers")
    positive_count = int((labels == 1).sum())
    ranking = np.lexsort((candidates, -scores))
    gains = (labels[ranking] == 1).astype(np.float64)
    result = {"candidates": len(candidates), "exposed": int((labels >= 0).sum()),
              "positives": positive_count, "zero_positive": positive_count == 0}
    for k in ks:
        discounts = 1.0 / np.log2(np.arange(2, min(k, len(gains)) + 2))
        result[f"ndcg@{k}"] = (
            float(gains[:k] @ discounts / discounts[:min(k, positive_count)].sum())
            if positive_count else None
        )
        result[f"recall@{k}"] = float(gains[:k].sum() / positive_count) if positive_count else None
    return result


def paired_uid_interval(uids, rebuild, reuse, *, samples=2000, seed=17) -> dict:
    """User-cluster bootstrap of the paired REQUEST-macro difference.

    Resample users; preserve all their requests and the request-weighted
    estimand. Input includes only the shared positive-request evaluation set.
    """
    uids, rebuild, reuse = map(np.asarray, (uids, rebuild, reuse))
    if uids.ndim != 1 or not len(uids) or uids.shape != rebuild.shape or reuse.shape != rebuild.shape:
        raise ValueError("Expected nonempty matched request arrays")
    if not np.isfinite(rebuild).all() or not np.isfinite(reuse).all() or samples < 1:
        raise ValueError("Expected finite metrics and positive bootstrap count")
    _, inverse = np.unique(uids, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=rebuild - reuse)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for index in range(samples):
        chosen = rng.integers(0, len(counts), size=len(counts))
        draws[index] = sums[chosen].sum() / counts[chosen].sum()
    return {"difference": float((rebuild - reuse).mean()),
            "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "users": len(counts), "requests": len(uids), "samples": samples, "seed": seed}
