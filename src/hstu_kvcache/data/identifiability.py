"""Leak-resistant simple-feature and continuous-matching primitives.

These helpers never see an assembled panel slot.  Proposal rank must be
supplied from a causal Q_main map; missing items use ``missing_rank``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

IDENTIFIABILITY_FEATURES = (
    "seen_flag",
    "recent_flag",
    "old_flag",
    "log1p_item_count",
    "log1p_artist_count",
    "last_seen_recency_days",
    "artist_last_seen_recency_days",
    "log1p_global_popularity",
    "log_proposal_rank",
)
MATCH_FEATURES = (
    "log1p_item_count",
    "log1p_artist_count",
    "last_seen_recency_days",
    "artist_last_seen_recency_days",
    "log1p_global_popularity",
    "log_proposal_rank",
)
UNSEEN_RECENCY_DAYS = 365.0
SECONDS_PER_DAY = 86400.0
MISSING_PROPOSAL_RANK = 1001
MATCH_COMPETITORS = 8
MATCH_CALIPERS = (0.25, 0.5, 1.0, 1.5, 2.0)
MAX_ABS_SMD = 0.25
MAX_MATCHED_CAUC = 0.60
MIN_HOLDOUT_COMPLETE = 400
MIN_STRATUM_COMPLETE = 80


@dataclass(frozen=True)
class ItemHistoryStats:
    count: int
    last_ts: int | None
    artist_count: int
    artist_last_ts: int | None
    in_recent: bool
    in_old: bool


def history_item_stats(
    history: Sequence[tuple[int, int, int]],
    artist_by_item: Sequence[int],
    *,
    recent_window: int = 32,
    max_history: int = 512,
) -> dict[int, ItemHistoryStats]:
    prefix = list(history)[-max_history:]
    recent = {int(item) for item, _ts, _beh in prefix[-recent_window:]}
    old_only = {int(item) for item, _ts, _beh in prefix[:-recent_window]} - recent
    counts: dict[int, list[int]] = {}
    artist_counts: dict[int, list[int]] = {}
    for item, timestamp, _behavior in prefix:
        item = int(item)
        timestamp = int(timestamp)
        bucket = counts.setdefault(item, [0, timestamp])
        bucket[0] += 1
        bucket[1] = timestamp
        if item < len(artist_by_item):
            artist_id = int(artist_by_item[item])
            if artist_id >= 0:
                artist_bucket = artist_counts.setdefault(artist_id, [0, timestamp])
                artist_bucket[0] += 1
                artist_bucket[1] = timestamp
    stats = {}
    for item, (count, last_ts) in counts.items():
        artist_id = int(artist_by_item[item]) if item < len(artist_by_item) else -1
        artist = artist_counts.get(artist_id, [0, None]) if artist_id >= 0 else [0, None]
        stats[item] = ItemHistoryStats(
            count=count,
            last_ts=last_ts,
            artist_count=int(artist[0]),
            artist_last_ts=None if artist[1] is None else int(artist[1]),
            in_recent=item in recent,
            in_old=item in old_only,
        )
    return stats


def _recency_days(last_ts: int | None, query_ts: int) -> float:
    if last_ts is None:
        return UNSEEN_RECENCY_DAYS
    return max(0.0, float(query_ts - last_ts) / SECONDS_PER_DAY)


def identifiability_vector(
    item: int,
    *,
    query_ts: int,
    stats: dict[int, ItemHistoryStats],
    popularity: Sequence[int],
    qmain_ranks: dict[int, int],
    artist_by_item: Sequence[int],
    missing_rank: int = MISSING_PROPOSAL_RANK,
) -> tuple[float, ...]:
    """Return the frozen simple-feature vector for one candidate.

    The vector is a function of the item and the causal request state.  It
    does not depend on the item's position in any assembled panel.
    """
    item = int(item)
    current = stats.get(item)
    if current is None:
        artist_id = int(artist_by_item[item]) if item < len(artist_by_item) else -1
        artist_hits = [
            value
            for other, value in stats.items()
            if other < len(artist_by_item) and int(artist_by_item[other]) == artist_id and artist_id >= 0
        ]
        artist_count = sum(hit.count for hit in artist_hits)
        artist_last = max((hit.artist_last_ts or 0 for hit in artist_hits), default=0) or None
        current = ItemHistoryStats(0, None, artist_count, artist_last, False, False)
    rank = int(qmain_ranks.get(item, missing_rank))
    if rank < 1:
        raise ValueError("proposal rank must be a positive causal Q_main rank")
    popularity_count = int(popularity[item]) if item < len(popularity) else 0
    return (
        float(current.in_recent or current.in_old),
        float(current.in_recent),
        float(current.in_old),
        math.log1p(float(current.count)),
        math.log1p(float(current.artist_count)),
        _recency_days(current.last_ts, query_ts),
        _recency_days(current.artist_last_ts, query_ts),
        math.log1p(float(max(popularity_count, 0))),
        math.log(float(rank)),
    )


def feature_index(name: str) -> int:
    return IDENTIFIABILITY_FEATURES.index(name)


def match_feature_indices() -> tuple[int, ...]:
    return tuple(IDENTIFIABILITY_FEATURES.index(name) for name in MATCH_FEATURES)


def uid_fold(uid: int, *, folds: int = 5, seed: int = 1) -> int:
    if folds < 2:
        raise ValueError("need at least two folds")
    material = f"{seed}:uid-fold:{int(uid)}".encode()
    digest = __import__("hashlib").sha256(material).digest()
    return int.from_bytes(digest[:8], "little") % folds


def grouped_folds(uids: Sequence[int], *, folds: int = 5, seed: int = 1) -> list[tuple[list[int], list[int]]]:
    """Return (train_indices, test_indices) with no uid on both sides."""
    assignments = [uid_fold(uid, folds=folds, seed=seed) for uid in uids]
    splits = []
    for fold in range(folds):
        train = [index for index, assigned in enumerate(assignments) if assigned != fold]
        test = [index for index, assigned in enumerate(assignments) if assigned == fold]
        if not test or not train:
            continue
        train_uids = {int(uids[index]) for index in train}
        test_uids = {int(uids[index]) for index in test}
        if train_uids & test_uids:
            raise RuntimeError("grouped fold leaked a uid into both sides")
        splits.append((train, test))
    if len(splits) < 2:
        raise RuntimeError("grouped fold construction failed")
    return splits


def request_conditional_metrics(scores: Sequence[Sequence[float]], *, k: int = 10) -> dict[str, float]:
    """Within-request metrics. Column 0 is the target."""
    if not scores:
        raise ValueError("scores must not be empty")
    cauc = []
    ndcg = []
    hr = []
    mrr = []
    percentile = []
    for row in scores:
        if len(row) < 2:
            raise ValueError("each request needs a target and at least one competitor")
        target = float(row[0])
        negatives = [float(value) for value in row[1:]]
        wins = sum(1.0 for value in negatives if target > value)
        ties = sum(1.0 for value in negatives if target == value)
        rank = 1 + sum(1 for value in negatives if value >= target)
        cauc.append((wins + 0.5 * ties) / len(negatives))
        ndcg.append((1.0 / math.log2(rank + 1.0)) if rank <= k else 0.0)
        hr.append(1.0 if rank <= k else 0.0)
        mrr.append(1.0 / rank)
        percentile.append((rank - 1) / len(negatives))
    return {
        "cauc": sum(cauc) / len(cauc),
        "ndcg@10": sum(ndcg) / len(ndcg),
        "hr@10": sum(hr) / len(hr),
        "mrr": sum(mrr) / len(mrr),
        "target_rank_percentile": sum(percentile) / len(percentile),
        "requests": float(len(scores)),
    }


def nearest_within_caliper(
    target: Sequence[float],
    candidates: Sequence[Sequence[float]],
    *,
    k: int,
    caliper: float,
) -> list[int]:
    if k < 1:
        raise ValueError("matched diagnostic needs a positive competitor quota")
    if caliper <= 0:
        raise ValueError("caliper must be positive")
    distances = []
    for index, candidate in enumerate(candidates):
        if len(candidate) != len(target):
            raise ValueError("target and candidate feature widths differ")
        dist = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(target, candidate, strict=True)))
        distances.append((dist, index))
    distances.sort()
    chosen = [index for dist, index in distances if dist <= caliper]
    return chosen[:k]


def standardized_mean_difference(target: Sequence[float], competitors: Sequence[float]) -> float:
    if not target or not competitors:
        return float("nan")
    t_mean = sum(target) / len(target)
    c_mean = sum(competitors) / len(competitors)
    values = list(target) + list(competitors)
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    denom = math.sqrt(var)
    if denom < 1e-12:
        return 0.0 if abs(t_mean - c_mean) < 1e-12 else float("inf")
    return (t_mean - c_mean) / denom


def ks_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return float("nan")
    points = sorted(set(left) | set(right))
    n_left = len(left)
    n_right = len(right)
    best = 0.0
    for point in points:
        cdf_left = sum(1 for value in left if value <= point) / n_left
        cdf_right = sum(1 for value in right if value <= point) / n_right
        best = max(best, abs(cdf_left - cdf_right))
    return best


def select_caliper(
    summaries: Sequence[dict],
    *,
    min_complete: int = MIN_HOLDOUT_COMPLETE,
    min_stratum: int = MIN_STRATUM_COMPLETE,
    max_abs_smd: float = MAX_ABS_SMD,
) -> dict:
    """Choose a caliper from design-set balance only. Never inspect AUC."""
    feasible = [
        row
        for row in summaries
        if row["complete"] >= min_complete
        and min(row["stratum_complete"].values()) >= min_stratum
        and row["max_abs_smd"] <= max_abs_smd
    ]
    if feasible:
        chosen = min(feasible, key=lambda row: (row["mean_abs_smd"], row["caliper"]))
        return {**chosen, "status": "balanced", "selection": "min_mean_abs_smd_among_feasible"}
    if not summaries:
        raise ValueError("no caliper summaries")
    fallback = min(summaries, key=lambda row: (row["max_abs_smd"], -row["complete"], row["caliper"]))
    return {**fallback, "status": "coverage_or_balance_failed", "selection": "least_imbalanced_fallback"}
