"""Small, label-free data contracts for CC qualification.

This module intentionally stops at per-query proposal records.  It does not
build a catalog, train a model, or register results; those decisions belong to
the frozen workload contract and its future orchestration layer.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class QMainCandidate:
    """One candidate drawn from the causal ``Q_main`` proposal."""

    item_id: int
    proposal_rank: int
    weight: float
    log_q_main: float
    causal_cutoff: int

    def __post_init__(self) -> None:
        if self.item_id < 1:
            raise ValueError("candidate item IDs must be positive")
        if self.proposal_rank < 1:
            raise ValueError("proposal_rank is one-based and must be positive")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("Q_main weights must be finite and positive")
        if not math.isfinite(self.log_q_main):
            raise ValueError("log_q_main must be finite")
        if self.causal_cutoff < 0:
            raise ValueError("causal_cutoff must be non-negative")


@dataclass(frozen=True)
class CCProposal:
    """A target-free candidate panel plus its causal boundary."""

    candidates: tuple[QMainCandidate, ...]
    causal_cutoff: int

    def __post_init__(self) -> None:
        if self.causal_cutoff < 0:
            raise ValueError("causal_cutoff must be non-negative")
        if not self.candidates:
            raise ValueError("a proposal panel must contain candidates")
        item_ids = [candidate.item_id for candidate in self.candidates]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Q_main candidates must be sampled without replacement")
        ranks = [candidate.proposal_rank for candidate in self.candidates]
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError("proposal ranks must be a complete one-based ordering")
        if any(candidate.causal_cutoff != self.causal_cutoff for candidate in self.candidates):
            raise ValueError("candidate causal cutoffs must equal the panel cutoff")
        if any(
            not math.isclose(candidate.log_q_main, math.log(candidate.weight), rel_tol=1e-6, abs_tol=1e-6)
            for candidate in self.candidates
        ):
            raise ValueError("log_q_main must be the log of the recorded proposal weight")
        if not math.isclose(sum(candidate.weight for candidate in self.candidates), 1.0, rel_tol=1e-6):
            raise ValueError("Q_main proposal weights must sum to one")

    @property
    def item_ids(self) -> tuple[int, ...]:
        return tuple(candidate.item_id for candidate in self.candidates)

    def negatives_for(self, current_positive_id: int) -> tuple[QMainCandidate, ...]:
        """Return negatives while excluding only the current positive.

        Historical/seen items remain eligible.  The caller may separately
        inject the positive into a quality manifest; this method never puts a
        target or label into a model query token.
        """
        return tuple(candidate for candidate in self.candidates if candidate.item_id != current_positive_id)

    def validate_query_timestamp(self, query_timestamp: int) -> None:
        if query_timestamp < self.causal_cutoff:
            raise ValueError("query timestamp precedes the proposal causal cutoff")


def build_q_main_rank_decay(
    item_ids: Iterable[int],
    causal_cutoff: int,
    *,
    decay: float = 1.0,
    exclude_item_id: int | None = None,
) -> CCProposal:
    """Create a deterministic target-free rank-decay proposal panel.

    ``item_ids`` must already be ordered using only information visible at the
    causal cutoff (for example, catalog/popularity/seen-history state).  This
    helper preserves that order and intentionally has no ``seen`` filter.
    """
    if decay <= 0 or not math.isfinite(decay):
        raise ValueError("decay must be finite and positive")
    if causal_cutoff < 0:
        raise ValueError("causal_cutoff must be non-negative")
    ordered = []
    seen: set[int] = set()
    for raw_item_id in item_ids:
        item_id = int(raw_item_id)
        if item_id < 1:
            raise ValueError("candidate item IDs must be positive")
        if item_id in seen:
            raise ValueError("Q_main candidates must be sampled without replacement")
        seen.add(item_id)
        if exclude_item_id is not None and item_id == exclude_item_id:
            continue
        ordered.append(item_id)
    if not ordered:
        raise ValueError("proposal has no candidates after positive exclusion")
    unnormalised = [float(rank) ** (-decay) for rank in range(1, len(ordered) + 1)]
    normalizer = sum(unnormalised)
    candidates = tuple(
        QMainCandidate(
            item_id=item_id,
            proposal_rank=rank,
            weight=weight / normalizer,
            log_q_main=math.log(weight / normalizer),
            causal_cutoff=causal_cutoff,
        )
        for rank, (item_id, weight) in enumerate(
            zip(ordered, unnormalised, strict=True),
            start=1,
        )
    )
    return CCProposal(candidates=candidates, causal_cutoff=causal_cutoff)


# ---------------------------------------------------------------------------
# P5 seen-aware competing-candidate protocol
# ---------------------------------------------------------------------------

RECENT_WINDOW = 32
OLD_WINDOW = 480
MAX_HISTORY = 512
SEENMIX_COMPETITOR_SLOTS = 99
SEENMIX_MIN_DISCOVERY = 32
SEENMIX_DECAY = 0.5
SEENMIX_QUOTA_GRID_OLD = (24, 16, 12, 8, 4)
SEENMIX_QUOTA_GRID_RECENT = (16, 12, 8, 4)


@dataclass(frozen=True)
class SeenMixQuotas:
    """Fixed competitor quotas. Chosen from coverage, never from model gaps."""

    m_recent: int
    m_old: int
    m_discovery: int

    def __post_init__(self) -> None:
        if self.m_recent < 0 or self.m_old < 0 or self.m_discovery < 0:
            raise ValueError("seenmix quotas must be non-negative")
        if self.m_recent + self.m_old + self.m_discovery != SEENMIX_COMPETITOR_SLOTS:
            raise ValueError("seenmix competitor quotas must sum to 99")
        if self.m_discovery < SEENMIX_MIN_DISCOVERY:
            raise ValueError("discovery quota is below the frozen floor")

    @property
    def competitors(self) -> int:
        return self.m_recent + self.m_old + self.m_discovery


@dataclass(frozen=True)
class SeenPools:
    """Causal recent-seen / old-seen pools, excluding the current positive."""

    recent_items: tuple[int, ...]
    old_items: tuple[int, ...]
    recent_counts: tuple[int, ...]
    old_counts: tuple[int, ...]
    target_stratum: str

    def __post_init__(self) -> None:
        if self.target_stratum not in {"recent_seen", "old_only", "unseen"}:
            raise ValueError("unknown target stratum")
        if len(self.recent_items) != len(self.recent_counts):
            raise ValueError("recent pool counts must align with items")
        if len(self.old_items) != len(self.old_counts):
            raise ValueError("old pool counts must align with items")
        overlap = set(self.recent_items) & set(self.old_items)
        if overlap:
            raise ValueError("recent-seen and old-seen pools must be disjoint")


@dataclass(frozen=True)
class SeenMixPanel:
    """One frozen seenmix panel. Incomplete panels are dropped, not backfilled."""

    item_ids: tuple[int, ...]
    roles: tuple[str, ...]
    complete: bool
    missing: dict[str, int]
    target_stratum: str
    target_injected: bool
    pre_injection_hit: bool | None

    def __post_init__(self) -> None:
        if len(self.item_ids) != len(self.roles):
            raise ValueError("item_ids and roles must have the same length")
        if len(self.item_ids) != len(set(self.item_ids)):
            raise ValueError("seenmix panels must not contain duplicates")
        if any(item <= 0 for item in self.item_ids):
            raise ValueError("candidate item IDs must be positive")


def split_seen_pools(
    history: Sequence[tuple[int, int, int]],
    target_item: int | None,
    *,
    recent_window: int = RECENT_WINDOW,
    max_history: int = MAX_HISTORY,
) -> SeenPools:
    """Split a causal prefix into recent-seen and old-only-seen item pools.

    ``history`` must already be strictly before the query.  The current
    positive is excluded from both competitor pools but still determines
    ``target_stratum``.  Old-seen items that also appear in Recent-32 stay
    in the recent pool only.
    """
    if recent_window < 1 or max_history < recent_window:
        raise ValueError("history windows are inconsistent")
    if target_item is not None and int(target_item) < 1:
        raise ValueError("target item IDs must be positive")
    prefix = list(history)[-max_history:]
    recent_events = prefix[-recent_window:]
    old_events = prefix[:-recent_window] if len(prefix) > recent_window else []

    def _counts(events: Sequence[tuple[int, int, int]]) -> dict[int, list[int]]:
        observed: dict[int, list[int]] = {}
        for position, (item, _timestamp, _behavior) in enumerate(events):
            item = int(item)
            if item < 1:
                raise ValueError("history item IDs must be positive")
            bucket = observed.setdefault(item, [0, position])
            bucket[0] += 1
            bucket[1] = position
        return observed

    recent_raw = _counts(recent_events)
    old_raw = _counts(old_events)
    target = None if target_item is None else int(target_item)
    if target is not None and target in recent_raw:
        stratum = "recent_seen"
    elif target is not None and target in old_raw:
        stratum = "old_only"
    else:
        stratum = "unseen"

    def _ordered(raw: dict[int, list[int]], exclude: set[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        items = [item for item in raw if item not in exclude]
        items.sort(key=lambda item: (-raw[item][0], -raw[item][1], item))
        return tuple(items), tuple(raw[item][0] for item in items)

    exclude = set() if target is None else {target}
    recent_items, recent_counts = _ordered(recent_raw, exclude)
    old_only = {item: value for item, value in old_raw.items() if item not in recent_raw}
    old_items, old_counts = _ordered(old_only, exclude)
    return SeenPools(
        recent_items=recent_items,
        old_items=old_items,
        recent_counts=recent_counts,
        old_counts=old_counts,
        target_stratum=stratum,
    )


def _weighted_sample_without_replacement(
    items: Sequence[int],
    weights: Sequence[float],
    count: int,
    rng: random.Random,
) -> list[int]:
    if count < 0:
        raise ValueError("sample count must be non-negative")
    if count == 0 or not items:
        return []
    if len(items) != len(weights):
        raise ValueError("items and weights must align")
    if count >= len(items):
        return list(items)
    keyed = []
    for item, weight in zip(items, weights, strict=True):
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("sample weights must be finite and positive")
        keyed.append((rng.expovariate(1.0) / weight, int(item)))
    keyed.sort()
    return [item for _key, item in keyed[:count]]


def _rank_decay_sample(
    items: Sequence[int],
    count: int,
    rng: random.Random,
    *,
    decay: float = SEENMIX_DECAY,
) -> list[int]:
    if count == 0 or not items:
        return []
    weights = [float(rank) ** (-decay) for rank in range(1, len(items) + 1)]
    return _weighted_sample_without_replacement(items, weights, count, rng)


def seenmix_quota_grid() -> tuple[SeenMixQuotas, ...]:
    """Largest-old-then-recent grid used to freeze quotas from coverage."""
    options = []
    for m_old in SEENMIX_QUOTA_GRID_OLD:
        for m_recent in SEENMIX_QUOTA_GRID_RECENT:
            m_discovery = SEENMIX_COMPETITOR_SLOTS - m_old - m_recent
            if m_discovery < SEENMIX_MIN_DISCOVERY:
                continue
            options.append(SeenMixQuotas(m_recent=m_recent, m_old=m_old, m_discovery=m_discovery))
    return tuple(options)


def build_seenmix_panel(
    pools: SeenPools,
    discovery_ordered: Sequence[int],
    quotas: SeenMixQuotas,
    *,
    rng: random.Random,
    target_item: int | None = None,
    inject_target: bool = False,
) -> SeenMixPanel:
    """Assemble one causal seenmix panel without cross-stratum backfill.

    Discovery is the leftover Q_main proposal after removing the target and
    every item already in the recent or old seen pools.  A short pool makes
    the panel incomplete; callers must drop it rather than steal from
    another stratum.
    """
    if inject_target and target_item is None:
        raise ValueError("quality seenmix requires a target to inject")
    target = None if target_item is None else int(target_item)
    seen_blocked = set(pools.recent_items) | set(pools.old_items)
    if target is not None:
        seen_blocked.add(target)
    discovery = []
    discovery_seen: set[int] = set()
    for raw_item in discovery_ordered:
        item = int(raw_item)
        if item < 1:
            raise ValueError("discovery item IDs must be positive")
        if item in seen_blocked or item in discovery_seen:
            continue
        discovery.append(item)
        discovery_seen.add(item)

    missing = {
        "recent": max(0, quotas.m_recent - len(pools.recent_items)),
        "old": max(0, quotas.m_old - len(pools.old_items)),
        "discovery": max(0, quotas.m_discovery - len(discovery)),
    }
    complete = all(value == 0 for value in missing.values())
    recent = _weighted_sample_without_replacement(
        pools.recent_items, [float(count) for count in pools.recent_counts], quotas.m_recent, rng
    ) if quotas.m_recent else []
    old = _weighted_sample_without_replacement(
        pools.old_items, [float(count) for count in pools.old_counts], quotas.m_old, rng
    ) if quotas.m_old else []
    chosen_discovery = _rank_decay_sample(discovery, quotas.m_discovery, rng)
    roles: list[str] = []
    items: list[int] = []
    pre_injection_hit = None
    if inject_target:
        assert target is not None
        items.append(target)
        roles.append("target")
        pre_injection_hit = target in set(recent) | set(old) | set(chosen_discovery)
    items.extend(recent)
    roles.extend(["recent_seen"] * len(recent))
    items.extend(old)
    roles.extend(["old_seen"] * len(old))
    items.extend(chosen_discovery)
    roles.extend(["discovery"] * len(chosen_discovery))
    return SeenMixPanel(
        item_ids=tuple(items),
        roles=tuple(roles),
        complete=complete and len(items) == (1 if inject_target else 0) + quotas.competitors,
        missing=missing,
        target_stratum=pools.target_stratum,
        target_injected=inject_target,
        pre_injection_hit=pre_injection_hit,
    )


def recency_bin(delta_seconds: float | None) -> str:
    if delta_seconds is None:
        return "unseen"
    if delta_seconds < 3600:
        return "lt_1h"
    if delta_seconds < 86400:
        return "lt_1d"
    if delta_seconds < 604800:
        return "lt_7d"
    return "ge_7d"


def count_bin(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count < 4:
        return "2_3"
    if count < 8:
        return "4_7"
    return "8p"


def popularity_bin(global_count: int) -> str:
    if global_count <= 0:
        return "0"
    magnitude = int(math.log10(global_count))
    if magnitude <= 0:
        return "1e0"
    if magnitude == 1:
        return "1e1"
    if magnitude == 2:
        return "1e2"
    if magnitude == 3:
        return "1e3"
    return "1e4p"


def artist_familiarity(artist_id: int, recent_artists: set[int], old_artists: set[int]) -> str:
    if artist_id < 0:
        return "unknown_artist"
    if artist_id in recent_artists:
        return "recent_artist"
    if artist_id in old_artists:
        return "old_artist"
    return "unseen_artist"


def match_key(
    *,
    stratum: str,
    item_count: int,
    recency_seconds: float | None,
    familiarity: str,
    global_count: int,
) -> tuple[str, str, str, str, str]:
    return (
        stratum,
        count_bin(item_count),
        recency_bin(recency_seconds),
        familiarity,
        popularity_bin(global_count),
    )


def build_history_matched_panel(
    target_item: int,
    target_key: tuple[str, str, str, str, str],
    competitor_items: Sequence[int],
    competitor_keys: Sequence[tuple[str, str, str, str, str]],
    *,
    competitor_slots: int,
    rng: random.Random,
) -> SeenMixPanel:
    """Match competitors to the target on frozen discrete bins.

    This is a mechanism diagnostic, not a serving workload.  Only same-key
    competitors are eligible.  Shortfalls are not backfilled from other bins.
    """
    if competitor_slots < 1:
        raise ValueError("matched diagnostic needs a positive competitor quota")
    if len(competitor_items) != len(competitor_keys):
        raise ValueError("matched competitors and keys must align")
    target = int(target_item)
    eligible = [
        int(item)
        for item, key in zip(competitor_items, competitor_keys, strict=True)
        if key == target_key and int(item) != target
    ]
    unique: list[int] = []
    seen: set[int] = set()
    for item in eligible:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    chosen = list(unique)
    rng.shuffle(chosen)
    chosen = chosen[:competitor_slots]
    complete = len(chosen) == competitor_slots
    return SeenMixPanel(
        item_ids=(target, *chosen),
        roles=("target", *(["matched"] * len(chosen))),
        complete=complete,
        missing={"matched": max(0, competitor_slots - len(chosen))},
        target_stratum=target_key[0],
        target_injected=True,
        pre_injection_hit=target in set(unique),
    )
