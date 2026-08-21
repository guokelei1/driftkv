"""Label-disciplined primitives for the frozen P7 Yambda workloads.

These helpers construct one causal request at a time.  They do not scan a
dataset, generate experiment manifests, fit a base model, or train HSTU.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Event = tuple[int, int, int]  # item_id, timestamp, behavior

FEEDBACK_HISTORY_STRATA_V2 = (
    "recent_seen",
    "old_seen",
    "seen_only_before_512",
    "never_seen",
)


def feedback_history_stratum_v2(
    listens: Sequence[Event],
    candidate_item_id: int,
    query_timestamp: int,
    *,
    recent_tokens: int = 32,
    max_history: int = 512,
) -> str:
    """Classify a candidate using the complete causal history before capping."""
    if candidate_item_id < 1 or query_timestamp < 0:
        raise ValueError("feedback query item and timestamp must be valid")
    if recent_tokens < 1 or max_history < recent_tokens:
        raise ValueError("history windows are inconsistent")
    causal = tuple(event for event in listens if event[1] < query_timestamp)
    if not causal:
        raise ValueError("feedback stratum requires an earlier listen")
    if any(causal[index][1] > causal[index + 1][1] for index in range(len(causal) - 1)):
        raise ValueError("listens must be chronological")
    last_position = next(
        (index for index in range(len(causal) - 1, -1, -1) if causal[index][0] == candidate_item_id),
        None,
    )
    if last_position is None:
        return "never_seen"
    if last_position >= len(causal) - recent_tokens:
        return "recent_seen"
    if last_position >= len(causal) - max_history:
        return "old_seen"
    return "seen_only_before_512"


@dataclass(frozen=True)
class FamiliarCandidate:
    item_id: int
    stratum: str
    item_count: int
    artist_count: int
    item_recency_seconds: int
    artist_recency_seconds: int
    global_popularity: int
    proposal_rank: int
    artist_missing: bool = False

    def __post_init__(self) -> None:
        if self.item_id < 1 or self.stratum not in {"recent_seen", "old_seen"}:
            raise ValueError("invalid familiar candidate")
        if min(self.item_count, self.artist_count, self.global_popularity) < 0:
            raise ValueError("count features must be non-negative")
        if min(self.item_recency_seconds, self.artist_recency_seconds) < 0:
            raise ValueError("recency features must be non-negative")
        if self.proposal_rank < 1:
            raise ValueError("proposal_rank must be one-based")

    def base_features(self) -> tuple[float, ...]:
        """Return the registered R-workload feature order."""
        return (
            math.log1p(self.item_count),
            math.log1p(self.artist_count),
            math.log1p(self.item_recency_seconds),
            0.0 if self.artist_missing else math.log1p(self.artist_recency_seconds),
            math.log1p(self.global_popularity),
            math.log1p(self.proposal_rank),
            float(self.artist_missing),
        )


@dataclass(frozen=True)
class ReturnToFamiliarRequest:
    query_timestamp: int
    inactivity_gap_seconds: int
    candidates: tuple[FamiliarCandidate, ...]

    @property
    def item_ids(self) -> tuple[int, ...]:
        return tuple(candidate.item_id for candidate in self.candidates)

    def quality_target_index(self, target_item_id: int) -> int | None:
        """Locate an organic familiar-return target without injecting it."""
        try:
            return self.item_ids.index(int(target_item_id))
        except ValueError:
            return None


def build_return_to_familiar_request(
    history: Sequence[Event],
    query_timestamp: int,
    artist_by_item: Mapping[int, int],
    global_popularity: Mapping[int, int],
    *,
    inactivity_gap_seconds: int = 259_200,
    recent_tokens: int = 32,
    max_history: int = 512,
) -> ReturnToFamiliarRequest:
    """Build the target-free candidate universe for one eligible R request."""
    if query_timestamp < 0 or inactivity_gap_seconds < 1:
        raise ValueError("query time and inactivity gap must be valid")
    if recent_tokens < 1 or max_history < recent_tokens:
        raise ValueError("history windows are inconsistent")
    prefix = list(history)[-max_history:]
    if not prefix:
        raise ValueError("return requests require a non-empty causal prefix")
    if any(item < 1 or timestamp < 0 for item, timestamp, _ in prefix):
        raise ValueError("history contains an invalid item or timestamp")
    if any(prefix[index][1] > prefix[index + 1][1] for index in range(len(prefix) - 1)):
        raise ValueError("history must be chronological")
    if any(timestamp >= query_timestamp for _, timestamp, _ in prefix):
        raise ValueError("history must be strictly before the query")
    gap = query_timestamp - prefix[-1][1]
    if gap < inactivity_gap_seconds:
        raise ValueError("query does not meet the frozen inactivity gap")

    item_counts = Counter(item for item, _, _ in prefix)
    item_last = {item: timestamp for item, timestamp, _ in prefix}
    artists = {item: int(artist_by_item.get(item, -1)) for item in item_counts}
    artist_counts = Counter(artists[item] for item, _, _ in prefix if artists[item] >= 0)
    artist_last: dict[int, int] = {}
    for item, timestamp, _ in prefix:
        if artists[item] >= 0:
            artist_last[artists[item]] = timestamp
    recent_items = {item for item, _, _ in prefix[-recent_tokens:]}
    ordered = sorted(
        item_counts,
        key=lambda item: (-item_counts[item], query_timestamp - item_last[item], item),
    )
    if len(ordered) < 2:
        raise ValueError("return requests require at least two familiar candidates")
    candidates = tuple(
        FamiliarCandidate(
            item_id=item,
            stratum="recent_seen" if item in recent_items else "old_seen",
            item_count=item_counts[item],
            artist_count=artist_counts[artists[item]] if artists[item] >= 0 else 0,
            item_recency_seconds=query_timestamp - item_last[item],
            artist_recency_seconds=(
                query_timestamp - artist_last[artists[item]] if artists[item] >= 0 else 0
            ),
            global_popularity=int(global_popularity.get(item, 0)),
            proposal_rank=rank,
            artist_missing=artists[item] < 0,
        )
        for rank, item in enumerate(ordered, start=1)
    )
    return ReturnToFamiliarRequest(query_timestamp, gap, candidates)


@dataclass(frozen=True)
class ExplicitFeedbackQuery:
    candidate_item_id: int
    query_timestamp: int
    label: int | None
    causal_prefix: tuple[Event, ...]
    candidate_history_position: str
    coincident_target_listens_excluded: int


def build_explicit_feedback_query(
    listens: Sequence[Event],
    candidate_item_id: int,
    query_timestamp: int,
    *,
    label: int | None,
    recent_tokens: int = 32,
    max_history: int = 512,
) -> ExplicitFeedbackQuery:
    """Construct one F request while excluding coincident/future listens."""
    if candidate_item_id < 1 or query_timestamp < 0:
        raise ValueError("feedback query item and timestamp must be valid")
    if label not in {None, 0, 1}:
        raise ValueError("feedback labels are binary or absent for fidelity")
    if recent_tokens < 1 or max_history < recent_tokens:
        raise ValueError("history windows are inconsistent")
    causal = tuple(event for event in listens if event[1] < query_timestamp)
    if not causal:
        raise ValueError("feedback query requires an earlier listen")
    capped = causal[-max_history:]
    position = feedback_history_stratum_v2(
        causal,
        candidate_item_id,
        query_timestamp,
        recent_tokens=recent_tokens,
        max_history=max_history,
    )
    coincident = sum(
        item == candidate_item_id and timestamp == query_timestamp
        for item, timestamp, _ in listens
    )
    return ExplicitFeedbackQuery(
        candidate_item_id=candidate_item_id,
        query_timestamp=query_timestamp,
        label=label,
        causal_prefix=capped,
        candidate_history_position=position,
        coincident_target_listens_excluded=coincident,
    )
