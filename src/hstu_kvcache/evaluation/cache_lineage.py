"""Version-aware rolling cache primitives with timestamp-group atomicity."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import Iterable, Iterator

import torch

from hstu_kvcache.models.hstu import HSTU
from hstu_kvcache.models.kv_cache import HSTUKVCache
from hstu_kvcache.models.state_transition import append_with_rolling_cap


Event = tuple[int, int, int]  # timestamp, item_idx, behavior
ROLLING_PATHS = (
    "parent_exact_rolling",
    "current_exact_rolling",
    "one_hop_reuse_rolling",
    "recursive_reuse_rolling",
)


def timestamp_groups(events: Iterable[Event]) -> Iterator[tuple[int, tuple[Event, ...]]]:
    """Yield canonical simultaneous-event groups without row-order semantics."""
    ordered = sorted(events, key=lambda value: (int(value[0]), int(value[1]), int(value[2])))
    for timestamp, values in groupby(ordered, key=lambda value: int(value[0])):
        yield timestamp, tuple(values)


@dataclass(frozen=True)
class VersionedCacheState:
    cache: HSTUKVCache
    last_timestamp: int
    producer_versions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.last_timestamp < 0 or len(self.producer_versions) != self.cache.seq_len:
            raise ValueError("versioned cache metadata does not match cache")

    def producer_counts(self) -> dict[str, int]:
        return {version: self.producer_versions.count(version) for version in sorted(set(self.producer_versions))}


@torch.no_grad()
def materialize_state(
    model: HSTU,
    events: Iterable[Event],
    *,
    producer_version: str,
    max_length: int,
) -> VersionedCacheState:
    """Exactly materialize a single-user cutover prefix."""
    ordered = [event for _, group in timestamp_groups(events) for event in group][-max_length:]
    if not ordered:
        raise ValueError("cannot materialize an empty prefix")
    device = next(model.parameters()).device
    timestamps = torch.tensor([[event[0] for event in ordered]], device=device)
    deltas = torch.zeros_like(timestamps, dtype=torch.float32)
    if len(ordered) > 1:
        deltas[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
    items = torch.tensor([[event[1] for event in ordered]], dtype=torch.long, device=device)
    behaviors = torch.tensor([[event[2] for event in ordered]], dtype=torch.long, device=device)
    cache = model.compute_kv(items, behaviors, deltas)
    return VersionedCacheState(
        cache, int(ordered[-1][0]), (producer_version,) * len(ordered)
    )


@torch.no_grad()
def observe_rolling(
    current_model: HSTU,
    state: VersionedCacheState,
    *,
    candidate_id: int,
    query_timestamp: int,
) -> tuple[float, torch.Tensor]:
    if query_timestamp <= state.last_timestamp:
        raise ValueError("rolling query must be strictly after its prefix")
    candidate = torch.tensor([[candidate_id]], dtype=torch.long, device=state.cache.k.device)
    delta = torch.tensor([float(query_timestamp - state.last_timestamp)], device=state.cache.k.device)
    scores, readout = current_model.observe_cc_reuse(state.cache, candidate, delta)
    return float(scores[0, 0]), readout[0, 0].detach().cpu()


@dataclass
class OneHopRollingBundle:
    """The four rolling paths; request-local Parent/Current are computed separately."""

    parent_exact: VersionedCacheState
    current_exact: VersionedCacheState
    one_hop_reuse: VersionedCacheState
    recursive_reuse: VersionedCacheState

    @classmethod
    def at_cutover(
        cls,
        parent_model: HSTU,
        current_model: HSTU,
        events: Iterable[Event],
        *,
        parent_version: str,
        current_version: str,
        max_length: int,
        recursive_state: VersionedCacheState | None = None,
    ) -> "OneHopRollingBundle":
        values = tuple(events)
        parent = materialize_state(
            parent_model, values, producer_version=parent_version, max_length=max_length
        )
        current = materialize_state(
            current_model, values, producer_version=current_version, max_length=max_length
        )
        return cls(parent, current, parent, recursive_state or parent)

    def observe(self, parent_model: HSTU, current_model: HSTU, *, candidate_id: int, query_timestamp: int):
        return {
            "parent_exact_rolling": observe_rolling(parent_model, self.parent_exact, candidate_id=candidate_id, query_timestamp=query_timestamp),
            "current_exact_rolling": observe_rolling(current_model, self.current_exact, candidate_id=candidate_id, query_timestamp=query_timestamp),
            "one_hop_reuse_rolling": observe_rolling(current_model, self.one_hop_reuse, candidate_id=candidate_id, query_timestamp=query_timestamp),
            "recursive_reuse_rolling": observe_rolling(current_model, self.recursive_reuse, candidate_id=candidate_id, query_timestamp=query_timestamp),
        }

    def append_group(
        self, parent_model: HSTU, current_model: HSTU, events: Iterable[Event], *,
        parent_version: str, current_version: str, max_length: int,
    ) -> None:
        values = tuple(events)
        self.parent_exact = append_timestamp_group(parent_model, self.parent_exact, values, producer_version=parent_version, max_length=max_length)
        for name in ("current_exact", "one_hop_reuse", "recursive_reuse"):
            setattr(self, name, append_timestamp_group(current_model, getattr(self, name), values, producer_version=current_version, max_length=max_length))


@torch.no_grad()
def append_timestamp_group(
    model: HSTU,
    state: VersionedCacheState,
    events: Iterable[Event],
    *,
    producer_version: str,
    max_length: int,
) -> VersionedCacheState:
    """Append one already-scored timestamp group in canonical token order."""
    values = tuple(events)
    if not values:
        return state
    timestamps = {int(value[0]) for value in values}
    if len(timestamps) != 1:
        raise ValueError("append_timestamp_group requires one timestamp")
    timestamp = timestamps.pop()
    if timestamp < state.last_timestamp:
        raise ValueError("timestamp regressed")
    canonical = sorted(values, key=lambda value: (int(value[1]), int(value[2])))
    current = state.cache
    producers = list(state.producer_versions)
    previous = state.last_timestamp
    for event_timestamp, item, behavior in canonical:
        delta = float(max(0, min(7 * 86_400, int(event_timestamp) - previous)))
        items = torch.tensor([[int(item)]], dtype=torch.long, device=current.k.device)
        behaviors = torch.tensor([[int(behavior)]], dtype=torch.long, device=current.k.device)
        deltas = torch.tensor([[delta]], dtype=torch.float32, device=current.k.device)
        current = append_with_rolling_cap(model, current, items, behaviors, deltas, max_length)
        if len(producers) >= max_length:
            producers = producers[-(max_length - 1) :] if max_length > 1 else []
        producers.append(producer_version)
        previous = int(event_timestamp)
    return VersionedCacheState(current, timestamp, tuple(producers))
