"""Causal request-time features and schemas for the Small foundation chain."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass


DAY_SECONDS = 86_400
WINDOWS = (
    ("foundation", 0, 217 * DAY_SECONDS),
    ("update1", 217 * DAY_SECONDS, 224 * DAY_SECONDS),
    ("update2", 224 * DAY_SECONDS, 231 * DAY_SECONDS),
    ("evaluation2", 231 * DAY_SECONDS, 238 * DAY_SECONDS),
)
SNAPSHOT_DAYS = (217, 224, 231)
BASE_FEATURE_NAMES = (
    "log1p_item_count",
    "log1p_artist_count",
    "log1p_item_recency_seconds",
    "log1p_artist_recency_seconds",
    "log1p_global_popularity_asof",
    "item_history_missing",
    "artist_history_or_mapping_missing",
)


def time_block(timestamp: int) -> str | None:
    for name, start, end in WINDOWS:
        if start <= int(timestamp) < end:
            return name
    return None


def foundation_request_id(uid: int, timestamp: int, raw_item_id: int) -> str:
    value = f"yambda500m-small-f-v1:{int(uid)}:{int(timestamp)}:{int(raw_item_id)}"
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class HistoryEvent:
    timestamp: int
    raw_item_id: int
    artist_id: int
    item_idx: int


class CausalFeatureState:
    """Bounded user history plus strictly-prior global popularity."""

    def __init__(self, max_history: int = 512) -> None:
        if max_history < 1:
            raise ValueError("max_history must be positive")
        self.max_history = max_history
        self.histories: dict[int, deque[HistoryEvent]] = defaultdict(deque)
        self.item_times: dict[int, dict[int, deque[int]]] = defaultdict(lambda: defaultdict(deque))
        self.artist_times: dict[int, dict[int, deque[int]]] = defaultdict(lambda: defaultdict(deque))
        self.global_popularity: Counter[int] = Counter()

    def append_listen(
        self,
        *,
        uid: int,
        timestamp: int,
        raw_item_id: int,
        item_idx: int,
        artist_id: int,
    ) -> None:
        history = self.histories[int(uid)]
        if history and int(timestamp) < history[-1].timestamp:
            raise ValueError("listen timestamp regressed")
        if len(history) >= self.max_history:
            removed = history.popleft()
            item_queue = self.item_times[int(uid)][removed.raw_item_id]
            if item_queue.popleft() != removed.timestamp:
                raise AssertionError("item recency index diverged")
            if not item_queue:
                del self.item_times[int(uid)][removed.raw_item_id]
            if removed.artist_id >= 0:
                artist_queue = self.artist_times[int(uid)][removed.artist_id]
                if artist_queue.popleft() != removed.timestamp:
                    raise AssertionError("artist recency index diverged")
                if not artist_queue:
                    del self.artist_times[int(uid)][removed.artist_id]
        event = HistoryEvent(int(timestamp), int(raw_item_id), int(artist_id), int(item_idx))
        history.append(event)
        self.item_times[int(uid)][int(raw_item_id)].append(int(timestamp))
        if artist_id >= 0:
            self.artist_times[int(uid)][int(artist_id)].append(int(timestamp))
        self.global_popularity[int(raw_item_id)] += 1

    def request_features(
        self, *, uid: int, timestamp: int, raw_item_id: int, artist_id: int
    ) -> tuple[float, ...]:
        history = self.histories.get(int(uid), ())
        if history and history[-1].timestamp >= int(timestamp):
            raise ValueError("request features require a strictly prior history state")
        item_queue = self.item_times[int(uid)].get(int(raw_item_id), ())
        artist_queue = (
            self.artist_times[int(uid)].get(int(artist_id), ()) if int(artist_id) >= 0 else ()
        )
        item_missing = not item_queue
        artist_missing = int(artist_id) < 0 or not artist_queue
        return (
            math.log1p(len(item_queue)),
            math.log1p(len(artist_queue)) if not artist_missing else 0.0,
            math.log1p(int(timestamp) - item_queue[-1]) if not item_missing else 0.0,
            math.log1p(int(timestamp) - artist_queue[-1]) if not artist_missing else 0.0,
            math.log1p(self.global_popularity[int(raw_item_id)]),
            float(item_missing),
            float(artist_missing),
        )

    def history_summary(self, uid: int) -> dict[str, int | float | None]:
        history = self.histories.get(int(uid), ())
        oov = sum(event.item_idx == 0 for event in history)
        return {
            "history_length": len(history),
            "last_timestamp": int(history[-1].timestamp) if history else None,
            "history_oov_tokens": oov,
            "history_oov_fraction": float(oov / len(history)) if history else None,
        }
