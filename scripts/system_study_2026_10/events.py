"""CPU event ordering for the existing on-demand history service policy.

Behavior entries announce arrivals; they do not request immediate model writes.
The model handler will advance strict-past history before recommendations and
publications, and advance every user at the final drain. This preserves the
existing replay policy instead of silently changing its append-band geometry.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence


@dataclass(frozen=True)
class TraceEvent:
    kind: str
    logical_time: int
    arrival_time: float
    sequence: int
    version: str
    uid: int | None = None
    requests: tuple[dict, ...] = ()
    behavior_indices: tuple[int, ...] = ()


def build_trace(
    histories: Mapping[int, Sequence],
    requests: Mapping[int, Sequence[Mapping]],
    releases: Sequence[tuple[str, int]],
    end_time: int,
    *,
    arrival_scale: float = 1.0,
) -> list[TraceEvent]:
    """Merge existing histories and request rows into a causal global trace.

    ``arrival_scale`` multiplies elapsed logical seconds, not model timestamps.
    At a tied timestamp publication precedes recommendation, then behavior.
    Candidate order is the input order within each round. Feedback rows without
    a round_id form one recommendation per user/version/timestamp. The final
    drain is exclusive of behavior exactly at ``end_time``, like advance().
    """
    if not isfinite(arrival_scale) or arrival_scale <= 0:
        raise ValueError("arrival_scale must be finite and positive")
    releases = [(str(version), int(at)) for version, at in releases]
    times = [at for _, at in releases]
    if not times or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("releases must have strictly increasing timestamps")
    if end_time <= times[-1]:
        raise ValueError("end_time must follow the final release")
    start = times[0]

    def version_at(timestamp):
        return releases[bisect_right(times, timestamp) - 1][0]

    # Insertion order breaks ties between independent users and rounds.
    entries = [dict(kind="publication", logical_time=at, version=version)
               for version, at in releases]
    groups = {}
    for uid, rows in requests.items():
        uid = int(uid)
        if uid not in histories:
            raise ValueError(f"request user {uid} has no history")
        for row in rows:
            timestamp = int(row["query_timestamp"])
            if not start <= timestamp < end_time:
                continue
            if int(row["uid"]) != uid:
                raise ValueError("request user differs from its containing group")
            version = str(row["version"])
            if version != version_at(timestamp):
                raise ValueError("request version does not match its logical timestamp")
            key = (uid, version, timestamp, row.get("round_id"))
            groups.setdefault(key, []).append(dict(row))
    for (uid, version, timestamp, _), rows in groups.items():
        entries.append(dict(kind="recommendation", logical_time=timestamp,
                            version=version, uid=uid, requests=tuple(rows)))

    for uid, columns in histories.items():
        timestamps = columns[0]
        if any(a > b for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("behavior timestamps must be in original causal order")
        same_time = {}
        for index, timestamp in enumerate(timestamps):
            timestamp = int(timestamp)
            if start <= timestamp < end_time:
                same_time.setdefault(timestamp, []).append(index)
        for timestamp, indices in same_time.items():
            entries.append(dict(kind="behavior", logical_time=timestamp,
                                version=version_at(timestamp), uid=int(uid),
                                behavior_indices=tuple(indices)))
    entries.append(dict(kind="drain", logical_time=int(end_time),
                        version=releases[-1][0]))
    priority = {"publication": 0, "recommendation": 1, "behavior": 2, "drain": 3}
    entries.sort(key=lambda e: (e["logical_time"], priority[e["kind"]]))
    return [TraceEvent(**entry, sequence=sequence,
                       arrival_time=(entry["logical_time"] - start) * arrival_scale)
            for sequence, entry in enumerate(entries)]
