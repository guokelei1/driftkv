"""One FIFO worker with externally scheduled arrivals; no model handler here.

Synthetic service durations exercise queue arithmetic only. Real measurements
use run_queue and an actual handler; a GPU caller must pass a completion
synchronizer so host launch time is not mistaken for completed device work.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import time
from typing import Callable, Iterable

from system_study_2026_10.events import TraceEvent


@dataclass(frozen=True)
class Timing:
    sequence: int
    kind: str
    uid: int | None
    logical_time: int
    arrival_time: float
    start_time: float
    finish_time: float
    queue_seconds: float
    service_seconds: float
    latency_seconds: float
    timing_source: str
    completion_synchronized: bool


def _record(event, start, finish, source, synchronized):
    return Timing(event.sequence, event.kind, event.uid, event.logical_time,
                  event.arrival_time, start, finish, start - event.arrival_time,
                  finish - start, finish - event.arrival_time, source, synchronized)


def _ordered(events):
    previous = (-1.0, -1)
    for event in events:
        current = (event.arrival_time, event.sequence)
        if not isfinite(event.arrival_time) or event.arrival_time < 0 or current < previous:
            raise ValueError("events must retain nonnegative arrival and trace order")
        previous = current
        yield event


def simulate_queue(
    events: Iterable[TraceEvent],
    service_seconds: Callable[[TraceEvent], float],
) -> list[Timing]:
    """Synthetic queue check, with zero service for on-demand behavior markers.

    Marker queue/latency fields describe observation by the serial worker, not
    completion of a behavior write. Writes are charged to the later consumer.
    This function never supplies measured CPU/GPU performance.
    """
    finish = 0.0
    records = []
    for event in _ordered(events):
        duration = 0.0 if event.kind == "behavior" else float(service_seconds(event))
        if not isfinite(duration) or duration < 0:
            raise ValueError("synthetic service durations must be finite and nonnegative")
        start = max(event.arrival_time, finish)
        finish = start + duration
        records.append(_record(event, start, finish, "synthetic_service_durations", False))
    return records


def run_queue(
    events: Iterable[TraceEvent],
    handler: Callable[[TraceEvent], None],
    *,
    synchronize: Callable[[], None] | None = None,
) -> list[Timing]:
    """Measure a synchronous handler under an open arrival schedule.

    Waiting for one job never moves later arrivals. Publication and drain are
    ordinary blocking jobs in this serial worker, so their work delays later
    requests. Initialization/JIT warmup belongs before this function. Behavior
    markers have zero service and invoke neither handler nor synchronizer.
    Their latency is observation delay, not completed model-write latency;
    actual writes are measured inside the later consuming handler.
    """
    if synchronize is not None:
        synchronize()
    epoch = time.perf_counter()
    records = []
    for event in _ordered(events):
        remaining = epoch + event.arrival_time - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        start = time.perf_counter() - epoch
        if event.kind == "behavior":
            finish = start
        else:
            handler(event)
            if synchronize is not None:
                synchronize()
            finish = time.perf_counter() - epoch
        records.append(_record(event, start, finish, "wall_clock_handler",
                               event.kind != "behavior" and synchronize is not None))
    return records
