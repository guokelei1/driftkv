from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from system_study_2026_10.events import TraceEvent
from system_study_2026_10.queueing import run_queue, simulate_queue


def event(kind, arrival, sequence):
    return TraceEvent(kind, int(arrival), arrival, sequence, "v3")


def test_backlog_publication_barrier_and_end_drain():
    events = [event("recommendation", 0, 0), event("publication", 1, 1),
              event("recommendation", 1, 2), event("drain", 3, 3)]
    durations = {0: 2., 1: 3., 2: 1., 3: .5}
    rows = simulate_queue(events, lambda e: durations[e.sequence])
    assert [r.start_time for r in rows] == [0., 2., 5., 6.]
    assert [r.finish_time for r in rows] == [2., 5., 6., 6.5]
    assert rows[2].queue_seconds == 4.
    assert rows[2].latency_seconds == 5.
    assert all(r.timing_source == "synthetic_service_durations" for r in rows)
    assert all(not r.completion_synchronized for r in rows)


def test_idle_worker_does_not_move_future_arrivals():
    rows = simulate_queue([event("recommendation", 0, 0), event("recommendation", 4, 1)],
                          lambda e: 1.)
    assert [r.start_time for r in rows] == [0., 4.]
    assert [r.queue_seconds for r in rows] == [0., 0.]


def test_rejects_out_of_order_trace():
    with pytest.raises(ValueError, match="order"):
        simulate_queue([event("publication", 1, 0), event("recommendation", 0, 1)], lambda e: 0.)


def test_behavior_marker_does_not_add_synthetic_service():
    called = []

    def duration(e):
        called.append(e.kind)
        return 1.

    rows = simulate_queue([event("recommendation", 0, 0), event("behavior", .5, 1),
                           event("recommendation", .5, 2)], duration)
    assert called == ["recommendation", "recommendation"]
    assert rows[1].service_seconds == 0.
    assert rows[2].start_time == 1.


def test_behavior_markers_do_not_dispatch_or_synchronize_gpu(monkeypatch):
    import system_study_2026_10.queueing as queueing
    now = [100.]
    calls = []
    synchronization_calls = []
    monkeypatch.setattr(queueing.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(queueing.time, "sleep", lambda duration: now.__setitem__(0, now[0] + duration))

    def synchronize():
        synchronization_calls.append(now[0])
        now[0] += .5

    rows = run_queue([event("behavior", 0, 0), event("behavior", .1, 1),
                      event("recommendation", .2, 2)],
                     lambda e: calls.append(e.kind), synchronize=synchronize)
    assert calls == ["recommendation"]
    assert len(synchronization_calls) == 2  # Initial completion and recommendation only.
    assert [r.service_seconds for r in rows[:2]] == [0., 0.]
    assert all(not r.completion_synchronized for r in rows[:2])
    assert rows[2].service_seconds == pytest.approx(.5)


def test_real_handler_waits_for_completion_without_synthetic_durations(monkeypatch):
    import system_study_2026_10.queueing as queueing
    now = [100.]
    calls = []
    monkeypatch.setattr(queueing.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(queueing.time, "sleep", lambda duration: now.__setitem__(0, now[0] + duration))

    def handle(e):
        calls.append(e.kind)
        now[0] += .25

    def synchronize():
        now[0] += .5

    rows = run_queue([event("publication", 0, 0), event("recommendation", .2, 1),
                      event("drain", 2, 2)], handle, synchronize=synchronize)
    assert calls == ["publication", "recommendation", "drain"]
    assert [r.start_time for r in rows] == pytest.approx([0., .75, 2.])
    assert rows[1].queue_seconds == pytest.approx(.55)
    assert all(r.service_seconds == pytest.approx(.75) for r in rows)
    assert all(r.timing_source == "wall_clock_handler" and r.completion_synchronized for r in rows)
