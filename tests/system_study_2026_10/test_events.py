from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from system_study_2026_10.events import build_trace


def row(uid, timestamp, item, *, version="v3", round_id=None):
    result = dict(uid=uid, query_timestamp=timestamp, item_idx=item,
                  request_id=f"{uid}:{timestamp}:{item}", version=version, label=0)
    if round_id is not None:
        result["round_id"] = round_id
    return result


def test_tied_times_keep_strict_past_and_original_behavior_order():
    histories = {1: ([9, 10, 10, 20], [1, 2, 3, 4], [0, 0, 0, 0])}
    requests = {1: [row(1, 10, 8), row(1, 10, 7)]}
    trace = build_trace(histories, requests, [("v3", 10)], 20)
    assert [e.kind for e in trace] == ["publication", "recommendation", "behavior", "drain"]
    assert [r["item_idx"] for r in trace[1].requests] == [8, 7]
    assert trace[2].behavior_indices == (1, 2)
    assert trace[-1].logical_time == 20  # Behavior at end is excluded.
    # A strict-past consumer at the recommendation cannot see tied behaviors.
    visible = [t for t in histories[1][0] if t < trace[1].logical_time]
    assert visible == [9]


def test_round_grouping_never_combines_users_or_distinct_rounds():
    histories = {1: ([9], [1], [0]), 2: ([9], [1], [0])}
    requests = {1: [row(1, 12, 4, round_id="a"), row(1, 12, 5, round_id="b"),
                    row(1, 12, 6, round_id="a")],
                2: [row(2, 12, 8, round_id="a")]}
    rounds = [e for e in build_trace(histories, requests, [("v3", 10)], 20)
              if e.kind == "recommendation"]
    assert [(e.uid, [r["item_idx"] for r in e.requests]) for e in rounds] == [
        (1, [4, 6]), (1, [5]), (2, [8])]


def test_arrival_scaling_keeps_logical_inputs_and_release_order():
    histories = {1: ([9, 15, 20, 24], [1, 2, 3, 4], [0] * 4)}
    requests = {1: [row(1, 20, 8, version="v4")]}
    regular = build_trace(histories, requests, [("v3", 10), ("v4", 20)], 30)
    scaled = build_trace(histories, requests, [("v3", 10), ("v4", 20)], 30, arrival_scale=.01)
    assert [(e.kind, e.logical_time, e.version, e.requests, e.behavior_indices) for e in regular] == [
        (e.kind, e.logical_time, e.version, e.requests, e.behavior_indices) for e in scaled]
    assert [e.arrival_time for e in scaled] == pytest.approx([e.arrival_time * .01 for e in regular])
    assert [e.kind for e in scaled if e.logical_time == 20] == ["publication", "recommendation", "behavior"]


def test_rejects_request_from_wrong_release():
    with pytest.raises(ValueError, match="version"):
        build_trace({1: ([9], [1], [0])}, {1: [row(1, 20, 5)]},
                    [("v3", 10), ("v4", 20)], 30)
