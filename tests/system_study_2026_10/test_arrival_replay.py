from pathlib import Path
import sys

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "scripts"),
                str(Path(__file__).resolve().parents[2] / "src")]
from system_study_2026_10.arrival_replay import annotate_trace, compare_scores, summarize_timings
from system_study_2026_10.events import TraceEvent


def test_first_recommendation_label_resets_per_user_and_release():
    trace = [TraceEvent("recommendation", 10, 0., 0, "v3", uid=1),
             TraceEvent("recommendation", 11, 1., 1, "v3", uid=1),
             TraceEvent("recommendation", 11, 1., 2, "v3", uid=2),
             TraceEvent("recommendation", 12, 2., 3, "v4", uid=1)]
    result = annotate_trace(trace)
    assert [result[i]["request_class"] for i in range(4)] == [
        "first_recommendation_in_release", "warm_recommendation",
        "first_recommendation_in_release", "first_recommendation_in_release"]


def test_measured_scores_check_identity_and_tolerance():
    assert compare_scores({"a": 1.}, {"a": 1.}, atol=1e-5, rtol=1e-5)["status"] == "passed"
    with pytest.raises(AssertionError, match="identities"):
        compare_scores({"a": 1.}, {"b": 1.}, atol=1e-5, rtol=1e-5)
    with pytest.raises(AssertionError, match="mismatch"):
        compare_scores({"a": 2.}, {"a": 1.}, atol=1e-5, rtol=1e-5)


def test_latency_summary_excludes_behavior_markers_from_recommendations():
    base = dict(queue_seconds=.01, service_seconds=.02, latency_seconds=.03,
                arrival_time=1., finish_time=1.03)
    rows = [dict(base, kind="recommendation", request_class="first_recommendation_in_release"),
            dict(base, kind="behavior", request_class=None, latency_seconds=100.),
            dict(base, kind="drain", request_class=None, arrival_time=2., finish_time=2.03)]
    value = summarize_timings(rows)
    assert value["groups"]["recommendation"]["latency_seconds"]["count"] == 1
    assert value["groups"]["recommendation"]["latency_seconds"]["p99"] == .03
    assert value["end_queue_seconds"] == .01
