"""Cold windows require zero history consumers, not merely absent queries."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
from experiments.archive.query_relay.analyze_cold_boundaries import group_windows, length_bucket


def test_skipped_windows_and_first_consumer_respect_publish_time_and_behavior():
    releases = (("v3", 10), ("v4", 20), ("v5", 30))
    # No requests in V3, but a behavior at 10 prepares codes, so V3 is active.
    windows = group_windows(np.array([1, 2, 10, 30, 40]), {"v5": 30},
                            releases=releases, end=40, cap=4)
    assert windows["v3"]["first_consumer"] == "behavior_before_recommendation"
    assert windows["v3"]["first_read_history_length"] == 2
    assert windows["v4"]["cold_group"] == "idle_no_consumer"
    assert windows["v5"]["cold_group"] == "skip_1"
    assert windows["v5"]["first_consumer"] == "recommendation"
    assert windows["v5"]["first_read_history_length"] == 3
    assert windows["v5"]["behavior_events"] == 1  # Event at END excluded.
    cold = group_windows(np.array([1, 2, 30]), {"v5": 30}, releases=releases, end=40)
    assert cold["v5"]["cold_group"] == "skip_2"
    assert length_bucket(None) == "idle_no_consumer"
    assert length_bucket(0) == "0"
    assert length_bucket(256) == "1_to_256"
    assert length_bucket(512) == "257_to_512"
    assert length_bucket(1024) == "513_to_1024"
