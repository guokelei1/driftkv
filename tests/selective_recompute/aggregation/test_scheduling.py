"""Scheduling changes execution order while preserving the fixed canary subset."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from selective_recompute_2026_09.scheduling import ordered_uids


def rows(prefix, appends):
    return [{"cache_length": min(1024, prefix + count), "append_count_since_cutover": count,
             "rolling_evictions": max(0, prefix + count - 1024)} for count in appends]


def test_orders_workload_without_changing_selected_users_or_request_rows():
    by_user = {1: rows(1024, [2, 3]), 2: rows(1024, [500]),
               3: rows(20, [1500]), 4: rows(1024, [2000])}
    selected = [1, 3, 2]  # A previous request-count selection excludes user 4.
    assert ordered_uids(by_user, uids=selected) == [2, 1, 3]
    assert selected == [1, 3, 2]
    assert [len(by_user[uid]) for uid in selected] == [2, 1, 1]
    assert ordered_uids(by_user) == [4, 2, 1, 3]


def test_inconsistent_saved_stream_counts_fail_before_scheduling():
    broken = rows(1024, [2, 3])
    broken[1]["rolling_evictions"] += 1
    with pytest.raises(RuntimeError, match="inconsistent saved initial cache lengths"):
        ordered_uids({1: broken})
