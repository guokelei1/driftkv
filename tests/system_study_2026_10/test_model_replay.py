from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "scripts"),
                str(Path(__file__).resolve().parents[2] / "src")]
from system_study_2026_10.model_replay import choose_users, compare_snapshot, state_snapshot


def test_smoke_selection_uses_first_registered_profile_not_quality():
    profiles = [dict(uid=7, stratum="short"), dict(uid=2, stratum="short"),
                dict(uid=5, stratum="long")]
    assert choose_users(dict(profiles=profiles)) == [7, 5]


def test_snapshot_comparison_checks_actual_tensors_and_operation_counts():
    target = dict(metadata=dict(cursor=3, history_rows=3, ledger=[["append", 1]]),
                  tensors=dict(k=torch.ones(2), v=torch.zeros(2), codes=torch.ones(3)))
    same = dict(metadata=dict(target["metadata"]),
                tensors={key: value.clone() for key, value in target["tensors"].items()})
    assert compare_snapshot(same, target, atol=1e-5, rtol=1e-5)["status"] == "passed"
    same["tensors"]["codes"][0] += .1
    with pytest.raises(AssertionError):
        compare_snapshot(same, target, atol=1e-5, rtol=1e-5)
    same["metadata"]["cursor"] = 4
    with pytest.raises(AssertionError, match="metadata"):
        compare_snapshot(same, target, atol=1e-5, rtol=1e-5)


def test_boundary_snapshot_does_not_alias_later_mutable_state():
    cache = SimpleNamespace(k=torch.ones(1), v=torch.zeros(1), seq_len=1)
    trajectory = SimpleNamespace(cache=cache, version=3, cursor=1, time=20,
        producers=np.array([2]), appended_tokens=0, evicted_tokens=0, ledger={})
    session = SimpleNamespace(trajectory=trajectory, state=None,
                              events=[dict(action="publish", version=3)])
    captured = state_snapshot(session)
    session.events[0]["version"] = 4
    session.events.append(dict(action="publish", version=4))
    cache.k[0] = 9
    assert captured["metadata"]["events"] == [dict(action="publish", version=3)]
    assert captured["tensors"]["k"].item() == 1


def test_raw_trajectory_history_arrays_are_not_session_event_log():
    cache = SimpleNamespace(k=torch.ones(1), v=torch.zeros(1), seq_len=1)
    trajectory = SimpleNamespace(cache=cache, version=3, cursor=1, time=20,
        producers=np.array([2]), appended_tokens=0, evicted_tokens=0, ledger={},
        events=(np.array([19]), np.array([5]), np.array([1])))
    captured = state_snapshot(trajectory)
    assert captured["metadata"]["events"] == []
    assert compare_snapshot(captured, captured, atol=1e-5, rtol=1e-5)["status"] == "passed"
