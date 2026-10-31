import pytest

from experiments.recflow_request_ranking.code import run_after_pilot
from experiments.recflow_request_ranking.code.data_common import json_save


def test_waits_for_actual_predecessor_even_after_completion_file_appears(tmp_path, monkeypatch):
    path = tmp_path / "completion.json"
    monkeypatch.setattr(run_after_pilot, "process_start_ticks", lambda pid: "original")
    assert run_after_pilot.predecessor_state(123, "original", path).startswith("waiting")
    json_save(path, {"status": "completed"})
    assert run_after_pilot.predecessor_state(123, "original", path).startswith("waiting")


def test_missing_predecessor_without_completion_never_restarts_it(tmp_path, monkeypatch):
    monkeypatch.setattr(run_after_pilot, "process_start_ticks", lambda pid: None)
    with pytest.raises(RuntimeError, match="without completion"):
        run_after_pilot.predecessor_state(123, "original", tmp_path / "completion.json")


@pytest.mark.parametrize("gap_passed", [False, True])
def test_successful_predecessor_can_release_barrier_regardless_of_cache_gap(tmp_path, monkeypatch, gap_passed):
    path = tmp_path / "completion.json"
    monkeypatch.setattr(run_after_pilot, "process_start_ticks", lambda pid: "recycled_other_process")
    json_save(path, {"status": "completed", "quality_passed": gap_passed})
    assert run_after_pilot.predecessor_state(123, "original", path) == "ready"
