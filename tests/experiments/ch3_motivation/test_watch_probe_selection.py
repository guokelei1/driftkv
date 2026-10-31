"""The durable watcher distinguishes completed evidence from successful fits."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import watch_probe_selection as watcher


def setup(tmp_path, failed=False):
    queue = tmp_path / "queue"
    queue.mkdir()
    inventory, edges = [], []
    for edge in range(1, 6):
        points = []
        for index in range(48):
            path = tmp_path / f"weight_{edge}_{index}.pt"
            point = dict(checkpoint=str(path), checkpoint_sha256=f"sha-{edge}-{index}", status="validated")
            if failed and edge == 5 and index == 0:
                point.update(status="failed", error="numerical failure")
            points.append(point)
            inventory.append(dict(edge=edge, checkpoint=str(path), checkpoint_sha256=point["checkpoint_sha256"]))
        failures = sum(point["status"] == "failed" for point in points)
        summary = dict(status="complete_with_failures" if failures else "complete", common_validation=dict(edge=edge),
                       points=points, failed=failures, validated=48-failures)
        path = tmp_path / f"common/edge_{edge}/summary.json"
        watcher.write_json(path, summary)
        edges.append(dict(edge=edge, status=summary["status"], points=48, failed=failures,
                          summary=str(path), summary_sha256=watcher.sha256(path)))
    watcher.write_json(queue / "checkpoint_inventory.json", inventory)
    watcher.write_json(queue / "summary.json", dict(status="complete_with_failures" if failed else "complete",
        points=240, edges=edges, quality_auc_computed=False))
    (queue / "completed.txt").write_text("finished\n")
    return SimpleNamespace(queue=queue, common_root=tmp_path / "common", state=tmp_path / "state",
        output=tmp_path / "output", wait_pid=123456789,
        selector_sha256=watcher.sha256(Path(watcher.__file__).with_name("select_probe_common.py")))


def mock_selector(monkeypatch, args):
    commands = []

    def run(command, *, check, env):
        commands.append(command)
        assert check is True and env["CUDA_VISIBLE_DEVICES"] == ""
        watcher.write_json(args.output / "selection.json", dict(status="recommendation_for_review",
            all_five_edges_complete=True, recommendation=dict(recipe="fixture", correction_seeds=[17, 23, 29])))

    monkeypatch.setattr(watcher.subprocess, "run", run)
    return commands


@pytest.mark.parametrize("failed", [False, True])
def test_completed_and_failed_validation_have_distinct_markers(tmp_path, monkeypatch, failed):
    args = setup(tmp_path, failed)
    commands = mock_selector(monkeypatch, args)
    code = watcher.run(args)
    assert len(commands) == 1
    summary = json.loads((args.state / "summary.json").read_text())
    assert summary["gpu_training_started"] is False
    if failed:
        assert code == 2 and summary["status"] == "complete_with_validation_failures"
        assert (args.state / "review_required.txt").exists()
        assert not (args.state / "completed.txt").exists()
        assert json.loads((args.state / "input_audit.json").read_text())["failed"] == 1
    else:
        assert code == 0 and summary["status"] == "complete_for_root_review"
        assert (args.state / "completed.txt").exists()


def test_dead_predecessor_without_marker_is_a_failure(tmp_path, monkeypatch):
    args = setup(tmp_path)
    (args.queue / "completed.txt").unlink()
    monkeypatch.setattr(watcher.os, "kill", lambda *a: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(watcher.time, "sleep", lambda *a: pytest.fail("should not wait on a dead predecessor"))
    with pytest.raises(RuntimeError, match="without a completion marker"):
        watcher.run(args)
    assert (args.state / "failure.json").exists() and not (args.state / "completed.txt").exists()


def test_summary_tamper_stops_before_selector(tmp_path, monkeypatch):
    args = setup(tmp_path)
    (args.common_root / "edge_1/summary.json").write_text("changed")
    monkeypatch.setattr(watcher.subprocess, "run", lambda *a, **k: pytest.fail("must stop before selection"))
    with pytest.raises(ValueError, match="summary changed"):
        watcher.run(args)
    assert (args.state / "failure.json").exists() and not (args.output / "selection.json").exists()


def test_frozen_selector_hash_failure_is_not_a_completed_job(tmp_path):
    args = setup(tmp_path)
    args.selector_sha256 = "not-the-frozen-source"
    with pytest.raises(ValueError, match="selector source changed"):
        watcher.run(args)
    assert (args.state / "failure.json").exists() and not (args.state / "completed.txt").exists()
