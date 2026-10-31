import json
from types import SimpleNamespace

import pytest
import torch

from experiments.recflow_full_population.code.batches import RequestBatches
from experiments.recflow_full_population.code.data_common import json_save
from experiments.recflow_full_population.code.runtime import load_checkpoint
from experiments.recflow_release_design.code.run_branches import Runner


def test_checkpoint_load_ignores_historical_digest_and_still_requires_readable_model(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": {"weight": torch.tensor([3.0])}}, checkpoint)
    json_save(checkpoint.with_suffix(".json"), {"status": "committed", "checkpoint_sha256": "old-record"})
    payload, _ = load_checkpoint(checkpoint)
    assert torch.equal(payload["model"]["weight"], torch.tensor([3.0]))
    checkpoint.write_bytes(b"")
    with pytest.raises(EOFError):
        load_checkpoint(checkpoint)


def test_request_loader_ignores_digest_but_keeps_actual_request_count_check(tmp_path):
    import numpy as np

    np.save(tmp_path / "requests.npy", np.zeros(2, dtype=[("rid", "i8")]))
    manifest = {"datasets": {"0:32": {"file": "requests.npy", "requests": 2, "sha256": "old-record"}}}
    json_save(tmp_path / "selection_manifest.json", manifest)
    assert len(RequestBatches(tmp_path, [32])) == 2
    manifest["datasets"]["0:32"]["requests"] = 3
    json_save(tmp_path / "selection_manifest.json", manifest)
    with pytest.raises(ValueError, match="count"):
        RequestBatches(tmp_path, [32])


def test_completed_stages_finalize_despite_old_paper_and_source_hashes(tmp_path, monkeypatch):
    """Exercise the coordinator's finalization without dispatching training."""
    monkeypatch.setattr("experiments.recflow_release_design.code.run_branches.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=100 * 1024**3))
    directory = tmp_path / "branches" / "example"
    training = directory / "training"
    training.mkdir(parents=True)
    spec = directory / "spec.json"
    json_save(spec, {"branch": "example", "first_pass_checkpoint_path": "/models/first.pt", "train_day": 32})
    json_save(training / "completion.json", {"status": "completed", "next_epoch": 1, "next_step": 0})
    json_save(training / "latest.json", {"checkpoint": "step_1.pt"})
    json_save(training / "step_1.json", {"checkpoint_sha256": "record-only"})
    metrics = {"current": {"ndcg@10": 0.04}, "reuse": {"ndcg@10": 0.05}}
    json_save(directory / "evaluation/original_vs_extra_epoch.json", {
        "branch": "example", "old_metrics": metrics, "new_metrics": metrics,
        "new_relative_loss_pct": -25.0, "new_gap_ci95": [-0.02, -0.001],
        "new_minus_old_rebuild": {"difference": 0.0, "ci95": [-0.01, 0.01]},
    })
    json_save(tmp_path / "plan.json", {
        "branch_specs": [str(spec)], "binding": {"/no/longer/present/source.py": "old-record"},
        "paper_sha256": "old-paper-record",
    })
    runner = Runner(tmp_path)
    monkeypatch.setattr(runner, "evaluate", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "stage", lambda *args, **kwargs: None)
    runner.execute()
    completion = json.loads((tmp_path / "completion.json").read_text())
    assert completion["status"] == "completed" and completion["completed_branches"] == 1
    assert "-25.0" in (tmp_path / "branch_metrics.csv").read_text()
