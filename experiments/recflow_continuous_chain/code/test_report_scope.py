"""CPU-only reporting regressions; synthetic evidence never writes a real run."""
import copy
import csv
import json

import numpy as np
import pytest

from . import export_chain as exporter
from . import validate_chain as validator


class ReportingInspector:
    """Evidence stub to isolate scope/date reporting from unchanged GPU audits."""
    def model(self, path):
        if not path:
            raise validator.Unproven("No checkpoint")
        index = int(validator.resolve(path).stem[1:])
        return {"checkpoint": str(validator.resolve(path)), "train_end_day": 27 + index,
                "actual_parent": None, "train_days": [27 + index]}

    def ancestry(self, target, source):
        return [self.model(source), self.model(target)]

    def original(self, directory, source, target, day):
        return {"metrics": {"point_positive": True}}


@pytest.fixture
def scoped_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "Inspector", ReportingInspector)
    revision = tmp_path / "revision.json"
    revision.write_text(json.dumps(validator.read(validator.BASE / "objective_revision_2026_10_29.json")))
    manifest = copy.deepcopy(validator.read(validator.BASE / "chain_manifest.json"))
    manifest["objective_revision"] = str(revision)
    manifest.pop("planned_release_scenario", None)
    for index, node in enumerate(manifest["nodes"]):
        node["checkpoint"] = str(tmp_path / f"V{index}.pt") if index < 7 else None
    for index, edge in enumerate(manifest["edges"]):
        edge["evaluation_day"] = 29 + index
        edge["evidence"] = {"kind": "original_full_population", "directory": "dummy"} if index < 6 else {}
    return manifest


def fill(manifest, count):
    root = validator.resolve(manifest["nodes"][0]["checkpoint"]).parent
    for index in range(count):
        manifest["nodes"][index]["checkpoint"] = str(root / f"V{index}.pt")
    for edge in manifest["edges"][:count - 1]:
        edge["evidence"] = {"kind": "original_full_population", "directory": "dummy"}


def add_review(manifest, tmp_path):
    nodes = [node for node in manifest["nodes"] if node["checkpoint"]]
    edges = manifest["edges"][:len(nodes) - 1]
    evidence = tmp_path / "external_remaining_windows.json"
    evidence.write_text('{"status": "completed", "note": "synthetic test only"}')
    review = {"status": "completed", "reason": "data_exhaustion",
              "actual_model_checkpoints": [node["checkpoint"] for node in nodes],
              "actual_edge_ids": [edge["id"] for edge in edges],
              "evaluation_days": [edge["evaluation_day"] for edge in edges],
              "last_available_data_day": 37, "latest_training_cutoff_with_future_evaluation": 36,
              "remaining_window_review": "External reviewer assessed retained failed and final available windows.",
              "evidence_paths": [str(evidence)], "no_pending_work": True}
    path = tmp_path / "external_review.json"
    path.write_text(json.dumps(review))
    manifest["data_exhaustion_review"] = str(path)
    return path, review


def test_current_prefix_is_valid_but_ongoing(scoped_manifest):
    result = validator.inspect_manifest(scoped_manifest)
    assert result["actual_prefix_structural_and_point_requirements_proven"]
    assert (result["actual_accepted_model_count"], result["actual_accepted_edge_count"]) == (7, 6)
    assert result["status"] == "ongoing"
    assert result["data_exhaustion_review"]["status"] == "missing_external_review"
    assert not result["preferred_target_achieved"]
    assert not result["goal_requirements_proven"]
    assert validator.cli_exit_code(result) == 2


def test_two_day_release_ignores_old_pending_day_but_rejects_real_duplicate(scoped_manifest):
    fill(scoped_manifest, 8)
    scoped_manifest["edges"][6]["evaluation_day"] = 36
    result = validator.inspect_manifest(scoped_manifest)
    assert result["actual_prefix_structural_and_point_requirements_proven"]
    assert result["edges"][7]["evaluation_day"] is None
    assert result["edges"][7]["preferred_placeholder_day"] == 36
    fill(scoped_manifest, 9)
    result = validator.inspect_manifest(scoped_manifest)
    assert result["status"] == "contradicted"
    assert result["actual_accepted_edge_count"] == 7
    assert any("strictly increasing" in issue["detail"] for issue in result["issues"])


def test_checkpoint_hole_cannot_count_later_nodes(scoped_manifest):
    scoped_manifest["nodes"][3]["checkpoint"] = None
    result = validator.inspect_manifest(scoped_manifest)
    assert result["status"] == "contradicted"
    assert (result["actual_accepted_model_count"], result["actual_accepted_edge_count"]) == (3, 2)


def test_full_preferred_count_still_does_not_prove_stability(scoped_manifest):
    fill(scoped_manifest, 10)
    result = validator.inspect_manifest(scoped_manifest)
    assert result["preferred_target_achieved"]
    assert result["terminal_basis"] == "preferred_count"
    assert validator.cli_exit_code(result) == 3
    assert not result["goal_requirements_proven"]


def test_external_review_binds_short_chain_without_proving_goal(scoped_manifest, tmp_path):
    scoped_manifest["nodes"] = scoped_manifest["nodes"][:7]
    scoped_manifest["edges"] = scoped_manifest["edges"][:6]
    add_review(scoped_manifest, tmp_path)
    result = validator.inspect_manifest(scoped_manifest)
    assert result["data_exhaustion_review"]["references_verified"]
    assert not result["data_exhaustion_review"]["independently_proven_by_this_tool"]
    assert result["terminal_basis"] == "externally_reviewed_data_exhaustion"
    assert validator.cli_exit_code(result) == 3
    assert not result["preferred_target_achieved"] and not result["goal_requirements_proven"]


@pytest.mark.parametrize("fault", ["wrong_chain", "missing_evidence", "no_window_review", "pending_work", "live_run"])
def test_missing_or_stale_terminal_evidence_remains_ongoing(scoped_manifest, tmp_path, fault):
    path, review = add_review(scoped_manifest, tmp_path)
    if fault == "wrong_chain":
        review["evaluation_days"][-1] = 35
    elif fault == "missing_evidence":
        review["evidence_paths"] = [str(tmp_path / "missing.json")]
    elif fault == "no_window_review":
        review["remaining_window_review"] = ""
    elif fault == "pending_work":
        review["no_pending_work"] = False
    else:
        scoped_manifest["planned_release_scenario"] = {"current_training_run": str(tmp_path / "running")}
    path.write_text(json.dumps(review))
    result = validator.inspect_manifest(scoped_manifest)
    assert result["status"] == "ongoing"
    assert not result["data_exhaustion_review"]["references_verified"]
    assert not result["structural_and_point_requirements_proven"]


def test_real_current_export_keeps_metrics_and_separates_slots(tmp_path):
    """Inspect completed local artifacts only; temporary export, no model loads."""
    baseline = list(csv.DictReader((validator.BASE / "chain_edges.csv").open()))
    manifest = validator.read(validator.BASE / "chain_manifest.json")
    expected_models = sum(bool(node.get("checkpoint")) for node in manifest["nodes"])
    result = exporter.export(validator.BASE / "chain_manifest.json", tmp_path)
    report = validator.read(tmp_path / "chain_report.json")
    assert result["actual_prefix_structural_and_point_requirements_proven"]
    assert (len(report["actual_chain_models"]), len(report["actual_chain_edges"])) == (expected_models, expected_models - 1)
    assert (len(report["models"]), len(report["edges"])) == (10, 9)
    assert result["preferred_target_achieved"] == (expected_models == 10)
    assert not result["goal_requirements_proven"]
    for index, edge in enumerate(report["actual_chain_edges"]):
        if baseline[index]["ndcg_relative_loss_pct"]:
            assert edge["ndcg_relative_loss_pct"] == float(baseline[index]["ndcg_relative_loss_pct"])
    for edge in report["preferred_pending_edge_slots"]:
        assert edge["evaluation_day"] is None
        assert edge["window_status"] == "preferred_placeholder_not_scheduled"
    if not manifest.get("data_exhaustion_review") and expected_models < 10:
        assert result["status"] == "ongoing"
        assert report["data_exhaustion_review"]["status"] == "missing_external_review"
    assert np.isfinite([edge["ndcg_relative_loss_pct"] for edge in report["actual_chain_edges"]]).all()
