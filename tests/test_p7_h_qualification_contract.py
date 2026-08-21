from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from hstu_kvcache.data.p7_training import P7Request

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluator():
    spec = importlib.util.spec_from_file_location("eval_p7_h_raw", ROOT / "scripts/eval_p7_h_raw.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(*, kind: str, target: int | None) -> P7Request:
    return P7Request(
        request_id="r",
        workload="N",
        uid=1,
        query_timestamp=1000,
        history_items=np.arange(1, 41, dtype=np.int64),
        history_behaviors=np.ones(40, dtype=np.int64),
        history_time_deltas=np.arange(40, dtype=np.float32),
        query_time_delta=5.0,
        candidate_ids=np.asarray([4, 5], dtype=np.int64),
        base_features=np.zeros((2, 7), dtype=np.float32),
        target_index=target,
        label=None,
        request_weight=1.0,
        manifest_kind=kind,
    )


def test_p7_8_roles_seed_policy_and_no_model_training_are_frozen() -> None:
    contract = yaml.safe_load((ROOT / "configs/contracts/p7_8_h_qualification_contract_v1.yaml").read_text())
    assert contract["model_roles"]["primary_workload_models"] == {"R": "M0-R", "F": "M0-F"}
    assert contract["seed_policy"]["qualification_scope"] == "all_frozen_seeds"
    assert contract["seed_policy"]["qualification_H_filtering"] == "forbidden"
    assert contract["authorization"]["train_or_refit_any_model"] is False
    assert contract["authorization"]["theta1_theta2"] is False


def test_fidelity_requests_reject_targets_and_recent_path_resets_boundary_delta() -> None:
    evaluator = _evaluator()
    fidelity = _request(kind="fidelity", target=None)
    tensors = evaluator.collate([fidelity], torch.device("cpu"), history_tokens=32)
    assert tensors["items"].tolist() == [list(range(9, 41))]
    assert tensors["deltas"][0, 0].item() == 0.0
    with pytest.raises(ValueError, match="must not carry targets"):
        _request(kind="fidelity", target=0)


def test_identity_metrics_and_raw_fidelity_schema_are_label_free() -> None:
    evaluator = _evaluator()
    scores = np.asarray([0.2, -0.1, 1.0], dtype=np.float32)
    metrics = evaluator.identity_metrics(scores, scores.copy(), "N")
    assert all(value == 0.0 for value in metrics.values())
    forbidden = set(evaluator.quality_schema().names) - set(evaluator.common_schema().names)
    assert forbidden
    assert not forbidden & set(evaluator.common_schema().names)


def test_p7_8_run_plan_covers_all_frozen_checkpoints_and_forbids_follow_on_work() -> None:
    plan_path = ROOT / "configs/contracts/p7_8_qualification_run_plan_v1.json"
    plan = json.loads(plan_path.read_text())
    assert _sha256(plan_path) == "1f7bc0176e3cbd8e30c7b0ca5d69a0e36fd5a5892c31898ac9061e73590bf683"
    assert len(plan["checkpoints"]) == 12
    assert {int(name.rsplit("seed", 1)[1]) for name in plan["checkpoints"]} == {17, 37, 71}
    assert plan["post_reveal_actions_authorized"] == []


def test_p7_8_raw_seal_is_complete_and_fidelity_artifacts_are_target_free() -> None:
    seal_path = ROOT / "results/p7/h_qualification/raw_score_seal_v1.json"
    seal = json.loads(seal_path.read_text())
    assert _sha256(seal_path) == "a59a76df8ff055e96d7ebc7de1cae4a8599b6723ba89958c8f89697a7a4a7427"
    assert seal["status"] == "sealed_all_raw_scores_before_metrics"
    assert seal["raw_files"] == len(seal["artifacts"]) == 42
    assert seal["metrics_computed"] is False
    forbidden = {"label", "target_index", "target_item", "rankable"}
    fidelity = [artifact for artifact in seal["artifacts"] if artifact["view"] == "fidelity"]
    assert len(fidelity) == 12
    assert all(not (forbidden & set(artifact["schema"])) for artifact in fidelity)


def test_p7_8_adjudication_preserves_all_seeds_and_stops_before_version_chain() -> None:
    report_path = ROOT / "results/p7/h_qualification/adjudication_report_v1.json"
    report = json.loads(report_path.read_text())
    assert _sha256(report_path) == "4ebb18dd3aae4761a8bc7c5e03295cfc7cb13795961c376ab921e31802aae8b1"
    assert report["adjudication_branch"] == "A_primary_M0_workload_passed"
    assert report["version_chain_eligible_conditions"] == ["F"]
    assert report["theta1_theta2_started"] is False
    assert report["requires_new_human_authorization"] is True
    conditions = {
        (entry["model_condition"], entry["workload"]): entry
        for entry in report["all_seed_results"]
    }
    assert set(conditions) == {
        ("m0_n", "N"), ("m0_r", "R"), ("m0_f", "F"),
        ("m1", "N"), ("m1", "R"), ("m1", "F"),
    }
    assert all([seed["seed"] for seed in entry["per_seed"]] == [17, 37, 71] for entry in conditions.values())
    assert conditions[("m0_f", "F")]["classification"] == "provisional_2_of_3"
    assert conditions[("m1", "F")]["classification"] == "robust_3_of_3"


def test_p7_8_result_contract_hashes_and_post_reveal_limitation_are_frozen() -> None:
    result = yaml.safe_load((ROOT / "configs/contracts/p7_8_h_qualification_result_v1.yaml").read_text())
    assert result["authorization_after_this_result"]["theta1_theta2_started"] is False
    assert result["authorization_after_this_result"]["F_development_version_chain"] == "eligible_pending_explicit_human_authorization"
    for artifact in result["artifacts"].values():
        path = ROOT / artifact["path"]
        assert _sha256(path) == artifact["sha256"]
    audit = json.loads((ROOT / "results/p7/h_qualification/post_reveal_audit_v1.json").read_text())
    assert audit["gate_impact"] == "none"
    assert audit["post_reveal_rescoring_or_raw_rewrite"] is False
