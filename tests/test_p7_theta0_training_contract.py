from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def test_p7_7_contract_never_authorizes_qualification_or_H_selection() -> None:
    contract = yaml.safe_load((ROOT / "configs/contracts/p7_7_theta0_training_contract_v1.yaml").read_text())
    assert contract["authorization"]["qualification_read_or_scoring"] is False
    assert contract["authorization"]["theta1_theta2"] is False
    assert contract["model"]["context"] == "Full-512_only"
    assert "Full_minus_Recent" in contract["checkpoint_selection"]["forbidden_metrics"]


def test_p7_7_frozen_bundle_contains_all_model_seed_runs() -> None:
    bundle = _json("results/p7/theta0_training/frozen_theta0_bundle_v1/bundle_manifest.json")
    identities = {(row["model"], row["seed"]) for row in bundle["artifacts"]}
    assert identities == {
        (model, seed)
        for model in ("m0_n", "m0_r", "m0_f", "m1")
        for seed in (17, 37, 71)
    }
    assert bundle["qualification_scored"] is False


def test_p7_7_budget_and_development_evidence_boundaries() -> None:
    budget = _json("results/p7/theta0_training/theta0_training_budget_audit_v1.json")
    assert budget["status"] == "passed"
    assert budget["comparison"]["m1_vs_each_task_specific_m0_compute_matched"] is False
    assert budget["comparison"]["m1_vs_three_m0_aggregate_query_candidate_history_token_work_matched"] is True
    for row in budget["seeds"]:
        for key in ("unique_queries", "query_presentations", "candidate_rows", "history_tokens", "token_layer_work"):
            assert row["three_m0_aggregate"][key] == row["m1_aggregate"][key]

    sanity = _json("results/p7/theta0_training/theta0_dev_sanity_v1.json")
    assert sanity["qualification_scored"] is False
    assert sanity["recent32_or_H_evaluated"] is False
    assert sanity["status"] == "passed_development_only_not_H"
