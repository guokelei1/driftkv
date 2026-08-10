from __future__ import annotations

from runpy import run_path

MATERIALIZER = run_path("scripts/materialize_evokv_qk_theta2_negative_strength.py")
SUMMARIZER = run_path("scripts/summarize_evokv_qk_theta2_negative_strength.py")
CANDIDATES = MATERIALIZER["CANDIDATES"]
PRIMARY_GATE = SUMMARIZER["_primary_gate"]


def test_negative_strength_bindings_match_frozen_matrix() -> None:
    entries = (
        "scripts/train_evokv_qk_theta2_strength_e2_lr075_n64.py",
        "scripts/train_evokv_qk_theta2_strength_e2_lr075_n96.py",
        "scripts/train_evokv_qk_theta2_strength_e3_lr075_n64.py",
        "scripts/train_evokv_qk_theta2_strength_e2_lr100_n64.py",
    )
    bindings = tuple(run_path(path)["BINDING"] for path in entries)
    observed = {
        value.candidate_name: (
            value.epochs,
            value.dense_learning_rate,
            value.projection_learning_rate,
            value.embedding_learning_rate,
            value.train_negative_count,
        )
        for value in bindings
    }
    assert observed == CANDIDATES
    assert {value.training_seed for value in bindings} == {2026080611}
    assert {value.negative_seed for value in bindings} == {2026080623}


def test_primary_gate_requires_range_and_both_intervals() -> None:
    row = {
        "positive_targets": 77479,
        "gaps": {
            "ndcg_at_10": {
                "relative_percent": 6.0,
                "positive_direction_with_ci": True,
            },
            "mrr": {
                "relative_percent": 4.0,
                "positive_direction_with_ci": True,
            },
        },
    }
    assert PRIMARY_GATE(row, [5.0, 10.0], 5000)["status"] == "pass"
    row["gaps"]["mrr"]["positive_direction_with_ci"] = False
    assert PRIMARY_GATE(row, [5.0, 10.0], 5000)["status"] == "fail"
    row["gaps"]["mrr"]["positive_direction_with_ci"] = True
    row["gaps"]["ndcg_at_10"]["relative_percent"] = 2.0
    assert PRIMARY_GATE(row, [5.0, 10.0], 5000)["status"] == "fail"
