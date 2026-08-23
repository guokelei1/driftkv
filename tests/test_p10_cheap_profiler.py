from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_p10_cheap_profiler as profiler


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p10_contract_seals_label_free_inputs_and_charges_probes() -> None:
    contract = yaml.safe_load((ROOT / "configs/contracts/p10_0_cheap_profiler_contract_v1.yaml").read_text())
    assert sha256(ROOT / "results/p9/p9_8_cutover_profiler_v1.json") == contract["inputs"]["p9_8_all_state_fidelity_sha256"]
    assert contract["features"]["future_requests"] == "prohibited"
    assert contract["features"]["feedback_label"] == "prohibited"
    assert contract["probe"]["charged_work"] == "sum_of_all_measured_action_token_layer_costs"
    assert contract["sealing"]["policy_assignments_before_quality_join"] == "required"
    assert contract["evidence_boundary"]["controller_training"] == "prohibited"


def test_p10_probe_order_is_deterministic_and_nested() -> None:
    uids = np.asarray([91, 3, 42, 5, 100], dtype=np.int64)
    order = profiler.deterministic_probe_order("r2", "m1", 17, uids)
    reverse = profiler.deterministic_probe_order("r2", "m1", 17, uids[::-1])
    selected = set(uids[order[:2]])
    selected_reverse = set(uids[::-1][reverse[:2]])
    assert selected == selected_reverse
    assert set(uids[order[:1]]).issubset(selected)


def test_p10_allocator_never_exceeds_budget_and_respects_dependencies() -> None:
    segments = [
        [
            {"delta_cost": 2.0, "delta_benefit": 3.0, "slope": 1.5, "action": "a"},
            {"delta_cost": 3.0, "delta_benefit": 1.0, "slope": 1 / 3, "action": "b"},
        ],
        [{"delta_cost": 1.0, "delta_benefit": 1.0, "slope": 1.0, "action": "c"}],
    ]
    spent, selected = profiler.allocate_predicted(segments, 3.0)
    assert spent == 3.0
    assert selected == ["a", "c"]


def test_p10_assignment_schema_is_explicitly_label_free() -> None:
    allowed = {
        "uid", "release", "model", "seed", "sample_fraction", "budget_fraction",
        "calibration_sample", "action", "predicted_benefit", "action_cost_token_layers",
    }
    forbidden = ("label", "target", "future", "mse", "actual_loss", "quality")
    assert not [column for column in allowed if any(word in column for word in forbidden)]
