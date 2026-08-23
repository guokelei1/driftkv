from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import adjudicate_p10_scheduler_baselines as baselines


def test_p10_3_contract_freezes_primary_and_nonlearning_family() -> None:
    contract = yaml.safe_load((ROOT / "configs/contracts/p10_3_scheduler_baseline_gate_contract_v1.yaml").read_text())
    assert contract["primary_policy"]["sample_fraction"] == 0.01
    assert contract["companion_policy"]["sample_fraction"] == 0.02
    assert len(contract["equal_cost_baselines"]["metadata_only_zero_probe_Exact"]["strategies"]) == 4
    assert contract["uncertainty"]["target_free_comparison"] == "paired_user_bootstrap"
    assert contract["evidence_boundary"]["blind_edge"] == "prohibited"


def test_metadata_exact_allocator_is_deterministic_and_never_exceeds_budget() -> None:
    costs = np.asarray([4.0, 3.0, 2.0, 1.0])
    order = np.asarray([0, 1, 2, 3])
    selected = baselines.exact_allocation(order, costs, 6.0)
    assert selected.tolist() == [True, False, True, False]
    assert costs[selected].sum() <= 6.0


def test_paired_bootstrap_preserves_strictly_positive_delta() -> None:
    result = baselines.paired_bootstrap(np.ones(20), "fixed", 200)
    assert result["p2_5"] > 0
