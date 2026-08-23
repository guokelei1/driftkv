from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p10_quality_contract_uses_presealed_policy_and_prohibits_refit() -> None:
    contract = yaml.safe_load((ROOT / "configs/contracts/p10_1_policy_quality_contract_v1.yaml").read_text())
    assert sha256(ROOT / "results/p10/p10_0_cheap_profiler_full_seal_v1.json") == contract["inputs"]["p10_0_full_policy_seal_sha256"]
    assert sha256(ROOT / "results/p9/p9_9_heldout_rolling_quality_raw_seal_v1.json") == contract["inputs"]["p9_9_heldout_quality_raw_seal_sha256"]
    assert contract["join"]["policy_refit_after_labels"] == "prohibited"
    assert contract["join"]["policy_selection_after_labels"] == "prohibited"
    assert contract["scope"]["all_144_frozen_policies_required"] is True
    assert contract["evidence_boundary"]["dislike_only_caveat_mandatory"] is True
    assert contract["adjudication"]["controller_authorized_by_this_step"] is False
