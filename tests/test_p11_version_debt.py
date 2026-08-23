from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_p11_contract_distinguishes_direct_diagnostic_and_recursive_lineage() -> None:
    contract = yaml.safe_load((ROOT / "configs/contracts/p11_0_version_debt_contract_v1.yaml").read_text())
    assert contract["lineages"]["RecursiveMixed"]["service_between_releases"] == "append_each_event_with_theta1_and_rolling_cap_512"
    assert contract["interpretation"]["DirectAge2Diagnostic_is_deployable_lineage"] is False
    assert contract["interpretation"]["RecursiveMixed_is_deployable_NoOp_lineage"] is True
    assert contract["prohibited"][0] == "theta3_blind_edge"


def test_p11_population_contract_keeps_recursive_lineage_and_actions_label_free() -> None:
    contract = yaml.safe_load((ROOT / "configs/contracts/p11_1_recursive_population_contract_v1.yaml").read_text())
    assert contract["scope"]["expected_population"] == 8229
    assert contract["scope"]["lineages"]["recursive_noop"].startswith("theta0_materialized")
    assert contract["scope"]["recursive_actions"][-1] == "exact_all"
    assert contract["interpretation"]["direct_age2_is_deployable"] is False
    assert "future_labels" in contract["prohibited"]
