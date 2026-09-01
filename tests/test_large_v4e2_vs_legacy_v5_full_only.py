from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _comparison():
    from run_yambda500m_large_v4e2_vs_legacy_v5_full_only import FullComparison

    return FullComparison()


def test_comparison_freezes_non_lineage_boundary_and_horizons() -> None:
    comparison = _comparison()
    contract = comparison.contract
    assert contract["models"]["comparison_direct_parent_is_original_v4_e1_not_reference_v4_e2"]
    assert contract["evaluation"]["horizons"]["E7"]["days_half_open"] == [287, 294]
    assert contract["evaluation"]["horizons"]["E7"]["primary"]
    assert contract["evaluation"]["horizons"]["E14_partial"]["days_half_open"] == [287, 301]
    assert not contract["evaluation"]["horizons"]["E14_partial"]["primary"]
    assert contract["evaluation"]["reuse"] == "prohibited"
    assert comparison.formal_horizons == ("E14_partial",)
    assert comparison.scope["scope_change"]["E7_formal"] == "excluded_before_raw_or_label_read"


def test_comparison_command_is_two_model_full_only() -> None:
    comparison = _comparison()
    command = comparison.eval_command("E7", comparison.horizon_dir("E7"), canary=False)
    assert command.count("--parent") == 1
    assert command.count("--current") == 1
    assert command[command.index("--start-day") + 1] == "287"
    assert command[command.index("--end-day") + 1] == "294"
    assert not any("reuse" in value.lower() for value in command)


def test_comparison_formal_requires_explicit_acknowledgement() -> None:
    import pytest

    comparison = _comparison()
    with pytest.raises(RuntimeError, match="requires --acknowledge"):
        comparison.formal(None)
