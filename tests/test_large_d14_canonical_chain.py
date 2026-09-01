from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_canonical_large_d14_chain_is_closed_and_current() -> None:
    from validate_yambda500m_large_d14_canonical_chain import validate

    result = validate(
        ROOT / "configs/contracts/yambda500m_large_d14_canonical_v0_v5_v1.yaml",
        verify_checkpoints=False,
    )
    assert result["status"] == "canonical_large_D14_v0_v5_chain_valid"
    assert [row["version"] for row in result["versions"]] == [f"v{index}" for index in range(6)]
    assert [row["epochs"] for row in result["versions"]] == [1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
    assert result["current_version"] == "v5"
    assert result["all_canonical_edges_AUC_positive"]
