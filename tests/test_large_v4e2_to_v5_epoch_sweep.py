from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _sweep():
    from run_yambda500m_large_v4e2_to_v5_epoch_sweep import V5EpochSweep

    return V5EpochSweep()


def test_v5_sweep_is_direct_parent_continuous_and_e14_partial_only() -> None:
    sweep = _sweep()
    assert sweep.epochs == (1.0, 2.0)
    assert sweep.contract["scope"]["expected_parent_version"] == "v4"
    assert sweep.contract["scope"]["branches"]["D14"]["training_days_half_open"] == [273, 287]
    assert sweep.contract["evaluation"]["day_range_half_open"] == [287, 301]
    assert sweep.contract["evaluation"]["horizon"] == "E14_partial"
    assert sweep.contract["evaluation"]["reuse"] == "prohibited"


def test_v5_sweep_training_and_eval_commands_preserve_scope() -> None:
    sweep = _sweep()
    train = sweep.train_command(sweep.checkpoint_dir, canary=False)
    assert train[train.index("--version") + 1] == "v5"
    assert train[train.index("--passes") + 1] == "2"
    assert train[train.index("--checkpoint-epochs") + 1] == "1.0,2.0"
    assert train[train.index("--parent") + 1] == str(sweep.parent)
    currents = {f"v5_e{epoch:.1f}".replace(".", "p"): sweep.checkpoint(epoch) for epoch in sweep.epochs}
    evaluate = sweep.eval_command(sweep.full_dir, currents, canary=False)
    assert evaluate.count("--current") == 2
    assert evaluate[evaluate.index("--start-day") + 1] == "287"
    assert evaluate[evaluate.index("--end-day") + 1] == "301"
    assert not any("reuse" in value.lower() for value in evaluate)


def test_v5_sweep_formal_requires_acknowledgement() -> None:
    import pytest

    with pytest.raises(RuntimeError, match="requires --acknowledge"):
        _sweep().formal(None)
