from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _sweep():
    from run_yambda500m_large_v3_v4_epoch_sweep import EpochSweep

    return EpochSweep()


def test_epoch_sweep_contract_and_windows_are_frozen() -> None:
    sweep = _sweep()
    assert sweep.epochs == (0.5, 1.0, 1.5, 2.0)
    assert sweep.contract["scope"]["branches"]["D14"]["training_days_half_open"] == [259, 273]
    assert sweep.contract["evaluation"]["day_range_half_open"] == [273, 287]
    assert sweep.contract["evaluation"]["reuse"] == "prohibited"
    assert sweep.contract["authorization"]["formal_long_training"].startswith("requires_new_explicit")


def test_epoch_sweep_training_is_one_two_epoch_trajectory() -> None:
    sweep = _sweep()
    command = sweep.train_command(sweep.checkpoint_dir, canary=False)
    assert command.count("scripts/train_yambda500m_foundation_fsdp.py") == 1
    assert command[command.index("--passes") + 1] == "2"
    assert command[command.index("--checkpoint-epochs") + 1] == "0.5,1.0,1.5,2.0"
    assert command[command.index("--parent") + 1] == str(sweep.parent)
    assert "--canary-steps" not in command


def test_epoch_sweep_full_only_command_has_all_candidates_and_no_reuse() -> None:
    sweep = _sweep()
    currents = {
        f"v4_e{str(epoch).replace('.', 'p')}": sweep.checkpoint(epoch)
        for epoch in sweep.epochs
    }
    command = sweep.eval_command(sweep.full_dir, currents, canary=False)
    assert command.count("--current") == 4
    assert command[command.index("--start-day") + 1] == "273"
    assert command[command.index("--end-day") + 1] == "287"
    assert not any("reuse" in value.lower() for value in command)


def test_epoch_sweep_formal_requires_explicit_acknowledgement() -> None:
    import pytest

    sweep = _sweep()
    with pytest.raises(RuntimeError, match="requires --acknowledge"):
        sweep.formal(None)
