from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts/train_yambda500m_foundation_fsdp.py"
    spec = importlib.util.spec_from_file_location("foundation_fsdp", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_uid_assignment_is_deterministic_balanced_and_user_closed() -> None:
    module = _load_script()
    uids = np.asarray([1, 2, 3, 4, 5, 6])
    counts = np.asarray([9, 8, 7, 6, 5, 4])
    first = module.balanced_uid_assignment(uids, counts, 4)
    second = module.balanced_uid_assignment(uids, counts, 4)
    assert first == second and set(first) == set(uids.tolist())
    loads = [sum(int(counts[index]) for index, uid in enumerate(uids) if first[int(uid)] == rank) for rank in range(4)]
    assert max(loads) - min(loads) <= int(counts.max())


def test_staged_epoch_schedule_is_contract_bound_and_keeps_one_epoch_boundary() -> None:
    module = _load_script()
    recipe = {
        "checkpoint_epochs": [0.5, 1.0, 1.5, 2.0],
    }
    epochs = module.parse_checkpoint_epochs("0.5,1.0,1.5,2.0", recipe, 2)
    assert epochs == (0.5, 1.0, 1.5, 2.0)
    schedule = module.checkpoint_step_schedule(3597, epochs)
    assert schedule == {1799: 0.5, 3597: 1.0, 5396: 1.5, 7194: 2.0}
    assert module.epoch_checkpoint_name(0.5) == "checkpoint_epoch_0p5.pt"
    assert module.epoch_checkpoint_name(2.0) == "checkpoint_epoch_2.pt"


def test_staged_epoch_schedule_rejects_uncontracted_or_incomplete_endpoints() -> None:
    import pytest

    module = _load_script()
    recipe = {"checkpoint_epochs": [0.5, 1.0, 1.5, 2.0]}
    with pytest.raises(RuntimeError):
        module.parse_checkpoint_epochs("0.5,1.0,2.0", recipe, 2)
    with pytest.raises(RuntimeError):
        module.parse_checkpoint_epochs("0.5,1.0,1.5,2.0", recipe, 3)


def test_weight_only_continuation_reports_cumulative_epochs_but_schedules_local_steps() -> None:
    module = _load_script()
    recipe = {
        "initial_window_epochs": 1,
        "optimizer_reset_at_start": True,
        "checkpoint_epochs": [1],
    }
    local_epochs = module.parse_checkpoint_epochs(None, recipe, 1)
    schedule = module.checkpoint_step_schedule(3597, local_epochs)
    assert schedule == {3597: 1.0}
    metadata = module.window_epoch_metadata(recipe, 1, schedule[3597])
    assert metadata == {
        "initial_window_epochs": 1.0,
        "optimizer_reset_at_start": True,
        "local_training_epochs_completed": 1.0,
        "local_training_epoch_target": 1.0,
        "training_epochs_completed": 2.0,
        "training_epoch_target": 2.0,
    }
    assert module.epoch_checkpoint_name(metadata["training_epochs_completed"]) == "checkpoint_epoch_2.pt"
    unchanged = module.window_epoch_metadata({}, 2, 1.5)
    assert unchanged["training_epochs_completed"] == 1.5
    assert unchanged["training_epoch_target"] == 2.0


def test_cumulative_window_epochs_reject_invalid_initial_exposure() -> None:
    import pytest

    module = _load_script()
    for initial in (-1, float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="initial_window_epochs"):
            module.window_epoch_metadata({"initial_window_epochs": initial}, 1, 1.0)
