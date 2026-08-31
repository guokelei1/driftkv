from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts/train_yambda500m_foundation_fsdp.py"
    spec = importlib.util.spec_from_file_location("foundation_fsdp", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_launch_contract_binds_parent_and_keeps_theta3_locked() -> None:
    launch = yaml.safe_load((ROOT / "configs/contracts/yambda500m_small_seed17_launch_v1.yaml").read_text())
    parent = ROOT / launch["parent_contract"]
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == launch["parent_contract_sha256"]
    assert launch["scope"]["default_versions"] == ["v0", "v1"]
    assert launch["scope"]["v3_theta3_or_later"] == "prohibited"


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
