import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
SCRIPT = SCRIPT_DIR / "run_cohortkv_stage5_real_edge_smoke.py"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "run_cohortkv_stage5_real_edge_smoke",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def frozen_args(**overrides):
    values = {
        "device": "cuda:0",
        "batch_size": MODULE.BATCH_SIZE,
        "seed": 0,
        "smoke_test": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def plan(retained=3, delta=2, latest=1):
    return SimpleNamespace(
        migration_eligible=True,
        retained_tokens=retained,
        delta_tokens=delta,
        latest_tokens=latest,
        final_tokens=retained + delta + latest,
    )


def test_runner_requires_explicit_available_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match="requires --smoke-test"):
        MODULE.validate_args(frozen_args(smoke_test=False))
    with pytest.raises(ValueError, match="explicit CUDA index"):
        MODULE.validate_args(frozen_args(device="cuda:1"))
    assert MODULE.validate_args(frozen_args()) == torch.device("cuda:0")


def test_selection_uses_program_selection_migrant_and_scheduled_exact() -> None:
    plans = {3: plan(), 5: plan(), 8: plan()}
    selection = SimpleNamespace(
        migrate_ids=(3, 5),
        scheduled_exact_ids=(8,),
    )
    records = [
        {"record_id": 3, "evaluation_role": "final_test"},
        {"record_id": 5, "evaluation_role": "program_selection"},
        {"record_id": 8, "evaluation_role": "certificate"},
    ]

    assert MODULE.select_real_edge_records(
        plans,
        selection,
        records,
    ) == (5, 8)


def test_selection_rejects_non_role_canary_or_no_delta() -> None:
    plans = {3: plan(delta=0), 8: plan()}
    selection = SimpleNamespace(
        migrate_ids=(3,),
        scheduled_exact_ids=(8,),
    )
    records = [
        {"record_id": 3, "evaluation_role": "program_selection"},
        {"record_id": 8, "evaluation_role": "certificate"},
    ]

    with pytest.raises(RuntimeError, match="program-selection migrant"):
        MODULE.select_real_edge_records(plans, selection, records)


def test_requests_bind_retained_and_final_lengths() -> None:
    plans = {5: plan(retained=4, delta=2), 8: plan(retained=6, delta=1)}

    migrate, exact = MODULE.build_requests(5, 8, plans)

    assert migrate.requested_action == "migrate"
    assert migrate.retained_tokens == 4
    assert migrate.final_tokens == 7
    assert exact.requested_action == "exact"
    assert exact.retained_tokens == 6
    assert exact.final_tokens == 8


def test_event_sequence_requires_retained_guard_append_per_extent() -> None:
    identity = {
        "record_ids": [5],
        "action": "migrate",
        "cohort_id": "theta0-cohort",
    }
    events = [
        {"kind": "retained", **identity},
        {"kind": "guard", **identity},
        {"kind": "append", **identity},
    ]

    assert MODULE.validate_event_sequence(events)
    events[1] = {"kind": "append", **identity}
    assert not MODULE.validate_event_sequence(events)


def test_canary_config_is_fixed_and_non_scientific() -> None:
    config, digest = MODULE.load_canary_config(MODULE.CANARY_CONFIG)

    assert config["selection_role"] == "program_selection"
    assert config["labels_used"] is False
    assert config["scientific_result"] is False
    assert len(digest) == 64
