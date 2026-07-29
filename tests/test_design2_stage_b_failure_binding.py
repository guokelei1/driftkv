import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LAUNCHER = _load(
    "cohortkv_design2_stage_b_launcher_failure_test",
    ROOT / "scripts/launch_cohortkv_design2_stage_b.py",
)
FREEZER = _load(
    "cohortkv_design2_stage_b_freezer_failure_test",
    ROOT / "scripts/freeze_cohortkv_design2_stage_b.py",
)


def _payload():
    result = LAUNCHER.ProcessResult(
        returncode=1,
        elapsed_seconds=4.0,
        timed_out=False,
        stdout="",
        stderr="local_rank: 1 exitcode: 23",
        process_group_alive_after_exit=False,
        cleanup_signal=None,
        process_group_alive_after_cleanup=False,
    )
    uuids = ("GPU-a", "GPU-b")
    return (
        LAUNCHER._hard_failure_payload(
            2,
            ("0", "1"),
            uuids,
            ["torchrun"],
            result,
            timeout_seconds=45.0,
            worker_timeout_seconds=120.0,
        ),
        uuids,
    )


def test_stage_b_failure_artifact_binds_nccl_and_physical_devices() -> None:
    payload, uuids = _payload()
    summary = FREEZER._validate_failure(payload, 2, list(uuids))
    assert summary["backend"] == "nccl"
    assert summary["device_uuids"] == list(uuids)


@pytest.mark.parametrize("field", ("backend", "device_uuids"))
def test_stage_b_failure_binding_rejects_substitution(field: str) -> None:
    payload, uuids = _payload()
    payload["launch"][field] = "gloo" if field == "backend" else ["GPU-x"]
    payload["content_sha256"] = LAUNCHER._payload_sha256(payload)
    with pytest.raises(ValueError):
        FREEZER._validate_failure(payload, 2, list(uuids))
