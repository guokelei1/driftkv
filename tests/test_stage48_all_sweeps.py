import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_cohortkv_stage4_8_all_sweeps.py"
SPEC = importlib.util.spec_from_file_location(
    "run_cohortkv_stage4_8_all_sweeps",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def args(**overrides):
    values = {
        "devices": None,
        "force": False,
        "smoke_test": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_family_command_forwards_only_requested_options() -> None:
    script = MODULE.FAMILIES[0][1]
    command = MODULE.family_command(
        script,
        args(
            devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
            force=True,
            smoke_test=True,
        ),
    )

    assert command == [
        sys.executable,
        str(script),
        "--devices",
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
        "--force",
        "--smoke-test",
    ]


def test_run_all_waits_for_each_family_in_declared_order(
    monkeypatch,
) -> None:
    calls = []

    def run(command, cwd, check):
        calls.append((command, cwd, check))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    assert MODULE.run_all(args()) == 0
    assert [Path(value[0][1]).name for value in calls] == [
        path.name for _, path in MODULE.FAMILIES
    ]
    assert all(cwd == ROOT and check is False for _, cwd, check in calls)


def test_run_all_stops_immediately_after_a_family_failure(
    monkeypatch,
) -> None:
    returncodes = iter((0, 7))
    calls = []

    def run(command, cwd, check):
        calls.append(command)
        return subprocess.CompletedProcess(command, next(returncodes))

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    assert MODULE.run_all(args()) == 7
    assert len(calls) == 2
