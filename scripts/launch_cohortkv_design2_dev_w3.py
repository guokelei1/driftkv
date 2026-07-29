from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from hstu_kvcache.migration.design2_plan import (
    canonical_sha256,
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/run_cohortkv_design2_distributed_tests.py"
DEFAULT_SAMPLE_INPUTS = "configs/cohortkv_d2/dev_w3_sample_inputs.json"
DEFAULT_OUTPUT_DIR = "configs/cohortkv_d2"
SAMPLE_PROTOCOL = "cohortkv_d2_dev_w3_sample_inputs_v1"
WORKER_PROTOCOL = "cohortkv_d2_stage_b_distributed_primitives_v1"
NORMAL_PROTOCOL = "cohortkv_d2_dev_w3_distributed_diagnostic_v1"
HARD_FAILURE_PROTOCOL = "cohortkv_d2_dev_w3_hard_failure_v1"
WORLD_SIZE = 3


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    elapsed_seconds: float
    timed_out: bool
    stdout: str
    stderr: str
    process_group_alive_after_exit: bool
    cleanup_signal: str | None
    process_group_alive_after_cleanup: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("normal", "hard_failure"),
        default=("normal", "hard_failure"),
    )
    parser.add_argument(
        "--visible-devices",
        nargs="+",
        default=("0", "1", "3"),
    )
    parser.add_argument("--sample-inputs", default=DEFAULT_SAMPLE_INPUTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=120.0,
    )
    parser.add_argument(
        "--normal-timeout-seconds",
        type=float,
        default=600.0,
    )
    parser.add_argument(
        "--hard-failure-timeout-seconds",
        type=float,
        default=45.0,
    )
    parser.add_argument(
        "--termination-grace-seconds",
        type=float,
        default=5.0,
    )
    return parser.parse_args(argv)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _unique_cases(
    values: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    prepared = tuple(values)
    if len(set(prepared)) != len(prepared):
        raise ValueError("D2 W3 development cases must be unique")
    return prepared


def _parse_devices(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    devices = tuple(
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    )
    if len(devices) != WORLD_SIZE or len(set(devices)) != WORLD_SIZE:
        raise ValueError(
            "D2 W3 development diagnostic requires three unique devices"
        )
    return devices


def _torchrun() -> str:
    executable = shutil.which("torchrun")
    if executable is None:
        raise FileNotFoundError("torchrun is unavailable")
    return executable


def _command(
    torchrun: str,
    case: str,
    worker_timeout_seconds: float,
    sample_inputs: Path,
    output: Path | None,
) -> list[str]:
    command = [
        torchrun,
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(WORKER),
        "--case",
        case,
        "--timeout-seconds",
        str(worker_timeout_seconds),
        "--sample-inputs",
        str(sample_inputs),
    ]
    if output is not None:
        command.extend(("--output", str(output)))
    return command


def _process_group_alive(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(
    process_group: int,
    process: subprocess.Popen[str],
    grace_seconds: float,
) -> tuple[str, str, str]:
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        stdout, stderr = process.communicate()
        return "none", stdout, stderr
    try:
        stdout, stderr = process.communicate(timeout=grace_seconds)
        return "SIGTERM", stdout, stderr
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        return "SIGKILL", stdout, stderr


def _run_process(
    command: list[str],
    environment: dict[str, str],
    timeout_seconds: float,
    grace_seconds: float,
) -> ProcessResult:
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    cleanup_signal = None
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        cleanup_signal, stdout, stderr = _terminate_process_group(
            process.pid,
            process,
            grace_seconds,
        )
    alive_after_exit = _process_group_alive(process.pid)
    if alive_after_exit:
        cleanup_signal, extra_stdout, extra_stderr = (
            _terminate_process_group(
                process.pid,
                process,
                grace_seconds,
            )
        )
        stdout += extra_stdout
        stderr += extra_stderr
    alive_after_cleanup = _process_group_alive(process.pid)
    elapsed = time.perf_counter() - started
    return ProcessResult(
        returncode=(
            process.returncode if process.returncode is not None else -1
        ),
        elapsed_seconds=elapsed,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        process_group_alive_after_exit=alive_after_exit,
        cleanup_signal=cleanup_signal,
        process_group_alive_after_cleanup=alive_after_cleanup,
    )


def _tail(value: str, lines: int = 80, characters: int = 20_000) -> str:
    prepared = "\n".join(value.splitlines()[-lines:])
    return prepared[-characters:]


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _payload_sha256(value: dict[str, object]) -> str:
    return canonical_sha256(
        {
            key: item
            for key, item in value.items()
            if key != "content_sha256"
        }
    )


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _environment(devices: tuple[str, ...]) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _normal_artifact_path(output_dir: Path) -> Path:
    return output_dir / "dev_w3_primitives.json"


def _hard_failure_artifact_path(output_dir: Path) -> Path:
    return output_dir / "dev_w3_hard_failure.json"


def _temporary_normal_path(output_dir: Path) -> Path:
    return output_dir / (
        f".dev_w3_primitives.{uuid.uuid4().hex}.run.json"
    )


def _validate_sample(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            "D2 W3 development sample is missing; materialize it first"
        )
    value = json.loads(path.read_text())
    expected_content = canonical_sha256(
        {
            key: item
            for key, item in value.items()
            if key != "content_sha256"
        }
    )
    diagnostic = value.get("development_diagnostic", {})
    if (
        value.get("protocol") != SAMPLE_PROTOCOL
        or value.get("status") != "complete"
        or value.get("scientific_result") is not False
        or value.get("content_sha256") != expected_content
        or str(WORLD_SIZE) not in value.get("selections", {})
        or diagnostic.get("formal_stage_b_gate") is not False
        or diagnostic.get("substitute_for_w4") is not False
    ):
        raise RuntimeError("D2 W3 development sample failed validation")
    return value


def _validate_worker_artifact(
    path: Path,
    sample_path: Path,
) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError("D2 W3 normal worker produced no artifact")
    value = json.loads(path.read_text())
    if (
        value.get("protocol") != WORKER_PROTOCOL
        or value.get("status") != "complete"
        or value.get("scientific_result") is not False
        or value.get("configuration", {}).get("world_size") != WORLD_SIZE
        or value.get("sample_inputs", {}).get("sha256")
        != file_sha256(sample_path)
        or not value.get("checks")
        or not all(value["checks"].values())
    ):
        raise RuntimeError("D2 W3 normal worker artifact failed validation")
    return value


def _decorate_normal_artifact(
    value: dict[str, object],
    devices: tuple[str, ...],
    sample_path: Path,
    command: list[str],
    result: ProcessResult,
) -> dict[str, object]:
    value["protocol"] = NORMAL_PROTOCOL
    value["worker_protocol"] = WORKER_PROTOCOL
    value["scientific_result"] = False
    value["development_diagnostic"] = {
        "world_size": WORLD_SIZE,
        "formal_stage_b_gate": False,
        "substitute_for_w4": False,
        "stage_c_development_evidence": True,
        "stage_c_evaluation_authorized": False,
        "formal_stage_b_summary_eligible": False,
        "pending_formal_evidence": [
            "stage_b_w4_primitives.json",
            "stage_b_w4_hard_failure.json",
            "stage_b_summary.json",
        ],
    }
    value["development_launch"] = {
        "command": command,
        "visible_devices": list(devices),
        "sample_inputs": str(
            sample_path.relative_to(ROOT)
            if sample_path.is_relative_to(ROOT)
            else sample_path
        ),
        "elapsed_seconds": result.elapsed_seconds,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "process_group_alive_after_cleanup": (
            result.process_group_alive_after_cleanup
        ),
        "stdout_sha256": _text_sha256(result.stdout),
        "stderr_sha256": _text_sha256(result.stderr),
    }
    value["scope"]["development_w3_only"] = True
    value["scope"]["formal_w4_topology_covered"] = False
    value["scope"]["formal_stage_b_gate"] = False
    value["content_sha256"] = _payload_sha256(value)
    return value


def _run_normal(
    torchrun: str,
    devices: tuple[str, ...],
    sample_path: Path,
    output_dir: Path,
    worker_timeout_seconds: float,
    timeout_seconds: float,
    grace_seconds: float,
) -> None:
    target = _normal_artifact_path(output_dir)
    temporary = _temporary_normal_path(output_dir)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    command = _command(
        torchrun,
        "normal",
        worker_timeout_seconds,
        sample_path,
        temporary,
    )
    result = _run_process(
        command,
        _environment(devices),
        timeout_seconds,
        grace_seconds,
    )
    if (
        result.returncode != 0
        or result.timed_out
        or result.process_group_alive_after_cleanup
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "D2 W3 development normal failed "
            f"returncode={result.returncode} "
            f"timed_out={result.timed_out}\n"
            f"{_tail(result.stderr)}"
        )
    try:
        value = _validate_worker_artifact(temporary, sample_path)
        value = _decorate_normal_artifact(
            value,
            devices,
            sample_path,
            command,
            result,
        )
        _write_json_atomic(target, value)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        "W3 development normal complete "
        f"elapsed_seconds={result.elapsed_seconds:.3f} "
        f"artifact={target.relative_to(ROOT)}",
        flush=True,
    )


def _rank_one_exit_23(value: str) -> bool:
    rank_one = re.search(
        r"(?:local_)?rank\s*(?::|=)\s*1\b",
        value,
        flags=re.IGNORECASE,
    )
    exit_23 = re.search(
        r"exitcode\s*(?::|=)\s*23\b",
        value,
        flags=re.IGNORECASE,
    )
    return rank_one is not None and exit_23 is not None


def _hard_failure_payload(
    devices: tuple[str, ...],
    command: list[str],
    result: ProcessResult,
    timeout_seconds: float,
    worker_timeout_seconds: float,
) -> dict[str, object]:
    combined = f"{result.stdout}\n{result.stderr}"
    checks = {
        "rank1_exit_23_observed": _rank_one_exit_23(combined),
        "launcher_returncode_nonzero": result.returncode != 0,
        "subprocess_did_not_timeout": not result.timed_out,
        "peer_termination_bounded": (
            not result.timed_out
            and result.elapsed_seconds <= timeout_seconds
            and not result.process_group_alive_after_cleanup
        ),
    }
    payload: dict[str, object] = {
        "protocol": HARD_FAILURE_PROTOCOL,
        "status": "complete" if all(checks.values()) else "failed",
        "scientific_result": False,
        "development_diagnostic": {
            "world_size": WORLD_SIZE,
            "formal_stage_b_gate": False,
            "substitute_for_w4": False,
            "stage_c_development_evidence": True,
            "stage_c_evaluation_authorized": False,
            "formal_stage_b_summary_eligible": False,
        },
        "scope": {
            "failure_propagation_only": True,
            "performance_result": False,
            "target_epoch_published": False,
            "formal_w4_topology_covered": False,
        },
        "world_size": WORLD_SIZE,
        "case": "hard_failure",
        "injected_failure": {
            "rank": 1,
            "mechanism": "os._exit",
            "exit_code": 23,
        },
        "launch": {
            "command": command,
            "visible_devices": list(devices),
            "worker_timeout_seconds": worker_timeout_seconds,
            "subprocess_timeout_seconds": timeout_seconds,
            "returncode": result.returncode,
            "elapsed_seconds": result.elapsed_seconds,
            "timed_out": result.timed_out,
            "process_group_alive_after_exit": (
                result.process_group_alive_after_exit
            ),
            "cleanup_signal": result.cleanup_signal,
            "process_group_alive_after_cleanup": (
                result.process_group_alive_after_cleanup
            ),
            "stdout_sha256": _text_sha256(result.stdout),
            "stderr_sha256": _text_sha256(result.stderr),
            "stdout_tail": _tail(result.stdout),
            "stderr_tail": _tail(result.stderr),
        },
        "checks": checks,
    }
    payload["content_sha256"] = _payload_sha256(payload)
    return payload


def _run_hard_failure(
    torchrun: str,
    devices: tuple[str, ...],
    sample_path: Path,
    output_dir: Path,
    worker_timeout_seconds: float,
    timeout_seconds: float,
    grace_seconds: float,
) -> None:
    command = _command(
        torchrun,
        "hard_failure",
        worker_timeout_seconds,
        sample_path,
        None,
    )
    result = _run_process(
        command,
        _environment(devices),
        timeout_seconds,
        grace_seconds,
    )
    payload = _hard_failure_payload(
        devices,
        command,
        result,
        timeout_seconds,
        worker_timeout_seconds,
    )
    target = _hard_failure_artifact_path(output_dir)
    _write_json_atomic(target, payload)
    if payload["status"] != "complete":
        raise RuntimeError(
            "D2 W3 development hard failure check failed: "
            f"{payload['checks']}"
        )
    print(
        "W3 development hard_failure complete "
        f"elapsed_seconds={result.elapsed_seconds:.3f} "
        f"artifact={target.relative_to(ROOT)}",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    cases = _unique_cases(args.cases)
    if min(
        args.worker_timeout_seconds,
        args.normal_timeout_seconds,
        args.hard_failure_timeout_seconds,
        args.termination_grace_seconds,
    ) <= 0:
        raise ValueError(
            "D2 W3 development launcher timeouts must be positive"
        )
    devices = _parse_devices(args.visible_devices)
    sample_path = _path(args.sample_inputs)
    _validate_sample(sample_path)
    output_dir = _path(args.output_dir)
    torchrun = _torchrun()
    for case in cases:
        print(
            f"W3 development {case} start "
            f"visible_devices={','.join(devices)}",
            flush=True,
        )
        if case == "normal":
            _run_normal(
                torchrun,
                devices,
                sample_path,
                output_dir,
                args.worker_timeout_seconds,
                args.normal_timeout_seconds,
                args.termination_grace_seconds,
            )
        else:
            _run_hard_failure(
                torchrun,
                devices,
                sample_path,
                output_dir,
                args.worker_timeout_seconds,
                args.hard_failure_timeout_seconds,
                args.termination_grace_seconds,
            )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
