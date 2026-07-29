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

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/run_cohortkv_design2_distributed_tests.py"
DEFAULT_OUTPUT_DIR = "configs/cohortkv_d2"
NORMAL_PROTOCOL = "cohortkv_d2_stage_b_distributed_primitives_v1"
HARD_FAILURE_PROTOCOL = "cohortkv_d2_stage_b_hard_failure_v1"


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
        "--world-sizes",
        nargs="+",
        type=int,
        choices=(1, 2, 4),
        default=(1, 2, 4),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("normal", "hard_failure"),
        default=("normal", "hard_failure"),
    )
    parser.add_argument("--visible-devices", nargs="+")
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


def _unique_values(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    prepared = tuple(values)
    if len(set(prepared)) != len(prepared):
        raise ValueError("D2 Stage B world sizes must be unique")
    return tuple(sorted(prepared))


def _unique_cases(
    values: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    prepared = tuple(values)
    if len(set(prepared)) != len(prepared):
        raise ValueError("D2 Stage B cases must be unique")
    return prepared


def _parse_devices(values: list[str] | None) -> tuple[str, ...]:
    if values is None:
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        if inherited is None:
            return ()
        values = [inherited]
    devices = tuple(
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    )
    if len(set(devices)) != len(devices):
        raise ValueError("D2 Stage B visible devices must be unique")
    return devices


def _resolve_device_uuids(
    devices: tuple[str, ...],
) -> tuple[str, ...]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    by_token = {}
    for line in result.stdout.splitlines():
        index, uuid_value = (value.strip() for value in line.split(",", 1))
        by_token[index] = uuid_value
        by_token[uuid_value] = uuid_value
    try:
        resolved = tuple(by_token[value] for value in devices)
    except KeyError as error:
        raise ValueError(
            f"D2 Stage B device token is unresolved: {error.args[0]}"
        ) from error
    if len(set(resolved)) != len(resolved):
        raise ValueError("D2 Stage B physical devices must be unique")
    return resolved


def _selected_devices(
    devices: tuple[str, ...],
    world_size: int,
) -> tuple[str, ...]:
    if devices:
        if len(devices) < world_size:
            raise ValueError(
                f"D2 Stage B W{world_size} needs {world_size} visible devices"
            )
        return devices[:world_size]
    return tuple(str(rank) for rank in range(world_size))


def _torchrun() -> str:
    executable = shutil.which("torchrun")
    if executable is None:
        raise FileNotFoundError("torchrun is unavailable")
    return executable


def _command(
    torchrun: str,
    world_size: int,
    case: str,
    worker_timeout_seconds: float,
    output: Path | None,
) -> list[str]:
    command = [
        torchrun,
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={world_size}",
        str(WORKER),
        "--case",
        case,
        "--timeout-seconds",
        str(worker_timeout_seconds),
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
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _environment(devices: tuple[str, ...]) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _normal_artifact_path(
    output_dir: Path,
    world_size: int,
) -> Path:
    return output_dir / f"stage_b_w{world_size}_primitives.json"


def _hard_failure_artifact_path(
    output_dir: Path,
    world_size: int,
) -> Path:
    return output_dir / f"stage_b_w{world_size}_hard_failure.json"


def _temporary_normal_path(
    output_dir: Path,
    world_size: int,
) -> Path:
    return output_dir / (
        f".stage_b_w{world_size}_primitives."
        f"{uuid.uuid4().hex}.run.json"
    )


def _validate_normal_artifact(
    path: Path,
    world_size: int,
) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError("D2 Stage B normal worker produced no artifact")
    value = json.loads(path.read_text())
    if (
        value.get("protocol") != NORMAL_PROTOCOL
        or value.get("status") != "complete"
        or value.get("scientific_result") is not False
        or value.get("configuration", {}).get("world_size") != world_size
        or not value.get("checks")
        or not all(value["checks"].values())
    ):
        raise RuntimeError("D2 Stage B normal artifact failed validation")
    return value


def _run_normal(
    torchrun: str,
    world_size: int,
    devices: tuple[str, ...],
    output_dir: Path,
    worker_timeout_seconds: float,
    timeout_seconds: float,
    grace_seconds: float,
) -> None:
    target = _normal_artifact_path(output_dir, world_size)
    temporary = _temporary_normal_path(output_dir, world_size)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    command = _command(
        torchrun,
        world_size,
        "normal",
        worker_timeout_seconds,
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
            f"D2 Stage B W{world_size} normal failed "
            f"returncode={result.returncode} "
            f"timed_out={result.timed_out}\n"
            f"{_tail(result.stderr)}"
        )
    try:
        _validate_normal_artifact(temporary, world_size)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"W{world_size} normal complete "
        f"elapsed_seconds={result.elapsed_seconds:.3f} "
        f"artifact={target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}",
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
    world_size: int,
    devices: tuple[str, ...],
    device_uuids: tuple[str, ...],
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
        "scope": {
            "failure_propagation_only": True,
            "performance_result": False,
            "target_epoch_published": False,
        },
        "world_size": world_size,
        "case": "hard_failure",
        "injected_failure": {
            "rank": 1,
            "mechanism": "os._exit",
            "exit_code": 23,
        },
        "launch": {
            "command": command,
            "backend": "nccl",
            "visible_devices": list(devices),
            "device_uuids": list(device_uuids),
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
    world_size: int,
    devices: tuple[str, ...],
    output_dir: Path,
    worker_timeout_seconds: float,
    timeout_seconds: float,
    grace_seconds: float,
) -> None:
    device_uuids = _resolve_device_uuids(devices)
    command = _command(
        torchrun,
        world_size,
        "hard_failure",
        worker_timeout_seconds,
        None,
    )
    result = _run_process(
        command,
        _environment(devices),
        timeout_seconds,
        grace_seconds,
    )
    payload = _hard_failure_payload(
        world_size,
        devices,
        device_uuids,
        command,
        result,
        timeout_seconds,
        worker_timeout_seconds,
    )
    target = _hard_failure_artifact_path(output_dir, world_size)
    _write_json_atomic(target, payload)
    if payload["status"] != "complete":
        raise RuntimeError(
            f"D2 Stage B W{world_size} hard failure check failed: "
            f"{payload['checks']}"
        )
    print(
        f"W{world_size} hard_failure complete "
        f"elapsed_seconds={result.elapsed_seconds:.3f} "
        f"artifact={target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    world_sizes = _unique_values(args.world_sizes)
    cases = _unique_cases(args.cases)
    if min(
        args.worker_timeout_seconds,
        args.normal_timeout_seconds,
        args.hard_failure_timeout_seconds,
        args.termination_grace_seconds,
    ) <= 0:
        raise ValueError("D2 Stage B launcher timeouts must be positive")
    devices = _parse_devices(args.visible_devices)
    maximum_world_size = max(world_sizes)
    if devices and len(devices) < maximum_world_size:
        raise ValueError(
            "D2 Stage B visible devices do not cover selected world sizes"
        )
    output_dir = _path(args.output_dir)
    torchrun = _torchrun()
    jobs = [
        (world_size, case)
        for world_size in world_sizes
        for case in cases
        if not (world_size == 1 and case == "hard_failure")
    ]
    if not jobs:
        raise ValueError("D2 Stage B launcher has no valid jobs")
    for world_size, case in jobs:
        selected = _selected_devices(devices, world_size)
        print(
            f"W{world_size} {case} start "
            f"visible_devices={','.join(selected)}",
            flush=True,
        )
        if case == "normal":
            _run_normal(
                torchrun,
                world_size,
                selected,
                output_dir,
                args.worker_timeout_seconds,
                args.normal_timeout_seconds,
                args.termination_grace_seconds,
            )
        else:
            _run_hard_failure(
                torchrun,
                world_size,
                selected,
                output_dir,
                args.worker_timeout_seconds,
                args.hard_failure_timeout_seconds,
                args.termination_grace_seconds,
            )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
