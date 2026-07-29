from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from hstu_kvcache.migration.design2_dev_wave import D2_DEV_WAVE_PROTOCOL
from hstu_kvcache.migration.design2_plan import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/run_cohortkv_design2_dev_c0.py"
DEFAULT_OUTPUT_DIR = "configs/cohortkv_d2/development"
STATUS_PROTOCOL = "cohortkv_d2_dev_c0_status_v1"
COMPLETE_WORLD_SIZES = (1, 2, 3)
COMPLETE_ABORT_WORLD_SIZES = (3,)


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    timed_out: bool
    stdout: str
    stderr: str
    cleanup_signal: str | None
    process_group_alive_after_cleanup: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world-sizes",
        nargs="+",
        type=int,
        choices=(1, 2, 3),
        default=(1, 2, 3),
    )
    parser.add_argument(
        "--pre-commit-abort-world-sizes",
        nargs="+",
        type=int,
        choices=(1, 2, 3),
        default=(3,),
    )
    parser.add_argument("--visible-devices", nargs="+")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--worker-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--launch-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--termination-grace-seconds", type=float, default=5.0)
    return parser.parse_args(argv)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _unique(values: list[object] | tuple[object, ...]) -> tuple[object, ...]:
    prepared = tuple(values)
    if len(set(prepared)) != len(prepared):
        raise ValueError("D2 dev launcher values must be unique")
    return prepared


def _devices(values: list[str] | None) -> tuple[str, ...]:
    if values is None:
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        if inherited is not None:
            values = [inherited]
    if values is None:
        return ("0", "1", "3")
    devices = tuple(
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    )
    if len(set(devices)) != len(devices):
        raise ValueError("D2 dev visible devices must be unique")
    return devices


def _is_complete_matrix(
    world_sizes: tuple[int, ...],
    abort_world_sizes: tuple[int, ...],
) -> bool:
    return (
        world_sizes == COMPLETE_WORLD_SIZES
        and abort_world_sizes == COMPLETE_ABORT_WORLD_SIZES
    )


def _torchrun() -> str:
    executable = shutil.which("torchrun")
    if executable is None:
        raise FileNotFoundError("torchrun is unavailable")
    return executable


def _tail(value: str, lines: int = 100) -> str:
    return "\n".join(value.splitlines()[-lines:])[-30_000:]


def _artifact_path(
    output_dir: Path,
    world_size: int,
    case: str,
) -> Path:
    return output_dir / f"dev_c0_w{world_size}_{case}.json"


def _temporary_path(
    output_dir: Path,
    world_size: int,
    case: str,
) -> Path:
    return output_dir / (
        f".dev_c0_w{world_size}_{case}.{uuid.uuid4().hex}.run.json"
    )


def _validate(
    path: Path,
    world_size: int,
    case: str,
) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError("D2 dev C0 worker produced no artifact")
    value = json.loads(path.read_text())
    expected_content_sha256 = canonical_sha256(
        {
            key: item
            for key, item in value.items()
            if key != "content_sha256"
        }
    )
    if (
        value.get("protocol") != D2_DEV_WAVE_PROTOCOL
        or value.get("status") != "complete"
        or value.get("case") != case
        or value.get("scientific_result") is not False
        or value.get("formal_stage_c") is not False
        or value.get("configuration", {}).get("world_size") != world_size
        or value.get("content_sha256") != expected_content_sha256
        or not value.get("epoch_integration", {}).get("connected")
        or not value.get("checks")
        or not all(value["checks"].values())
    ):
        raise RuntimeError("D2 dev C0 artifact failed validation")
    if value["scope"]["target_epoch_published"] is not False:
        raise RuntimeError("D2 dev artifact made a formal publication claim")
    target_visible = value["scope"][
        "development_target_pointer_published"
    ]
    if target_visible != (case == "normal"):
        raise RuntimeError("D2 dev C0 visibility differs from case")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _process_group_alive(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(
    process: subprocess.Popen[str],
    grace_seconds: float,
) -> tuple[str, str, str]:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        stdout, stderr = process.communicate()
        return "none", stdout, stderr
    try:
        stdout, stderr = process.communicate(timeout=grace_seconds)
        return "SIGTERM", stdout, stderr
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
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
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    cleanup_signal = None
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        cleanup_signal, stdout, stderr = _terminate_process_group(
            process,
            grace_seconds,
        )
    if _process_group_alive(process.pid):
        cleanup_signal, extra_stdout, extra_stderr = (
            _terminate_process_group(process, grace_seconds)
        )
        stdout += extra_stdout
        stderr += extra_stderr
    alive_after_cleanup = _process_group_alive(process.pid)
    return ProcessResult(
        returncode=(
            process.returncode if process.returncode is not None else -1
        ),
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        cleanup_signal=cleanup_signal,
        process_group_alive_after_cleanup=alive_after_cleanup,
    )


def _run(
    torchrun: str,
    world_size: int,
    case: str,
    devices: tuple[str, ...],
    output_dir: Path,
    worker_timeout_seconds: float,
    launch_timeout_seconds: float,
    termination_grace_seconds: float,
) -> tuple[Path, dict[str, object]]:
    if len(devices) < world_size:
        raise ValueError(
            f"D2 dev C0 W{world_size} needs {world_size} visible devices"
        )
    selected = devices[:world_size]
    temporary = _temporary_path(output_dir, world_size, case)
    target = _artifact_path(output_dir, world_size, case)
    temporary.parent.mkdir(parents=True, exist_ok=True)
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
        "--output",
        str(temporary),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    environment["PYTHONUNBUFFERED"] = "1"
    result = _run_process(
        command,
        environment,
        launch_timeout_seconds,
        termination_grace_seconds,
    )
    if (
        result.returncode != 0
        or result.timed_out
        or result.process_group_alive_after_cleanup
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"D2 dev C0 W{world_size} {case} failed "
            f"returncode={result.returncode} "
            f"timed_out={result.timed_out} "
            f"residual={result.process_group_alive_after_cleanup}\n"
            f"{_tail(result.stderr)}"
        )
    try:
        _validate(temporary, world_size, case)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"W{world_size} {case} complete "
        f"devices={','.join(selected)} "
        f"artifact={target}",
        flush=True,
    )
    return target, {
        "world_size": world_size,
        "case": case,
        "visible_devices": list(selected),
        "timed_out": result.timed_out,
        "cleanup_signal": result.cleanup_signal,
        "process_group_alive_after_cleanup": (
            result.process_group_alive_after_cleanup
        ),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    world_sizes = tuple(
        sorted(int(value) for value in _unique(args.world_sizes))
    )
    abort_world_sizes = tuple(
        sorted(
            int(value)
            for value in _unique(args.pre_commit_abort_world_sizes)
        )
    )
    devices = _devices(args.visible_devices)
    torchrun = _torchrun()
    output_dir = _path(args.output_dir)
    complete_matrix = _is_complete_matrix(
        world_sizes,
        abort_world_sizes,
    )
    if (
        output_dir.resolve() == _path(DEFAULT_OUTPUT_DIR).resolve()
        and not complete_matrix
    ):
        raise ValueError(
            "canonical D2 dev C0 output requires the complete matrix; "
            "use a separate --output-dir for partial diagnostics"
        )
    outputs = []
    launch_evidence = []
    for world_size in world_sizes:
        output, evidence = _run(
            torchrun,
            world_size,
            "normal",
            devices,
            output_dir,
            args.worker_timeout_seconds,
            args.launch_timeout_seconds,
            args.termination_grace_seconds,
        )
        outputs.append(output)
        launch_evidence.append(evidence)
    for world_size in abort_world_sizes:
        output, evidence = _run(
            torchrun,
            world_size,
            "pre_commit_abort",
            devices,
            output_dir,
            args.worker_timeout_seconds,
            args.launch_timeout_seconds,
            args.termination_grace_seconds,
        )
        outputs.append(output)
        launch_evidence.append(evidence)
    artifact_descriptors = []
    for path in outputs:
        value = json.loads(path.read_text())
        artifact_descriptors.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _file_sha256(path),
                "content_sha256": value["content_sha256"],
                "world_size": value["configuration"]["world_size"],
                "case": value["case"],
                "status": value["status"],
            }
        )
    status: dict[str, object] = {
        "protocol": STATUS_PROTOCOL,
        "status": "complete" if complete_matrix else "partial",
        "scientific_result": False,
        "formal_stage_c": False,
        "gates": {
            "stage_c_development_entry": (
                "go" if complete_matrix else "unchanged_by_partial_run"
            ),
            "stage_c_evaluation_entry": "blocked",
            "formal_stage_b_freeze": "blocked",
        },
        "completed": {
            "complete_development_matrix": complete_matrix,
            "sample_integrated_wave": True,
            "record_owner_strict_cow_lpt_world_sizes": list(world_sizes),
            "normal_epoch_publication_world_sizes": list(world_sizes),
            "pre_commit_abort_world_sizes": list(abort_world_sizes),
            "source_fixture_unchanged_before_decision": True,
            "abort_private_target_references_released": True,
        },
        "pending": [
            "formal physical W4 Stage B normal evidence",
            "formal physical W4 Stage B hard-failure evidence",
            "Stage B summary freeze and check",
            "formal Stage C protocol freeze",
            "full-682 integrated-wave evidence",
            "paper-comparable timing and capacity evidence",
            "independent one-shot target-exact numerical reference closure",
            "zero-delta compiled or scheduled append branch closure",
        ],
        "unsupported_claims": [
            "W3 substitutes for formal W4",
            "development pointer is a formal target epoch publication",
            "source fixture release proves HBM allocator reclaim",
            "sample execution is a full-cohort result",
            "development artifact is a timing result",
        ],
        "artifacts": artifact_descriptors,
        "launch_evidence": launch_evidence,
        "checks": {
            "complete_development_matrix": complete_matrix,
            "all_launches_completed": len(outputs)
            == len(world_sizes) + len(abort_world_sizes),
            "no_residual_process_groups": all(
                not value["process_group_alive_after_cleanup"]
                for value in launch_evidence
            ),
            "no_launcher_timeouts": all(
                not value["timed_out"] for value in launch_evidence
            ),
        },
    }
    status["content_sha256"] = canonical_sha256(status)
    status_path = output_dir / "c0_status.json"
    _write_json_atomic(status_path, status)
    print(
        json.dumps(
            {
                "status": status["status"],
                "scientific_result": False,
                "formal_stage_c": False,
                "artifacts": [str(value) for value in outputs],
                "status_artifact": str(status_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
