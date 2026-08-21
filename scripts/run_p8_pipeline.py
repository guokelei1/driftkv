#!/usr/bin/env python3
"""Conservative status/next-step driver for the frozen P8 release chain.

This driver never selects seeds from H/S, never overwrites artifacts, and runs at
most one missing job per invocation.  It is intended for low-context monitoring.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("m0_f", "m1")
SEEDS = (17, 37, 71)
STAGES = ("r1_edge1", "r1_edge2", "r2")


@dataclass(frozen=True)
class Job:
    kind: str
    release: str
    model: str | None
    seed: int | None
    artifact: Path

    def command(self, device: str) -> list[str]:
        if self.kind == "train":
            return ["python", "scripts/train_p8_release.py", "--model", str(self.model), "--seed", str(self.seed), "--release", self.release, "--device", device]
        if self.kind == "raw":
            return ["python", "scripts/eval_p8_release_raw.py", "--model", str(self.model), "--seed", str(self.seed), "--release", self.release, "--device", device]
        if self.kind == "seal":
            return ["python", "scripts/seal_p8_release_raw.py", "--release", self.release]
        if self.kind == "adjudicate":
            return ["python", "scripts/adjudicate_p8_hs.py", "--release", self.release]
        raise ValueError(self.kind)


def result_json(path: Path) -> dict:
    return json.loads(path.read_text())


def r0_passed(root: Path = ROOT) -> bool:
    path = root / "results/p8/r0_control/adjudication_v1.json"
    return path.exists() and result_json(path).get("status") == "R0_blocking_control_passed"


def release_jobs(release: str, root: Path = ROOT) -> list[Job]:
    jobs = []
    for model in MODELS:
        for seed in SEEDS:
            jobs.append(Job("train", release, model, seed, root / f"results/p8/release_training/{release}/{model}_seed{seed}/train_result.json"))
    for model in MODELS:
        for seed in SEEDS:
            jobs.append(Job("raw", release, model, seed, root / f"results/p8/staleness_raw/{release}/{model}_seed{seed}/raw_manifest.json"))
    jobs.append(Job("seal", release, None, None, root / f"results/p8/{release}/raw_score_seal_v1.json"))
    jobs.append(Job("adjudicate", release, None, None, root / f"results/p8/{release}/hs_adjudication_v1.json"))
    return jobs


def admission_block(release: str, root: Path = ROOT) -> list[str]:
    rejected = []
    for model in MODELS:
        for seed in SEEDS:
            path = root / f"results/p8/release_training/{release}/{model}_seed{seed}/train_result.json"
            if path.exists() and not result_json(path).get("admitted", False):
                rejected.append(f"{release}/{model}/seed{seed}")
    return rejected


def next_job(root: Path = ROOT) -> tuple[Job | None, str | None]:
    if not r0_passed(root):
        return None, "BLOCKED: R0 blocking control is absent or failed"
    for stage_index, release in enumerate(STAGES):
        if stage_index:
            previous = STAGES[stage_index - 1]
            previous_result = root / f"results/p8/{previous}/hs_adjudication_v1.json"
            if not previous_result.exists():
                return None, f"BLOCKED: {previous} must be adjudicated before {release}"
            rejected = admission_block(previous, root)
            if rejected:
                return None, "BLOCKED: rejected parent release(s) require human adjudication: " + ", ".join(rejected)
        jobs = release_jobs(release, root)
        training = jobs[:6]
        raw = jobs[6:12]
        if any(not job.artifact.exists() for job in training):
            return next(job for job in training if not job.artifact.exists()), None
        rejected = admission_block(release, root)
        if rejected:
            return None, "BLOCKED: release admission failed; do not score primary S automatically: " + ", ".join(rejected)
        if any(not job.artifact.exists() for job in raw):
            return next(job for job in raw if not job.artifact.exists()), None
        if not jobs[12].artifact.exists():
            return jobs[12], None
        if not jobs[13].artifact.exists():
            return jobs[13], None
    return None, "COMPLETE: P8 H-S adjudication finished; tomography/controller remain unauthorized"


def next_wave(root: Path = ROOT) -> tuple[list[Job], str | None]:
    job, message = next_job(root)
    if job is None:
        return [], message
    if job.kind in {"seal", "adjudicate"}:
        return [job], None
    jobs = release_jobs(job.release, root)
    phase = jobs[:6] if job.kind == "train" else jobs[6:12]
    return [candidate for candidate in phase if not candidate.artifact.exists()], None


def active_p8_processes() -> list[str]:
    output = subprocess.run(["ps", "-eo", "pid,etime,args"], check=True, text=True, capture_output=True).stdout
    return [line.strip() for line in output.splitlines() if ("train_p8_release.py" in line or "eval_p8_release_raw.py" in line) and "run_p8_pipeline.py" not in line]


def status_payload(root: Path = ROOT) -> dict:
    job, message = next_job(root)
    stages = {}
    for release in STAGES:
        jobs = release_jobs(release, root)
        stages[release] = {
            "train_complete": sum(job.artifact.exists() for job in jobs[:6]),
            "raw_complete": sum(job.artifact.exists() for job in jobs[6:12]),
            "raw_sealed": jobs[12].artifact.exists(),
            "adjudicated": jobs[13].artifact.exists(),
            "rejected": admission_block(release, root),
        }
    return {
        "r0_passed": r0_passed(root), "stages": stages,
        "active_processes": active_p8_processes() if root == ROOT else [],
        "next_job": None if job is None else {
            "kind": job.kind, "release": job.release, "model": job.model, "seed": job.seed,
            "artifact": str(job.artifact),
        },
        "message": message,
        "scientific_stop_boundary": "stop after r2 H-S adjudication; no tomography/controller",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--next-command", action="store_true")
    parser.add_argument("--next-wave-commands", action="store_true")
    parser.add_argument("--run-next", action="store_true")
    parser.add_argument("--launch-wave", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    args = parser.parse_args()
    payload = status_payload()
    job, message = next_job()
    if args.run_next:
        if payload["active_processes"]:
            raise RuntimeError("refusing to launch while P8 train/eval processes are active")
        if job is None:
            print(message)
            return
        command = job.command(args.device)
        print("RUNNING:", shlex.join(["env", "PYTHONPATH=src", *command]), flush=True)
        subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": "src"}, check=True)
        return
    if args.next_wave_commands or args.launch_wave:
        if payload["active_processes"]:
            raise RuntimeError("refusing to emit or launch a wave while P8 train/eval processes are active")
        wave, wave_message = next_wave()
        if not wave:
            print(wave_message)
            return
        devices = [value.strip() for value in args.devices.split(",") if value.strip()]
        if not devices:
            raise ValueError("--devices must contain at least one device")
        selected = wave[: len(devices)] if wave[0].kind in {"train", "raw"} else wave[:1]
        commands = [job.command(devices[index % len(devices)]) for index, job in enumerate(selected)]
        if args.next_wave_commands:
            for command in commands:
                print(shlex.join(["env", "PYTHONPATH=src", *command]))
            return
        log_root = ROOT / "results/p8/monitor_logs"
        log_root.mkdir(parents=True, exist_ok=True)
        launched = []
        for job, command in zip(selected, commands, strict=True):
            suffix = f"_{job.model}_seed{job.seed}" if job.model is not None else ""
            log_path = log_root / f"{job.release}_{job.kind}{suffix}.log"
            stream = log_path.open("ab")
            process = subprocess.Popen(
                command, cwd=ROOT, env={**os.environ, "PYTHONPATH": "src"},
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            )
            stream.close()
            launched.append({"pid": process.pid, "job": f"{job.release}/{job.kind}/{job.model}/seed{job.seed}", "log": str(log_path)})
        print(json.dumps({"launched": launched}, indent=2))
        return
    if args.next_command and payload["active_processes"]:
        raise RuntimeError("refusing to emit a launch command while P8 train/eval processes are active")
    if args.next_command and job is not None:
        print(shlex.join(["env", "PYTHONPATH=src", *job.command(args.device)]))
        return
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
