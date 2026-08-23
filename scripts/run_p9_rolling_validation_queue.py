#!/usr/bin/env python3
"""Run pending P9.5 cells concurrently on the frozen GPU 0/1 allowlist."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p9/rolling_validation_raw"


def jobs() -> list[tuple[str, str, int]]:
    return [
        (release, model, seed)
        for release in ("r0", "r1_edge1", "r1_edge2", "r2")
        for model in ("m0_f", "m1")
        for seed in (17, 37, 71)
    ]


def run(job: tuple[str, str, int], device: int, threads: int) -> tuple[tuple[str, str, int], int]:
    release, model, seed = job
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:scripts"
    command = [
        "python", "scripts/eval_p9_rolling_validation_cell.py",
        "--release", release, "--model", model, "--seed", str(seed),
        "--device", f"cuda:{device}", "--threads", str(threads),
    ]
    log = OUTPUT / "logs" / f"{release}_{model}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return job, result.returncode


def run_device_queue(
    assigned: list[tuple[str, str, int]], device: int, threads: int
) -> list[tuple[tuple[str, str, int], int]]:
    results = []
    for job in assigned:
        result = run(job, device, threads)
        print(result[0], "passed" if result[1] == 0 else f"failed({result[1]})", flush=True)
        results.append(result)
        if result[1]:
            break
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads-per-job", type=int, default=12)
    args = parser.parse_args()
    pending = [job for job in jobs() if not (OUTPUT / job[0] / f"{job[1]}_seed{job[2]}" / "result.json").exists()]
    failures = []
    queues = [pending[0::2], pending[1::2]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(run_device_queue, queues[device], device, args.threads_per_job): device
            for device in (0, 1)
        }
        for future in as_completed(futures):
            for job, code in future.result():
                if code:
                    failures.append(job)
    if failures:
        raise SystemExit(f"P9.5 failed cells: {failures}")


if __name__ == "__main__":
    main()
