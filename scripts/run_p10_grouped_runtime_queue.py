#!/usr/bin/env python3
"""Replay the frozen ten P10 policies with semantics-preserving grouped batching."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess

import run_p10_mixed_policy_runtime_queue as reference


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p10/mixed_policy_runtime/grouped"


def run(job: reference.Job, physical_gpu: int) -> tuple[str, int]:
    output = OUTPUT / job.name
    if (output / "result.json").exists():
        return job.name, 0
    log = OUTPUT / "logs" / f"{job.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:scripts"
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    command = [
        "python", "scripts/eval_p10_mixed_policy_runtime.py",
        "--release", job.release, "--model", job.model, "--seed", str(job.seed),
        "--sample-fraction", str(job.sample), "--budget-fraction", str(job.budget),
        "--batching-mode", "grouped", "--device", "cuda:0", "--output", str(output),
    ]
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return job.name, result.returncode


def worker(queue: list[reference.Job], physical_gpu: int) -> list[tuple[str, int]]:
    rows = []
    for job in queue:
        row = run(job, physical_gpu)
        print(f"gpu{physical_gpu} {row[0]} {'passed' if row[1] == 0 else f'failed({row[1]})'}", flush=True)
        rows.append(row)
        if row[1]:
            break
    return rows


def main() -> None:
    ledger = reference.jobs()
    queues = [ledger[0::2], ledger[1::2]]
    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, queues[index], index) for index in (0, 1)]
        for future in as_completed(futures):
            failures.extend(name for name, code in future.result() if code)
    if failures:
        raise SystemExit(f"P10.5 grouped runtime failed: {failures}")


if __name__ == "__main__":
    main()
