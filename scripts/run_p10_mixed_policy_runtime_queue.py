#!/usr/bin/env python3
"""Run the ten frozen P10.2 mixed-policy runtime jobs on physical GPUs 0/1."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p10/mixed_policy_runtime/full"


@dataclass(frozen=True)
class Job:
    release: str
    model: str
    seed: int
    sample: float
    budget: float

    @property
    def name(self) -> str:
        return f"{self.release}_{self.model}_seed{self.seed}_sample{int(round(self.sample*100)):02d}_budget{int(round(self.budget*100)):02d}"


def jobs() -> list[Job]:
    output = []
    for sample in (0.01, 0.02):
        for budget in (0.05, 0.10, 0.25):
            output.append(Job("r2", "m1", 17, sample, budget))
        output.append(Job("r2", "m0_f", 17, sample, 0.10))
        output.append(Job("r1_edge2", "m1", 17, sample, 0.10))
    return output


def run(job: Job, physical_gpu: int) -> tuple[str, int]:
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
        "--device", "cuda:0", "--output", str(output),
    ]
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return job.name, result.returncode


def worker(queue: list[Job], physical_gpu: int) -> list[tuple[str, int]]:
    rows = []
    for job in queue:
        row = run(job, physical_gpu)
        print(f"gpu{physical_gpu} {row[0]} {'passed' if row[1] == 0 else f'failed({row[1]})'}", flush=True)
        rows.append(row)
        if row[1]:
            break
    return rows


def main() -> None:
    ledger = jobs()
    queues = [ledger[0::2], ledger[1::2]]
    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, queues[index], index) for index in (0, 1)]
        for future in as_completed(futures):
            failures.extend(name for name, code in future.result() if code)
    if failures:
        raise SystemExit(f"P10.2 runtime failed: {failures}")


if __name__ == "__main__":
    main()
