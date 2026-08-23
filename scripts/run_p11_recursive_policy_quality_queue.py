#!/usr/bin/env python3
"""Run six P11.4 recursive-quality cells sequentially on GPU 0/1."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p11/p11_4_recursive_policy_quality_raw/full"
JOBS = [(model, seed) for model in ("m0_f", "m1") for seed in (17, 37, 71)]


def run(job, device):
    model, seed = job
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:scripts"
    env["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 32) // 2))
    log = OUTPUT / "logs" / f"{model}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = ["python", "scripts/eval_p11_recursive_policy_quality_raw.py",
               "--model", model, "--seed", str(seed), "--device", f"cuda:{device}"]
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return job, result.returncode


def worker(jobs, device):
    results = []
    for job in jobs:
        result = run(job, device)
        print(result[0], "passed" if result[1] == 0 else f"failed({result[1]})", flush=True)
        results.append(result)
        if result[1]:
            break
    return results


def main():
    pending = [job for job in JOBS if not (OUTPUT / f"{job[0]}_seed{job[1]}/raw_manifest.json").exists()]
    queues = [pending[0::2], pending[1::2]]
    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, queues[device], device) for device in (0, 1)]
        for future in as_completed(futures):
            failures.extend(job for job, code in future.result() if code)
    if failures:
        raise SystemExit(f"P11.4 failed cells: {failures}")


if __name__ == "__main__":
    main()
