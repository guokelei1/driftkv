#!/usr/bin/env python3
"""Run the 24 frozen P9.9 cells sequentially per allowed GPU."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p9/heldout_rolling_quality_raw/full"


def jobs():
    return [
        (release, model, seed)
        for release in ("r0", "r1_edge1", "r1_edge2", "r2")
        for model in ("m0_f", "m1")
        for seed in (17, 37, 71)
    ]


def run(job, device):
    release, model, seed = job
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:scripts"
    command = [
        "python", "scripts/eval_p9_heldout_rolling_quality_raw.py",
        "--release", release, "--model", model, "--seed", str(seed),
        "--device", f"cuda:{device}",
        "--output", str(OUTPUT / release / f"{model}_seed{seed}"),
    ]
    log = OUTPUT / "logs" / f"{release}_{model}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return job, result.returncode


def worker(assigned, device):
    output = []
    for job in assigned:
        result = run(job, device)
        print(result[0], "passed" if result[1] == 0 else f"failed({result[1]})", flush=True)
        output.append(result)
        if result[1]:
            break
    return output


def main() -> None:
    pending = [
        job for job in jobs()
        if not (OUTPUT / job[0] / f"{job[1]}_seed{job[2]}" / "raw_manifest.json").exists()
    ]
    queues = [pending[0::2], pending[1::2]]
    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(worker, queues[device], device): device for device in (0, 1)}
        for future in as_completed(futures):
            failures.extend(job for job, code in future.result() if code)
    if failures:
        raise SystemExit(f"P9.9 failed cells: {failures}")


if __name__ == "__main__":
    main()
