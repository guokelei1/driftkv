#!/usr/bin/env python3
"""Run the three frozen P9.10 runtime conditions on GPU 0/1."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p9/full_population_runtime/full"
CONDITIONS = (
    "edge1_m0_r2_seed17",
    "edge1_m1_r2_seed17",
    "edge2_m1_r1_edge2_seed17",
)


def run(condition, device):
    output = OUTPUT / condition
    if (output / "result.json").exists():
        return condition, 0
    log = OUTPUT / "logs" / f"{condition}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:scripts"
    command = [
        "python", "scripts/eval_p9_full_population_runtime.py",
        "--condition", condition, "--device", f"cuda:{device}",
        "--output", str(output),
    ]
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return condition, result.returncode


def worker(values, device):
    rows = []
    for value in values:
        row = run(value, device)
        print(row[0], "passed" if row[1] == 0 else f"failed({row[1]})", flush=True)
        rows.append(row)
        if row[1]:
            break
    return rows


def main() -> None:
    queues = [CONDITIONS[0::2], CONDITIONS[1::2]]
    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, queues[device], device) for device in (0, 1)]
        for future in as_completed(futures):
            failures.extend(name for name, code in future.result() if code)
    if failures:
        raise SystemExit(f"P9.10 failed conditions: {failures}")


if __name__ == "__main__":
    main()
