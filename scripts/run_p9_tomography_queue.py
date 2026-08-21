#!/usr/bin/env python3
"""One bounded physical-GPU worker for the frozen P9.2 coarse job ledger."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import run_p9_tomography as ledger

ROOT = Path(__file__).resolve().parents[1]


def wait_for_first(job: ledger.Job, pid: int) -> None:
    """Do not overlap a just-launched first job; fail closed if it fails."""
    manifest = job.output / "raw_manifest.json"
    while not manifest.exists():
        try:
            os.kill(pid, 0)
        except ProcessLookupError as error:
            raise RuntimeError(f"predecessor {pid} exited without {manifest}") from error
        time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--parity", type=int, choices=(0, 1), required=True)
    parser.add_argument("--wait-for-first-index", type=int, required=True)
    parser.add_argument("--wait-for-pid", type=int, required=True)
    args = parser.parse_args()
    jobs = ledger.jobs()
    first = jobs[args.wait_for_first_index]
    if args.wait_for_first_index % 2 != args.parity:
        raise ValueError("first index and queue parity differ")
    wait_for_first(first, args.wait_for_pid)
    selected = [job for index, job in enumerate(jobs) if index % 2 == args.parity and index > args.wait_for_first_index]
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    })
    for job in selected:
        if ledger.complete(job):
            print(f"skip already complete: {job}", flush=True)
            continue
        command = [
            sys.executable, "scripts/eval_p9_tomography_raw.py", "--release", job.release,
            "--model", job.model, "--seed", str(job.seed), "--device", "cuda:0",
        ]
        print(f"start physical_gpu={args.physical_gpu}: {job}", flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        if not ledger.complete(job):
            raise RuntimeError(f"job returned without complete manifest: {job}")
        print(f"complete: {job}", flush=True)


if __name__ == "__main__":
    main()
