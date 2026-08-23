#!/usr/bin/env python3
"""Run one deterministic P9.4 ledger parity on physical GPU 0 or 1."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import run_p9_executor as ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--parity", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": str(ledger.ROOT / "src"), "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    })
    for index, job in enumerate(ledger.jobs()):
        if index % 2 != args.parity or ledger.complete(job):
            continue
        print(f"start physical_gpu={args.physical_gpu}: {job}", flush=True)
        subprocess.run([
            sys.executable, "scripts/eval_p9_executor_raw.py", "--release", job.release,
            "--model", job.model, "--seed", str(job.seed), "--device", "cuda:0",
        ], cwd=ledger.ROOT, env=environment, check=True)
        if not ledger.complete(job):
            raise RuntimeError(f"job returned without complete output: {job}")
        print(f"complete: {job}", flush=True)


if __name__ == "__main__":
    main()
