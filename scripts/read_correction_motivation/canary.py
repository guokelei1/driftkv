#!/usr/bin/env python3
"""Run one complete fixed-budget fit and a 128-user numerical/resource check."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from read_correction_motivation.run_population import OUTPUT, edge_name, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "max"), required=True)
    parser.add_argument("--budget", type=int, choices=(64, 256), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    base = ["--scale", args.scale, "--edge", "1", "--gpu", str(args.gpu)]
    folder = OUTPUT / "canary" / f"c{args.budget}" / args.scale / edge_name(1)
    folder.mkdir(parents=True, exist_ok=True)
    calibration = OUTPUT / "calibration" / f"c{args.budget}" / args.scale / edge_name(1)
    commands = [
        ("fit", ["scripts/read_correction_motivation/calibrate.py", *base,
                 "--budget", str(args.budget), "--output-root", str(OUTPUT / "calibration")]),
        ("evaluate", ["scripts/read_correction_v4/evaluate_full.py", *base,
                      "--calibration-dir", str(calibration), "--output", str(folder / "evaluation"),
                      "--variants", "map_all", "--limit-users", "128", "--unit-users", "128"]),
    ]
    for stage, argv in commands:
        started = time.perf_counter()
        with (folder / f"{stage}.log").open("w") as log:
            code = subprocess.run([sys.executable, *argv], cwd=ROOT,
                env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"},
                stdout=log, stderr=subprocess.STDOUT).returncode
        record = {"exit_code": code, "elapsed_seconds": time.perf_counter() - started,
                  "argv": [sys.executable, *argv]}
        write_json(folder / f"{stage}.exit.json", record)
        print(json.dumps({"scale": args.scale, "budget": args.budget, "stage": stage, **record}), flush=True)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
