#!/usr/bin/env python3
"""Run the predeclared daily1/3-epoch development branches, retaining every result."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/recflow/development/window_6l_daily_seed17"
CONFIG = ROOT / "configs/recflow/window_6l_daily_seed17.json"
CONTINUATION = ROOT / "configs/recflow/window_6l_daily_continuation_seed17.json"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(name, command):
    started = time.time()
    save(BASE / "progress.json", dict(stage=name, started_unix=started, command=command))
    print(json.dumps(dict(stage=name, started_unix=started)), flush=True)
    with (BASE / f"{name}.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"),
            stdout=log, stderr=subprocess.STDOUT)
    (BASE / f"{name}.exit_status.txt").write_text(str(result.returncode) + "\n")
    save(BASE / f"{name}.execution.json", dict(command=command, started_unix=started,
        elapsed_seconds=time.time() - started, exit_code=result.returncode))
    if result.returncode:
        raise SystemExit(result.returncode)


def train(name, phase, checkpoint, epochs, config):
    run(name, ["torchrun", "--standalone", "--nproc-per-node=4",
        "scripts/recflow/window_chain.py", "--config", str(config), "--phase", phase,
        "--checkpoint", str(checkpoint), "--epochs", str(epochs),
        "--output", str(BASE / name), "--eval-devices", "0", "1", "2", "3"])
    assert json.loads((BASE / name / "summary.json").read_text())["complete"]


def compare(name, parent, current):
    run(name, [sys.executable, "scripts/recflow/evaluation_pair.py",
        "--parent", str(parent), "--current", str(current), "--output", str(BASE / name)])


def main():
    os.chdir(ROOT)
    if BASE.exists():
        raise SystemExit("Use the retained logs to inspect an existing run; never overwrite it.")
    settings = json.loads(CONFIG.read_text())
    parent = ROOT / settings["initial_checkpoint"]
    with parent.open("rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest() == settings["initial_checkpoint_sha256"]
    canary_path = ROOT / "results/recflow/development/window_6l_daily_canary_seed17/summary.json"
    canary = json.loads(canary_path.read_text())
    assert canary["complete"] and canary["configuration"]["canary"]
    assert canary["configuration"]["source_checkpoint_sha256"] == settings["initial_checkpoint_sha256"]
    assert canary["configuration"]["training_window_days"] == [19, 19]
    assert canary["configuration"]["evaluation_window_days"] == [20, 20]
    assert canary["epochs"][0]["training"]["processed_requests"] == 263
    BASE.mkdir(parents=True)
    shutil.copyfile(__file__, BASE / "launcher_source.py")
    save(BASE / "launch.json", dict(authorization=settings["authorization"],
        canary=str(canary_path), canary_complete=True,
        setting_files={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (CONFIG, CONTINUATION)},
        resource_estimate=settings["resource_estimate"],
        branching="All four predeclared daily endpoints are development probes; none is an admitted serving release."))
    train("B_day19", "B", parent, 3, CONFIG)
    for epochs in (1, 3):
        compare(f"AB_{epochs}epoch", BASE / "B_day19/parent_A_d20_20",
                BASE / f"B_day19/B_epoch{epochs}/evaluation")
    for epochs in (1, 3):
        name = f"C_day20_{epochs}epoch"
        train(name, "C", BASE / f"B_day19/B_epoch{epochs}/checkpoint.pt", epochs, CONTINUATION)
        compare(f"BC_{epochs}epoch", BASE / name / "parent_B_d21_21",
                BASE / name / f"C_epoch{epochs}/evaluation")
    save(BASE / "progress.json", dict(complete=True, completed_unix=time.time(),
        comparisons=[f"{edge}_{epochs}epoch" for edge in ("AB", "BC") for epochs in (1, 3)]))
    print("All daily endpoints and comparisons completed", flush=True)


if __name__ == "__main__":
    main()
