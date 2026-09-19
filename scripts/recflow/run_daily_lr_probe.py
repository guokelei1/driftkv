#!/usr/bin/env python3
"""Two predeclared daily update learning rates, with unchanged future panels."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/recflow/development/window_6l_daily_lr_seed17"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(directory, name, command):
    started = time.time()
    save(BASE / "progress.json", dict(setting=directory.name, stage=name,
        started_unix=started, command=command))
    print(json.dumps(dict(setting=directory.name, stage=name, started_unix=started)), flush=True)
    with (directory / f"{name}.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"), stdout=log, stderr=subprocess.STDOUT)
    (directory / f"{name}.exit_status.txt").write_text(str(result.returncode) + "\n")
    save(directory / f"{name}.execution.json", dict(command=command, started_unix=started,
        elapsed_seconds=time.time() - started, exit_code=result.returncode))
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    os.chdir(ROOT)
    if BASE.exists():
        raise SystemExit("Existing LR probe found; inspect retained evidence instead of overwriting it.")
    canary = ROOT / "results/recflow/development/window_6l_daily_lr_canary_seed17"
    checked = json.loads((canary / "optimizer_check.json").read_text())
    assert checked["passed"] and checked["saved_learning_rates"] == [0.0001]
    assert checked["saved_optimizer_steps"] == [4068]
    settings_files = [ROOT / f"configs/recflow/window_6l_daily_{key}_seed17.json"
                      for key in ("lr1e4", "lr3e5")]
    settings = [json.loads(path.read_text()) for path in settings_files]
    parent = ROOT / settings[0]["initial_checkpoint"]
    with parent.open("rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest() == settings[0]["initial_checkpoint_sha256"]
    BASE.mkdir(parents=True)
    shutil.copyfile(__file__, BASE / "launcher_source.py")
    save(BASE / "launch.json", dict(authorization=settings[0]["authorization"],
        canary=str(canary), actual_optimizer_check=checked,
        configurations={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in settings_files},
        resource_estimate=settings[0]["resource_estimate"],
        interpretation="All planned learning rates and both edges retained; no automatic release admission or preferred-gain gate."))
    for key, config, setting in zip(("lr1e4", "lr3e5"), settings_files, settings, strict=True):
        directory = BASE / key
        directory.mkdir()
        old_evaluation = ROOT / setting["reused_A_day20_evaluation"]
        for phase in ("B", "C"):
            source = parent if phase == "B" else directory / "B/B_epoch1/checkpoint.pt"
            command = ["torchrun", "--standalone", "--nproc-per-node=4", "scripts/recflow/window_chain.py",
                "--config", str(config), "--phase", phase, "--checkpoint", str(source),
                "--epochs", "1", "--output", str(directory / phase), "--eval-devices", "0", "1", "2", "3"]
            if phase == "B":
                command += ["--parent-evaluation", str(old_evaluation)]
            run(directory, phase, command)
            assert json.loads((directory / phase / "summary.json").read_text())["complete"]
            parent_evaluation = old_evaluation if phase == "B" else directory / "C/parent_B_d21_21"
            name = "AB_comparison" if phase == "B" else "BC_comparison"
            run(directory, name, [sys.executable, "scripts/recflow/evaluation_pair.py",
                "--parent", str(parent_evaluation),
                "--current", str(directory / phase / f"{phase}_epoch1/evaluation"),
                "--output", str(directory / name)])
    save(BASE / "progress.json", dict(complete=True, completed_unix=time.time(),
        settings=["lr1e4", "lr3e5"], phases=["B", "C"]))
    print("Both learning rates and all paired comparisons completed", flush=True)


if __name__ == "__main__":
    main()
