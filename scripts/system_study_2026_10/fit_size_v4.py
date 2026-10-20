#!/usr/bin/env python3
"""Fixed-predecessor V4 fit-size probe; original calibration remains untouched."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]


def fit(args):
    from unified_reuse_2026_09.common import sha256, write_json
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "configuration.json", dict(config=config,
        config_sha256=sha256(args.config), entry_sha256=sha256(Path(__file__)),
        source_pack_sha256=sha256(ROOT/config["source_pack"]),
        origin_sha256=sha256(ROOT/config["origin"]),
        evaluation_scope="Frozen original development panels; V4-only changed fit. V3 remains original64. No blind confirmation."))
    processes = []
    for count in config["fit_sizes"]:
        source_pack = config.get("source_packs", {}).get(str(count), config["source_pack"])
        folder = args.output / f"fit{count}/calibration"
        folder.mkdir(parents=True)
        (folder / "v3").symlink_to((ROOT/config["origin"]).parent, target_is_directory=True)
        cmd = [sys.executable, "scripts/query_relay_2026_10/calibrate.py", "--output", str(folder/"v4"),
            "--panel", config["panel"], "--origin", config["origin"], "--source-pack", source_pack,
            "--target", "v4", "--fit-users", str(count), "--validation-users", str(len(config["validation_uids"])),
            "--threads", "2"]
        for key in ("mode", "feature_mode", "gate_source", "gate_normalization", "read_shared", "read_private",
                    "release_initialization", "seed", "local_epochs", "joint_epochs", "rollout_steps", "batch_size", "learning_rate"):
            cmd += ["--"+key.replace("_", "-"), str(config[key])]
        log = args.output / f"fit{count}.log"
        began = time.perf_counter()
        with log.open("w") as stream:
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            record = dict(fit_users=count, pid=child.pid, command=cmd, log=str(log))
            processes.append(record)
            write_json(args.output / "processes.json", processes)
            print(json.dumps(record), flush=True)
            child.wait()
        record.update(exit_code=child.returncode, process_wall_seconds=time.perf_counter()-began)
        write_json(args.output / "processes.json", processes)
        if child.returncode:
            raise RuntimeError(f"fit{count} failed; see {log}")
    print(json.dumps(dict(status="fitting_complete", output=str(args.output))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/system_study_2026_10/fit_size_v4.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit(args)
