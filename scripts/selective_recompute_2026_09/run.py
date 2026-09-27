#!/usr/bin/env python3
"""Prepare/probe the experiment; the formal queue requires explicit approval."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from selective_recompute_2026_09.common import OUTPUT, edge_name, plan, sha256, sources, write_json

HERE = Path(__file__).resolve().parent


def preflight():
    """Verify the 15 frozen panels and each distinct selected weight once."""
    config = plan()
    verified = {}
    panels = []
    for scale in config["scales"]:
        for edge in config["edges"]:
            path = OUTPUT / "panels" / scale / edge_name(edge) / "binding.json"
            record = json.loads(path.read_text())
            for entry in [record["users_file"], *record["files"].values(),
                          *(record["sources"][k] for k in ("parent", "current", "dataset"))]:
                name = str(Path(entry["path"]).resolve())
                if name not in verified:
                    verified[name] = sha256(name)
                if verified[name] != entry["sha256"]:
                    raise RuntimeError(f"prepared source changed: {name}")
            panels.append({"scale": scale, "edge": record["edge"], "binding_sha256": sha256(path),
                           "users": record["panels"]["evaluation"]["users"],
                           "requests": record["panels"]["evaluation"]["requests"]})
    result = {"status": "pass", "checked_at": datetime.now(timezone.utc).isoformat(),
              "files_checked": len(verified), "panels": panels, "execution_sources": sources()}
    write_json(OUTPUT / "preflight.json", result)
    return result


def status(output_root, active=None):
    completed = []
    progress = []
    for scale in plan()["scales"]:
        for edge in plan()["edges"]:
            directory = output_root / "runtime" / scale / edge_name(edge)
            if (directory / "summary.json").exists():
                completed.append({"scale": scale, "edge": edge_name(edge)})
            for p in sorted(directory.glob("rank*/progress.json")):
                progress.append(json.loads(p.read_text()))
    result = {"updated_at": datetime.now(timezone.utc).isoformat(), "active": active,
              "completed_edges": len(completed), "completed_baseline_curves": 4 * len(completed),
              "completed": completed, "progress": progress}
    write_json(output_root / "status.json", result)
    return result


def execute(commands, logs, output_root, active):
    children, streams = [], []
    try:
        for command, log_path in zip(commands, logs):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("a")
            streams.append(stream)
            children.append(subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT))
        last_report = time.monotonic()
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError(f"worker failed; inspect {[str(p) for p in logs]}")
            if time.monotonic() - last_report >= plan()["report_interval_seconds"]:
                info = status(output_root, active)
                print(json.dumps({"stage_report": active, "completed_edges": info["completed_edges"],
                                  "progress": info["progress"][-4:]}), flush=True)
                last_report = time.monotonic()
            time.sleep(5)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError(f"worker failed; inspect {[str(p) for p in logs]}")
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        for stream in streams:
            stream.close()


def launch_edge(scale, edge, output_root, *, probe=False):
    from selective_recompute_2026_09.adjudicate import adjudicate
    config = plan()
    name = edge_name(edge)
    directory = output_root / "runtime" / scale / name
    active = {"scale": scale, "edge": name, "phase": "calibration", "probe": probe}
    status(output_root, active)
    command = [sys.executable, str(HERE / "calibrate.py"), "--scale", scale, "--edge", str(edge),
               "--gpu", str(config["gpus"][0]), "--output-root", str(output_root)]
    if probe:
        command += ["--limit-users", "2"]
    execute([command], [directory / "calibration.log"], output_root, active)
    active["phase"] = "four_gpu_evaluation"
    status(output_root, active)
    commands, logs = [], []
    for rank, gpu in enumerate(config["gpus"]):
        command = [sys.executable, str(HERE / "worker.py"), "--scale", scale, "--edge", str(edge),
                   "--rank", str(rank), "--world-size", str(len(config["gpus"])), "--gpu", str(gpu),
                   "--output-root", str(output_root), "--calibration-root", str(output_root),
                   "--partition", "canary" if probe else "evaluation"]
        if probe:
            command += ["--limit-users", "4", "--unit-users", "4", "--verify", "--stress"]
        commands.append(command)
        logs.append(directory / f"rank{rank}" / "runtime.log")
    try:
        execute(commands, logs, output_root, active)
        result = adjudicate(scale, edge, output_root, world_size=len(config["gpus"]))
    except Exception as exc:
        write_json(directory / "failure.json", {"error": str(exc), "resumable": True})
        status(output_root, {**active, "failed": True})
        raise
    failure = directory / "failure.json"
    if failure.exists():
        failure.unlink()
    status(output_root)
    print(json.dumps({"completed": name, "scale": scale, "probe": probe}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--probe", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--status", action="store_true")
    parser.add_argument("--user-approved", action="store_true", help="formal launch only after user OK")
    parser.add_argument("--scales", nargs="+", choices=("medium", "large", "max"))
    args = parser.parse_args()
    config = plan()
    if args.status:
        print(json.dumps(status(OUTPUT), indent=2))
        return
    if args.prepare:
        from selective_recompute_2026_09.prepare import prepare
        for scale in config["scales"]:
            for edge in config["edges"]:
                prepare(scale, edge, OUTPUT / "panels", users=config["evaluation_users_per_edge"],
                        calibration_users=config["calibration_users_per_edge"],
                        canary_users=config["canary_users_per_edge"])
        result = preflight()
        print(f"All15 panels ready; {result['files_checked']} files verified. No GPU run started.")
    elif args.probe:
        preflight()
        root = OUTPUT / "probes" / "four_gpu"
        for scale in args.scales or config["scales"]:
            launch_edge(scale, 1, root, probe=True)
        print("Bounded canaries finished. Formal evaluation has not started.")
    else:
        if not args.user_approved:
            parser.error("formal run awaits the user's OK; then pass --user-approved")
        if args.scales:
            parser.error("formal queue covers all three scales")
        ready_path = OUTPUT / "readiness.json"
        if not ready_path.exists():
            raise RuntimeError("missing preparation/canary readiness report")
        ready = json.loads(ready_path.read_text())
        if ready.get("status") != "ready" or ready["execution_sources"] != sources():
            raise RuntimeError("readiness report does not match the execution sources")
        preflight()
        for scale in config["scales"]:
            for edge in config["edges"]:
                launch_edge(scale, edge, OUTPUT)
        subprocess.run([sys.executable, str(ROOT / "figures/src/selective_recompute_2026_09.py"),
                        "--input", str(OUTPUT / "summary.json"),
                        "--out", str(ROOT / "figures/out/selective_recompute_2026_09")], check=True, cwd=ROOT)
        status(OUTPUT)
        print("All60 baseline curves completed;12 figures exported.")


if __name__ == "__main__":
    main()
