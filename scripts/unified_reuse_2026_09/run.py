#!/usr/bin/env python3
"""Prepare or resume the 15-edge queue. Full runs require an explicit --run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adjudicate import adjudicate
from common import DEFAULT_OUTPUT, PLAN, binding, sha256, write_json
from prepare import prepare

SCALES = ("medium", "large", "max")
HERE = Path(__file__).resolve().parent


def status(root: Path, active: dict | None = None) -> dict:
    complete, partial = [], []
    for scale in SCALES:
        for edge in range(1, 6):
            b = binding(scale, edge)
            directory = root / scale / b["edge"]
            summary = directory / "summary.json"
            if summary.exists():
                value = json.loads(summary.read_text())
                complete.append({key: value[key] for key in ("scale", "edge", "requests", "exact_minus_reuse_auc_pp")})
            elif directory.exists():
                progress = []
                for rank in range(4):
                    path = directory / f"rank{rank}/progress.json"
                    if path.exists():
                        progress.append(json.loads(path.read_text()))
                if progress:
                    partial.append({"scale": scale, "edge": b["edge"], "ranks": progress})
    result = {"updated_at_utc": datetime.now(timezone.utc).isoformat(),
              "plan_sha256": sha256(PLAN), "completed_edges": len(complete),
              "total_edges": 15, "completed": complete, "partial": partial, "active": active}
    write_json(root / "status.json", result)
    return result


def launch_edge(scale: str, edge: int, output_root: Path, *, probe: bool, probe_users: int):
    b = binding(scale, edge)
    edge_dir = output_root / scale / b["edge"]
    if not probe and (edge_dir / "summary.json").exists():
        return
    work_root = output_root / "probes" / f"queue_{probe_users}" if probe else output_root
    if not probe:
        status(output_root, {"scale": scale, "edge": b["edge"], "probe": False})
        failure = edge_dir / "failure.json"
        if failure.exists():
            failure.unlink()
    children = []
    logs = []
    try:
        for rank, gpu in enumerate(b["gpus"]):
            log_path = work_root / scale / b["edge"] / f"rank{rank}/runtime.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("a")
            command = [sys.executable, str(HERE / "worker.py"), "--scale", scale,
                       "--edge", str(edge), "--rank", str(rank), "--gpu", str(gpu),
                       "--prepared-root", str(output_root), "--work-root", str(work_root)]
            if probe:
                command += ["--limit-users", str(probe_users), "--unit-users", str(probe_users), "--verify"]
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
            logs.append(log)
        last_report = time.monotonic()
        interval = b["report_interval_seconds"]
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError(f"worker failed; inspect {work_root / scale / b['edge']}")
            if time.monotonic() - last_report >= interval:
                info = status(output_root, {"scale": scale, "edge": b["edge"], "probe": probe})
                current = next((row for row in info["partial"] if row["scale"] == scale
                                and row["edge"] == b["edge"]), None)
                print(json.dumps({"stage_report": info["active"],
                                  "completed_edges": info["completed_edges"],
                                  "latest_completed": info["completed"][-1:],
                                  "current_progress": current}), flush=True)
                last_report = time.monotonic()
            time.sleep(10)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError(f"worker failed; inspect {work_root / scale / b['edge']}")
    except Exception as exc:
        if not probe:
            write_json(edge_dir / "failure.json", {"scale": scale, "edge": b["edge"],
                       "error": repr(exc), "resumable": True})
            status(output_root, {"scale": scale, "edge": b["edge"], "failed": True})
        raise
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        for log in logs:
            log.close()
    if not probe:
        try:
            adjudicate(scale, edge, output_root)
        except Exception as exc:
            write_json(edge_dir / "failure.json", {"scale": scale, "edge": b["edge"],
                       "error": repr(exc), "resumable": True})
            status(output_root, {"scale": scale, "edge": b["edge"], "failed": True})
            raise
        status(output_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="bind all 15 request panels without GPUs")
    parser.add_argument("--probe", action="store_true", help="short correctness/resource run on one edge per scale")
    parser.add_argument("--run", action="store_true", help="resume the full 15-edge GPU queue")
    parser.add_argument("--probe-users-per-rank", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if sum((args.prepare, args.probe, args.run)) != 1:
        parser.error("choose exactly one of --prepare, --probe or --run")
    for scale in SCALES:
        for edge in range(1, 6):
            prepare(scale, edge, args.output_root / scale / f"v{edge-1}_to_v{edge}")
    status(args.output_root)
    if args.prepare:
        print("All 15 Full request panels and selected checkpoints are bound; no GPU evaluation started.")
        return
    if args.probe:
        for scale in SCALES:
            launch_edge(scale, 5, args.output_root, probe=True, probe_users=args.probe_users_per_rank)
        print("Three-scale probes finished; no full evaluation started.")
    else:
        for scale in SCALES:
            for edge in range(1, 6):
                launch_edge(scale, edge, args.output_root, probe=False, probe_users=0)
        print("All 15 Reuse edges completed.")


if __name__ == "__main__":
    main()
