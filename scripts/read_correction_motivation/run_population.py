#!/usr/bin/env python3
"""Complete two additional budgets of the fixed history motivation probe."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from read_correction_v4.common import edge_name, sha256, sources as original_sources, write_json
from read_correction_v4.probe_nonlinear import nonlinear_configuration

OUTPUT = ROOT / "results/read_correction_2026_09/motivation_final"
JOBS = [(scale, edge, budget) for scale in ("max", "large", "medium")
        for edge in range(1, 6) for budget in (256, 64)]


def now():
    return datetime.now(timezone.utc).isoformat()


def sources():
    result = original_sources()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        result[str(path.relative_to(ROOT))] = sha256(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not os.environ.get("TMUX"):
        raise RuntimeError("run the population queue inside tmux")
    plan = json.loads((root / "plan.json").read_text())
    if plan["execution_sources"] != sources() or plan["jobs"] != [list(job) for job in JOBS]:
        raise RuntimeError("execution sources or fixed budget jobs changed")
    for budget in (64, 128, 256):
        expected = {**nonlinear_configuration(), "calibration_users": budget}
        if plan["settings"][str(budget)] != expected:
            raise RuntimeError(f"H method settings changed for C{budget}")
    for record in plan["reused_sources"]:
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"reused Q/H source changed: {record['path']}")
    canary_path = root / "canary.pass.json"
    if sha256(canary_path) != plan["canary_sha256"]:
        raise RuntimeError("the passing resource/numerical canary changed")
    canary = json.loads(canary_path.read_text())
    if canary["status"] != "pass" or canary["execution_sources"] != sources():
        raise RuntimeError("a passing canary for these sources is required")

    began = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    jobs, events = queue.Queue(), queue.Queue()
    for job in JOBS:
        jobs.put(job)
    stop = threading.Event()
    completed, active, failures = [], {}, []
    write_json(root / "launch_record.json", {"status": "running", "at": now(),
        "run_id": run_id, "tmux": os.environ["TMUX"], "plan_sha256": sha256(root / "plan.json"),
        "new_points": 30, "reused_Q_points": 60, "reused_H_points": 15})

    def command(gpu, scale, edge, budget, stage, argv):
        while not stop.is_set():
            info = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
                "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"], text=True)
            free, total = map(float, info.strip().split(","))
            if free / total >= .75:
                break
            stop.wait(15)
        if stop.is_set():
            raise RuntimeError("queue stopped after an execution failure")
        log = root / "runtime" / f"c{budget}" / scale / edge_name(edge) / f"{stage}_{run_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        events.put({"stage": stage, "gpu": gpu, "scale": scale, "edge": edge_name(edge),
            "budget": budget, "log": str(log), "at": now()})
        started = time.perf_counter()
        with log.open("w") as stream:
            code = subprocess.run([sys.executable, *argv], cwd=ROOT,
                env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"},
                stdout=stream, stderr=subprocess.STDOUT).returncode
        write_json(log.with_suffix(".exit.json"), {"exit_code": code, "at": now(),
            "elapsed_seconds": time.perf_counter() - started, "argv": [sys.executable, *argv]})
        if code:
            raise RuntimeError(f"{scale}/{edge_name(edge)} C{budget} {stage} failed: {log}")

    def worker(gpu):
        while not stop.is_set():
            try:
                scale, edge, budget = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                base = ["--scale", scale, "--edge", str(edge), "--gpu", str(gpu)]
                calibration = root / "calibration" / f"c{budget}" / scale / edge_name(edge)
                output = root / "evaluation" / f"c{budget}" / scale / edge_name(edge)
                command(gpu, scale, edge, budget, "calibrate", [
                    "scripts/read_correction_motivation/calibrate.py", *base,
                    "--budget", str(budget), "--output-root", str(root / "calibration")])
                command(gpu, scale, edge, budget, "evaluate", [
                    "scripts/read_correction_v4/evaluate_full.py", *base,
                    "--calibration-dir", str(calibration), "--output", str(output),
                    "--variants", "map_all"])
                record = json.loads((output / "summary.json").read_text())
                if (record["status"] != "complete" or record["users"] != 3000
                    or len(record["points"]) != 1 or record["points"][0]["variant"] != "map_all"):
                    raise RuntimeError("budget point does not contain the complete fixed panel")
                events.put({"stage": "complete", "gpu": gpu, "scale": scale,
                    "edge": edge_name(edge), "budget": budget, "points": record["points"], "at": now()})
            except Exception as error:
                events.put({"stage": "failed", "gpu": gpu, "scale": scale,
                    "edge": edge_name(edge), "budget": budget, "error": str(error), "at": now()})
                stop.set()
                return

    next_report = time.monotonic() + 7200
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, gpu) for gpu in range(4)]
        while any(not future.done() for future in futures) or not events.empty():
            try:
                event = events.get(timeout=5)
            except queue.Empty:
                event = None
            if event is not None:
                print(json.dumps(event), flush=True)
                if event["stage"] == "complete":
                    completed.append(event)
                    active.pop(event["gpu"], None)
                elif event["stage"] == "failed":
                    failures.append(event)
                    active.pop(event["gpu"], None)
                else:
                    active[event["gpu"]] = event
                write_json(root / "progress.json", {"at": now(), "complete_new_points": len(completed),
                    "total_new_points": len(JOBS), "active": active, "failures": failures})
            if time.monotonic() >= next_report:
                write_json(root / "reports" / f"report_{int(time.perf_counter()-began):06d}.json",
                    {"at": now(), "completed": completed, "active": active, "failures": failures})
                next_report += 7200
        for future in futures:
            future.result()
    if not failures and len(completed) != len(JOBS):
        failures.append({"error": "queue ended before all 30 new points completed"})
    result = {"status": "failed" if failures else "complete", "at": now(),
        "elapsed_seconds": time.perf_counter() - began, "completed": completed, "failures": failures}
    write_json(root / "complete.json", result)
    if failures:
        return 1
    log = root / "figure_generation.log"
    with log.open("w") as stream:
        code = subprocess.run([sys.executable, "figures/src/read_correction_motivation_2026_09.py",
            "--input-root", str(root), "--output-dir", str(ROOT / "figures/out/read_correction_2026_09/motivation_final"),
            "--require-complete"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
    write_json(root / "figure_generation.exit.json", {"exit_code": code, "at": now()})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
