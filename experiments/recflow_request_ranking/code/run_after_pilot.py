"""Durable, CPU-only predecessor barrier before the already reviewed full chain."""
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .data_common import json_save, sha256
from .version_chain import (
    assert_no_orphan_worker,
    interrupt_handler,
    native_admission,
    process_start_ticks,
    read_json,
    terminate_group,
)


def predecessor_state(pid, start, completion_path):
    if start and process_start_ticks(pid) == start:
        return "waiting_for_exact_predecessor_exit"
    if not completion_path.exists():
        raise RuntimeError("Existing pilot process ended without completion; do not relaunch or dispatch")
    if read_json(completion_path)["status"] != "completed":
        raise RuntimeError("Existing pilot did not finish successfully")
    return "ready"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, interrupt_handler)
    signal.signal(signal.SIGINT, interrupt_handler)
    with (args.output / "driver.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_no_orphan_worker(args.output)
        if (args.output / "plan.json").exists():
            raise ValueError("Preserve the existing queue attempt and inspect its state before retrying")
        preparation = read_json(args.chain / "preparation.json")
        if Path(preparation["pilot"]) != args.pilot or Path(preparation["output"]) != args.chain:
            raise ValueError("Prepared chain belongs to a different pilot or output")
        selected, native = native_admission(args.pilot)
        decision_path = args.pilot / "foundation_data_decision.json"
        decision = read_json(decision_path)
        if (decision["selected_checkpoint_sha256"] != selected["checkpoint_sha256"]
                or decision["selected_by_max_cache_gap"] or decision["Max_cache_metrics_exist_at_decision"]
                or decision["confirmation_scores_opened"]):
            raise ValueError("Foundation data decision was not fixed independently of cache/confirmation results")
        pid = read_json(args.pilot / "driver_pid.json")["pid"]
        start = process_start_ticks(pid)
        if start:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
            if "experiments.recflow_request_ranking.code.run_max_pilot" not in command:
                raise ValueError("Pilot PID was reused by another process; inspect the original run")
        original = read_json(args.pilot / "plan.json")
        hashes = dict(preparation["driver_sources"], **preparation["source_binding"])
        hashes.update(read_json(args.pilot / "source_snapshot/manifest.json"))
        for path in [Path(__file__), args.chain / "preparation.json", decision_path,
                     args.pilot / "native_selection.json", args.pilot / "m0_native_d27/summary.json"]:
            hashes[str(path)] = sha256(path)
        for path, expected in hashes.items():
            if sha256(path) != expected:
                raise ValueError(f"Reviewed source has already changed: {path}")
        plan = {"status": "reviewed_native_quality_and_budget_waiting_for_existing_pilot",
                "pilot_pid": pid, "pilot_process_start_ticks": start, "pilot": str(args.pilot),
                "chain": str(args.chain), "selected": selected, "native_admission": native,
                "source_and_decision_sha256": hashes,
                "cache_gap_positive_required_for_dispatch": False,
                "deadline_unix": original["started_unix"] + original["pilot_total_budget_seconds"]
                                 - original["prior_small_pilot_spent_seconds_conservative"] + 180,
                "started_unix": time.time()}
        json_save(args.output / "plan.json", plan)
        shutil.copyfile(__file__, args.output / "driver_snapshot.py")
        while True:
            state = predecessor_state(pid, start, args.pilot / "completion.json")
            json_save(args.output / "progress.json", {"status": state, "checked_unix": time.time(),
                                                       "predecessor_pid": pid})
            if state == "ready":
                break
            if time.time() > plan["deadline_unix"]:
                raise TimeoutError("Predecessor exceeded its declared budget; inspect the existing job")
            time.sleep(5)
        for path, expected in hashes.items():
            if sha256(path) != expected:
                raise ValueError(f"Reviewed source or decision changed while waiting: {path}")
        env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        command = [sys.executable, "-m", "experiments.recflow_request_ranking.code.version_chain",
                   "--data", preparation["data"], "--pilot", str(args.pilot), "--output", str(args.chain),
                   "--wall-hours", str(preparation["maximum_total_seconds"] / 3600), "--execute"]
        begin = time.time()
        with (args.output / "chain.log").open("xb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            active = {"pid": child.pid, "process_start_ticks": process_start_ticks(child.pid), "status": "running",
                      "command": command, "started_unix": begin}
            json_save(args.output / "active_process.json", active)
            try:
                code = child.wait()
            except BaseException:
                terminate_group(child)
                json_save(args.output / "active_process.json", dict(active, status="interrupted", returncode=child.returncode))
                raise
        json_save(args.output / "active_process.json", dict(active, status="completed" if code == 0 else "failed", returncode=code))
        json_save(args.output / "completion.json", {"status": "completed" if code == 0 else "failed",
                  "returncode": code, "chain_wall_seconds": time.time() - begin, "chain": str(args.chain)})
        if code:
            raise RuntimeError("Full-chain admission or execution failed; inspect retained outputs")


if __name__ == "__main__":
    main()
