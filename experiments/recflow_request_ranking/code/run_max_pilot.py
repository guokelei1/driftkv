"""Budgeted actual Max learning and daily snapshot evaluation, no blind chain."""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .batches import RequestBatches
from .data_common import json_save, sha256
from .evaluate import summarize


def derive_first_requests(directory, days, data):
    """Use the original input ledger, never the first positive/valid successor."""
    expected = RequestBatches(data, days, role=2, first_only=True)
    wanted = {(int(r["uid"]), int(r["day"]), int(r["rid"])) for r in expected.records}
    integer = {"uid", "rid", "day", "candidates", "exposed", "positives", "history_length"}
    selected = []
    paths = sorted(directory.glob("rank_*.csv"))
    for path in paths:
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if (int(row["uid"]), int(row["day"]), int(row["rid"])) in wanted:
                    selected.append({key: (int(value) if key in integer else float(value)) if value else None
                                     for key, value in row.items()})
    if len(selected) != len(wanted):
        raise ValueError("Full-day scores omitted or duplicated an original first request")
    summary = json.loads((directory/"summary.json").read_text())
    result = summarize(selected, list(summary["metrics"]))
    result.update(status="passed", days=days, checkpoint_sha256=summary["checkpoint_sha256"],
                  parent_sha256=summary["parent_sha256"],
                  source_csv_sha256={str(path): sha256(path) for path in paths},
                  selection="original first all-stage request before quarantine, never a later substitute")
    json_save(directory/"first_request_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/recflow_request_ranking_v1"))
    args = parser.parse_args()
    root, data = args.output, args.data
    plan = json.loads((root/"plan.json").read_text())
    config = Path(plan["config"])
    if sha256(config) != plan["config_sha256"]:
        raise ValueError("Frozen Max configuration changed")
    json_save(root/"driver_binding.json", {"sha256": sha256(__file__), "plan_sha256": sha256(root/"plan.json")})
    maximum = plan["pilot_total_budget_seconds"]-plan["prior_small_pilot_spent_seconds_conservative"]
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", "-m"]
    train_module = "experiments.recflow_request_ranking.code.train"
    eval_module = "experiments.recflow_request_ranking.code.evaluate"

    def elapsed():
        return time.time()-plan["started_unix"]

    def remaining():
        return maximum-elapsed()

    def latest(directory):
        return directory/json.loads((directory/"latest.json").read_text())["checkpoint"]

    def run(name, module, arguments):
        if remaining() < 180:
            raise TimeoutError("Total pilot budget reached before next operation")
        command = prefix+[module]+list(map(str, arguments))
        json_save(root/f"{name}_command.json", command)
        begin = time.time()
        with (root/f"{name}.log").open("xb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            json_save(root/"active_process.json", {"name": name, "pid": child.pid, "command": command})
            try:
                code = child.wait(timeout=max(1, remaining()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                json_save(root/f"{name}_exit.json", {"returncode": child.returncode, "reason": "total_budget_timeout"})
                raise
        json_save(root/f"{name}_exit.json", {"returncode": code, "wall_seconds": time.time()-begin})
        if code:
            raise RuntimeError(f"{name} failed; no successor is dispatched")

    def evaluate(name, checkpoint, days, *, parent=None, role=2, first=False, history=1024):
        command = ["--data", data, "--checkpoint", checkpoint, "--output", root/name, "--days", *days,
                   "--role", role, "--batch-size", 16, "--history-limit", history]
        if role == 1:
            command += ["--limit", 4096]
        if parent:
            command += ["--parent", parent]
        if first:
            command += ["--first-only"]
        run(name, eval_module, command)
        result = json.loads((root/name/"summary.json").read_text())
        if role == 2 and not first:
            derive_first_requests(root/name, days, data)
        return result

    while not (root/"warmup/completion.json").exists():
        pid = json.loads((root/"warmup_pid.json").read_text())["pid"]
        os.kill(pid, 0)
        if remaining() < 120:
            raise TimeoutError("Warmup exhausted the bounded pilot budget")
        time.sleep(5)
    if json.loads((root/"warmup/completion.json").read_text())["status"] != "completed":
        raise RuntimeError("Warmup stopped before completing the predeclared requests")
    warmup = latest(root/"warmup")
    evaluate("warmup_native_validation", warmup, [24, 25, 26], role=1)
    run("alignment", train_module, ["--data", data, "--config", config, "--output", root/"alignment",
         "--days", 24, 25, 26, "--phase", "alignment", "--scale", "max", "--parent", warmup,
         "--request-limit", plan["alignment_requests"], "--epochs", 2, "--listwise-weight", plan["listwise_weight"],
         "--micro-batch", 16, "--workers", 4, "--wall-hours", maximum/3600,
         "--budget-spent-seconds", elapsed(), "--checkpoint-every", 2048, "--log-every", 100])
    if json.loads((root/"alignment/completion.json").read_text())["status"] != "completed":
        raise RuntimeError("Alignment stopped early; preserve partial checkpoints without promoting them")
    choices = []
    for epoch, step in [(1, 2048), (2, 4096)]:
        checkpoint = root/"alignment"/f"step_{step:06d}.pt"
        summary = evaluate(f"alignment_epoch{epoch}_validation", checkpoint, [24, 25, 26], role=1)
        choices.append({"epoch": epoch, "checkpoint": str(checkpoint), "checkpoint_sha256": summary["checkpoint_sha256"],
                        "native_validation_ndcg10": summary["metrics"]["current"]["ndcg@10"]})
    selected = sorted(choices, key=lambda choice: (-choice["native_validation_ndcg10"], choice["epoch"]))[0]
    json_save(root/"native_selection.json", {"choices": choices, "selected": selected,
              "selection_before_cache_metrics": True, "confirmation_unopened": True})
    parent = Path(selected["checkpoint"])
    evaluate("m0_native_d27", parent, [27])
    edges = []
    for day in plan["updates"]:
        # Conservative admission for a complete next daily update + scores/save.
        requests = len(RequestBatches(data, [day], role=0))
        estimated = 1.25*((requests+63)//64*.883+50+35+300)
        if remaining() < estimated:
            json_save(root/"bounded_stop.json", {"reason": "next_complete_edge_exceeds_remaining_budget",
                      "next_day": day, "remaining_seconds": remaining(), "estimated_seconds": estimated})
            break
        name = f"update_d{day}"
        run(name, train_module, ["--data", data, "--config", config, "--output", root/name,
            "--days", day, "--phase", "update", "--scale", "max", "--parent", parent,
            "--listwise-weight", plan["listwise_weight"], "--micro-batch", 16, "--workers", 4,
            "--wall-hours", min(2, remaining()/3600), "--checkpoint-every", 2000, "--log-every", 100])
        if json.loads((root/name/"completion.json").read_text())["status"] != "completed":
            raise RuntimeError("Daily update stopped before its epoch ended")
        checkpoint = latest(root/name)
        summary = evaluate(f"edge_d{day}_full_day_{day+1}", checkpoint, [day+1], parent=parent)
        short = evaluate(f"edge_d{day}_first_history128", checkpoint, [day+1], parent=parent, first=True, history=128)
        differences = summary["paired_current_minus"]
        native = all(differences[method]["ci95"][0] > 0 for method in ["random", "popularity"])
        compatibility = differences["reuse"]["ci95"][0] > 0
        edges.append({"update_day": day, "native_beats_baselines": native,
                      "reuse_gap_positive_interval": compatibility, "full_day_summary": summary,
                      "first_request_summary": json.loads((root/f"edge_d{day}_full_day_{day+1}"/"first_request_summary.json").read_text()),
                      "first_history128_summary": short})
        json_save(root/"gate_progress.json", {"edges": edges, "quality_passed": native and compatibility})
        if native and compatibility:
            break
        parent = checkpoint
    json_save(root/"completion.json", {"status": "completed", "quality_passed": any(
        edge["native_beats_baselines"] and edge["reuse_gap_positive_interval"] for edge in edges),
        "edges": edges, "selected": selected, "confirmation_unopened": True,
        "formal_chain_dispatched": False, "Max_wall_seconds": elapsed(),
        "total_pilot_wall_seconds_conservative": elapsed()+plan["prior_small_pilot_spent_seconds_conservative"]})


if __name__ == "__main__":
    main()
