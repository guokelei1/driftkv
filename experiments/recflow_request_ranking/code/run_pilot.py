"""Bounded two-arm pilot; validation selects the loss before any cache metrics."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .data_common import json_save, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    plan = json.loads((root / "plan.json").read_text())
    started = time.monotonic()
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", "-m"]
    training = "experiments.recflow_request_ranking.code.train"
    evaluation = "experiments.recflow_request_ranking.code.evaluate"
    json_save(root / "driver_binding.json", {"sha256": sha256(__file__), "plan_sha256": sha256(root / "plan.json")})
    while not (root / "warmup/completion.json").exists():
        pid = json.loads((root / "warmup_pid.json").read_text())["pid"]
        os.kill(pid, 0)
        if time.monotonic()-started > 3600:
            raise TimeoutError("Warmup did not finish within the pilot limit")
        time.sleep(5)
    completed = json.loads((root / "warmup/completion.json").read_text())
    if completed["status"] != "completed":
        raise RuntimeError("Warmup did not complete; do not silently promote a partial model")
    prior = completed["elapsed_budget_seconds"]

    def run(name, module, arguments):
        if prior+time.monotonic()-started > plan["wall_limit_seconds"]-180:
            raise TimeoutError("Total pilot budget exhausted before next operation")
        command = prefix+[module]+list(map(str, arguments))
        json_save(root / f"{name}_command.json", command)
        with (root / f"{name}.log").open("xb") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                    timeout=max(1, plan["wall_limit_seconds"]-prior-(time.monotonic()-started)))
        json_save(root / f"{name}_exit.json", {"returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed; preserve evidence and inspect the log")

    def latest(directory):
        return directory / json.loads((directory / "latest.json").read_text())["checkpoint"]

    def evaluate(name, checkpoint, days, *, parent=None, role=1, first=False, history=1024):
        output = root / name
        command = ["--checkpoint", checkpoint, "--output", output, "--days", *days,
                   "--role", role, "--batch-size", 16, "--history-limit", history]
        if role == 1:
            command += ["--limit", 2048]
        if parent:
            command += ["--parent", parent]
        if first:
            command += ["--first-only"]
        run(name, evaluation, command)
        return json.loads((output / "summary.json").read_text())

    warmup = latest(root / "warmup")
    for step in [1000, 2000]:
        evaluate(f"warmup_val_{step}", root / "warmup" / f"step_{step:06d}.pt", [24, 25, 26])
    evaluate("warmup_val_final", warmup, [24, 25, 26])
    arms = []
    for weight in plan["alignment"]["lambda_arms"]:
        name = f"align_lambda_{weight:g}"
        run(name, training, ["--output", root/name, "--days", 24, 25, 26, "--phase", "alignment",
                             "--parent", warmup, "--pilot", "--epochs", 2, "--listwise-weight", weight,
                             "--micro-batch", 16, "--workers", 4, "--wall-hours", 3, "--log-every", 100])
        evaluate(f"{name}_val_1000", root/name/"step_001000.pt", [24, 25, 26])
        checkpoint = latest(root/name)
        summary = evaluate(f"{name}_val_final", checkpoint, [24, 25, 26])
        arms.append({"weight": weight, "checkpoint": str(checkpoint),
                     "native_validation_ndcg10": summary["metrics"]["current"]["ndcg@10"],
                     "checkpoint_sha256": summary["checkpoint_sha256"]})
    selected = sorted(arms, key=lambda arm: (-arm["native_validation_ndcg10"], arm["weight"]))[0]
    json_save(root / "native_selection.json", {"arms": arms, "selected": selected,
              "selection_before_cache_metrics": True, "confirmation_unopened": True})
    parent = Path(selected["checkpoint"])
    evaluate("m0_native_d27", parent, [27], role=2, first=True)
    edges = []
    for day in [27, 28]:
        name = f"update_d{day}"
        run(name, training, ["--output", root/name, "--days", day, "--phase", "update",
                             "--parent", parent, "--pilot", "--listwise-weight", selected["weight"],
                             "--micro-batch", 16, "--workers", 4, "--wall-hours", 1, "--log-every", 100])
        checkpoint = latest(root/name)
        summary = evaluate(f"edge_d{day}_eval_d{day+1}", checkpoint, [day+1], parent=parent, role=2, first=True)
        short = evaluate(f"edge_d{day}_eval_d{day+1}_history128", checkpoint, [day+1], parent=parent,
                         role=2, first=True, history=128)
        differences = summary["paired_current_minus"]
        native_pass = all(differences[method]["ci95"][0] > 0 for method in ["random", "popularity"])
        gap_pass = differences["reuse"]["ci95"][0] > 0
        edges.append({"update_day": day, "native_beats_baselines": native_pass,
                      "reuse_gap_positive_interval": gap_pass, "summary": summary,
                      "history128_summary": short})
        json_save(root / "gate_progress.json", {"edges": edges, "quality_passed": native_pass and gap_pass})
        if native_pass and gap_pass:
            break
        parent = checkpoint
    json_save(root / "completion.json", {"status": "completed", "quality_passed": any(
        edge["native_beats_baselines"] and edge["reuse_gap_positive_interval"] for edge in edges),
        "edges": edges, "selected": selected, "confirmation_unopened": True,
        "formal_chain_dispatched": False, "wall_seconds_conservative": prior+time.monotonic()-started})


if __name__ == "__main__":
    main()
