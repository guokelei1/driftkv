"""Bounded, fixed-nine-window temporal evaluation of already trained Max models."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .audit_temporal_cohort import audit
from .data_common import json_save, sha256
from .runtime import source_binding
from .temporal_cohort import PROTOCOL, SEED, TemporalCohortBatches, read_plan, select_users
from .version_chain import process_start_ticks, selection_digest, terminate_group

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def read_json(path):
    return json.loads(Path(path).read_text())


def prepare(root, data, cpu_report):
    if root.exists():
        raise ValueError("Keep existing attempts; preparation requires a fresh directory")
    original_path = BASE / "runs/version_chain_verification/attempt_1/completion_audit.json"
    original = read_json(original_path)
    if (original["status"] != "artifacts_verified" or original["total_models"] != 10
            or original["total_audited_windows"] != 19):
        raise ValueError("The original ten models and nineteen windows must be fully verified")
    checks = read_json(cpu_report)
    if checks["status"] != "passed" or checks["returncode"] != 0:
        raise ValueError("New population/cutoff/parity tests must pass before preparation")
    pilot = read_json(BASE / "runs/max_budgeted/attempt_1/plan.json")
    deadline = pilot["started_unix"] + 36000 - pilot["prior_small_pilot_spent_seconds_conservative"]
    root.mkdir(parents=True)
    population_path = root / "population.npz"
    with np.load(data / "history/population.npz") as population:
        selected = select_users(population)
        eligible = int((population["role"] == 0).sum())
        reserved = np.union1d(population["cache_fit_uids"], population["cache_validation_uids"])
    np.savez(population_path, uids=selected)
    inputs = {str(population_path): sha256(population_path),
              str(data / "history/population.npz"): sha256(data / "history/population.npz"),
              str(original_path): sha256(original_path), str(cpu_report): sha256(cpu_report)}
    sources = {str(HERE / name): sha256(HERE / name) for name in [
        "temporal_cohort.py", "evaluate_temporal_cohort.py", "audit_temporal_cohort.py", "temporal_cohort_chain.py",
        "evaluate.py", "audit_scored_requests.py", "version_chain.py", "data_common.py"]}
    sources.update(source_binding(data, "vocabulary_max"))
    for name, digest in checks["source_sha256"].items():
        if sha256(name) != digest:
            raise ValueError("Tested source changed before population freeze")
        sources[name] = digest
    models = original["models"]
    if [row["version"] for row in models] != list(range(10)):
        raise ValueError("Incomplete or reordered verified model chain")
    for row in models:
        sidecar = Path(row["checkpoint"]).with_suffix(".json")
        inputs[str(sidecar)] = sha256(sidecar)
    schedule = []
    for day in range(28, 37):
        dataset = TemporalCohortBatches(data, [day], population_path)
        index = data / dataset.stage_dir / f"d{day:02d}/index.npy"
        inputs[str(index)] = sha256(index)
        version = day - 27
        current, parent = models[version], models[version-1]
        schedule.append({"version": version, "day": day, "checkpoint": current["checkpoint"],
                         "checkpoint_sha256": current["sha256"], "parent": parent["checkpoint"],
                         "parent_sha256": parent["sha256"], "requests": len(dataset),
                         "selection_sha256": selection_digest(dataset.records),
                         "users_with_requests": int(len(np.unique(dataset.records["uid"]))),
                         "zero_exposure_requests": int((dataset.records["observed"] == 0).sum()),
                         "history_min": int(dataset.records["history_length"].min()),
                         "history_max": int(dataset.records["history_length"].max())})
    # Completed original timings include model loading and score writing. Scaling
    # that whole time by request count, then adding 25%, is conservative here.
    prior = read_json(BASE / "runs/version_chain/attempt_1/completion.json")
    rate = max(row["summary"]["wall_seconds"] / row["summary"]["requests"]
               for row in prior["evaluations"] if row["role"] == 3)
    estimate = sum(row["requests"] for row in schedule)*rate*1.25 + 600
    remaining = deadline-time.time()
    plan = {"status": "frozen_before_cohort_scores", "protocol": PROTOCOL,
            "created_unix": time.time(), "data": str(data), "population": str(population_path),
            "population_size": len(selected), "population_seed": SEED, "training_residents": eligible,
            "reserved_calibration_users_excluded": len(reserved), "population_reserved_overlap": 0,
            "selection": "smallest stable UID hashes among D26 role0 residents excluding calibration UID lists",
            "independence": "temporal holdout of training users; separate adaptive extension, not unseen-user confirmation",
            "scope": "all nine immediate-parent pairs; matched causal 1024-prefix snapshots; not persistent serving",
            "metrics": "observed-positive request-macro NDCG@10; Recall@10 secondary; all @5/@20 retained",
            "intervals": "paired UID bootstrap, seed17,2000 samples; nominal 95%, no familywise significance claim",
            "all_signs_retained": True, "stop_or_select_by_gap": False,
            "history": 1024, "batch_size": 16, "world_size": 4, "schedule": schedule,
            "driver_sources": sources, "input_sha256": inputs, "original_audit": str(original_path),
            "original_unseen_confirmation_clear_edges": original["clear_confirmation_cache_edges"],
            "cpu_report": str(cpu_report), "global_deadline_unix": deadline,
            "global_maximum_seconds": 36000, "remaining_seconds_at_freeze": remaining,
            "projected_seconds_with_margin": estimate, "budget_admitted": estimate < remaining,
            "prior_evaluation_seconds_per_request_max": rate,
            "parity_reference": str(BASE / "runs/max_budgeted/attempt_1/edge_d27_full_day_28")}
    json_save(root / "plan.json", plan)
    if not plan["budget_admitted"]:
        raise TimeoutError("Input-only projection exceeds remaining total budget; no cohort scores run")
    return plan


def execute(root):
    plan_path = root / "plan.json"
    plan = read_plan(plan_path)
    if not plan["budget_admitted"] or (root / "coordinator_started.json").exists():
        raise ValueError("Unadmitted or previously started attempt; do not overwrite partial evidence")
    json_save(root / "coordinator_started.json", {"pid": os.getpid(), "process_start_ticks": process_start_ticks(os.getpid()),
                                                "started_unix": time.time(), "plan_sha256": sha256(plan_path)})
    env = dict(os.environ, PYTHONPATH="src", CUDA_VISIBLE_DEVICES="0,1,2,3", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")

    def remaining():
        return plan["global_deadline_unix"]-time.time()

    def run(case, output, parity=False):
        read_plan(plan_path)
        if remaining() < 180:
            raise TimeoutError("Global ten-hour budget reached")
        memory = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
        if len(memory.splitlines()) != 4 or min(map(int, memory.splitlines())) < 16384:
            raise RuntimeError("Insufficient measured memory on the four authorized GPUs")
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", "-m",
                   "experiments.recflow_request_ranking.code.evaluate_temporal_cohort",
                   "--plan", str(plan_path), "--data", plan["data"], "--days", str(case["day"]),
                   "--checkpoint", case["checkpoint"], "--parent", case["parent"], "--output", str(output)]
        if parity:
            command += ["--parity-reference", plan["parity_reference"]]
        marker = root / "operations" / ("parity.json" if parity else f"day{case['day']}.json")
        marker.parent.mkdir(exist_ok=True)
        begin = time.time()
        with marker.with_suffix(".log").open("xb") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state = {"status": "running", "pid": child.pid, "process_start_ticks": process_start_ticks(child.pid),
                     "command": command, "started_unix": begin}
            json_save(marker, state)
            json_save(root / "active_process.json", state)
            try:
                code = child.wait(timeout=remaining())
            except BaseException:
                terminate_group(child)
                json_save(marker, dict(state, status="interrupted", returncode=child.returncode))
                raise
        state.update(status="completed" if code == 0 else "failed", returncode=code, wall_seconds=time.time()-begin)
        json_save(marker, state)
        json_save(root / "active_process.json", state)
        if code:
            raise RuntimeError(f"Evaluation exited {code}; see {marker.with_suffix('.log')}")

    run(plan["schedule"][0], root / "parity", parity=True)
    parity_path = root / "parity/parity.json"
    if read_json(parity_path)["status"] != "passed":
        raise ValueError("Actual original-score parity failed")
    json_save(root / "readiness.json", {"status": "passed", "plan_sha256": sha256(plan_path),
              "parity_report": str(parity_path), "parity_sha256": sha256(parity_path),
              "cpu_report": plan["cpu_report"], "cpu_report_sha256": sha256(plan["cpu_report"])})
    rows, audits = [], []
    for case in plan["schedule"]:
        directory = root / "evaluation" / f"d{case['day']}"
        run(case, directory)
        result = audit(directory, Path(plan["data"]))
        audit_path = root / "score_audits" / f"d{case['day']}.json"
        json_save(audit_path, result)
        summary = read_json(directory / "summary.json")
        comparison = summary["paired_current_minus"]["reuse"]
        native = all(summary["paired_current_minus"][name]["ci95"][0] > 0 for name in ["random", "popularity"])
        rows.append({"version": case["version"], "day": case["day"], "requests": summary["requests"],
                     "positive_requests": summary["positive_requests"], "positive_users": summary["positive_users"],
                     "rebuild_ndcg10": summary["metrics"]["current"]["ndcg@10"],
                     "reuse_ndcg10": summary["metrics"]["reuse"]["ndcg@10"],
                     "rebuild_minus_reuse": comparison["difference"], "ci95_low": comparison["ci95"][0],
                     "ci95_high": comparison["ci95"][1], "native_beats_baselines": native,
                     "clear_cache_loss": comparison["ci95"][0] > 0, "summary": str(directory / "summary.json"),
                     "score_audit": str(audit_path)})
        audits.append({"path": str(audit_path), "sha256": sha256(audit_path)})
        json_save(root / "progress.json", {"status": "evaluating", "completed_windows": len(rows),
                  "windows": rows, "global_elapsed_seconds": 36000-remaining()})
    read_plan(plan_path)
    with (root / "all_windows.csv").open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    useful = sum(row["clear_cache_loss"] and row["native_beats_baselines"] for row in rows)
    complete = {"status": "artifacts_verified", "protocol": PROTOCOL, "plan_sha256": sha256(plan_path),
                "windows": rows, "score_audits": audits, "total_windows": len(rows), "population_size": 2048,
                "useful_temporal_cache_edges": useful, "original_unseen_confirmation_clear_edges": 2,
                "new_training_performed": False, "global_elapsed_seconds": 36000-remaining(),
                "within_total_budget": remaining() > 0, "requested_several_temporal_edges_observed": useful >= 3,
                "uncertainty": plan["intervals"], "holdout": plan["independence"]}
    json_save(root / "completion_audit.json", complete)
    json_save(root / "progress.json", dict(complete, status="completed"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/recflow_request_ranking_v1"))
    parser.add_argument("--cpu-report", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        plan = prepare(args.output, args.data, args.cpu_report)
        print(json.dumps({key: plan[key] for key in ["population_size", "remaining_seconds_at_freeze", "projected_seconds_with_margin", "budget_admitted"]}))
        return
    with (args.output / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            execute(args.output)
        except BaseException as error:
            json_save(args.output / "failure.json", {"type": type(error).__name__, "message": str(error), "unix": time.time()})
            raise


if __name__ == "__main__":
    main()
