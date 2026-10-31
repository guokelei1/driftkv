"""Prepare and continue a fixed nine-update chain, preserving every edge.

Preparation is CPU-only. Execution requires the bounded Max experiment to have
finished and its native-quality evidence to pass; a cache-gap sign is never an
admission or stopping rule. Existing checkpoints remain at their original paths.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .batches import RequestBatches
from .data_common import json_save, sha256
from .protocol import day_date
from .run_max_pilot import derive_first_requests
from .runtime import source_binding

HERE = Path(__file__).resolve().parent
FIELDS = ("uid", "rid", "day", "ts", "history_length")


def read_json(path):
    return json.loads(Path(path).read_text())


def selection_digest(records):
    digest = hashlib.sha256()
    for name in FIELDS:
        digest.update(name.encode())
        digest.update(np.asarray(records[name], dtype="<i8").tobytes())
    return digest.hexdigest()


def checkpoint_info(path, *, expected_parent=None, verify_bytes=False):
    path = Path(path)
    info = read_json(path.with_suffix(".json"))
    metadata = info["metadata"]
    if (info["status"] != "committed" or not path.is_file() or path.stat().st_size != info["bytes"]
            or metadata["scale"] != "max" or metadata.get("quality_use_permitted") is not True):
        raise ValueError("Expected a committed, quality-eligible actual Max checkpoint")
    if expected_parent is not None and metadata["parent_checkpoint_sha256"] != expected_parent:
        raise ValueError("Checkpoint's actual training parent differs from this chain")
    if verify_bytes and sha256(path) != info["checkpoint_sha256"]:
        raise ValueError("Checkpoint bytes changed")
    return info


def audit_update(directory, parent_sha, day, expected, config, binding):
    """A completed endpoint, never an interrupted point or a different subset."""
    directory = Path(directory)
    plan, completion = read_json(directory / "plan.json"), read_json(directory / "completion.json")
    path = directory / read_json(directory / "latest.json")["checkpoint"]
    info = checkpoint_info(path, expected_parent=parent_sha)
    metadata, arguments = info["metadata"], plan["arguments"]
    required = {"phase": "update", "days": [day], "scale": "max", "epochs": 1,
                "global_batch": 64, "micro_batch": 16, "listwise_weight": .1,
                "pilot": False, "request_limit": None, "max_steps": None,
                "dtype": "bfloat16", "communication": "rows", "seed": 17,
                "diagnostic_state": False, "stop_after_steps": None}
    steps = math.ceil(expected["requests"] / 64)
    if (completion["status"] != "completed" or completion["steps"] != steps
            or completion["next_epoch"] != 1 or completion["next_step"] != 0
            or metadata["total_step"] != steps or metadata["days"] != [day]
            or metadata["phase"] != "update" or metadata["proposal"] != config
            or plan["world_size"] != 4 or plan["requests"] != expected["requests"]
            or plan["planned_steps"] != steps or plan["parent_checkpoint_sha256"] != parent_sha
            or plan["trainer_sha256"] != sha256(HERE / "train.py")
            or metadata["source_binding"] != binding or plan["source_binding"] != binding
            or any(arguments.get(key) != value for key, value in required.items())):
        raise ValueError("Daily update is partial or differs from the frozen recipe")
    selected_path = directory / "request_selection.npz"
    if sha256(selected_path) != plan["request_selection_sha256"]:
        raise ValueError("Saved training selection changed")
    with np.load(selected_path) as selected:
        if selection_digest(selected) != expected["selection_sha256"]:
            raise ValueError("Daily update did not use the planned request population/order")
    return {"checkpoint": str(path), "sha256": info["checkpoint_sha256"], "day": day}


def audit_evaluation(directory, checkpoint, parent, day, role, expected_count, binding):
    summary = read_json(Path(directory) / "summary.json")
    if (summary["status"] != "passed" or summary["checkpoint_sha256"] != checkpoint["sha256"]
            or summary["parent_sha256"] != (parent["sha256"] if parent else None)
            or summary["days"] != [day] or summary["role"] != role
            or summary["first_only"] or summary["history_limit"] != 1024
            or summary["requests"] != expected_count or summary["arguments"]["limit"] is not None
            or summary["source_binding"] != binding or summary["world_size"] != 4
            or summary["evaluator_sha256"] != sha256(HERE / "evaluate.py")
            or summary["precision"] != "float32_parameters_and_KV_tf32_disabled"):
        raise ValueError("Evaluation has a different checkpoint, window, population, or backend")
    return summary


def native_admission(pilot):
    """Review native learning independently of the first two cache-gap results."""
    selection = read_json(pilot / "native_selection.json")
    selected = selection["selected"]
    best = sorted(selection["choices"], key=lambda v: (-v["native_validation_ndcg10"], v["epoch"]))[0]
    if selected != best or not selection["selection_before_cache_metrics"] or not selection["confirmation_unopened"]:
        raise ValueError("Foundation was not selected by the frozen native-validation rule")
    validation_path = pilot / f"alignment_epoch{selected['epoch']}_validation/summary.json"
    validation = read_json(validation_path)
    if (validation["status"] != "passed" or validation["checkpoint_sha256"] != selected["checkpoint_sha256"] or validation["role"] != 1
            or validation["days"] != [24, 25, 26] or validation["requests"] != 4096
            or validation["history_limit"] != 1024
            or validation["metrics"]["current"]["ndcg@10"] != selected["native_validation_ndcg10"]):
        raise ValueError("Selected foundation validation evidence differs")
    native = validation["metrics"]["current"]["ndcg@10"]
    if not all(native > validation["metrics"][name]["ndcg@10"] for name in ["random", "popularity"]):
        raise ValueError("Foundation native validation does not beat both simple baselines")
    paths = [pilot / "m0_native_d27/summary.json"]
    paths += sorted(pilot.glob("edge_d*_full_day_*/summary.json"))
    evidence = []
    for path in paths:
        value = read_json(path)
        differences = value["paired_current_minus"]
        evidence.append({"path": str(path), "sha256": sha256(path),
                         "native_beats_both_baselines": all(differences[m]["ci95"][0] > 0
                                                           for m in ["random", "popularity"])})
    if not any(item["native_beats_both_baselines"] for item in evidence):
        raise ValueError("No completed full-day native model clearly beats both simple baselines")
    return selected, {"validation": str(validation_path), "validation_sha256": sha256(validation_path),
                      "full_day_native_evidence": evidence, "cache_gap_used_for_admission": False}


def prepare(data, pilot, output, hours):
    original = read_json(pilot / "plan.json")
    if sha256(original["config"]) != original["config_sha256"]:
        raise ValueError("Bounded Max configuration changed")
    schedule = []
    for day in range(27, 36):
        train = RequestBatches(data, [day], role=0)
        counts = {str(role): len(RequestBatches(data, [day + 1], role=role, allow_confirmation=role == 3))
                  for role in [2, 3]}
        train_seconds = math.ceil(len(train) / 64) * .883 + 50 + 35
        eval_seconds = sum(math.ceil(count / 64) * .539 + 70 for count in counts.values())
        schedule.append({"version": day - 26, "update_day": day, "update_date": day_date(day),
                         "evaluation_day": day + 1, "evaluation_date": day_date(day + 1),
                         "requests": len(train), "selection_sha256": selection_digest(train.records),
                         "evaluation_requests_by_role": counts,
                         "projected_seconds_with_25percent_margin": 1.25 * (train_seconds + eval_seconds)})
    extra_count = len(RequestBatches(data, [37], role=3, allow_confirmation=True))
    extra_seconds = 1.25 * (math.ceil(extra_count / 64) * .539 + 70)
    return {"status": "prepared_not_dispatched", "data": str(data), "pilot": str(pilot),
            "output": str(output), "config": original["config"], "config_sha256": original["config_sha256"],
            "world_size": 4, "maximum_total_seconds": hours * 3600,
            "maximum_foundation_seconds": 10 * 3600, "maximum_daily_training_seconds": 2 * 3600,
            "schedule": schedule, "extra_d37_confirmation_requests": extra_count,
            "all_nine_updates_and_evaluations_projection_seconds": sum(
                row["projected_seconds_with_25percent_margin"] for row in schedule) + extra_seconds,
            "projection_basis": "actual Max p95 .883s train/64 requests, .539s dual-model read/64; load/save +25%",
            "selection": "M0 native pre-release validation; every daily update one full epoch, never select by gap",
            "cache_semantics": "per-request matched causal 1024-prefix snapshots, not persistent serving",
            "metric": "observed-positive request-macro NDCG@10; paired UID bootstrap",
            "confirmation": "all nine edges after recipe freeze; v0->v1 primary, all others secondary; D37 extra",
            "secondary_ci_interpretation": "nominal per-edge intervals; positive-edge counts are descriptive, not familywise claims",
            "policy_revision": "Updated user goal asks for several useful edges among ten models, not every edge. "
                               "Two inconclusive preliminary cache edges do not veto an otherwise useful complete chain.",
            "stop_on_cache_gap_sign": False, "confirmation_scores_opened": False,
            "source_binding": source_binding(data, "vocabulary_max"),
            "driver_sources": {str(path): sha256(path) for path in [Path(__file__), HERE / "run_max_pilot.py",
                                                                    HERE / "train.py", HERE / "evaluate.py"]}}


def terminate_group(child):
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def execute(plan, root):
    data, pilot = Path(plan["data"]), Path(plan["pilot"])
    if not (root / "plan.json").exists():
        if not (pilot / "completion.json").exists():
            raise RuntimeError("Bounded Max is still running; no successor or confirmation freeze created")
        completion = read_json(pilot / "completion.json")
        if completion["status"] != "completed" or not completion["confirmation_unopened"]:
            raise ValueError("Wait for the bounded Max experiment to finish before continuing")
        selected, evidence = native_admission(pilot)
        foundation = checkpoint_info(selected["checkpoint"], verify_bytes=True)
        if foundation["checkpoint_sha256"] != selected["checkpoint_sha256"]:
            raise ValueError("Selected foundation checkpoint changed")
        metadata = foundation["metadata"]
        if (metadata["source_binding"] != plan["source_binding"] or metadata["proposal"] != read_json(plan["config"])
                or metadata["phase"] != "alignment" or metadata["days"] != [24, 25, 26]):
            raise ValueError("Foundation numerical sources, data, or training protocol differ")
        parent = {"checkpoint": selected["checkpoint"], "sha256": selected["checkpoint_sha256"], "day": 26}
        audit_evaluation(pilot / "m0_native_d27", parent, None, 27, 2,
                         len(RequestBatches(data, [27], role=2)), plan["source_binding"])
        for edge in completion["edges"]:
            day = edge["update_day"]
            if day not in [27, 28] or day != parent["day"] + 1:
                raise ValueError("Pilot update sequence is not the declared adjacent-version chain")
            row = plan["schedule"][day - 27]
            current = audit_update(pilot / f"update_d{day}", parent["sha256"], day, row,
                                   metadata["proposal"], plan["source_binding"])
            checkpoint_info(current["checkpoint"], expected_parent=parent["sha256"], verify_bytes=True)
            audit_evaluation(pilot / f"edge_d{day}_full_day_{day + 1}", current, parent, day + 1, 2,
                             row["evaluation_requests_by_role"]["2"], plan["source_binding"])
            parent = current
        spent = completion["total_pilot_wall_seconds_conservative"]
        if foundation["elapsed_budget_seconds"] > plan["maximum_foundation_seconds"]:
            raise ValueError("Foundation exceeded the new ten-hour limit")
        if spent + plan["all_nine_updates_and_evaluations_projection_seconds"] > plan["maximum_total_seconds"]:
            raise ValueError("Conservative full-chain projection exceeds the overall time budget")
        plan.update(status="frozen_before_confirmation", started_unix=time.time(), prior_spent_seconds=spent,
                    selected=selected, native_admission=evidence,
                    foundation={"checkpoint": selected["checkpoint"], "sha256": selected["checkpoint_sha256"], "day": 26})
        json_save(root / "plan.json", plan)
        shutil.copyfile(__file__, root / "driver_snapshot.py")
    else:
        frozen = read_json(root / "plan.json")
        for key in ["data", "pilot", "output", "config_sha256", "source_binding", "driver_sources", "schedule", "maximum_total_seconds"]:
            if frozen[key] != plan[key]:
                raise ValueError(f"Frozen chain plan changed: {key}")
        plan = frozen
    freeze = {"status": "frozen_before_confirmation", "plan": str(root / "plan.json"),
              "plan_sha256": sha256(root / "plan.json"), "foundation_sha256": plan["foundation"]["sha256"],
              "config_sha256": plan["config_sha256"], "primary_edge": "v0_to_v1", "all_nine_edges_reported": True}
    freeze_path = data / "formal_recipe_freeze.json"
    if freeze_path.exists() and read_json(freeze_path) != freeze:
        raise ValueError("A different confirmation recipe is already frozen")
    if not freeze_path.exists():
        json_save(freeze_path, freeze)
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", "-m"]

    def remaining():
        return plan["maximum_total_seconds"] - plan["prior_spent_seconds"] - (time.time() - plan["started_unix"])

    def run(name, module, arguments):
        if remaining() < 180:
            raise TimeoutError("Overall chain budget reached; committed checkpoints retained")
        for path, digest in plan["driver_sources"].items():
            if sha256(path) != digest:
                raise ValueError("Frozen execution source changed")
        if source_binding(data, "vocabulary_max") != plan["source_binding"]:
            raise ValueError("Frozen numerical sources or data changed")
        free = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
        needed_mib = 32768 if module == "train" else 16384
        if len(free.splitlines()) != 4 or min(map(int, free.splitlines())) < needed_mib:
            raise RuntimeError("Four-card job has insufficient free GPU memory; no duplicate worker launched")
        folder = root / "operations" / name
        folder.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while (folder / f"attempt_{attempt}.json").exists():
            attempt += 1
        marker = folder / f"attempt_{attempt}.json"
        command = prefix + [f"experiments.recflow_request_ranking.code.{module}"] + list(map(str, arguments))
        begin = time.time()
        with marker.with_suffix(".log").open("xb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            state = {"command": command, "pid": child.pid, "started_unix": begin, "name": name,
                     "status": "running", "process_start_ticks": process_start_ticks(child.pid)}
            json_save(marker, state)
            json_save(root / "active_process.json", state)
            try:
                code = child.wait(timeout=remaining())
            except BaseException:
                terminate_group(child)
                stopped = dict(state, status="interrupted", returncode=child.returncode, wall_seconds=time.time() - begin)
                json_save(marker, stopped)
                json_save(root / "active_process.json", stopped)
                raise
        finished = dict(state, status="completed" if code == 0 else "failed", returncode=code, wall_seconds=time.time() - begin)
        json_save(marker, finished)
        json_save(root / "active_process.json", finished)
        if code:
            raise RuntimeError(f"{name} failed; later versions not launched")

    config = read_json(plan["config"])
    versions, results = [plan["foundation"]], []

    def evaluate(current, parent, day, role, count, inherited=None):
        if inherited and (inherited / "summary.json").exists():
            directory = inherited
        else:
            folder = root / f"evaluation/role{role}_d{day}"
            attempts = sorted(folder.glob("attempt_*"), key=lambda path: int(path.name.split("_")[-1]))
            completed = [p for p in attempts if (p / "summary.json").exists()]
            if completed:
                directory = completed[-1]
            else:
                directory = folder / f"attempt_{len(attempts) + 1}"
                arguments = ["--data", data, "--checkpoint", current["checkpoint"], "--parent", parent["checkpoint"],
                             "--output", directory, "--days", day, "--role", role, "--batch-size", 16,
                             "--history-limit", 1024]
                if role == 3:
                    arguments += ["--allow-confirmation"]
                run(f"eval_role{role}_d{day}", "evaluate", arguments)
        summary = audit_evaluation(directory, current, parent, day, role, count, plan["source_binding"])
        if role == 2 and not (directory / "first_request_summary.json").exists():
            derive_first_requests(directory, [day], data)
        results.append({"day": day, "role": role, "directory": str(directory), "summary_sha256": sha256(directory / "summary.json"),
                        "summary": summary})
        json_save(root / "progress.json", {"versions": versions, "evaluations": results,
                                           "remaining_seconds": remaining(), "all_signs_retained": True})

    for row in plan["schedule"]:
        day, parent = row["update_day"], versions[-1]
        inherited = pilot / f"update_d{day}"
        reusable = (inherited / "completion.json").exists() and read_json(inherited / "completion.json")["status"] == "completed"
        directory = inherited if reusable else root / f"training/v{row['version']}"
        complete = directory / "completion.json"
        if not complete.exists() or read_json(complete)["status"] != "completed":
            if remaining() < row["projected_seconds_with_25percent_margin"]:
                raise TimeoutError("Next complete daily update/evaluation exceeds remaining budget")
            arguments = ["--data", data, "--config", plan["config"], "--output", directory, "--days", day,
                         "--phase", "update", "--scale", "max", "--epochs", 1, "--listwise-weight", .1,
                         "--global-batch", 64, "--micro-batch", 16, "--workers", 4,
                         "--wall-hours", 2, "--checkpoint-every", 2000, "--log-every", 100]
            if (directory / "plan.json").exists():
                latest = directory / read_json(directory / "latest.json")["checkpoint"]
                arguments += ["--resume", latest]
            else:
                arguments += ["--parent", parent["checkpoint"]]
            run(f"train_d{day}", "train", arguments)
        current = audit_update(directory, parent["sha256"], day, row, config, plan["source_binding"])
        versions.append(current)
        evaluate(current, parent, day + 1, 2, row["evaluation_requests_by_role"]["2"],
                 inherited=pilot / f"edge_d{day}_full_day_{day + 1}")
    # Confirmation cannot influence training, early stopping, or version count:
    # all ten models and all nine development edges already exist at this point.
    for row in plan["schedule"]:
        version = row["version"]
        evaluate(versions[version], versions[version - 1], row["evaluation_day"], 3,
                 row["evaluation_requests_by_role"]["3"])
    evaluate(versions[-1], versions[-2], 37, 3, plan["extra_d37_confirmation_requests"])
    rows = []
    for item in results:
        summary = item["summary"]
        gap = summary["paired_current_minus"]["reuse"]
        rows.append({"day": item["day"], "role": item["role"], "requests": summary["requests"],
                     "positive_users": summary["positive_users"], "rebuild": summary["metrics"]["current"]["ndcg@10"],
                     "reuse": summary["metrics"]["reuse"]["ndcg@10"], "difference": gap["difference"],
                     "ci_low": gap["ci95"][0], "ci_high": gap["ci95"][1]})
    with (root / "all_edges.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_save(root / "completion.json", {"status": "completed", "versions": versions, "evaluations": results,
              "positive_nominal_confirmation_intervals": sum(row["role"] == 3 and row["day"] <= 36 and row["ci_low"] > 0 for row in rows),
              "quality_claim": "Report every edge; secondary nominal intervals are not familywise significance",
              "total_wall_seconds_conservative": plan["maximum_total_seconds"] - remaining()})


def process_start_ticks(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()
        return fields[19] if fields[0] != "Z" else None
    except FileNotFoundError:
        return None


def assert_no_orphan_worker(root):
    marker = root / "active_process.json"
    if marker.exists():
        active = read_json(marker)
        start = active.get("process_start_ticks")
        if active.get("status") == "running" and start and process_start_ticks(active["pid"]) == start:
            raise RuntimeError("A prior coordinator's exact worker is still running; do not duplicate it")


def interrupt_handler(signum, frame):
    raise KeyboardInterrupt(f"Chain coordinator received signal {signum}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/recflow_request_ranking_v1"))
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-hours", type=float, default=10)
    parser.add_argument("--execute", action="store_true", help="Run only after bounded Max completion and native-quality review")
    args = parser.parse_args()
    if not 0 < args.wall_hours <= 10:
        raise ValueError("New whole-chain budget must be positive and at most ten hours")
    args.output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, interrupt_handler)
    signal.signal(signal.SIGINT, interrupt_handler)
    with (args.output / "driver.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_no_orphan_worker(args.output)
        plan = prepare(args.data, args.pilot, args.output, args.wall_hours)
        if args.execute:
            execute(plan, args.output)
        else:
            json_save(args.output / "preparation.json", plan)
            print(json.dumps({"status": plan["status"], "updates": len(plan["schedule"]),
                              "nine_updates_and_evaluations_projected_hours": plan["all_nine_updates_and_evaluations_projection_seconds"] / 3600}))


if __name__ == "__main__":
    main()
