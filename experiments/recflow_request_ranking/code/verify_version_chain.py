"""Read-only CPU verification of every completed window and the final chain."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import time
from pathlib import Path

from .audit_scored_requests import audit
from .data_common import json_save, sha256
from .version_chain import audit_update, checkpoint_info, process_start_ticks, read_json

HERE = Path(__file__).resolve().parent
EXPECTED_WINDOWS = {(role, day) for role in [2, 3] for day in range(28, 37)} | {(3, 37)}


def validate_completed_scope(progress):
    days = [entry["day"] for entry in progress["versions"]]
    windows = [(entry["role"], entry["day"]) for entry in progress["evaluations"]]
    if days != list(range(26, 36)) or len({entry["sha256"] for entry in progress["versions"]}) != 10:
        raise ValueError("Completion requires all ten actual versions in the fixed daily order")
    if set(windows) != EXPECTED_WINDOWS or len(windows) != len(EXPECTED_WINDOWS):
        raise ValueError("Completion requires all nine development, nine confirmation, and extra D37 windows")


def existing_audit_matches(value, entry):
    summary = str(Path(entry["directory"]) / "summary.json")
    if (value["status"] != "passed" or value["audit_sha256"] != sha256(HERE / "audit_scored_requests.py")
            or value["source_artifact_sha256"].get(summary) != entry["summary_sha256"]):
        return False
    return all(sha256(path) == digest for path, digest in value["source_artifact_sha256"].items())


def verify_window(entry, data, output, seed_audits):
    key = f"role{entry['role']}_d{entry['day']}"
    target = output / "score_audits" / f"{key}.json"
    if sha256(Path(entry["directory"]) / "summary.json") != entry["summary_sha256"]:
        raise ValueError("Controller's committed evaluation summary changed")
    if target.exists():
        value = read_json(target)
        if not existing_audit_matches(value, entry):
            raise ValueError("A previously verified score artifact changed")
        return target
    for path in seed_audits:
        value = read_json(path)
        if value["evaluation"] == entry["directory"] and existing_audit_matches(value, entry):
            json_save(target, dict(value, reused_verified_audit=str(path), reused_audit_sha256=sha256(path)))
            return target
    value = audit(Path(entry["directory"]), data)
    json_save(target, value)
    return target


def finish(chain, output, audits):
    plan, completion = read_json(chain / "plan.json"), read_json(chain / "completion.json")
    if completion["status"] != "completed":
        raise ValueError("The training/evaluation coordinator has not completed")
    validate_completed_scope(completion)
    if set(audits) != EXPECTED_WINDOWS:
        raise ValueError("Not every required scoring window has an independent audit")
    for entry in completion["evaluations"]:
        if not existing_audit_matches(read_json(audits[(entry["role"], entry["day"])]), entry):
            raise ValueError("A scored artifact changed after its independent audit")
    data = Path(plan["data"])
    freeze = read_json(data / "formal_recipe_freeze.json")
    if freeze["plan_sha256"] != sha256(chain / "plan.json"):
        raise ValueError("Confirmation used a different frozen chain plan")
    versions = completion["versions"]
    config = read_json(plan["config"])
    checkpoint_evidence = []
    for index, version in enumerate(versions):
        info = checkpoint_info(version["checkpoint"], expected_parent=versions[index - 1]["sha256"] if index else None,
                               verify_bytes=True)
        if (info["checkpoint_sha256"] != version["sha256"] or info["metadata"]["proposal"] != config
                or info["metadata"]["source_binding"] != plan["source_binding"]):
            raise ValueError("Completed model bytes or configuration changed")
        if index:
            actual = audit_update(Path(version["checkpoint"]).parent, versions[index - 1]["sha256"],
                                  version["day"], plan["schedule"][index - 1], config, plan["source_binding"])
            if actual != version:
                raise ValueError("Completed update's actual parent/selection differs")
        elif (info["metadata"]["phase"] != "alignment" or info["metadata"]["days"] != [24, 25, 26]
              or version["sha256"] != plan["selected"]["checkpoint_sha256"]):
            raise ValueError("Foundation does not match the native-validation selection")
        checkpoint_evidence.append({"version": index, "checkpoint": version["checkpoint"], "sha256": version["sha256"],
                                    "bytes": info["bytes"], "actual_training_days": info["metadata"]["days"],
                                    "parent_sha256": info["metadata"]["parent_checkpoint_sha256"]})
    rows = []
    for entry in completion["evaluations"]:
        summary = read_json(Path(entry["directory"]) / "summary.json")
        current = summary["metrics"]["current"]["ndcg@10"]
        reuse = summary["metrics"]["reuse"]["ndcg@10"]
        difference = summary["paired_current_minus"]["reuse"]
        version = min(entry["day"] - 27, 9)
        if (summary["checkpoint_sha256"] != versions[version]["sha256"]
                or summary["parent_sha256"] != versions[version - 1]["sha256"]
                or summary["history_limit"] != 1024 or summary["first_only"]):
            raise ValueError("Completed quality comparison uses the wrong model pair or history")
        native = all(summary["paired_current_minus"][name]["ci95"][0] > 0 for name in ["random", "popularity"])
        rows.append({"version": version, "role": entry["role"], "evaluation_day": entry["day"],
                     "requests": summary["requests"], "positive_requests": summary["positive_requests"],
                     "positive_users": summary["positive_users"], "parent_ndcg10": summary["metrics"]["parent"]["ndcg@10"],
                     "rebuild_ndcg10": current, "reuse_ndcg10": reuse, "rebuild_minus_reuse": difference["difference"],
                     "reuse_relative_loss_percent": 100 * (current - reuse) / current if current else None,
                     "ci95_low": difference["ci95"][0], "ci95_high": difference["ci95"][1],
                     "native_beats_both_baselines": native,
                     "clear_cache_loss": difference["ci95"][0] > 0, "summary": str(Path(entry["directory"]) / "summary.json"),
                     "score_audit": str(audits[(entry["role"], entry["day"])])})
    rows.sort(key=lambda row: (row["role"], row["evaluation_day"]))
    confirmation = [row for row in rows if row["role"] == 3 and row["evaluation_day"] <= 36]
    development = [row for row in rows if row["role"] == 2]
    useful = sum(row["clear_cache_loss"] and row["native_beats_both_baselines"] for row in confirmation)
    # Actual continuous elapsed time also includes predecessor handoff/admission,
    # which is outside the coordinator's own timer. Audit this gap explicitly.
    pilot = read_json(Path(plan["pilot"]) / "plan.json")
    end = plan["started_unix"] + completion["total_wall_seconds_conservative"] - plan["prior_spent_seconds"]
    continuous_seconds = end - pilot["started_unix"] + pilot["prior_small_pilot_spent_seconds_conservative"]
    foundation_upper = read_json(Path(plan["pilot"]) / "completion.json")["Max_wall_seconds"]
    within_budget = continuous_seconds <= plan["maximum_total_seconds"] and foundation_upper <= plan["maximum_foundation_seconds"]
    with (output / "all_windows.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"status": "artifacts_verified", "models": checkpoint_evidence, "windows": rows,
              "total_models": len(versions), "total_audited_windows": len(rows),
              "history": 1024, "scale": "max", "native_confirmation_baselines_passed": sum(row["native_beats_both_baselines"] for row in confirmation),
              "clear_development_cache_edges": sum(row["clear_cache_loss"] for row in development),
              "clear_confirmation_cache_edges": sum(row["clear_cache_loss"] for row in confirmation),
              "useful_confirmation_cache_edges": useful, "requested_several_edges_observed": useful >= 3,
              "primary_confirmation_v0_to_v1": confirmation[0],
              "uncertainty": "nominal paired UID 95% intervals; secondary edge counts are descriptive, not familywise significance",
              "continuous_training_evaluation_seconds_including_small_pilot": continuous_seconds,
              "coordinator_reported_seconds": completion["total_wall_seconds_conservative"],
              "handoff_admission_seconds_added": continuous_seconds - completion["total_wall_seconds_conservative"],
              "foundation_conservative_upper_seconds": foundation_upper,
              "foundation_upper_bound_includes": "the entire bounded Max experiment, including its first update and development scores",
              "within_frozen_training_budget": within_budget,
              "goal_quality_and_artifacts_satisfied": useful >= 3 and within_budget,
              "scope": "observed-positive full logged-candidate ranking; matched-prefix snapshots, not persistent serving",
              "plan_sha256": sha256(chain / "plan.json"), "completion_sha256": sha256(chain / "completion.json"),
              "verifier_sha256": sha256(__file__)}
    json_save(output / "completion_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--seed-audits", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "driver.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "plan.json").exists():
            raise ValueError("Preserve the verifier attempt; inspect it before choosing a new output")
        plan = read_json(args.chain / "plan.json")
        start = process_start_ticks(args.watch_pid) if args.watch_pid else None
        if args.watch_pid and not start and not (args.chain / "completion.json").exists():
            raise ValueError("The specified coordinator is not live and has no completion")
        hashes = dict(plan["source_binding"], **plan["driver_sources"])
        hashes.update(read_json(Path(plan["pilot"]) / "source_snapshot/manifest.json"))
        for path in [Path(__file__), HERE / "audit_scored_requests.py", args.chain / "plan.json"]:
            hashes[str(path)] = sha256(path)
        json_save(args.output / "plan.json", {"chain": str(args.chain), "pid": os.getpid(), "watch_pid": args.watch_pid,
                  "watch_process_start_ticks": start, "source_sha256": hashes, "minimum_useful_confirmation_edges": 3,
                  "confirmation_unopened_at_creation": not any(entry["role"] == 3 for entry in read_json(args.chain / "progress.json")["evaluations"])})
        shutil.copyfile(__file__, args.output / "verifier_snapshot.py")
        audited = {}
        while True:
            for path, digest in hashes.items():
                if sha256(path) != digest:
                    raise ValueError(f"Frozen verification input/source changed: {path}")
            progress = read_json(args.chain / "completion.json" if (args.chain / "completion.json").exists() else args.chain / "progress.json")
            for entry in progress["evaluations"]:
                key = (entry["role"], entry["day"])
                if key not in EXPECTED_WINDOWS:
                    raise ValueError("Controller added an undeclared evaluation window")
                if key not in audited:
                    audited[key] = verify_window(entry, Path(plan["data"]), args.output, args.seed_audits)
                    json_save(args.output / "progress.json", {"status": "watching", "verified_windows": [list(key) for key in audited],
                              "last_completed_audit": str(audited[key]), "checked_unix": time.time()})
                    print(json.dumps({"status": "score_audit_passed", "window": key, "audit": str(audited[key])}), flush=True)
            if (args.chain / "completion.json").exists():
                result = finish(args.chain, args.output, audited)
                print(json.dumps({key: result[key] for key in ["status", "total_models", "total_audited_windows",
                      "useful_confirmation_cache_edges", "within_frozen_training_budget", "goal_quality_and_artifacts_satisfied"]}), flush=True)
                break
            if not args.watch_pid:
                json_save(args.output / "incomplete.json", {"status": "waiting_for_remaining_models_and_windows",
                          "verified_windows": [list(key) for key in audited]})
                break
            if process_start_ticks(args.watch_pid) != start:
                if (args.chain / "completion.json").exists():
                    continue
                raise RuntimeError("Watched coordinator ended without completion; existing artifacts preserved")
            time.sleep(10)


if __name__ == "__main__":
    main()
