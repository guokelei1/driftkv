"""Prepare one complete daily update and its next-day 4096-UID pilot.

Preparation only reads CPU metadata and request indices. Actual model/AdamW
loading remains the existing trainer's responsibility. No successor is queued.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from experiments.recflow_full_population.code.configuration import load_proposal
from experiments.recflow_full_population.code.protocol import day_bounds_ms, day_date
from experiments.recflow_release_design.code.interval_protocol import ancestry


BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
ORIGINAL = REPO / "experiments/recflow_full_population"
DATA = ORIGINAL / "data"


def read(path):
    return json.loads(Path(path).read_text())


def resolve_recorded(path):
    path = Path(path)
    return (path if path.is_absolute() else REPO / path).resolve()


def cache_source_path(training_parent, cache_source=None):
    """Check the existing optimizer path; never substitute publication labels."""
    parent = resolve_recorded(training_parent)
    source = parent if cache_source is None else resolve_recorded(cache_source)
    if source != parent:
        path = ancestry(parent, source)
    else:
        if not parent.is_file():
            raise FileNotFoundError(parent)
        meta = read(parent.with_suffix(".json"))["metadata"]
        days = meta.get("days")
        if not days or any(type(day) is not int or not 1 <= day <= 37 for day in days):
            raise ValueError("Training parent lacks valid actual training days")
        raw_parent = meta.get("arguments", {}).get("parent")
        path = [{"checkpoint": str(parent), "train_days": days, "train_end_day": max(days),
                 "actual_parent": str(resolve_recorded(raw_parent)) if raw_parent else None}]
    for entry in path:
        sidecar = read(Path(entry["checkpoint"]).with_suffix(".json"))
        meta = sidecar["metadata"]
        if (sidecar.get("status") != "committed" or meta.get("quality_use_permitted") is not True
                or meta.get("scale") != "max"
                or meta.get("proposal", {}).get("history", {}).get("max_events") != 1024):
            raise ValueError("Cache-source ancestry requires committed quality Max/1024 checkpoints")
        if resolve_recorded(meta["arguments"]["data"]) != DATA.resolve():
            raise ValueError("Cache source and training parent must use the same full-population data")
    return path


def prepare(args):
    evaluation_name = args.name + "_pilot_4096"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,69}", evaluation_name):
        raise ValueError("Name plus _pilot_4096 must be a directory name of at most 70 safe characters")
    if not args.source_label.strip() or not args.target_label.strip():
        raise ValueError("Source and target labels must be nonempty")
    if args.question is not None and not args.question.strip():
        raise ValueError("An explicit experimental question must be nonempty")
    if not 28 <= args.day <= 36:
        raise ValueError("Daily updates use D28..36, followed by evaluation on D29..37")
    if not math.isfinite(args.listwise_weight) or args.listwise_weight < 0:
        raise ValueError("Listwise weight must be finite and nonnegative")
    if args.dry_run and args.launch:
        raise ValueError("Dry-run cannot launch")
    run = BASE / "training_runs" / args.name
    evaluation = BASE / "runs" / evaluation_name
    if run.exists() or evaluation.exists():
        raise FileExistsError("Preserve existing training and evaluation directories; choose a new name")

    baseline = ORIGINAL / "runs/attempt_1/training" / f"m{args.day - 27}_d{args.day}" / "plan.json"
    original_plan = read(baseline)
    original_args = original_plan["arguments"]
    expected = {"days": [args.day], "phase": "update", "scale": "max", "seed": 17,
                "epochs": 1, "global_batch": 64, "micro_batch": 16, "workers": 4,
                "communication": "rows", "dtype": "bfloat16", "resume": None,
                "max_steps": None, "request_limit": None, "stop_after_steps": None,
                "pilot": False, "diagnostic_state": False}
    if any(original_args.get(key) != value for key, value in expected.items()):
        raise ValueError("Original same-day plan is not the complete one-epoch Max training recipe")
    if original_plan["world_size"] != 4 or original_plan["history_limit"] != 1024:
        raise ValueError("Expected original four-rank, history-1024 training")
    if resolve_recorded(original_args["data"]) != DATA.resolve():
        raise ValueError("Original plan uses another data root")
    config_path = resolve_recorded(original_args["config"])
    config = load_proposal(config_path)

    parent = args.parent_checkpoint.resolve()
    if not parent.is_file():
        raise FileNotFoundError(f"Actual parent checkpoint is missing: {parent}")
    sidecar = read(parent.with_suffix(".json"))
    metadata = sidecar["metadata"]
    days = metadata.get("days")
    if (not isinstance(days, list) or not days
            or any(type(day) is not int or not 1 <= day <= 37 for day in days)
            or max(days) != args.day - 1):
        raise ValueError("Parent's recorded training cutoff must be exactly the preceding day")
    if sidecar.get("status") != "committed" or metadata.get("quality_use_permitted") is not True:
        raise ValueError("Parent must be a committed quality checkpoint")
    if resolve_recorded(metadata["arguments"]["data"]) != DATA.resolve():
        raise ValueError("Parent belongs to another data/cohort lineage")
    parent_config = metadata["proposal"]
    if (metadata.get("scale") != "max"
            or parent_config["architecture"]["max"] != config["architecture"]["max"]
            or parent_config["history"]["max_events"] != 1024):
        raise ValueError("Parent must retain the actual Max backbone and history cap 1024")
    existing_path = cache_source_path(parent, getattr(args, "cache_source_checkpoint", None))
    cache_source = Path(existing_path[0]["checkpoint"])

    manifest = read(DATA / "selection_manifest.json")["datasets"]
    with np.load(DATA / "population.npz") as population:
        train_uids = np.array(population["train"], copy=True)
    if len(train_uids) != 30000 or len(np.unique(train_uids)) != 30000:
        raise ValueError("Expected the existing fixed 30000-user training cohort")
    selections = {}
    for role, day in ((0, args.day), (2, args.day + 1)):
        entry = manifest[f"{role}:{day}"]
        path = DATA / entry["file"]
        rows = np.load(path, mmap_mode="r")
        # The row's inherited role field is not this full-population selection
        # role. The manifest key chooses training/evaluation request indices.
        if (len(rows) != entry["requests"] or not len(rows)
                or not np.all(rows["day"] == day)
                or not np.isin(rows["uid"], train_uids).all()):
            raise ValueError(f"Actual request selection disagrees with role {role}, day {day}, or cohort")
        selections[role] = {"path": str(path), "requests": len(rows),
                            "active_users": int(len(np.unique(rows["uid"])))}
        if role == 2:
            chosen = np.random.default_rng(20261029).permutation(np.sort(train_uids))[:4096]
            selected = rows[np.isin(rows["uid"], chosen)]
            selections[role].update(pilot_requests=len(selected),
                                    pilot_active_users=int(len(np.unique(selected["uid"]))))
    requests = selections[0]["requests"]
    steps = math.ceil(requests / original_args["global_batch"])
    if requests != original_plan["requests"] or steps != original_plan["planned_steps"]:
        raise ValueError("Actual complete-day request count/steps differ from the original same-day plan")

    training_output = run / "training"
    training_args = dict(original_args, parent=str(parent), listwise_weight=args.listwise_weight,
                         output=str(training_output), log_every=50)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               "-m", "experiments.recflow_full_population.code.train"]
    for key, value in sorted(training_args.items()):
        if value is None or value is False:
            continue
        command.append("--" + key.replace("_", "-"))
        if value is not True:
            command.extend(map(str, value if isinstance(value, list) else [value]))
    question = args.question or (
        f"Does the complete one-pass D{args.day} update with listwise weight {args.listwise_weight} "
        f"produce positive Rebuild minus Reuse NDCG@10 against the declared published cache source on D{args.day + 1}? "
        "Retain every outcome; native quality is diagnostic, not a predecessor-superiority requirement.")
    scenario = (
        f"Published cache source has training cutoff D{existing_path[0]['train_end_day']}. "
        f"The {len(existing_path)-1} subsequent existing optimizer checkpoint(s) are treated as unpublished "
        f"in this release scenario; the actual training parent has cutoff D{max(days)}. "
        f"Train that parent on complete D{args.day} once and compare the resulting target's Rebuild "
        f"with the published source's cache on complete D{args.day+1}."
    )
    spec = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "question": question, "parent": str(parent), "parent_training_cutoff_day": max(days),
            "training_parent": str(parent), "cache_source": str(cache_source),
            "cache_source_to_training_parent_path": existing_path,
            "cache_source_training_cutoff_day": existing_path[0]["train_end_day"],
            "publication_scenario": scenario,
            "training_day": args.day, "training_date": day_date(args.day),
            "listwise_weight": args.listwise_weight, "training_output": str(training_output),
            "training_command": command, "training_requests": requests, "training_steps": steps,
            "epochs": 1, "baseline_plan": str(baseline),
            "training_argument_changes": {key: {"original": original_args[key], "new": value}
                                          for key, value in training_args.items() if value != original_args[key]},
            "training_selection": selections[0], "evaluation_directory": str(evaluation),
            "evaluation_day": args.day + 1, "evaluation_date": day_date(args.day + 1),
            "evaluation_interval_ms_half_open": list(day_bounds_ms(args.day + 1)),
            "evaluation_users": 4096, "evaluation_seed": 20261029, "evaluation_user_offset": 0,
            "evaluation_selection_preview": selections[2],
            "evaluation_rule": "All requests and all logged candidates of the fixed chosen users on the next complete day",
            "source_label": args.source_label.strip(), "target_label": args.target_label.strip(),
            "physical_gpus": [0, 1, 2, 3], "history_limit": 1024,
            "optimizer": {"state": "Existing trainer loads the actual parent model and complete AdamW state",
                          "daily_scheduler": config["optimization_proposal"]["scheduler_on_update"],
                          "learning_rate": config["optimization_proposal"]["update_learning_rate"],
                          "warmup_fraction": config["optimization_proposal"]["warmup_fraction"]},
            "preparation_limit": "CPU sidecars and request indices checked; checkpoint tensors are loaded by the trainer at execution",
            "automatic_successors": False,
            "claim": "One actual daily update and one development pilot; no replication or following model is automatic"}
    driver = [sys.executable, "-m", "experiments.recflow_continuous_chain.code.run_train_pair", "--run", str(run)]
    if not args.dry_run:
        run.parent.mkdir(parents=True, exist_ok=True)
        run.mkdir()
        (run / "experiment.json").write_text(json.dumps(spec, indent=2) + "\n")
        if args.launch:
            script = run / "launch.sh"
            script.write_text("#!/bin/bash\nset -e\ncd " + shlex.quote(str(REPO)) +
                              "\nexport PYTHONPATH=" + shlex.quote(str(REPO / "src") + ":" + str(REPO)) +
                              "\nexec " + shlex.join(driver) + " > " + shlex.quote(str(run / "driver.log")) + " 2>&1\n")
            subprocess.run(["tmux", "new-session", "-d", "-s", "continuous-update-" + args.name,
                            "bash", str(script)], check=True)
    return {"dry_run": args.dry_run, "launched": args.launch, "run": str(run),
            "driver_command": driver, "experiment": spec}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", "--parent", type=Path, required=True)
    parser.add_argument("--cache-source-checkpoint", "--cache-source", type=Path,
                        help="Published cache source; defaults to the training parent, otherwise must be its actual ancestor")
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--listwise-weight", type=float, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--question")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Read and print the complete plan without writing any files")
    mode.add_argument("--launch", action="store_true", help="Explicitly launch only this update and its next-day pilot in tmux")
    print(json.dumps(prepare(parser.parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()
