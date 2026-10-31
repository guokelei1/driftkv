"""Prepare one explicitly chosen causal release pair; launch only with --launch."""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.recflow_full_population.code.data_common import json_save
from experiments.recflow_full_population.code.protocol import day_bounds_ms, day_date
from experiments.recflow_release_design.code.interval_protocol import ancestry, prepare_selection


BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
DATA = REPO / "experiments/recflow_full_population/data"


def read(path):
    return json.loads(Path(path).read_text())


def validate_day(day, target_cutoff, scenario):
    day = target_cutoff + 1 if day is None else day
    if not 28 <= day <= 37 or day <= target_cutoff:
        raise ValueError("Evaluation must follow target training and use D28..37 after D27 catalog/cohort availability")
    if day != target_cutoff + 1 and not (scenario or "").strip():
        raise ValueError("A later evaluation day requires --scenario explaining its publication or serving context")
    return day


def validate_data(lineage):
    for entry in lineage:
        metadata = read(Path(entry["checkpoint"]).with_suffix(".json"))["metadata"]
        recorded = metadata.get("arguments", {}).get("data")
        if not recorded:
            raise ValueError("Checkpoint lacks its actual data-root record")
        path = Path(recorded)
        path = path if path.is_absolute() else REPO / path
        if path.resolve() != DATA.resolve():
            raise ValueError("This helper uses full_population vocabulary/cohort; another data lineage cannot be substituted")


def validate_reference(reference, checkpoint, day):
    if reference is None:
        return None
    reference = reference.resolve()
    summary = read(reference / "summary.json")
    recorded = summary.get("checkpoint") or summary.get("arguments", {}).get("checkpoint")
    if (not recorded or Path(recorded).resolve() != checkpoint.resolve()
            or summary.get("days") != [day]
            or summary.get("status") not in ("passed", "completed")):
        raise ValueError("Native reference must be completed for this actual target and exact evaluation day")
    if summary.get("role") != 2 or summary.get("first_only") or summary.get("world_size") != 4:
        raise ValueError("Native reference must use the original full-day four-rank evaluation layout")
    manifest = read(DATA / "selection_manifest.json")
    if summary.get("requests") != manifest["datasets"][f"2:{day}"]["requests"]:
        raise ValueError("Native reference does not cover the original complete future-day selection")
    for rank in range(4):
        if not (reference / f"rank_{rank}_scores/batch_000000.npz").is_file():
            raise ValueError("Native reference is missing an original first score batch")
    return str(reference)


def prepare(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,69}", args.name):
        raise ValueError("Experiment name must be one short alphanumeric/underscore/hyphen directory name")
    if not all(value.strip() for value in (args.source_label, args.target_label, args.question)):
        raise ValueError("Source label, target label and experimental question must be nonempty")
    if args.users < 1 or args.offset < 0 or args.offset + args.users > 30000:
        raise ValueError("Requested UID slice must lie within the fixed 30000-user permutation")
    run = BASE / "runs" / args.name
    if run.exists():
        raise FileExistsError("Keep the existing experiment; choose a new name")
    source, target = args.source_checkpoint.resolve(), args.target_checkpoint.resolve()
    lineage = ancestry(target, source)
    validate_data(lineage)
    cutoff = lineage[-1]["train_end_day"]
    day = validate_day(args.day, cutoff, args.scenario)
    reference = validate_reference(args.native_reference, target, day)
    selection = prepare_selection(DATA, day, run, args.users, seed=20261029, user_offset=args.offset)
    scenario = (args.scenario or "Target publication after its training cutoff; evaluate every selected user's request on the immediately following complete natural day.").strip()
    plan = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "question": args.question.strip(), "source_label": args.source_label.strip(),
            "target_label": args.target_label.strip(), "checkpoint": str(target),
            "cache_source": str(source), "eval_day": day, "evaluation_date": day_date(day),
            "evaluation_interval_ms_half_open": list(day_bounds_ms(day)),
            "data": str(DATA), "native_reference": reference, "selection": selection,
            "batch_size": 16, "physical_gpus": [0, 1, 2, 3], "actual_training_path": lineage,
            "primary_metric": "request-macro observed-positive NDCG@10", "secondary_metric": "Recall@10",
            "history_limit": 1024, "development_only": True, "automatic_successors": False,
            "scenario": scenario, "target_training_cutoff_day": cutoff,
            "evaluation_delay_days_after_training_cutoff": day-cutoff,
            "window_choice": "One explicitly requested complete day; no automatic window selection or metric-based window search",
            "input_availability": {"vocabulary_and_fixed_population_cutoff_day": 27,
                                   "earliest_allowed_evaluation_day": 28},
            "subgroups": "chosen UID order even/odd; full-day first12h/second12h, all reported",
            "protocol": "matched-prefix snapshot for the actual recorded ancestor-to-descendant path; not persistent serving"}
    # Labels describe publications, while versions are only supplied when the
    # checkpoint is one of the original numbered full-population models.
    inventory = REPO / "experiments/recflow_full_population/runs/attempt_1/model_chain_inventory.json"
    versions = {str(Path(row["checkpoint"]).resolve()): row["version"] for row in read(inventory)["versions"]}
    plan.update(source_version=versions.get(str(source)), target_version=versions.get(str(target)))
    json_save(run / "plan.json", plan)
    if args.launch:
        command = [sys.executable, "-m", "experiments.recflow_continuous_chain.code.run_one", "--run", str(run)]
        script = run / "launch.sh"
        script.write_text("#!/bin/bash\nset -e\ncd " + shlex.quote(str(REPO)) +
                          "\nexport PYTHONPATH=src:.\nexec " + shlex.join(command) +
                          " > " + shlex.quote(str(run / "driver.log")) + " 2>&1\n")
        subprocess.run(["tmux", "new-session", "-d", "-s", "continuous-"+args.name,
                        "bash", str(script)], check=True)
    return {"run": str(run), "selection": selection, "eval_day": day,
            "scenario": scenario, "launched": bool(args.launch), "automatic_successors": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--users", type=int, default=4096)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--question", required=True)
    parser.add_argument("--day", type=int)
    parser.add_argument("--scenario", help="Publication/serving justification; required for a non-next-day window")
    parser.add_argument("--native-reference", type=Path)
    parser.add_argument("--launch", action="store_true")
    print(json.dumps(prepare(parser.parse_args())), flush=True)


if __name__ == "__main__":
    main()
