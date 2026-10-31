"""Prepare and optionally dispatch one explicitly selected release-pair experiment."""
import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.recflow_full_population.code.data_common import json_save
from .interval_protocol import ancestry, prepare_selection


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=int, required=True)
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--users", type=int, default=4096)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--name", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--launch", action="store_true")
    args = p.parse_args()
    base = Path(__file__).resolve().parents[1]
    repo = base.parents[1]
    data = repo / "experiments/recflow_full_population/data"
    original = repo / "experiments/recflow_full_population/runs/attempt_1"
    versions = {v["version"]: v for v in json.loads((original / "model_chain_inventory.json").read_text())["versions"]}
    target, source = versions[args.target], versions[args.source]
    lineage = ancestry(target["checkpoint"], source["checkpoint"])
    day = lineage[-1]["train_end_day"] + 1
    run = base / "release_interval/runs" / args.name
    if run.exists():
        raise ValueError("Retain previous results; use a new experiment name")
    selection = prepare_selection(data, day, run, args.users, user_offset=args.offset)
    plan = dict(created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                question=args.question, source_version=args.source, target_version=args.target,
                checkpoint=target["checkpoint"], cache_source=source["checkpoint"], eval_day=day,
                data=str(data), native_reference=target["evaluation"], selection=selection,
                batch_size=16, physical_gpus=[0, 1, 2, 3], actual_training_path=lineage,
                primary_metric="request-macro observed-positive NDCG@10", secondary_metric="Recall@10",
                history_limit=1024, development_only=True, automatic_successors=False,
                subgroups="chosen UID order even/odd; full-day first12h/second12h, all reported")
    json_save(run / "plan.json", plan)
    if args.launch:
        command = [sys.executable, "-m", "experiments.recflow_release_design.code.interval_one", "--run", str(run)]
        script = run / "launch.sh"
        script.write_text("#!/bin/bash\nset -e\ncd " + shlex.quote(str(repo)) +
                          "\nexport PYTHONPATH=src:.\nexec " + shlex.join(command) +
                          " > " + shlex.quote(str(run / "driver.log")) + " 2>&1\n")
        subprocess.run(["tmux", "new-session", "-d", "-s", "interval-" + args.name, "bash", str(script)], check=True)
    print(json.dumps(dict(run=str(run), selection=selection, launched=args.launch)))


if __name__ == "__main__":
    main()
