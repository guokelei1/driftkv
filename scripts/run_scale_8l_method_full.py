#!/usr/bin/env python3
"""One-command resumable 8L EvoKV method-validation queue.

Stages are mechanical and fail closed: R0 training, four action canaries, four
all-state raw action cells, raw seal/adjudication, frozen scheduler replay,
three rolling quality cells, and final quality adjudication.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/scale_8l_v1"
LOGS = RESULT / "method_logs"
RELEASES = ("r0", "r1_edge1", "r1_edge2", "r2")


def env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": "src:scripts", "OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "16"}


def run(command: list[str], log: Path | None = None) -> None:
    print("RUN", shlex.join(command), flush=True)
    if log is None:
        subprocess.run(command, cwd=ROOT, env=env(), check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env(), stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"command failed ({result.returncode}); inspect {log}")


def parallel(commands: list[tuple[list[str], Path]]) -> None:
    with ThreadPoolExecutor(max_workers=len(commands)) as pool:
        futures = {pool.submit(run, command, log): log for command, log in commands}
        for future in as_completed(futures): future.result()


def artifact(path: str) -> bool: return (ROOT / path).exists()


def stages() -> list[dict]:
    return [
        {"name":"population", "done":artifact("data/manifests/scale_8l_population_v1/materialization_summary.json")},
        {"name":"r0_training", "done":artifact("results/scale_8l_v1/releases/r0/m0_f_seed17/train_result.json")},
        {"name":"action_canaries", "done":all(artifact(f"results/scale_8l_v1/actions_raw/canary16/{r}/m0_f_seed17/raw_manifest.json") for r in RELEASES)},
        {"name":"action_raw_full", "done":all(artifact(f"results/scale_8l_v1/actions_raw/full/{r}/m0_f_seed17/raw_manifest.json") for r in RELEASES)},
        {"name":"action_adjudication", "done":artifact("results/scale_8l_v1/actions_adjudication_v1.json")},
        {"name":"scheduler", "done":artifact("results/scale_8l_v1/scheduler/assignment_seal.json")},
        {"name":"mixed_policy_runtime", "done":all(artifact(f"results/scale_8l_v1/policy_runtime/{r}/m0_f_seed17/result.json") for r in RELEASES[1:])},
        {"name":"policy_quality_raw", "done":all(artifact(f"results/scale_8l_v1/policy_quality_raw/{r}/m0_f_seed17/raw_manifest.json") for r in RELEASES[1:])},
        {"name":"policy_quality_adjudication", "done":artifact("results/scale_8l_v1/policy_quality_adjudication_v1.json")},
        {"name":"user_weighted_quality_v2", "done":artifact("results/scale_8l_v1/policy_quality_adjudication_v2.json")},
        {"name":"final_summary", "done":artifact("results/scale_8l_v1/method_summary_v2.json")},
    ]


def execute() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    if not artifact("data/manifests/scale_8l_population_v1/materialization_summary.json"):
        run(["python","scripts/build_scale_8l_population.py"])
    if not artifact("results/scale_8l_v1/releases/r0/m0_f_seed17/train_result.json"):
        run(["torchrun","--standalone","--nproc_per_node=4","scripts/train_scale_8l_fsdp_release.py",
            "--release","r0","--model","m0_f","--seed","17"], LOGS/"r0_training.log")
    pending=[]
    for gpu, release in enumerate(RELEASES):
        if not artifact(f"results/scale_8l_v1/actions_raw/canary16/{release}/m0_f_seed17/raw_manifest.json"):
            pending.append((["python","scripts/eval_scale_8l_actions_raw.py","--release",release,
                "--device",f"cuda:{gpu}","--state-limit","16"], LOGS/f"canary_{release}.log"))
    if pending: parallel(pending)
    pending=[]
    for gpu, release in enumerate(RELEASES):
        if not artifact(f"results/scale_8l_v1/actions_raw/full/{release}/m0_f_seed17/raw_manifest.json"):
            pending.append((["python","scripts/eval_scale_8l_actions_raw.py","--release",release,
                "--device",f"cuda:{gpu}"], LOGS/f"actions_full_{release}.log"))
    if pending: parallel(pending)
    if not artifact("results/scale_8l_v1/actions_raw_seal_v1.json"): run(["python","scripts/seal_scale_8l_actions_raw.py"])
    if not artifact("results/scale_8l_v1/actions_adjudication_v1.json"): run(["python","scripts/adjudicate_scale_8l_actions.py"])
    if not artifact("results/scale_8l_v1/scheduler/assignment_seal.json"): run(["python","scripts/eval_scale_8l_scheduler.py"])
    pending=[]
    for gpu, release in enumerate(RELEASES[1:]):
        if not artifact(f"results/scale_8l_v1/policy_runtime/{release}/m0_f_seed17/result.json"):
            pending.append((["python","scripts/eval_scale_8l_policy_runtime.py","--release",release,
                "--device",f"cuda:{gpu}"], LOGS/f"policy_runtime_{release}.log"))
    if pending: parallel(pending)
    pending=[]
    for gpu, release in enumerate(RELEASES[1:]):
        if not artifact(f"results/scale_8l_v1/policy_quality_raw/{release}/m0_f_seed17/raw_manifest.json"):
            pending.append((["python","scripts/eval_scale_8l_policy_quality_raw.py","--release",release,
                "--device",f"cuda:{gpu}"], LOGS/f"policy_quality_{release}.log"))
    if pending: parallel(pending)
    if not artifact("results/scale_8l_v1/policy_quality_adjudication_v1.json"):
        run(["python","scripts/adjudicate_scale_8l_policy_quality.py"])
    if not artifact("results/scale_8l_v1/method_summary_v1.json"):
        run(["python","scripts/summarize_scale_8l_method.py"])
    if not artifact("results/scale_8l_v1/policy_quality_adjudication_v2.json"):
        run(["python","scripts/adjudicate_scale_8l_policy_quality_v2.py"])
    if not artifact("results/scale_8l_v1/method_summary_v2.json"):
        run(["python","scripts/summarize_scale_8l_method_v2.py"])
    print(json.dumps({"status":"scale_8l_method_full_queue_complete","stages":stages()},indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run",action="store_true"); parser.add_argument("--status",action="store_true")
    args=parser.parse_args()
    if args.run: execute()
    else: print(json.dumps({"status":"ready" if not all(s["done"] for s in stages()) else "complete","stages":stages(),
        "launch_command":"PYTHONPATH=src:scripts python scripts/run_scale_8l_method_full.py --run",
        "GPUs":[0,1,2,3],"resumable":True,"theta3_access":False},indent=2))


if __name__=="__main__": main()
