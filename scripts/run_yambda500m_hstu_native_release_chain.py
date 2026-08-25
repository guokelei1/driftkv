#!/usr/bin/env python3
"""Train an HSTU-native v0 and independently admit one complete-epoch v1."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_release_chain_v1.yaml"
MANIFESTS = ROOT / "data/manifests/yambda500m_small_five_version_v1"
RESULTS = ROOT / "results/yambda500m_small_seed17/hstu_native_release_chain_v1"


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def train(version: str, output: Path, env: dict[str, str], *, parent: Path | None = None, canary: bool = False) -> None:
    command = ["torchrun", "--standalone", "--nproc_per_node=4", "scripts/train_yambda500m_foundation_fsdp.py",
               "--version", version, "--launch-contract", str(CONTRACT), "--manifest-dir", str(MANIFESTS),
               "--output", str(output), "--oov-buckets", "256", "--passes", "1"]
    if parent:
        command.extend(["--parent", str(parent)])
    if version == "v1":
        command.extend(["--train-start-day", "217", "--train-end-day", "222"])
    if canary:
        command.extend(["--canary-steps", "2", "--batch-size", "8"])
    run(command, env)


def admission(v0: Path, v1: Path, env: dict[str, str]) -> Path:
    output = RESULTS / "v1_causal_admission"; report = output / "adjudication.json"
    if report.exists(): return report
    if output.exists(): raise RuntimeError(f"partial admission result requires audit: {output}")
    command = ["torchrun", "--standalone", "--nproc_per_node=4", "scripts/evaluate_yambda500m_release_candidates_raw.py",
               "--stage", "hstu_native_v1_admission", "--block", "update1", "--training-block", "update1",
               "--start-day", "222", "--end-day", "224", "--manifest-dir", str(MANIFESTS),
               "--parent", f"v0={v0}", "--current", f"candidate_v1={v1}"]
    with tempfile.TemporaryDirectory(prefix="evokv_native_admission_canary_") as temporary:
        probe = Path(temporary) / "probe"
        run([*command, "--output", str(probe), "--max-users", "2", "--batch-size", "8"], env)
        run([sys.executable, "scripts/adjudicate_yambda500m_release_candidates.py", "--raw", str(probe / "raw.parquet"),
             "--seal", str(probe / "raw.seal.json"), "--labels", str(MANIFESTS / "requests_quality.parquet"),
             "--output", str(probe / "adjudication.json")], env)
    run([*command, "--output", str(output), "--batch-size", "64"], env)
    run([sys.executable, "scripts/adjudicate_yambda500m_release_candidates.py", "--raw", str(output / "raw.parquet"),
         "--seal", str(output / "raw.seal.json"), "--labels", str(MANIFESTS / "requests_quality.parquet"),
         "--output", str(report)], env)
    return report


def seal_decision(report_path: Path) -> None:
    report = json.loads(report_path.read_text()); candidate = report["candidates"]["candidate_v1"]
    paired = candidate["paired_release_gain"]["parent_minus_current_log_loss"]
    lower = paired["user_cluster_bootstrap_95CI"]["p2_5"]; event_gain = paired["event_weighted_mean"]
    parent_brier = report["parent_absolute"]["hstu_native"]["Brier"]
    current_brier = candidate["absolute"]["hstu_native"]["Brier"]
    accepted = lower > 0 and event_gain > 0 and current_brier <= parent_brier
    (report_path.parent / "admission.seal.json").write_text(json.dumps({
        "status": "accepted_release" if accepted else "rejected_candidate", "architecture": "hstu_native_cc",
        "contract": str(CONTRACT.relative_to(ROOT)), "admission_report": str(report_path.relative_to(ROOT)),
        "candidate": "candidate_v1", "parent": "v0", "contains_reuse": False, "reuse_unlocked": accepted,
        "criteria": {"user_equal_bootstrap_lower": lower, "event_weighted_parent_minus_current_log_loss": event_gain,
                     "parent_brier": parent_brier, "current_brier": current_brier},
    }, indent=2) + "\n")


def main() -> None:
    env = {**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": "0,1,2,3", "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    v0 = RESULTS / "v0" / "checkpoint_100.pt"; v1 = RESULTS / "v1_epoch1_candidate" / "checkpoint_100.pt"
    if not v0.exists():
        if (RESULTS / "v0").exists(): raise RuntimeError("partial v0 result requires audit")
        with tempfile.TemporaryDirectory(prefix="evokv_native_v0_canary_") as temporary: train("v0", Path(temporary) / "v0", env, canary=True)
        train("v0", RESULTS / "v0", env)
    if not v1.exists():
        if (RESULTS / "v1_epoch1_candidate").exists(): raise RuntimeError("partial v1 result requires audit")
        with tempfile.TemporaryDirectory(prefix="evokv_native_v1_canary_") as temporary: train("v1", Path(temporary) / "v1", env, parent=v0, canary=True)
        train("v1", RESULTS / "v1_epoch1_candidate", env, parent=v0)
    seal_decision(admission(v0, v1, env))


if __name__ == "__main__":
    main()
