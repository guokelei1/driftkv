#!/usr/bin/env python3
"""Complete only missing cells of the sealed D=14 one-hop Reuse diagnostic."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from run_yambda500m_hstu_native_d14_onehop_reuse import MANIFEST, MATRIX, ROOT, checkpoint, sha256


CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_d14_onehop_reuse_completion_v2.yaml"
OUTPUT = MATRIX / "d14_onehop_reuse_completion_v2"
CELLS = ((2, (1, 2, 3, 4, 5)), (4, (5,)), (7, (3, 4, 5)), (14, (3, 4, 5)))


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def evaluate(*, edge: int, horizon: int, env: dict[str, str], output: Path, cohort_size: int,
             force_fallback: bool) -> None:
    cutover = 217 + edge * 14
    command = [
        "torchrun", "--standalone", "--nproc_per_node=4", "scripts/evaluate_yambda500m_hstu_native_onehop_reuse_raw.py",
        "--stage", f"d14_onehop_completion_e{horizon}_edge{edge}", "--edge", f"v{edge - 1}_to_v{edge}",
        "--cutover-day", str(cutover), "--start-day", str(cutover), "--end-day", str(cutover + horizon),
        "--manifest-dir", str(MANIFEST), "--parent", str(checkpoint(edge - 1)), "--current", str(checkpoint(edge)),
        "--cohort-size", str(cohort_size), "--output", str(output),
    ]
    if force_fallback:
        command.insert(-2, "--force-fallback")
    run(command, env)
    run([
        sys.executable, "scripts/adjudicate_yambda500m_hstu_native_onehop_reuse.py",
        "--raw", str(output / "raw.parquet"), "--seal", str(output / "raw.seal.json"),
        "--labels", str(MANIFEST / "requests_quality.parquet"), "--output", str(output / "adjudication.json"),
    ], env)


def main() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    frozen = contract["frozen_inputs"]
    if sha256(MANIFEST / "manifest.json") != frozen["matrix_manifest_sha256"] or sha256(checkpoint(0)) != frozen["v0_sha256"]:
        raise RuntimeError("frozen D14 inputs differ from the completion contract")
    for version in range(1, 6):
        if not checkpoint(version).exists():
            raise RuntimeError(f"D14 checkpoint v{version} is absent")
    env = {**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": "0,1,2,3", "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    marker = OUTPUT / "preflight_complete.json"
    if not marker.exists():
        prior = MATRIX / "d14_onehop_reuse_diagnostic_v1" / "preflight_complete.json"
        if not prior.exists():
            raise RuntimeError("the required prior four-rank two-path canary is absent")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "status": "prior_four_rank_two_path_canary_reused",
            "prior_canary": str(prior.relative_to(ROOT)),
            "prior_canary_sha256": sha256(prior),
            "contract_sha256": sha256(CONTRACT),
            "reason": "completion reuses the already-canary-passed CurrentExactRolling_vs_OneHopReuseRolling evaluator and frozen D14 inputs",
        }, indent=2) + "\n")
    for horizon, edges in CELLS:
        for edge in edges:
            output = OUTPUT / f"eval_{horizon}d" / f"v{edge - 1}_to_v{edge}"
            report = output / "adjudication.json"
            if report.exists():
                continue
            if output.exists():
                raise RuntimeError(f"partial completion output requires audit: {output}")
            # The trace and scoring semantics are invariant to this batching
            # choice.  Long post-cutover windows use a smaller cohort solely
            # to bound peak attention/reuse allocation on the fixed four GPUs.
            cohort_size = 1 if edge == 5 and horizon >= 4 else (32 if horizon >= 4 else 128)
            evaluate(
                edge=edge, horizon=horizon, env=env, output=output, cohort_size=cohort_size,
                force_fallback=edge == 5 and horizon >= 4,
            )
            run([sys.executable, "scripts/summarize_yambda500m_hstu_native_d14_auc_coverage.py"], env)
    run([sys.executable, "scripts/summarize_yambda500m_hstu_native_d14_auc_coverage.py"], env)


if __name__ == "__main__":
    main()
