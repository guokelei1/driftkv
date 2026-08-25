#!/usr/bin/env python3
"""Seal D=14 matched-rolling AUC release-gain and one-hop-debt diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from run_yambda500m_hstu_native_d14_onehop_reuse import (
    MANIFEST, MATRIX, ROOT, atomic_text, checkpoint, run, sha256,
)


CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_d14_auc_release_debt_diagnostic_v2.yaml"
OUTPUT = MATRIX / "d14_auc_release_debt_diagnostic_v2"


def evaluate(*, edge: int, horizon: int, env: dict[str, str], max_users: int = 0,
             destination: Path | None = None) -> None:
    cutover = 217 + edge * 14
    output = destination or OUTPUT / f"eval_{horizon}d" / f"v{edge - 1}_to_v{edge}"
    command = [
        "torchrun", "--standalone", "--nproc_per_node=4",
        "scripts/evaluate_yambda500m_hstu_native_onehop_reuse_raw.py",
        "--stage", f"d14_auc_release_debt_e{horizon}_edge{edge}",
        "--edge", f"v{edge - 1}_to_v{edge}",
        "--cutover-day", str(cutover), "--start-day", str(cutover),
        "--end-day", str(cutover + horizon), "--manifest-dir", str(MANIFEST),
        "--parent", str(checkpoint(edge - 1)), "--current", str(checkpoint(edge)),
        "--include-parent-exact", "--output", str(output),
    ]
    if max_users:
        command.extend(["--max-users", str(max_users)])
    run(command, env)
    run([
        sys.executable, "scripts/adjudicate_yambda500m_hstu_native_onehop_reuse.py",
        "--raw", str(output / "raw.parquet"), "--seal", str(output / "raw.seal.json"),
        "--labels", str(MANIFEST / "requests_quality.parquet"),
        "--output", str(output / "adjudication.json"),
    ], env)


def results() -> list[dict]:
    rows = []
    for report in sorted(OUTPUT.glob("eval_*d/v*_to_v*/adjudication.json")):
        payload = json.loads(report.read_text())
        debt = payload["release_debt_auc"]
        rows.append({
            "edge": payload["edge"],
            "evaluation_days": payload["evaluation_day_range"][1] - payload["evaluation_day_range"][0],
            **debt,
        })
    return sorted(rows, key=lambda row: (int(row["edge"].split("_to_")[0][1:]), row["evaluation_days"]))


def render(rows: list[dict]) -> str:
    lines = [
        "# D=14 matched-rolling AUC release gain and One-hop Reuse debt",
        "",
        "For every request, the three paths share user, prefix, query, target and timestamp-group-atomic append stream. `Parent Exact` continues the parent model; `Current Exact` uses current-model recompute; `One-hop Reuse` begins with the parent-produced cutover K/V, then appends with the current model.",
        "",
        "- **Release gain** = Current Exact AUC − Parent Exact AUC (pp). Positive means the new model is better.",
        "- **Reuse loss** = Current Exact AUC − One-hop Reuse AUC (pp). Positive means Reuse is worse.",
        "- **Erased release gain** = Reuse loss / Release gain. It is a percentage only when release gain is positive; signed fraction is retained for auditing mixed edges.",
        "",
        "All results are post-hoc diagnostics, not Full-only admission and not cache-lineage promotion.",
        "",
        "| Edge | E (days) | Release gain (pp) | Reuse loss (pp) | Reuse loss / release gain |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        percent = row["reuse_harm_over_current_gain_percent"]
        value = "N/A" if percent is None else f"{percent:+.1f}%"
        lines.append(
            f"| {row['edge'].replace('_to_', ' → ')} | {row['evaluation_days']} | "
            f"{row['current_minus_parent_ROC_AUC_pp']:+.6f} | "
            f"{row['current_minus_reuse_ROC_AUC_pp']:+.6f} | {value} |"
        )
    lines.append("")
    return "\n".join(lines)


def emit() -> None:
    rows = results()
    atomic_text(OUTPUT / "auc_release_debt_table.json", json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    atomic_text(OUTPUT / "auc_release_debt_table.md", render(rows))
    if rows:
        print("SEALED_AUC_RELEASE_DEBT", json.dumps(rows[-1], sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 7, 14])
    args = parser.parse_args()
    contract = yaml.safe_load(CONTRACT.read_text())
    frozen = contract["frozen_inputs"]
    if sha256(MANIFEST / "manifest.json") != frozen["matrix_manifest_sha256"]:
        raise RuntimeError("D14 diagnostic manifest hash differs from its contract")
    if sha256(checkpoint(0)) != frozen["v0_sha256"]:
        raise RuntimeError("D14 diagnostic v0 hash differs from its contract")
    if any(edge not in range(1, 6) for edge in args.edges):
        raise ValueError("requested D14 edge is outside this immutable contract")
    allowed_horizons = set(contract["scope"]["evaluation_days"])
    if any(horizon not in allowed_horizons for horizon in args.horizons):
        raise ValueError("requested horizon is outside this immutable contract")
    for version in range(1, 6):
        if not checkpoint(version).exists():
            raise RuntimeError(f"D14 checkpoint v{version} is absent")
    env = {**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": "0,1,2,3",
           "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    marker = OUTPUT / "preflight_complete.json"
    if not marker.exists():
        with tempfile.TemporaryDirectory(prefix="evokv_d14_auc_debt_canary_") as temporary:
            evaluate(edge=1, horizon=1, env=env, max_users=32, destination=Path(temporary) / "canary")
        atomic_text(marker, json.dumps({"status": "four_rank_matched_rolling_canary_passed", "contract_sha256": sha256(CONTRACT)}, indent=2) + "\n")
    for horizon in args.horizons:
        for edge in args.edges:
            report = OUTPUT / f"eval_{horizon}d" / f"v{edge - 1}_to_v{edge}" / "adjudication.json"
            if not report.exists():
                if report.parent.exists():
                    raise RuntimeError(f"partial diagnostic requires audit: {report.parent}")
                evaluate(edge=edge, horizon=horizon, env=env)
            emit()
    emit()


if __name__ == "__main__":
    main()
