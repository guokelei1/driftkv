#!/usr/bin/env python3
"""D=14/E=14 direct (non-recursive) long-age KV Reuse matrix for v3/v4."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from run_yambda500m_hstu_native_d14_onehop_reuse import MANIFEST, MATRIX, ROOT, checkpoint


CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_d14_direct_long_age_reuse_v1.yaml"
OUTPUT = MATRIX / "d14_direct_long_age_reuse_v1"
CELLS = ((0, 3, 259), (1, 3, 259), (0, 4, 273), (1, 4, 273), (2, 4, 273))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def evaluate(*, producer: int, current: int, cutover: int, output: Path, env: dict[str, str],
             max_users: int = 0, force_fallback: bool = False) -> None:
    command = [
        "torchrun", "--standalone", "--nproc_per_node=4", "scripts/evaluate_yambda500m_hstu_native_onehop_reuse_raw.py",
        "--stage", f"d14_direct_long_age_v{producer}_to_v{current}", "--edge", f"v{producer}_to_v{current}",
        "--cutover-day", str(cutover), "--start-day", str(cutover), "--end-day", str(cutover + 14),
        "--manifest-dir", str(MANIFEST), "--parent", str(checkpoint(producer)), "--current", str(checkpoint(current)),
        "--cohort-size", "32", "--output", str(output),
    ]
    if max_users:
        command.extend(["--max-users", str(max_users)])
    if force_fallback:
        command.insert(-2, "--force-fallback")
    run(command, env)
    run([
        sys.executable, "scripts/adjudicate_yambda500m_hstu_native_onehop_reuse.py",
        "--raw", str(output / "raw.parquet"), "--seal", str(output / "raw.seal.json"),
        "--labels", str(MANIFEST / "requests_quality.parquet"), "--output", str(output / "adjudication.json"),
    ], env)


def report_row(report: Path) -> dict:
    payload = json.loads(report.read_text())
    values = payload["reuse_minus_recompute"]
    producer, current = payload["edge"].split("_to_")
    return {
        "producer": producer,
        "current": current,
        "version_gap": int(current[1:]) - int(producer[1:]),
        "current_minus_direct_reuse_ROC_AUC_pp": values["current_minus_reuse_ROC_AUC_pp"],
        "current_minus_direct_reuse_PR_AUC_pp": values["current_minus_reuse_dislike_PR_AUC_pp"],
        "direct_reuse_minus_current_event_log_loss": values["event_weighted_log_loss"],
        "direct_reuse_minus_current_user_log_loss": values["paired_harm"]["user_weighted_mean"],
        "mean_Bernoulli_JS": values["mean_Bernoulli_JS"],
    }


def emit() -> None:
    rows = []
    # The adjacent D=14/E=14 cells are already sealed under the initial and
    # completion diagnostics; do not duplicate or overwrite them.
    adjacent = {
        ("v2", "v3"): MATRIX / "d14_onehop_reuse_completion_v2" / "eval_14d" / "v2_to_v3" / "adjudication.json",
        ("v3", "v4"): MATRIX / "d14_onehop_reuse_completion_v2" / "eval_14d" / "v3_to_v4" / "adjudication.json",
    }
    for report in adjacent.values():
        if report.exists():
            rows.append(report_row(report))
    for report in OUTPUT.glob("v*_to_v*/adjudication.json"):
        rows.append(report_row(report))
    rows.sort(key=lambda row: (int(row["current"][1:]), int(row["producer"][1:])))
    lines = [
        "# D=14/E=14 direct long-age KV Reuse",
        "",
        "Every row uses the current model's exact rolling cache as Recompute. For Direct Reuse, the named producer recomputes the **entire** pre-cutover prefix; the current model then reads that producer KV and appends every post-cutover event. This is direct long-age Reuse, not recursive lineage.",
        "",
        "| Current | KV producer | Version gap | Current − Direct Reuse ROC-AUC (pp) | Current − Direct Reuse PR-AUC (pp) | Direct Reuse − Current event log-loss | User-equal log-loss | JS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['current']} | {row['producer']} | {row['version_gap']} | "
            f"{row['current_minus_direct_reuse_ROC_AUC_pp']:+.6f} | "
            f"{row['current_minus_direct_reuse_PR_AUC_pp']:+.6f} | "
            f"{row['direct_reuse_minus_current_event_log_loss']:+.6f} | "
            f"{row['direct_reuse_minus_current_user_log_loss']:+.6f} | {row['mean_Bernoulli_JS']:.2e} |"
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "direct_long_age_matrix.json").write_text(json.dumps(rows, indent=2) + "\n")
    (OUTPUT / "direct_long_age_matrix.md").write_text("\n".join(lines) + "\n")
    if rows:
        print("SEALED_DIRECT_LONG_AGE", json.dumps(rows[-1], sort_keys=True), flush=True)


def main() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    frozen = contract["frozen_inputs"]
    if sha256(MANIFEST / "manifest.json") != frozen["matrix_manifest_sha256"]:
        raise RuntimeError("matrix manifest differs from frozen long-age contract")
    if sha256(checkpoint(0)) != frozen["v0_sha256"]:
        raise RuntimeError("v0 differs from frozen long-age contract")
    env = {**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": "0,1,2,3", "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    marker = OUTPUT / "preflight_complete.json"
    if not marker.exists():
        with tempfile.TemporaryDirectory(prefix="evokv_d14_long_age_canary_") as temporary:
            evaluate(producer=1, current=3, cutover=259, output=Path(temporary) / "canary", env=env, max_users=16)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"status": "four_rank_direct_long_age_canary_passed", "contract_sha256": sha256(CONTRACT)}, indent=2) + "\n")
    for producer, current, cutover in CELLS:
        output = OUTPUT / f"v{producer}_to_v{current}"
        report = output / "adjudication.json"
        if report.exists():
            emit()
            continue
        if output.exists():
            raise RuntimeError(f"partial long-age result requires audit: {output}")
        evaluate(
            producer=producer, current=current, cutover=cutover, output=output, env=env,
            force_fallback=current == 4,
        )
        emit()
    emit()


if __name__ == "__main__":
    main()
