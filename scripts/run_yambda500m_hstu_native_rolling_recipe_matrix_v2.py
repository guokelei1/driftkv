#!/usr/bin/env python3
"""Run the v2 rolling HSTU-native release-recipe matrix.

Each candidate is trained and then immediately evaluated against its parent.
The evaluation days deliberately overlap the following candidate's training
window.  This is an upstream Full-only recipe scan: it never performs
admission, Reuse, or cache-compatibility analysis.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_rolling_recipe_matrix_v2.yaml"
MANIFESTS = ROOT / "data/manifests/yambda500m_small_hstu_native_rolling_matrix_fast_v2"
V0 = ROOT / "results/yambda500m_small_seed17/hstu_native_release_chain_v1/v0/checkpoint_100.pt"
OUTPUT = ROOT / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def train(*, duration: int, version: int, parent: Path, output: Path, env: dict[str, str], canary: bool) -> None:
    start = 217 + (version - 1) * duration
    end = start + duration
    command = [
        "torchrun", "--standalone", "--nproc_per_node=4", "scripts/train_yambda500m_foundation_fsdp.py",
        "--version", f"v{version}", "--launch-contract", str(CONTRACT), "--manifest-dir", str(MANIFESTS),
        "--training-block", "matrix_horizon", "--parent", str(parent), "--output", str(output),
        "--oov-buckets", "256", "--passes", "1", "--train-start-day", str(start), "--train-end-day", str(end),
    ]
    if canary:
        command.extend(["--canary-steps", "2", "--batch-size", "8"])
    run(command, env)


def evaluate(*, duration: int, horizon: int, edge: int, parent: Path, current: Path, env: dict[str, str]) -> dict:
    cutover = 217 + edge * duration
    directory = OUTPUT / f"train_{duration}d" / f"eval_{horizon}d" / f"v{edge - 1}_to_v{edge}"
    report = directory / "adjudication.json"
    if report.exists():
        return json.loads(report.read_text(encoding="utf-8"))
    if directory.exists():
        raise RuntimeError(f"partial evaluation requires audit: {directory}")
    command = [
        "torchrun", "--standalone", "--nproc_per_node=4", "scripts/evaluate_yambda500m_release_candidates_raw.py",
        "--stage", f"rolling_matrix_v2_d{duration}_e{horizon}_edge{edge}", "--block", "matrix_horizon",
        "--training-block", "matrix_horizon", "--start-day", str(cutover), "--end-day", str(cutover + horizon),
        "--manifest-dir", str(MANIFESTS), "--parent", f"v{edge - 1}={parent}", f"--current", f"v{edge}={current}",
        "--output", str(directory), "--batch-size", "64",
    ]
    run(command, env)
    run([
        sys.executable, "scripts/adjudicate_yambda500m_release_candidates.py", "--raw", str(directory / "raw.parquet"),
        "--seal", str(directory / "raw.seal.json"), "--labels", str(MANIFESTS / "requests_quality.parquet"),
        "--output", str(report),
    ], env)
    return json.loads(report.read_text(encoding="utf-8"))


def metric_delta(report: dict) -> dict[str, float]:
    candidate = next(iter(report["candidates"].values()))
    parent = report["parent_absolute"]["hstu_native"]
    current = candidate["absolute"]["hstu_native"]
    return {
        "roc_auc_delta_pp": (current["ROC_AUC"] - parent["ROC_AUC"]) * 100,
        "dislike_pr_auc_delta_pp": (current["dislike_PR_AUC"] - parent["dislike_PR_AUC"]) * 100,
        "log_loss_delta": current["log_loss"] - parent["log_loss"],
        "brier_delta": current["Brier"] - parent["Brier"],
    }


def finished_rows(contract: dict) -> list[dict]:
    rows: list[dict] = []
    scope = contract["scope"]
    for duration in scope["training_days"]:
        updates = int(scope["total_versions_including_v0_by_training_days"][str(duration)]) - 1
        for horizon in scope["evaluation_days_by_training_days"][str(duration)]:
            for edge in range(1, updates + 1):
                report = OUTPUT / f"train_{duration}d" / f"eval_{horizon}d" / f"v{edge - 1}_to_v{edge}" / "adjudication.json"
                if report.exists():
                    rows.append({
                        "training_days": duration,
                        "evaluation_days": horizon,
                        "edge": f"v{edge - 1}_to_v{edge}",
                        **metric_delta(json.loads(report.read_text(encoding="utf-8"))),
                    })
    return rows


def render_summary(contract: dict, rows: list[dict]) -> str:
    rows_by_key = {(row["training_days"], row["evaluation_days"], row["edge"]): row for row in rows}
    labels = [
        ("roc_auc_delta_pp", "ROC-AUC Δ (pp)"),
        ("dislike_pr_auc_delta_pp", "dislike PR-AUC Δ (pp)"),
        ("log_loss_delta", "log loss Δ"),
        ("brier_delta", "Brier Δ"),
    ]
    lines = [
        f"# {contract['contract']}: sealed core deltas",
        "",
        "Every value is **Current Full minus Parent Full**. Positive AUC / PR-AUC is better; negative log loss / Brier is better.",
        "This is an upstream Full-only scan: it contains neither Reuse nor admission.",
        "",
    ]
    scope = contract["scope"]
    for duration in scope["training_days"]:
        horizons = scope["evaluation_days_by_training_days"][str(duration)]
        updates = int(scope["total_versions_including_v0_by_training_days"][str(duration)]) - 1
        lines.extend([f"## D={duration}: v0 through v{updates}", ""])
        for key, title in labels:
            lines.extend([f"### {title}", "", "| Edge | " + " | ".join(f"E={h}" for h in horizons) + " |", "| --- | " + " | ".join("---:" for _ in horizons) + " |"])
            for edge in range(1, updates + 1):
                values = []
                for horizon in horizons:
                    row = rows_by_key.get((duration, horizon, f"v{edge - 1}_to_v{edge}"))
                    values.append("—" if row is None else f"{row[key]:+.4f}")
                lines.append(f"| v{edge - 1} → v{edge} | " + " | ".join(values) + " |")
            lines.append("")
    return "\n".join(lines)


def emit_summary(contract: dict) -> None:
    rows = finished_rows(contract)
    atomic_json(OUTPUT / "matrix_progress.json", rows)
    atomic_text(OUTPUT / "core_delta_matrix.md", render_summary(contract, rows))
    if rows:
        newest = rows[-1]
        print("SEALED_EDGE", json.dumps(newest, sort_keys=True), flush=True)


def main() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if sha256(V0) != contract["scope"]["parent_model_sha256"]:
        raise RuntimeError("fixed v0 hash differs from matrix contract")
    manifest = MANIFESTS / "manifest.json"
    if not manifest.exists():
        raise RuntimeError("matrix manifest is not materialized")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_payload["contract_sha256"] != sha256(CONTRACT):
        raise RuntimeError("matrix manifest was not built from this frozen contract")
    scope = contract["scope"]
    maximum_day = int(contract["windows_days_half_open"]["maximum_timestamp_exclusive_day"])
    for duration in scope["training_days"]:
        updates = int(scope["total_versions_including_v0_by_training_days"][str(duration)]) - 1
        for horizon in scope["evaluation_days_by_training_days"][str(duration)]:
            if 217 + updates * duration + horizon > maximum_day:
                raise RuntimeError(f"D={duration}, E={horizon} exceeds frozen source window")
    env = {**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": "0,1,2,3", "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    preflight_marker = OUTPUT / "preflight_complete.json"
    if not preflight_marker.exists():
        with tempfile.TemporaryDirectory(prefix="evokv_matrix_v2_preflight_") as temporary:
            train(duration=1, version=1, parent=V0, output=Path(temporary) / "v1", env=env, canary=True)
        atomic_json(preflight_marker, {"status": "four_rank_canary_passed", "contract_sha256": sha256(CONTRACT)})
    for duration in scope["training_days"]:
        updates = int(scope["total_versions_including_v0_by_training_days"][str(duration)]) - 1
        horizons = [int(value) for value in scope["evaluation_days_by_training_days"][str(duration)]]
        chain_dir = OUTPUT / f"train_{duration}d" / "checkpoints"
        parent = V0
        for version in range(1, updates + 1):
            candidate = chain_dir / f"v{version}" / "checkpoint_100.pt"
            if not candidate.exists():
                if candidate.parent.exists():
                    raise RuntimeError(f"partial candidate requires audit: {candidate.parent}")
                train(duration=duration, version=version, parent=parent, output=candidate.parent, env=env, canary=False)
            for horizon in horizons:
                evaluate(duration=duration, horizon=horizon, edge=version, parent=parent, current=candidate, env=env)
                emit_summary(contract)
            parent = candidate
    rows = finished_rows(contract)
    atomic_json(OUTPUT / "matrix_result.json", rows)
    emit_summary(contract)


if __name__ == "__main__":
    main()
