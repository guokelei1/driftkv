#!/usr/bin/env python3
"""Run the sealed D=14 one-hop Reuse diagnostic across all edges and E."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_d14_onehop_reuse_diagnostic_v1.yaml"
MATRIX = ROOT / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3"
MANIFEST = ROOT / "data/manifests/yambda500m_small_hstu_native_rolling_matrix_fast_v3"
OUTPUT = MATRIX / "d14_onehop_reuse_diagnostic_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def checkpoint(version: int) -> Path:
    if version == 0:
        return ROOT / "results/yambda500m_small_seed17/hstu_native_release_chain_v1/v0/checkpoint_100.pt"
    return MATRIX / "train_14d" / "checkpoints" / f"v{version}" / "checkpoint_100.pt"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def results() -> list[dict]:
    rows = []
    for report in sorted(OUTPUT.glob("eval_*d/v*_to_v*/adjudication.json")):
        payload = json.loads(report.read_text())
        values = payload["reuse_minus_recompute"]
        harm = values["paired_harm"]
        rows.append({
            "edge": payload["edge"], "evaluation_days": payload["evaluation_day_range"][1] - payload["evaluation_day_range"][0],
            "user_equal_reuse_minus_recompute_log_loss": harm["user_weighted_mean"],
            "event_reuse_minus_recompute_log_loss": values["event_weighted_log_loss"],
            "current_minus_reuse_ROC_AUC_pp": values["current_minus_reuse_ROC_AUC_pp"],
            "current_minus_reuse_dislike_PR_AUC_pp": values["current_minus_reuse_dislike_PR_AUC_pp"],
            "reuse_minus_current_Brier": values["reuse_minus_current_Brier"],
            "mean_Bernoulli_JS": values["mean_Bernoulli_JS"],
            "mean_absolute_logit_shift": values["mean_absolute_logit_shift"],
        })
    return rows


def render(rows: list[dict]) -> str:
    by_key = {(row["edge"], row["evaluation_days"]): row for row in rows}
    edges = [f"v{index}_to_v{index + 1}" for index in range(5)]
    horizons = (1, 4, 7, 14)
    metrics = (
        ("user_equal_reuse_minus_recompute_log_loss", "User-equal Reuse − Recompute log loss (positive = Reuse harms)"),
        ("event_reuse_minus_recompute_log_loss", "Event-weighted Reuse − Recompute log loss (positive = Reuse harms)"),
        ("current_minus_reuse_ROC_AUC_pp", "Current Recompute − Reuse ROC-AUC (pp; positive = Reuse harms)"),
        ("current_minus_reuse_dislike_PR_AUC_pp", "Current Recompute − Reuse dislike PR-AUC (pp; positive = Reuse harms)"),
        ("reuse_minus_current_Brier", "Reuse − Recompute Brier (positive = Reuse harms)"),
        ("mean_Bernoulli_JS", "Bernoulli JS"),
    )
    lines = [
        "# D=14 one-hop Reuse diagnostic",
        "",
        "Comparison: the current HSTU query/head reads either its own exact rolling prefix KV (**Recompute**) or the parent-produced cutover KV followed by current-model appends (**One-hop Reuse**).",
        "All values are post-hoc diagnostic observations, not release admission or recursive-lineage results.",
        "",
    ]
    for key, title in metrics:
        lines.extend([f"## {title}", "", "| Edge | " + " | ".join(f"E={value}" for value in horizons) + " |", "| --- | " + " | ".join("---:" for _ in horizons) + " |"])
        for edge in edges:
            formatted = []
            for horizon in horizons:
                row = by_key.get((edge, horizon))
                formatted.append("—" if row is None else f"{row[key]:+.6f}")
            lines.append(f"| {edge.replace('_to_', ' → ')} | " + " | ".join(formatted) + " |")
        lines.append("")
    return "\n".join(lines)


def emit() -> None:
    rows = results()
    atomic_text(OUTPUT / "reuse_vs_recompute_matrix.json", json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    atomic_text(OUTPUT / "reuse_vs_recompute_matrix.md", render(rows))
    if rows:
        print("SEALED_REUSE_EDGE", json.dumps(rows[-1], sort_keys=True), flush=True)


def evaluate(*, edge: int, horizon: int, env: dict[str, str], max_users: int = 0, destination: Path | None = None) -> None:
    cutover = 217 + edge * 14
    output = destination or OUTPUT / f"eval_{horizon}d" / f"v{edge - 1}_to_v{edge}"
    command = [
        "torchrun", "--standalone", "--nproc_per_node=4", "scripts/evaluate_yambda500m_hstu_native_onehop_reuse_raw.py",
        "--stage", f"d14_onehop_e{horizon}_edge{edge}", "--edge", f"v{edge - 1}_to_v{edge}",
        "--cutover-day", str(cutover), "--start-day", str(cutover), "--end-day", str(cutover + horizon),
        "--manifest-dir", str(MANIFEST), "--parent", str(checkpoint(edge - 1)), "--current", str(checkpoint(edge)),
        "--output", str(output),
    ]
    if max_users:
        command.extend(["--max-users", str(max_users)])
    run(command, env)
    run([
        sys.executable, "scripts/adjudicate_yambda500m_hstu_native_onehop_reuse.py", "--raw", str(output / "raw.parquet"),
        "--seal", str(output / "raw.seal.json"), "--labels", str(MANIFEST / "requests_quality.parquet"),
        "--output", str(output / "adjudication.json"),
    ], env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 7, 14])
    args = parser.parse_args()
    contract = yaml.safe_load(CONTRACT.read_text())
    frozen = contract["frozen_inputs"]
    if sha256(MANIFEST / "manifest.json") != frozen["matrix_manifest_sha256"]:
        raise RuntimeError("D14 diagnostic manifest hash differs from its contract")
    if sha256(checkpoint(0)) != frozen["v0_sha256"]:
        raise RuntimeError("D14 diagnostic v0 hash differs from its contract")
    for version in range(1, 6):
        if not checkpoint(version).exists():
            raise RuntimeError(f"D14 checkpoint v{version} is absent")
    env = {**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": "0,1,2,3", "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    allowed_horizons = set(contract["scope"]["evaluation_days"])
    if any(edge not in range(1, 6) for edge in args.edges) or any(horizon not in allowed_horizons for horizon in args.horizons):
        raise ValueError("requested D14 edges or horizons are outside the frozen diagnostic contract")
    marker = OUTPUT / "preflight_complete.json"
    if not marker.exists():
        with tempfile.TemporaryDirectory(prefix="evokv_d14_onehop_canary_") as temporary:
            evaluate(edge=1, horizon=1, env=env, max_users=32, destination=Path(temporary) / "canary")
        atomic_text(marker, json.dumps({"status": "four_rank_onehop_canary_passed", "contract_sha256": sha256(CONTRACT)}, indent=2) + "\n")
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
