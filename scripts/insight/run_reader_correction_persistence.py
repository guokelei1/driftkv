#!/usr/bin/env python3
"""Run five-edge real-request reader-correction persistence observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_reader_compatibility_correction_v1.yaml"
MANIFEST = ROOT / "data/manifests/yambda500m_small_hstu_native_rolling_matrix_fast_v3"
OUTPUT = ROOT / "results/yambda500m_small_seed17/insight_reader_compatibility_correction_v1"
STAGE_CANARY = OUTPUT / "canary_stage/summary.json"
FORMAL_CONTROLLED = OUTPUT / "formal_controlled_stage/summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(version: int) -> Path:
    if version == 0:
        return ROOT / "results/yambda500m_small_seed17/hstu_native_release_chain_v1/v0/checkpoint_100.pt"
    return (
        ROOT
        / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3"
        / f"train_14d/checkpoints/v{version}/checkpoint_100.pt"
    )


def verify(scope: str) -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    for name, record in contract["frozen_inputs"].items():
        if name == "checkpoints":
            continue
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"frozen {name} differs")
        required = record.get("required_status")
        if required and json.loads((ROOT / record["path"]).read_text())["status"] != required:
            raise RuntimeError(f"frozen {name} status differs")
    for version in range(6):
        record = contract["frozen_inputs"]["checkpoints"][f"v{version}"]
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"frozen v{version} checkpoint differs")
    canary = json.loads(STAGE_CANARY.read_text())
    if (
        canary["status"] != "reader_correction_canary_passed"
        or canary["contract_sha256"] != sha256(CONTRACT)
    ):
        raise RuntimeError("stage canary did not unlock real-request observation")
    if scope == "formal":
        controlled = json.loads(FORMAL_CONTROLLED.read_text())
        if (
            controlled["status"] != "reader_correction_formal_controlled_passed"
            or controlled["contract_sha256"] != sha256(CONTRACT)
        ):
            raise RuntimeError("formal controlled stage observation is incomplete")
        real_canary = json.loads((OUTPUT / "canary_persistence/summary.json").read_text())
        if (
            real_canary["status"] != "reader_correction_real_canary_passed"
            or real_canary["contract_sha256"] != sha256(CONTRACT)
        ):
            raise RuntimeError("real-request persistence canary did not pass")
    return contract


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def evaluate(edge: int, scope: str, env: dict[str, str]) -> None:
    edge_name = f"v{edge - 1}_to_v{edge}"
    cutover = 217 + edge * 14
    root = OUTPUT / ("canary_persistence" if scope == "canary" else "formal_real")
    target = root / "eval_14d" / edge_name
    if (target / "raw.seal.json").exists():
        return
    if target.exists():
        raise RuntimeError(f"partial real-request output requires audit: {target}")
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=4",
        "scripts/insight/evaluate_reader_correction_persistence_raw.py",
        "--edge",
        edge_name,
        "--cutover-day",
        str(cutover),
        "--end-day",
        str(cutover + 14),
        "--manifest-dir",
        str(MANIFEST),
        "--parent",
        str(checkpoint(edge - 1)),
        "--current",
        str(checkpoint(edge)),
        "--output",
        str(target),
        "--cohort-size",
        "8" if scope == "canary" else "16",
    ]
    if scope == "canary":
        command.extend(["--max-users", "8"])
    run(command, env)
    for transient in (
        target / ".directory_ready",
        target / ".raw_complete",
        *target.glob("progress_rank*.json"),
    ):
        transient.unlink(missing_ok=True)


def emit_summary(scope: str) -> None:
    root = OUTPUT / ("canary_persistence" if scope == "canary" else "formal_real")
    seals = []
    scores = []
    energies = []
    persistence = []
    for edge in range(1, 6):
        directory = root / "eval_14d" / f"v{edge - 1}_to_v{edge}"
        seal_path = directory / "raw.seal.json"
        if not seal_path.exists():
            continue
        seals.append(json.loads(seal_path.read_text()))
        scores.append(pd.read_parquet(directory / "stage_score.parquet"))
        energies.append(pd.read_parquet(directory / "stage_energy.parquet"))
        persistence.append(pd.read_parquet(directory / "persistence.parquet"))
    if not seals:
        return
    score = pd.concat(scores, ignore_index=True)
    energy = pd.concat(energies, ignore_index=True)
    pairs = pd.concat(persistence, ignore_index=True)
    root.mkdir(parents=True, exist_ok=True)
    score.to_parquet(root / "stage_score_all_edges.parquet", index=False)
    energy.to_parquet(root / "stage_energy_all_edges.parquet", index=False)
    pairs.to_parquet(root / "persistence_all_edges.parquet", index=False)
    passed = len(seals) == 5 and max(seal["correctness_max_abs_error"] for seal in seals) <= 2e-5
    status = (
        f"reader_correction_real_{scope}_passed"
        if passed
        else f"reader_correction_real_{scope}_incomplete_or_failed"
    )
    summary = {
        "status": status,
        "scope": scope,
        "contract_sha256": sha256(CONTRACT),
        "edges": len(seals),
        "users_sum_across_edges": sum(seal["users"] for seal in seals),
        "request_groups": sum(seal["request_groups"] for seal in seals),
        "full_history_persistence_pairs": sum(
            seal["full_history_persistence_pairs"] for seal in seals
        ),
        "labels_read": False,
        "correctness": {
            "passed": passed,
            "max_abs_error": max(seal["correctness_max_abs_error"] for seal in seals),
        },
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    focus_score = (
        score[score.stage != "reuse"]
        .groupby(["edge", "stage"], as_index=False)
        .agg(
            mean_stage_gap=("mean_abs_probability_gap", "mean"),
            mean_reuse_gap=("reuse_probability_gap", "mean"),
        )
    )
    focus_score["gap_recovery_over_reuse"] = 1.0 - (
        focus_score.mean_stage_gap / focus_score.mean_reuse_gap.clip(lower=1e-12)
    )
    focus_pair = (
        pairs.groupby(["edge", "stage"], as_index=False)
        .agg(
            median_cosine=("adjacent_request_direction_cosine", "median"),
            median_scaled_recovery=("coverage_scaled_prior_gap_recovery", "median"),
        )
    )
    lines = [
        f"# Real-request reader-correction {scope}",
        "",
        f"Correctness: **{'PASS' if passed else 'FAIL'}**; labels read: **no**.",
        "",
        "## Same-request stage recovery",
        "",
        "| edge | stage | mean recovery |",
        "| --- | --- | ---: |",
    ]
    for row in focus_score.itertuples(index=False):
        lines.append(f"| {row.edge} | {row.stage} | {row.gap_recovery_over_reuse:.6f} |")
    lines.extend(
        [
            "",
            "## Adjacent-request persistence",
            "",
            "| edge | stage | median cosine | median coverage-scaled recovery |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in focus_pair.itertuples(index=False):
        lines.append(
            f"| {row.edge} | {row.stage} | {row.median_cosine:.6f} | "
            f"{row.median_scaled_recovery:.6f} |"
        )
    (root / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("canary", "formal"), required=True)
    parser.add_argument("--edges", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    if any(edge not in range(1, 6) for edge in args.edges):
        raise ValueError("edges must be drawn from 1..5")
    verify(args.scope)
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "OMP_NUM_THREADS": "2",
        "PYTHONUNBUFFERED": "1",
    }
    for edge in args.edges:
        evaluate(edge, args.scope, env)
        emit_summary(args.scope)
    emit_summary(args.scope)


if __name__ == "__main__":
    main()
