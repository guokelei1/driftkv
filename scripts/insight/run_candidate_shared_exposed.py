#!/usr/bin/env python3
"""Run the prospective five-edge real-exposed signed causal evaluation."""

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
CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_candidate_shared_causal_v1.yaml"
MANIFEST = ROOT / "data/manifests/yambda500m_small_hstu_native_rolling_matrix_fast_v3"
OUTPUT = ROOT / "results/yambda500m_small_seed17/insight_candidate_shared_causal_v1/formal_exposed"
CONTROLLED = ROOT / "results/yambda500m_small_seed17/insight_candidate_shared_causal_v1/formal_controlled/summary.json"
EVALUATOR_CANARY = ROOT / "results/yambda500m_small_seed17/insight_candidate_shared_causal_v1/exposed_evaluator_canary/raw.seal.json"


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


def verify_unlock() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    frozen = contract["frozen_inputs"]
    for name in (
        "source_observation_contract",
        "population",
        "matrix_manifest",
        "requests_fidelity",
        "requests_quality",
    ):
        record = frozen[name]
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"frozen {name} differs before real-exposed evaluation")
    for version in range(6):
        record = frozen["checkpoints"][f"v{version}"]
        if sha256(checkpoint(version)) != record["sha256"]:
            raise RuntimeError(f"frozen v{version} checkpoint differs")
    controlled = json.loads(CONTROLLED.read_text())
    if (
        controlled["status"] != "formal_controlled_signed_causal_passed"
        or controlled["contract_sha256"] != sha256(CONTRACT)
    ):
        raise RuntimeError("formal controlled signed causal gate did not pass")
    evaluator = json.loads(EVALUATOR_CANARY.read_text())
    if (
        evaluator["native_score_max_abs_error"] > 2e-5
        or evaluator["full_delta_reconstruction_max_abs_error"] > 2e-5
    ):
        raise RuntimeError("distributed real-exposed evaluator canary failed")
    return contract


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def evaluate(edge: int, env: dict[str, str]) -> None:
    edge_name = f"v{edge - 1}_to_v{edge}"
    cutover = 217 + edge * 14
    output = OUTPUT / "eval_14d" / edge_name
    raw = output / "raw.parquet"
    if not raw.exists():
        if output.exists():
            raise RuntimeError(f"partial real-exposed output requires audit: {output}")
        run(
            [
                "torchrun",
                "--standalone",
                "--nproc_per_node=4",
                "scripts/insight/evaluate_candidate_shared_exposed_raw.py",
                "--stage",
                f"candidate_shared_real_exposed_e14_edge{edge}",
                "--edge",
                edge_name,
                "--cutover-day",
                str(cutover),
                "--start-day",
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
                str(output),
                "--cohort-size",
                "32",
            ],
            env,
        )
    adjudication = output / "adjudication"
    if not adjudication.exists():
        run(
            [
                sys.executable,
                "scripts/insight/adjudicate_candidate_shared_exposed.py",
                "--raw",
                str(raw),
                "--seal",
                str(output / "raw.seal.json"),
                "--labels",
                str(MANIFEST / "requests_quality.parquet"),
                "--output-dir",
                str(adjudication),
            ],
            env,
        )


def emit_summary() -> None:
    metric_frames = []
    paired_frames = []
    seals = []
    for edge in range(1, 6):
        directory = OUTPUT / "eval_14d" / f"v{edge - 1}_to_v{edge}"
        metric = directory / "adjudication/quality_by_width.csv"
        paired = directory / "adjudication/paired_fidelity.csv"
        if not metric.exists() or not paired.exists():
            continue
        metric_frames.append(pd.read_csv(metric))
        paired_frames.append(pd.read_csv(paired))
        seals.append(json.loads((directory / "raw.seal.json").read_text()))
    if not metric_frames:
        return
    metrics = pd.concat(metric_frames, ignore_index=True)
    paired = pd.concat(paired_frames, ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "quality_all_edges.csv", index=False)
    paired.to_csv(OUTPUT / "paired_fidelity_all_edges.csv", index=False)
    focus = metrics[
        metrics.path.isin(["current_exact", "reuse", "shared_only", "residual_only"])
    ][["edge", "width", "path", "requests", "users", "ROC_AUC", "log_loss", "within_bank_pairwise_accuracy"]]
    lines = [
        "# Five-edge real-exposed signed causal evaluation",
        "",
        "All raw scores were sealed before label join. Candidate banks contain only real same-UID, same-timestamp requests; no sampled negatives were added. Shared/residual are oracle interventions, not executable state actions.",
        "",
        "| edge | width | path | requests | users | ROC_AUC | log_loss | within_bank_pairwise_accuracy |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in focus.itertuples(index=False):
        lines.append(
            f"| {row.edge} | {row.width} | {row.path} | {row.requests} | {row.users} | "
            f"{'N/A' if pd.isna(row.ROC_AUC) else f'{row.ROC_AUC:.6f}'} | {row.log_loss:.6f} | "
            f"{'N/A' if pd.isna(row.within_bank_pairwise_accuracy) else f'{row.within_bank_pairwise_accuracy:.6f}'} |"
        )
    summary = {
        "status": "candidate_shared_real_exposed_five_edge_complete" if len(seals) == 5 else "candidate_shared_real_exposed_partial",
        "edges": len(seals),
        "users_sum_across_edges": sum(seal["users"] for seal in seals),
        "banks_across_widths": sum(seal["banks_across_widths"] for seal in seals),
        "selected_requests_across_widths": sum(
            seal["selected_requests_across_widths"] for seal in seals
        ),
        "labels_joined_only_after_each_raw_seal": True,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    if any(edge not in range(1, 6) for edge in args.edges):
        raise ValueError("edges must be drawn from 1..5")
    verify_unlock()
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "OMP_NUM_THREADS": "2",
        "PYTHONUNBUFFERED": "1",
    }
    for edge in args.edges:
        evaluate(edge, env)
        emit_summary()
    emit_summary()


if __name__ == "__main__":
    main()
