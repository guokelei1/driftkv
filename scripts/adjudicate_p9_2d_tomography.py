#!/usr/bin/env python3
"""Aggregate sealed P9.3 layer-by-position recovery across all seeds."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import adjudicate_p9_tomography as cell_adjudication
import eval_p9_2d_tomography_raw as evaluator
import run_p9_2d_tomography as ledger
import train_p7_theta0 as p7

ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / "results/p9/p9_3_2d_tomography_raw_seal_v1.json"
COARSE = ROOT / "results/p9/p9_2_coarse_tomography_v1.json"
OUTPUT = ROOT / "results/p9/p9_3_2d_tomography_v1.json"


def summarize(job: ledger.Job) -> dict:
    raw = job.output / "F_fidelity_2d_tomography.parquet"
    return cell_adjudication.summarize_raw(raw)


def finite_mean(values: list[float | None]) -> float | None:
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite P9.3 result: {args.output}")
    if not SEAL.exists():
        raise FileNotFoundError("seal P9.3 raw matrix first")
    if not 1 <= args.workers <= 24:
        raise ValueError("workers must be in [1,24]")
    seal = json.loads(SEAL.read_text())
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        summaries = list(executor.map(summarize, ledger.jobs()))
    grouped: dict[tuple[str, str, str], list[tuple[int, dict]]] = defaultdict(list)
    cells = []
    for job, summary in zip(ledger.jobs(), summaries, strict=True):
        cells.append({
            "release": job.release, "model": job.model, "seed": job.seed,
            "requests": summary["requests"], "raw_sha256": summary["raw_sha256"],
        })
        for action, values in summary["actions"].items():
            grouped[(job.release, job.model, action)].append((job.seed, values))
    coarse = json.loads(COARSE.read_text())
    coarse_lookup = {(row["release"], row["model"], row["action"]): row for row in coarse["aggregate"]}
    aggregate = []
    for (release, model, action), seed_rows in sorted(grouped.items()):
        seed_rows.sort()
        layer, segment = evaluator.parse_action(action)
        stale = [values["stale_JS"]["mean_equal_user"] for _, values in seed_rows]
        residual = [values["residual_JS"]["mean_equal_user"] for _, values in seed_rows]
        recovery = [values["absolute_recovery_JS"]["mean_equal_user"] for _, values in seed_rows]
        stale_mean, recovery_mean = finite_mean(stale), finite_mean(recovery)
        layer_parent = coarse_lookup[(release, model, f"layer_{layer}")]["absolute_recovery_JS_equal_seed_mean"]
        segment_parent = coarse_lookup[(release, model, segment)]["absolute_recovery_JS_equal_seed_mean"]
        aggregate.append({
            "release": release, "model": model, "action": action, "layer": layer, "segment": segment,
            "seed_order": [seed for seed, _ in seed_rows],
            "stale_JS_seed_points": stale, "stale_JS_equal_seed_mean": stale_mean,
            "residual_JS_seed_points": residual, "residual_JS_equal_seed_mean": finite_mean(residual),
            "absolute_recovery_JS_seed_points": recovery,
            "absolute_recovery_JS_equal_seed_mean": recovery_mean,
            "positive_recovery_seed_count": int(sum(value > 0 for value in recovery)),
            "ratio_of_mean_recovery_to_mean_stale": recovery_mean / stale_mean if stale_mean and stale_mean > 1e-8 else None,
            "coarse_parent_recovery": {"layer_only": layer_parent, "segment_only": segment_parent},
            "fraction_of_layer_only_recovery": recovery_mean / layer_parent if abs(layer_parent) > 1e-12 else None,
            "fraction_of_segment_only_recovery": recovery_mean / segment_parent if abs(segment_parent) > 1e-12 else None,
            "recovery_minus_better_coarse_parent": recovery_mean - max(layer_parent, segment_parent),
        })
    conditions = sorted({(row["release"], row["model"]) for row in aggregate})
    best = []
    stability = []
    for release, model in conditions:
        rows = [row for row in aggregate if row["release"] == release and row["model"] == model]
        winner = max(rows, key=lambda row: row["absolute_recovery_JS_equal_seed_mean"])
        best.append({
            "release": release, "model": model, "diagnostic_best_2d_action": winner["action"],
            "absolute_recovery_JS_equal_seed_mean": winner["absolute_recovery_JS_equal_seed_mean"],
            "ratio_of_mean_recovery_to_mean_stale": winner["ratio_of_mean_recovery_to_mean_stale"],
            "positive_recovery_seed_count": winner["positive_recovery_seed_count"],
        })
        seed_vectors = [np.asarray([row["absolute_recovery_JS_seed_points"][i] for row in rows]) for i in range(3)]
        pairs = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            value = spearmanr(seed_vectors[left], seed_vectors[right]).statistic
            pairs.append(float(value) if np.isfinite(value) else None)
        stability.append({"release": release, "model": model, "seed_pair_rank_spearman": pairs})
    payload = {
        "status": "P9_3_diagnostic_2d_tomography_aggregated_stop_before_dependency_closure",
        "raw_seal_hash": p7.sha256_file(SEAL), "contract_hash": seal["contract_hash"],
        "diagnostic_not_executable_action": True,
        "cells": cells, "aggregate": aggregate,
        "best_by_semantic_condition_diagnostic_only": best,
        "cross_seed_action_rank_stability": stability,
        "notes": [
            "All 12 semantic seed-cells and all 24 layer-by-segment interventions are included.",
            "Two-dimensional recovery is compared with its broader layer-only and segment-only diagnostic parents.",
            "Best labels are descriptive and cannot select an executable migration action.",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells), "conditions": len(conditions), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
