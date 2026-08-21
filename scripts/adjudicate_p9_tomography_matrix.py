#!/usr/bin/env python3
"""Build per-cell summaries and aggregate the sealed P9.2 matrix by seed."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import train_p7_theta0 as p7

import adjudicate_p9_tomography as cell_adjudication
import run_p9_tomography as ledger

ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / "results/p9/p9_2_tomography_raw_seal_v1.json"
OUTPUT = ROOT / "results/p9/p9_2_coarse_tomography_v1.json"


def build_or_load(job: ledger.Job) -> dict:
    path = job.output / "tomography_summary.json"
    raw = job.output / "F_fidelity_tomography.parquet"
    if path.exists():
        payload = json.loads(path.read_text())
        if payload["raw_sha256"] != p7.sha256_file(raw):
            raise RuntimeError(f"cell summary/raw hash mismatch: {job}")
        return payload
    payload = cell_adjudication.summarize_raw(raw)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not SEAL.exists():
        raise FileNotFoundError("seal P9.2 raw matrix before aggregation")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite P9.2 matrix result: {args.output}")
    if args.workers < 1 or args.workers > 24:
        raise ValueError("workers must be in [1,24]")
    jobs = ledger.jobs()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        summaries = list(executor.map(build_or_load, jobs))
    grouped: dict[tuple[str, str, str], list[tuple[int, dict]]] = defaultdict(list)
    cells = []
    for job, summary in zip(jobs, summaries, strict=True):
        cells.append({
            "release": job.release, "model": job.model, "seed": job.seed,
            "requests": summary["requests"],
            "summary": str((job.output / "tomography_summary.json").relative_to(ROOT)),
            "summary_sha256": p7.sha256_file(job.output / "tomography_summary.json"),
        })
        for action, values in summary["actions"].items():
            grouped[(job.release, job.model, action)].append((job.seed, values))
    aggregate = []
    for (release, model, action), seed_rows in sorted(grouped.items()):
        seed_rows.sort()
        recovery = [row["absolute_recovery_JS"]["mean_equal_user"] for _, row in seed_rows]
        residual = [row["residual_JS"]["mean_equal_user"] for _, row in seed_rows]
        stale = [row["stale_JS"]["mean_equal_user"] for _, row in seed_rows]
        relative = [row["relative_recovery_on_S_above_floor"]["mean_equal_user"] for _, row in seed_rows]
        aggregate.append({
            "release": release, "model": model, "action": action,
            "seed_order": [seed for seed, _ in seed_rows],
            "stale_JS_seed_points": stale, "stale_JS_equal_seed_mean": finite_mean(stale),
            "residual_JS_seed_points": residual, "residual_JS_equal_seed_mean": finite_mean(residual),
            "absolute_recovery_JS_seed_points": recovery, "absolute_recovery_JS_equal_seed_mean": finite_mean(recovery),
            "positive_recovery_seed_count": sum(float(value) > 0 for value in recovery),
            "relative_recovery_seed_points": relative, "relative_recovery_equal_seed_mean": finite_mean(relative),
        })
    best_by_condition = []
    conditions = sorted({(row["release"], row["model"]) for row in aggregate})
    for release, model in conditions:
        candidates = [row for row in aggregate if row["release"] == release and row["model"] == model]
        best = max(candidates, key=lambda row: row["absolute_recovery_JS_equal_seed_mean"])
        best_by_condition.append({
            "release": release, "model": model, "diagnostic_best_action": best["action"],
            "absolute_recovery_JS_equal_seed_mean": best["absolute_recovery_JS_equal_seed_mean"],
            "positive_recovery_seed_count": best["positive_recovery_seed_count"],
        })
    seal = json.loads(SEAL.read_text())
    payload = {
        "status": "P9_2_coarse_diagnostic_tomography_aggregated_stop_before_P9_3",
        "raw_seal_hash": p7.sha256_file(SEAL), "contract_hash": seal["contract_hash"],
        "diagnostic_not_executable_action": True, "cells": cells, "aggregate": aggregate,
        "best_by_condition_diagnostic_only": best_by_condition,
        "notes": [
            "All 24 frozen F cells and all three seeds are included.",
            "Positive absolute recovery means the diagnostic splice moved output toward Current Full; negative means it worsened fidelity.",
            "Best-action labels are descriptive diagnostics and do not authorize a partial migration action.",
            "P9.3 remains pending human review of cross-seed and cross-release structure.",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells), "conditions": len(best_by_condition), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
