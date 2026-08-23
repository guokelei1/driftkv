#!/usr/bin/env python3
"""P9.1-C label-free user-risk concentration and diagnostic recovery capture."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

import adjudicate_p8_hs as hs
import run_p9_tomography as ledger
import train_p7_theta0 as p7

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_2_closure_contract_v1.yaml"
OUTPUT = ROOT / "results/p9/p9_1_risk_concentration_v1.json"
NUMERIC_FLOOR = 1e-8


def validate_contract_inputs() -> dict[str, Any]:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p9_contract": ROOT / "configs/contracts/p9_tomography_contract_v1.yaml",
        "p9_1_distribution_v2": ROOT / "results/p9/p9_1_hs_distribution_v2.json",
        "p9_2_raw_seal": ROOT / "results/p9/p9_2_tomography_raw_seal_v1.json",
        "p9_2_coarse_result": ROOT / "results/p9/p9_2_coarse_tomography_v1.json",
        "p8_r0_raw_seal": ROOT / "results/p8/r0_control/raw_score_seal_v1.json",
        "p8_r1_edge1_raw_seal": ROOT / "results/p8/r1_edge1/raw_score_seal_v1.json",
        "p8_r1_edge2_raw_seal": ROOT / "results/p8/r1_edge2/raw_score_seal_v1.json",
        "p8_r2_raw_seal": ROOT / "results/p8/r2/raw_score_seal_v1.json",
    }
    for name, path in paths.items():
        if p7.sha256_file(path) != contract["input_hashes"][name]:
            raise RuntimeError(f"P9.1-C input hash mismatch: {name}")
    return contract


def user_means(uids: np.ndarray, values: np.ndarray) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for uid, value in zip(uids, values, strict=True):
        grouped[int(uid)].append(float(value))
    return {uid: float(np.mean(rows)) for uid, rows in grouped.items()}


def gini(values: np.ndarray) -> float | None:
    values = np.sort(np.asarray(values, dtype=np.float64))
    total = float(values.sum())
    if total <= NUMERIC_FLOOR:
        return None
    n = len(values)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float(np.sum((2.0 * ranks - n - 1.0) * values) / (n * total))


def top_indices(risk_by_uid: dict[int, float], fraction: float) -> list[int]:
    ordered = sorted(risk_by_uid, key=lambda uid: (-risk_by_uid[uid], uid))
    return ordered[: max(1, math.ceil(len(ordered) * fraction))]


def build_cell(job: ledger.Job) -> dict[str, Any]:
    path = job.output / "F_fidelity_tomography.parquet"
    frame = pq.read_table(path).to_pandas()
    expected_actions = yaml.safe_load(CONTRACT.read_text())["scope"]["actions"]
    baseline = frame.drop_duplicates("request_id", keep="first")
    if len(frame) != len(baseline) * len(expected_actions):
        raise RuntimeError(f"tomography row conservation failed: {job}")
    stale_request = np.asarray([
        hs.js_divergence(np.asarray([reuse]), np.asarray([full]), "F")
        for full, reuse in zip(baseline["full_logit"], baseline["reuse_logit"], strict=True)
    ], dtype=np.float64)
    risk = user_means(baseline["uid"].to_numpy(), stale_request)
    uids = sorted(risk)
    risks = np.asarray([risk[uid] for uid in uids], dtype=np.float64)
    total = float(risks.sum())
    fractions = yaml.safe_load(CONTRACT.read_text())["risk_concentration"]["top_fractions"]
    concentration = {}
    for fraction in fractions:
        chosen = top_indices(risk, float(fraction))
        share = float(sum(risk[uid] for uid in chosen) / total) if total > NUMERIC_FLOOR else None
        concentration[f"top_{int(round(100 * float(fraction)))}pct"] = {
            "users": len(chosen), "population_fraction": len(chosen) / len(uids),
            "risk_share": share,
            "lift_over_uniform": share / (len(chosen) / len(uids)) if share is not None else None,
        }
    action_recovery: dict[str, dict[int, float]] = {}
    for action in expected_actions:
        selected = frame[frame["action"] == action]
        if not np.array_equal(selected["request_id"].to_numpy(), baseline["request_id"].to_numpy()):
            raise RuntimeError(f"action request order mismatch: {job} {action}")
        residual = np.asarray([
            hs.js_divergence(np.asarray([diagnostic]), np.asarray([full]), "F")
            for full, diagnostic in zip(selected["full_logit"], selected["diagnostic_logit"], strict=True)
        ], dtype=np.float64)
        action_recovery[action] = user_means(selected["uid"].to_numpy(), stale_request - residual)
    recovery_capture = {}
    for action, recovery in action_recovery.items():
        signed_all = float(sum(recovery.values()) / total) if total > NUMERIC_FLOOR else None
        by_budget = {}
        for fraction in fractions:
            chosen = top_indices(risk, float(fraction))
            signed = float(sum(recovery[uid] for uid in chosen) / total) if total > NUMERIC_FLOOR else None
            positive = float(sum(max(0.0, recovery[uid]) for uid in chosen) / total) if total > NUMERIC_FLOOR else None
            by_budget[f"top_{int(round(100 * float(fraction)))}pct_by_S"] = {
                "signed_global_stale_recovery_fraction": signed,
                "positive_only_global_stale_recovery_fraction": positive,
            }
        recovery_values = np.asarray([recovery[uid] for uid in uids], dtype=np.float64)
        recovery_capture[action] = {
            "signed_all_user_recovery_fraction": signed_all,
            "risk_recovery_pearson": float(np.corrcoef(risks, recovery_values)[0, 1]) if np.std(risks) > 0 and np.std(recovery_values) > 0 else None,
            "top_risk_allocation_companion": by_budget,
        }
    best = {uid: max(action_recovery[action][uid] for action in expected_actions) for uid in uids}
    return {
        "release": job.release, "model": job.model, "seed": job.seed,
        "source": str(path.relative_to(ROOT)), "source_sha256": p7.sha256_file(path),
        "users": len(uids), "requests": len(baseline),
        "mean_equal_user_S": float(np.mean(risks)), "total_equal_user_S_mass": total,
        "above_numeric_floor_users": int(np.sum(risks > NUMERIC_FLOOR)),
        "above_numeric_floor_fraction": float(np.mean(risks > NUMERIC_FLOOR)),
        "Gini": gini(risks),
        "effective_risk_user_count": float(total * total / np.sum(risks * risks)) if total > NUMERIC_FLOOR else None,
        "concentration": concentration,
        "diagnostic_action_recovery_capture": recovery_capture,
        "diagnostic_per_user_best_action_recovery_fraction": float(sum(best.values()) / total) if total > NUMERIC_FLOOR else None,
    }


def aggregate(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for release in ("r0", "r1_edge1", "r1_edge2", "r2"):
        for model in ("m0_f", "m1"):
            rows = sorted([row for row in cells if row["release"] == release and row["model"] == model], key=lambda row: row["seed"])
            shares = {}
            for key in ("top_1pct", "top_5pct", "top_10pct"):
                points = [row["concentration"][key]["risk_share"] for row in rows]
                finite = [point for point in points if point is not None]
                shares[key] = {"seed_points": points, "equal_seed_mean": float(np.mean(finite)) if finite else None}
            output.append({
                "release": release, "model": model, "seed_order": [row["seed"] for row in rows],
                "risk_share": shares,
                "Gini_seed_points": [row["Gini"] for row in rows],
                "effective_risk_user_count_seed_points": [row["effective_risk_user_count"] for row in rows],
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite P9.1-C result: {args.output}")
    if not 1 <= args.workers <= 24:
        raise ValueError("workers must be in [1,24]")
    validate_contract_inputs()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        cells = list(executor.map(build_cell, ledger.jobs()))
    payload = {
        "status": "P9_1_label_free_user_risk_concentration_complete",
        "contract_hash": p7.sha256_file(CONTRACT),
        "state_unit": "uid_equal_weighted_mean_request_S",
        "diagnostic_recovery_is_not_executable_or_a_cost_frontier": True,
        "cells": cells, "aggregate": aggregate(cells),
        "notes": [
            "Risk ordering uses only sealed F fidelity outputs and breaks ties by ascending uid.",
            "R0 totals at or below the numeric floor report null concentration rather than arbitrary shares.",
            "Recovery capture assumes a diagnostic splice only for selected users and is an opportunity companion, not an executable action result.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
