#!/usr/bin/env python3
"""Describe all-state cutover risk concentration and fixed-action opportunity.

This is an offline-oracle analysis of already sealed P9.8 state metrics.  It
does not train a scheduler and does not turn CurrentExact probes into online
features.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
P9_8 = ROOT / "results/p9/p9_8_cutover_profiler_v1.json"
POPULATION = ROOT / "data/manifests/p9_full_population_v1"
OUTPUT = ROOT / "results/p9/p9_8_cutover_opportunity_v1.json"
FRACTIONS = (0.01, 0.05, 0.10, 0.20)
PARTIAL_ACTIONS = (
    "layer0_recent128", "layer0_middle", "layer0_full", "hybrid_tail128"
)


def gini(values: np.ndarray) -> float | None:
    values = np.sort(np.asarray(values, dtype=np.float64))
    total = float(values.sum())
    if total <= 1e-20:
        return None
    n = len(values)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float(np.sum((2 * ranks - n - 1) * values) / (n * total))


def action_work(action: str, lengths: np.ndarray) -> np.ndarray:
    lengths = lengths.astype(np.int64)
    if action == "noop":
        return np.zeros_like(lengths)
    if action == "layer0_recent128":
        return np.minimum(lengths, 128)
    if action == "layer0_middle":
        start = lengths // 4
        stop = np.maximum(start + 1, (3 * lengths + 1) // 4)
        return stop - start
    if action == "layer0_full":
        return lengths
    if action == "hybrid_tail128":
        return 4 * np.minimum(lengths, 128)
    if action == "exact_all":
        return 4 * lengths
    raise ValueError(action)


def analyze_cell(cell: dict) -> dict:
    frame = pq.read_table(ROOT / cell["state_metrics_path"]).to_pandas()
    edge = "edge2" if cell["release"] == "r1_edge2" else "edge1"
    population = pq.read_table(
        POPULATION / edge / "states.parquet", columns=["uid", "effective_prefix_length"]
    ).to_pandas()
    lengths = dict(zip(population["uid"].astype(int), population["effective_prefix_length"].astype(int)))
    pivot = frame.pivot(index="uid", columns="action", values="mse").sort_index()
    uid = pivot.index.to_numpy(dtype=np.int64)
    if any(int(value) not in lengths for value in uid):
        raise RuntimeError("state metric uid absent from full migration population")
    state_lengths = np.asarray([lengths[int(value)] for value in uid], dtype=np.int64)
    noop = pivot["noop"].to_numpy(dtype=np.float64)
    total_risk = float(noop.sum())
    exact_work = action_work("exact_all", state_lengths)
    total_exact_work = float(exact_work.sum())
    order = np.lexsort((uid, -noop))
    concentration = {}
    for fraction in FRACTIONS:
        count = max(1, math.ceil(len(uid) * fraction))
        chosen = order[:count]
        concentration[f"top_{int(100*fraction)}pct"] = {
            "states": count,
            "risk_share": float(noop[chosen].sum() / total_risk) if total_risk > 1e-20 else None,
        }
    actions = {}
    for action in PARTIAL_ACTIONS + ("exact_all",):
        residual = pivot[action].to_numpy(dtype=np.float64)
        benefit = noop - residual
        work = action_work(action, state_lengths)
        allocations = {}
        for fraction in FRACTIONS:
            count = max(1, math.ceil(len(uid) * fraction))
            chosen = order[:count]
            allocations[f"top_{int(100*fraction)}pct_by_noop_risk"] = {
                "global_risk_recovery_fraction": (
                    float(benefit[chosen].sum() / total_risk) if total_risk > 1e-20 else None
                ),
                "exact_equivalent_token_layer_work_fraction": float(work[chosen].sum() / total_exact_work),
            }
        actions[action] = {
            "uniform_recovery_fraction": (
                float(benefit.sum() / total_risk) if total_risk > 1e-20 else None
            ),
            "uniform_exact_equivalent_token_layer_work_fraction": float(work.sum() / total_exact_work),
            "positive_benefit_state_fraction": float(np.mean(benefit > 0)) if total_risk > 1e-20 else None,
            "risk_benefit_pearson": (
                float(np.corrcoef(noop, benefit)[0, 1])
                if np.std(noop) > 0 and np.std(benefit) > 0 else None
            ),
            "top_risk_allocations": allocations,
        }
    return {
        "release": cell["release"], "model": cell["model"], "seed": cell["seed"],
        "states": len(uid), "mean_noop_MSE": float(np.mean(noop)),
        "Gini_noop_MSE": gini(noop),
        "effective_risk_state_count": (
            float(total_risk * total_risk / np.sum(noop * noop)) if total_risk > 1e-20 else None
        ),
        "concentration": concentration, "actions": actions,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    source = json.loads(P9_8.read_text())
    with ProcessPoolExecutor(max_workers=12) as pool:
        cells = list(pool.map(analyze_cell, source["cells"]))
    aggregates = []
    for release in ("r0", "r1_edge1", "r1_edge2", "r2"):
        for model in ("m0_f", "m1"):
            group = sorted(
                [row for row in cells if row["release"] == release and row["model"] == model],
                key=lambda row: row["seed"],
            )
            aggregates.append({
                "release": release, "model": model,
                "seed_order": [row["seed"] for row in group],
                "Gini_seed_points": [row["Gini_noop_MSE"] for row in group],
                "top_risk_share": {
                    f"top_{int(100*fraction)}pct": [
                        row["concentration"][f"top_{int(100*fraction)}pct"]["risk_share"]
                        for row in group
                    ] for fraction in FRACTIONS
                },
                "actions": {
                    action: {
                        "uniform_recovery_seed_points": [row["actions"][action]["uniform_recovery_fraction"] for row in group],
                        "positive_benefit_state_fraction_seed_points": [row["actions"][action]["positive_benefit_state_fraction"] for row in group],
                        "risk_benefit_pearson_seed_points": [row["actions"][action]["risk_benefit_pearson"] for row in group],
                        "uniform_work_fraction": group[0]["actions"][action]["uniform_exact_equivalent_token_layer_work_fraction"],
                        "top_risk_allocations_seed_points": {
                            f"top_{int(100*fraction)}pct": [
                                row["actions"][action]["top_risk_allocations"][f"top_{int(100*fraction)}pct_by_noop_risk"]
                                for row in group
                            ] for fraction in FRACTIONS
                        },
                    } for action in PARTIAL_ACTIONS + ("exact_all",)
                },
            })
    payload = {
        "status": "P9_8_all_state_offline_opportunity_characterized",
        "p9_8_adjudication_sha256": p7.sha256_file(P9_8),
        "evidence_boundary": {
            "exact_probe_and_risk_ordering": "offline_oracle_only",
            "future_request_or_label_used": False,
            "scheduler_trained_or_authorized": False,
            "top_risk_allocation": "oracle_opportunity_upper_bound_not_deployable_policy",
        },
        "cells": cells, "aggregates": aggregates,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
