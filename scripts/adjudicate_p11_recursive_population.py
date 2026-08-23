#!/usr/bin/env python3
"""Adjudicate all six sealed P11.1 target-free population cells."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p11_1_recursive_population_contract_v1.yaml"
RAW = ROOT / "results/p11/p11_1_recursive_population_raw/full"
OUTPUT = ROOT / "results/p11/p11_1_recursive_population_adjudication_v1.json"
MODELS = ("m0_f", "m1")
SEEDS = (17, 37, 71)


def quantiles(values):
    return {
        "mean": float(np.mean(values)), "P50": float(np.quantile(values, 0.50)),
        "P90": float(np.quantile(values, 0.90)), "P95": float(np.quantile(values, 0.95)),
        "P99": float(np.quantile(values, 0.99)),
    }


def concentration(values):
    ordered = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    total = float(ordered.sum())
    output = {}
    for fraction in (0.01, 0.05, 0.10):
        count = max(1, int(np.ceil(len(ordered) * fraction)))
        output[f"top_{int(fraction * 100)}pct_fraction"] = float(ordered[:count].sum() / total) if total else 0.0
    return output


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    cells, manifests = [], []
    for model in MODELS:
        for seed in SEEDS:
            directory = RAW / f"{model}_seed{seed}"
            manifest_path = directory / "raw_manifest.json"
            raw_path = directory / "state_metrics.parquet"
            if not manifest_path.exists() or not raw_path.exists():
                raise RuntimeError(f"missing P11.1 raw cell: {model} seed{seed}")
            manifest = json.loads(manifest_path.read_text())
            if manifest["raw_sha256"] != p7.sha256_file(raw_path) or manifest["contract_sha256"] != p7.sha256_file(CONTRACT):
                raise RuntimeError(f"unsealed P11.1 raw cell: {model} seed{seed}")
            rows = pq.read_table(raw_path).to_pylist()
            by_action = {}
            for row in rows:
                by_action.setdefault(row["action"], []).append(row)
            noop_mean = float(np.mean([row["bernoulli_js"] for row in by_action["recursive_noop"]]))
            action_summaries = []
            for action, action_rows in sorted(by_action.items()):
                js = [row["bernoulli_js"] for row in action_rows]
                mean_js = float(np.mean(js))
                action_summaries.append({
                    "action": action, "states": len(action_rows), "JS": quantiles(js),
                    "normalized_RMS": quantiles([row["normalized_rms"] for row in action_rows]),
                    "mean_abs_probability_shift": quantiles([
                        row["mean_abs_probability_shift"] for row in action_rows
                    ]),
                    "recovery_fraction_vs_recursive_noop": (
                        float(1.0 - mean_js / noop_mean) if noop_mean else 0.0
                    ),
                    "risk_concentration": concentration(js),
                })
            direct = float(np.mean([row["bernoulli_js"] for row in by_action["direct_age2"]]))
            onehop = float(np.mean([row["bernoulli_js"] for row in by_action["one_hop"]]))
            cells.append({
                "model": model, "seed": seed, "states": manifest["states"],
                "suffix_events": manifest["suffix_events"], "actions": action_summaries,
                "lineage_JS_ratios": {
                    "recursive_over_one_hop": float(noop_mean / onehop) if onehop else None,
                    "direct_age2_over_recursive": float(direct / noop_mean) if noop_mean else None,
                },
            })
            manifests.append({"model": model, "seed": seed, "sha256": p7.sha256_file(manifest_path)})
    aggregate = []
    actions = [row["action"] for row in cells[0]["actions"]]
    for model in MODELS:
        model_cells = [cell for cell in cells if cell["model"] == model]
        for action in actions:
            per_seed = [next(row for row in cell["actions"] if row["action"] == action) for cell in model_cells]
            recovery = [row["recovery_fraction_vs_recursive_noop"] for row in per_seed]
            aggregate.append({
                "model": model, "action": action,
                "mean_JS_across_seed_means": float(np.mean([row["JS"]["mean"] for row in per_seed])),
                "recovery_fraction_seed_mean": float(np.mean(recovery)),
                "recovery_fraction_seed_min": float(np.min(recovery)),
                "positive_recovery_seeds": int(sum(value > 0 for value in recovery)),
            })
    payload = {
        "status": "P11_1_recursive_population_target_free_adjudicated",
        "cells": cells, "seed_aggregate": aggregate, "raw_manifests": manifests,
        "contract_sha256": p7.sha256_file(CONTRACT), "quality_labels_read": False,
        "scheduler_applied": False, "blind_edge_executed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells), "aggregate_rows": len(aggregate)}, indent=2))


if __name__ == "__main__":
    main()
