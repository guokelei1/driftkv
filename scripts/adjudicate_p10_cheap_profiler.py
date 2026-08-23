#!/usr/bin/env python3
"""Compare sealed P10 policies with P9.11 target-free frontier references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p10_0_cheap_profiler_contract_v1.yaml"
P9_11 = ROOT / "results/p9/p9_11_frontier_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("canary", "full"), required=True)
    args = parser.parse_args()
    output = ROOT / f"results/p10/p10_0_cheap_profiler_{args.mode}_v1.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    contract = yaml.safe_load(CONTRACT.read_text())
    if p7.sha256_file(P9_11) != contract["inputs"]["post_seal_oracle_reference_sha256"]:
        raise RuntimeError("P9.11 oracle reference changed")
    seal_path = ROOT / f"results/p10/p10_0_cheap_profiler_{args.mode}_seal_v1.json"
    seal = json.loads(seal_path.read_text())
    frontier = json.loads(P9_11.read_text())
    reference = {(cell["release"], cell["model"], int(cell["seed"])): cell for cell in frontier["cells"]}
    cells = []
    for artifact in seal["artifacts"]:
        result = json.loads((ROOT / artifact["result"]).read_text())
        oracle_cell = reference[(artifact["release"], artifact["model"], int(artifact["seed"]))]
        oracle_budget = {float(row["budget_fraction"]): row for row in (oracle_cell["allocations"] or [])}
        policies = []
        for row in result["policies"]:
            match = oracle_budget.get(float(row["budget_fraction"]))
            enriched = dict(row)
            if match is not None:
                enriched.update({
                    "version_level_recovery_fraction": match["version_level_recovery_fraction"],
                    "random_exact_recovery_mean": match["random_exact_recovery_mean"],
                    "offline_oracle_recovery_fraction": match["near_optimal_recovery_fraction"],
                    "fraction_of_offline_oracle_recovery": (
                        row["risk_recovery_fraction"] / match["near_optimal_recovery_fraction"]
                        if row["risk_recovery_fraction"] is not None and match["near_optimal_recovery_fraction"] > 1e-20
                        else None
                    ),
                })
            policies.append(enriched)
        cells.append({
            "release": artifact["release"], "model": artifact["model"], "seed": artifact["seed"],
            "states": artifact["states"], "policies": policies,
        })
    aggregates = []
    for release in ("r0", "r1_edge1", "r1_edge2", "r2"):
        for model in ("m0_f", "m1"):
            group = [row for row in cells if row["release"] == release and row["model"] == model]
            if not group:
                continue
            for sample_fraction in map(float, contract["probe"]["sample_fractions"]):
                for budget_fraction in map(float, contract["allocation"]["budgets_exact_fraction"]):
                    points = [
                        next(policy for policy in cell["policies"] if policy["sample_fraction"] == sample_fraction and policy["budget_fraction"] == budget_fraction)
                        for cell in group
                    ]
                    recovery = [row["risk_recovery_fraction"] for row in points]
                    aggregates.append({
                        "release": release, "model": model,
                        "sample_fraction": sample_fraction, "budget_fraction": budget_fraction,
                        "seed_order": [row["seed"] for row in group],
                        "risk_recovery_seed_points": recovery,
                        "risk_recovery_equal_seed_mean": float(np.mean(recovery)) if all(value is not None for value in recovery) else None,
                        "fraction_of_offline_oracle_seed_points": [row.get("fraction_of_offline_oracle_recovery") for row in points],
                    })
    payload = {
        "status": f"P10_0_{args.mode}_cheap_profiler_target_free_adjudicated_after_policy_seal",
        "contract_sha256": p7.sha256_file(CONTRACT),
        "policy_seal_sha256": p7.sha256_file(seal_path),
        "cells": cells,
        "aggregates": aggregates,
        "quality_joined": False,
        "controller_authorized": False,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells), "aggregates": len(aggregates)}, indent=2))


if __name__ == "__main__":
    main()
