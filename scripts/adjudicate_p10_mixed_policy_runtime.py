#!/usr/bin/env python3
"""Compare sealed mixed-policy runtimes with P9.10 Exact-All baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / "results/p10/p10_2_mixed_policy_runtime_raw_seal_v1.json"
P9_10 = ROOT / "results/p9/p9_10_full_population_runtime_v1.json"
P10_FIDELITY = ROOT / "results/p10/p10_0_cheap_profiler_full_v1.json"
P10_QUALITY = ROOT / "results/p10/p10_1_policy_quality_v1.json"
OUTPUT = ROOT / "results/p10/p10_2_mixed_policy_runtime_v1.json"


def exact_baseline(runtime: dict, release: str, model: str) -> float:
    if release == "r1_edge2":
        name = "edge2_m1_r1_edge2_seed17"
    elif model == "m0_f":
        name = "edge1_m0_r2_seed17"
    else:
        name = "edge1_m1_r2_seed17"
    condition = next(row for row in runtime["conditions"] if row["condition"]["name"] == name)
    exact = next(row for row in condition["actions"] if row["action"] == "exact_all")
    return float(exact["rollout_seconds_median"])


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    seal = json.loads(SEAL.read_text())
    if seal["status"] != "P10_2_all_10_mixed_policy_runtime_results_sealed":
        raise RuntimeError("P10.2 runtime is not sealed")
    baseline = json.loads(P9_10.read_text())
    fidelity = json.loads(P10_FIDELITY.read_text())
    quality = json.loads(P10_QUALITY.read_text())
    fidelity_cells = {
        (cell["release"], cell["model"], int(cell["seed"])): cell
        for cell in fidelity["cells"]
    }
    quality_cells = {
        (cell["release"], cell["model"], int(cell["seed"])): cell
        for cell in quality["cells"]
    }
    rows = []
    for artifact in seal["artifacts"]:
        result = json.loads((ROOT / artifact["path"]).read_text())
        condition = result["condition"]
        key = (condition["release"], condition["model"], int(condition["seed"]))
        sample = float(condition["sample_fraction"])
        budget = float(condition["budget_fraction"])
        fidelity_policy = next(
            row for row in fidelity_cells[key]["policies"]
            if row["sample_fraction"] == sample and row["budget_fraction"] == budget
        )
        quality_policy = next(
            row for row in quality_cells[key]["evaluations"]
            if row["sample_fraction"] == sample and row["budget_fraction"] == budget
        )
        exact_seconds = exact_baseline(baseline, condition["release"], condition["model"])
        mixed_seconds = float(result["mixed_rollout_seconds_median"])
        rows.append({
            "condition": condition,
            "states": result["states"],
            "logical_token_layer_fraction_of_Exact": result["logical_token_layer_fraction_of_Exact"],
            "mixed_rollout_seconds_median": mixed_seconds,
            "Exact_All_rollout_seconds_median": exact_seconds,
            "runtime_fraction_of_Exact": mixed_seconds / exact_seconds,
            "runtime_saving_fraction_vs_Exact": 1.0 - mixed_seconds / exact_seconds,
            "target_free_risk_recovery_fraction": fidelity_policy["risk_recovery_fraction"],
            "Noop_minus_policy_logloss": quality_policy["Noop_minus_policy"]["log_loss"],
            "policy_minus_CurrentExact_logloss": quality_policy["policy_minus_CurrentExact"]["log_loss"],
            "Noop_minus_policy_ROC_AUC": quality_policy["Noop_minus_policy"]["ROC_AUC"],
            "Noop_minus_policy_dislike_PR_AUC": quality_policy["Noop_minus_policy"]["dislike_PR_AUC"],
            "Noop_minus_policy_dislike_only_logloss": quality_policy["Noop_minus_policy"]["dislike_only_log_loss"],
            "operation_state_counts": result["operation_state_counts"],
        })
    m1_r2 = [row for row in rows if row["condition"]["release"] == "r2" and row["condition"]["model"] == "m1"]
    payload = {
        "status": "P10_2_mixed_policy_runtime_cost_fidelity_quality_frontier_adjudicated",
        "runtime_seal_sha256": p7.sha256_file(SEAL),
        "rows": rows,
        "summary": {
            "runtime_fraction_of_Exact_range": [
                float(min(row["runtime_fraction_of_Exact"] for row in rows)),
                float(max(row["runtime_fraction_of_Exact"] for row in rows)),
            ],
            "runtime_saving_fraction_vs_Exact_range": [
                float(min(row["runtime_saving_fraction_vs_Exact"] for row in rows)),
                float(max(row["runtime_saving_fraction_vs_Exact"] for row in rows)),
            ],
            "M1_R2_by_sample_budget": [
                {
                    "sample_fraction": row["condition"]["sample_fraction"],
                    "budget_fraction": row["condition"]["budget_fraction"],
                    "runtime_fraction_of_Exact": row["runtime_fraction_of_Exact"],
                    "risk_recovery_fraction": row["target_free_risk_recovery_fraction"],
                    "Noop_minus_policy_logloss": row["Noop_minus_policy_logloss"],
                }
                for row in m1_r2
            ],
        },
        "logical_budget_does_not_equal_runtime_fraction": True,
        "quality_labels_used_for_policy_selection": False,
        "controller_authorized": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
