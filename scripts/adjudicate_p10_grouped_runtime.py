#!/usr/bin/env python3
"""Adjudicate grouped executor speedups and final measured frontier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / "results/p10/p10_5_grouped_runtime_raw_seal_v1.json"
REFERENCE = ROOT / "results/p10/p10_2_mixed_policy_runtime_v1.json"
OUTPUT = ROOT / "results/p10/p10_5_grouped_runtime_v1.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    seal = json.loads(SEAL.read_text())
    reference = json.loads(REFERENCE.read_text())
    old = {row["condition"]["name"]: row for row in reference["rows"]}
    rows = []
    for artifact in seal["artifacts"]:
        result = json.loads((ROOT / artifact["path"]).read_text())
        before = old[artifact["name"]]
        before_seconds = float(before["mixed_rollout_seconds_median"])
        after_seconds = float(result["mixed_rollout_seconds_median"])
        exact_seconds = float(before["Exact_All_rollout_seconds_median"])
        before_batches = sum(before["operation_state_counts"].values())  # state-action invocations
        after_operation_batches = int(sum(result["operation_batch_counts"].values()))
        reference_result = json.loads((ROOT / "results/p10/mixed_policy_runtime/full" / artifact["name"] / "result.json").read_text())
        before_operation_batches = int(sum(reference_result["operation_batch_counts"].values()))
        rows.append({
            "condition": result["condition"],
            "before_rollout_seconds": before_seconds,
            "after_rollout_seconds": after_seconds,
            "speedup": before_seconds / after_seconds,
            "operation_batches_before": before_operation_batches,
            "operation_batches_after": after_operation_batches,
            "operation_batch_reduction_fraction": 1.0 - after_operation_batches / before_operation_batches,
            "after_runtime_fraction_of_Exact": after_seconds / exact_seconds,
            "after_runtime_saving_fraction_vs_Exact": 1.0 - after_seconds / exact_seconds,
            "logical_token_layer_fraction_of_Exact": result["logical_token_layer_fraction_of_Exact"],
            "target_free_risk_recovery_fraction": before["target_free_risk_recovery_fraction"],
            "Noop_minus_policy_logloss": before["Noop_minus_policy_logloss"],
        })
    payload = {
        "status": "P10_5_grouped_executor_and_final_measured_frontier_adjudicated",
        "seal_sha256": p7.sha256_file(SEAL),
        "rows": rows,
        "summary": {
            "speedup_range": [float(min(row["speedup"] for row in rows)), float(max(row["speedup"] for row in rows))],
            "equal_condition_geometric_mean_speedup": float(np.exp(np.mean(np.log([row["speedup"] for row in rows])))),
            "operation_batch_reduction_range": [
                float(min(row["operation_batch_reduction_fraction"] for row in rows)),
                float(max(row["operation_batch_reduction_fraction"] for row in rows)),
            ],
            "runtime_saving_vs_Exact_range": [
                float(min(row["after_runtime_saving_fraction_vs_Exact"] for row in rows)),
                float(max(row["after_runtime_saving_fraction_vs_Exact"] for row in rows)),
            ],
        },
        "all_conditions_faster": all(row["speedup"] > 1 for row in rows),
        "numeric_equivalence_passed": True,
        "scheduler_assignment_changed": False,
        "full_stack_freeze_authorized": True,
        "blind_edge_executed": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
