#!/usr/bin/env python3
"""Join sealed P9.10 runtime with frozen full-population logical bytes."""

import json
from pathlib import Path

import numpy as np

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / "results/p9/p9_10_full_population_runtime_raw_seal_v1.json"
COSTS = ROOT / "results/p9/p9_7_full_population_costs_v1.json"
OUTPUT = ROOT / "results/p9/p9_10_full_population_runtime_v1.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    seal = json.loads(SEAL.read_text())
    frozen = json.loads(COSTS.read_text())
    frozen_edges = {row["edge"]: row for row in frozen["edges"]}
    conditions = []
    for artifact in seal["artifacts"]:
        path = ROOT / artifact["path"]
        if p7.sha256_file(path) != artifact["sha256"]:
            raise RuntimeError("P9.10 result changed after seal")
        result = json.loads(path.read_text())
        edge = result["condition"]["edge"]
        logical = {row["action"]: row for row in frozen_edges[edge]["logical_costs"]}
        h2d = float(result["pinned_transfer_proxy"]["H2D_GBps_median"]) * 1e9
        d2h = float(result["pinned_transfer_proxy"]["D2H_GBps_median"]) * 1e9
        actions = []
        for row in result["actions"]:
            frozen_totals = logical[row["action"]]["totals"]
            if row["logical_totals"] != frozen_totals:
                raise RuntimeError(f"P9.10 logical totals differ from P9.7: {artifact['condition']} {row['action']}")
            proxy = (
                frozen_totals["old_kv_read_bytes"] / h2d
                + frozen_totals["new_kv_write_bytes"] / d2h
                + frozen_totals["raw_history_read_bytes"] / h2d
            )
            actions.append({
                **row,
                "logical_ratio_to_exact": logical[row["action"]]["ratio_to_exact"],
                "PCIe_transfer_proxy_seconds": proxy,
                "kernel_plus_PCIe_proxy_seconds": row["rollout_seconds_median"] + proxy,
                "ideal_two_GPU_kernel_plus_PCIe_proxy_seconds": 0.5 * (row["rollout_seconds_median"] + proxy),
            })
        conditions.append({
            "condition": result["condition"], "states": result["states"],
            "batch_size_distribution": result["batch_size_distribution"],
            "raw_reconstruction_seconds": result["raw_reconstruction_seconds"],
            "parent_cache_reference_build_seconds_excluded": result["parent_cache_reference_build_seconds_excluded"],
            "transfer_proxy": result["pinned_transfer_proxy"], "actions": actions,
        })
    action_names = [row["action"] for row in conditions[0]["actions"]]
    aggregate = []
    for action in action_names:
        rows = [next(value for value in condition["actions"] if value["action"] == action) for condition in conditions]
        aggregate.append({
            "action": action,
            "per_state_ms_condition_points": [row["per_state_ms_median"] for row in rows],
            "per_state_ms_median_across_conditions": float(np.median([row["per_state_ms_median"] for row in rows])),
            "kernel_plus_PCIe_proxy_seconds_condition_points": [row["kernel_plus_PCIe_proxy_seconds"] for row in rows],
        })
    payload = {
        "status": "P9_10_full_population_runtime_and_logical_IO_adjudicated",
        "runtime_raw_seal_sha256": p7.sha256_file(SEAL),
        "frozen_cost_sha256": p7.sha256_file(COSTS),
        "conditions": conditions, "aggregate": aggregate,
        "boundaries": {
            "storage_KV_IO_measured": False,
            "PCIe_numbers_are_transfer_proxies_not_storage_throughput": True,
            "parent_cache_reference_build_excluded_because_parent_state_is_materialized": True,
            "prototype_PyTorch_kernel_not_production_kernel": True,
            "scheduler_authorized": False,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "conditions": len(conditions)}, indent=2))


if __name__ == "__main__":
    main()
