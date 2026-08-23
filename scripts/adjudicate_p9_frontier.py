#!/usr/bin/env python3
"""Build P9.11 uniform and offline state-action work/fidelity frontiers."""

from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_11_frontier_contract_v1.yaml"
P9_7 = ROOT / "results/p9/p9_7_full_population_costs_v1.json"
P9_8 = ROOT / "results/p9/p9_8_cutover_profiler_v1.json"
P9_9 = ROOT / "results/p9/p9_9_heldout_rolling_quality_v1.json"
P9_10 = ROOT / "results/p9/p9_10_full_population_runtime_v1.json"
POPULATION = ROOT / "data/manifests/p9_full_population_v1"
OUTPUT = ROOT / "results/p9/p9_11_frontier_v1.json"
ACTIONS = ("noop", "layer0_recent128", "layer0_middle", "layer0_full", "hybrid_tail128", "exact_all")


def validate() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p9_7_population_cost_sha256": P9_7,
        "p9_8_all_state_fidelity_sha256": P9_8,
        "p9_8_opportunity_sha256": ROOT / "results/p9/p9_8_cutover_opportunity_v1.json",
        "p9_9_heldout_quality_sha256": P9_9,
        "p9_10_runtime_sha256": P9_10,
        "p9_10_runtime_seal_sha256": ROOT / "results/p9/p9_10_full_population_runtime_raw_seal_v1.json",
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P9.11 input hash mismatch: {key}")
    return contract


def token_layer_cost(action: str, lengths: np.ndarray) -> np.ndarray:
    if action == "noop": return np.zeros_like(lengths)
    if action == "layer0_recent128": return np.minimum(lengths, 128)
    if action == "layer0_middle":
        start = lengths // 4
        return np.maximum(start + 1, (3 * lengths + 1) // 4) - start
    if action == "layer0_full": return lengths
    if action == "hybrid_tail128": return 4 * np.minimum(lengths, 128)
    if action == "exact_all": return 4 * lengths
    raise ValueError(action)


def concave_segments(costs, benefits, actions):
    best_by_cost = {}
    for cost, benefit, action in zip(costs, benefits, actions, strict=True):
        if cost not in best_by_cost or benefit > best_by_cost[cost][0]:
            best_by_cost[cost] = (benefit, action)
    points = []
    best_benefit = -np.inf
    for cost in sorted(best_by_cost):
        benefit, action = best_by_cost[cost]
        if benefit > best_benefit + 1e-20:
            points.append((float(cost), float(benefit), action))
            best_benefit = benefit
    hull = []
    for point in points:
        while len(hull) >= 2:
            left, middle = hull[-2], hull[-1]
            slope_left = (middle[1] - left[1]) / (middle[0] - left[0])
            slope_right = (point[1] - middle[1]) / (point[0] - middle[0])
            if slope_left <= slope_right:
                hull.pop()
            else:
                break
        hull.append(point)
    return [
        {
            "delta_cost": hull[i][0] - hull[i - 1][0],
            "delta_benefit": hull[i][1] - hull[i - 1][1],
            "slope": (hull[i][1] - hull[i - 1][1]) / (hull[i][0] - hull[i - 1][0]),
            "action": hull[i][2],
        }
        for i in range(1, len(hull))
        if hull[i][1] > hull[i - 1][1]
    ]


def near_optimal(segments, budget):
    heap = []
    for state, values in enumerate(segments):
        if values:
            heapq.heappush(heap, (-values[0]["slope"], state, 0))
    spent = benefit = 0.0
    action_counts = {action: 0 for action in ACTIONS}
    current_action = ["noop"] * len(segments)
    while heap:
        _, state, index = heapq.heappop(heap)
        segment = segments[state][index]
        if spent + segment["delta_cost"] <= budget + 1e-9:
            spent += segment["delta_cost"]
            benefit += segment["delta_benefit"]
            current_action[state] = segment["action"]
            if index + 1 < len(segments[state]):
                nxt = segments[state][index + 1]
                heapq.heappush(heap, (-nxt["slope"], state, index + 1))
    for action in current_action:
        action_counts[action] += 1
    return spent, benefit, action_counts


def runtime_condition(runtime: dict, release: str, model: str) -> dict:
    if release == "r1_edge2":
        name = "edge2_m1_r1_edge2_seed17"
    elif model == "m0_f":
        name = "edge1_m0_r2_seed17"
    else:
        name = "edge1_m1_r2_seed17"
    return next(row for row in runtime["conditions"] if row["condition"]["name"] == name)


def cell_frontier(cell, contract, costs, quality, runtime):
    release, model, seed = cell["release"], cell["model"], cell["seed"]
    edge = "edge2" if release == "r1_edge2" else "edge1"
    state = pq.read_table(ROOT / cell["state_metrics_path"]).to_pandas()
    pivot = state.pivot(index="uid", columns="action", values="mse").sort_index()
    population = pq.read_table(POPULATION / edge / "states.parquet", columns=["uid", "effective_prefix_length"]).to_pandas()
    length_by_uid = dict(zip(population["uid"].astype(int), population["effective_prefix_length"].astype(int)))
    lengths = np.asarray([length_by_uid[int(uid)] for uid in pivot.index], dtype=np.int64)
    noop = pivot["noop"].to_numpy(dtype=np.float64)
    total_risk = float(noop.sum())
    exact_work = token_layer_cost("exact_all", lengths)
    total_work = float(exact_work.sum())
    cost_rows = {row["action"]: row for row in costs[edge]["logical_costs"]}
    quality_rows = {
        row["action"]: row for row in quality
        if row["release"] == release and row["model"] == model
    }
    runtime_rows = {row["action"]: row for row in runtime_condition(runtime, release, model)["actions"]}
    uniform = []
    for action in ACTIONS:
        residual = pivot[action].to_numpy(dtype=np.float64)
        q = quality_rows[action]
        seed_index = q["seed_order"].index(seed)
        uniform.append({
            "action": action,
            "token_layer_work_fraction": cost_rows[action]["ratio_to_exact"]["recomputed_token_layers"],
            "population_MSE": float(np.mean(residual)),
            "risk_recovery_fraction": float((total_risk - residual.sum()) / total_risk) if total_risk > 1e-20 else None,
            "measured_kernel_rollout_seconds": runtime_rows[action]["rollout_seconds_median"],
            "kernel_plus_PCIe_proxy_seconds": runtime_rows[action]["kernel_plus_PCIe_proxy_seconds"],
            "action_minus_current_logloss": q["action_minus_current_logloss_seed_points"][seed_index],
            "action_minus_current_ROC_AUC": q["action_minus_current_ROC_AUC_seed_points"][seed_index],
            "action_minus_current_dislike_PR_AUC": q["action_minus_current_dislike_PR_AUC_seed_points"][seed_index],
        })
    if total_risk <= 1e-20:
        return {"release": release, "model": model, "seed": seed, "states": len(pivot), "uniform": uniform, "allocations": None}
    action_costs = {action: token_layer_cost(action, lengths) for action in ACTIONS}
    benefits = {action: noop - pivot[action].to_numpy(dtype=np.float64) for action in ACTIONS}
    segments = []
    for index in range(len(lengths)):
        segments.append(concave_segments(
            [action_costs[action][index] for action in ACTIONS],
            [benefits[action][index] for action in ACTIONS], ACTIONS,
        ))
    budgets = [float(value) for value in contract["cost_axes"]["budgets_exact_fraction"]]
    order = np.lexsort((pivot.index.to_numpy(dtype=np.int64), -noop))
    allocations = []
    random_repetitions = int(contract["policies"]["random_exact_allocation"]["repetitions"])
    for fraction in budgets:
        budget = fraction * total_work
        best_uniform = min(
            [row for row in uniform if row["token_layer_work_fraction"] <= fraction + 1e-12],
            key=lambda row: row["population_MSE"],
        )
        spent = recovered = 0.0
        for index in order:
            if spent + exact_work[index] <= budget + 1e-9:
                spent += exact_work[index]
                recovered += noop[index]
        random_points = []
        for repetition in range(random_repetitions):
            key = f"{release}:{model}:{seed}:{fraction}:{repetition}".encode()
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
            random_order = rng.permutation(len(lengths))
            random_spent = random_recovered = 0.0
            for index in random_order:
                if random_spent + exact_work[index] <= budget + 1e-9:
                    random_spent += exact_work[index]
                    random_recovered += noop[index]
            random_points.append(random_recovered / total_risk)
        near_spent, near_benefit, counts = near_optimal(segments, budget)
        allocations.append({
            "budget_fraction": fraction,
            "version_level_best_action": best_uniform["action"],
            "version_level_recovery_fraction": best_uniform["risk_recovery_fraction"],
            "top_risk_exact_recovery_fraction": recovered / total_risk,
            "top_risk_exact_spent_fraction": spent / total_work,
            "random_exact_recovery_mean": float(np.mean(random_points)),
            "random_exact_recovery_seed_points": random_points,
            "near_optimal_recovery_fraction": near_benefit / total_risk,
            "near_optimal_spent_fraction": near_spent / total_work,
            "near_optimal_action_counts": counts,
        })
    return {"release": release, "model": model, "seed": seed, "states": len(pivot), "uniform": uniform, "allocations": allocations}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    contract = validate()
    p8 = json.loads(P9_8.read_text())
    cost_payload = json.loads(P9_7.read_text())
    costs = {row["edge"]: row for row in cost_payload["edges"]}
    quality = json.loads(P9_9.read_text())["aggregates"]
    runtime = json.loads(P9_10.read_text())
    cells = [cell_frontier(cell, contract, costs, quality, runtime) for cell in p8["cells"]]
    aggregate = []
    for release in ("r0", "r1_edge1", "r1_edge2", "r2"):
        for model in ("m0_f", "m1"):
            group = sorted([row for row in cells if row["release"] == release and row["model"] == model], key=lambda row: row["seed"])
            entry = {"release": release, "model": model, "seed_order": [row["seed"] for row in group]}
            if group[0]["allocations"] is not None:
                entry["budgets"] = []
                for index, fraction in enumerate(contract["cost_axes"]["budgets_exact_fraction"]):
                    rows = [cell["allocations"][index] for cell in group]
                    entry["budgets"].append({
                        "budget_fraction": float(fraction),
                        "version_level_recovery_seed_points": [row["version_level_recovery_fraction"] for row in rows],
                        "top_risk_exact_recovery_seed_points": [row["top_risk_exact_recovery_fraction"] for row in rows],
                        "random_exact_recovery_seed_means": [row["random_exact_recovery_mean"] for row in rows],
                        "near_optimal_recovery_seed_points": [row["near_optimal_recovery_fraction"] for row in rows],
                    })
            aggregate.append(entry)
    payload = {
        "status": "P9_11_uniform_and_offline_state_action_frontiers_adjudicated",
        "contract_sha256": p7.sha256_file(CONTRACT), "cells": cells, "aggregate": aggregate,
        "state_level_policy_is_offline_oracle": True,
        "quality_labels_used_for_policy_selection": False,
        "scheduler_authorized": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells)}, indent=2))


if __name__ == "__main__":
    main()
