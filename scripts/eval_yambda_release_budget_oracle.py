#!/usr/bin/env python3
"""Compute all-state release-budget frontiers with an exact binary knapsack oracle.

The input risk is a target-independent cutover-probe fidelity loss.  Exact
recompute makes that loss zero; no-op reuse retains it.  The optimizer is a
0-1 knapsack, not a benefit/cost ordering heuristic, because state lengths
and therefore recompute work differ.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import Bounds, LinearConstraint, milp


EDGES = ("theta0_theta1", "theta1_theta2")
BUDGETS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
PRIMARY = "cutover_top10_regret"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def greedy_indices(order: np.ndarray, costs: np.ndarray, capacity: float) -> np.ndarray:
    selected = []
    used = 0.0
    for index in order:
        if used + costs[index] <= capacity:
            selected.append(index)
            used += costs[index]
    return np.asarray(selected, dtype=np.int64)


def exact_knapsack(benefits: np.ndarray, costs: np.ndarray, capacity: float) -> tuple[np.ndarray, dict]:
    if capacity <= 0:
        return np.empty(0, dtype=np.int64), {"solver": "boundary", "optimal": True, "mip_gap": 0.0}
    if capacity >= float(costs.sum()):
        return np.arange(len(costs), dtype=np.int64), {"solver": "boundary", "optimal": True, "mip_gap": 0.0}
    solution = milp(
        c=-benefits,
        integrality=np.ones(len(costs)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(costs[None, :], -np.inf, capacity),
        options={"mip_rel_gap": 0.0, "time_limit": 45},
    )
    if solution.x is None:
        raise RuntimeError(f"knapsack returned no feasible allocation: {solution.message}")
    mip_gap = None if solution.mip_gap is None else float(solution.mip_gap)
    return np.flatnonzero(solution.x > 0.5), {
        "solver": "scipy_highs_milp",
        "termination": str(solution.message),
        "optimal": bool(solution.success and mip_gap == 0.0),
        "mip_gap": mip_gap,
        "mip_nodes": None if solution.mip_node_count is None else int(solution.mip_node_count),
    }


def summarize(name: str, selected: np.ndarray, losses: np.ndarray, costs: np.ndarray, total_work: float) -> dict:
    residual = losses.copy()
    residual[selected] = 0.0
    used = float(costs[selected].sum())
    return {
        "policy": name,
        "exact_state_count": int(len(selected)),
        "exact_state_fraction": float(len(selected) / len(losses)),
        "exact_equivalent_work": used,
        "exact_equivalent_work_ratio": used / total_work,
        "residual_primary_fidelity_loss": {
            "mean": float(residual.mean()),
            "p50": float(np.quantile(residual, 0.50)),
            "p95": float(np.quantile(residual, 0.95)),
            "p99": float(np.quantile(residual, 0.99)),
            "total": float(residual.sum()),
        },
    }


def random_selection(costs: np.ndarray, capacity: float, seed: int) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(len(costs))
    return greedy_indices(order, costs, capacity)


def evaluate_edge(edge: str, validity: dict, snapshot_path: Path) -> dict:
    snapshot = pq.read_table(snapshot_path).to_pydict()
    rows = validity["edges"][edge]["records"]
    by_uid = {int(uid): index for index, uid in enumerate(snapshot["uid"])}
    order = np.asarray([by_uid[int(row["uid"])] for row in rows], dtype=np.int64)
    if len(order) != len(snapshot["uid"]) or len(set(order.tolist())) != len(order):
        raise ValueError("validity records and release snapshot differ")
    costs = np.asarray(snapshot["exact_token_layer_work"], dtype=np.float64)[order]
    losses = np.asarray([float(row[PRIMARY]) for row in rows], dtype=np.float64)
    last_activity_age = np.asarray(snapshot["last_activity_age_seconds"], dtype=np.float64)[order]
    activity_7d = np.asarray(snapshot["events_last_7d"], dtype=np.float64)[order]
    if np.any(costs <= 0) or np.any(losses < 0):
        raise ValueError("work and primary fidelity loss must be non-negative")
    total_work = float(costs.sum())
    policies = {}
    for budget_ratio in BUDGETS:
        capacity = total_work * budget_ratio
        oracle_selected, solver = exact_knapsack(losses, costs, capacity)
        random_runs = [summarize("random", random_selection(costs, capacity, seed), losses, costs, total_work) for seed in range(10)]
        random_mean = {
            key: float(np.mean([run[key] for run in random_runs]))
            for key in ("exact_state_count", "exact_state_fraction", "exact_equivalent_work", "exact_equivalent_work_ratio")
        }
        for key in ("mean", "p50", "p95", "p99", "total"):
            random_mean.setdefault("residual_primary_fidelity_loss", {})[key] = float(
                np.mean([run["residual_primary_fidelity_loss"][key] for run in random_runs])
            )
        policies[str(budget_ratio)] = {
            "budget_ratio_requested": budget_ratio,
            "budget_work_capacity": capacity,
            "user_level_oracle": {**summarize("user_level_oracle", oracle_selected, losses, costs, total_work), "solver": solver},
            "reuse_all": summarize("reuse_all", np.empty(0, dtype=np.int64), losses, costs, total_work),
            "exact_all": summarize("exact_all", np.arange(len(losses), dtype=np.int64), losses, costs, total_work),
            "random_mean_10_seeds": random_mean,
            "longest_prefix_first": summarize("longest_prefix_first", greedy_indices(np.argsort(-costs, kind="stable"), costs, capacity), losses, costs, total_work),
            "most_recently_active_first": summarize("most_recently_active_first", greedy_indices(np.argsort(last_activity_age, kind="stable"), costs, capacity), losses, costs, total_work),
            "highest_pre_release_activity_first": summarize("highest_pre_release_activity_first", greedy_indices(np.argsort(-activity_7d, kind="stable"), costs, capacity), losses, costs, total_work),
            "version_level_uniform_policy": summarize(
                "version_level_uniform_policy",
                np.arange(len(losses), dtype=np.int64) if budget_ratio >= 1.0 else np.empty(0, dtype=np.int64),
                losses,
                costs,
                total_work,
            ),
        }
    return {
        "state_count": len(losses),
        "exact_all_token_layer_work": total_work,
        "cost_definition": "effective_prefix_tokens * 4 HSTU layers; readout token excluded",
        "primary_fidelity": "current_model_topk_regret",
        "policies": policies,
    }


def main() -> None:
    root = Path("results/data_audit/yambda50m_v2")
    validity_path = root / "cutover_probe_validity_v1.json"
    validity = json.loads(validity_path.read_text())
    result = {
        "status": "all_materialized_state_budget_knapsack_frontier_development",
        "primary_population": "all_materialized_states_at_release",
        "primary_fidelity": "current_model_topk_regret",
        "target_injected": False,
        "budget_definition": "exact_equivalent_token_layer_work_ratio",
        "budget_points": list(BUDGETS),
        "oracle_label_policy": "Only MIP points with mip_gap == 0 are exact oracles; other points are gap-reported feasible allocations.",
        "validity_result_hash": sha256_file(validity_path),
        "edges": {},
    }
    csv_rows = []
    for edge in EDGES:
        snapshot_path = Path(f"data/manifests/yambda50m_v2_release_snapshot_{edge}.parquet")
        result["edges"][edge] = evaluate_edge(edge, validity, snapshot_path)
        result["edges"][edge]["snapshot_hash"] = sha256_file(snapshot_path)
        for budget, policy_set in result["edges"][edge]["policies"].items():
            for name, value in policy_set.items():
                if name in {"budget_ratio_requested", "budget_work_capacity"}:
                    continue
                if name == "user_level_oracle":
                    value = {key: item for key, item in value.items() if key != "solver"}
                csv_rows.append({
                    "edge_id": edge,
                    "budget_ratio_requested": budget,
                    "policy": name,
                    "exact_equivalent_work_ratio": value["exact_equivalent_work_ratio"],
                    "residual_mean_topk_regret": value["residual_primary_fidelity_loss"]["mean"],
                    "residual_p95_topk_regret": value["residual_primary_fidelity_loss"]["p95"],
                    "residual_total_topk_regret": value["residual_primary_fidelity_loss"]["total"],
                })
    output = root / "release_snapshot_budget_oracle_v1.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    with (root / "release_snapshot_budget_oracle_v1.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({edge: value["exact_all_token_layer_work"] for edge, value in result["edges"].items()}, indent=2))


if __name__ == "__main__":
    main()
