#!/usr/bin/env python3
"""Test whether Q_main development-panel scheduling transfers to held-out panels.

Selections are made only from development-half mean regret (or from frozen
pre-release metadata baselines).  Their residual fidelity is always measured
on the disjoint held-out panel half.  This is a development opportunity test,
not a learned controller or a qualification result.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from eval_yambda_release_budget_oracle import BUDGETS, exact_knapsack, greedy_indices, random_selection, summarize


ROOT = Path("results/data_audit/yambda50m_v2")
EDGES = ("theta0_theta1", "theta1_theta2")


def policy(name: str, selected: np.ndarray, held_losses: np.ndarray, costs: np.ndarray, total_work: float) -> dict:
    return summarize(name, selected, held_losses, costs, total_work)


def random_mean(costs, capacity, losses, total_work, seed):
    runs = [policy("random", random_selection(costs, capacity, seed + index), losses, costs, total_work) for index in range(64)]
    return {
        "policy": "random_budget_matched_mean_64_seeds",
        "exact_state_count": float(np.mean([run["exact_state_count"] for run in runs])),
        "exact_state_fraction": float(np.mean([run["exact_state_fraction"] for run in runs])),
        "exact_equivalent_work": float(np.mean([run["exact_equivalent_work"] for run in runs])),
        "exact_equivalent_work_ratio": float(np.mean([run["exact_equivalent_work_ratio"] for run in runs])),
        "residual_primary_fidelity_loss": {
            key: float(np.mean([run["residual_primary_fidelity_loss"][key] for run in runs]))
            for key in ("mean", "p50", "p95", "p99", "total")
        },
    }


def bootstrap_difference(selected: np.ndarray, baseline: np.ndarray, losses: np.ndarray, seed: int) -> dict:
    """CI for held-out mean residual improvement of selected over baseline."""
    residual_selected = losses.copy(); residual_selected[selected] = 0.0
    residual_baseline = losses.copy(); residual_baseline[baseline] = 0.0
    difference = residual_baseline - residual_selected
    rng = np.random.default_rng(seed)
    values = [float(difference[rng.integers(0, len(difference), len(difference))].mean()) for _ in range(500)]
    return {"mean_improvement": float(difference.mean()), "p2_5": float(np.quantile(values, .025)), "p97_5": float(np.quantile(values, .975))}


def frontier_area(points: list[dict], policy_name: str) -> float:
    x = np.asarray([point[policy_name]["exact_equivalent_work_ratio"] for point in points])
    y = np.asarray([point[policy_name]["residual_primary_fidelity_loss"]["mean"] for point in points])
    return float(np.trapezoid(y[np.argsort(x)], x[np.argsort(x)]))


def evaluate(edge: str) -> tuple[dict, list[dict]]:
    risk = pq.read_table(ROOT / f"multi_panel_risk_v1_{edge}.parquet").to_pydict()
    snapshot = pq.read_table(f"data/manifests/yambda50m_v2_release_snapshot_{edge}.parquet").to_pydict()
    snapshot_by_uid = {int(uid): index for index, uid in enumerate(snapshot["uid"])}
    order = np.asarray([snapshot_by_uid[int(uid)] for uid in risk["uid"]], dtype=np.int64)
    costs = np.asarray(snapshot["exact_token_layer_work"], dtype=float)[order]
    prefix = np.asarray(snapshot["effective_prefix_length"], dtype=float)[order]
    recent_age = np.asarray(snapshot["last_activity_age_seconds"], dtype=float)[order]
    activity = np.asarray(snapshot["events_last_7d"], dtype=float)[order]
    dev_losses = np.asarray(risk["dev_mean"], dtype=float)
    held_losses = np.asarray(risk["heldout_mean"], dtype=float)
    if np.any(dev_losses < 0) or np.any(held_losses < 0) or np.any(costs <= 0):
        raise ValueError("non-negative fidelity labels and positive costs required")
    total_work = float(costs.sum())
    points, csv_rows = [], []
    for budget in BUDGETS:
        capacity = total_work * budget
        selected_dev, solver = exact_knapsack(dev_losses, costs, capacity)
        selected_held, held_solver = exact_knapsack(held_losses, costs, capacity)
        longest = greedy_indices(np.argsort(-prefix, kind="stable"), costs, capacity)
        recent = greedy_indices(np.argsort(recent_age, kind="stable"), costs, capacity)
        active = greedy_indices(np.argsort(-activity, kind="stable"), costs, capacity)
        values = {
            "development_selected_near_oracle": {**policy("development_selected_near_oracle", selected_dev, held_losses, costs, total_work), "selection_label": "development_panel_mean_regret", "solver": solver},
            "heldout_near_oracle_upper_bound": {**policy("heldout_near_oracle_upper_bound", selected_held, held_losses, costs, total_work), "selection_label": "heldout_panel_mean_regret_inadmissible_upper_bound", "solver": held_solver},
            "random_budget_matched_mean_64_seeds": random_mean(costs, capacity, held_losses, total_work, seed=17 + int(1000 * budget)),
            "longest_prefix_first": policy("longest_prefix_first", longest, held_losses, costs, total_work),
            "most_recently_active_first": policy("most_recently_active_first", recent, held_losses, costs, total_work),
            "highest_pre_release_activity_first": policy("highest_pre_release_activity_first", active, held_losses, costs, total_work),
            "version_level_uniform_policy": policy(
                "version_level_uniform_policy",
                np.arange(len(costs), dtype=np.int64) if budget >= 1.0 else np.empty(0, dtype=np.int64),
                held_losses, costs, total_work,
            ),
        }
        baseline_sets = {"random": random_selection(costs, capacity, 981 + int(1000 * budget)), "longest_prefix_first": longest, "most_recently_active_first": recent, "highest_pre_release_activity_first": active}
        values["development_selection_bootstrap_improvement"] = {
            name: bootstrap_difference(selected_dev, indices, held_losses, 5000 + index + int(1000 * budget))
            for index, (name, indices) in enumerate(baseline_sets.items())
        }
        point = {"budget_ratio_requested": budget, "budget_work_capacity": capacity, **values}
        points.append(point)
        for name, value in values.items():
            if name == "development_selection_bootstrap_improvement":
                continue
            csv_rows.append({
                "edge_id": edge,
                "budget_ratio_requested": budget,
                "policy": name,
                "exact_equivalent_work_ratio": value["exact_equivalent_work_ratio"],
                "heldout_residual_mean_topk_regret": value["residual_primary_fidelity_loss"]["mean"],
                "heldout_residual_p95_topk_regret": value["residual_primary_fidelity_loss"]["p95"],
                "heldout_residual_p99_topk_regret": value["residual_primary_fidelity_loss"]["p99"],
            })
    areas = {name: frontier_area(points, name) for name in ("development_selected_near_oracle", "random_budget_matched_mean_64_seeds", "longest_prefix_first", "most_recently_active_first", "highest_pre_release_activity_first")}
    return {
        "states": len(costs),
        "primary_fidelity": "heldout_panel_mean_current_model_topk_regret",
        "selection_semantics": "development panels only for admissible near-oracle selection; held-out panels only for evaluation",
        "exact_all_token_layer_work": total_work,
        "budget_points": points,
        "frontier_area_mean_residual_regret": areas,
    }, csv_rows


def main() -> None:
    result = {
        "status": "q_main_heldout_panel_budget_opportunity_development",
        "distribution": "Q_main_rank_decay_v1",
        "population": "all_materialized_states_at_release",
        "primary_fidelity": "mean current-model Top-K regret over held-out panels 16..31",
        "selection_fidelity": "mean current-model Top-K regret over development panels 0..15",
        "budget_definition": "exact_equivalent_token_layer_work_ratio",
        "version_level_policy_note": "within one edge, version-level policy is only Reuse All (budget 0) or Exact All (budget 1); intermediate entries are endpoint references rather than a budget-matched curve",
        "edges": {},
    }
    rows = []
    for edge in EDGES:
        result["edges"][edge], edge_rows = evaluate(edge)
        rows.extend(edge_rows)
    (ROOT / "qmain_heldout_panel_frontier_v1.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ROOT / "qmain_heldout_panel_frontier_v1.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({edge: value["frontier_area_mean_residual_regret"] for edge, value in result["edges"].items()}, indent=2))


if __name__ == "__main__":
    main()
