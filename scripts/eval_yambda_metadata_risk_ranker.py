#!/usr/bin/env python3
"""Cross-edge metadata-only risk-ranker development on Q_main labels.

The training label is complete-snapshot cutover mean Top-K regret over Q_main
development panels.  Evaluation uses the other release edge's held-out-panel
mean regret.  UID, future activity/proxy information, candidates and Full
outputs are deliberately unavailable to every model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from eval_yambda_multi_panel_risk import top_recall
from eval_yambda_release_budget_oracle import BUDGETS, greedy_indices, random_selection, summarize


ROOT = Path("results/data_audit/yambda50m_v2")
EDGES = ("theta0_theta1", "theta1_theta2")
FEATURES = (
    "effective_prefix_length", "raw_prefix_length", "history_cap_hit",
    "last_activity_age_seconds", "events_last_1d", "events_last_7d", "events_last_30d",
    "unique_items_last_7d", "unique_artists_last_7d", "repeat_ratio_last_7d",
    "organic_ratio_last_7d", "exact_recompute_cost",
)


def load(edge: str) -> dict:
    risk = pq.read_table(ROOT / f"multi_panel_risk_v1_{edge}.parquet").to_pydict()
    snapshot = pq.read_table(f"data/manifests/yambda50m_v2_release_snapshot_{edge}.parquet").to_pydict()
    by_uid = {int(uid): index for index, uid in enumerate(snapshot["uid"])}
    indices = np.asarray([by_uid[int(uid)] for uid in risk["uid"]])
    def col(name): return np.asarray(snapshot[name], dtype=float)[indices]
    data = {
        "uid": np.asarray(risk["uid"], dtype=np.int64),
        "label_dev": np.asarray(risk["dev_mean"], dtype=float),
        "label_held": np.asarray(risk["heldout_mean"], dtype=float),
        "cost": col("exact_token_layer_work"),
        "prefix": col("effective_prefix_length"),
        "age": col("last_activity_age_seconds"),
        "activity": col("events_last_7d"),
    }
    data.update({
        "effective_prefix_length": data["prefix"],
        "raw_prefix_length": col("raw_prefix_length"),
        "history_cap_hit": (data["prefix"] >= 512).astype(float),
        "last_activity_age_seconds": data["age"],
        "events_last_1d": col("events_last_1d"),
        "events_last_7d": data["activity"],
        "events_last_30d": col("events_last_30d"),
        "unique_items_last_7d": col("unique_items_last_7d"),
        "unique_artists_last_7d": col("unique_artists_last_7d"),
        "repeat_ratio_last_7d": col("repeat_ratio_last_7d"),
        "organic_ratio_last_7d": col("organic_ratio_last_7d"),
        "exact_recompute_cost": data["cost"],
    })
    data["X"] = np.column_stack([data[name] for name in FEATURES])
    if np.any(data["cost"] <= 0) or np.any(data["label_dev"] < 0) or np.any(data["label_held"] < 0):
        raise ValueError("invalid non-negative loss/cost contract")
    return data


def models() -> dict:
    scaled = lambda estimator: Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", estimator)])
    return {
        "ridge_alpha_1": scaled(Ridge(alpha=1.0)),
        "elasticnet_alpha_1e-3_l1_0.5": scaled(ElasticNet(alpha=1e-3, l1_ratio=.5, max_iter=5000, random_state=37)),
        "tree_depth_3_minleaf_64": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", DecisionTreeRegressor(max_depth=3, min_samples_leaf=64, random_state=37))]),
        "hgb_80_leaf8": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=8, min_samples_leaf=64, learning_rate=.05, l2_regularization=1.0, random_state=37))]),
    }


def selected_from_prediction(prediction: np.ndarray, costs: np.ndarray, capacity: float) -> np.ndarray:
    # One declared approximation scheduler is shared by all learned models.
    # It is benefit/cost greedy, never called an oracle.
    priority = np.maximum(prediction, 0.0) / costs
    return greedy_indices(np.argsort(-priority, kind="stable"), costs, capacity)


def area(points: list[dict], name: str) -> float:
    x = np.asarray([row[name]["exact_equivalent_work_ratio"] for row in points])
    y = np.asarray([row[name]["residual_primary_fidelity_loss"]["mean"] for row in points])
    order = np.argsort(x)
    return float(np.trapezoid(y[order], x[order]))


def bootstrap_improvement(selected, baseline, losses, seed):
    a = losses.copy(); a[selected] = 0
    b = losses.copy(); b[baseline] = 0
    values = b - a
    rng = np.random.default_rng(seed)
    sampled = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(500)]
    return {"mean": float(values.mean()), "p2_5": float(np.quantile(sampled, .025)), "p97_5": float(np.quantile(sampled, .975))}


def evaluate_direction(train_edge: str, test_edge: str) -> tuple[dict, list[dict]]:
    train, test = load(train_edge), load(test_edge)
    total_work = float(test["cost"].sum())
    result, csv_rows = {"train_edge": train_edge, "test_edge": test_edge, "feature_set": list(FEATURES), "label_train": "development_panel_mean_topk_regret", "label_test": "heldout_panel_mean_topk_regret", "scheduler": "benefit_cost_greedy_shared_by_learned_models", "models": {}}, []
    for name, model in models().items():
        model.fit(train["X"], train["label_dev"])
        prediction = np.maximum(model.predict(test["X"]), 0.0)
        ranking = {"spearman": float(spearmanr(prediction, test["label_held"]).statistic), "top10_high_risk_recall": top_recall(prediction, test["label_held"], .1), "top20_high_risk_recall": top_recall(prediction, test["label_held"], .2)}
        points = []
        for budget in BUDGETS:
            capacity = total_work * budget
            selected = selected_from_prediction(prediction, test["cost"], capacity)
            longest = greedy_indices(np.argsort(-test["prefix"], kind="stable"), test["cost"], capacity)
            recent = greedy_indices(np.argsort(test["age"], kind="stable"), test["cost"], capacity)
            active = greedy_indices(np.argsort(-test["activity"], kind="stable"), test["cost"], capacity)
            random_runs = [summarize("random", random_selection(test["cost"], capacity, 1000 + run), test["label_held"], test["cost"], total_work) for run in range(64)]
            random_mean = {
                "policy": "random_budget_matched_mean_64_seeds",
                "exact_state_count": float(np.mean([run["exact_state_count"] for run in random_runs])),
                "exact_state_fraction": float(np.mean([run["exact_state_fraction"] for run in random_runs])),
                "exact_equivalent_work": float(np.mean([run["exact_equivalent_work"] for run in random_runs])),
                "exact_equivalent_work_ratio": float(np.mean([run["exact_equivalent_work_ratio"] for run in random_runs])),
                "residual_primary_fidelity_loss": {key: float(np.mean([run["residual_primary_fidelity_loss"][key] for run in random_runs])) for key in ("mean", "p50", "p95", "p99", "total")},
            }
            point = {
                "budget_ratio_requested": budget,
                "ranker": summarize(name, selected, test["label_held"], test["cost"], total_work),
                "random_budget_matched_mean_64_seeds": random_mean,
                "longest_prefix_first": summarize("longest_prefix_first", longest, test["label_held"], test["cost"], total_work),
                "most_recently_active_first": summarize("most_recently_active_first", recent, test["label_held"], test["cost"], total_work),
                "highest_pre_release_activity_first": summarize("highest_pre_release_activity_first", active, test["label_held"], test["cost"], total_work),
                "ranker_vs_best_heuristic_bootstrap": None,
            }
            # Baseline is identified from its residual mean at this fixed budget.
            candidate_baselines = {"longest": longest, "recent": recent, "activity": active}
            best_name, best_indices = min(candidate_baselines.items(), key=lambda pair: summarize(pair[0], pair[1], test["label_held"], test["cost"], total_work)["residual_primary_fidelity_loss"]["mean"])
            point["ranker_vs_best_heuristic_bootstrap"] = {"baseline": best_name, **bootstrap_improvement(selected, best_indices, test["label_held"], 4000 + int(1000 * budget))}
            points.append(point)
            for policy_name in ("ranker", "random_budget_matched_mean_64_seeds", "longest_prefix_first", "most_recently_active_first", "highest_pre_release_activity_first"):
                value = point[policy_name]
                csv_rows.append({"train_edge": train_edge, "test_edge": test_edge, "model": name, "budget_ratio_requested": budget, "policy": policy_name, "work_ratio": value["exact_equivalent_work_ratio"], "mean_residual_regret": value["residual_primary_fidelity_loss"]["mean"], "p95_residual_regret": value["residual_primary_fidelity_loss"]["p95"], "p99_residual_regret": value["residual_primary_fidelity_loss"]["p99"]})
        result["models"][name] = {"ranking": ranking, "budget_frontier": points, "frontier_area": {"ranker": area(points, "ranker"), "random": area(points, "random_budget_matched_mean_64_seeds"), "longest": area(points, "longest_prefix_first"), "recent": area(points, "most_recently_active_first"), "activity": area(points, "highest_pre_release_activity_first")}}
    return result, csv_rows


def main() -> None:
    result = {"status": "metadata_only_cross_edge_risk_ranker_development", "warning": "No model is frozen and theta3 remains unused; choose any freeze candidate only after both-direction review.", "features_pre_release_only": True, "forbidden_features": ["uid", "future_served_status", "future_proxy_availability", "future_append_count", "future_target", "future_candidates", "current_full_output"], "directions": {}}
    rows = []
    for train, test in ((EDGES[0], EDGES[1]), (EDGES[1], EDGES[0])):
        direction, direction_rows = evaluate_direction(train, test)
        result["directions"][f"{train}_to_{test}"] = direction
        rows.extend(direction_rows)
    (ROOT / "metadata_risk_ranker_cross_edge_v1.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ROOT / "metadata_risk_ranker_frontier_v1.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({key: {model: value["frontier_area"] for model, value in direction["models"].items()} for key, direction in result["directions"].items()}, indent=2))


if __name__ == "__main__":
    main()
