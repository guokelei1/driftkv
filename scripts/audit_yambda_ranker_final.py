#!/usr/bin/env python3
"""Finalize development-only controller audit before any theta3 decision.

Produces the required cross-edge frontier table, feature provenance/ablation,
permutation importance, and a UID-hash-disjoint companion.  UID is used only
to form the companion split and is never passed to the model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

from eval_yambda_metadata_risk_ranker import FEATURES, area, load, models, selected_from_prediction
from eval_yambda_release_budget_oracle import BUDGETS, greedy_indices, summarize


ROOT = Path("results/data_audit/yambda50m_v2")
GROUPS = {
    "history_and_cost": ("effective_prefix_length", "raw_prefix_length", "history_cap_hit", "exact_recompute_cost"),
    "recency_and_activity": ("last_activity_age_seconds", "events_last_1d", "events_last_7d", "events_last_30d"),
    "diversity_and_behavior": ("unique_items_last_7d", "unique_artists_last_7d", "repeat_ratio_last_7d", "organic_ratio_last_7d"),
}
FEATURE_PROVENANCE = {
    "effective_prefix_length": "release snapshot exact-parent state header; pre-release",
    "raw_prefix_length": "release snapshot raw history counter; pre-release",
    "history_cap_hit": "derived from effective_prefix_length >= 512; pre-release",
    "last_activity_age_seconds": "release cutoff minus pre-release last activity; pre-release",
    "events_last_1d": "pre-release rolling event counter",
    "events_last_7d": "pre-release rolling event counter",
    "events_last_30d": "pre-release rolling event counter",
    "unique_items_last_7d": "pre-release rolling distinct counter",
    "unique_artists_last_7d": "pre-release artist mapping/count; pre-release",
    "repeat_ratio_last_7d": "pre-release rolling count derived statistic",
    "organic_ratio_last_7d": "pre-release rolling behavior statistic",
    "exact_recompute_cost": "effective_prefix_length * 4 layers; deterministic release-time cost",
}


def user_half(uid: np.ndarray) -> np.ndarray:
    return np.asarray([int.from_bytes(hashlib.sha256(str(int(value)).encode()).digest()[:8], "little") % 2 for value in uid])


def subset(data: dict, mask: np.ndarray) -> dict:
    return {key: (value[mask] if isinstance(value, np.ndarray) else value) for key, value in data.items()}


def points(prediction, test):
    total = float(test["cost"].sum())
    rows = []
    selected_by_budget = {}
    for budget in BUDGETS:
        capacity = total * budget
        selected = selected_from_prediction(prediction, test["cost"], capacity)
        longest = greedy_indices(np.argsort(-test["prefix"], kind="stable"), test["cost"], capacity)
        recent = greedy_indices(np.argsort(test["age"], kind="stable"), test["cost"], capacity)
        active = greedy_indices(np.argsort(-test["activity"], kind="stable"), test["cost"], capacity)
        rows.append({
            "budget_ratio_requested": budget,
            "gbdt": summarize("gbdt", selected, test["label_held"], test["cost"], total),
            "longest_prefix": summarize("longest_prefix", longest, test["label_held"], test["cost"], total),
            "recent_active": summarize("recent_active", recent, test["label_held"], test["cost"], total),
            "highest_activity": summarize("highest_activity", active, test["label_held"], test["cost"], total),
        })
        selected_by_budget[budget] = selected
    return rows, selected_by_budget


def train_test(train, test, feature_indices=None):
    model = models()["hgb_80_leaf8"]
    if feature_indices is None:
        feature_indices = np.arange(len(FEATURES))
    model.fit(train["X"][:, feature_indices], train["label_dev"])
    prediction = np.maximum(model.predict(test["X"][:, feature_indices]), 0.0)
    frontier, selected = points(prediction, test)
    return model, prediction, frontier, selected


def compact_point(row: dict, name: str) -> dict:
    value = row[name]
    loss = value["residual_primary_fidelity_loss"]
    return {"work_ratio": value["exact_equivalent_work_ratio"], "mean": loss["mean"], "p95": loss["p95"], "p99": loss["p99"]}


def audit_direction(train_edge: str, test_edge: str) -> dict:
    train, test = load(train_edge), load(test_edge)
    model, prediction, frontier, selections = train_test(train, test)
    perm = permutation_importance(model, test["X"], test["label_held"], n_repeats=20, random_state=37, scoring="neg_mean_absolute_error")
    ablations = {}
    for name, names in GROUPS.items():
        indices = np.asarray([FEATURES.index(feature) for feature in names])
        _, _, group_frontier, _ = train_test(train, test, indices)
        ablations[name] = area(group_frontier, "gbdt")
    singles = {}
    for index, feature in enumerate(FEATURES):
        _, _, one_frontier, _ = train_test(train, test, np.asarray([index]))
        singles[feature] = area(one_frontier, "gbdt")
    train_mask = user_half(train["uid"]) == 0
    test_mask = user_half(test["uid"]) == 1
    disjoint_train, disjoint_test = subset(train, train_mask), subset(test, test_mask)
    _, _, disjoint_frontier, _ = train_test(disjoint_train, disjoint_test)
    full_area = {name: area(frontier, name) for name in ("gbdt", "longest_prefix", "recent_active", "highest_activity")}
    near_oracle = json.loads((ROOT / "qmain_heldout_panel_frontier_v1.json").read_text())["edges"][test_edge]
    near_points = near_oracle["budget_points"]
    table = []
    for row, upper in zip(frontier, near_points):
        record = {"budget_ratio_requested": row["budget_ratio_requested"]}
        for name in ("gbdt", "longest_prefix", "recent_active", "highest_activity"):
            record[name] = compact_point(row, name)
        value = upper["development_selected_near_oracle"]
        record["near_optimal_development_selection"] = {"work_ratio": value["exact_equivalent_work_ratio"], **value["residual_primary_fidelity_loss"]}
        table.append(record)
    return {
        "train_edge": train_edge,
        "test_edge": test_edge,
        "full_cross_edge": {
            "frontier_area": full_area,
            "ranking_spearman": float(spearmanr(prediction, test["label_held"]).statistic),
            "table": table,
        },
        "near_optimal_reference": {"frontier_area": near_oracle["frontier_area_mean_residual_regret"]["development_selected_near_oracle"], "solver_note": "MIP selections on development-panel label; held-out-panel fidelity evaluation"},
        "permutation_importance_neg_mae": {feature: {"mean": float(mean), "std": float(std)} for feature, mean, std in zip(FEATURES, perm.importances_mean, perm.importances_std)},
        "single_feature_frontier_area": singles,
        "feature_group_ablation_frontier_area": ablations,
        "user_hash_disjoint_companion": {
            "train_users": int(len(np.unique(train["uid"][train_mask]))),
            "test_users": int(len(np.unique(test["uid"][test_mask]))),
            "uid_hash_rule": "sha256(uid) low bit; UID excluded from features",
            "frontier_area": {name: area(disjoint_frontier, name) for name in ("gbdt", "longest_prefix", "recent_active", "highest_activity")},
        },
    }


def main() -> None:
    result = {
        "status": "ranker_final_audit_development_only",
        "primary_label": "Q_main development-panel mean current-model Top-K regret",
        "test_label": "Q_main held-out-panel mean current-model Top-K regret",
        "area_definition": "trapezoidal integral of mean held-out residual regret over actual exact-equivalent work ratio at [0, .1, .25, .5, .75, 1]",
        "feature_whitelist": list(FEATURES),
        "feature_provenance": FEATURE_PROVENANCE,
        "explicitly_absent": ["uid", "edge_id", "release_cutoff", "absolute_timestamp", "future_served_status", "proxy_availability", "proxy_delay", "future_append_count", "future_candidate", "future_target", "current_full_output", "full_reuse_divergence"],
        "directions": {},
    }
    for train, test in (("theta0_theta1", "theta1_theta2"), ("theta1_theta2", "theta0_theta1")):
        result["directions"][f"{train}_to_{test}"] = audit_direction(train, test)
    (ROOT / "ranker_final_audit_v1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value["full_cross_edge"]["frontier_area"] for key, value in result["directions"].items()}, indent=2))


if __name__ == "__main__":
    main()
