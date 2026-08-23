#!/usr/bin/env python3
"""P10.3 equal-cost nonlearning baseline gate and dislike attribution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

import train_p7_theta0 as p7
from adjudicate_p9_frontier import ACTIONS, token_layer_cost


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p10_3_scheduler_baseline_gate_contract_v1.yaml"
P10 = ROOT / "results/p10/p10_0_cheap_profiler_full_v1.json"
POLICY_SEAL = ROOT / "results/p10/p10_0_cheap_profiler_full_seal_v1.json"
QUALITY = ROOT / "results/p10/p10_1_policy_quality_v1.json"
RUNTIME = ROOT / "results/p10/p10_2_mixed_policy_runtime_v1.json"
P9_11 = ROOT / "results/p9/p9_11_frontier_v1.json"
P9_8 = ROOT / "results/p9/p9_8_cutover_profiler_v1.json"
POPULATION = ROOT / "data/manifests/p9_full_population_v1"
OUTPUT = ROOT / "results/p10/p10_3_scheduler_baseline_gate_v1.json"


def validate() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p10_0_target_free_sha256": P10,
        "p10_0_policy_seal_sha256": POLICY_SEAL,
        "p10_1_quality_sha256": QUALITY,
        "p10_2_runtime_sha256": RUNTIME,
        "p9_11_oracle_frontier_sha256": P9_11,
        "edge1_state_manifest_sha256": POPULATION / "edge1/states.parquet",
        "edge2_state_manifest_sha256": POPULATION / "edge2/states.parquet",
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P10.3 input mismatch: {key}")
    return contract


def exact_allocation(order: np.ndarray, costs: np.ndarray, budget: float) -> np.ndarray:
    selected = np.zeros(len(costs), dtype=bool)
    spent = 0.0
    for index in order:
        if spent + costs[index] <= budget + 1e-9:
            selected[index] = True
            spent += costs[index]
    return selected


def paired_bootstrap(delta: np.ndarray, key: str, repetitions: int) -> dict:
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    points = []
    for begin in range(0, repetitions, 100):
        width = min(100, repetitions - begin)
        sampled = rng.integers(0, len(delta), size=(width, len(delta)))
        points.extend(np.mean(delta[sampled], axis=1).tolist())
    return {
        "repetitions": repetitions,
        "mean": float(np.mean(delta)),
        "p2_5": float(np.quantile(points, 0.025)),
        "p97_5": float(np.quantile(points, 0.975)),
    }


def metadata_orders(frame) -> dict[str, np.ndarray]:
    uid = frame["uid"].to_numpy(dtype=np.int64)
    return {
        "longest_effective_prefix_first": np.lexsort((uid, -frame["effective_prefix_length"].to_numpy(dtype=np.int64))),
        "oldest_state_first": np.lexsort((uid, -frame["last_activity_age_seconds"].to_numpy(dtype=np.int64))),
        "most_active_30d_first": np.lexsort((uid, -frame["events_last_30d"].to_numpy(dtype=np.int64))),
        "most_unique_items_7d_first": np.lexsort((uid, -frame["unique_items_last_7d"].to_numpy(dtype=np.int64))),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    contract = validate()
    p10 = json.loads(P10.read_text())
    policy_seal = json.loads(POLICY_SEAL.read_text())
    p9 = json.loads(P9_11.read_text())
    p8 = json.loads(P9_8.read_text())
    quality = json.loads(QUALITY.read_text())
    assignment_artifacts = {(row["release"], row["model"], int(row["seed"])): row for row in policy_seal["artifacts"]}
    p9_cells = {(row["release"], row["model"], int(row["seed"])): row for row in p9["cells"]}
    p8_cells = {(row["release"], row["model"], int(row["seed"])): row for row in p8["cells"]}
    quality_cells = {(row["release"], row["model"], int(row["seed"])): row for row in quality["cells"]}
    rows = []
    repetitions = int(contract["uncertainty"]["repetitions"])
    for key in sorted(assignment_artifacts):
        release, model, seed = key
        if release == "r0":
            continue
        edge = "edge2" if release == "r1_edge2" else "edge1"
        population = pq.read_table(POPULATION / edge / "states.parquet").to_pandas().sort_values("uid").reset_index(drop=True)
        state = pq.read_table(ROOT / p8_cells[key]["state_metrics_path"], columns=["uid", "action", "mse"]).to_pandas()
        pivot = state.pivot(index="uid", columns="action", values="mse").reindex(population["uid"].astype(int))
        if pivot.isna().any().any():
            raise RuntimeError(f"population join failed: {key}")
        noop = pivot["noop"].to_numpy(dtype=np.float64)
        total_risk = float(noop.sum())
        lengths = population["effective_prefix_length"].to_numpy(dtype=np.int64)
        exact_cost = token_layer_cost("exact_all", lengths).astype(np.float64)
        total_exact = float(exact_cost.sum())
        assignments = pq.read_table(ROOT / assignment_artifacts[key]["assignments"]).to_pandas()
        action_values = {action: noop - pivot[action].to_numpy(dtype=np.float64) for action in ACTIONS}
        metadata = metadata_orders(population)
        p9_cell = p9_cells[key]
        p10_cell = next(cell for cell in p10["cells"] if (cell["release"], cell["model"], int(cell["seed"])) == key)
        for budget in map(float, contract["primary_policy"]["budgets_exact_fraction"]):
            policy = assignments[
                np.isclose(assignments["sample_fraction"], float(contract["primary_policy"]["sample_fraction"]))
                & np.isclose(assignments["budget_fraction"], budget)
            ].sort_values("uid")
            if not np.array_equal(policy["uid"].to_numpy(dtype=np.int64), population["uid"].to_numpy(dtype=np.int64)):
                raise RuntimeError(f"policy population mismatch: {key}:{budget}")
            ridge_benefit = np.asarray([
                action_values[str(action)][index] for index, action in enumerate(policy["action"])
            ], dtype=np.float64)
            p9_budget = next(row for row in p9_cell["allocations"] if float(row["budget_fraction"]) == budget)
            version_action = p9_budget["version_level_best_action"]
            deterministic = {
                "release_level_best_uniform": {
                    "action": version_action,
                    "benefit": action_values[version_action],
                }
            }
            for name, order in metadata.items():
                selected = exact_allocation(order, exact_cost, budget * total_exact)
                deterministic[name] = {
                    "action": "state_selected_exact",
                    "benefit": noop * selected,
                }
            baseline_summaries = []
            for name, value in deterministic.items():
                benefit = value["benefit"]
                baseline_summaries.append({
                    "name": name,
                    "action": value["action"],
                    "risk_recovery_fraction": float(benefit.sum() / total_risk),
                })
            strongest = max(baseline_summaries, key=lambda row: row["risk_recovery_fraction"])
            strongest_benefit = deterministic[strongest["name"]]["benefit"]
            comparison = paired_bootstrap(
                ridge_benefit - strongest_benefit,
                f"{release}:{model}:{seed}:{budget}:{strongest['name']}", repetitions,
            )
            comparison["normalized_by_mean_noop_risk"] = comparison["mean"] / float(np.mean(noop))
            ridge_policy = next(row for row in p10_cell["policies"] if row["sample_fraction"] == 0.01 and row["budget_fraction"] == budget)
            rows.append({
                "release": release, "model": model, "seed": seed, "budget_fraction": budget,
                "Ridge_risk_recovery_fraction": float(ridge_benefit.sum() / total_risk),
                "Ridge_charged_cost_fraction": ridge_policy["charged_cost_fraction"],
                "deterministic_nonlearning_baselines": baseline_summaries,
                "strongest_nonlearning_baseline": strongest,
                "Ridge_minus_strongest_paired_user_bootstrap": comparison,
                "random_Exact_recovery_mean": p9_budget["random_exact_recovery_mean"],
                "offline_oracle_recovery_fraction": p9_budget["near_optimal_recovery_fraction"],
            })
    condition_gates = []
    for release in ("r1_edge1", "r1_edge2", "r2"):
        for model in ("m0_f", "m1"):
            group = [row for row in rows if row["release"] == release and row["model"] == model]
            budgets = []
            for budget in map(float, contract["primary_policy"]["budgets_exact_fraction"]):
                points = [row for row in group if row["budget_fraction"] == budget]
                deltas = [row["Ridge_risk_recovery_fraction"] - row["strongest_nonlearning_baseline"]["risk_recovery_fraction"] for row in points]
                budgets.append({
                    "budget_fraction": budget,
                    "seed_order": [row["seed"] for row in points],
                    "recovery_difference_seed_points": deltas,
                    "equal_seed_mean_difference": float(np.mean(deltas)),
                    "positive_seed_count": int(sum(value > 0 for value in deltas)),
                    "positive_bootstrap_CI_seed_count": int(sum(row["Ridge_minus_strongest_paired_user_bootstrap"]["p2_5"] > 0 for row in points)),
                })
            passed = any(row["equal_seed_mean_difference"] > 0 and row["positive_seed_count"] >= 2 for row in budgets)
            condition_gates.append({"release": release, "model": model, "budgets": budgets, "passed": passed})
    quality_attribution = []
    for key, cell in sorted(quality_cells.items()):
        release, model, seed = key
        if release == "r0":
            continue
        for evaluation in cell["evaluations"]:
            if evaluation["sample_fraction"] != 0.01:
                continue
            policy_minus_exact = evaluation["policy_minus_CurrentExact"]["dislike_only_log_loss"]
            noop_minus_policy = evaluation["Noop_minus_policy"]["dislike_only_log_loss"]
            noop_minus_exact = noop_minus_policy + policy_minus_exact
            quality_attribution.append({
                "release": release, "model": model, "seed": seed,
                "budget_fraction": evaluation["budget_fraction"],
                "CurrentExact_minus_Noop_dislike_logloss": -noop_minus_exact,
                "Policy_minus_CurrentExact_dislike_logloss": policy_minus_exact,
                "Policy_minus_Noop_dislike_logloss": -noop_minus_policy,
            })
    passed_conditions = sum(row["passed"] for row in condition_gates)
    m1_r2_passed = next(row["passed"] for row in condition_gates if row["release"] == "r2" and row["model"] == "m1")
    global_passed = bool(m1_r2_passed and passed_conditions >= 4)
    payload = {
        "status": "P10_3_equal_cost_nonlearning_baseline_gate_adjudicated",
        "contract_sha256": p7.sha256_file(CONTRACT),
        "primary_policy_sample_fraction": 0.01,
        "rows": rows,
        "condition_gates": condition_gates,
        "global_gate": {
            "M1_R2_passed": m1_r2_passed,
            "passed_release_model_conditions": passed_conditions,
            "required_release_model_conditions": 4,
            "passed": global_passed,
        },
        "dislike_attribution": quality_attribution,
        "quality_labels_used_for_policy_or_baseline_selection": False,
        "scheduler_freeze_authorized": global_passed,
        "blind_edge_authorized_by_this_artifact": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"], "rows": len(rows),
        "condition_gates": [{"release": row["release"], "model": row["model"], "passed": row["passed"]} for row in condition_gates],
        "global_gate": payload["global_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
