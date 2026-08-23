#!/usr/bin/env python3
"""Same-charged-cost gate for the sealed recursive-lineage scheduler."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

import adjudicate_p10_scheduler_baselines as p10base
import train_p7_theta0 as p7
from adjudicate_p9_frontier import ACTIONS, concave_segments, near_optimal, token_layer_cost


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p11_3_recursive_scheduler_baseline_gate_v1.yaml"
P11_1 = ROOT / "results/p11/p11_1_recursive_population_adjudication_v1.json"
P11_2_CONTRACT = ROOT / "configs/contracts/p11_2_recursive_scheduler_replay_v1.yaml"
SEAL = ROOT / "results/p11/p11_2_recursive_scheduler_full_seal_v1.json"
RAW_METRICS = ROOT / "results/p11/p11_1_recursive_population_raw/full"
POPULATION = ROOT / "data/manifests/p9_full_population_v1/edge2/states.parquet"
OUTPUT = ROOT / "results/p11/p11_3_recursive_scheduler_baseline_gate_v1.json"


def validate():
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {"p11_1_adjudication_sha256": P11_1, "p11_2_contract_sha256": P11_2_CONTRACT,
             "p11_2_assignment_seal_sha256": SEAL}
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P11.3 input mismatch: {key}")
    return contract, yaml.safe_load(P11_2_CONTRACT.read_text())


def allocate_exact(order, costs, budget):
    selected = np.zeros(len(costs), dtype=bool)
    spent = 0.0
    for index in order:
        if spent + costs[index] <= budget + 1e-9:
            selected[index] = True
            spent += costs[index]
    return selected, spent


def random_exact(noop, costs, budget, key, repetitions):
    points = []
    for repetition in range(repetitions):
        digest = hashlib.sha256(f"{key}:{repetition}".encode()).digest()
        order = np.random.default_rng(int.from_bytes(digest[:8], "little")).permutation(len(costs))
        selected, _ = allocate_exact(order, costs, budget)
        points.append(float(noop[selected].sum()))
    return points


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    contract, scheduler = validate()
    seal = json.loads(SEAL.read_text())
    mapping = scheduler["action_mapping"]
    population_all = pq.read_table(POPULATION).to_pandas()
    rows = []
    for artifact in seal["artifacts"]:
        model, seed = artifact["model"], int(artifact["seed"])
        metrics = pq.read_table(RAW_METRICS / f"{model}_seed{seed}/state_metrics.parquet").to_pandas()
        pivot = metrics.pivot(index="uid", columns="action", values="mse")
        population = population_all[population_all["uid"].astype(int).isin(pivot.index.astype(int))]
        population = population.sort_values("uid").reset_index(drop=True)
        pivot = pivot.reindex(population["uid"].astype(int))
        uids = population["uid"].to_numpy(dtype=np.int64)
        lengths = population["effective_prefix_length"].to_numpy(dtype=np.int64)
        loss = {action: pivot[source].to_numpy(dtype=np.float64) for action, source in mapping.items()}
        noop, total_risk = loss["noop"], float(loss["noop"].sum())
        costs = {action: token_layer_cost(action, lengths).astype(np.float64) for action in ACTIONS}
        exact_cost, total_exact = costs["exact_all"], float(costs["exact_all"].sum())
        benefits = {action: noop - loss[action] for action in ACTIONS}
        assignments = pq.read_table(ROOT / artifact["assignments"]).to_pandas()
        metadata = p10base.metadata_orders(population)
        for sample_fraction in (
            float(scheduler["frozen_scheduler"]["primary_probe_rate"]),
            float(scheduler["frozen_scheduler"]["companion_probe_rate"]),
        ):
            for budget_fraction in map(float, scheduler["frozen_scheduler"]["budgets_exact_fraction"]):
                policy = assignments[
                    np.isclose(assignments["sample_fraction"], sample_fraction)
                    & np.isclose(assignments["budget_fraction"], budget_fraction)
                ].sort_values("uid")
                if not np.array_equal(policy["uid"].to_numpy(dtype=np.int64), uids):
                    raise RuntimeError("P11.3 assignment/population mismatch")
                chosen = policy["action"].astype(str).to_numpy()
                ridge_benefit = np.asarray([benefits[action][index] for index, action in enumerate(chosen)])
                charged = float(next(
                    row["charged_cost_token_layers"] for row in json.loads((ROOT / artifact["result"]).read_text())["policies"]
                    if row["sample_fraction"] == sample_fraction and row["budget_fraction"] == budget_fraction
                ))
                baselines = []
                baseline_vectors = {}
                for action in ACTIONS:
                    cost = float(costs[action].sum())
                    if cost <= charged + 1e-9:
                        vector = benefits[action]
                        name = f"uniform_{action}"
                        baseline_vectors[name] = vector
                        baselines.append({"name": name, "spent_token_layers": cost,
                                          "recovery_fraction": float(vector.sum() / total_risk)})
                for name, order in metadata.items():
                    selected, spent = allocate_exact(order, exact_cost, charged)
                    vector = noop * selected
                    baseline_vectors[name] = vector
                    baselines.append({"name": name, "spent_token_layers": spent,
                                      "recovery_fraction": float(vector.sum() / total_risk)})
                strongest = max(baselines, key=lambda row: row["recovery_fraction"])
                delta = ridge_benefit - baseline_vectors[strongest["name"]]
                bootstrap = p10base.paired_bootstrap(
                    delta, f"recursive:{model}:{seed}:{sample_fraction}:{budget_fraction}:{strongest['name']}",
                    int(contract["uncertainty"]["paired_user_bootstrap_repetitions"]),
                )
                bootstrap["normalized_by_mean_noop_risk"] = bootstrap["mean"] / float(np.mean(noop))
                random_points = random_exact(
                    noop, exact_cost, charged, f"recursive:{model}:{seed}:{sample_fraction}:{budget_fraction}",
                    int(contract["same_charged_cost_baselines"]["random_exact_repetitions"]),
                )
                segments = [concave_segments(
                    [costs[action][index] for action in ACTIONS],
                    [benefits[action][index] for action in ACTIONS], ACTIONS,
                ) for index in range(len(uids))]
                oracle_spent, oracle_benefit, oracle_counts = near_optimal(segments, charged)
                rows.append({
                    "model": model, "seed": seed, "sample_fraction": sample_fraction,
                    "budget_fraction": budget_fraction, "charged_cost_fraction": charged / total_exact,
                    "Ridge_recovery_fraction": float(ridge_benefit.sum() / total_risk),
                    "Ridge_action_counts": {action: int(np.sum(chosen == action)) for action in ACTIONS},
                    "deterministic_baselines": baselines, "strongest_deterministic": strongest,
                    "Ridge_minus_strongest_paired_user_bootstrap": bootstrap,
                    "random_Exact_recovery_mean": float(np.mean(random_points) / total_risk),
                    "random_Exact_recovery_points": [float(value / total_risk) for value in random_points],
                    "offline_near_optimal_recovery_fraction": float(oracle_benefit / total_risk),
                    "offline_near_optimal_spent_fraction": float(oracle_spent / total_exact),
                    "offline_near_optimal_action_counts": oracle_counts,
                })
    model_gates = []
    primary = float(contract["primary_policy"]["probe_fraction"])
    for model in scheduler["scope"]["models"]:
        budgets = []
        for budget in map(float, contract["primary_policy"]["budgets_exact_fraction"]):
            points = sorted([
                row for row in rows if row["model"] == model and row["sample_fraction"] == primary
                and row["budget_fraction"] == budget
            ], key=lambda row: row["seed"])
            differences = [
                row["Ridge_recovery_fraction"] - row["strongest_deterministic"]["recovery_fraction"]
                for row in points
            ]
            budgets.append({"budget_fraction": budget, "seed_order": [row["seed"] for row in points],
                            "recovery_difference_seed_points": differences,
                            "equal_seed_mean_difference": float(np.mean(differences)),
                            "positive_seed_count": int(sum(value > 0 for value in differences)),
                            "positive_bootstrap_CI_seed_count": int(sum(
                                row["Ridge_minus_strongest_paired_user_bootstrap"]["p2_5"] > 0 for row in points
                            ))})
        passed = any(row["equal_seed_mean_difference"] > 0 and row["positive_seed_count"] >= 2 for row in budgets)
        model_gates.append({"model": model, "budgets": budgets, "passed": passed})
    payload = {
        "status": "P11_3_recursive_same_cost_scheduler_adjudicated",
        "contract_sha256": p7.sha256_file(CONTRACT), "assignment_seal_sha256": p7.sha256_file(SEAL),
        "rows": rows, "model_gates": model_gates, "quality_labels_read": False,
        "scheduler_retuned": False, "theta3_executed": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "model_gates": model_gates}, indent=2))


if __name__ == "__main__":
    main()
