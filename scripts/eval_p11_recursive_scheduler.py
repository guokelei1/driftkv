#!/usr/bin/env python3
"""Generate frozen P10 scheduler assignments for P11 recursive lineage."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import yaml

import eval_p10_cheap_profiler as p10
import train_p7_theta0 as p7
from adjudicate_p9_frontier import ACTIONS, concave_segments, token_layer_cost


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p11_2_recursive_scheduler_replay_v1.yaml"
RAW = ROOT / "results/p11/p11_1_recursive_population_raw/full"
POPULATION = ROOT / "data/manifests/p9_full_population_v1/edge2/states.parquet"
OUTPUT_ROOT = ROOT / "results/p11/p11_2_recursive_scheduler_raw"


def validate():
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p10_scheduler_freeze_sha256": ROOT / "configs/contracts/p10_4_minimal_scheduler_freeze_v1.yaml",
        "p10_profiler_contract_sha256": ROOT / "configs/contracts/p10_0_cheap_profiler_contract_v1.yaml",
        "p11_1_contract_sha256": ROOT / "configs/contracts/p11_1_recursive_population_contract_v1.yaml",
        "p11_1_adjudication_sha256": ROOT / "results/p11/p11_1_recursive_population_adjudication_v1.json",
        "edge2_state_manifest_sha256": POPULATION,
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P11.2 input mismatch: {key}")
    return contract


def probe_order(release_id, model, seed, uids):
    keyed = []
    for index, uid in enumerate(uids):
        digest = hashlib.sha256(f"{release_id}:{model}:{seed}:{int(uid)}".encode()).digest()
        keyed.append((digest, int(uid), index))
    return np.asarray([row[2] for row in sorted(keyed)], dtype=np.int64)


def evaluate_cell(args):
    model, seed, contract, output_text, limit = args
    output = Path(output_text)
    raw_path = RAW / f"{model}_seed{seed}/state_metrics.parquet"
    metrics = pq.read_table(raw_path, columns=["uid", "action", "mse"]).to_pandas()
    mapping = contract["action_mapping"]
    pivot = metrics.pivot(index="uid", columns="action", values="mse")
    population = pq.read_table(POPULATION).to_pandas()
    population = population[population["uid"].astype(int).isin(pivot.index.astype(int))]
    if limit is not None:
        allowed = sorted(
            population["uid"].astype(int), key=lambda uid: hashlib.sha256(str(uid).encode()).digest()
        )[:limit]
        population = population[population["uid"].astype(int).isin(allowed)]
    population = population.sort_values("uid").reset_index(drop=True)
    pivot = pivot.reindex(population["uid"].astype(int))
    if len(population) != (limit or int(contract["scope"]["states"])) or pivot.isna().any().any():
        raise RuntimeError(f"P11.2 population join failed: {model} seed{seed}")
    uids = population["uid"].to_numpy(dtype=np.int64)
    lengths = population["effective_prefix_length"].to_numpy(dtype=np.int64)
    x = p10.feature_matrix(population)
    loss = {action: pivot[source].to_numpy(dtype=np.float64) for action, source in mapping.items()}
    noop = loss["noop"]
    action_cost = {action: token_layer_cost(action, lengths).astype(np.float64) for action in ACTIONS}
    total_exact = float(action_cost["exact_all"].sum())
    actual_benefit = {action: noop - loss[action] for action in ACTIONS}
    order = probe_order(contract["scope"]["release_id"], model, seed, uids)
    assignments, policies = [], []
    for sample_fraction in (
        float(contract["frozen_scheduler"]["primary_probe_rate"]),
        float(contract["frozen_scheduler"]["companion_probe_rate"]),
    ):
        count = int(math.ceil(sample_fraction * len(uids)))
        sampled = np.zeros(len(uids), dtype=bool)
        sampled[order[:count]] = True
        scaler = StandardScaler().fit(x[sampled])
        sample_x, all_x = scaler.transform(x[sampled]), scaler.transform(x)
        scale = max(float(np.mean(noop[sampled])), 1e-20)
        predicted = {"noop": np.zeros(len(uids), dtype=np.float64)}
        trace = {}
        for action in ACTIONS[1:]:
            estimator = Ridge(alpha=1.0, solver="lsqr")
            estimator.fit(sample_x, actual_benefit[action][sampled] / scale)
            predicted[action] = estimator.predict(all_x) * scale
            trace[action] = {"intercept": float(estimator.intercept_),
                             "coefficients": [float(value) for value in estimator.coef_]}
        probe_cost = float(sum(action_cost[action][sampled].sum() for action in ACTIONS[1:]))
        unsampled = np.flatnonzero(~sampled)
        segments = [concave_segments(
            [action_cost[action][index] for action in ACTIONS],
            [predicted[action][index] for action in ACTIONS], ACTIONS,
        ) for index in unsampled]
        for budget_fraction in map(float, contract["frozen_scheduler"]["budgets_exact_fraction"]):
            budget = budget_fraction * total_exact
            _, chosen = p10.allocate_predicted(segments, max(0.0, budget - probe_cost))
            selected = np.full(len(uids), "noop", dtype=object)
            selected[sampled] = "exact_all"
            selected[unsampled] = np.asarray(chosen, dtype=object)
            selected_cost = np.asarray([action_cost[action][index] for index, action in enumerate(selected)])
            charged = probe_cost + float(selected_cost[~sampled].sum())
            if charged > budget + 1e-6 or not np.all(selected[sampled] == "exact_all"):
                raise RuntimeError("P11.2 budget or sampled-terminal-action gate failed")
            policies.append({
                "sample_fraction": sample_fraction, "sample_count": count,
                "budget_fraction": budget_fraction, "probe_cost_token_layers": probe_cost,
                "charged_cost_token_layers": charged, "charged_cost_fraction": charged / total_exact,
                "action_counts": {action: int(np.sum(selected == action)) for action in ACTIONS},
                "predictor_target_scale": scale, "predictor_coefficients": trace,
            })
            for index, uid in enumerate(uids):
                action = str(selected[index])
                assignments.append({
                    "uid": int(uid), "release": contract["scope"]["release_id"],
                    "model": model, "seed": seed, "sample_fraction": sample_fraction,
                    "budget_fraction": budget_fraction, "calibration_sample": bool(sampled[index]),
                    "action": action, "predicted_benefit": float(predicted[action][index]),
                    "action_cost_token_layers": float(action_cost[action][index]),
                })
    cell = output / f"{model}_seed{seed}"
    cell.mkdir(parents=True, exist_ok=False)
    assignment_path = cell / "assignments.parquet"
    table = pa.Table.from_pylist(assignments)
    pq.write_table(table, assignment_path, compression="zstd")
    payload = {
        "status": "passed_recursive_assignments_unsealed", "model": model, "seed": seed,
        "states": len(uids), "policies": policies,
        "assignments": str(assignment_path.relative_to(ROOT)),
        "assignments_sha256": p7.sha256_file(assignment_path), "assignment_columns": table.column_names,
        "contract_sha256": p7.sha256_file(CONTRACT), "quality_joined": False,
        "unsampled_actual_loss_exported": False,
    }
    result_path = cell / "result.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    return {"model": model, "seed": seed, "states": len(uids),
            "result": str(result_path.relative_to(ROOT)), "result_sha256": p7.sha256_file(result_path),
            "assignments": payload["assignments"], "assignments_sha256": payload["assignments_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("canary", "full"), required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    contract = validate()
    limit = int(contract["canary"]["states"]) if args.mode == "canary" else None
    cells = [("m1", 17)] if args.mode == "canary" else [
        (model, seed) for model in contract["scope"]["models"] for seed in contract["scope"]["seeds"]
    ]
    output = (args.output_root or OUTPUT_ROOT / args.mode).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    work = [(model, int(seed), contract, str(output), limit) for model, seed in cells]
    if args.workers == 1 or len(work) == 1:
        completed = [evaluate_cell(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(work))) as pool:
            completed = list(pool.map(evaluate_cell, work))
    summary = {"status": f"P11_2_{args.mode}_assignments_generated_unsealed",
               "contract_sha256": p7.sha256_file(CONTRACT), "cells": completed, "quality_joined": False}
    path = output / "run_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "cells": len(completed), "output": str(output)}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
