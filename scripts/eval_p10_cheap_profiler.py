#!/usr/bin/env python3
"""Generate label-free P10 cheap-profiler state-action assignments."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import heapq
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import yaml

import train_p7_theta0 as p7
from adjudicate_p9_frontier import ACTIONS, concave_segments, token_layer_cost


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p10_0_cheap_profiler_contract_v1.yaml"
P9_7 = ROOT / "results/p9/p9_7_full_population_costs_v1.json"
P9_8 = ROOT / "results/p9/p9_8_cutover_profiler_v1.json"
POPULATION = ROOT / "data/manifests/p9_full_population_v1"
DEFAULT_OUTPUT = ROOT / "results/p10/p10_0_cheap_profiler_raw/full"
FEATURES = (
    "log1p_effective_prefix_length",
    "log1p_last_activity_age_seconds",
    "log1p_events_last_1d",
    "log1p_events_last_7d",
    "log1p_events_last_30d",
    "log1p_unique_items_last_7d",
    "organic_ratio_last_7d",
    "repeat_ratio_last_7d",
)
PROBE_ACTIONS = tuple(action for action in ACTIONS if action != "noop")


def validate_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p9_7_population_cost_sha256": P9_7,
        "p9_8_all_state_fidelity_sha256": P9_8,
        "edge1_state_manifest_sha256": POPULATION / "edge1/states.parquet",
        "edge2_state_manifest_sha256": POPULATION / "edge2/states.parquet",
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P10 input hash mismatch: {key}")
    return contract


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        np.log1p(frame["effective_prefix_length"].to_numpy(dtype=np.float64)),
        np.log1p(frame["last_activity_age_seconds"].to_numpy(dtype=np.float64)),
        np.log1p(frame["events_last_1d"].to_numpy(dtype=np.float64)),
        np.log1p(frame["events_last_7d"].to_numpy(dtype=np.float64)),
        np.log1p(frame["events_last_30d"].to_numpy(dtype=np.float64)),
        np.log1p(frame["unique_items_last_7d"].to_numpy(dtype=np.float64)),
        frame["organic_ratio_last_7d"].to_numpy(dtype=np.float64),
        frame["repeat_ratio_last_7d"].to_numpy(dtype=np.float64),
    ])


def deterministic_probe_order(release: str, model: str, seed: int, uids: np.ndarray) -> np.ndarray:
    keys = []
    for index, uid in enumerate(uids):
        raw = f"{release}:{model}:{seed}:{int(uid)}".encode()
        keys.append((hashlib.sha256(raw).digest(), int(uid), index))
    return np.asarray([row[2] for row in sorted(keys)], dtype=np.int64)


def allocate_predicted(segments: list[list[dict]], budget: float) -> tuple[float, list[str]]:
    heap: list[tuple[float, int, int]] = []
    for state, values in enumerate(segments):
        if values:
            heapq.heappush(heap, (-values[0]["slope"], state, 0))
    spent = 0.0
    selected = ["noop"] * len(segments)
    while heap:
        _, state, index = heapq.heappop(heap)
        segment = segments[state][index]
        if spent + segment["delta_cost"] <= budget + 1e-9:
            spent += segment["delta_cost"]
            selected[state] = segment["action"]
            if index + 1 < len(segments[state]):
                nxt = segments[state][index + 1]
                heapq.heappush(heap, (-nxt["slope"], state, index + 1))
    return spent, selected


def _cell_key(cell: dict) -> tuple[str, str, int]:
    return cell["release"], cell["model"], int(cell["seed"])


def evaluate_cell(arguments: tuple[dict, dict, str]) -> dict:
    cell, contract, output_root_text = arguments
    output_root = Path(output_root_text)
    release, model, seed = _cell_key(cell)
    edge = "edge2" if release == "r1_edge2" else "edge1"
    population = pq.read_table(POPULATION / edge / "states.parquet").to_pandas().sort_values("uid").reset_index(drop=True)
    state = pq.read_table(ROOT / cell["state_metrics_path"], columns=["uid", "action", "mse"]).to_pandas()
    pivot = state.pivot(index="uid", columns="action", values="mse").reindex(population["uid"].astype(int)).reset_index(drop=True)
    if pivot.isna().any().any():
        raise RuntimeError(f"state/population join failed: {release}:{model}:{seed}")
    uids = population["uid"].to_numpy(dtype=np.int64)
    lengths = population["effective_prefix_length"].to_numpy(dtype=np.int64)
    x = feature_matrix(population)
    noop = pivot["noop"].to_numpy(dtype=np.float64)
    total_risk = float(noop.sum())
    exact_cost = token_layer_cost("exact_all", lengths).astype(np.float64)
    total_exact_cost = float(exact_cost.sum())
    action_cost = {action: token_layer_cost(action, lengths).astype(np.float64) for action in ACTIONS}
    actual_benefit = {action: noop - pivot[action].to_numpy(dtype=np.float64) for action in ACTIONS}
    assignments = []
    policies = []
    probe_order = deterministic_probe_order(release, model, seed, uids)
    for sample_fraction in map(float, contract["probe"]["sample_fractions"]):
        sample_count = int(math.ceil(sample_fraction * len(uids))) if release != "r0" else 0
        sampled = np.zeros(len(uids), dtype=bool)
        sampled[probe_order[:sample_count]] = True
        predicted = {"noop": np.zeros(len(uids), dtype=np.float64)}
        coefficient_trace = {}
        if sample_count:
            scaler = StandardScaler().fit(x[sampled])
            transformed_sample = scaler.transform(x[sampled])
            transformed_all = scaler.transform(x)
            target_scale = max(float(np.mean(noop[sampled])), 1e-20)
            for action in PROBE_ACTIONS:
                estimator = Ridge(alpha=float(contract["predictor"]["alpha"]), solver=contract["predictor"]["solver"])
                estimator.fit(transformed_sample, actual_benefit[action][sampled] / target_scale)
                predicted[action] = estimator.predict(transformed_all) * target_scale
                coefficient_trace[action] = {
                    "intercept": float(estimator.intercept_),
                    "coefficients": [float(value) for value in estimator.coef_],
                }
            probe_cost = float(sum(action_cost[action][sampled].sum() for action in PROBE_ACTIONS))
        else:
            target_scale = None
            probe_cost = 0.0
            for action in PROBE_ACTIONS:
                predicted[action] = np.zeros(len(uids), dtype=np.float64)
        unsampled_indices = np.flatnonzero(~sampled)
        segments = []
        for index in unsampled_indices:
            segments.append(concave_segments(
                [action_cost[action][index] for action in ACTIONS],
                [predicted[action][index] for action in ACTIONS],
                ACTIONS,
            ))
        for budget_fraction in map(float, contract["allocation"]["budgets_exact_fraction"]):
            total_budget = budget_fraction * total_exact_cost
            remaining = max(total_budget - probe_cost, 0.0)
            _, unsampled_actions = allocate_predicted(segments, remaining)
            selected = np.full(len(uids), "noop", dtype=object)
            selected[sampled] = "exact_all"
            selected[unsampled_indices] = np.asarray(unsampled_actions, dtype=object)
            selected_cost = np.asarray([action_cost[action][index] for index, action in enumerate(selected)], dtype=np.float64)
            migration_cost = float(selected_cost.sum())
            total_charged = probe_cost + float(selected_cost[~sampled].sum())
            if total_charged > total_budget + 1e-6:
                raise RuntimeError(f"budget exceeded: {release}:{model}:{seed}:{sample_fraction}:{budget_fraction}")
            recovered = float(sum(actual_benefit[action][index] for index, action in enumerate(selected)))
            action_counts = {action: int(np.sum(selected == action)) for action in ACTIONS}
            policies.append({
                "sample_fraction": sample_fraction,
                "sample_count": sample_count,
                "budget_fraction": budget_fraction,
                "probe_cost_token_layers": probe_cost,
                "migration_cost_token_layers": migration_cost,
                "charged_cost_token_layers": total_charged,
                "charged_cost_fraction": total_charged / total_exact_cost,
                "unused_budget_fraction": (total_budget - total_charged) / total_exact_cost,
                "risk_recovery_fraction": recovered / total_risk if total_risk > 1e-20 else None,
                "residual_population_MSE": (total_risk - recovered) / len(uids),
                "action_counts": action_counts,
                "predictor_target_scale": target_scale,
                "predictor_coefficients": coefficient_trace,
            })
            for index, uid in enumerate(uids):
                action = str(selected[index])
                assignments.append({
                    "uid": int(uid),
                    "release": release,
                    "model": model,
                    "seed": seed,
                    "sample_fraction": sample_fraction,
                    "budget_fraction": budget_fraction,
                    "calibration_sample": bool(sampled[index]),
                    "action": action,
                    "predicted_benefit": float(predicted[action][index]),
                    "action_cost_token_layers": float(action_cost[action][index]),
                })
    cell_root = output_root / release / f"{model}_seed{seed}"
    cell_root.mkdir(parents=True, exist_ok=False)
    assignment_path = cell_root / "assignments.parquet"
    table = pa.Table.from_pylist(assignments)
    pq.write_table(table, assignment_path, compression="zstd")
    payload = {
        "status": "passed_target_free_assignments_unsealed",
        "release": release,
        "model": model,
        "seed": seed,
        "states": len(uids),
        "total_noop_risk": total_risk,
        "total_exact_token_layers": total_exact_cost,
        "features": list(FEATURES),
        "policies": policies,
        "assignments_path": str(assignment_path.relative_to(ROOT)),
        "assignments_sha256": p7.sha256_file(assignment_path),
        "assignment_columns": table.column_names,
        "quality_joined": False,
        "contract_sha256": p7.sha256_file(CONTRACT),
    }
    result_path = cell_root / "result.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    return {
        "release": release, "model": model, "seed": seed, "states": len(uids),
        "result": str(result_path.relative_to(ROOT)), "result_sha256": p7.sha256_file(result_path),
        "assignments": payload["assignments_path"], "assignments_sha256": payload["assignments_sha256"],
    }


def selected_cells(p8: dict, mode: str, contract: dict) -> list[dict]:
    if mode == "full":
        return list(p8["cells"])
    allowed = {(row["release"], row["model"], int(row["seed"])) for row in contract["canary"]["semantic_cells"]}
    return [cell for cell in p8["cells"] if _cell_key(cell) in allowed]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("canary", "full"), required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    contract = validate_contract()
    output_root = (args.output_root or (
        DEFAULT_OUTPUT if args.mode == "full" else ROOT / "results/p10/p10_0_cheap_profiler_raw/canary"
    )).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    cells = selected_cells(json.loads(P9_8.read_text()), args.mode, contract)
    work = [(cell, contract, str(output_root)) for cell in cells]
    workers = min(max(args.workers, 1), len(work))
    if workers == 1:
        completed = [evaluate_cell(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            completed = list(pool.map(evaluate_cell, work))
    summary = {
        "status": f"P10_0_{args.mode}_target_free_assignments_generated_unsealed",
        "mode": args.mode,
        "contract_sha256": p7.sha256_file(CONTRACT),
        "cells": sorted(completed, key=lambda row: (row["release"], row["model"], row["seed"])),
        "quality_joined": False,
    }
    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "cells": len(completed), "output": str(output_root)}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
