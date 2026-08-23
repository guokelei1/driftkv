#!/usr/bin/env python3
"""Replay the frozen target-free Ridge scheduler and same-cost baselines at 8L."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import yaml

import train_p7_theta0 as p7

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/scale_8l_method_v1.yaml"
ACTIONS_RESULT = ROOT / "results/scale_8l_v1/actions_adjudication_v1.json"
POPULATION = ROOT / "data/manifests/scale_8l_population_v1"
OUTPUT = ROOT / "results/scale_8l_v1/scheduler"
ACTIONS = ("noop", "layer0_recent128", "layer0_middle", "layer0_full", "hybrid_tail128", "exact_all")
FEATURES = ("log1p_effective_prefix_length", "log1p_last_activity_age_seconds", "log1p_events_last_1d",
    "log1p_events_last_7d", "log1p_events_last_30d", "log1p_unique_items_last_7d", "organic_ratio_last_7d", "repeat_ratio_last_7d")


def costs(action: str, lengths: np.ndarray) -> np.ndarray:
    if action == "noop": return np.zeros_like(lengths)
    if action == "layer0_recent128": return np.minimum(lengths, 128)
    if action == "layer0_middle": return np.maximum(lengths // 4 + 1, (3 * lengths + 1) // 4) - lengths // 4
    if action == "layer0_full": return lengths
    if action == "hybrid_tail128": return 8 * np.minimum(lengths, 128)
    if action == "exact_all": return 8 * lengths
    raise ValueError(action)


def features(frame) -> np.ndarray:
    return np.column_stack([np.log1p(frame.effective_prefix_length), np.log1p(frame.last_activity_age_seconds),
        np.log1p(frame.events_last_1d), np.log1p(frame.events_last_7d), np.log1p(frame.events_last_30d),
        np.log1p(frame.unique_items_last_7d), frame.organic_ratio_last_7d, frame.repeat_ratio_last_7d]).astype(np.float64)


def probe_order(release: str, uids: np.ndarray) -> np.ndarray:
    return np.asarray(sorted(range(len(uids)), key=lambda i: (hashlib.sha256(f"{release}:m0_f:17:{int(uids[i])}".encode()).digest(), int(uids[i]))), dtype=np.int64)


def hull_segments(action_cost, benefit, index):
    best_by_cost = {}
    for action in ACTIONS:
        cost, value = float(action_cost[action][index]), float(benefit[action][index])
        if cost not in best_by_cost or value > best_by_cost[cost][0]:
            best_by_cost[cost] = (value, action)
    points = sorted(best_by_cost.items())
    monotone = []; best = -np.inf
    for cost, (value, action) in points:
        if value > best + 1e-20: monotone.append((cost, value, action)); best = value
    hull = []
    for point in monotone:
        while len(hull) >= 2 and ((hull[-1][1]-hull[-2][1])/(hull[-1][0]-hull[-2][0]) <= (point[1]-hull[-1][1])/(point[0]-hull[-1][0])): hull.pop()
        hull.append(point)
    return [{"delta_cost": hull[i][0]-hull[i-1][0], "slope": (hull[i][1]-hull[i-1][1])/(hull[i][0]-hull[i-1][0]), "action": hull[i][2]}
        for i in range(1, len(hull)) if hull[i][1] > hull[i-1][1]]


def allocate(segments, budget):
    heap = []; selected = ["noop"] * len(segments); spent = 0.0
    for i, rows in enumerate(segments):
        if rows: heapq.heappush(heap, (-rows[0]["slope"], i, 0))
    while heap:
        _, state, position = heapq.heappop(heap); segment = segments[state][position]
        if spent + segment["delta_cost"] <= budget + 1e-9:
            spent += segment["delta_cost"]; selected[state] = segment["action"]
            if position + 1 < len(segments[state]): heapq.heappush(heap, (-segments[state][position+1]["slope"], state, position+1))
    return spent, np.asarray(selected, dtype=object)


def exact_by_order(order, exact_cost, budget):
    selected = np.zeros(len(order), dtype=bool); spent = 0.0
    for index in order:
        if spent + exact_cost[index] <= budget + 1e-9: selected[index] = True; spent += exact_cost[index]
    return selected


def main() -> None:
    if OUTPUT.exists(): raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    contract = yaml.safe_load(CONTRACT.read_text()); source = json.loads(ACTIONS_RESULT.read_text())
    OUTPUT.mkdir(parents=True); all_results = []
    modes = (("rate_1pct", .01), ("rate_2pct", .02), ("fixed_count_64", 64), ("fixed_count_128", 128),
        ("fixed_count_256", 256), ("capped_rate_min_1pct_128", "cap128"))
    for cell in source["cells"]:
        release = cell["release"]; edge = "edge2" if release == "r1_edge2" else "edge1"
        population = pq.read_table(POPULATION / edge / "states.parquet").to_pandas().sort_values("uid").reset_index(drop=True)
        state = pq.read_table(ROOT / cell["state_metrics_path"], columns=["uid", "action", "mse"]).to_pandas()
        pivot = state.pivot(index="uid", columns="action", values="mse").reindex(population.uid.astype(int))
        if pivot.isna().any().any(): raise RuntimeError(f"population join failed: {release}")
        uids = population.uid.to_numpy(dtype=np.int64); lengths = population.effective_prefix_length.to_numpy(dtype=np.int64)
        x = features(population); noop = pivot.noop.to_numpy(dtype=np.float64); total_risk = float(noop.sum())
        action_cost = {a: costs(a, lengths).astype(np.float64) for a in ACTIONS}
        benefit = {a: noop - pivot[a].to_numpy(dtype=np.float64) for a in ACTIONS}; exact_cost = action_cost["exact_all"]
        total_exact = float(exact_cost.sum()); order = probe_order(release, uids); policies = []; assignments = []
        if release == "r0":
            all_results.append({"release": release, "states": len(uids), "metadata_gate_action": "noop", "policies": []}); continue
        for mode, value in modes:
            count = int(math.ceil(value * len(uids))) if isinstance(value, float) else (min(int(math.ceil(.01 * len(uids))), 128) if value == "cap128" else min(int(value), len(uids)))
            sampled = np.zeros(len(uids), dtype=bool); sampled[order[:count]] = True
            scaler = StandardScaler().fit(x[sampled]); xs, xa = scaler.transform(x[sampled]), scaler.transform(x)
            target_scale = max(float(noop[sampled].mean()), 1e-20); predicted = {"noop": np.zeros(len(uids))}
            for action in ACTIONS[1:]:
                model = Ridge(alpha=1.0, solver="auto").fit(xs, benefit[action][sampled] / target_scale)
                predicted[action] = model.predict(xa) * target_scale
            probe_cost = float(sum(action_cost[a][sampled].sum() for a in ACTIONS[1:])); unsampled = np.flatnonzero(~sampled)
            segments = [hull_segments(action_cost, predicted, i) for i in unsampled]
            for budget_fraction in contract["scheduler"]["budgets_exact_fraction"]:
                total_budget = float(budget_fraction) * total_exact; _, chosen = allocate(segments, max(0., total_budget-probe_cost))
                selected = np.full(len(uids), "noop", dtype=object); selected[sampled] = "exact_all"; selected[unsampled] = chosen
                charged = probe_cost + float(sum(action_cost[str(selected[i])][i] for i in unsampled))
                recovered_vector = np.asarray([benefit[str(selected[i])][i] for i in range(len(uids))])
                uniform = []
                for action in ACTIONS:
                    total_cost = float(action_cost[action].sum())
                    if total_cost <= total_budget + 1e-9: uniform.append((float(benefit[action].sum()), action, total_cost))
                best_uniform = max(uniform)
                orders = {
                    "longest_prefix_exact_first": np.lexsort((uids, -lengths)),
                    "oldest_state_exact_first": np.lexsort((uids, -population.last_activity_age_seconds.to_numpy())),
                    "most_active_30d_exact_first": np.lexsort((uids, -population.events_last_30d.to_numpy())),
                    "most_unique_items_7d_exact_first": np.lexsort((uids, -population.unique_items_last_7d.to_numpy())),
                    "deterministic_random_exact": order,
                }
                baselines = [{"name": "best_uniform_action", "action": best_uniform[1], "recovery_fraction": best_uniform[0]/total_risk}]
                for name, ordering in orders.items():
                    mask = exact_by_order(ordering, exact_cost, total_budget); baselines.append({"name": name, "action": "selected_exact", "recovery_fraction": float(noop[mask].sum()/total_risk)})
                oracle_segments = [hull_segments(action_cost, benefit, i) for i in range(len(uids))]
                _, oracle_action = allocate(oracle_segments, total_budget)
                oracle_recovery = float(sum(benefit[str(oracle_action[i])][i] for i in range(len(uids))) / total_risk)
                strongest = max(baselines, key=lambda row: row["recovery_fraction"])
                policy = {"probe_mode": mode, "probe_count": count, "budget_fraction": float(budget_fraction),
                    "probe_cost_fraction": probe_cost/total_exact, "charged_cost_fraction": charged/total_exact,
                    "risk_recovery_fraction": float(recovered_vector.sum()/total_risk), "strongest_nonlearning_baseline": strongest,
                    "ridge_minus_strongest_recovery": float(recovered_vector.sum()/total_risk - strongest["recovery_fraction"]),
                    "offline_oracle_recovery_fraction": oracle_recovery,
                    "action_counts": {a: int(np.sum(selected == a)) for a in ACTIONS}}
                policies.append(policy)
                if mode == "rate_1pct":
                    for i, uid in enumerate(uids): assignments.append({"uid": int(uid), "release": release,
                        "budget_fraction": float(budget_fraction), "calibration_sample": bool(sampled[i]), "action": str(selected[i])})
        root = OUTPUT / release; root.mkdir()
        path = root / "primary_assignments.parquet"; pq.write_table(pa.Table.from_pylist(assignments), path, compression="zstd")
        all_results.append({"release": release, "states": len(uids), "total_noop_risk": total_risk,
            "total_exact_token_layers": total_exact, "policies": policies,
            "primary_assignments": str(path.relative_to(ROOT)), "primary_assignments_sha256": p7.sha256_file(path)})
    payload = {"status": "scale_8l_frozen_scheduler_target_free_assignments_written", "contract_sha256": p7.sha256_file(CONTRACT),
        "action_result_sha256": p7.sha256_file(ACTIONS_RESULT), "features": list(FEATURES), "cells": all_results,
        "quality_labels_used": False, "assignments_sealed_for_quality_validation": True}
    path = OUTPUT / "scheduler_result.json"; path.write_text(json.dumps(payload, indent=2) + "\n")
    (OUTPUT / "assignment_seal.json").write_text(json.dumps({"status": "sealed_before_quality",
        "scheduler_result": str(path.relative_to(ROOT)), "scheduler_result_sha256": p7.sha256_file(path),
        "artifacts": [{"release": row["release"], "path": row.get("primary_assignments"), "sha256": row.get("primary_assignments_sha256")} for row in all_results]}, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "primary": [{"release": row["release"],
        "budgets": [{k: p[k] for k in ("budget_fraction", "risk_recovery_fraction", "ridge_minus_strongest_recovery")} for p in row["policies"] if p["probe_mode"] == "rate_1pct"]} for row in all_results]}, indent=2))


if __name__ == "__main__": main()
