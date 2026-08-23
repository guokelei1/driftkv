#!/usr/bin/env python3
"""Compute P9.8 all-state target-free metrics after the raw seal."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / "results/p9/p9_8_cutover_profiler_raw_seal_v1.json"
STATE_ROOT = ROOT / "results/p9/cutover_profiler_state_metrics"
OUTPUT = ROOT / "results/p9/p9_8_cutover_profiler_v1.json"
ACTIONS = ("noop", "layer0_recent128", "layer0_middle", "layer0_full", "hybrid_tail128", "exact_all")


def quantiles(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)), "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)), "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)), "max": float(np.max(values)),
    }


def cell_metrics(artifact: dict) -> dict:
    raw_path = ROOT / artifact["raw"]
    if p7.sha256_file(raw_path) != artifact["raw_sha256"]:
        raise RuntimeError("P9.8 raw changed after seal")
    frame = pq.read_table(raw_path).to_pandas()
    action_data = {}
    pair_i, pair_j = np.triu_indices(16, k=1)
    state_rows = []
    reference_uids = None
    for action in ACTIONS:
        selected = frame[frame["action"] == action].sort_values(["uid", "candidate_position"])
        uids = selected["uid"].to_numpy()[::16].astype(np.int64)
        if len(selected) != len(uids) * 16:
            raise RuntimeError("P9.8 candidate rows do not reshape to 16")
        if reference_uids is None:
            reference_uids = uids
        elif not np.array_equal(reference_uids, uids):
            raise RuntimeError("P9.8 actions have different state populations")
        current = selected["current_logit"].to_numpy(dtype=np.float64).reshape(-1, 16)
        value = selected["action_logit"].to_numpy(dtype=np.float64).reshape(-1, 16)
        delta = value - current
        mse = np.mean(delta * delta, axis=1)
        current_rms = np.sqrt(np.mean(current * current, axis=1))
        normalized_rms = np.sqrt(mse) / (current_rms + 1e-8)
        current_pair = current[:, pair_i] - current[:, pair_j]
        action_pair = value[:, pair_i] - value[:, pair_j]
        valid = current_pair != 0
        inversion = np.sum((current_pair * action_pair < 0) & valid, axis=1) / np.maximum(1, np.sum(valid, axis=1))
        p95_shift = np.quantile(np.abs(delta), 0.95, axis=1)
        action_data[action] = {
            "uids": uids, "mse": mse, "normalized_rms": normalized_rms,
            "pairwise_inversion": inversion, "p95_abs_shift": p95_shift,
        }
    noop = action_data["noop"]["mse"]
    summaries = []
    for action in ACTIONS:
        data = action_data[action]
        recovery = np.full_like(noop, np.nan, dtype=np.float64)
        valid_noop = noop > 1e-20
        recovery[valid_noop] = (noop[valid_noop] - data["mse"][valid_noop]) / noop[valid_noop]
        for index, uid in enumerate(data["uids"]):
            state_rows.append({
                "uid": int(uid), "action": action,
                "mse": float(data["mse"][index]),
                "normalized_rms": float(data["normalized_rms"][index]),
                "pairwise_inversion": float(data["pairwise_inversion"][index]),
                "p95_abs_shift": float(data["p95_abs_shift"][index]),
                "recovery_fraction_vs_noop": None if not np.isfinite(recovery[index]) else float(recovery[index]),
            })
        summaries.append({
            "action": action,
            "mse": quantiles(data["mse"]),
            "normalized_rms": quantiles(data["normalized_rms"]),
            "pairwise_inversion": quantiles(data["pairwise_inversion"]),
            "p95_abs_shift": quantiles(data["p95_abs_shift"]),
            "recovery_from_population_mean_MSE": (
                float((np.mean(noop) - np.mean(data["mse"])) / np.mean(noop))
                if np.mean(noop) > 1e-20 else None
            ),
            "positive_recovery_state_fraction": (
                float(np.mean(recovery > 0)) if np.isfinite(recovery).any() else None
            ),
        })
    state_path = STATE_ROOT / artifact["release"] / f"{artifact['model']}_seed{artifact['seed']}.parquet"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(state_rows), state_path, compression="zstd")
    return {
        "release": artifact["release"], "model": artifact["model"], "seed": artifact["seed"],
        "states": len(reference_uids), "summaries": summaries,
        "state_metrics_path": str(state_path.relative_to(ROOT)),
        "state_metrics_sha256": p7.sha256_file(state_path),
    }


def main() -> None:
    seal = json.loads(SEAL.read_text())
    if seal["status"] != "P9_8_all_24_full_population_cutover_raw_cells_sealed_before_metrics":
        raise RuntimeError("P9.8 raw is not sealed")
    if STATE_ROOT.exists() or OUTPUT.exists():
        raise FileExistsError("refusing to overwrite P9.8 adjudication")
    cells = [cell_metrics(artifact) for artifact in seal["artifacts"]]
    aggregates = []
    for release in ("r0", "r1_edge1", "r1_edge2", "r2"):
        for model in ("m0_f", "m1"):
            group = sorted(
                [cell for cell in cells if cell["release"] == release and cell["model"] == model],
                key=lambda row: row["seed"],
            )
            for action in ACTIONS:
                rows = [next(x for x in cell["summaries"] if x["action"] == action) for cell in group]
                noop_rows = [next(x for x in cell["summaries"] if x["action"] == "noop") for cell in group]
                action_equal_seed_mse = float(np.mean([row["mse"]["mean"] for row in rows]))
                noop_equal_seed_mse = float(np.mean([row["mse"]["mean"] for row in noop_rows]))
                aggregates.append({
                    "release": release, "model": model, "action": action,
                    "seed_order": [cell["seed"] for cell in group],
                    "population_mean_MSE_seed_points": [row["mse"]["mean"] for row in rows],
                    "population_mean_MSE_equal_seed_mean": action_equal_seed_mse,
                    "recovery_from_equal_seed_population_mean_MSE": (
                        (noop_equal_seed_mse - action_equal_seed_mse) / noop_equal_seed_mse
                        if noop_equal_seed_mse > 1e-20 else None
                    ),
                    "recovery_seed_points": [row["recovery_from_population_mean_MSE"] for row in rows],
                    "positive_recovery_state_fraction_seed_points": [row["positive_recovery_state_fraction"] for row in rows],
                })
    payload = {
        "status": "P9_8_full_population_cutover_profiler_adjudicated",
        "raw_seal_sha256": p7.sha256_file(SEAL),
        "cells": cells, "aggregates": aggregates,
        "all_state_exact_role": "offline_oracle_only",
        "scheduler_authorized": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells)}, indent=2))


if __name__ == "__main__":
    main()
