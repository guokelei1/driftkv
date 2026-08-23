#!/usr/bin/env python3
"""Compute all-state fidelity and recovery only after the 8L raw seal."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import train_p7_theta0 as p7

ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / "results/scale_8l_v1/actions_raw_seal_v1.json"
STATE_ROOT = ROOT / "results/scale_8l_v1/action_state_metrics"
OUTPUT = ROOT / "results/scale_8l_v1/actions_adjudication_v1.json"
ACTIONS = ("noop", "layer0_recent128", "layer0_middle", "layer0_full", "hybrid_tail128", "exact_all")


def summary(values: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "p50": float(np.quantile(values, .5)),
        "p90": float(np.quantile(values, .9)), "p95": float(np.quantile(values, .95)),
        "p99": float(np.quantile(values, .99)), "max": float(values.max())}


def cell(artifact: dict) -> dict:
    raw = ROOT / artifact["raw"]
    if p7.sha256_file(raw) != artifact["raw_sha256"]: raise RuntimeError("sealed raw changed")
    frame = pq.read_table(raw).to_pandas(); state_rows = []; values = {}; reference = None
    pair_i, pair_j = np.triu_indices(16, k=1)
    for action in ACTIONS:
        selected = frame[frame.action == action].sort_values(["uid", "candidate_position"])
        uids = selected.uid.to_numpy(dtype=np.int64)[::16]
        if len(selected) != len(uids) * 16: raise RuntimeError("candidate panel does not reshape to 16")
        if reference is None: reference = uids
        elif not np.array_equal(reference, uids): raise RuntimeError("action populations differ")
        current = selected.current_logit.to_numpy(dtype=np.float64).reshape(-1, 16)
        action_score = selected.action_logit.to_numpy(dtype=np.float64).reshape(-1, 16)
        delta = action_score - current; mse = np.mean(delta * delta, axis=1)
        current_rms = np.sqrt(np.mean(current * current, axis=1))
        inversion = np.mean(((current[:, pair_i] - current[:, pair_j]) *
            (action_score[:, pair_i] - action_score[:, pair_j])) < 0, axis=1)
        values[action] = {"mse": mse, "normalized_rms": np.sqrt(mse) / (current_rms + 1e-8),
            "pairwise_inversion": inversion, "p95_abs_shift": np.quantile(np.abs(delta), .95, axis=1)}
    noop = values["noop"]["mse"]; summaries = []
    for action in ACTIONS:
        recovery = np.where(noop > 1e-20, (noop - values[action]["mse"]) / noop, np.nan)
        for index, uid in enumerate(reference):
            state_rows.append({"uid": int(uid), "action": action,
                "mse": float(values[action]["mse"][index]),
                "normalized_rms": float(values[action]["normalized_rms"][index]),
                "pairwise_inversion": float(values[action]["pairwise_inversion"][index]),
                "p95_abs_shift": float(values[action]["p95_abs_shift"][index]),
                "recovery_fraction_vs_noop": None if not np.isfinite(recovery[index]) else float(recovery[index])})
        summaries.append({"action": action, "mse": summary(values[action]["mse"]),
            "normalized_rms": summary(values[action]["normalized_rms"]),
            "pairwise_inversion": summary(values[action]["pairwise_inversion"]),
            "p95_abs_shift": summary(values[action]["p95_abs_shift"]),
            "recovery_from_population_mean_MSE": ((float(noop.mean()) - float(values[action]["mse"].mean())) / float(noop.mean())) if noop.mean() > 1e-20 else None,
            "positive_recovery_state_fraction": float(np.mean(recovery > 0)) if np.isfinite(recovery).any() else None})
    path = STATE_ROOT / artifact["release"] / "m0_f_seed17.parquet"; path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(state_rows), path, compression="zstd")
    manifest = json.loads((ROOT / artifact["manifest"]).read_text())
    return {"release": artifact["release"], "model": "m0_f", "seed": 17, "states": len(reference),
        "summaries": summaries, "state_metrics_path": str(path.relative_to(ROOT)),
        "state_metrics_sha256": p7.sha256_file(path),
        "measured_grouped_transition_runtime_seconds": manifest["transition_runtime_seconds"],
        "logical_work": manifest["logical_work"], "full_cell_wall_seconds": manifest["wall_seconds"]}


def main() -> None:
    if OUTPUT.exists() or STATE_ROOT.exists(): raise FileExistsError("refusing to overwrite adjudication")
    seal = json.loads(SEAL.read_text())
    if seal["status"] != "scale_8l_all_action_raw_cells_sealed_before_metrics": raise RuntimeError("raw not sealed")
    cells = [cell(row) for row in seal["artifacts"]]
    payload = {"status": "scale_8l_frozen_actions_adjudicated", "raw_seal_sha256": p7.sha256_file(SEAL),
        "cells": cells, "scheduler_authorized": True, "paper_qualification": False}
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": [{"release": c["release"],
        "recovery": {s["action"]: s["recovery_from_population_mean_MSE"] for s in c["summaries"]}} for c in cells]}, indent=2))


if __name__ == "__main__": main()
