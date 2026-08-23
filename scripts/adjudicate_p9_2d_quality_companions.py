#!/usr/bin/env python3
"""Quality companions for every sealed P9.3 2-D diagnostic intervention."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import adjudicate_p9_quality_companions as quality
import eval_p9_2d_tomography_raw as evaluator
import run_p9_2d_tomography as ledger
import train_p7_theta0 as p7

ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / "results/p9/p9_3_2d_tomography_raw_seal_v1.json"
OUTPUT = ROOT / "results/p9/p9_3_2d_quality_companions_v1.json"


def build_cell(job: ledger.Job) -> dict:
    label_map, _, _ = quality.validate_and_map_labels(job)
    raw = job.output / "F_fidelity_2d_tomography.parquet"
    frame = pq.read_table(raw).to_pandas()
    actions = list(evaluator.action_names_2d(4))
    if set(frame["action"].unique()) != set(actions):
        raise RuntimeError(f"P9.3 action set mismatch: {job}")
    if not (frame.groupby("request_id", sort=False).size() == len(actions)).all():
        raise RuntimeError(f"P9.3 action row conservation failed: {job}")
    frame["label"] = frame["request_id"].astype(str).map(label_map)
    if frame["label"].isna().any():
        raise RuntimeError(f"P9.3 request missing sealed label: {job}")
    baseline = frame.drop_duplicates("request_id", keep="first")
    request_order = baseline["request_id"].astype(str).to_numpy()
    labels = baseline["label"].to_numpy(dtype=np.int64)
    uids = baseline["uid"].to_numpy(dtype=np.int64)
    full = quality.binary_metrics(labels, baseline["full_logit"].to_numpy(), uids)
    reuse = quality.binary_metrics(labels, baseline["reuse_logit"].to_numpy(), uids)
    full_gain = {metric: quality.quality_gain(reuse[metric], full[metric], metric) for metric in quality.METRICS}
    action_rows = {}
    for action in actions:
        selected = frame[frame["action"] == action]
        if not np.array_equal(selected["request_id"].astype(str).to_numpy(), request_order):
            raise RuntimeError(f"P9.3 request order differs: {job} {action}")
        values = quality.binary_metrics(labels, selected["diagnostic_logit"].to_numpy(), uids)
        action_rows[action] = {
            "absolute": values,
            "quality_gain_vs_reuse": {
                metric: quality.quality_gain(reuse[metric], values[metric], metric) for metric in quality.METRICS
            },
            "quality_gain_vs_current_full": {
                metric: quality.quality_gain(full[metric], values[metric], metric) for metric in quality.METRICS
            },
        }
    return {
        "release": job.release, "model": job.model, "seed": job.seed,
        "requests": len(baseline), "users": len(np.unique(uids)),
        "current_full": full, "reuse": reuse, "current_full_quality_gain_vs_reuse": full_gain,
        "actions": action_rows, "raw_sha256": p7.sha256_file(raw),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite P9.3 quality result: {args.output}")
    if not SEAL.exists():
        raise FileNotFoundError("seal P9.3 raw matrix first")
    if not 1 <= args.workers <= 24:
        raise ValueError("workers must be in [1,24]")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        cells = list(executor.map(build_cell, ledger.jobs()))
    aggregate = []
    for release, model in ledger.SEMANTIC_CELLS:
        rows = sorted([row for row in cells if row["release"] == release and row["model"] == model], key=lambda row: row["seed"])
        for action in evaluator.action_names_2d(4):
            metrics = {}
            for metric in quality.METRICS:
                points = [row["actions"][action]["quality_gain_vs_reuse"][metric] for row in rows]
                metrics[metric] = {
                    "seed_points": points, "equal_seed_mean": float(np.mean(points)),
                    "positive_seed_count": int(np.sum(np.asarray(points) > 0)),
                }
            aggregate.append({
                "release": release, "model": model, "action": action,
                "seed_order": [row["seed"] for row in rows], "quality_gain_vs_reuse": metrics,
            })
    payload = {
        "status": "P9_3_all_2d_diagnostic_quality_companions_complete",
        "raw_seal_hash": p7.sha256_file(SEAL),
        "diagnostic_not_executable_action": True,
        "labels_used_only_for_posthoc_quality_companions": True,
        "cells": cells, "aggregate": aggregate,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
