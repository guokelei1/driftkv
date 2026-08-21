#!/usr/bin/env python3
"""Summarize a P9 diagnostic tomography raw artifact without quality labels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import train_p7_theta0 as p7

import adjudicate_p8_hs as hs

ROOT = Path(__file__).resolve().parents[1]


def user_means(rows: list[dict], key: str) -> np.ndarray:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[key])
        if np.isfinite(value):
            grouped[int(row["uid"])].append(value)
    return np.asarray([np.mean(grouped[uid]) for uid in sorted(grouped)], dtype=np.float64)


def summarize(values: np.ndarray) -> dict:
    if not len(values):
        return {
            "users": 0, "mean_equal_user": None, "P50_equal_user": None,
            "P90_equal_user": None, "P95_equal_user": None,
        }
    return {
        "users": int(len(values)), "mean_equal_user": float(np.mean(values)),
        "P50_equal_user": float(np.percentile(values, 50)), "P90_equal_user": float(np.percentile(values, 90)),
        "P95_equal_user": float(np.percentile(values, 95)),
    }


def summarize_raw(path: Path) -> dict:
    path = path.resolve()
    frame = pq.read_table(path).to_pandas()
    rows_by_action: dict[str, list[dict]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        full = np.asarray([row.full_logit], dtype=np.float64)
        reuse = np.asarray([row.reuse_logit], dtype=np.float64)
        diagnostic = np.asarray([row.diagnostic_logit], dtype=np.float64)
        stale = hs.js_divergence(reuse, full, "F")
        residual = hs.js_divergence(diagnostic, full, "F")
        rows_by_action[str(row.action)].append({
            "uid": int(row.uid), "stale_JS": stale, "residual_JS": residual,
            "absolute_recovery_JS": stale - residual,
            "relative_recovery": (stale - residual) / stale if stale > 1e-8 else float("nan"),
        })
    actions = {}
    for action, rows in sorted(rows_by_action.items()):
        actions[action] = {
            "requests": len(rows),
            "stale_JS": summarize(user_means(rows, "stale_JS")),
            "residual_JS": summarize(user_means(rows, "residual_JS")),
            "absolute_recovery_JS": summarize(user_means(rows, "absolute_recovery_JS")),
            "relative_recovery_on_S_above_floor": summarize(user_means(rows, "relative_recovery")),
        }
    identity = frame[["request_id", "full_logit", "reuse_logit"]].drop_duplicates()
    return {
        "status": "P9_2_diagnostic_tomography_summarized",
        "diagnostic_not_executable_action": True,
        "raw_path": str(path.relative_to(ROOT)), "raw_sha256": p7.sha256_file(path),
        "requests": int(identity.shape[0]), "actions": actions,
        "notes": [
            "Recovery is a target-free diagnostic exact-KV splice, not an executable partial migration result.",
            "relative recovery omits requests with stale JS at or below the frozen 1e-8 numeric floor.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.raw.exists():
        raise FileNotFoundError(args.raw)
    payload = summarize_raw(args.raw)
    if args.output is None:
        print(json.dumps(payload, indent=2))
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite tomography summary: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "requests": payload["requests"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
