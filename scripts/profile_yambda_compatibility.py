#!/usr/bin/env python3
"""Summarize label-free one-hop compatibility features from a chain screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_edge(name: str, edge: dict) -> dict:
    records = edge.get("compatibility_records", [])
    if not records:
        return {"edge": name, "status": "no_records"}
    label_free = ("score_rms", "score_max_abs", "js_divergence", "top10_overlap")
    summary = {"edge": name, "evaluated_users": len(records)}
    for key in label_free:
        values = np.asarray([float(row[key]) for row in records], dtype=np.float64)
        summary[key] = {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }
    summary["top10_changed_fraction"] = float(
        np.mean([float(row["top10_overlap"]) < 1.0 for row in records])
    )
    summary["js_nontrivial_fraction_gt_1e-6"] = float(
        np.mean([float(row["js_divergence"]) > 1e-6 for row in records])
    )
    summary["uses_future_labels_for_action_selection"] = False
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/data_audit/yambda50m_v2/two_edge_chain_screen.json"))
    parser.add_argument("--output", type=Path, default=Path("results/data_audit/yambda50m_v2/compatibility_profile_screen.json"))
    args = parser.parse_args()
    chain = json.loads(args.input.read_text())
    result = {
        "status": "label_free_compatibility_profile_screen",
        "source": str(args.input),
        "features": ["score_rms", "score_max_abs", "js_divergence", "top10_overlap"],
        "action_thresholds_fitted": False,
        "target_kv_mapping_fitted": False,
        "edges": [
            summarize_edge("theta0_to_theta1", chain["edge_theta0_theta1"]),
            summarize_edge("theta1_to_theta2", chain["edge_theta1_theta2"]),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
