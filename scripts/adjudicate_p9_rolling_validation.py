#!/usr/bin/env python3
"""Aggregate the sealed P9.5 rolling-lineage validation cells."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / "results/p9/p9_5_rolling_validation_raw_seal_v1.json"
OUTPUT = ROOT / "results/p9/p9_5_rolling_validation_v1.json"


def main() -> None:
    seal = json.loads(SEAL.read_text())
    if seal["status"] != "P9_5_all_24_materialized_rolling_lineage_cells_sealed":
        raise RuntimeError("P9.5 raw matrix is not sealed")
    cells = []
    for artifact in seal["artifacts"]:
        path = ROOT / artifact["result"]
        if p7.sha256_file(path) != artifact["result_sha256"]:
            raise RuntimeError("P9.5 result changed after seal")
        cells.append(json.loads(path.read_text()))
    aggregates = []
    for release in ("r0", "r1_edge1", "r1_edge2", "r2"):
        for model in ("m0_f", "m1"):
            group = [cell for cell in cells if cell["release"] == release and cell["model"] == model]
            for view in ("fidelity", "quality"):
                noop_points = []
                for cell in sorted(group, key=lambda row: row["seed"]):
                    noop_points.append(next(
                        row for row in cell["summaries"]
                        if row["view"] == view and row["action"] == "noop"
                    )["online_JS_mean"])
                noop_equal_seed_mean = float(np.mean(noop_points))
                for action in (
                    "noop", "layer0_recent128", "layer0_middle", "layer0_full",
                    "hybrid_tail128", "exact_all",
                ):
                    points = []
                    for cell in sorted(group, key=lambda row: row["seed"]):
                        summary = next(
                            row for row in cell["summaries"]
                            if row["view"] == view and row["action"] == action
                        )
                        points.append(summary)
                    aggregate = {
                        "release": release, "model": model, "view": view, "action": action,
                        "seed_order": [row["seed"] for row in sorted(group, key=lambda row: row["seed"])],
                        "online_JS_mean_seed_points": [row["online_JS_mean"] for row in points],
                        "online_JS_equal_seed_mean": float(np.mean([row["online_JS_mean"] for row in points])),
                        "recovery_fraction_seed_points": [row["signed_JS_recovery_fraction_vs_noop"] for row in points],
                        "current_online_vs_request_local_full_JS_seed_points": [
                            row["current_online_vs_request_local_full_JS_mean"] for row in points
                        ],
                    }
                    finite = [value for value in aggregate["recovery_fraction_seed_points"] if value is not None]
                    aggregate["recovery_fraction_equal_seed_mean"] = float(np.mean(finite)) if finite else None
                    aggregate["recovery_from_equal_seed_JS_means"] = (
                        (noop_equal_seed_mean - aggregate["online_JS_equal_seed_mean"])
                        / noop_equal_seed_mean
                        if noop_equal_seed_mean > 1e-15 else None
                    )
                    if view == "quality":
                        aggregate["action_minus_current_logloss_seed_points"] = [
                            row["action_minus_current_logloss"] for row in points
                        ]
                    aggregates.append(aggregate)
    payload = {
        "status": "P9_5_expanded_rolling_lineage_validation_adjudicated",
        "raw_seal": str(SEAL.relative_to(ROOT)),
        "raw_seal_sha256": p7.sha256_file(SEAL),
        "cells": len(cells),
        "aggregates": aggregates,
        "evidence_level": "expanded_development_validation_not_full_population",
        "scheduler_authorized": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": payload["cells"]}, indent=2))


if __name__ == "__main__":
    main()
