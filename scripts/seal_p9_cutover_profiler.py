#!/usr/bin/env python3
"""Seal all 24 P9.8 raw cutover-profiler cells before metrics."""

from __future__ import annotations

import json
from pathlib import Path

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_8_cutover_profiler_contract_v1.yaml"
RAW = ROOT / "results/p9/cutover_profiler_raw/full"
OUTPUT = ROOT / "results/p9/p9_8_cutover_profiler_raw_seal_v1.json"


def main() -> None:
    artifacts, errors = [], []
    for release in ("r0", "r1_edge1", "r1_edge2", "r2"):
        for model in ("m0_f", "m1"):
            for seed in (17, 37, 71):
                root = RAW / release / f"{model}_seed{seed}"
                manifest_path = root / "raw_manifest.json"
                if not manifest_path.exists():
                    errors.append(f"missing:{release}:{model}:{seed}")
                    continue
                manifest = json.loads(manifest_path.read_text())
                raw_path = ROOT / manifest["raw_path"]
                if manifest["status"] != "passed_raw_scores_unadjudicated":
                    errors.append(f"status:{release}:{model}:{seed}")
                if manifest["metrics_computed"]:
                    errors.append(f"metrics:{release}:{model}:{seed}")
                if manifest["contract_hash"] != p7.sha256_file(CONTRACT):
                    errors.append(f"contract:{release}:{model}:{seed}")
                if manifest["raw_sha256"] != p7.sha256_file(raw_path):
                    errors.append(f"hash:{release}:{model}:{seed}")
                artifacts.append({
                    "release": release, "model": model, "seed": seed,
                    "states": manifest["states"], "candidate_rows": manifest["candidate_rows"],
                    "manifest": str(manifest_path.relative_to(ROOT)),
                    "manifest_sha256": p7.sha256_file(manifest_path),
                    "raw": manifest["raw_path"], "raw_sha256": manifest["raw_sha256"],
                })
    if errors or len(artifacts) != 24:
        raise RuntimeError(f"P9.8 seal refused: {errors}; cells={len(artifacts)}")
    payload = {
        "status": "P9_8_all_24_full_population_cutover_raw_cells_sealed_before_metrics",
        "contract_hash": p7.sha256_file(CONTRACT),
        "cells": len(artifacts),
        "state_evaluations": sum(row["states"] for row in artifacts),
        "candidate_action_rows": sum(row["candidate_rows"] for row in artifacts),
        "artifacts": artifacts,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "status", "cells", "state_evaluations", "candidate_action_rows"
    )}, indent=2))


if __name__ == "__main__":
    main()
