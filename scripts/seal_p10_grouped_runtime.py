#!/usr/bin/env python3
"""Seal the ten semantics-preserving grouped-runtime replays."""

from __future__ import annotations

import json
from pathlib import Path

import train_p7_theta0 as p7
import run_p10_mixed_policy_runtime_queue as queue


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/p10/mixed_policy_runtime/grouped"
CONTRACT = ROOT / "configs/contracts/p10_5_executor_batching_optimization_contract_v1.yaml"
OUTPUT = ROOT / "results/p10/p10_5_grouped_runtime_raw_seal_v1.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    artifacts = []
    for job in queue.jobs():
        path = RAW / job.name / "result.json"
        result = json.loads(path.read_text())
        if result["status"] != "P10_2_mixed_policy_runtime_measured" or result["batching_mode"] != "grouped":
            raise RuntimeError(f"bad grouped result: {job.name}")
        if result["optimization_contract_sha256"] != p7.sha256_file(CONTRACT):
            raise RuntimeError(f"optimization contract mismatch: {job.name}")
        if result["quality_labels_read"]:
            raise RuntimeError(f"quality boundary violated: {job.name}")
        artifacts.append({
            "name": job.name, "path": str(path.relative_to(ROOT)), "sha256": p7.sha256_file(path),
            "states": result["states"], "rollout_seconds_median": result["mixed_rollout_seconds_median"],
            "logical_fraction": result["logical_token_layer_fraction_of_Exact"],
        })
    payload = {
        "status": "P10_5_all_10_grouped_runtime_results_sealed",
        "contract_sha256": p7.sha256_file(CONTRACT),
        "numeric_canary_sha256": p7.sha256_file(ROOT / "results/p10/p10_5_batching_equivalence_canary_v1.json"),
        "cells": len(artifacts), "artifacts": artifacts,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": payload["cells"]}, indent=2))


if __name__ == "__main__":
    main()
