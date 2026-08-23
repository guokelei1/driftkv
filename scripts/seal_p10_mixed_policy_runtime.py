#!/usr/bin/env python3
"""Seal all ten frozen P10.2 mixed-policy runtime results."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import train_p7_theta0 as p7
import run_p10_mixed_policy_runtime_queue as queue


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p10_2_mixed_policy_runtime_contract_v1.yaml"
RAW = ROOT / "results/p10/mixed_policy_runtime/full"
OUTPUT = ROOT / "results/p10/p10_2_mixed_policy_runtime_raw_seal_v1.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    contract = yaml.safe_load(CONTRACT.read_text())
    expected = queue.jobs()
    artifacts, evaluator_hashes = [], set()
    for job in expected:
        path = RAW / job.name / "result.json"
        if not path.exists():
            raise RuntimeError(f"missing runtime result: {job.name}")
        result = json.loads(path.read_text())
        condition = result["condition"]
        if result["status"] != "P10_2_mixed_policy_runtime_measured":
            raise RuntimeError(f"bad runtime status: {job.name}")
        if result["contract_sha256"] != p7.sha256_file(CONTRACT):
            raise RuntimeError(f"contract mismatch: {job.name}")
        if result["quality_labels_read"] or result["controller_authorized"]:
            raise RuntimeError(f"evidence boundary violated: {job.name}")
        if not (
            condition["release"] == job.release
            and condition["model"] == job.model
            and int(condition["seed"]) == job.seed
            and abs(float(condition["sample_fraction"]) - job.sample) < 1e-12
            and abs(float(condition["budget_fraction"]) - job.budget) < 1e-12
        ):
            raise RuntimeError(f"condition mismatch: {job.name}")
        logical_fraction = float(result["logical_token_layer_fraction_of_Exact"])
        if logical_fraction > job.budget + 1e-12 or job.budget - logical_fraction > 1e-5:
            raise RuntimeError(f"logical budget mismatch: {job.name}")
        if not all(float(value) >= 0 for value in result["mixed_rollout_seconds_points"]):
            raise RuntimeError(f"invalid timing: {job.name}")
        evaluator_hashes.add(result["evaluator_sha256"] if "evaluator_sha256" in result else None)
        artifacts.append({
            "name": job.name,
            "path": str(path.relative_to(ROOT)),
            "sha256": p7.sha256_file(path),
            "states": result["states"],
            "logical_fraction": result["logical_token_layer_fraction_of_Exact"],
            "rollout_seconds_median": result["mixed_rollout_seconds_median"],
        })
    payload = {
        "status": "P10_2_all_10_mixed_policy_runtime_results_sealed",
        "contract_sha256": p7.sha256_file(CONTRACT),
        "cells": len(artifacts),
        "artifacts": artifacts,
        "quality_labels_read": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": payload["cells"]}, indent=2))


if __name__ == "__main__":
    main()
