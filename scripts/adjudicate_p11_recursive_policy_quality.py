#!/usr/bin/env python3
"""Join sealed P11.2 policies with sealed P11.4 recursive quality logits."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import adjudicate_p10_policy_quality as p10q
import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p11_4_recursive_policy_quality_v1.yaml"
POLICY_SEAL = ROOT / "results/p11/p11_2_recursive_scheduler_full_seal_v1.json"
QUALITY_SEAL = ROOT / "results/p11/p11_4_recursive_policy_quality_raw_seal_v1.json"
OUTPUT = ROOT / "results/p11/p11_4_recursive_policy_quality_v1.json"


def evaluate_cell(args):
    model, seed, policy_artifact, quality_artifact = args
    policy = pq.read_table(ROOT / policy_artifact["assignments"]).to_pandas()
    raw = pq.read_table(ROOT / quality_artifact["raw_path"]).to_pandas()
    evaluations = [
        p10q.evaluate_policy(raw, policy, "r1_recursive_age2", model, seed, sample, budget)
        for sample in (0.01, 0.02) for budget in (0.05, 0.10, 0.25)
    ]
    return {"model": model, "seed": seed, "evaluations": evaluations}


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    policies = json.loads(POLICY_SEAL.read_text())
    qualities = json.loads(QUALITY_SEAL.read_text())
    if policies["contract_sha256"] != p7.sha256_file(ROOT / "configs/contracts/p11_2_recursive_scheduler_replay_v1.yaml"):
        raise RuntimeError("P11.2 policy seal contract mismatch")
    if qualities["contract_sha256"] != p7.sha256_file(CONTRACT):
        raise RuntimeError("P11.4 quality seal contract mismatch")
    policy_by = {(row["model"], int(row["seed"])): row for row in policies["artifacts"]}
    quality_by = {(row["model"], int(row["seed"])): row for row in qualities["artifacts"]}
    if set(policy_by) != set(quality_by):
        raise RuntimeError("P11.4 policy/quality cell mismatch")
    work = [(model, seed, policy_by[(model, seed)], quality_by[(model, seed)])
            for model, seed in sorted(policy_by)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        cells = list(pool.map(evaluate_cell, work))
    aggregates = []
    for model in ("m0_f", "m1"):
        group = sorted([cell for cell in cells if cell["model"] == model], key=lambda row: row["seed"])
        for sample in (0.01, 0.02):
            for budget in (0.05, 0.10, 0.25):
                values = [next(row for row in cell["evaluations"]
                               if row["sample_fraction"] == sample and row["budget_fraction"] == budget)
                          for cell in group]
                aggregates.append({
                    "model": model, "sample_fraction": sample, "budget_fraction": budget,
                    "seed_order": [cell["seed"] for cell in group],
                    "RecursiveNoop_minus_policy_logloss_seed_points": [row["Noop_minus_policy"]["log_loss"] for row in values],
                    "policy_minus_CurrentExact_logloss_seed_points": [row["policy_minus_CurrentExact"]["log_loss"] for row in values],
                    "RecursiveNoop_minus_policy_ROC_AUC_seed_points": [row["Noop_minus_policy"]["ROC_AUC"] for row in values],
                    "RecursiveNoop_minus_policy_dislike_PR_AUC_seed_points": [row["Noop_minus_policy"]["dislike_PR_AUC"] for row in values],
                    "RecursiveNoop_minus_policy_Brier_seed_points": [row["Noop_minus_policy"]["Brier"] for row in values],
                    "RecursiveNoop_minus_policy_dislike_only_logloss_seed_points": [row["Noop_minus_policy"]["dislike_only_log_loss"] for row in values],
                    "mean_Bernoulli_JS_to_CurrentExact_seed_points": [row["mean_Bernoulli_JS_to_CurrentExact"] for row in values],
                })
    payload = {"status": "P11_4_all_presealed_recursive_policies_joined_to_quality",
               "contract_sha256": p7.sha256_file(CONTRACT),
               "policy_seal_sha256": p7.sha256_file(POLICY_SEAL),
               "quality_seal_sha256": p7.sha256_file(QUALITY_SEAL),
               "cells": cells, "aggregates": aggregates,
               "policy_or_hyperparameter_selection_after_labels": False,
               "development_only": True, "theta3_executed": False}
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells), "policies": 36}, indent=2))


if __name__ == "__main__":
    main()
