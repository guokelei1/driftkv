#!/usr/bin/env python3
"""Join sealed P10 policies with sealed P9.9 held-out rolling logits."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

import train_p7_theta0 as p7
from adjudicate_p9_heldout_rolling_quality import (
    ACTIONS,
    bernoulli_js,
    logloss,
    metrics,
    user_cluster_bootstrap_ci,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p10_1_policy_quality_contract_v1.yaml"
POLICY_SEAL = ROOT / "results/p10/p10_0_cheap_profiler_full_seal_v1.json"
QUALITY_SEAL = ROOT / "results/p9/p9_9_heldout_rolling_quality_raw_seal_v1.json"
PROFILER_CONTRACT = ROOT / "configs/contracts/p10_0_cheap_profiler_contract_v1.yaml"
OUTPUT = ROOT / "results/p10/p10_1_policy_quality_v1.json"


def validate() -> tuple[dict, dict, dict]:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p10_0_full_policy_seal_sha256": POLICY_SEAL,
        "p9_9_heldout_quality_raw_seal_sha256": QUALITY_SEAL,
        "p10_0_profiler_contract_sha256": PROFILER_CONTRACT,
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P10.1 input changed: {key}")
    return contract, json.loads(POLICY_SEAL.read_text()), json.loads(QUALITY_SEAL.read_text())


def sliced_logloss(frame, logits, labels) -> dict:
    masks = {
        "prior_30m_same_item": frame["prior_30m_same_item"].to_numpy(dtype=bool),
        "non_prior_30m_same_item": ~frame["prior_30m_same_item"].to_numpy(dtype=bool),
        "latest_item": frame["latest_item"].to_numpy(dtype=bool),
        "non_latest_item": ~frame["latest_item"].to_numpy(dtype=bool),
        "organic": frame["is_organic"].to_numpy(dtype=np.int64) == 1,
        "recommendation_driven": frame["is_organic"].to_numpy(dtype=np.int64) == 0,
    }
    losses = logloss(logits, labels)
    return {
        name: {"requests": int(mask.sum()), "log_loss": float(np.mean(losses[mask])) if mask.any() else None}
        for name, mask in masks.items()
    }


def evaluate_policy(raw_frame, assignment_frame, release: str, model: str, seed: int, sample: float, budget: float) -> dict:
    assignment = assignment_frame[
        np.isclose(assignment_frame["sample_fraction"], sample)
        & np.isclose(assignment_frame["budget_fraction"], budget)
    ][["uid", "action"]]
    if assignment["uid"].duplicated().any():
        raise RuntimeError("policy has multiple actions per uid")
    reference = raw_frame[raw_frame["action"] == "exact_all"].sort_values("request_id").reset_index(drop=True)
    uids = reference["uid"].to_numpy(dtype=np.int64)
    labels = reference["label"].to_numpy(dtype=np.int64)
    current = reference["current_exact_logit"].to_numpy(dtype=np.float64)
    action_by_uid = dict(zip(assignment["uid"].astype(int), assignment["action"].astype(str)))
    try:
        selected_actions = np.asarray([action_by_uid[int(uid)] for uid in uids], dtype=object)
    except KeyError as error:
        raise RuntimeError(f"heldout uid absent from sealed policy: {error}") from error
    selected_logits = np.empty(len(reference), dtype=np.float64)
    logits_by_action = {}
    for action in ACTIONS:
        selected = raw_frame[raw_frame["action"] == action].sort_values("request_id").reset_index(drop=True)
        if not np.array_equal(selected["request_id"].to_numpy(), reference["request_id"].to_numpy()):
            raise RuntimeError("P9.9 action request populations differ")
        logits_by_action[action] = selected["action_logit"].to_numpy(dtype=np.float64)
        mask = selected_actions == action
        selected_logits[mask] = logits_by_action[action][mask]
    noop = logits_by_action["noop"]
    policy_metrics = metrics(selected_logits, labels)
    current_metrics = metrics(current, labels)
    noop_metrics = metrics(noop, labels)
    policy_loss = logloss(selected_logits, labels)
    current_loss = logloss(current, labels)
    noop_loss = logloss(noop, labels)
    return {
        "sample_fraction": sample,
        "budget_fraction": budget,
        "requests": len(reference),
        "users": int(np.unique(uids).size),
        "action_counts_on_heldout_users": {action: int(np.sum(selected_actions == action)) for action in ACTIONS},
        "mean_Bernoulli_JS_to_CurrentExact": float(np.mean(bernoulli_js(selected_logits, current))),
        "quality": policy_metrics,
        "policy_minus_CurrentExact": {key: policy_metrics[key] - current_metrics[key] for key in current_metrics},
        "Noop_minus_policy": {key: noop_metrics[key] - policy_metrics[key] for key in current_metrics},
        "policy_logloss_slices": sliced_logloss(reference, selected_logits, labels),
        "Noop_minus_policy_logloss_slices": {
            name: noop_value["log_loss"] - policy_value["log_loss"]
            if noop_value["log_loss"] is not None and policy_value["log_loss"] is not None else None
            for name, noop_value, policy_value in (
                (name, noop_slice, policy_slice)
                for (name, noop_slice), policy_slice in zip(
                    sliced_logloss(reference, noop, labels).items(),
                    sliced_logloss(reference, selected_logits, labels).values(),
                    strict=True,
                )
            )
        },
        "user_cluster_bootstrap_95CI": {
            "policy_minus_CurrentExact_logloss": user_cluster_bootstrap_ci(
                uids, policy_loss - current_loss, f"p10:{release}:{model}:{seed}:{sample}:{budget}:current"
            ),
            "Noop_minus_policy_logloss": user_cluster_bootstrap_ci(
                uids, noop_loss - policy_loss, f"p10:{release}:{model}:{seed}:{sample}:{budget}:noop"
            ),
        },
    }


def evaluate_cell(arguments: tuple[tuple[str, str, int], dict, dict, dict]) -> dict:
    key, policy_artifact, quality_artifact, contract = arguments
    release, model, seed = key
    policy = pq.read_table(ROOT / policy_artifact["assignments"]).to_pandas()
    raw = pq.read_table(ROOT / quality_artifact["raw_path"]).to_pandas()
    evaluations = [
        evaluate_policy(raw, policy, release, model, seed, float(sample), float(budget))
        for sample in contract["scope"]["sample_fractions"]
        for budget in contract["scope"]["budgets_exact_fraction"]
    ]
    return {"release": release, "model": model, "seed": seed, "evaluations": evaluations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    contract, policy_seal, quality_seal = validate()
    policy_artifacts = {(row["release"], row["model"], int(row["seed"])): row for row in policy_seal["artifacts"]}
    quality_artifacts = {(row["release"], row["model"], int(row["seed"])): row for row in quality_seal["artifacts"]}
    if set(policy_artifacts) != set(quality_artifacts):
        raise RuntimeError("policy and quality cell populations differ")
    work = [(key, policy_artifacts[key], quality_artifacts[key], contract) for key in sorted(policy_artifacts)]
    with ProcessPoolExecutor(max_workers=min(max(args.workers, 1), len(work))) as pool:
        cells = list(pool.map(evaluate_cell, work))
    aggregates = []
    for release in ("r0", "r1_edge1", "r1_edge2", "r2"):
        for model in ("m0_f", "m1"):
            group = sorted([row for row in cells if row["release"] == release and row["model"] == model], key=lambda row: row["seed"])
            for sample in map(float, contract["scope"]["sample_fractions"]):
                for budget in map(float, contract["scope"]["budgets_exact_fraction"]):
                    rows = [next(value for value in cell["evaluations"] if value["sample_fraction"] == sample and value["budget_fraction"] == budget) for cell in group]
                    aggregates.append({
                        "release": release, "model": model, "sample_fraction": sample, "budget_fraction": budget,
                        "seed_order": [cell["seed"] for cell in group],
                        "Noop_minus_policy_logloss_seed_points": [row["Noop_minus_policy"]["log_loss"] for row in rows],
                        "policy_minus_CurrentExact_logloss_seed_points": [row["policy_minus_CurrentExact"]["log_loss"] for row in rows],
                        "Noop_minus_policy_ROC_AUC_seed_points": [row["Noop_minus_policy"]["ROC_AUC"] for row in rows],
                        "Noop_minus_policy_dislike_PR_AUC_seed_points": [row["Noop_minus_policy"]["dislike_PR_AUC"] for row in rows],
                        "Noop_minus_policy_Brier_seed_points": [row["Noop_minus_policy"]["Brier"] for row in rows],
                        "Noop_minus_policy_dislike_only_logloss_seed_points": [row["Noop_minus_policy"]["dislike_only_log_loss"] for row in rows],
                    })
    payload = {
        "status": "P10_1_all_144_presealed_profiler_policies_joined_to_heldout_rolling_quality",
        "contract_sha256": p7.sha256_file(CONTRACT),
        "policy_seal_sha256": p7.sha256_file(POLICY_SEAL),
        "quality_raw_seal_sha256": p7.sha256_file(QUALITY_SEAL),
        "cells": cells,
        "aggregates": aggregates,
        "policy_or_hyperparameter_selection_after_labels": False,
        "controller_authorized": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells), "policies": sum(len(row["evaluations"]) for row in cells)}, indent=2))


if __name__ == "__main__":
    main()
