#!/usr/bin/env python3
"""Compute P9.9 quality and fidelity metrics from the sealed raw logits."""

import json
import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / "results/p9/p9_9_heldout_rolling_quality_raw_seal_v1.json"
OUTPUT = ROOT / "results/p9/p9_9_heldout_rolling_quality_v1.json"
ACTIONS = ("noop", "layer0_recent128", "layer0_middle", "layer0_full", "hybrid_tail128", "exact_all")


def sigmoid(logits):
    logits = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(logits)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp = np.exp(logits[~positive])
    output[~positive] = exp / (1.0 + exp)
    return output


def logloss(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return np.maximum(logits, 0) - labels * logits + np.log1p(np.exp(-np.abs(logits)))


def bernoulli_js(left, right):
    left, right = sigmoid(left), sigmoid(right)
    midpoint = 0.5 * (left + right)
    epsilon = 1e-15
    left, right, midpoint = [np.clip(value, epsilon, 1 - epsilon) for value in (left, right, midpoint)]
    kl_left = left * np.log(left / midpoint) + (1 - left) * np.log((1 - left) / (1 - midpoint))
    kl_right = right * np.log(right / midpoint) + (1 - right) * np.log((1 - right) / (1 - midpoint))
    return 0.5 * (kl_left + kl_right)


def metrics(logits, labels):
    probabilities = sigmoid(logits)
    losses = logloss(logits, labels)
    dislikes = labels == 0
    return {
        "log_loss": float(np.mean(losses)),
        "ROC_AUC": float(roc_auc_score(labels, probabilities)),
        "dislike_PR_AUC": float(average_precision_score(dislikes, 1 - probabilities)),
        "Brier": float(np.mean((probabilities - labels) ** 2)),
        "dislike_only_log_loss": float(np.mean(losses[dislikes])),
    }


def user_cluster_bootstrap_ci(uids, values, key, repetitions=2000):
    unique, inverse = np.unique(uids, return_inverse=True)
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=np.float64))
    counts = np.bincount(inverse)
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    estimates = []
    for begin in range(0, repetitions, 100):
        width = min(100, repetitions - begin)
        sampled = rng.integers(0, len(unique), size=(width, len(unique)))
        estimates.extend((sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)).tolist())
    return {
        "repetitions": repetitions,
        "p2_5": float(np.quantile(estimates, 0.025)),
        "p97_5": float(np.quantile(estimates, 0.975)),
    }


def cell_metrics(artifact):
    raw_path = ROOT / artifact["raw_path"]
    if p7.sha256_file(raw_path) != artifact["raw_sha256"]:
        raise RuntimeError("P9.9 raw changed after seal")
    frame = pq.read_table(raw_path).to_pandas()
    reference = frame[frame["action"] == "exact_all"].sort_values("request_id")
    labels = reference["label"].to_numpy(dtype=np.int64)
    uids = reference["uid"].to_numpy(dtype=np.int64)
    current = reference["current_exact_logit"].to_numpy(dtype=np.float64)
    noop_selected = frame[frame["action"] == "noop"].sort_values("request_id")
    noop_logits = noop_selected["action_logit"].to_numpy(dtype=np.float64)
    noop_losses = logloss(noop_logits, labels)
    current_losses = logloss(current, labels)
    summaries = []
    noop_js = None
    for action in ACTIONS:
        selected = frame[frame["action"] == action].sort_values("request_id")
        if not np.array_equal(selected["request_id"].to_numpy(), reference["request_id"].to_numpy()):
            raise RuntimeError("P9.9 action request populations differ")
        logits = selected["action_logit"].to_numpy(dtype=np.float64)
        js = bernoulli_js(logits, current)
        action_metrics = metrics(logits, labels)
        current_metrics = metrics(current, labels)
        action_losses = logloss(logits, labels)
        if action == "noop":
            noop_js = float(np.mean(js))
        summaries.append({
            "action": action, "requests": len(selected),
            "mean_JS_to_CurrentExact": float(np.mean(js)),
            "p95_JS_to_CurrentExact": float(np.quantile(js, 0.95)),
            "quality": action_metrics,
            "action_minus_current_quality": {
                key: action_metrics[key] - current_metrics[key] for key in current_metrics
            },
            "user_cluster_bootstrap_95CI": {
                "action_minus_current_logloss": user_cluster_bootstrap_ci(
                    uids, action_losses - current_losses,
                    f"{artifact['release']}:{artifact['model']}:{artifact['seed']}:{action}:current",
                ),
                "noop_minus_action_logloss_recovery": user_cluster_bootstrap_ci(
                    uids, noop_losses - action_losses,
                    f"{artifact['release']}:{artifact['model']}:{artifact['seed']}:{action}:noop",
                ),
            },
        })
    assert noop_js is not None
    for row in summaries:
        row["JS_recovery_fraction_vs_noop"] = (
            (noop_js - row["mean_JS_to_CurrentExact"]) / noop_js if noop_js > 1e-20 else None
        )
    return {
        "release": artifact["release"], "model": artifact["model"], "seed": artifact["seed"],
        "users": artifact["users"], "requests": artifact["requests"], "summaries": summaries,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    seal = json.loads(SEAL.read_text())
    if seal["status"] != "P9_9_all_24_heldout_rolling_quality_raw_cells_sealed_before_metrics":
        raise RuntimeError("P9.9 raw is not sealed")
    cells = [cell_metrics(artifact) for artifact in seal["artifacts"]]
    aggregates = []
    for release in ("r0", "r1_edge1", "r1_edge2", "r2"):
        for model in ("m0_f", "m1"):
            group = sorted([row for row in cells if row["release"] == release and row["model"] == model], key=lambda row: row["seed"])
            for action in ACTIONS:
                rows = [next(value for value in cell["summaries"] if value["action"] == action) for cell in group]
                aggregates.append({
                    "release": release, "model": model, "action": action,
                    "seed_order": [cell["seed"] for cell in group],
                    "JS_recovery_seed_points": [row["JS_recovery_fraction_vs_noop"] for row in rows],
                    "action_minus_current_logloss_seed_points": [row["action_minus_current_quality"]["log_loss"] for row in rows],
                    "action_minus_current_ROC_AUC_seed_points": [row["action_minus_current_quality"]["ROC_AUC"] for row in rows],
                    "action_minus_current_dislike_PR_AUC_seed_points": [row["action_minus_current_quality"]["dislike_PR_AUC"] for row in rows],
                    "action_minus_current_Brier_seed_points": [row["action_minus_current_quality"]["Brier"] for row in rows],
                    "action_minus_current_dislike_only_logloss_seed_points": [row["action_minus_current_quality"]["dislike_only_log_loss"] for row in rows],
                })
    payload = {
        "status": "P9_9_heldout_rolling_quality_adjudicated",
        "raw_seal_sha256": p7.sha256_file(SEAL),
        "cells": cells, "aggregates": aggregates,
        "labels_used_for_action_or_state_selection": False,
        "scheduler_authorized": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells)}, indent=2))


if __name__ == "__main__":
    main()
