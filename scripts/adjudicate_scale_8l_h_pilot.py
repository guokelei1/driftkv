#!/usr/bin/env python3
"""Adjudicate the sealed, development-only 8L M0-F seed-17 H pilot."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/scale_8l_h_pilot_v1.yaml"
RAW_ROOT = ROOT / "results/scale_8l_v1/pilot/h_m0_f_seed17"
SEAL = RAW_ROOT / "raw_score_seal.json"
OUTPUT = ROOT / "results/scale_8l_v1/pilot/s3_m0_f_seed17_h_adjudication.json"
REPLICATES = 2000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def bernoulli_js(recent, full):
    p = sigmoid(recent)
    q = sigmoid(full)
    middle = 0.5 * (p + q)
    return 0.5 * (
        p * np.log(p / middle)
        + (1.0 - p) * np.log((1.0 - p) / (1.0 - middle))
        + q * np.log(q / middle)
        + (1.0 - q) * np.log((1.0 - q) / (1.0 - middle))
    )


def rng_for(name: str):
    seed = int.from_bytes(hashlib.sha256(f"scale-8l-H-pilot-v1:{name}".encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def user_mean_summary(uid, values, name: str):
    grouped: dict[int, list[float]] = defaultdict(list)
    for user, value in zip(uid, values, strict=True):
        grouped[int(user)].append(float(value))
    user_values = np.asarray([np.mean(grouped[user]) for user in sorted(grouped)], dtype=np.float64)
    rng = rng_for(name)
    draws = user_values[rng.integers(0, len(user_values), size=(REPLICATES, len(user_values)))].mean(axis=1)
    return {
        "point": float(user_values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "users": len(user_values),
    }


def weighted_auc(labels, recent, full, weights):
    return {
        "ROC_AUC_gain": float(
            roc_auc_score(labels, full, sample_weight=weights)
            - roc_auc_score(labels, recent, sample_weight=weights)
        ),
        "dislike_PR_AUC_gain": float(
            average_precision_score(1 - labels, 1 - full, sample_weight=weights)
            - average_precision_score(1 - labels, 1 - recent, sample_weight=weights)
        ),
    }


def classification_bootstrap(frame):
    labels = frame["label"].to_numpy(dtype=np.int64)
    uid = frame["uid"].to_numpy(dtype=np.int64)
    recent = sigmoid(frame["recent32_deployment_logit"].to_numpy(dtype=np.float64))
    full = sigmoid(frame["full1024_deployment_logit"].to_numpy(dtype=np.float64))
    weights = frame["request_weight"].to_numpy(dtype=np.float64)
    users = np.asarray(sorted(set(uid)), dtype=np.int64)
    user_index = {int(user): index for index, user in enumerate(users)}
    row_user = np.asarray([user_index[int(user)] for user in uid], dtype=np.int64)
    point = weighted_auc(labels, recent, full, weights)
    draws = {key: np.empty(REPLICATES, dtype=np.float64) for key in point}
    rng = rng_for("classification")
    for replicate in range(REPLICATES):
        sampled = rng.integers(0, len(users), size=len(users))
        counts = np.bincount(sampled, minlength=len(users))
        boot_weights = weights * counts[row_user]
        values = weighted_auc(labels, recent, full, boot_weights)
        for key, value in values.items():
            draws[key][replicate] = value
    return {
        key: {
            "point": point[key],
            "ci95": [float(np.quantile(draws[key], 0.025)), float(np.quantile(draws[key], 0.975))],
            "users": len(users),
        }
        for key in point
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract["sealed_inputs"]["adjudicator_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("adjudicator changed after pilot contract freeze")
    seal = json.loads(SEAL.read_text())
    if seal["contract_sha256"] != sha256_file(CONTRACT) or seal["metrics_computed"] is not False:
        raise RuntimeError("raw seal is not valid for this contract")
    artifacts = {row["view"]: row for row in seal["artifacts"]}
    for row in artifacts.values():
        if sha256_file(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError("sealed raw artifact changed")

    fidelity = pq.read_table(ROOT / artifacts["fidelity"]["path"]).to_pandas()
    recent_f = fidelity["recent32_deployment_logit"].to_numpy(dtype=np.float64)
    full_f = fidelity["full1024_deployment_logit"].to_numpy(dtype=np.float64)
    uid_f = fidelity["uid"].to_numpy(dtype=np.int64)
    js = bernoulli_js(recent_f, full_f)
    probability_delta = np.abs(sigmoid(full_f) - sigmoid(recent_f))
    logit_delta = full_f - recent_f
    full_std = max(float(np.std(full_f)), 1e-3)
    fidelity_metrics = {
        "output_js_divergence": user_mean_summary(uid_f, js, "fidelity-js"),
        "normalized_score_rms": user_mean_summary(uid_f, np.abs(logit_delta) / full_std, "fidelity-rms"),
        "absolute_probability_difference": user_mean_summary(uid_f, probability_delta, "fidelity-probability"),
    }
    panels = []
    for panel in range(4):
        mask = np.asarray(
            [int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], "big") % 4 == panel for value in fidelity["request_id"]],
            dtype=bool,
        )
        panels.append(float(np.mean(js[mask])))

    quality = pq.read_table(ROOT / artifacts["quality"]["path"]).to_pandas()
    labels = quality["label"].to_numpy(dtype=np.int64)
    uid_q = quality["uid"].to_numpy(dtype=np.int64)
    recent_logit = quality["recent32_deployment_logit"].to_numpy(dtype=np.float64)
    full_logit = quality["full1024_deployment_logit"].to_numpy(dtype=np.float64)
    recent_probability = sigmoid(recent_logit)
    full_probability = sigmoid(full_logit)
    recent_loss = np.logaddexp(0.0, recent_logit) - labels * recent_logit
    full_loss = np.logaddexp(0.0, full_logit) - labels * full_logit
    quality_metrics = {
        "log_loss_gain": user_mean_summary(uid_q, recent_loss - full_loss, "quality-logloss"),
        "Brier_gain": user_mean_summary(
            uid_q,
            (recent_probability - labels) ** 2 - (full_probability - labels) ** 2,
            "quality-brier",
        ),
        **classification_bootstrap(quality),
    }
    dislike = labels == 0
    quality_metrics["dislike_only_log_loss_gain"] = user_mean_summary(
        uid_q[dislike], (recent_loss - full_loss)[dislike], "quality-dislike-logloss"
    )

    floors = contract["gates"]["numeric_floors"]
    components = {
        "H_JS_CI_above_floor": fidelity_metrics["output_js_divergence"]["ci95"][0] > floors["output_js_divergence"],
        "H_probability_shift_CI_above_floor": fidelity_metrics["absolute_probability_difference"]["ci95"][0] > floors["absolute_probability_difference"],
        "H_repeatable_panels": sum(point > floors["output_js_divergence"] for point in panels) >= 3,
        "quality_log_loss_CI_positive": quality_metrics["log_loss_gain"]["ci95"][0] > 0.0,
        "ROC_AUC_noninferior": quality_metrics["ROC_AUC_gain"]["ci95"][0] >= -0.005,
        "dislike_PR_AUC_noninferior": quality_metrics["dislike_PR_AUC_gain"]["ci95"][0] >= -0.01,
    }
    passed = all(components.values())
    result = {
        "status": "passed_scale_H_pilot_gate" if passed else "failed_scale_H_pilot_gate_stop",
        "evidence_level": "development_single_seed_cost_control_gate_not_paper_qualification",
        "model_condition": "m0_f",
        "seed": 17,
        "comparison": "Base+Full1024_vs_Base+Recent32",
        "qualification_or_theta3_read": False,
        "contract_sha256": sha256_file(CONTRACT),
        "raw_score_seal_sha256": sha256_file(SEAL),
        "fidelity": fidelity_metrics,
        "fidelity_JS_panel_points": panels,
        "quality": quality_metrics,
        "gate_components": components,
        "pilot_H_passed": passed,
        "authorization": "seed17_release_chain_pilot_may_start" if passed else "stop_before_release_training",
        "limitations": [
            "development split was used because qualification and theta3 remain prohibited",
            "one seed cannot establish cross-seed scale robustness",
            "dislike-only log loss is a mandatory companion and not a retroactive gate",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
