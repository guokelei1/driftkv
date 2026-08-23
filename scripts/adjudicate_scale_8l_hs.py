#!/usr/bin/env python3
"""Compute frozen single-seed 8L H/S and quality companions after raw sealing."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/scale_8l_hs_v1.yaml"
SEAL = ROOT / "results/scale_8l_v1/hs_raw_seal_v1.json"
OUTPUT_ROOT = ROOT / "results/scale_8l_v1/pilot"
REPLICATES = 2000
COMPARISONS = {
    "H_request_local": ("request_local_current_full1024_logit", "current_recent32_logit"),
    "S_rolling": ("current_exact_rolling_logit", "reuse_parent_rolling_logit"),
    "release_quality": ("request_local_current_full1024_logit", "previous_full1024_logit"),
}


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


def bernoulli_js(reference, alternative):
    p, q = sigmoid(reference), sigmoid(alternative)
    m = 0.5 * (p + q)
    return 0.5 * (
        p * np.log(p / m) + (1 - p) * np.log((1 - p) / (1 - m))
        + q * np.log(q / m) + (1 - q) * np.log((1 - q) / (1 - m))
    )


def rng_for(*parts):
    token = "scale-8l-HS-v1|" + "|".join(str(part) for part in parts)
    return np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big"))


def user_summary(uid, values, namespace):
    grouped = defaultdict(list)
    for user, value in zip(uid, values, strict=True):
        if np.isfinite(value):
            grouped[int(user)].append(float(value))
    points = np.asarray([np.mean(grouped[user]) for user in sorted(grouped)], dtype=np.float64)
    rng = rng_for(*namespace)
    draws = points[rng.integers(0, len(points), size=(REPLICATES, len(points)))].mean(axis=1)
    return {
        "point": float(points.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "users": len(points),
    }


def weighted_classification(frame, current_column, alternative_column, namespace):
    uid = frame["uid"].to_numpy(dtype=np.int64)
    labels = frame["label"].to_numpy(dtype=np.int64)
    current = sigmoid(frame[current_column].to_numpy(dtype=np.float64))
    alternative = sigmoid(frame[alternative_column].to_numpy(dtype=np.float64))
    base_weights = frame["request_weight"].to_numpy(dtype=np.float64)
    users = np.asarray(sorted(set(uid)), dtype=np.int64)
    index = {int(user): position for position, user in enumerate(users)}
    row_user = np.asarray([index[int(user)] for user in uid], dtype=np.int64)

    def metrics(weights):
        dislike = 1 - labels
        return {
            "ROC_AUC_gain": float(
                roc_auc_score(labels, current, sample_weight=weights)
                - roc_auc_score(labels, alternative, sample_weight=weights)
            ),
            "dislike_PR_AUC_gain": float(
                average_precision_score(dislike, 1 - current, sample_weight=weights)
                - average_precision_score(dislike, 1 - alternative, sample_weight=weights)
            ),
        }

    point = metrics(base_weights)
    draws = {key: np.empty(REPLICATES) for key in point}
    rng = rng_for(*namespace, "classification")
    for replicate in range(REPLICATES):
        sampled = rng.integers(0, len(users), size=len(users))
        counts = np.bincount(sampled, minlength=len(users))
        result = metrics(base_weights * counts[row_user])
        for key in point:
            draws[key][replicate] = result[key]
    return {
        key: {
            "point": point[key],
            "ci95": [float(np.quantile(draws[key], 0.025)), float(np.quantile(draws[key], 0.975))],
            "users": len(users),
        }
        for key in point
    }


def artifact(seal, release, view):
    matches = [row for row in seal["artifacts"] if row["release"] == release and row["view"] == view]
    if len(matches) != 1:
        raise RuntimeError(f"sealed artifact lookup failed: {release}/{view}")
    path = ROOT / matches[0]["path"]
    if sha256_file(path) != matches[0]["sha256"]:
        raise RuntimeError("raw artifact changed after seal")
    return path


def evaluate_comparison(seal, release, name):
    current_column, alternative_column = COMPARISONS[name]
    fidelity = pq.read_table(artifact(seal, release, "fidelity")).to_pandas()
    current = fidelity[current_column].to_numpy(dtype=np.float64)
    alternative = fidelity[alternative_column].to_numpy(dtype=np.float64)
    uid = fidelity["uid"].to_numpy(dtype=np.int64)
    js = bernoulli_js(current, alternative)
    probability = np.abs(sigmoid(current) - sigmoid(alternative))
    denominator = max(float(np.std(current)), 1e-3)
    result = {
        "fidelity": {
            "output_js_divergence": user_summary(uid, js, (release, name, "js")),
            "normalized_score_rms": user_summary(uid, np.abs(current - alternative) / denominator, (release, name, "rms")),
            "absolute_probability_difference": user_summary(uid, probability, (release, name, "probability")),
            "request_percentiles_JS": {
                "P50": float(np.quantile(js, 0.50)), "P90": float(np.quantile(js, 0.90)),
                "P95": float(np.quantile(js, 0.95)), "P99": float(np.quantile(js, 0.99)),
            },
            "panel_points_JS": [
                float(np.mean(js[np.asarray([
                    int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], "big") % 4 == panel
                    for value in fidelity["request_id"]
                ], dtype=bool)])) for panel in range(4)
            ],
        }
    }
    quality = pq.read_table(artifact(seal, release, "quality")).to_pandas()
    labels = quality["label"].to_numpy(dtype=np.int64)
    uid_q = quality["uid"].to_numpy(dtype=np.int64)
    current_logit = quality[current_column].to_numpy(dtype=np.float64)
    alternative_logit = quality[alternative_column].to_numpy(dtype=np.float64)
    current_probability = sigmoid(current_logit)
    alternative_probability = sigmoid(alternative_logit)
    current_loss = np.logaddexp(0.0, current_logit) - labels * current_logit
    alternative_loss = np.logaddexp(0.0, alternative_logit) - labels * alternative_logit
    result["quality"] = {
        "log_loss_gain": user_summary(uid_q, alternative_loss - current_loss, (release, name, "logloss")),
        "Brier_gain": user_summary(
            uid_q,
            (alternative_probability - labels) ** 2 - (current_probability - labels) ** 2,
            (release, name, "brier"),
        ),
        **weighted_classification(quality, current_column, alternative_column, (release, name)),
    }
    dislike = labels == 0
    result["quality"]["dislike_only_log_loss_gain"] = user_summary(
        uid_q[dislike], (alternative_loss - current_loss)[dislike], (release, name, "dislike-logloss")
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=("r1_edge1", "r1_edge2", "r2"), required=True)
    args = parser.parse_args()
    output = OUTPUT_ROOT / f"s4_{args.release}_m0_f_seed17_adjudication.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract["sealed_inputs"]["adjudicator_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("adjudicator changed after H/S contract freeze")
    seal = json.loads(SEAL.read_text())
    if seal["contract_sha256"] != sha256_file(CONTRACT) or seal["metrics_computed"] is not False:
        raise RuntimeError("raw seal is invalid")
    run = next(row for row in seal["runs"] if row["release"] == args.release)
    comparisons = {
        name: evaluate_comparison(seal, args.release, name) for name in COMPARISONS
    }
    floor = float(contract["gates"]["numeric_floor_JS"])
    h = comparisons["H_request_local"]
    s = comparisons["S_rolling"]
    components = {
        "model_admitted": True,
        "Base_Full_Recent_identical": run["base_full_recent_max_abs_delta"] == 0.0,
        "current_H_JS_CI_above_floor": h["fidelity"]["output_js_divergence"]["ci95"][0] > floor,
        "current_H_probability_CI_above_floor": h["fidelity"]["absolute_probability_difference"]["ci95"][0] > 1e-7,
        "rolling_S_JS_CI_above_floor": s["fidelity"]["output_js_divergence"]["ci95"][0] > floor,
        "rolling_S_minimum_panels": sum(point > floor for point in s["fidelity"]["panel_points_JS"]) >= 3,
    }
    passed = all(components.values())
    h_point = h["fidelity"]["output_js_divergence"]["point"]
    s_point = s["fidelity"]["output_js_divergence"]["point"]
    result = {
        "status": "scale_HS_gate_passed" if passed else "scale_HS_gate_failed_stop",
        "evidence_level": "development_scale_pilot_single_seed_not_paper_qualification",
        "release": args.release, "model": "m0_f", "seed": 17,
        "qualification_or_theta3_read": False,
        "contract_sha256": sha256_file(CONTRACT), "raw_seal_sha256": sha256_file(SEAL),
        "comparisons": comparisons,
        "S_over_H_companion": s_point / h_point if h_point > 0 else None,
        "request_local_full_vs_exact_rolling_max_abs_logit_companion": run["request_local_full_companion_max_abs_logit"],
        "gate_components": components,
        "HS_gate_passed": passed,
        "authorization": "frozen_scale_actions_may_be_replayed_on_this_edge" if passed else "stop_before_partial_actions",
        "mandatory_caveats": [
            "single seed is a cost-control scale pilot",
            "quality harm from Reuse is a companion rather than an S existence gate",
            "dislike-only log loss is always reported",
            "request-local Full1024 is used for H; true rolling Exact is used as the S reference",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
