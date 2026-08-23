#!/usr/bin/env python3
"""Join sealed P9.2 diagnostic logits to sealed F labels and report quality.

The join is deliberately strict: P8 fidelity and quality views must be exactly
row-aligned on all causal identifiers and all shared logits before the label is
associated with a fidelity request id. Labels never select a state or action.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml

import run_p9_tomography as ledger
import train_p7_theta0 as p7

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_2_closure_contract_v1.yaml"
OUTPUT = ROOT / "results/p9/p9_2_quality_companions_v1.json"
P8_ROOT = ROOT / "results/p8/staleness_raw"
LOWER_IS_BETTER = {"aggregate_logloss", "Brier", "dislike_only_logloss"}
METRICS = ("aggregate_logloss", "ROC_AUC", "dislike_PR_AUC", "Brier", "dislike_only_logloss")


def validate_contract_inputs() -> dict[str, Any]:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p9_contract": ROOT / "configs/contracts/p9_tomography_contract_v1.yaml",
        "p9_1_distribution_v2": ROOT / "results/p9/p9_1_hs_distribution_v2.json",
        "p9_2_raw_seal": ROOT / "results/p9/p9_2_tomography_raw_seal_v1.json",
        "p9_2_coarse_result": ROOT / "results/p9/p9_2_coarse_tomography_v1.json",
        "p8_r0_raw_seal": ROOT / "results/p8/r0_control/raw_score_seal_v1.json",
        "p8_r1_edge1_raw_seal": ROOT / "results/p8/r1_edge1/raw_score_seal_v1.json",
        "p8_r1_edge2_raw_seal": ROOT / "results/p8/r1_edge2/raw_score_seal_v1.json",
        "p8_r2_raw_seal": ROOT / "results/p8/r2/raw_score_seal_v1.json",
    }
    for name, path in paths.items():
        if p7.sha256_file(path) != contract["input_hashes"][name]:
            raise RuntimeError(f"P9.2 closure input hash mismatch: {name}")
    return contract


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp = np.exp(values[~positive])
    output[~positive] = exp / (1.0 + exp)
    return output


def equal_user_weights(uids: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    selected = np.ones(len(uids), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    weights = np.zeros(len(uids), dtype=np.float64)
    unique, counts = np.unique(uids[selected], return_counts=True)
    inverse = {int(uid): 1.0 / int(count) for uid, count in zip(unique, counts, strict=True)}
    for index in np.flatnonzero(selected):
        weights[index] = inverse[int(uids[index])]
    return weights


def binary_metrics(labels: np.ndarray, logits: np.ndarray, uids: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    uids = np.asarray(uids, dtype=np.int64)
    probabilities = sigmoid(logits)
    weights = equal_user_weights(uids)
    loss = np.logaddexp(0.0, logits) - labels * logits
    dislike = labels == 0
    dislike_weights = equal_user_weights(uids, dislike)
    return {
        "aggregate_logloss": float(np.average(loss, weights=weights)),
        "ROC_AUC": float(roc_auc_score(labels, probabilities, sample_weight=weights)),
        "dislike_PR_AUC": float(average_precision_score(1 - labels, 1.0 - probabilities, sample_weight=weights)),
        "Brier": float(np.average((probabilities - labels) ** 2, weights=weights)),
        "dislike_only_logloss": float(np.average(loss[dislike], weights=dislike_weights[dislike])),
    }


def quality_gain(reference: float, candidate: float, metric: str) -> float:
    """Positive means candidate quality is better than reference quality."""
    return reference - candidate if metric in LOWER_IS_BETTER else candidate - reference


def validate_and_map_labels(job: ledger.Job) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    root = P8_ROOT / job.release / f"{job.model}_seed{job.seed}"
    fidelity_path, quality_path = root / "F_fidelity.parquet", root / "F_quality.parquet"
    raw_manifest = json.loads((root / "raw_manifest.json").read_text())
    sealed = {(row["workload"], row["view"]): row["sha256"] for row in raw_manifest["artifacts"]}
    if p7.sha256_file(fidelity_path) != sealed[("F", "fidelity")]:
        raise RuntimeError(f"P8 F fidelity changed after sealing: {job}")
    if p7.sha256_file(quality_path) != sealed[("F", "quality")]:
        raise RuntimeError(f"P8 F quality changed after sealing: {job}")
    identifier = [
        "uid", "query_timestamp", "candidate_position", "candidate_id",
        "prefix_tokens_at_cutover", "suffix_tokens_after_cutover",
    ]
    scores = [
        "base_logit", "previous_full_logit", "current_recent32_logit",
        "current_full512_logit", "reuse_parent_kv_logit",
    ]
    fidelity = pq.read_table(fidelity_path, columns=["request_id", *identifier, *scores]).to_pandas()
    quality = pq.read_table(quality_path, columns=[*identifier, *scores, "label"]).to_pandas()
    if len(fidelity) != len(quality):
        raise RuntimeError(f"F view row count mismatch: {job}")
    for name in identifier:
        if not np.array_equal(fidelity[name].to_numpy(), quality[name].to_numpy()):
            raise RuntimeError(f"F view identifier mismatch in {name}: {job}")
    for name in scores:
        if not np.array_equal(fidelity[name].to_numpy(), quality[name].to_numpy()):
            raise RuntimeError(f"F view score mismatch in {name}: {job}")
    if fidelity["request_id"].duplicated().any():
        raise RuntimeError(f"F fidelity request ids are not unique: {job}")
    labels = quality["label"].to_numpy(dtype=np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError(f"F quality lacks both labels: {job}")
    mapping = dict(zip(fidelity["request_id"].astype(str), labels, strict=True))
    return mapping, fidelity["uid"].to_numpy(dtype=np.int64), labels


def build_cell(job: ledger.Job) -> dict[str, Any]:
    label_map, _, _ = validate_and_map_labels(job)
    raw = job.output / "F_fidelity_tomography.parquet"
    frame = pq.read_table(raw).to_pandas()
    expected_actions = yaml.safe_load(CONTRACT.read_text())["scope"]["actions"]
    action_count = len(expected_actions)
    if set(frame["action"].unique()) != set(expected_actions):
        raise RuntimeError(f"diagnostic action set mismatch: {job}")
    if not (frame.groupby("request_id", sort=False).size() == action_count).all():
        raise RuntimeError(f"diagnostic action row conservation failed: {job}")
    frame["label"] = frame["request_id"].astype(str).map(label_map)
    if frame["label"].isna().any():
        raise RuntimeError(f"tomography request missing sealed label mapping: {job}")
    baseline = frame.drop_duplicates("request_id", keep="first")
    uids = baseline["uid"].to_numpy(dtype=np.int64)
    labels = baseline["label"].to_numpy(dtype=np.int64)
    full = binary_metrics(labels, baseline["full_logit"].to_numpy(), uids)
    reuse = binary_metrics(labels, baseline["reuse_logit"].to_numpy(), uids)
    full_gain = {metric: quality_gain(reuse[metric], full[metric], metric) for metric in METRICS}
    actions = {}
    for action in expected_actions:
        selected = frame[frame["action"] == action]
        if not np.array_equal(selected["request_id"].astype(str).to_numpy(), baseline["request_id"].astype(str).to_numpy()):
            raise RuntimeError(f"diagnostic request order differs by action: {job} {action}")
        values = binary_metrics(labels, selected["diagnostic_logit"].to_numpy(), uids)
        gain = {metric: quality_gain(reuse[metric], values[metric], metric) for metric in METRICS}
        actions[action] = {
            "absolute": values,
            "quality_gain_vs_reuse": gain,
            "quality_gain_vs_current_full": {
                metric: quality_gain(full[metric], values[metric], metric) for metric in METRICS
            },
            "recovery_fraction_of_full_gain_companion": {
                metric: (gain[metric] / full_gain[metric] if abs(full_gain[metric]) > 1e-12 else None)
                for metric in METRICS
            },
        }
    return {
        "release": job.release, "model": job.model, "seed": job.seed,
        "requests": int(len(baseline)), "users": int(len(np.unique(uids))),
        "labels": {"like": int(np.sum(labels == 1)), "dislike": int(np.sum(labels == 0))},
        "sources": {
            "tomography": str(raw.relative_to(ROOT)), "tomography_sha256": p7.sha256_file(raw),
            "p8_fidelity": str((P8_ROOT / job.release / f"{job.model}_seed{job.seed}" / "F_fidelity.parquet").relative_to(ROOT)),
            "p8_quality": str((P8_ROOT / job.release / f"{job.model}_seed{job.seed}" / "F_quality.parquet").relative_to(ROOT)),
        },
        "view_equivalence": "all causal identifiers and shared logits exactly equal by frozen row ordinal",
        "current_full": full, "reuse": reuse, "current_full_quality_gain_vs_reuse": full_gain,
        "actions": actions,
    }


def aggregate(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for release in ("r0", "r1_edge1", "r1_edge2", "r2"):
        for model in ("m0_f", "m1"):
            rows = sorted([row for row in cells if row["release"] == release and row["model"] == model], key=lambda row: row["seed"])
            for action in yaml.safe_load(CONTRACT.read_text())["scope"]["actions"]:
                metrics = {}
                for metric in METRICS:
                    points = [row["actions"][action]["quality_gain_vs_reuse"][metric] for row in rows]
                    metrics[metric] = {
                        "seed_points": points, "equal_seed_mean": float(np.mean(points)),
                        "positive_seed_count": int(np.sum(np.asarray(points) > 0)),
                    }
                output.append({
                    "release": release, "model": model, "action": action,
                    "seed_order": [row["seed"] for row in rows], "quality_gain_vs_reuse": metrics,
                })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite P9.2-Q result: {args.output}")
    if not 1 <= args.workers <= 24:
        raise ValueError("workers must be in [1,24]")
    contract = validate_contract_inputs()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        cells = list(executor.map(build_cell, ledger.jobs()))
    payload = {
        "status": "P9_2_diagnostic_quality_companions_complete",
        "contract_hash": p7.sha256_file(CONTRACT),
        "diagnostic_not_executable_action": True,
        "labels_used_only_for_posthoc_quality_companions": True,
        "cells": cells, "aggregate": aggregate(cells),
        "notes": [
            "All 24 frozen cells, all actions, and all seeds are reported; labels select neither states nor actions.",
            "Positive quality gain means the diagnostic action is better than Reuse for that metric.",
            "Recovery fractions are companions and are unstable when Current Full and Reuse have near-zero quality difference.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
