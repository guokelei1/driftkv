#!/usr/bin/env python3
"""Adjudicate sealed HSTU-native one-hop Reuse raw observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from hstu_kvcache.evaluation import bernoulli_js, binary_metrics, paired_harm, stable_log_loss
from evaluate_yambda500m_hstu_native_onehop_reuse_raw import (
    PAIR_PATHS,
    RELEASE_DEBT_PATHS,
    sha256_file,
    validate_pair_raw,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seal = json.loads(args.seal.read_text())
    if seal["raw_sha256"] != sha256_file(args.raw):
        raise RuntimeError("raw artifact differs from its pre-label seal")
    raw = pq.read_table(args.raw)
    expected_paths = RELEASE_DEBT_PATHS if seal.get("contains_parent_exact_rolling") else PAIR_PATHS
    pro_path = seal.get("pro_lazy_path") if seal.get("contains_pro_lazy") else None
    if pro_path is not None:
        expected_paths = (*expected_paths, str(pro_path))
    validate_pair_raw(raw, expected_paths=expected_paths)
    labels = pq.read_table(args.labels, columns=["request_id", "label"])
    label_map = dict(zip(labels["request_id"].to_pylist(), labels["label"].to_pylist(), strict=True))
    rows = raw.to_pylist()
    current = sorted((row for row in rows if row["path"] == PAIR_PATHS[0]), key=lambda row: row["request_id"])
    reuse = sorted((row for row in rows if row["path"] == PAIR_PATHS[1]), key=lambda row: row["request_id"])
    if [row["request_id"] for row in current] != [row["request_id"] for row in reuse]:
        raise RuntimeError("Reuse and recompute request pairs are not aligned")
    targets = np.asarray([label_map[row["request_id"]] for row in current], dtype=np.int64)
    current_logits = np.asarray([row["hstu_logit"] for row in current], dtype=np.float64)
    reuse_logits = np.asarray([row["hstu_logit"] for row in reuse], dtype=np.float64)
    current_metrics = binary_metrics(targets, current_logits)
    reuse_metrics = binary_metrics(targets, reuse_logits)
    loss_delta = stable_log_loss(reuse_logits, targets) - stable_log_loss(current_logits, targets)
    report = {
        "status": "native_onehop_reuse_adjudicated",
        "raw_sha256": seal["raw_sha256"],
        "stage": seal["stage"], "edge": seal["edge"], "evaluation_day_range": seal["evaluation_day_range"],
        "comparison": "CurrentExactRolling_vs_OneHopReuseRolling",
        "absolute_metrics": {"current_exact_rolling": current_metrics, "one_hop_reuse_rolling": reuse_metrics},
        "reuse_minus_recompute": {
            "paired_harm": paired_harm(uids=np.asarray([row["uid"] for row in current]), labels=targets, reuse_logits=reuse_logits, current_logits=current_logits, namespace=f"onehop:{seal['stage']}"),
            "event_weighted_log_loss": float(loss_delta.mean()),
            "current_minus_reuse_ROC_AUC_pp": (current_metrics["ROC_AUC"] - reuse_metrics["ROC_AUC"]) * 100,
            "current_minus_reuse_dislike_PR_AUC_pp": (current_metrics["dislike_PR_AUC"] - reuse_metrics["dislike_PR_AUC"]) * 100,
            "reuse_minus_current_Brier": reuse_metrics["Brier"] - current_metrics["Brier"],
            "mean_Bernoulli_JS": float(bernoulli_js(reuse_logits, current_logits).mean()),
            "mean_absolute_logit_shift": float(np.abs(reuse_logits - current_logits).mean()),
        },
    }
    if pro_path is not None:
        pro = sorted(
            (row for row in rows if row["path"] == pro_path),
            key=lambda row: row["request_id"],
        )
        if [row["request_id"] for row in pro] != [row["request_id"] for row in current]:
            raise RuntimeError("PRO and Current rolling request pairs are not aligned")
        pro_logits = np.asarray([row["hstu_logit"] for row in pro], dtype=np.float64)
        pro_metrics = binary_metrics(targets, pro_logits)
        report["absolute_metrics"][str(pro_path)] = pro_metrics
        report["PRO"] = {
            "path": str(pro_path),
            "plan": seal.get("pro_lazy_plan"),
            "PRO_minus_reuse_ROC_AUC_pp": (
                pro_metrics["ROC_AUC"] - reuse_metrics["ROC_AUC"]
            ) * 100,
            "PRO_minus_reuse_log_loss": (
                pro_metrics["log_loss"] - reuse_metrics["log_loss"]
            ),
            "current_minus_PRO_ROC_AUC_pp": (
                current_metrics["ROC_AUC"] - pro_metrics["ROC_AUC"]
            ) * 100,
            "PRO_minus_current_log_loss": (
                pro_metrics["log_loss"] - current_metrics["log_loss"]
            ),
            "mean_Bernoulli_JS_to_Current": float(
                bernoulli_js(pro_logits, current_logits).mean()
            ),
            "mean_absolute_logit_gap_to_Current": float(
                np.abs(pro_logits - current_logits).mean()
            ),
        }
    parent = sorted((row for row in rows if row["path"] == "parent_exact_rolling"), key=lambda row: row["request_id"])
    if parent:
        if [row["request_id"] for row in parent] != [row["request_id"] for row in current]:
            raise RuntimeError("Parent and Current rolling request pairs are not aligned")
        parent_logits = np.asarray([row["hstu_logit"] for row in parent], dtype=np.float64)
        parent_metrics = binary_metrics(targets, parent_logits)
        current_gain_pp = (current_metrics["ROC_AUC"] - parent_metrics["ROC_AUC"]) * 100
        reuse_harm_pp = report["reuse_minus_recompute"]["current_minus_reuse_ROC_AUC_pp"]
        log_loss_gain = parent_metrics["log_loss"] - current_metrics["log_loss"]
        reuse_auc_gain_pp = (reuse_metrics["ROC_AUC"] - parent_metrics["ROC_AUC"]) * 100
        report["absolute_metrics"]["parent_exact_rolling"] = parent_metrics
        report["release_debt_auc"] = {
            "comparison": "ParentExactRolling_vs_CurrentExactRolling_vs_OneHopReuseRolling",
            "current_minus_parent_ROC_AUC_pp": current_gain_pp,
            "current_minus_reuse_ROC_AUC_pp": reuse_harm_pp,
            "reuse_harm_over_current_gain_fraction": None if abs(current_gain_pp) < 1e-12 else reuse_harm_pp / current_gain_pp,
            "reuse_harm_over_current_gain_percent": None if current_gain_pp <= 0.0 else 100.0 * reuse_harm_pp / current_gain_pp,
            "reuse_gain_retained_percent": None if current_gain_pp <= 0.0 else 100.0 * reuse_auc_gain_pp / current_gain_pp,
            "interpretation": "percent is defined only for a positive current-versus-parent AUC release gain; fraction preserves the signed diagnostic ratio for every edge",
        }
        report["three_path_summary"] = {
            "requests": len(current),
            "old_parent": {
                "ROC_AUC": parent_metrics["ROC_AUC"],
                "log_loss": parent_metrics["log_loss"],
            },
            "new_current": {
                "ROC_AUC": current_metrics["ROC_AUC"],
                "log_loss": current_metrics["log_loss"],
            },
            "adjacent_one_hop_reuse": {
                "ROC_AUC": reuse_metrics["ROC_AUC"],
                "log_loss": reuse_metrics["log_loss"],
            },
            "current_minus_old_ROC_AUC_pp": current_gain_pp,
            "reuse_minus_old_ROC_AUC_pp": reuse_auc_gain_pp,
            "reuse_AUC_gain_retained_percent": (
                None if current_gain_pp <= 0.0 else 100.0 * reuse_auc_gain_pp / current_gain_pp
            ),
            "old_minus_current_log_loss": log_loss_gain,
            "old_minus_reuse_log_loss": parent_metrics["log_loss"] - reuse_metrics["log_loss"],
            "reuse_log_loss_gain_retained_percent": (
                None
                if log_loss_gain <= 0.0
                else 100.0 * (parent_metrics["log_loss"] - reuse_metrics["log_loss"])
                / log_loss_gain
            ),
        }
        if pro_path is not None:
            pro_metrics = report["absolute_metrics"][str(pro_path)]
            report["three_path_summary"]["PRO"] = {
                "path": str(pro_path),
                "ROC_AUC": pro_metrics["ROC_AUC"],
                "log_loss": pro_metrics["log_loss"],
                "AUC_gain_retained_percent": (
                    None
                    if current_gain_pp <= 0.0
                    else 100.0
                    * (pro_metrics["ROC_AUC"] - parent_metrics["ROC_AUC"])
                    / (current_metrics["ROC_AUC"] - parent_metrics["ROC_AUC"])
                ),
                "log_loss_gain_retained_percent": (
                    None
                    if log_loss_gain <= 0.0
                    else 100.0
                    * (parent_metrics["log_loss"] - pro_metrics["log_loss"])
                    / log_loss_gain
                ),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "edge": report["edge"], "event_logloss_reuse_minus_recompute": report["reuse_minus_recompute"]["event_weighted_log_loss"]}, indent=2))


if __name__ == "__main__":
    main()
