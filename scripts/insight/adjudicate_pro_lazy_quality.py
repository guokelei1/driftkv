#!/usr/bin/env python3
"""Adjudicate one frozen lightweight-PRO rolling edge.

Canary mode is label-free.  Formal mode verifies both the new raw seal and the
frozen Design-0 replay before it opens the separately frozen label artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_yambda500m_hstu_native_onehop_reuse_raw import (  # noqa: E402
    PRO_LAZY_PATHS,
    REFINEMENT_PATHS,
    sha256_file,
    validate_pair_raw,
)
from hstu_kvcache.evaluation import (  # noqa: E402
    bernoulli_js,
    binary_metrics,
    stable_log_loss,
)
from insight.one_release_refinement import OUR_PATH  # noqa: E402
from insight.pro_lazy_reader import PRO_PATH  # noqa: E402


BASELINE_PATHS = (
    "parent_exact_rolling",
    "current_exact_rolling",
    "one_hop_reuse_rolling",
)


def _by_path(rows: list[dict]) -> dict[str, list[dict]]:
    return {
        path: sorted(
            (row for row in rows if row["path"] == path),
            key=lambda row: row["request_id"],
        )
        for path in sorted({str(row["path"]) for row in rows})
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if abs(denominator) < 1e-12 else numerator / denominator


def _full_only_gain(path: Path, edge: str) -> float:
    rows = json.loads(path.read_text())
    matched = [
        row for row in rows
        if row["edge"] == edge and int(row["evaluation_days"]) == 14
    ]
    if len(matched) != 1:
        raise RuntimeError(f"missing unique E14 Full-only release gain for {edge}")
    return float(matched[0]["full_only_current_minus_parent_ROC_AUC_pp"]) / 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--prior-design0-raw", type=Path, required=True)
    parser.add_argument("--prior-design0-sha256", required=True)
    parser.add_argument("--full-only-gain-table", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--require-exact-request-set", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    seal = json.loads(args.seal.read_text())
    if seal["raw_sha256"] != sha256_file(args.raw):
        raise RuntimeError("PRO raw differs from its pre-label seal")
    if seal.get("contains_pro_lazy") is not True:
        raise RuntimeError("raw seal does not declare the frozen PRO path")
    plan = seal.get("pro_lazy_plan", {})
    if (
        plan.get("repair_width"),
        plan.get("carriers"),
        plan.get("represented_mass"),
        plan.get("materialized_translated_prefix_positions"),
        plan.get("underfull_rule"),
    ) != (128, 32, 4, 0, "reuse"):
        raise RuntimeError("raw seal PRO plan differs from the frozen quality contract")
    if sha256_file(args.prior_design0_raw) != args.prior_design0_sha256:
        raise RuntimeError("Design-0 raw differs from the prospective contract")

    raw = pq.read_table(args.raw)
    if "label" in raw.column_names:
        raise RuntimeError("raw artifact contains a label")
    validate_pair_raw(raw, expected_paths=PRO_LAZY_PATHS)
    paths = _by_path(raw.to_pylist())
    request_ids = [row["request_id"] for row in paths["current_exact_rolling"]]
    if any([row["request_id"] for row in paths[path]] != request_ids for path in PRO_LAZY_PATHS):
        raise RuntimeError("PRO paths are not request-aligned")

    prior_raw = pq.read_table(args.prior_design0_raw)
    validate_pair_raw(prior_raw, expected_paths=REFINEMENT_PATHS)
    prior = _by_path(prior_raw.to_pylist())
    prior_ids = {row["request_id"] for row in prior["current_exact_rolling"]}
    if set(request_ids) - prior_ids:
        raise RuntimeError("PRO evaluation contains requests absent from frozen Design 0")
    if args.require_exact_request_set and set(request_ids) != prior_ids:
        raise RuntimeError("formal PRO request set differs from frozen Design 0")
    prior_maps = {
        path: {row["request_id"]: row for row in prior[path]}
        for path in REFINEMENT_PATHS
    }
    replay_errors = {
        path: max(
            abs(
                float(row["hstu_logit"])
                - float(prior_maps[path][row["request_id"]]["hstu_logit"])
            )
            for row in paths[path]
        )
        for path in BASELINE_PATHS
    }
    if max(replay_errors.values()) > 2e-5:
        raise RuntimeError(f"baseline replay differs from frozen Design 0: {replay_errors}")

    logits = {
        path: np.asarray([float(row["hstu_logit"]) for row in rows], dtype=np.float64)
        for path, rows in paths.items()
    }
    logits[OUR_PATH] = np.asarray(
        [float(prior_maps[OUR_PATH][request_id]["hstu_logit"]) for request_id in request_ids],
        dtype=np.float64,
    )
    current = logits["current_exact_rolling"]
    current_probability = 1.0 / (1.0 + np.exp(-current))
    fidelity = {}
    for path, values in logits.items():
        probability = 1.0 / (1.0 + np.exp(-values))
        fidelity[path] = {
            "mean_abs_logit_gap_to_Current": float(np.mean(np.abs(values - current))),
            "mean_abs_probability_gap_to_Current": float(
                np.mean(np.abs(probability - current_probability))
            ),
            "mean_Bernoulli_JS_to_Current": float(bernoulli_js(values, current).mean()),
        }

    report = {
        "status": (
            "pro_lazy_quality_canary_adjudicated"
            if args.labels is None
            else "pro_lazy_rolling_quality_adjudicated"
        ),
        "edge": seal["edge"],
        "stage": seal["stage"],
        "requests": len(request_ids),
        "users": len({int(row["uid"]) for row in paths["current_exact_rolling"]}),
        "raw_sha256": seal["raw_sha256"],
        "Design0_raw_sha256": args.prior_design0_sha256,
        "labels_joined_after_raw_seal": args.labels is not None,
        "request_set_exactly_matches_Design0": set(request_ids) == prior_ids,
        "baseline_replay_max_absolute_logit_error": replay_errors,
        "structural_checks": {
            "materialized_translated_prefix_positions": 0,
            "carriers": 32,
            "represented_mass": 4,
            "underfull_rule": "reuse",
        },
        "fidelity": fidelity,
    }

    if args.labels is not None:
        labels = pq.read_table(args.labels, columns=["request_id", "label"])
        label_map = dict(zip(
            labels["request_id"].to_pylist(), labels["label"].to_pylist(), strict=True
        ))
        targets = np.asarray([label_map[request_id] for request_id in request_ids], dtype=np.int64)
        metrics = {path: binary_metrics(targets, values) for path, values in logits.items()}
        user_ids = np.asarray(
            [int(row["uid"]) for row in paths["current_exact_rolling"]], dtype=np.int64
        )
        user_equal_log_loss_delta = {}
        for path, values in logits.items():
            event_delta = stable_log_loss(values, targets) - stable_log_loss(current, targets)
            user_equal_log_loss_delta[path] = float(np.mean([
                float(event_delta[user_ids == uid].mean()) for uid in np.unique(user_ids)
            ]))

        auc = {path: float(value["ROC_AUC"]) for path, value in metrics.items()}
        log_loss = {path: float(value["log_loss"]) for path, value in metrics.items()}
        parent_path = "parent_exact_rolling"
        current_path = "current_exact_rolling"
        reuse_path = "one_hop_reuse_rolling"
        full_only_gain = _full_only_gain(args.full_only_gain_table, seal["edge"])
        implied_old_auc = auc[current_path] - full_only_gain
        reuse_harm = auc[current_path] - auc[reuse_path]
        matched_gain = auc[current_path] - auc[parent_path]
        report.update({
            "absolute_metrics": metrics,
            "user_equal_log_loss_delta_to_Current": user_equal_log_loss_delta,
            "requested_full_only_release_gain_retention": {
                "full_only_Current_minus_Parent_ROC_AUC_pp": full_only_gain * 100.0,
                "implied_old_ROC_AUC_on_rolling_axis": implied_old_auc,
                "Reuse_fraction": _safe_ratio(auc[reuse_path] - implied_old_auc, full_only_gain),
                "Design0_fraction": _safe_ratio(auc[OUR_PATH] - implied_old_auc, full_only_gain),
                "PRO_fraction": _safe_ratio(auc[PRO_PATH] - implied_old_auc, full_only_gain),
                "Reuse_percent_when_positive": (
                    None if full_only_gain <= 0
                    else 100.0 * (auc[reuse_path] - implied_old_auc) / full_only_gain
                ),
                "Design0_percent_when_positive": (
                    None if full_only_gain <= 0
                    else 100.0 * (auc[OUR_PATH] - implied_old_auc) / full_only_gain
                ),
                "PRO_percent_when_positive": (
                    None if full_only_gain <= 0
                    else 100.0 * (auc[PRO_PATH] - implied_old_auc) / full_only_gain
                ),
            },
            "rolling_quality_deltas": {
                "Current_minus_Reuse_ROC_AUC_pp": 100.0 * reuse_harm,
                "PRO_minus_Reuse_ROC_AUC_pp": 100.0 * (auc[PRO_PATH] - auc[reuse_path]),
                "PRO_minus_Design0_ROC_AUC_pp": 100.0 * (auc[PRO_PATH] - auc[OUR_PATH]),
                "Current_minus_PRO_ROC_AUC_pp": 100.0 * (auc[current_path] - auc[PRO_PATH]),
                "Reuse_harm_recovered_fraction": _safe_ratio(
                    auc[PRO_PATH] - auc[reuse_path], reuse_harm
                ),
                "PRO_minus_Reuse_log_loss": log_loss[PRO_PATH] - log_loss[reuse_path],
                "PRO_minus_Design0_log_loss": log_loss[PRO_PATH] - log_loss[OUR_PATH],
                "PRO_minus_Current_log_loss": log_loss[PRO_PATH] - log_loss[current_path],
            },
            "matched_rolling_release_gain": {
                "Current_minus_Parent_ROC_AUC_pp": 100.0 * matched_gain,
                "Reuse_retained_fraction": _safe_ratio(
                    auc[reuse_path] - auc[parent_path], matched_gain
                ),
                "Design0_retained_fraction": _safe_ratio(
                    auc[OUR_PATH] - auc[parent_path], matched_gain
                ),
                "PRO_retained_fraction": _safe_ratio(
                    auc[PRO_PATH] - auc[parent_path], matched_gain
                ),
            },
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "edge": report["edge"],
        "requests": report["requests"],
        "labels_joined": report["labels_joined_after_raw_seal"],
        "rolling_quality_deltas": report.get("rolling_quality_deltas"),
    }, indent=2))


if __name__ == "__main__":
    main()
