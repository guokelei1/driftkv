#!/usr/bin/env python3
"""Adjudicate the sealed full-population one-release refinement raw artifact."""

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
    REFINEMENT_PATHS, sha256_file, validate_pair_raw,
)
from hstu_kvcache.evaluation import (  # noqa: E402
    bernoulli_js, binary_metrics, stable_log_loss,
)
from insight.one_release_refinement import OUR_PATH  # noqa: E402


def _rows_by_path(rows: list[dict], path: str) -> list[dict]:
    return sorted((row for row in rows if row["path"] == path), key=lambda row: row["request_id"])


def _full_only_gain(path: Path, edge: str) -> float:
    rows = json.loads(path.read_text())
    matches = [
        row for row in rows
        if row["edge"] == edge and int(row["evaluation_days"]) == 14
    ]
    if len(matches) != 1:
        raise RuntimeError(f"missing unique E14 Full-only release gain for {edge}")
    return float(matches[0]["full_only_current_minus_parent_ROC_AUC_pp"]) / 100.0


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if abs(denominator) < 1e-12 else numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--prior-raw", type=Path, required=True)
    parser.add_argument("--prior-raw-sha256", required=True)
    parser.add_argument("--full-only-gain-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-exact-request-set", action="store_true")
    args = parser.parse_args()

    seal = json.loads(args.seal.read_text())
    if seal["raw_sha256"] != sha256_file(args.raw):
        raise RuntimeError("new raw artifact differs from its pre-label seal")
    if sha256_file(args.prior_raw) != args.prior_raw_sha256:
        raise RuntimeError("prior rolling raw differs from the prospective contract")

    raw = pq.read_table(args.raw)
    validate_pair_raw(raw, expected_paths=REFINEMENT_PATHS)
    rows = raw.to_pylist()
    paths = {path: _rows_by_path(rows, path) for path in REFINEMENT_PATHS}
    request_ids = [row["request_id"] for row in paths["current_exact_rolling"]]
    if any([row["request_id"] for row in values] != request_ids for values in paths.values()):
        raise RuntimeError("new rolling paths are not request-aligned")

    prior_rows = pq.read_table(args.prior_raw).to_pylist()
    prior = {
        path: {row["request_id"]: row for row in prior_rows if row["path"] == path}
        for path in ("current_exact_rolling", "one_hop_reuse_rolling")
    }
    prior_ids = set(prior["current_exact_rolling"])
    if set(request_ids) - prior_ids:
        raise RuntimeError("new evaluation contains requests absent from the sealed prior raw")
    if args.require_exact_request_set and set(request_ids) != prior_ids:
        raise RuntimeError("formal evaluation request set differs from the sealed E14 baseline")
    baseline_errors = {}
    for path in prior:
        baseline_errors[path] = max(
            abs(float(row["hstu_logit"]) - float(prior[path][row["request_id"]]["hstu_logit"]))
            for row in paths[path]
        )
    if max(baseline_errors.values()) > 2e-5:
        raise RuntimeError(f"baseline replay differs from sealed raw: {baseline_errors}")

    labels = pq.read_table(args.labels, columns=["request_id", "label"])
    label_map = dict(zip(
        labels["request_id"].to_pylist(), labels["label"].to_pylist(), strict=True
    ))
    targets = np.asarray([label_map[request_id] for request_id in request_ids], dtype=np.int64)
    logits = {
        path: np.asarray([float(row["hstu_logit"]) for row in values], dtype=np.float64)
        for path, values in paths.items()
    }
    metrics = {path: binary_metrics(targets, values) for path, values in logits.items()}
    auc = {path: values["ROC_AUC"] for path, values in metrics.items()}

    current_path = "current_exact_rolling"
    reuse_path = "one_hop_reuse_rolling"
    parent_path = "parent_exact_rolling"
    current_auc, reuse_auc = auc[current_path], auc[reuse_path]
    parent_auc, our_auc = auc[parent_path], auc[OUR_PATH]
    full_only_gain = _full_only_gain(args.full_only_gain_table, seal["edge"])
    implied_old_auc = current_auc - full_only_gain
    matched_gain = current_auc - parent_auc
    reuse_harm = current_auc - reuse_auc
    our_harm = current_auc - our_auc

    report = {
        "status": "one_release_refinement_auc_adjudicated",
        "raw_sha256": seal["raw_sha256"],
        "prior_raw_sha256": args.prior_raw_sha256,
        "stage": seal["stage"],
        "edge": seal["edge"],
        "evaluation_day_range": seal["evaluation_day_range"],
        "requests": len(request_ids),
        "baseline_replay_max_absolute_logit_error": baseline_errors,
        "absolute_metrics": metrics,
        "requested_full_only_reference_ratio": {
            "semantics": "same pre-existing D14 Full-only release-gain denominator plus matched rolling path deltas",
            "full_only_current_minus_parent_ROC_AUC_pp": full_only_gain * 100.0,
            "implied_old_ROC_AUC_on_rolling_axis": implied_old_auc,
            "reuse_gain_retained_fraction": _safe_ratio(reuse_auc - implied_old_auc, full_only_gain),
            "our_gain_retained_fraction": _safe_ratio(our_auc - implied_old_auc, full_only_gain),
            "reuse_gain_retained_percent": None if full_only_gain <= 0 else 100.0 * (reuse_auc - implied_old_auc) / full_only_gain,
            "our_gain_retained_percent": None if full_only_gain <= 0 else 100.0 * (our_auc - implied_old_auc) / full_only_gain,
        },
        "rolling_auc_deltas": {
            "current_minus_reuse_ROC_AUC_pp": reuse_harm * 100.0,
            "current_minus_our_ROC_AUC_pp": our_harm * 100.0,
            "our_minus_reuse_ROC_AUC_pp": (our_auc - reuse_auc) * 100.0,
            "reuse_harm_recovered_fraction": _safe_ratio(our_auc - reuse_auc, reuse_harm),
        },
        "matched_rolling_ratio": {
            "current_minus_parent_ROC_AUC_pp": matched_gain * 100.0,
            "reuse_gain_retained_fraction": _safe_ratio(reuse_auc - parent_auc, matched_gain),
            "our_gain_retained_fraction": _safe_ratio(our_auc - parent_auc, matched_gain),
            "interpretation": "reported as a companion; only positive, nontrivial Parent-to-Current gain supports a retained-gain percentage",
        },
        "fidelity_companions": {
            "reuse_minus_current_event_log_loss": float(
                (stable_log_loss(logits[reuse_path], targets) - stable_log_loss(logits[current_path], targets)).mean()
            ),
            "our_minus_current_event_log_loss": float(
                (stable_log_loss(logits[OUR_PATH], targets) - stable_log_loss(logits[current_path], targets)).mean()
            ),
            "reuse_mean_Bernoulli_JS_to_current": float(
                bernoulli_js(logits[reuse_path], logits[current_path]).mean()
            ),
            "our_mean_Bernoulli_JS_to_current": float(
                bernoulli_js(logits[OUR_PATH], logits[current_path]).mean()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "edge": report["edge"],
        "requests": report["requests"],
        "our_gain_retained_percent": report["requested_full_only_reference_ratio"]["our_gain_retained_percent"],
    }, indent=2))


if __name__ == "__main__":
    main()
