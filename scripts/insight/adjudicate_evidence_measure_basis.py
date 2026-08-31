#!/usr/bin/env python3
"""Adjudicate one sealed evidence-measure basis edge.

Canary mode is label-free and compares output fidelity.  Formal mode joins
labels only after the raw hash and Design-0 replay have been validated.
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
    EVIDENCE_MEASURE_PATHS,
    sha256_file,
    validate_pair_raw,
)
from hstu_kvcache.evaluation import bernoulli_js, binary_metrics, stable_log_loss  # noqa: E402
from insight.one_release_refinement import EVIDENCE_MEASURE_PATH, OUR_PATH  # noqa: E402


def _by_path(rows: list[dict]) -> dict[str, list[dict]]:
    observed = sorted({str(row["path"]) for row in rows})
    return {
        path: sorted((row for row in rows if row["path"] == path), key=lambda row: row["request_id"])
        for path in observed
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--prior-design0-raw", type=Path, required=True)
    parser.add_argument("--prior-design0-sha256", required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--require-exact-request-set", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    seal = json.loads(args.seal.read_text())
    if seal["raw_sha256"] != sha256_file(args.raw):
        raise RuntimeError("evidence-measure raw differs from its pre-label seal")
    if seal.get("contains_evidence_measure_basis") is not True:
        raise RuntimeError("raw seal does not declare the evidence-measure path")
    if sha256_file(args.prior_design0_raw) != args.prior_design0_sha256:
        raise RuntimeError("Design-0 raw differs from the prospective contract")

    raw = pq.read_table(args.raw)
    expected = EVIDENCE_MEASURE_PATHS if seal.get("contains_parent_exact_rolling") else (
        "current_exact_rolling",
        "one_hop_reuse_rolling",
        OUR_PATH,
        EVIDENCE_MEASURE_PATH,
    )
    validate_pair_raw(raw, expected_paths=expected)
    if "label" in raw.column_names:
        raise RuntimeError("raw artifact contains a label")
    paths = _by_path(raw.to_pylist())
    request_ids = [row["request_id"] for row in paths["current_exact_rolling"]]
    if any([row["request_id"] for row in rows] != request_ids for rows in paths.values()):
        raise RuntimeError("evidence-measure paths are not request-aligned")

    prior = _by_path(pq.read_table(args.prior_design0_raw).to_pylist())
    prior_ids = {row["request_id"] for row in prior["current_exact_rolling"]}
    if set(request_ids) - prior_ids:
        raise RuntimeError("evaluation contains requests absent from Design 0")
    if args.require_exact_request_set and set(request_ids) != prior_ids:
        raise RuntimeError("formal request set differs from Design 0")
    prior_maps = {
        path: {row["request_id"]: row for row in prior[path]}
        for path in ("current_exact_rolling", "one_hop_reuse_rolling", OUR_PATH)
    }
    if "parent_exact_rolling" in paths:
        prior_maps["parent_exact_rolling"] = {
            row["request_id"]: row for row in prior["parent_exact_rolling"]
        }
    replay_errors = {
        path: max(
            abs(float(row["hstu_logit"]) - float(prior_maps[path][row["request_id"]]["hstu_logit"]))
            for row in paths[path]
        )
        for path in prior_maps
    }
    if max(replay_errors.values()) > 2e-5:
        raise RuntimeError(f"baseline replay differs from Design 0: {replay_errors}")

    logits = {
        path: np.asarray([float(row["hstu_logit"]) for row in rows], dtype=np.float64)
        for path, rows in paths.items()
    }
    current = logits["current_exact_rolling"]
    current_probability = 1.0 / (1.0 + np.exp(-current))
    fidelity = {}
    for path, values in logits.items():
        probability = 1.0 / (1.0 + np.exp(-values))
        fidelity[path] = {
            "mean_abs_logit_gap_to_current": float(np.mean(np.abs(values - current))),
            "mean_abs_probability_gap_to_current": float(np.mean(np.abs(probability - current_probability))),
            "mean_readout_normalized_l2": float(
                np.mean([float(row["readout_normalized_l2"]) for row in paths[path]])
            ),
        }

    report = {
        "status": "evidence_measure_basis_canary_adjudicated" if args.labels is None else "evidence_measure_basis_formal_adjudicated",
        "edge": seal["edge"],
        "stage": seal["stage"],
        "requests": len(request_ids),
        "users": len({int(row["uid"]) for row in paths["current_exact_rolling"]}),
        "raw_sha256": seal["raw_sha256"],
        "Design0_raw_sha256": args.prior_design0_sha256,
        "baseline_replay_max_absolute_logit_error": replay_errors,
        "labels_joined_after_raw_seal": args.labels is not None,
        "fidelity": fidelity,
        "basis_fidelity_not_worse_than_Design0": (
            fidelity[EVIDENCE_MEASURE_PATH]["mean_abs_logit_gap_to_current"]
            <= fidelity[OUR_PATH]["mean_abs_logit_gap_to_current"]
        ),
    }

    if args.labels is not None:
        labels = pq.read_table(args.labels, columns=["request_id", "label"])
        label_map = dict(zip(
            labels["request_id"].to_pylist(), labels["label"].to_pylist(), strict=True
        ))
        targets = np.asarray([label_map[request_id] for request_id in request_ids], dtype=np.int64)
        metrics = {path: binary_metrics(targets, values) for path, values in logits.items()}
        user_ids = np.asarray([int(row["uid"]) for row in paths["current_exact_rolling"]])
        user_equal = {}
        for path, values in logits.items():
            event_delta = stable_log_loss(values, targets) - stable_log_loss(current, targets)
            per_user = [float(event_delta[user_ids == uid].mean()) for uid in np.unique(user_ids)]
            user_equal[path] = float(np.mean(per_user))
        report.update({
            "absolute_metrics": metrics,
            "fidelity_companions": {
                path: {
                    "event_log_loss_delta_to_current": float(
                        (stable_log_loss(values, targets) - stable_log_loss(current, targets)).mean()
                    ),
                    "user_equal_log_loss_delta_to_current": user_equal[path],
                    "mean_Bernoulli_JS_to_current": float(bernoulli_js(values, current).mean()),
                }
                for path, values in logits.items()
            },
            "formal_gate": {
                "basis_ROC_AUC_not_below_Design0": (
                    metrics[EVIDENCE_MEASURE_PATH]["ROC_AUC"] >= metrics[OUR_PATH]["ROC_AUC"]
                ),
                "basis_log_loss_not_above_Design0": (
                    metrics[EVIDENCE_MEASURE_PATH]["log_loss"] <= metrics[OUR_PATH]["log_loss"]
                ),
            },
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "edge": report["edge"],
        "requests": report["requests"],
        "basis_fidelity_not_worse_than_Design0": report["basis_fidelity_not_worse_than_Design0"],
        "formal_gate": report.get("formal_gate"),
    }, indent=2))


if __name__ == "__main__":
    main()
