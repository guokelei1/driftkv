#!/usr/bin/env python3
"""Join labels only after seal and adjudicate real-exposed signed interventions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from hstu_kvcache.evaluation import binary_metrics, stable_log_loss  # noqa: E402
from insight.evaluate_candidate_shared_exposed_raw import RAW_PATHS, sha256  # noqa: E402


def user_equal_mean(uids: np.ndarray, values: np.ndarray) -> float:
    frame = pd.DataFrame({"uid": uids, "value": values})
    return float(frame.groupby("uid", sort=False).value.mean().mean())


def pairwise_accuracy(frame: pd.DataFrame) -> tuple[int, int, float | None]:
    correct = 0
    pairs = 0
    banks = 0
    for _, group in frame.groupby("bank_id", sort=False):
        positive = group[group.label == 1].hstu_logit.to_numpy(dtype=np.float64)
        negative = group[group.label == 0].hstu_logit.to_numpy(dtype=np.float64)
        if not len(positive) or not len(negative):
            continue
        margins = positive[:, None] - negative[None, :]
        correct += int((margins > 0.0).sum())
        pairs += int(margins.size)
        banks += 1
    return banks, pairs, None if not pairs else correct / pairs


def metric_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    output = []
    for (edge, width, path), group in rows.groupby(
        ["edge", "width", "path"], sort=True
    ):
        labels = group.label.to_numpy(dtype=np.int64)
        logits = group.hstu_logit.to_numpy(dtype=np.float64)
        metrics = binary_metrics(labels, logits)
        banks, pairs, pair_accuracy = pairwise_accuracy(group)
        output.append(
            {
                "edge": edge,
                "width": int(width),
                "path": path,
                "requests": len(group),
                "users": int(group.uid.nunique()),
                "banks": int(group.bank_id.nunique()),
                **metrics,
                "mixed_label_banks": banks,
                "positive_negative_pairs": pairs,
                "within_bank_pairwise_accuracy": pair_accuracy,
            }
        )
    return output


def paired_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    index = ["edge", "width", "bank_id", "request_id", "uid", "label"]
    pivot = rows.pivot(index=index, columns="path", values="hstu_logit").reset_index()
    output = []
    exact = pivot["current_exact"].to_numpy(dtype=np.float64)
    labels = pivot.label.to_numpy(dtype=np.int64)
    for (edge, width), positions in pivot.groupby(["edge", "width"], sort=True).groups.items():
        selected = np.asarray(list(positions), dtype=np.int64)
        selected_labels = labels[selected]
        exact_logits = exact[selected]
        for path in RAW_PATHS:
            values = pivot[path].to_numpy(dtype=np.float64)[selected]
            loss_delta = stable_log_loss(values, selected_labels) - stable_log_loss(
                exact_logits, selected_labels
            )
            output.append(
                {
                    "edge": edge,
                    "width": int(width),
                    "path": path,
                    "mean_abs_logit_gap_to_exact": float(
                        np.mean(np.abs(values - exact_logits))
                    ),
                    "event_path_minus_exact_log_loss": float(loss_delta.mean()),
                    "user_equal_path_minus_exact_log_loss": user_equal_mean(
                        pivot.uid.to_numpy(dtype=np.int64)[selected], loss_delta
                    ),
                    "positive_request_fraction_path_worse": float(
                        np.mean(loss_delta > 0.0)
                    ),
                }
            )
    return output


def render_report(metrics: pd.DataFrame, paired: pd.DataFrame, seal: dict) -> str:
    focus = metrics[
        metrics.path.isin(["current_exact", "reuse", "shared_only", "residual_only"])
    ][
        [
            "edge",
            "width",
            "path",
            "requests",
            "users",
            "ROC_AUC",
            "log_loss",
            "within_bank_pairwise_accuracy",
        ]
    ]
    gap = paired[
        paired.path.isin(["reuse", "shared_only", "residual_only"])
    ][
        [
            "edge",
            "width",
            "path",
            "mean_abs_logit_gap_to_exact",
            "event_path_minus_exact_log_loss",
            "user_equal_path_minus_exact_log_loss",
        ]
    ]

    def table(frame: pd.DataFrame) -> list[str]:
        columns = list(frame.columns)
        output = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in frame.itertuples(index=False):
            values = []
            for value in row:
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    values.append("N/A")
                elif isinstance(value, float):
                    values.append(f"{value:.8g}")
                else:
                    values.append(str(value))
            output.append("| " + " | ".join(values) + " |")
        return output

    return "\n".join(
        [
            f"# Real-exposed signed causal quality: {seal['edge']}",
            "",
            "Raw rolling scores were sealed before labels were joined. Candidate groups are real same-UID, same-timestamp requests; no sampled negatives are introduced.",
            "",
            "## Absolute quality by candidate width",
            "",
            *table(focus),
            "",
            "## Paired fidelity and log-loss delta to Current Exact",
            "",
            *table(gap),
            "",
            "The shared/residual paths are oracle causal interventions, not executable migration actions.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite adjudication: {args.output_dir}")
    seal = json.loads(args.seal.read_text())
    if seal["status"] != "candidate_shared_real_exposed_raw_sealed_before_label_join":
        raise RuntimeError("unexpected exposed raw seal status")
    if seal["artifacts"]["raw"]["sha256"] != sha256(args.raw):
        raise RuntimeError("real-exposed raw differs from its pre-label seal")
    if seal["native_score_max_abs_error"] > 2e-5:
        raise RuntimeError("native signed trace exceeded the correctness threshold")
    if seal["full_delta_reconstruction_max_abs_error"] > 2e-5:
        raise RuntimeError("full signed delta failed to reconstruct Current Exact")

    rows = pq.read_table(args.raw).to_pandas()
    if "label" in rows.columns:
        raise RuntimeError("raw real-exposed artifact already contains labels")
    labels = pq.read_table(args.labels, columns=["request_id", "label"]).to_pandas()
    if labels.request_id.duplicated().any():
        raise RuntimeError("quality manifest contains duplicate request ids")
    label_map = labels.set_index("request_id").label
    missing = set(rows.request_id) - set(label_map.index)
    if missing:
        raise RuntimeError(f"{len(missing)} sealed request ids are absent from labels")
    rows["label"] = rows.request_id.map(label_map).astype(np.int64)
    if not rows.label.isin([0, 1]).all():
        raise RuntimeError("joined exposed labels are not binary")

    metrics = pd.DataFrame(metric_rows(rows))
    paired = pd.DataFrame(paired_rows(rows))
    args.output_dir.mkdir(parents=True)
    metrics.to_csv(args.output_dir / "quality_by_width.csv", index=False)
    paired.to_csv(args.output_dir / "paired_fidelity.csv", index=False)
    summary = {
        "status": "candidate_shared_real_exposed_adjudicated",
        "edge": seal["edge"],
        "raw_sha256": sha256(args.raw),
        "users": seal["users"],
        "banks_across_widths": seal["banks_across_widths"],
        "selected_requests_across_widths": seal["selected_requests_across_widths"],
        "widths": seal["widths"],
        "labels_joined_after_raw_seal": True,
    }
    (args.output_dir / "adjudication.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(render_report(metrics, paired, seal))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
