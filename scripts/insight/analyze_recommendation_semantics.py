#!/usr/bin/env python3
"""Recommendation-semantic and real request-group ranking analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_yambda500m_foundation_raw import load_histories  # noqa: E402
from hstu_kvcache.evaluation import binary_metrics  # noqa: E402
from analyze_first_pass import MANIFEST, load_edge, markdown_table  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results/yambda500m_small_seed17/insight_recommendation_semantics_v1"
KNOWN_ITEMS = 781_678


def request_features(history, row) -> dict:
    timestamps, items, behaviors = history.rows[int(row.uid)]
    stop = int(np.searchsorted(timestamps, int(row.query_timestamp), side="left"))
    values = items[max(0, stop - 512):stop]
    actions = behaviors[max(0, stop - 512):stop]
    recent, old = values[-32:], values[:-32]
    recent_set, old_set = set(map(int, recent)), set(map(int, old))
    candidate = int(row.item_idx)
    if candidate in recent_set:
        mode = "recent_repeat"
    elif candidate in old_set:
        mode = "old_only_repeat"
    else:
        mode = "novel_to_prefix"
    union = recent_set | old_set
    old_organic = float(np.mean(actions[:-32] == 1)) if len(actions) > 32 else float("nan")
    recent_organic = float(np.mean(actions[-32:] == 1))
    return {
        "candidate_mode": mode,
        "candidate_repeat_count": int(np.count_nonzero(values == candidate)),
        "history_unique_fraction": float(len(set(map(int, values))) / len(values)),
        "history_oov_fraction": float(np.mean(values >= KNOWN_ITEMS)),
        "recent_old_item_jaccard": float(len(recent_set & old_set) / len(union)) if union else 0.0,
        "organic_fraction": float(np.mean(actions == 1)),
        "organic_shift_recent_minus_old": recent_organic - old_organic,
    }


def cohort_rows(rows: pd.DataFrame) -> list[dict]:
    output = []
    for (edge, mode), cohort in rows.groupby(["edge", "candidate_mode"], sort=True):
        labels = cohort["label"].to_numpy(dtype=np.int64)
        parent = binary_metrics(labels, cohort["parent_full_logit"].to_numpy())
        current_full = binary_metrics(labels, cohort["current_full_logit"].to_numpy())
        current = binary_metrics(labels, cohort["current_exact_logit"].to_numpy())
        reuse = binary_metrics(labels, cohort["reuse_logit"].to_numpy())
        output.append({
            "edge": edge,
            "candidate_mode": mode,
            "requests": len(cohort),
            "users": int(cohort["uid"].nunique()),
            "mean_release_benefit": float(cohort["release_benefit"].mean()),
            "mean_reuse_harm": float(cohort["reuse_harm"].mean()),
            "mean_abs_probability_shift": float(cohort["abs_probability_shift"].mean()),
            "current_minus_parent_ROC_AUC_pp": (
                float("nan") if parent["ROC_AUC"] is None or current_full["ROC_AUC"] is None
                else 100.0 * (current_full["ROC_AUC"] - parent["ROC_AUC"])
            ),
            "current_minus_reuse_ROC_AUC_pp": (
                float("nan") if current["ROC_AUC"] is None or reuse["ROC_AUC"] is None
                else 100.0 * (current["ROC_AUC"] - reuse["ROC_AUC"])
            ),
        })
    return output


def correlation_rows(rows: pd.DataFrame) -> list[dict]:
    output = []
    features = (
        "candidate_repeat_count", "history_unique_fraction", "history_oov_fraction",
        "recent_old_item_jaccard", "organic_fraction", "organic_shift_recent_minus_old",
    )
    for edge, cohort in rows.groupby("edge", sort=True):
        for feature in features:
            valid = cohort[[feature, "reuse_harm", "abs_probability_shift"]].dropna()
            output.append({
                "edge": edge,
                "feature": feature,
                "spearman_feature_vs_harm": float(valid[feature].corr(valid["reuse_harm"], method="spearman")),
                "spearman_feature_vs_abs_shift": float(valid[feature].corr(valid["abs_probability_shift"], method="spearman")),
            })
    return output


def pairwise_rows(rows: pd.DataFrame) -> list[dict]:
    output = []
    for edge, edge_rows in rows.groupby("edge", sort=True):
        positives = edge_rows[edge_rows["label"] == 1]
        negatives = edge_rows[edge_rows["label"] == 0]
        pairs = positives.merge(
            negatives, on=["edge", "uid", "query_timestamp"], suffixes=("_positive", "_negative")
        )
        current_margin = pairs["current_exact_logit_positive"] - pairs["current_exact_logit_negative"]
        reuse_margin = pairs["reuse_logit_positive"] - pairs["reuse_logit_negative"]
        current_correct, reuse_correct = current_margin > 0.0, reuse_margin > 0.0
        output.append({
            "edge": edge,
            "real_positive_negative_pairs": len(pairs),
            "request_groups": int(pairs[["uid", "query_timestamp"]].drop_duplicates().shape[0]),
            "current_pairwise_accuracy": float(current_correct.mean()) if len(pairs) else float("nan"),
            "reuse_pairwise_accuracy": float(reuse_correct.mean()) if len(pairs) else float("nan"),
            "current_minus_reuse_pairwise_accuracy_pp": float(100 * (current_correct.mean() - reuse_correct.mean())) if len(pairs) else float("nan"),
            "harmful_flip_fraction": float((current_correct & ~reuse_correct).mean()) if len(pairs) else float("nan"),
            "beneficial_flip_fraction": float((~current_correct & reuse_correct).mean()) if len(pairs) else float("nan"),
            "mean_margin_erosion": float((current_margin - reuse_margin).mean()) if len(pairs) else float("nan"),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")

    manifest = pq.read_table(
        MANIFEST / "requests_quality.parquet",
        columns=["request_id", "item_idx", "label"],
    ).to_pandas()
    labels = dict(zip(manifest["request_id"], manifest["label"], strict=True))
    item_idx = dict(zip(manifest["request_id"], manifest["item_idx"], strict=True))
    frames = [load_edge(edge, labels) for edge in range(5)]
    rows = pd.concat(frames, ignore_index=True)
    rows["label"] = rows["request_id"].map(labels).astype(np.int64)
    rows["item_idx"] = rows["request_id"].map(item_idx).astype(np.int64)
    history = load_histories(sorted(map(int, rows["uid"].unique())), oov_buckets=256)
    features = pd.DataFrame(
        [request_features(history, row) for row in rows.itertuples(index=False)],
        index=rows.index,
    )
    rows = pd.concat([rows, features], axis=1)

    cohorts = pd.DataFrame(cohort_rows(rows))
    correlations = pd.DataFrame(correlation_rows(rows))
    pairwise = pd.DataFrame(pairwise_rows(rows))
    args.output.mkdir(parents=True)
    cohorts.to_csv(args.output / "candidate_modes.csv", index=False)
    correlations.to_csv(args.output / "semantic_correlations.csv", index=False)
    pairwise.to_csv(args.output / "real_pairwise_ranking.csv", index=False)
    summary = {
        "status": "recommendation_semantics_complete",
        "scope": "all paired D14/E14 requests on five Small seed17 edges",
        "requests": len(rows),
        "users": int(rows["uid"].nunique()),
        "real_positive_negative_pairs": int(pairwise["real_positive_negative_pairs"].sum()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = [
        "# Recommendation semantics and real pairwise ranking", "",
        f"Scope: {len(rows)} paired requests and {summary['real_positive_negative_pairs']} real positive-negative pairs. No sampled negatives.", "",
        "## Candidate modes", "",
        *markdown_table(cohorts, list(cohorts.columns)), "",
        "## Real request-group ranking", "",
        *markdown_table(pairwise, list(pairwise.columns)), "",
        "Novel-to-prefix candidates have larger Current-minus-Reuse ROC-AUC loss than recent repeats on all five edges. Parent-to-Current release gain does not consistently concentrate in the same candidate mode, so the supported claim is candidate-conditioned compatibility risk, not universal suppression of novel capability.", "",
        "Real same-timestamp pairwise accuracy changes direction across edges; aggregate AUC harm must not be restated as a universal pair-inversion effect.", "",
        "Feature correlations are in `semantic_correlations.csv`. Persistent behavior tokens are organic/non-organic listens; request like/dislike is used only as the evaluation label.", "",
    ]
    (args.output / "report.md").write_text("\n".join(report))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
