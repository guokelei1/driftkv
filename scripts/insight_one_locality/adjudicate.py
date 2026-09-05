#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insight_one_locality.common import (
    CONTRACT,
    EDGES,
    INPUT_MANIFEST,
    LOCALITY_CONFIGS,
    PATH_IDS,
    POPULATION,
    RESULT_ROOT,
    config_records,
    load_input_manifest,
    sha256_file,
)


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def bernoulli_js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    epsilon = 1e-12
    left = np.clip(left, epsilon, 1.0 - epsilon)
    right = np.clip(right, epsilon, 1.0 - epsilon)
    middle = 0.5 * (left + right)

    def kl(p, q):
        return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))

    return 0.5 * (kl(left, middle) + kl(right, middle))


def rank_correlation(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference_rank = np.argsort(np.argsort(reference, axis=1), axis=1).astype(np.float64)
    value_rank = np.argsort(np.argsort(values, axis=1), axis=1).astype(np.float64)
    reference_rank -= reference_rank.mean(axis=1, keepdims=True)
    value_rank -= value_rank.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.square(reference_rank).sum(axis=1) * np.square(value_rank).sum(axis=1)
    )
    return (reference_rank * value_rank).sum(axis=1) / np.maximum(denominator, 1e-12)


def top10_overlap(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.argpartition(reference, -10, axis=1)[:, -10:]
    observed = np.argpartition(values, -10, axis=1)[:, -10:]
    return np.asarray(
        [len(set(left.tolist()) & set(right.tolist())) / 10.0 for left, right in zip(ref, observed)],
        dtype=np.float64,
    )


def load_edge(raw: Path, edge: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    uids, scores, seals = [], [], []
    for rank in range(4):
        path = raw / f"rank{rank}" / f"{edge}.npz"
        with np.load(path, allow_pickle=False) as payload:
            if tuple(payload["path_ids"].tolist()) != PATH_IDS:
                raise RuntimeError(f"path IDs differ in {path}")
            uids.append(payload["uids"].astype(np.int64, copy=False))
            scores.append(payload["scores"].astype(np.float64, copy=False))
        seals.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    uid_array = np.concatenate(uids)
    score_array = np.concatenate(scores)
    order = np.argsort(uid_array)
    uid_array, score_array = uid_array[order], score_array[order]
    if len(uid_array) != POPULATION or len(np.unique(uid_array)) != POPULATION:
        raise RuntimeError(f"formal raw population is incomplete on {edge}")
    if score_array.shape != (POPULATION, len(PATH_IDS), 64):
        raise RuntimeError(f"formal raw score shape differs on {edge}: {score_array.shape}")
    return uid_array, score_array, seals


def metrics_for_path(
    edge: str,
    path_id: str,
    path_index: int,
    scores: np.ndarray,
    config_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reuse = scores[:, 0]
    exact = scores[:, 1]
    observed = scores[:, path_index]
    exact_probability = sigmoid(exact)
    observed_probability = sigmoid(observed)
    reuse_gap = np.abs(sigmoid(reuse) - exact_probability)
    gap = np.abs(observed_probability - exact_probability)
    denominator = float(reuse_gap.mean())
    if denominator <= 1e-12:
        raise RuntimeError(f"Reuse functional gap is numerically empty on {edge}")
    exact_top1 = exact.argmax(axis=1)
    config = config_by_id.get(path_id)
    return {
        "edge": edge,
        "config_id": path_id,
        "family": "anchor" if config is None else config["family"],
        "budget": path_id if config is None else config["budget"],
        "cost": 0.0 if path_id == "reuse" else 1.0 if path_id == "current_exact" else config["cost"],
        "users": len(scores),
        "candidates_per_user": scores.shape[2],
        "mean_abs_probability_gap": float(gap.mean()),
        "reuse_mean_abs_probability_gap": denominator,
        "probability_gap_recovery": float(1.0 - gap.mean() / denominator),
        "mean_abs_logit_gap": float(np.abs(observed - exact).mean()),
        "mean_Bernoulli_JS": float(bernoulli_js(observed_probability, exact_probability).mean()),
        "top1_agreement": float((observed.argmax(axis=1) == exact_top1).mean()),
        "top10_overlap": float(top10_overlap(exact, observed).mean()),
        "rank_correlation": float(rank_correlation(exact, observed).mean()),
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame[columns].itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RESULT_ROOT / "formal_raw")
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "analysis")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_name(args.output.name + ".partial").exists():
        raise FileExistsError(f"refusing to overwrite analysis: {args.output}")
    raw_summary_path = args.raw / "summary.json"
    raw_summary = json.loads(raw_summary_path.read_text(encoding="utf-8"))
    if (
        raw_summary.get("status") != "formal_raw_complete"
        or not raw_summary.get("passed")
        or raw_summary.get("contract_sha256") != sha256_file(CONTRACT)
    ):
        raise RuntimeError("formal raw output is incomplete or from another contract")
    _, manifest_uids, _, _ = load_input_manifest()
    records = config_records()
    config_by_id = {record["config_id"]: record for record in records}
    rows, raw_seals = [], []
    for edge in EDGES:
        uids, scores, seals = load_edge(args.raw, edge)
        if not np.array_equal(uids, np.sort(manifest_uids)):
            raise RuntimeError(f"formal raw UIDs differ from frozen manifest on {edge}")
        raw_seals.extend(seals)
        for path_index, path_id in enumerate(PATH_IDS):
            rows.append(metrics_for_path(edge, path_id, path_index, scores, config_by_id))
    metrics = pd.DataFrame(rows)
    locality = metrics[metrics.family != "anchor"].copy()
    best_indices = locality.groupby(["edge", "family", "budget"], sort=True).probability_gap_recovery.idxmax()
    best = locality.loc[best_indices].sort_values(["edge", "family", "cost"]).reset_index(drop=True)
    family_summary = (
        locality.groupby(["edge", "family", "budget", "cost"], sort=True)
        .agg(
            configs=("config_id", "size"),
            recovery_mean=("probability_gap_recovery", "mean"),
            recovery_min=("probability_gap_recovery", "min"),
            recovery_max=("probability_gap_recovery", "max"),
        )
        .reset_index()
    )
    edge_equal = (
        best.groupby(["family", "budget", "cost"], sort=True)
        .agg(
            edge_equal_best_observed_recovery=("probability_gap_recovery", "mean"),
            minimum_edge_recovery=("probability_gap_recovery", "min"),
            maximum_edge_recovery=("probability_gap_recovery", "max"),
        )
        .reset_index()
    )
    fixed_by_config = (
        locality.groupby(["family", "budget", "cost", "config_id"], sort=True)
        .probability_gap_recovery.mean()
        .reset_index(name="edge_equal_recovery")
    )
    fixed_winners = fixed_by_config.loc[
        fixed_by_config.groupby(["family", "budget"], sort=True).edge_equal_recovery.idxmax()
    ].sort_values(["family", "cost"])

    partial = args.output.with_name(args.output.name + ".partial")
    partial.mkdir(parents=True)
    metrics.to_csv(partial / "all_config_metrics.csv", index=False)
    best.to_csv(partial / "best_observed_by_edge.csv", index=False)
    family_summary.to_csv(partial / "family_mean_min_max.csv", index=False)
    edge_equal.to_csv(partial / "edge_equal_best_observed.csv", index=False)
    fixed_winners.to_csv(partial / "globally_fixed_config_winners.csv", index=False)
    summary = {
        "status": "medium_insight1_locality_analysis_complete",
        "contract_sha256": sha256_file(CONTRACT),
        "raw_summary_sha256": sha256_file(raw_summary_path),
        "input_manifest_sha256": sha256_file(INPUT_MANIFEST / "manifest.json"),
        "users": POPULATION,
        "edges": list(EDGES),
        "locality_configs": len(LOCALITY_CONFIGS),
        "raw_score_seals": raw_seals,
        "edge_equal_best_observed": edge_equal.to_dict("records"),
        "globally_fixed_winners": fixed_winners.to_dict("records"),
    }
    (partial / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# Medium Insight 1 locality diagnostic",
        "",
        "All five frozen D14 adjacent edges and all 34 pre-specified Exact-KV splice configurations are included. No request label is used.",
        "",
        "## Edge-equal best-observed frontier",
        "",
        *markdown_table(
            edge_equal,
            [
                "family",
                "budget",
                "cost",
                "edge_equal_best_observed_recovery",
                "minimum_edge_recovery",
                "maximum_edge_recovery",
            ],
        ),
        "",
        "## Globally fixed configuration winners",
        "",
        *markdown_table(
            fixed_winners,
            ["family", "budget", "cost", "config_id", "edge_equal_recovery"],
        ),
        "",
        "Exact-KV splices are optimistic diagnostic interventions, not dependency-closed migration actions. The x-axis is theoretical KV coverage, not GPU FLOPs or wall time.",
        "",
    ]
    (partial / "report.md").write_text("\n".join(report), encoding="utf-8")
    partial.rename(args.output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
