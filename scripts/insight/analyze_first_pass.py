#!/usr/bin/env python3
"""Analyze dilution and release-benefit/Reuse-harm overlap from sealed raw."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hstu_kvcache.evaluation import bernoulli_js, stable_log_loss


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3"
MANIFEST = ROOT / "data/manifests/yambda500m_small_hstu_native_rolling_matrix_fast_v3"
DEFAULT_OUTPUT = ROOT / "results/yambda500m_small_seed17/insight_first_pass_v1"
DAY = 86_400


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed_raw(directory: Path) -> Path:
    raw = directory / "raw.parquet"
    seal_path = directory / "raw.seal.json"
    if not raw.exists() or not seal_path.exists():
        raise FileNotFoundError(f"missing sealed raw in {directory}")
    seal = json.loads(seal_path.read_text())
    if seal.get("raw_sha256") != sha256(raw):
        raise RuntimeError(f"raw hash differs from seal: {raw}")
    return raw


def paths_for_edge(edge: int) -> tuple[Path, Path]:
    name = f"v{edge}_to_v{edge + 1}"
    full = MATRIX / "train_14d/eval_14d" / name
    reuse_run = "d14_onehop_reuse_diagnostic_v1" if edge < 2 else "d14_onehop_reuse_completion_v2"
    reuse = MATRIX / reuse_run / "eval_14d" / name
    return sealed_raw(full), sealed_raw(reuse)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def append_bucket(value: int) -> str:
    if value <= 1:
        return str(value)
    for low, high in ((2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127)):
        if value <= high:
            return f"{low}-{high}"
    return ">=128"


def remaining_bucket(value: float) -> str:
    if value <= 0.0:
        return "0"
    if value <= 0.25:
        return "(0,.25]"
    if value <= 0.50:
        return "(.25,.5]"
    if value <= 0.75:
        return "(.5,.75]"
    if value < 1.0:
        return "(.75,1)"
    return "1"


def load_edge(edge: int, labels: dict[str, int]) -> pd.DataFrame:
    full_path, reuse_path = paths_for_edge(edge)
    full = pq.read_table(full_path).to_pandas()
    reuse = pq.read_table(reuse_path).to_pandas()

    parent = full.loc[full["is_parent"], ["request_id", "hstu_logit"]].rename(
        columns={"hstu_logit": "parent_full_logit"}
    )
    current_full = full.loc[~full["is_parent"], ["request_id", "hstu_logit"]].rename(
        columns={"hstu_logit": "current_full_logit"}
    )
    metadata = [
        "request_id", "uid", "query_timestamp", "seconds_since_cutover",
        "append_count_since_cutover", "history_length", "cache_length", "rolling_evictions",
    ]
    current = reuse.loc[
        reuse["path"] == "current_exact_rolling", metadata + ["hstu_logit"]
    ].rename(columns={"hstu_logit": "current_exact_logit"})
    stale = reuse.loc[
        reuse["path"] == "one_hop_reuse_rolling", ["request_id", "hstu_logit"]
    ].rename(columns={"hstu_logit": "reuse_logit"})

    rows = parent.merge(current_full, on="request_id", validate="one_to_one")
    rows = rows.merge(current, on="request_id", validate="one_to_one")
    rows = rows.merge(stale, on="request_id", validate="one_to_one")
    if len(rows) != full["request_id"].nunique() or len(rows) != reuse["request_id"].nunique():
        raise RuntimeError(f"incomplete Full/Reuse request join for v{edge}_to_v{edge + 1}")
    try:
        target = np.asarray([labels[value] for value in rows["request_id"]], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError(f"missing label for request {error.args[0]}") from error

    parent_logit = rows["parent_full_logit"].to_numpy(dtype=np.float64)
    current_full_logit = rows["current_full_logit"].to_numpy(dtype=np.float64)
    current_exact_logit = rows["current_exact_logit"].to_numpy(dtype=np.float64)
    reuse_logit = rows["reuse_logit"].to_numpy(dtype=np.float64)
    rows["release_benefit"] = stable_log_loss(parent_logit, target) - stable_log_loss(current_full_logit, target)
    rows["reuse_harm"] = stable_log_loss(reuse_logit, target) - stable_log_loss(current_exact_logit, target)
    rows["abs_probability_shift"] = np.abs(sigmoid(reuse_logit) - sigmoid(current_exact_logit))
    rows["bernoulli_js"] = bernoulli_js(reuse_logit, current_exact_logit)
    rows["edge"] = f"v{edge}_to_v{edge + 1}"
    rows["day_since_cutover"] = np.floor(rows["seconds_since_cutover"] / DAY).astype(int)
    rows["cutover_old_count"] = (
        rows["cache_length"] - rows["append_count_since_cutover"] + rows["rolling_evictions"]
    )
    if (rows["cutover_old_count"] <= 0).any():
        raise RuntimeError("reconstructed cutover state must be nonempty")
    rows["remaining_old_count"] = np.maximum(
        rows["cutover_old_count"] - rows["rolling_evictions"], 0
    )
    rows["remaining_old_fraction"] = rows["remaining_old_count"] / rows["cutover_old_count"]
    rows["append_bucket"] = rows["append_count_since_cutover"].map(append_bucket)
    rows["remaining_bucket"] = rows["remaining_old_fraction"].map(remaining_bucket)
    return rows


def overlap_row(name: str, rows: pd.DataFrame) -> dict[str, float | int | str]:
    benefit = rows["release_benefit"].to_numpy()
    harm = rows["reuse_harm"].to_numpy()
    winners = benefit > 0.0
    positive_harm = np.maximum(harm, 0.0)
    total_positive_harm = float(positive_harm.sum())
    winner_share = float(winners.mean())
    harm_share = float(positive_harm[winners].sum() / total_positive_harm) if total_positive_harm else float("nan")
    return {
        "edge": name,
        "requests": len(rows),
        "users": int(rows["uid"].nunique()),
        "mean_release_benefit": float(benefit.mean()),
        "mean_reuse_harm": float(harm.mean()),
        "release_winner_fraction": winner_share,
        "reuse_harmed_fraction": float((harm > 0.0).mean()),
        "mean_harm_on_release_winners": float(harm[winners].mean()),
        "mean_harm_on_other_requests": float(harm[~winners].mean()),
        "positive_harm_on_release_winners_fraction": harm_share,
        "positive_harm_concentration_lift": harm_share / winner_share if winner_share else float("nan"),
        "spearman_G_H": float(rows[["release_benefit", "reuse_harm"]].corr(method="spearman").iloc[0, 1]),
    }


def grouped_dilution(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return rows.groupby(keys, observed=True, sort=True).agg(
        requests=("request_id", "size"),
        users=("uid", "nunique"),
        mean_reuse_harm=("reuse_harm", "mean"),
        mean_abs_probability_shift=("abs_probability_shift", "mean"),
        mean_bernoulli_js=("bernoulli_js", "mean"),
        mean_remaining_old_fraction=("remaining_old_fraction", "mean"),
        mean_append_count=("append_count_since_cutover", "mean"),
    ).reset_index()


def dilution_row(name: str, rows: pd.DataFrame) -> dict[str, float | int | str]:
    high = rows[rows["remaining_old_fraction"] > 0.75]
    empty = rows[rows["remaining_old_fraction"] == 0.0]
    return {
        "edge": name,
        "requests": len(rows),
        "spearman_remaining_vs_abs_probability_shift": float(
            rows[["remaining_old_fraction", "abs_probability_shift"]].corr(method="spearman").iloc[0, 1]
        ),
        "spearman_append_vs_remaining": float(
            rows[["append_count_since_cutover", "remaining_old_fraction"]].corr(method="spearman").iloc[0, 1]
        ),
        "mean_abs_probability_shift_remaining_gt_075": float(high["abs_probability_shift"].mean()),
        "mean_abs_probability_shift_remaining_zero": float(empty["abs_probability_shift"].mean()),
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        values = [f"{value:.6g}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")

    label_table = pq.read_table(MANIFEST / "requests_quality.parquet", columns=["request_id", "label"])
    labels = dict(zip(label_table["request_id"].to_pylist(), label_table["label"].to_pylist(), strict=True))
    edge_frames = [load_edge(edge, labels) for edge in range(5)]
    combined = pd.concat(edge_frames, ignore_index=True)
    overlap = pd.DataFrame([overlap_row(frame["edge"].iloc[0], frame) for frame in edge_frames])
    overlap = pd.concat([overlap, pd.DataFrame([overlap_row("pooled_descriptive", combined)])], ignore_index=True)
    by_day = grouped_dilution(combined, ["edge", "day_since_cutover"])
    by_remaining = grouped_dilution(combined, ["edge", "remaining_bucket"])
    dilution_grid = grouped_dilution(combined, ["edge", "remaining_bucket", "append_bucket"])
    dilution = pd.DataFrame([
        dilution_row(frame["edge"].iloc[0], frame) for frame in edge_frames
    ])

    args.output.mkdir(parents=True)
    overlap.to_csv(args.output / "benefit_harm_overlap.csv", index=False)
    by_day.to_csv(args.output / "dilution_by_day.csv", index=False)
    by_remaining.to_csv(args.output / "dilution_by_remaining_state.csv", index=False)
    dilution_grid.to_csv(args.output / "dilution_grid.csv", index=False)
    dilution.to_csv(args.output / "dilution_summary.csv", index=False)
    summary = {
        "status": "insight_first_pass_complete",
        "scope": "Yambda-500M Small seed17 D14/E14; descriptive discovery only",
        "requests": len(combined),
        "users": int(combined["uid"].nunique()),
        "edges": [f"v{edge}_to_v{edge + 1}" for edge in range(5)],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    columns = [
        "edge", "requests", "users", "mean_release_benefit", "mean_reuse_harm",
        "release_winner_fraction", "reuse_harmed_fraction",
        "positive_harm_on_release_winners_fraction", "positive_harm_concentration_lift", "spearman_G_H",
    ]
    report = [
        "# Insight first pass: dilution and benefit/harm overlap", "",
        "Scope: Yambda-500M Small, seed 17, D14/E14. This is descriptive discovery over existing sealed raw, not a new training result or a causal History Utility test.", "",
        "## Release benefit and Reuse harm", "",
        *markdown_table(overlap, columns), "",
        "Definitions: `G = loss(Parent Full) - loss(Current Full)`; `H = loss(Reuse) - loss(Current Exact Rolling)`. Concentration lift compares the share of positive harm on release winners with the winners' request share.", "",
        f"All five edges have positive G/H rank association ({overlap.iloc[:5]['spearman_G_H'].min():.3f} to {overlap.iloc[:5]['spearman_G_H'].max():.3f}) and positive-harm concentration lift above one ({overlap.iloc[:5]['positive_harm_concentration_lift'].min():.2f}x to {overlap.iloc[:5]['positive_harm_concentration_lift'].max():.2f}x). This supports benefit/harm overlap as a discovery signal; it is not yet a causal History Utility result.", "",
        "## Dilution outputs", "",
        "The CSV files report paired mean Reuse harm and output divergence by cutover day, remaining-old-state bucket, and the remaining-state x append-count grid. Inspect the grid before distinguishing eviction from current-version anchoring.", "",
        *markdown_table(dilution, list(dilution.columns)), "",
        "Remaining-old fraction is positively associated with Current-Reuse output shift on every edge, but append count and remaining-old fraction are strongly coupled in the observational rolling trace. This table alone cannot distinguish eviction from current-version anchoring; the separate controlled experiment is required.", "",
        "## Boundary", "",
        "History Utility is not measured in this pass. History length or stale-state volume must not be presented as a utility proxy.", "",
    ]
    (args.output / "report.md").write_text("\n".join(report))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
