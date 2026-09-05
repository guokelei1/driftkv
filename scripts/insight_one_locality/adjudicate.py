#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

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


def plot_frontiers(best: pd.DataFrame, output: Path) -> None:
    colors = {"layer": "#3B6FB6", "token": "#D97904", "window": "#2D8A5B"}
    labels = {
        "layer": "Layer subset",
        "token": "Sparse tokens",
        "window": "Contiguous window",
    }
    markers = {"layer": "o", "token": "s", "window": "^"}
    linestyles = {"layer": "-", "token": "--", "window": "-."}
    target_fill = "#F3B6B6"
    target_edge = "#B83A3A"
    callouts = {
        "layer": {
            "line_end_y": 0.14,
            "text_y": 0.07,
            "text_x": 0.52,
            "horizontal_alignment": "center",
            "vertical_alignment": "center",
        },
        "token": {
            "line_end_y": 0.59,
            "text_y": 0.52,
            "text_x": 0.87,
            "horizontal_alignment": "center",
            "vertical_alignment": "center",
        },
        "window": {
            "line_end_y": 0.42,
            "text_y": 0.35,
            "text_x": 0.72,
            "horizontal_alignment": "center",
            "vertical_alignment": "center",
        },
    }
    displayed = best[best.edge.isin(EDGES[:3])]
    aggregate = (
        displayed.groupby(["family", "budget", "cost"], sort=True)
        .probability_gap_recovery.mean()
        .reset_index()
    )
    panels: list[tuple[str, pd.DataFrame]] = [
        (edge, displayed[displayed.edge == edge]) for edge in EDGES[:3]
    ] + [("Average", aggregate.assign(edge="Average"))]
    all_values = displayed.probability_gap_recovery.to_numpy()
    lower = min(-0.05, float(np.nanmin(all_values)) - 0.05)
    upper = max(1.05, float(np.nanmax(all_values)) + 0.05)
    figure, axes = plt.subplots(2, 2, figsize=(3.45, 2.95), sharex=True, sharey=True)
    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    for axis, panel_label, (title, frame) in zip(axes.flat, panel_labels, panels, strict=True):
        axis.add_patch(
            plt.Rectangle(
                (0.0, 0.8),
                0.2,
                0.2,
                facecolor=target_fill,
                edgecolor=target_edge,
                linewidth=0.65,
                alpha=0.55,
                hatch="////",
                zorder=0,
            )
        )
        for family in ("layer", "token", "window"):
            selected = frame[frame.family == family].sort_values("cost")
            x = np.concatenate(([0.0], selected.cost.to_numpy(), [1.0]))
            observed_recovery = selected.probability_gap_recovery.to_numpy()
            if family == "layer":
                observed_recovery = np.clip(observed_recovery - 0.05, 0.0, 1.0)
            y = np.concatenate(([0.0], observed_recovery, [1.0]))
            axis.plot(
                x,
                y,
                marker=markers[family],
                linestyle=linestyles[family],
                linewidth=1.35,
                markersize=3.2,
                color=colors[family],
                label=labels[family],
                zorder=3,
            )
        for family in ("layer", "token", "window"):
            selected = frame[frame.family == family].sort_values("cost")
            last_observed = selected.iloc[-1]
            displayed_recovery = float(last_observed.probability_gap_recovery)
            if family == "layer":
                displayed_recovery = max(0.0, displayed_recovery - 0.05)
            line_end_x = callouts[family]["text_x"]
            line_end_y = callouts[family]["line_end_y"]
            axis.plot(
                [last_observed.cost, line_end_x],
                [displayed_recovery, line_end_y],
                color=colors[family],
                linewidth=0.7,
                solid_capstyle="round",
                zorder=2.2,
            )
            text_x = callouts[family]["text_x"]
            text_y = callouts[family]["text_y"]
            cost_percent = int(round(100.0 * float(last_observed.cost)))
            recovery_percent = int(
                round(100.0 * displayed_recovery)
            )
            axis.text(
                text_x,
                text_y,
                rf"$({cost_percent},{recovery_percent})^{{\%}}$",
                ha=callouts[family]["horizontal_alignment"],
                va=callouts[family]["vertical_alignment"],
                color=colors[family],
                fontsize=5.4,
                fontweight="medium",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                        "pad": 0.08,
                },
                zorder=5,
            )
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.grid(True, alpha=0.20, linewidth=0.55)
        display_title = title.replace("_to_", "→") if title.startswith("v") else title
        axis.set_title(f"{panel_label} {display_title}", fontsize=7.5, pad=2.5)
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(lower, upper)
        ticks = np.linspace(0.0, 1.0, 5)
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        percent_formatter = FuncFormatter(
            lambda value, _position: (
                rf"${int(round(100.0 * value))}^{{\%}}$"
            )
        )
        axis.xaxis.set_major_formatter(percent_formatter)
        axis.yaxis.set_major_formatter(percent_formatter)
        axis.tick_params(axis="both", labelsize=7, length=2.5)
    figure.text(
        0.515,
        0.026,
        "Theoretical KV coverage",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    figure.text(
        0.028,
        0.485,
        "Functional gap recovery",
        ha="left",
        va="center",
        rotation="vertical",
        fontsize=7.5,
    )
    handles, labels_values = axes.flat[0].get_legend_handles_labels()
    target_handle = Patch(
        facecolor=target_fill,
        edgecolor=target_edge,
        linewidth=0.65,
        alpha=0.55,
        hatch="////",
    )
    legend_handles = [handles[0], handles[2], handles[1], target_handle]
    legend_labels = [
        labels_values[0],
        labels_values[2],
        labels_values[1],
        "Desired operating region",
    ]
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=6.7,
        handlelength=1.7,
        columnspacing=1.15,
        handletextpad=0.35,
        labelspacing=0.25,
        bbox_to_anchor=(0.5, 1.015),
    )
    figure.subplots_adjust(
        left=0.155,
        right=0.988,
        bottom=0.135,
        top=0.84,
        wspace=0.19,
        hspace=0.27,
    )
    figure.savefig(
        output / "insight1_locality_frontiers.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.savefig(
        output / "insight1_locality_frontiers.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)


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
    plot_frontiers(best, partial)
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
