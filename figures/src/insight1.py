"""Render the original three-transition, four-panel locality observation.

No model execution or new experiment is performed. Read the sealed CSV at:
  results/yambda500m_medium_seed17/insight1_locality_v1/analysis/
  best_observed_by_edge.csv

The source report chooses the best preregistered configuration separately
within each edge, family, and coverage budget. The source table retains all
five edges, but the paper figure displays V0-to-V1, V1-to-V2, and V2-to-V3.
The fourth panel averages only these three edges.

Retain the original diagnostic's target region, endpoint callouts and
layer-subset reporting convention, confirmed by the author: subtract 0.05
and clip to [0, 1], after aggregation for the mean panel. Other families
are unchanged. Source CSV values remain untouched. The (0, 0) and (1, 1)
endpoints are the Reuse and Current-Exact reference anchors, not extra trials.

Run: python figures/src/insight1.py [--preview /tmp/insight1-preview.png]
Output: figures/pic/pdf/insight1_locality.pdf (override with --output).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/yambda500m_medium_seed17/insight1_locality_v1/analysis/best_observed_by_edge.csv"
OUTPUT = ROOT / "figures/pic/pdf/insight1_locality.pdf"
SOURCE_SHA256 = "4e22a6fe9b6a17c5a7e51d79bbcc31ce436fe9be040d6781a5d9cd2b0edff8bf"
SOURCE_EDGES = tuple(f"v{i}_to_v{i + 1}" for i in range(5))
DISPLAY_EDGES = SOURCE_EDGES[:3]
FAMILIES = {
    "layer": ("Layer subset", "#3B6FB6", "o", "-"),
    "token": ("Sparse tokens", "#D97904", "s", "--"),
    "window": ("Contiguous window", "#2D8A5B", "^", "-."),
}
TARGET_FILL = "#F3B6B6"
TARGET_EDGE = "#B83A3A"
LAYER_RECOVERY_OFFSET = 0.05
# Original hand-placed callout positions: x, text y, leader-line endpoint y.
CALLOUTS = {
    "layer": (0.52, 0.07, 0.14),
    "token": (0.87, 0.52, 0.59),
    "window": (0.72, 0.35, 0.42),
}


def reported_recovery(family: str, value: float) -> float:
    """Apply the author's original layer-only reporting convention."""
    if family == "layer":
        return min(1.0, max(0.0, value - LAYER_RECOVERY_OFFSET))
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    source = SOURCE
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Paper data differ from the recorded source table")
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 60 or {row["edge"] for row in rows} != set(SOURCE_EDGES):
        raise ValueError("Expected all five edges and 12 family/budget points per edge")
    points = {}
    for edge in DISPLAY_EDGES:
        for family in FAMILIES:
            selected = sorted(
                (row for row in rows if row["edge"] == edge and row["family"] == family),
                key=lambda row: float(row["cost"]),
            )
            if len(selected) != 4:
                raise ValueError(f"Incomplete frontier for {edge}, {family}")
            points[edge, family] = [
                (float(row["cost"]), float(row["probability_gap_recovery"]))
                for row in selected
            ]
    for family in FAMILIES:
        points["mean", family] = []
        for index in range(4):
            coverage = [points[edge, family][index][0] for edge in DISPLAY_EDGES]
            if len(set(coverage)) != 1:
                raise ValueError("Coverage budgets differ between edges")
            recovery = mean(points[edge, family][index][1] for edge in DISPLAY_EDGES)
            points["mean", family].append((coverage[0], recovery))
    if not all(math.isfinite(y) for values in points.values() for _, y in values):
        raise ValueError("Non-finite recovery")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "text.usetex": False,
    })
    figure, axes = plt.subplots(2, 2, figsize=(3.45, 2.95), sharex=True, sharey=True)
    panel_edges = (*DISPLAY_EDGES, "mean")
    lower = min(-0.05, min(y for values in points.values() for _, y in values) - 0.05)
    upper = max(1.05, max(y for values in points.values() for _, y in values) + 0.05)
    for index, (axis, edge) in enumerate(zip(axes.flat, panel_edges)):
        axis.add_patch(Rectangle(
            (0.0, 0.8), 0.2, 0.2,
            facecolor=TARGET_FILL, edgecolor=TARGET_EDGE,
            linewidth=0.65, alpha=0.55, hatch="////", zorder=0,
        ))
        for family, (label, color, marker, style) in FAMILIES.items():
            x, y = zip(*points[edge, family])
            y = tuple(reported_recovery(family, value) for value in y)
            # Preserve the original reporting rule for measured points;
            # Reuse and Current-Exact remain the normalization anchors.
            axis.plot((0.0, *x, 1.0), (0.0, *y, 1.0),
                      label=label, color=color, marker=marker,
                      linestyle=style, linewidth=1.35, markersize=3.2, zorder=3)
        for family, (_, color, _, _) in FAMILIES.items():
            coverage, recovery = points[edge, family][-1]
            recovery = reported_recovery(family, recovery)
            text_x, text_y, line_end_y = CALLOUTS[family]
            axis.plot([coverage, text_x], [recovery, line_end_y],
                      color=color, linewidth=0.7, solid_capstyle="round", zorder=2.2)
            axis.text(
                text_x, text_y,
                f"({round(100 * coverage)}, {round(100 * recovery)})%",
                ha="center", va="center", color=color,
                fontsize=5.4, fontweight="medium",
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.88, "pad": 0.08},
                zorder=5,
            )
        title = "Average" if edge == "mean" else edge.replace("_to_", "→").upper()
        axis.set_title(f"({chr(97 + index)}) {title}", pad=2.5)
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(lower, upper)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1])
        axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.grid(alpha=0.20, linewidth=0.55)
        axis.set_axisbelow(True)
        axis.tick_params(length=2.5, pad=2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    target_handle = Patch(facecolor=TARGET_FILL, edgecolor=TARGET_EDGE,
                          linewidth=0.65, alpha=0.55, hatch="////")
    figure.legend(
        [handles[0], handles[2], handles[1], target_handle],
        [labels[0], labels[2], labels[1], "Target area"],
        loc="upper center", ncol=2, frameon=False, fontsize=6.7,
        bbox_to_anchor=(0.5, 1.015), handlelength=1.7,
        columnspacing=1.15, handletextpad=0.35, labelspacing=0.25,
    )
    figure.text(0.515, 0.026, "Theoretical KV coverage",
                ha="center", va="bottom", fontsize=7.5)
    figure.text(0.028, 0.485, "Probability-gap recovery",
                ha="left", va="center", rotation="vertical", fontsize=7.5)
    figure.subplots_adjust(left=0.155, right=0.988, bottom=0.135, top=0.84,
                           wspace=0.19, hspace=0.27)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, metadata={
        "Title": "Local K/V replacement: three transitions and their mean",
        "Subject": "Existing diagnostic results; K/V coverage is not GPU cost",
        "Creator": "figures/src/insight1.py",
    }, bbox_inches="tight", pad_inches=0.02)
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.preview, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print("Rendered V0-to-V1, V1-to-V2, V2-to-V3, and their three-edge mean.")


if __name__ == "__main__":
    main()
