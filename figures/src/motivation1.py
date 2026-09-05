"""Motivation Figure 1: six-layer Medium cross-version K/V compatibility.

Read the completed Medium D14/E14 triangle and the matching adjacent
three-path reports. No experiment is run and no source result is modified.

The left column shows Current-minus-Parent ROC-AUC on the percentage scale
(e.g., 0.02356 becomes +2.356%), from each release's adjacent three-path
comparison. These are absolute AUC differences, not relative AUC growth.
All left cells have the same size and color; gain magnitude is not encoded
by bar length, position, or shading. A heatmap cell divides its
within-run Current-minus-Reuse AUC loss by that release's reference gain.
Older-producer runs did not recompute Parent; their ratios are a common
reference scale, not newly measured three-path retained-gain estimates.

Model: 30,000-user Yambda-500M Medium, 6L/H192/6 heads/C1024, seed 17.
V0 and every V1--V5 update use one pass. The V5 E14 row includes the observed,
incomplete day-300 tail, identified in the paper caption.
The two panels share model rows but not units: absolute AUC gains on the
left, negative percentages denoting lost improvement on the right. Bold typography and
the original red/blue palette and heatmap color mapping are retained.

Run: python figures/src/motivation1.py
Outputs: figures/pic/{jpg,pdf}/motivation1.{jpg,pdf}
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = Path("results/yambda500m_medium_seed17/full_reuse_matrix_v1/D14")
TRIANGLE_PATH = RESULT_ROOT / "direct_long_age_reuse_v1/summary.json"
ADJACENT_PATHS = {
    **{
        f"v{i + 1}": RESULT_ROOT / f"reuse/E14/v{i}_to_v{i + 1}/adjudication.json"
        for i in range(4)
    },
    "v5": RESULT_ROOT / "v5_extension_v1/reuse/E14_partial/v4_to_v5/adjudication.json",
}
JPG_OUT = ROOT / "figures/pic/jpg/motivation1.jpg"
PDF_OUT = ROOT / "figures/pic/pdf/motivation1.pdf"
CACHE_VERSIONS = tuple(f"v{i}" for i in range(5))
CURRENT_VERSIONS = tuple(f"v{i}" for i in range(1, 6))
CACHE_LABELS = ["C0", "C1", "C2", "C3", "C4"]
CURRENT_LABELS = ["M1", "M2", "M3", "M4", "M5"]
EXPECTED_PAIRS = {
    (f"v{current}", f"v{producer}")
    for current in range(1, 6)
    for producer in range(current)
}


def load_measurements(
    root: Path = ROOT,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Load matched release gains and within-run losses from sealed reports."""
    triangle = json.loads((root / TRIANGLE_PATH).read_text(encoding="utf-8"))
    if (
        triangle["status"] != "medium_D14_E14_direct_long_age_triangle_complete"
        or triangle["completed_triangle_cells"] != 15
        or triangle["primary_comparison"] != "within_run_current_exact_vs_direct_reuse"
        or not triangle["direct_non_recursive_reuse"]
    ):
        raise ValueError("Expected the complete, paired Medium direct-Reuse triangle")

    gains: dict[str, float] = {}
    adjacent = {}
    for current, path in ADJACENT_PATHS.items():
        record = json.loads((root / path).read_text(encoding="utf-8"))
        three = record["three_path_summary"]
        gain = 100.0 * (three["new_current"]["ROC_AUC"] - three["old_parent"]["ROC_AUC"])
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError(f"Release-gain normalization requires a positive gain: {current}")
        if not math.isclose(gain, three["current_minus_old_ROC_AUC_pp"], abs_tol=1e-10):
            raise ValueError(f"Release gain differs from its source report: {current}")
        gains[current] = gain
        adjacent[current] = three

    losses: dict[tuple[str, str], float] = {}
    for row in triangle["rows"]:
        pair = row["current"], row["producer"]
        if pair in losses or pair not in EXPECTED_PAIRS:
            raise ValueError(f"Unexpected or duplicate cache pair: {pair}")
        # Each subtraction stays within this row's paired evaluation.
        loss = 100.0 * (row["new_current"]["ROC_AUC"] - row["reuse"]["ROC_AUC"])
        if not math.isfinite(loss) or not math.isclose(
            loss, row["current_minus_reuse_ROC_AUC_pp"], abs_tol=1e-10
        ):
            raise ValueError(f"Paired loss differs from its source report: {pair}")
        losses[pair] = loss
        if row["version_gap"] == 1:
            three = adjacent[row["current"]]
            for row_key, three_key in (
                ("new_current", "new_current"),
                ("reuse", "adjacent_one_hop_reuse"),
            ):
                if not math.isclose(
                    row[row_key]["ROC_AUC"], three[three_key]["ROC_AUC"], abs_tol=1e-12
                ):
                    raise ValueError(f"Adjacent reports use different AUC values: {pair}")
    if set(losses) != EXPECTED_PAIRS:
        raise ValueError("The figure must include all 15 measured producer/Current pairs")
    return gains, losses


def build_loss_matrix(
    release_gains_pp: Mapping[str, float],
    paired_losses_pp: Mapping[tuple[str, str], float],
) -> np.ndarray:
    """Return signed loss/reference-gain fractions; unused cells remain NaN."""
    matrix = np.full((5, 5), np.nan, dtype=float)
    for (current, producer), loss in paired_losses_pp.items():
        row = CURRENT_VERSIONS.index(current)
        column = CACHE_VERSIONS.index(producer)
        gain = release_gains_pp[current]
        if column > row or not math.isfinite(gain) or gain <= 0:
            raise ValueError(f"Invalid cache pair or release gain: {(current, producer)}")
        matrix[row, column] = loss / gain
    return matrix


def build_color_positions(matrix: np.ndarray) -> np.ndarray:
    """Map loss fractions onto a two-segment [0, 1] colour scale."""
    positions = np.full_like(matrix, np.nan)
    finite = np.isfinite(matrix)
    below = finite & (matrix <= 1.0)
    above = finite & (matrix > 1.0)

    # Give 0--100% enough of the palette to remain readable, with a mild
    # power transform that makes the ordinary nonzero cells darker overall.
    split = 0.58
    positions[below] = split * np.power(np.clip(matrix[below], 0.0, 1.0), 0.55)

    # Start every >100% cell in the darker segment, then compress the long
    # above-100% tail logarithmically so its internal ordering stays visible.
    maximum = float(np.nanmax(matrix))
    if np.any(above):
        positions[above] = split + (1.0 - split) * np.power(
            np.log(matrix[above]) / np.log(maximum), 0.75
        )
    return positions


def build_figure(
    release_gains_pp: Mapping[str, float],
    paired_losses_pp: Mapping[tuple[str, str], float],
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Build row-aligned panels without mixing absolute gains and loss ratios."""
    matrix = build_loss_matrix(release_gains_pp, paired_losses_pp)
    rows = np.arange(len(CURRENT_VERSIONS))
    figure, (gain_axis, loss_axis) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(8.6, 4.05),
        dpi=180,
        gridspec_kw={"width_ratios": [0.9, 2.0], "wspace": 0.16},
    )
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.17, top=0.80)

    # Separate panels and a shared row scale express the reading order:
    # this release's gain -> how much of that gain each cache loses.
    gain_axis.set_title(
        "(a) Model-update improvement",
        fontsize=14,
        fontweight="bold",
        color="black",
        pad=15,
    )
    loss_axis.set_title(
        "(b) Lost improvement (%)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    # This is a numeric reference column, not a magnitude comparison chart.
    # Each gain has identical placement, type, cell size, and background.
    gain_axis.set_xlim(0.0, 1.0)
    gain_axis.set_xticks([])
    gain_axis.set_xlabel("AUC increase (%)", fontsize=14.5, fontweight="bold", labelpad=5)
    gain_axis.set_ylabel(
        "Current model version", fontsize=14.5, fontweight="bold", labelpad=6
    )
    gain_axis.set_yticks(rows, labels=CURRENT_LABELS)
    for row, version in enumerate(CURRENT_VERSIONS):
        gain_axis.add_patch(
            Rectangle(
                (0.0, row - 0.5),
                1.0,
                1.0,
                facecolor="#fff7f5",
                edgecolor="white",
                linewidth=1.2,
            )
        )
        gain_axis.text(
            0.5,
            row,
            f"+{release_gains_pp[version]:.3f}%",
            ha="center",
            va="center",
            color="#d62728",
            fontsize=14,
            fontweight="bold",
        )

    color_positions = build_color_positions(matrix)
    cmap = LinearSegmentedColormap.from_list(
        "reuse_loss", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    ).copy()
    cmap.set_bad("white")
    # Retain the original monotonic two-segment color mapping; the printed
    # percentages, not the nonlinear shading, give exact magnitudes.
    loss_axis.imshow(
        np.ma.masked_invalid(color_positions),
        cmap=cmap,
        norm=Normalize(vmin=0.0, vmax=1.0),
        aspect="auto",
        interpolation="nearest",
    )
    loss_axis.set_xlim(-0.5, len(CACHE_VERSIONS) - 0.5)
    loss_axis.set_xticks(range(len(CACHE_VERSIONS)), labels=CACHE_LABELS)
    loss_axis.set_xlabel(
        "Cache producer version", fontsize=14.5, fontweight="bold", labelpad=5
    )
    loss_axis.tick_params(axis="y", left=False, labelleft=False)

    for row in rows:
        for column in range(row + 1):
            loss_axis.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="white",
                    linewidth=1.2,
                    zorder=3,
                )
            )
            value = matrix[row, column]
            text_color = "white" if color_positions[row, column] >= 0.54 else "#123"
            loss_axis.text(
                column,
                row,
                f"−{value:.0%}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=14,
                fontweight="bold",
            )

    # Explain the normalization with one existing cell, not a new result.
    # The arithmetic deliberately uses the rounded labels visible in the
    # figure; the matrix itself still uses the full-precision measurements.
    example_row = CURRENT_VERSIONS.index("v3")
    example_column = CACHE_VERSIONS.index("v2")
    example_ratio_pct = float(f"{100.0 * matrix[example_row, example_column]:.0f}")
    example_improvement_pct = float(f"{release_gains_pp['v3']:.3f}")
    example_loss_pct = example_ratio_pct / 100.0 * example_improvement_pct
    callout_color = "#D5A021"
    loss_axis.add_patch(
        Rectangle(
            (example_column - 0.48, example_row - 0.48),
            0.96,
            0.96,
            fill=False,
            edgecolor=callout_color,
            linewidth=2.2,
            zorder=5,
        )
    )
    # Keep all callout text bold; only the improvement is red to link it
    # to the left panel. All other text and calculation symbols stay black.
    callout_style = {"fontsize": 11.5, "fontweight": "bold", "color": "black"}
    calculation = HPacker(
        children=[
            TextArea(value, textprops={**callout_style, "color": color})
            for value, color in (
                (f"−{example_ratio_pct:.0f}%", "black"),
                (" × ", "black"),
                (f"{example_improvement_pct:.3f}%", "#d62728"),
                (" ≈ ", "black"),
                (f"−{example_loss_pct:.3f}%", "black"),
                (".", "black"),
            )
        ],
        align="baseline", pad=0, sep=0,
    )
    explanation = VPacker(
        children=[
            TextArea("For example, reusing cache produced", textprops=callout_style),
            TextArea("by M2 in M3 changes AUC by", textprops=callout_style),
            calculation,
        ],
        align="right", pad=0, sep=3,
    )
    loss_axis.add_artist(AnnotationBbox(
        explanation, (4.38, -0.30), xycoords="data",
        box_alignment=(1, 1), frameon=False, pad=0, annotation_clip=False,
    ))
    # Lead from the outlined cell into the explanation in the empty triangle.
    loss_axis.annotate(
        "",
        xy=(3.5, 1.05),
        xytext=(example_column + 0.48, example_row - 0.46),
        arrowprops={
            "arrowstyle": "->",
            "color": callout_color,
            "linewidth": 1.8,
            "mutation_scale": 13,
            "connectionstyle": "arc3,rad=0.12",
            "shrinkA": 1,
            "shrinkB": 2,
        },
        zorder=6,
    )

    # Fix identical row centers after imshow has initialized the shared axis.
    gain_axis.set_ylim(len(CURRENT_VERSIONS) - 0.5, -0.5)
    for axis in (gain_axis, loss_axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["bottom"].set_linewidth(1.2)
        axis.spines["left"].set_linewidth(1.2)
        axis.tick_params(axis="x", labelsize=13, width=1.5, length=4)
        axis.tick_params(axis="y", labelsize=14, width=1.5, length=4, pad=7)
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontweight("bold")
    loss_axis.spines["left"].set_visible(False)
    for spine in gain_axis.spines.values():
        spine.set_visible(False)
    gain_axis.tick_params(axis="y", length=0)
    # Keep both bottom labels on one baseline despite removing gain ticks.
    gain_axis.xaxis.set_label_coords(0.5, -0.14)
    loss_axis.xaxis.set_label_coords(0.5, -0.14)
    return figure, (gain_axis, loss_axis)


def draw() -> None:
    release_gains_pp, paired_losses_pp = load_measurements()
    figure, _ = build_figure(release_gains_pp, paired_losses_pp)
    JPG_OUT.parent.mkdir(parents=True, exist_ok=True)
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(JPG_OUT, format="jpg", dpi=300, bbox_inches="tight")
    figure.savefig(PDF_OUT, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {JPG_OUT}")
    print(f"wrote {PDF_OUT}")


if __name__ == "__main__":
    draw()
