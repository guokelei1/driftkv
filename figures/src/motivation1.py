"""Motivation Figure 1: AUC loss from reusing an older persistent KV cache.

The matrix entry at row ``current_model`` and column ``cache_version`` is

    1 - AUC(current model + cache_version - old)
        / AUC(current model + recomputed cache - old)

which is reported here as the fraction of the Current-vs-Parent AUC release
gain lost by Reuse.  Diagonal zero cases are omitted because they do not reuse
an older cache.  Only sealed values from the D14/E14 motivation artifacts are
entered below; unmeasured cells are left blank instead of being interpolated
or inferred.

Run from the repository root:

    python figures/src/motivation1.py

Outputs:
    figures/pic/jpg/motivation1.jpg
    figures/pic/pdf/motivation1.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
JPG_OUT = ROOT / "figures" / "pic" / "jpg" / "motivation1.jpg"
PDF_OUT = ROOT / "figures" / "pic" / "pdf" / "motivation1.pdf"

# v0 can produce a cache but has no preceding release.  v5 cannot be an older
# cache for any current version shown here, so the historical-cache axis ends
# at v4.
CACHE_VERSIONS = ["v0", "v1", "v2", "v3", "v4"]
CURRENT_VERSIONS = ["v1", "v2", "v3", "v4", "v5"]
CACHE_LABELS = ["C0", "C1", "C2", "C3", "C4"]
CURRENT_LABELS = ["M1", "M2", "M3", "M4", "M5"]

# Current-vs-Parent Full ROC-AUC release gains from the fixed D14/E14 table,
# in percentage points.  This is shown as a line on the right.
RELEASE_GAIN_PP = {
    "v1": 1.199879,
    "v2": 0.626216,
    "v3": 0.310144,
    "v4": 0.046331,
    "v5": 0.220611,
}

# Reuse AUC harm (Current Exact Rolling - Reuse), in percentage points.
# Values are from the sealed D14/E14 one-hop and direct long-age artifacts.
DIRECT_REUSE_HARM_PP = {
    ("v1", "v0"): 0.3060142481099315,
    ("v2", "v0"): 0.5308523622681194,
    ("v2", "v1"): 0.19966160704484315,
    ("v3", "v0"): 0.45277684605078417,
    ("v3", "v1"): 0.22470826426196355,
    ("v3", "v2"): 0.14869348043838881,
    ("v4", "v0"): 0.6941849652866039,
    ("v4", "v1"): 0.3609879843375907,
    ("v4", "v2"): 0.25316363131886455,
    ("v4", "v3"): 0.1939839199060045,
    ("v5", "v0"): 0.8486791613806388,
    ("v5", "v1"): 0.34101845861639335,
    ("v5", "v2"): 0.2589748042084894,
    ("v5", "v3"): 0.2056131030367503,
    ("v5", "v4"): 0.06373634986358567,
}


def build_loss_matrix() -> np.ndarray:
    """Return loss fractions in [row=current model, column=cache]."""
    matrix = np.full(
        (len(CURRENT_VERSIONS), len(CACHE_VERSIONS)), np.nan, dtype=float
    )
    current_index = {
        version: index for index, version in enumerate(CURRENT_VERSIONS)
    }
    cache_index = {version: index for index, version in enumerate(CACHE_VERSIONS)}

    for (current, producer), harm_pp in DIRECT_REUSE_HARM_PP.items():
        # Express the calculation in the same form as the figure definition:
        # AUC(recompute-old) is the Full-only release gain, and Reuse loses
        # ``harm_pp`` of that gain.
        gain_pp = RELEASE_GAIN_PP[current]
        recompute_minus_old_pp = gain_pp
        reuse_minus_old_pp = recompute_minus_old_pp - harm_pp
        matrix[current_index[current], cache_index[producer]] = (
            1.0 - reuse_minus_old_pp / recompute_minus_old_pp
        )

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
    # 100--1500% tail logarithmically so its internal ordering stays visible.
    maximum = float(np.nanmax(matrix))
    if np.any(above):
        positions[above] = split + (1.0 - split) * np.power(
            np.log(matrix[above]) / np.log(maximum), 0.75
        )
    return positions


def draw() -> None:
    matrix = build_loss_matrix()
    # Use a compact landscape aspect ratio for paper placement.  The extra
    # width accommodates the release-gain annotations without increasing the
    # figure's vertical footprint.
    figure, axis = plt.subplots(figsize=(8.6, 3.45), dpi=180)
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.98)

    # Mask unmeasured cells.  They remain visibly blank.
    color_positions = build_color_positions(matrix)
    masked = np.ma.masked_invalid(color_positions)
    cmap = LinearSegmentedColormap.from_list(
        "reuse_loss", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    ).copy()
    cmap.set_bad("#eeeeee")
    # ``build_color_positions`` explicitly separates <=100% and >100% while
    # preserving a monotonic light-to-dark ordering in both segments.
    norm = Normalize(vmin=0.0, vmax=1.0)
    image = axis.imshow(masked, cmap=cmap, norm=norm)
    # Slightly wider cells make the complete figure more landscape-oriented
    # for paper placement while keeping the matrix easy to scan.
    axis.set_aspect(0.62)

    gain_x = -1.22
    axis.set_xticks(
        [gain_x, *range(len(CACHE_VERSIONS))],
        labels=["", *CACHE_LABELS],
    )
    axis.set_yticks(range(len(CURRENT_VERSIONS)), labels=CURRENT_LABELS)
    axis.set_xlabel("Cache version", fontsize=14.5, fontweight="bold", labelpad=1)
    axis.set_ylabel(
        "Current model version", fontsize=14.5, fontweight="bold", labelpad=4
    )
    axis.tick_params(axis="both", labelsize=14, width=1.7, length=5)
    axis.tick_params(axis="y", pad=10)
    for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
        tick_label.set_fontweight("bold")
    axis.text(
        gain_x,
        -0.04,
        "Model\nImprovement",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        color="#d62728",
        fontsize=12.0,
        fontweight="bold",
        linespacing=0.85,
        clip_on=False,
    )
    axis.text(
        gain_x,
        -0.145,
        "(vs. previous model)",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        color="#d62728",
        fontsize=12.0,
        fontweight="bold",
        clip_on=False,
    )

    legend = axis.legend(
        handles=[
            Patch(
                facecolor="#c6dbef",
                edgecolor="#6baed6",
                label=(
                    "Earlier-cache reuse impact\n"
                    "(% of model improvement)"
                ),
            )
        ],
        loc="upper right",
        bbox_to_anchor=(0.995, 0.985),
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.775,
        handleheight=2.775,
        prop={"size": 12.0, "weight": "bold"},
    )

    # Draw only the meaningful lower-triangular cell grid.  Future-cache cells
    # in the upper triangle are removed from the visual entirely.
    axis.grid(False)
    for row in range(len(CURRENT_VERSIONS)):
        for column in range(row + 1):
            axis.add_patch(
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
        for column in range(row + 1, len(CACHE_VERSIONS)):
            axis.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor="white",
                    edgecolor="white",
                    linewidth=0,
                    zorder=3,
                )
            )

    # Annotate measured values.  Values above 100%
    # are retained because a small Full-only release gain can be fully
    # overwhelmed by an otherwise real Reuse harm (v4 over older caches).
    axis.set_xticks(np.arange(-0.5, len(CACHE_VERSIONS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(CURRENT_VERSIONS), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="minor", bottom=False, left=False)

    for row in range(len(CURRENT_VERSIONS)):
        for column in range(len(CACHE_VERSIONS)):
            value = matrix[row, column]
            if np.isfinite(value):
                # Choose label contrast from the displayed colour rather than
                # the raw ratio because the power normalization is nonlinear.
                text_color = "white" if color_positions[row, column] >= 0.54 else "#123"
                axis.text(
                    column,
                    row,
                    f"−{value:.0%}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=13,
                    fontweight="bold",
                )
            # Missing measured cells remain blank; they are not zeroes.

    # Put the absolute Full-only model release gain in a dedicated column to
    # the left of the matrix's v0 column.
    gain_values = np.full(len(CURRENT_VERSIONS), np.nan, dtype=float)
    for row, version in enumerate(CURRENT_VERSIONS):
        if version in RELEASE_GAIN_PP:
            gain_values[row] = RELEASE_GAIN_PP[version]
    axis.set_xlim(-1.94, len(CACHE_VERSIONS) - 0.5)
    axis.axvspan(-1.94, -0.5, color="#fff7f5", zorder=0)
    axis.axvline(
        -0.5,
        color="#d62728",
        linestyle="--",
        linewidth=1.6,
        zorder=4,
    )
    for row, value in enumerate(gain_values):
        label = "—" if not np.isfinite(value) else f"+{value:.3f}%"
        if np.isfinite(value):
            axis.text(
                gain_x,
                row,
                label,
                ha="center",
                va="center",
                color="#d62728",
                fontsize=13.5,
                fontweight="bold",
            )
        # v0 has no parent-release gain; leave the corresponding annotation blank.

    figure.canvas.draw()
    JPG_OUT.parent.mkdir(parents=True, exist_ok=True)
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(JPG_OUT, format="jpg", dpi=300, bbox_inches="tight")
    figure.savefig(PDF_OUT, format="pdf", bbox_inches="tight")
    plt.close(figure)

    print(f"wrote {JPG_OUT}")
    print(f"wrote {PDF_OUT}")


if __name__ == "__main__":
    draw()
