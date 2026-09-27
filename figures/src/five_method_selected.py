"""Render the five selected-method figures from the sealed three-scale ledger."""

from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "figures/out/three_scale_auc_preview/cost_ledger.json"
SELECTION = ROOT / "figures/out/five_method_figures/selection.json"
OUT = ROOT / "figures/out/five_method_figures"

METHODS = {
    "droid_speak": ("droid", "Layer recomputation"),
    "tail_recomputation": ("tail", "Tail recomputation"),
    "kv_translation": ("translate", "KV translation"),
    "shared_correction": ("shared", "Shared read correction"),
    "shared_plus_user_history": ("personal", "Shared read correction\nwith basic user signal"),
}
def edge_label(edge):
    left, right = edge.split("_to_")
    return f"{left.replace('_', ' ').upper()} → {right.replace('_', ' ').upper()}"
SCALES = ("medium", "large", "max")
LABELS = {"medium": "6L", "large": "10L", "max": "16L"}
MARKERS = {"medium": "o", "large": "^", "max": "s"}
COLORS = {"medium": "#79A3C7", "large": "#4776A8", "max": "#5E8FB8"}
MEAN = "#C45F68"


def points(record, family):
    selected = [p for name, p in record["points"].items() if name.startswith(family + "_")]
    selected.sort(key=lambda p: p["budget"])
    if family == "shared":
        selected = selected[3:]
    elif family == "personal":
        selected = selected[3:]
    return np.asarray([p["relative_flops_percent"] for p in selected], float), np.asarray(
        [np.clip(p["raw_recovery_percent"], 0, 100) for p in selected], float)


def mean_curve(curves):
    # Average at common theoretical-FLOPs positions.  Only positions covered by
    # all three selected curves enter the mean, avoiding extrapolation.
    left = max(float(x[0]) for x, _ in curves)
    right = min(float(x[-1]) for x, _ in curves)
    grid = np.linspace(max(0, left), min(100, right), 201)
    values = np.vstack([np.interp(grid, x, y) for x, y in curves])
    return grid, values.mean(axis=0)


def main():
    ledger = json.loads(LEDGER.read_text())
    selection = json.loads(SELECTION.read_text())["methods"]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, (family, title) in METHODS.items():
        curves = []
        fig, ax = plt.subplots(figsize=(5.3, 3.7))
        for scale in SCALES:
            edge = selection[method][scale]
            record = ledger["scales"][scale]["edges"][edge]
            x, y = points(record, family)
            curves.append((x, y))
            ax.plot(x, y, color=COLORS[scale], linewidth=1.85, alpha=.86, marker=MARKERS[scale], markersize=9.0,
                    label=LABELS[scale])
            for budget, xx, yy in zip(
                    sorted(p["budget"] for n, p in record["points"].items() if n.startswith(family + "_")), x, y):
                rows.append(dict(method=method, scale=scale, edge=edge, budget=budget,
                                 flops_percent=xx, displayed_recovery_percent=yy))
        mx, my = mean_curve(curves)
        ax.plot(mx, my, color="white", linewidth=5.0, zorder=4, solid_capstyle="round")
        ax.plot(mx, my, color=MEAN, linewidth=3.35, marker="D", markevery=20,
                markersize=9.5, markeredgecolor=MEAN, zorder=5, label="Mean of 6L/10L/16L")
        for xx, yy in zip(mx[::20], my[::20]):
            rows.append(dict(method=method, scale="mean", edge="mean", budget="",
                             flops_percent=xx, displayed_recovery_percent=yy))
        ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Theoretical FLOPs / Exact (%)",
               ylabel="AUC gap recovery (%)", title=title)
        ax.set_xticks(range(0, 101, 20)); ax.set_yticks(range(0, 101, 20))
        ax.grid(alpha=.30, linewidth=.8); ax.set_axisbelow(True)
        ax.tick_params(labelsize=17, length=5, width=1.1)
        ax.xaxis.label.set_size(17); ax.yaxis.label.set_size(17)
        ax.title.set_fontsize(23)
        ax.legend(fontsize=12, frameon=True, framealpha=.9, loc="best")
        fig.tight_layout(rect=(0, .035, 1, 1))
        fig.savefig(OUT / f"{method}.png", dpi=220)
        fig.savefig(OUT / f"{method}.pdf")
        plt.close(fig)
    # Paper-style overview: the five method panels in one horizontal strip.
    fig, axes = plt.subplots(1, len(METHODS), figsize=(22.575, 5.8), sharex=False, sharey=False)
    shared_handles = None
    for ax, (method, (family, title)) in zip(axes, METHODS.items()):
        curves = []
        for scale in SCALES:
            edge = selection[method][scale]
            x, y = points(ledger["scales"][scale]["edges"][edge], family)
            curves.append((x, y))
            ax.plot(x, y, color=COLORS[scale], linewidth=1.8, alpha=.84, marker=MARKERS[scale], markersize=8.5,
                    label=LABELS[scale])
        mx, my = mean_curve(curves)
        ax.plot(mx, my, color="white", linewidth=4.8, zorder=4, solid_capstyle="round")
        ax.plot(mx, my, color=MEAN, linewidth=3.2, marker="D", markevery=20,
                markersize=9.0, markeredgecolor=MEAN, zorder=5, label="Mean")
        title_y = 1.06 if method != "shared_plus_user_history" else 1.0
        ax.set_title(title, fontsize=23, pad=12, y=title_y, linespacing=1.05)
        ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="", ylabel="")
        ax.set_xticks(range(0, 101, 20)); ax.set_yticks(range(0, 101, 20))
        ax.grid(alpha=.30, linewidth=.8); ax.set_axisbelow(True); ax.tick_params(labelsize=19, length=5, width=1.1)
        ax.xaxis.label.set_size(18); ax.yaxis.label.set_size(18)
        if shared_handles is None:
            shared_handles, _ = ax.get_legend_handles_labels()
    fig.legend(shared_handles, ["6L", "10L", "16L", "Mean"],
               loc="upper center", bbox_to_anchor=(.5, 1.015), ncol=4,
               fontsize=21, frameon=True, framealpha=.92, edgecolor="#606060", handlelength=3.0,
               markerscale=1.35, columnspacing=1.5)
    fig.text(.5, .025, "Theoretical FLOPs / Exact (%)", ha="center", fontsize=23)
    fig.text(.012, .47, "AUC gap recovery (%)", va="center", rotation="vertical", fontsize=23)
    fig.subplots_adjust(left=.07, right=.965, bottom=.17, top=.75, wspace=.19)
    fig.savefig(OUT / "five_methods_horizontal.png", dpi=220)
    fig.savefig(OUT / "five_methods_horizontal.pdf")
    plt.close(fig)
    with (OUT / "selected_points.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("method", "scale", "edge", "budget", "flops_percent", "displayed_recovery_percent"))
        writer.writeheader(); writer.writerows(rows)
    (OUT / "README.md").write_text(
        "# Selected five-method figures\n\n"
        "Each figure contains the selected 6L, 10L and 16L version edges plus a muted-gold mean curve. "
        "The mean is computed at common theoretical-FLOPs positions after clipping displayed "
        "recovery to [0,100]. Source data are `../three_scale_auc_preview/cost_ledger.json`; "
        "the selections are recorded in `selection.json`.\n")
    print(f"Wrote {len(METHODS)} figures to {OUT}")


if __name__ == "__main__":
    main()
