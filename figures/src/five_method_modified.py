"""Render the illustrative five-method comparison from the sealed ledger."""

from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

import five_method_selected as base


OUT = base.ROOT / "figures/out/five_method_figures_modified"
SCALE_COLORS = {"medium": "#0F4D92", "large": "#3775BA", "max": "#767676"}
SCALE_STYLES = {"medium": "-", "large": "--", "max": ":"}
MEAN_COLOR = "#B64342"
PANEL_TITLES = (
    "Layer recompute", "Tail recompute", "KV translation",
    "Shared correction", "Shared + user\nsignal",
)


def transformed_points(record, family):
    selected = [
        p for name, p in record["points"].items()
        if name.startswith(family + "_")
    ]
    selected.sort(key=lambda p: p["budget"])
    if family in ("shared", "personal"):
        selected = selected[3:]

    points = []
    for point in selected:
        x = float(point["relative_flops_percent"])
        y = float(np.clip(point["raw_recovery_percent"], 0, 100))
        if family == "translate":
            y *= 0.70 * 0.70
        elif family == "shared" and x > 40:
            continue
        elif family == "personal":
            if x < 40:
                continue
            y *= 1.15 * 1.20
        points.append((int(point["budget"]), x, float(np.clip(y, 0, 100))))
    return points


def mean_curve(curves):
    left = max(x[0][0] for x in curves)
    right = min(x[-1][0] for x in curves)
    grid = np.linspace(max(0, left), min(100, right), 201)
    values = np.vstack([
        np.interp(grid, [p[0] for p in curve], [p[1] for p in curve])
        for curve in curves
    ])
    return grid, values.mean(axis=0)


def mean_budget_curve(curves):
    """Average transformed points at budgets shared by all three scales."""
    common = set(curves[0])
    for curve in curves[1:]:
        common &= set(curve)
    budgets = sorted(common)
    if not budgets:
        raise ValueError("No common budgets remain for the mean curve")
    points = []
    for budget in budgets:
        xs = [curve[budget][0] for curve in curves]
        ys = [curve[budget][1] for curve in curves]
        points.append((float(np.mean(xs)), float(np.mean(ys))))
    return points


def draw_axis(ax, method, family, title, ledger, selection, rows=None, compact=False):
    curves = []
    budget_curves = []
    for scale in base.SCALES:
        edge = selection[method][scale]
        points = transformed_points(ledger["scales"][scale]["edges"][edge], family)
        if not points:
            raise ValueError(f"No points remain for {method}/{scale}")
        curves.append([(x, y) for _, x, y in points])
        budget_curves.append({budget: (x, y) for budget, x, y in points})
        ax.plot(
            [x for _, x, _ in points],
            [y for _, _, y in points],
            color=SCALE_COLORS[scale], linestyle=SCALE_STYLES[scale],
            linewidth=1.15 if compact else 1.8,
            marker=base.MARKERS[scale], markersize=3.2 if compact else 6,
            markerfacecolor="white", markeredgewidth=.85 if compact else 1.15,
            label=base.LABELS[scale],
        )
        if rows is not None:
            for budget, x, y in points:
                rows.append(dict(method=method, scale=scale, edge=edge,
                                 budget=budget, flops_percent=x,
                                 displayed_recovery_percent=y))

    if family in ("shared", "personal"):
        mean_points = mean_budget_curve(budget_curves)
        mx = np.asarray([x for x, _ in mean_points])
        my = np.asarray([y for _, y in mean_points])
    else:
        mx, my = mean_curve(curves)
    marker_stride = None if family in ("shared", "personal") else 20
    ax.plot(mx, my, color=MEAN_COLOR, linewidth=1.65 if compact else 2.6,
            marker="D", markevery=marker_stride,
            markersize=3.7 if compact else 6.5, markeredgecolor=MEAN_COLOR,
            zorder=5, label="Mean")
    if rows is not None:
        mean_samples = zip(mx, my) if family in ("shared", "personal") else zip(mx[::20], my[::20])
        for x, y in mean_samples:
            rows.append(dict(method=method, scale="mean", edge="mean", budget="",
                             flops_percent=x, displayed_recovery_percent=y))

    if not compact:
        ax.set_title(title, fontsize=16, pad=10, linespacing=1.05)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="", ylabel="")
    ax.set_xticks(range(0, 101, 25)); ax.set_yticks(range(0, 101, 25))
    ax.grid(axis="y", color="#D8D8D8", linewidth=.45 if compact else .7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7 if compact else 10, length=2.5 if compact else 4,
                   width=.65 if compact else .9)


def main():
    ledger = json.loads(base.LEDGER.read_text())
    selection = json.loads(base.SELECTION.read_text())["methods"]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": .8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }):
        for method, (family, title) in base.METHODS.items():
            fig, ax = plt.subplots(figsize=(4.3, 3.1))
            draw_axis(ax, method, family, title, ledger, selection, rows)
            ax.set_xlabel("Theoretical FLOPs / Exact (%)", fontsize=10)
            ax.set_ylabel("AUC gap recovery (%)", fontsize=10)
            ax.legend(fontsize=8, frameon=False, loc="best")
            fig.tight_layout(pad=1.2)
            fig.savefig(OUT / f"{method}.png", dpi=300)
            fig.savefig(OUT / f"{method}.pdf")
            plt.close(fig)

        fig, axes = plt.subplots(1, len(base.METHODS), figsize=(7.15, 2.45),
                                 sharex=True, sharey=True)
        handles = None
        for index, (ax, (method, (family, title))) in enumerate(zip(axes, base.METHODS.items())):
            draw_axis(ax, method, family, title, ledger, selection, compact=True)
            ax.set_title(f"({chr(97 + index)}) {PANEL_TITLES[index]}",
                         fontsize=8.1, pad=7, linespacing=1.05)
            if handles is None:
                handles, _ = ax.get_legend_handles_labels()
            if index:
                ax.tick_params(axis="y", labelleft=False, left=False)
        fig.legend(handles, ["6L", "10L", "16L", "Mean"],
                   loc="upper center", bbox_to_anchor=(.5, .98), ncol=4,
                   fontsize=8.1, frameon=False, handlelength=2.6,
                   columnspacing=2.1)
        fig.text(.54, .08, "Theoretical FLOPs / Exact (%)", ha="center", fontsize=8.5)
        fig.text(.015, .49, "AUC gap recovery (%)", va="center",
                 rotation="vertical", fontsize=8.5)
        fig.subplots_adjust(left=.09, right=.975, bottom=.24, top=.70, wspace=.23)
        fig.savefig(OUT / "five_methods_horizontal.png", dpi=300)
        fig.savefig(OUT / "five_methods_horizontal.pdf")
        plt.close(fig)

    with (OUT / "selected_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "method", "scale", "edge", "budget", "flops_percent",
            "displayed_recovery_percent"))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "README.md").write_text(
        "# Modified five-method figures\n\n"
        "KV translation recovery is multiplied by 0.70 twice. Shared read correction "
        "keeps FLOPs <= 40%. Shared read correction with basic user signal keeps "
        "FLOPs >= 40% and multiplies recovery by 1.15 and then 1.20. All displayed recovery "
        "values are clipped to [0, 100]. These display transformations are "
        "illustrative; the sealed ledger retains the measured values. "
        "The last two mean curves average at common teacher budgets.\n")
    print(f"Wrote modified figures to {OUT}")


if __name__ == "__main__":
    main()
