"""Compact six-panel motivation figure from completed, retained measurements.

Each scale curve averages its five adjacent edges; Mean averages all 15 edges.
Layer points align by the original requested layer fractions, including the two
fractions that executed the same one-layer Medium configuration. The display
omits the Q-only Max curve and floors displayed recovery at zero. Aggregation
and the saved aggregate table retain every original value, including negatives.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
SCALES = {"medium": 6, "large": 10, "max": 16}
EDGES = tuple(f"v{i}_to_v{i + 1}" for i in range(5))
METHODS = ("layer", "tail", "deviation", "query", "query_only", "history_conditioned")
TITLES = ("(a) Layer", "(b) Tail", "(c) Deviation-guided", "(d) Query-guided",
          "(e) Q correction", "(f) Q + history")
QH_BUDGETS = {"query_only": (128, 256, 512), "history_conditioned": (64, 128, 256)}
STYLES = {
    "all": dict(label="Mean", color="#C5264D", linestyle="-", marker="o", linewidth=1.5),
    "medium": dict(label="Medium", color="#2C6BA0", linestyle="--", marker="s", linewidth=.85),
    "large": dict(label="Large", color="#248477", linestyle=":", marker="^", linewidth=.95),
    "max": dict(label="Max", color="#B55D2E", linestyle="-.", marker="D", linewidth=.85),
}
INSET_COLOR = "#62758A"
INSETS = {
    "query_only": {"bounds": [.37, .24, .59, .63], "xlim": [1, 6], "ylim": [0, 100],
                   "xticks": [2, 4, 6], "yticks": [0, 50, 100],
                   "arrow_from": [.09, .42], "arrow_to": [.37, .43]},
    "history_conditioned": {"bounds": [.18, .20, .52, .34], "xlim": [59, 87], "ylim": [62, 82],
                            "xticks": [60, 70, 80], "yticks": [65, 75],
                            "arrow_from": [.73, .65], "arrow_to": [.60, .54]},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    baseline_path = ROOT / "results/selective_recompute_2026_09/summary.json"
    plan_path = ROOT / "configs/selective_recompute_2026_09/plan.json"
    qh_path = ROOT / "figures/out/read_correction_2026_09/motivation_final/points.csv"
    qh_manifest_path = qh_path.parent / "manifest.json"
    report = json.loads(baseline_path.read_text())
    plan = json.loads(plan_path.read_text())
    if report["status"] != "complete" or report["completed_curves"] != 60:
        raise ValueError("All 60 selective-recompute curves are required")
    baseline = [p for p in report["points"] if p["kind"] == "measurement"]
    with qh_path.open(newline="") as handle:
        qh = list(csv.DictReader(handle))
    if len(baseline) != 295 or len(qh) != 90:
        raise ValueError("Expected the retained 295 baseline and 90 Q/H measurements")
    rows = []
    for original in baseline + qh:
        p = dict(original)
        p["method"] = p.get("method", p["baseline"])
        for key in ("full_auc", "reuse_auc", "baseline_auc", "relative_flops_percent",
                    "recovery_percent", "extra_flops", "full_minus_reuse_flops"):
            p[key] = float(p[key])
        p["requests"], p["users"] = int(p["requests"]), int(p["users"])
        if p["scale"] not in SCALES or p["edge"] not in EDGES or p["users"] != 3000:
            raise ValueError("Unexpected scale, edge, or evaluation panel")
        expected_x = 100 * p["extra_flops"] / p["full_minus_reuse_flops"]
        expected_y = 100 * (p["baseline_auc"] - p["reuse_auc"]) / (p["full_auc"] - p["reuse_auc"])
        if not (math.isclose(expected_x, p["relative_flops_percent"], abs_tol=1e-9)
                and math.isclose(expected_y, p["recovery_percent"], abs_tol=1e-9)):
            raise ValueError("Stored percentages disagree with raw AUC/FLOP fields")
        rows.append(p)
    controls = {}
    for p in rows:
        key = p["scale"], p["edge"]
        current = tuple(p[k] for k in ("full_auc", "reuse_auc", "requests", "users", "full_minus_reuse_flops"))
        if key in controls and current != controls[key]:
            raise ValueError(f"Unpaired Full/Reuse controls: {key}")
        controls[key] = current
    sources = {str(p.resolve()): sha256(p) for p in (baseline_path, plan_path, qh_path, qh_manifest_path)}
    # The existing point table records the exact retained per-edge summaries.
    for p in qh:
        path = Path(p["source_path"])
        if str(path) not in sources:
            sources[str(path)] = sha256(path)
        if sources[str(path)] != p["source_sha256"]:
            raise ValueError(f"Q/H source summary changed: {path}")
    return rows, plan, sources


def aggregate(rows, plan):
    lookup = {(p["method"], p["scale"], p["edge"], str(p["budget"])): p for p in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate measurements")
    means, consumed = [], set()
    for method in METHODS:
        budgets = (plan["layer_fractions"] if method == "layer" else plan["token_fractions"]
                   if method in ("tail", "deviation", "query") else QH_BUDGETS[method])
        for nominal in budgets:
            groups = {}
            for scale, layers in SCALES.items():
                if method == "layer":
                    budget = f"layers_{min(layers - 1, max(1, math.ceil(layers * nominal)))}"
                elif method in ("tail", "deviation", "query"):
                    budget = f"fraction_{nominal:g}"
                else:
                    budget = str(nominal)
                keys = [(method, scale, edge, budget) for edge in EDGES]
                groups[scale] = [lookup[key] for key in keys]
                consumed.update(keys)
            groups["all"] = [p for scale in SCALES for p in groups[scale]]
            for scale in ("all", *SCALES):
                group = groups[scale]
                means.append({
                    "method": method, "scale": scale, "nominal_budget": nominal,
                    "budget_unit": ("layer_fraction" if method == "layer" else "token_fraction"
                                    if method in ("tail", "deviation", "query") else "calibration_users"),
                    "edges": len(group),
                    "relative_flops_percent": sum(p["relative_flops_percent"] for p in group) / len(group),
                    "recovery_percent": sum(p["recovery_percent"] for p in group) / len(group),
                    "executed_budgets": json.dumps({s: sorted({str(p["budget"]) for p in group if p["scale"] == s})
                                                     for s in SCALES if any(p["scale"] == s for p in group)}, sort_keys=True),
                })
    if consumed != set(lookup):
        raise ValueError("Some retained measurements were omitted from aggregation")
    return means


def draw(means):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 6.5,
        "axes.linewidth": .55, "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 6, figsize=(7, 1.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.060, right=.985, bottom=.245, top=.745, wspace=.15)
    xs = [0., 100.] + [p["relative_flops_percent"] for p in means]
    visible = [p for p in means if (p["method"], p["scale"]) != ("query_only", "max")]
    ys = [100.] + [max(0., p["recovery_percent"]) for p in visible]
    xlim = (0., max(xs) * 1.025)
    ylim = (0., math.ceil(max(ys) / 25) * 25 + 4.)

    def plot_curves(ax, method, *, zoom=False):
        for scale in (*SCALES, "all"):
            points = [p for p in visible if p["method"] == method and p["scale"] == scale]
            if not points:
                continue
            style = dict(STYLES[scale])
            if zoom:
                style["linewidth"] *= .8
            ax.plot([p["relative_flops_percent"] for p in points],
                    [max(0., p["recovery_percent"]) for p in points], **style,
                    markersize=(2.5 if scale == "all" else 2.0) * (.85 if zoom else 1), markeredgewidth=.35,
                    zorder=4 if scale == "all" else 3)

    for ax, method, title in zip(axes, METHODS, TITLES):
        plot_curves(ax, method)
        ax.set(title=title, xlim=xlim, ylim=ylim, xticks=(0, 50, 100), yticks=(0, 50, 100))
        ax.set_title(title, fontsize=6.5, pad=3)
        ax.tick_params(direction="out", length=2, width=.5, pad=1.4)
        ax.axhline(0, color="#D2D5D8", linewidth=.45, zorder=0)
        ax.axhline(100, color="#D2D5D8", linewidth=.45, linestyle=":", zorder=0)
        if method in INSETS:
            spec = INSETS[method]
            zoom = ax.inset_axes(spec["bounds"], zorder=5)
            plot_curves(zoom, method, zoom=True)
            zoom.set(xlim=spec["xlim"], ylim=spec["ylim"], xticks=spec["xticks"], yticks=spec["yticks"])
            zoom.set_facecolor("white")
            for spine in zoom.spines.values():
                spine.set_visible(True)
                spine.set_color(INSET_COLOR)
                spine.set_linewidth(.45)
            zoom.tick_params(direction="out", length=1.4, width=.4, pad=1, labelsize=5.2,
                             colors=INSET_COLOR)
            # A single quiet pointer identifies the enlarged region without
            # a second source rectangle or a perspective-like connector frame.
            ax.annotate("", xy=spec["arrow_to"], xytext=spec["arrow_from"],
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=.12",
                                    "color": INSET_COLOR, "lw": .65, "mutation_scale": 5.5,
                                    "shrinkA": 0, "shrinkB": 1.2}, zorder=6)
    handles = [Line2D([], [], **STYLES[scale], markersize=3) for scale in ("all", *SCALES)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1.00), ncol=4,
               borderaxespad=0, fontsize=6.5, handlelength=2.4, columnspacing=1.55, handletextpad=.55)
    fig.text(.53, .035, "Full−Reuse compute gap consumed (%)", ha="center", va="bottom", fontsize=6.5)
    fig.text(.010, .49, "AUC gap recovered (%)", ha="center", va="center", rotation=90, fontsize=6.5)
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer())
    if bounds.x0 < -.001 or bounds.y0 < -.001 or bounds.x1 > 7.001 or bounds.y1 > 1.401:
        raise ValueError(f"Figure text exceeds fixed page bounds: {bounds.bounds}")
    return fig, {"xlim": xlim, "ylim": ylim, "figure_inches": [7, 1.4],
                 "content_bounds_inches": list(bounds.bounds), "shared_x": True, "shared_y": True,
                 "insets": INSETS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "figures/out/motivation_six_methods")
    args = parser.parse_args()
    rows, plan, sources = collect()
    means = aggregate(rows, plan)
    fig, axes = draw(means)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "aggregate_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(means[0]))
        writer.writeheader()
        writer.writerows(means)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(args.out / f"quality_cost.{suffix}", dpi=350)
    plt.close(fig)
    manifest = {
        "sources": sources, "generator": str(Path(__file__).resolve()), "generator_sha256": sha256(Path(__file__)),
        "source_measurements": {"selective_recompute": 295, "query_only": 45, "history_conditioned": 45},
        "aggregate_points": len(means), "methods": list(METHODS), "curves": ["Mean", "Medium", "Large", "Max"],
        "axes": axes,
        "display": {
            "mean_color": STYLES["all"]["color"],
            "inset_axis_color": INSET_COLOR,
            "omitted_curve": {"method": "query_only", "scale": "max"},
            "recovery_floor_percent": 0,
            "floor_applied_after_aggregation": True,
            "mean_includes_omitted_curve": True,
            "omitted_points": [p for p in means if (p["method"], p["scale"]) == ("query_only", "max")],
            "floored_points": [p for p in means if (p["method"], p["scale"]) != ("query_only", "max")
                               and p["recovery_percent"] < 0],
        },
        "aggregation": "Arithmetic mean of per-edge normalized x and y at each original nominal budget: five edges per scale, all 15 for Mean. No pooled predictions or ratio of aggregate AUCs.",
        "layer_alignment": {"nominal_fractions": plan["layer_fractions"],
            "layer_counts": {s: [min(n - 1, max(1, math.ceil(n * f))) for f in plan["layer_fractions"]]
                             for s, n in SCALES.items()},
            "duplicate_policy": "Medium fractions 1/16 and 1/8 execute the same measured one-layer configuration. Reuse its observed point at both nominal budgets; this is not a second measurement or interpolation."},
        "cost_convention": "Extra FLOPs including calibration / (Full history recompute − rolling Reuse append FLOPs).",
        "quality_convention": "(method AUC − Reuse AUC) / (Full AUC − Reuse AUC), normalized per edge before averaging.",
        "policy": "All measurements and edges remain in aggregation and aggregate_points.csv. Display only: omit Q-only Max, floor plotted recovery at zero, and start the y-axis at zero. Mean still uses all 15 original edge values. No smoothing, interpolation, extrapolation, or controls appended to method curves.",
        "evaluation_role": "Outcome-conditioned 3,000-user development panel per edge; one training seed.",
        "plots": [f"quality_cost.{s}" for s in ("pdf", "png", "svg")],
        "data": "aggregate_points.csv",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(args.out), "aggregate_points": len(means), "axes": axes}))


if __name__ == "__main__":
    main()
