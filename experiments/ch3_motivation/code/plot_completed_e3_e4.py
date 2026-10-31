#!/usr/bin/env python3
"""Plot audited E3/E4 tables for inspection; do not change paper figures."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "experiments/ch3_motivation/runs"
E3 = RUNS / "e3/main_confirmation_bridge_queue/attempt_1/analysis"
E4 = RUNS / "e4/final_aggregate/attempt_1"
FAMILIES = (("layer", "Layer", "#2563a6"), ("tail", "Tail", "#a15c2f"),
            ("deviation", "Deviation-guided", "#8a4d98"), ("query", "Query-guided", "#198577"))


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def cost_curves(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.7), constrained_layout=True)
    for row, panel in enumerate(("main", "confirmation")):
        current = [r for r in rows if r["panel"] == panel]
        assert len(current) == 24
        for col, (cost, title) in enumerate((
                ("actual_percent_of_once_rebuild", "Actual event ledger"),
                ("normalized_230_percent_of_once_rebuild", "Analytic 230 candidate reads/user"))):
            ax = axes[row, col]
            for prefix, label, color in FAMILIES:
                points = sorted((r for r in current if r["method"].startswith(prefix + "_")),
                                key=lambda r: float(r[cost]))
                ax.plot([float(r[cost]) for r in points],
                        [float(r["recovery_percent"]) for r in points],
                        "o-", label=label, color=color, markersize=4)
            upper = max(100., max(float(r[cost]) for r in current))
            ax.plot([0, upper], [0, upper], "--", color="0.65", linewidth=1, label="Equal recovery / cost")
            ax.scatter([0, 100], [0, 100], color="0.2", marker="s", s=22, zorder=4)
            ax.axhline(0, color="0.8", linewidth=.7)
            ax.set(title=f"{panel.title()} (2,000 users/edge) — {title}",
                   xlabel="Complete added FLOPs / once-Rebuild (%)",
                   ylabel="Recovery of equal-edge mean AUC gap (%)")
            ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("E3: fresh five-edge results; persistent once-per-release repair", fontsize=13)
    fig.supxlabel("Point estimates; recovery intervals are unstable. All five edges retained, including E14_partial.", fontsize=9)
    save(fig, output, "e3_quality_cost")


def edge_gains(rows, output):
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.2), constrained_layout=True, sharey=True)
    for row, panel in enumerate(("main", "confirmation")):
        for edge in range(1, 6):
            ax = axes[row, edge - 1]
            current = [r for r in rows if r["panel"] == panel and int(r["edge"]) == edge]
            assert len(current) == 24
            for prefix, label, color in FAMILIES:
                pts = sorted((r for r in current if r["method"].startswith(prefix + "_")),
                             key=lambda r: float(r["normalized_230_percent_of_once_rebuild"]))
                x = [float(r["normalized_230_percent_of_once_rebuild"]) for r in pts]
                y = np.array([float(r["auc_gain_over_reuse"]) for r in pts])
                lo = np.array([float(r["auc_gain_over_reuse_ci95_low"]) for r in pts])
                hi = np.array([float(r["auc_gain_over_reuse_ci95_high"]) for r in pts])
                ax.plot(x, y, "o-", color=color, label=label, markersize=3, linewidth=1)
                ax.vlines(x, lo, hi, color=color, alpha=.45, linewidth=.9)
            rebuild = next(r for r in current if r["method"] == "rebuild")
            ax.axhline(float(rebuild["auc_gain_over_reuse"]), color="0.25", linestyle="--", linewidth=.8)
            ax.axhline(0, color="0.6", linewidth=.7)
            ax.set_title(f"V{edge - 1}→V{edge}" + (" (partial)" if edge == 5 else ""), fontsize=10)
            ax.grid(alpha=.18)
            if edge == 1:
                ax.set_ylabel(f"{panel.title()}\nAUC gain over Reuse")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("E3: every edge, with paired UID 95% intervals (1,000 draws)", fontsize=13)
    fig.supxlabel("Complete added FLOPs / once-Rebuild (%), analytic 230 reads/user. Dashed line: Rebuild gain.", fontsize=10)
    save(fig, output, "e3_all_edges")


def mismatch(rows, output):
    lookup = {(int(r["edge"]), r["metric"], r["statistic"]): float(r["value"])
              for r in rows if r["group"] == "all" and not r["layer"]}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for ax, suffix, scale, ylabel, reference, ref_label in (
            (axes[0], "top10_row_share", 100., "Top 10% rows' share of mismatch (%)", 16., "Prior expectation: 16%"),
            (axes[1], "max_layer_evenness", 1., "Maximum layer share / uniform share", 1.4, "Prior expectation: 1.4×")):
        for prefix, label, color, offset in (("absolute", "Absolute squared difference", "#2563a6", -.08),
                                             ("relative", "Relative squared difference", "#a15c2f", .08)):
            metric = f"{prefix}_{suffix}"
            med = np.array([lookup[e, metric, "median"] * scale for e in range(1, 6)])
            lo = np.array([lookup[e, metric, "q25"] * scale for e in range(1, 6)])
            hi = np.array([lookup[e, metric, "q75"] * scale for e in range(1, 6)])
            ax.errorbar(np.arange(1, 6) + offset, med, yerr=[med-lo, hi-med], fmt="o", color=color,
                        label=label, capsize=4, markersize=5)
        ax.axhline(reference, color="0.4", linestyle="--", linewidth=1, label=ref_label)
        ax.set_xticks(range(1, 6), [f"V{e-1}→V{e}" for e in range(1, 6)])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.18)
        ax.legend(fontsize=8)
    fig.suptitle("E4: 5,000 fixed users per edge; median and interquartile range", fontsize=13)
    fig.supxlabel("Bars show variation across users, not confidence intervals. Mismatch is not quality contribution.", fontsize=9)
    save(fig, output, "e4_mismatch_distribution")


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    sources = [E3 / "metrics_by_panel.csv", E3 / "metrics_by_edge.csv", E3 / "summary.json",
               E4 / "points.csv", E4 / "summary.json", Path(__file__).resolve()]
    assert json.loads(sources[2].read_text())["state"] == "complete"
    assert json.loads(sources[4].read_text())["status"] == "complete"
    cost_curves(read_csv(sources[0]), output)
    edge_gains(read_csv(sources[1]), output)
    mismatch(read_csv(sources[3]), output)
    manifest = dict(status="complete", scope="Derived review figures only; paper unchanged; no new quality inference",
                    sources_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
                    e3="Exact audited CSV values. Equal-edge AUC gaps, not mean edge recovery. Actual and analytic230 costs separate.",
                    e4="All5,000 users per edge; absolute and relative squared metrics shown separately; IQR is not CI.",
                    files=sorted(p.name for p in output.iterdir() if p.suffix in (".pdf", ".png")))
    (output / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output.resolve())
