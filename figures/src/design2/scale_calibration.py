"""Standalone research plots from frozen summary only; no experiment execution."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "figures/pic/design2/scale_calibration_01"
COLORS = {"raw": "#747b88", "background": "#db8b29", "calibrated": "#087f8c"}
LABELS = {"raw": "Raw geometry", "background": "Target/length background", "calibrated": "Background + geometry"}


def main():
    s = json.loads((ROOT / "results/design2/analysis/scale_calibration_01/summary.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for ax, j, title in ((axes[0, 0], 0, "A. Reuse risk: residual >= 0.5 (primary)"),
                         (axes[0, 1], 1, "B. Reuse risk: residual >= 0.1")):
        for method in LABELS:
            points = s["methods"][method]["acceptance"]
            ax.plot([100*r["coverage"] for r in points], [100*r["severe_rates"][j] for r in points],
                    "o-", color=COLORS[method], label=LABELS[method], markersize=4)
        ax.axhline(100*s["overall_severe_rates"][j], color="#555", ls=":", label="All-state risk")
        ax.set(title=title, xlabel="Retained state coverage (%)", ylabel="Serious residual rate (%)", xlim=(0, 102))
        ax.grid(alpha=.17)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    ax = axes[1, 0]
    for method in (*LABELS, "cost_only"):
        rows = s["budgets"][method]
        ax.plot([100*r["budget"] for r in rows], [100*r["recall"][0] for r in rows], "o-",
                color=COLORS.get(method, "#8056a4"), label=LABELS.get(method, "Cheapest rebuild first"), markersize=4)
    ax.set(title="C. Primary failures covered by rebuild ranking", xlabel="Rebuild-only FLOPs budget (%)",
           ylabel="Weighted severe-state coverage (%)", ylim=(0, 105))
    ax.text(.03, .08, "Geometry checks alone cost 26.12% of Exact-All.\nCurves here exclude checking/preparation.",
            transform=ax.transAxes, fontsize=9, color="#753731")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=.17)
    ax = axes[1, 1]
    groups = s["groups"]
    yy = np.arange(len(groups))
    for offset, method in ((-.17, "background"), (.17, "calibrated")):
        ax.barh(yy+offset, [100*g[method+"_retained_fraction"] for g in groups], height=.32,
                color=COLORS[method], label=LABELS[method])
    ax.set_yticks(yy, [g["group"].replace("_n", " / ").replace("1024-1024", "1024") for g in groups], fontsize=8)
    ax.invert_yaxis()
    ax.set(title="D. Group retention at frozen OOF 80% thresholds", xlabel="Within-group weighted retention (%)", xlim=(0, 103))
    ax.grid(axis="x", alpha=.17)
    fig.suptitle("One fixed scale calibration | 512 calibration + 1024 development UIDs\nNo new model forwards; empirical residual scores, not error bounds", fontsize=13)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "scale_calibration.png", dpi=180)
    fig.savefig(OUT / "scale_calibration.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
