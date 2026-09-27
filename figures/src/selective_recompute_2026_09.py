"""Plot measured selective-recomputation points: four methods x three scales.

Each standalone plot contains all five adjacent release edges. This generator
only reads completed experiment summaries; it never starts an experiment or
alters the paper. No clipping, edge selection, monotone envelopes, smoothing,
interpolation, or fabricated measurements are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from selective_recompute_2026_09.cost import normalized_point

METHODS = {
    "layer": "Layer-selective recomputation",
    "tail": "Tail recomputation",
    "deviation": "Deviation-guided recomputation",
    "query": "Query-guided recomputation",
}
SCALES = {"medium": "Medium (6 layers)", "large": "Large (10 layers)", "max": "Max (16 layers)"}
EDGES = tuple(f"v{i}_to_v{i + 1}" for i in range(5))
COLORS = ("#0F4D92", "#42949E", "#52894E", "#B64342", "#9A4D8E")
MARKERS = ("o", "s", "^", "D", "v")
REQUIRED = ("baseline", "scale", "edge", "budget", "full_auc", "reuse_auc", "baseline_auc",
            "extra_flops", "full_minus_reuse_flops")


def read_points(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    points = payload["points"]
    if not points:
        raise ValueError("input contains no measured points")
    records = []
    for point in points:
        missing = set(REQUIRED).difference(point)
        if missing:
            raise ValueError(f"point lacks required fields: {sorted(missing)}")
        if point["baseline"] not in METHODS or point["scale"] not in SCALES or point["edge"] not in EDGES:
            raise ValueError("point identifies an unknown baseline, scale, or adjacent edge")
        record = dict(point)
        record.update(normalized_point(**{key: float(record[key]) for key in REQUIRED[4:]}))
        if any(record[key] is None or not math.isfinite(record[key])
               for key in ("recovery_percent", "relative_flops_percent")):
            raise ValueError("an undefined/nonfinite cost or quality ratio must be resolved, not hidden")
        records.append(record)
    return records


def write_csv(records: list[dict], destination: Path) -> None:
    fields = list(REQUIRED) + sorted(set().union(*(set(row) for row in records)).difference(REQUIRED))
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def draw(records: list[dict], baseline: str, scale: str, output: Path,
         *, diagnostic_label: str = "Selected-user diagnostic") -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 13, "axes.linewidth": 1.5, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "svg.fonttype": "none",
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    handles, labels = [], []
    for index, edge in enumerate(EDGES):
        actual = [row for row in records if row["edge"] == edge and row.get("kind") != "reference"]
        if not actual:
            continue
        actual.sort(key=lambda row: (row["relative_flops_percent"], str(row["budget"])))
        line, = ax.plot([row["relative_flops_percent"] for row in actual],
                        [row["recovery_percent"] for row in actual],
                        color=COLORS[index], marker=MARKERS[index], markersize=6,
                        linewidth=1.8, markeredgewidth=.7, label=f"V{index} → V{index + 1}")
        handles.append(line)
        labels.append(line.get_label())
    references = [row for row in records if row.get("kind") == "reference"]
    if references:
        # These are controls generated from actual matched request metrics.
        # A Full control is separate from a method's full-budget measurement.
        for row in references:
            ax.scatter(row["relative_flops_percent"], row["recovery_percent"],
                       marker="X", color="#4D4D4D", s=49, zorder=4)
        handles.append(Line2D([], [], color="#4D4D4D", marker="X", linestyle="none", markersize=7))
        labels.append("Full / Reuse controls")
    x = [row["relative_flops_percent"] for row in records] + [0.0, 100.0]
    y = [row["recovery_percent"] for row in records] + [0.0, 100.0]
    xpad = max(4, .05 * (max(x) - min(x)))
    ypad = max(5, .07 * (max(y) - min(y)))
    ax.set_xlim(min(x) - xpad, max(x) + xpad)
    ax.set_ylim(min(y) - ypad, max(y) + ypad)
    ax.axhline(0, color="#CFCECE", linewidth=.8, zorder=0)
    ax.axhline(100, color="#767676", linewidth=.8, linestyle="--", zorder=0)
    ax.axvline(100, color="#767676", linewidth=.8, linestyle="--", zorder=0)
    ax.set_xlabel("Full−Reuse FLOP gap consumed (%)")
    ax.set_ylabel("Full−Reuse AUC gap recovered (%)")
    ax.set_title(f"{METHODS[baseline]}\n{SCALES[scale]}", pad=11, fontsize=16)
    ax.tick_params(direction="out", length=4, width=1)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .025),
               ncol=3, fontsize=10.5, handlelength=2.2, columnspacing=1.4)
    fig.text(.5, .008, diagnostic_label, ha="center", fontsize=9, color="#4D4D4D")
    fig.tight_layout(rect=(0, .145, 1, 1), pad=1.4)
    destination = output / baseline
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination / f"{scale}.{suffix}", dpi=300)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/selective_recompute_2026_09/summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "figures/out/selective_recompute_2026_09")
    parser.add_argument("--allow-partial", action="store_true", help="development-only: render available curves and record missing combinations")
    parser.add_argument("--diagnostic-label", default="Selected-user diagnostic")
    args = parser.parse_args(argv)
    records = read_points(args.input)
    present = {(row["baseline"], row["scale"], row["edge"])
               for row in records if row.get("kind") != "reference"}
    required = {(method, scale, edge) for method in METHODS for scale in SCALES for edge in EDGES}
    missing = sorted(required - present)
    if missing and not args.allow_partial:
        raise ValueError(f"missing {len(missing)} baseline/scale/edge curves; formal output requires all60")
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(records, args.out / "all_points.csv")
    plots = []
    for method in METHODS:
        for scale in SCALES:
            subset = [row for row in records if row["baseline"] == method and row["scale"] == scale]
            if not subset:
                continue
            draw(subset, method, scale, args.out, diagnostic_label=args.diagnostic_label)
            plots.append(f"{method}/{scale}")
    manifest = {
        "source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plots": plots, "points": len(records), "missing_curves": missing,
        "complete": not missing, "diagnostic_label": args.diagnostic_label,
        "policy": "All raw points; no clipping, smoothing, interpolation or selected-edge reporting. Controls are separate markers.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "README.md").write_text(
        "# Selective recomputation curves\n\n"
        f"Rendered {len(plots)} baseline/scale figures from `{args.input.resolve()}`. "
        "Each complete figure contains all five adjacent version edges. PNG (300dpi), PDF and SVG share the same data.\n\n"
        "`all_points.csv` preserves every input point and its raw metrics/cost components; "
        "`manifest.json` records the source hash and missing curves, if any. "
        "Negative recovery and costs/recovery above100% remain visible. "
        "Full/Reuse control markers do not replace actual method endpoints.\n\n"
        "Users were selected using existing Full/Reuse outcomes; this is a conditional diagnostic "
        "on the frozen cohort, not an unselected-population quality estimate.\n")
    print(json.dumps({"plots": len(plots), "points": len(records), "missing_curves": len(missing), "out": str(args.out)}))


if __name__ == "__main__":
    main()
