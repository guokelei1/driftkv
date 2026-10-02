"""Plot the fixed Q/H motivation probes from retained results only.

Q is the nonlinear-feature v5 correction; H is the nonlinear v4 all-token map.
Calibration-user budgets are connected only within the same method. No model
execution, outcome filtering, extrapolation, or averaging of unfinished groups.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "figures/src"), str(ROOT / "scripts")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from read_correction_common import EDGES, SCALES, write_csv, sha256, PAIRED_INTEGERS, paired
from read_correction_v3.summarize import validate_point

RESULTS = ROOT / "results/read_correction_2026_09"
BUDGETS = {"query_only": (128, 256, 512), "history_conditioned": (64, 128, 256)}
STYLES = {
    "query_only": {"color": "#0F4D92", "label": "Q: nonlinear Query correction"},
    "history_conditioned": {"color": "#CB742C", "label": "H: explicit history correction"},
}
MARKERS = {64: "v", 128: "o", 256: "s", 512: "^"}
EXPECTED = {(method, scale, edge, budget) for method, budgets in BUDGETS.items()
            for scale in SCALES for edge in EDGES for budget in budgets}
MEAN_FIELDS = ("full_auc", "reuse_auc", "baseline_auc", "auc_delta", "recovery_percent",
               "relative_flops_percent", "calibration_percent", "inference_percent")


def collect(input_root):
    rows, sources, evidence, references = [], {}, [], {}

    def read(path):
        path = path.resolve()
        sources[str(path)] = sha256(path)
        return json.loads(path.read_text())

    def add(original, budget, path):
        point = copy.deepcopy(original)
        validate_point(point)
        key = point["scale"], point["edge"]
        if key in references:
            paired(point, references[key])
        else:
            references[key] = point
        point.update(budget=budget, calibration_users=budget,
                     source_budget=original["budget"], source_path=str(path.resolve()),
                     source_sha256=sources[str(path.resolve())],
                     auc_delta=point["baseline_auc"] - point["reuse_auc"],
                     calibration_percent=100 * point["calibration_flops"] / point["full_minus_reuse_flops"],
                     inference_percent=100 * point["correction_flops"] / point["full_minus_reuse_flops"])
        rows.append(point)

    q_root = RESULTS / "v5/population_run"
    q_plan = read(q_root / "plan.json")["query"]
    if (q_plan["feature_mode"] != "cross_phi" or q_plan["joint_epochs"] != 8
            or q_plan["variant"] != "cross_phi_joint8" or q_plan["state_mode"] != "pure"
            or tuple(q_plan["budgets"]) != BUDGETS["query_only"]):
        raise ValueError("Q source must be the frozen v5 cross_phi joint8 three-budget run")
    q_signatures = set()
    for scale in SCALES:
        for edge in EDGES:
            edge_path = q_root / "evaluation" / scale / edge / "summary.json"
            edge_report = read(edge_path)
            if (edge_report["status"] != "complete" or edge_report["probe_only"]
                    or edge_report["users"] != 3000 or edge_report["partition"] != "evaluation"):
                raise ValueError(f"Q must use the complete fixed panel: {edge_path}")
            points = edge_report["points"]
            keys = {(p["method"], p["variant"], p["budget"]) for p in points}
            expected = {("query_only", "cross_phi_joint8", n) for n in BUDGETS["query_only"]}
            if len(points) != 3 or keys != expected:
                raise ValueError(f"Retain all three original Q-v5 budgets: {scale}/{edge}")
            for point in points:
                if (point["scale"], point["edge"], point["kind"]) != (scale, edge, "measurement"):
                    raise ValueError(f"Q point disagrees with containing directory: {edge_path}")
                score = edge_report["scores"][point["tag"]]
                score_path = Path(score["path"])
                if not score_path.is_absolute():
                    score_path = ROOT / score_path
                if sha256(score_path) != score["sha256"] or score["rows"] != point["requests"]:
                    raise ValueError(f"Q retained scores changed: {score_path}")
                add(point, int(point["budget"]), edge_path)
                rows[-1].update(scores_path=str(score_path.resolve()), scores_sha256=score["sha256"])
            q_signatures.add(json.dumps(edge_report["inputs"]["execution_sources"], sort_keys=True))
            evidence.append({"method": "query_only", "scale": scale, "edge": edge,
                "input_signature": edge_report["input_signature"],
                "inputs": {k: v for k, v in edge_report["inputs"].items() if k != "uids"},
                "scores": edge_report["scores"], "controls": edge_report["controls"]})
    if len(q_signatures) != 1:
        raise ValueError("Q-v5 edges mix execution sources")

    for budget in BUDGETS["history_conditioned"]:
        base = (RESULTS / "v4/population_run/evaluation" if budget == 128
                else input_root / "evaluation" / f"c{budget}")
        for scale in SCALES:
            for edge in EDGES:
                path = base / scale / edge / "summary.json"
                if not path.exists():
                    continue
                report = read(path)
                if report["status"] != "complete":
                    continue
                if report["probe_only"] or report["users"] != 3000 or report["partition"] != "evaluation":
                    raise ValueError(f"Only complete fixed panels enter this figure: {path}")
                points = [p for p in report["points"] if p["variant"] == "map_all"]
                if len(points) != 1 or points[0]["calibration_users"] != budget:
                    raise ValueError(f"Expected one H all-token map at C{budget}: {path}")
                point = points[0]
                if (point["scale"], point["edge"], point["method"]) != (scale, edge, "history_conditioned"):
                    raise ValueError(f"H point disagrees with containing directory: {path}")
                score = report["scores"]["map_all"]
                score_path = Path(score["path"])
                if not score_path.is_absolute():
                    score_path = ROOT / score_path
                if sha256(score_path) != score["sha256"] or score["rows"] != point["requests"]:
                    raise ValueError(f"H retained scores changed: {score_path}")
                add(point, budget, path)
                rows[-1].update(scores_path=str(score_path.resolve()), scores_sha256=score["sha256"])
                evidence.append({"method": "history_conditioned", "scale": scale, "edge": edge,
                    "budget": budget, "input_signature": report["input_signature"],
                    "inputs": {k: v for k, v in report["inputs"].items() if k != "uids"},
                    "scores": score, "controls": report["controls"]})
    keys = {(p["method"], p["scale"], p["edge"], p["budget"]) for p in rows}
    if len(keys) != len(rows) or not keys <= EXPECTED:
        raise ValueError("Unexpected or duplicate point")
    return rows, EXPECTED - keys, sources, evidence


def equal_edge_means(rows):
    means = []
    for scale in ("all", *SCALES):
        required = {(s, e) for s in SCALES if scale in ("all", s) for e in EDGES}
        for method, budgets in BUDGETS.items():
            for budget in budgets:
                group = [p for p in rows if p["method"] == method and p["budget"] == budget
                         and (p["scale"], p["edge"]) in required]
                if {(p["scale"], p["edge"]) for p in group} != required:
                    continue
                means.append({"method": method, "scale": scale, "edge": "equal_edge_mean",
                    "budget": budget, "edges": len(group),
                    **{field: sum(p[field] for p in group) / len(group) for field in MEAN_FIELDS}})
    return means


def limits(points):
    values = [0., 100.] + [p["recovery_percent"] for p in points]
    margin = max(6., (max(values) - min(values)) * .065)
    return min(values) - margin, max(values) + margin


def draw(rows, means, missing):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.2,
        "legend.frameon": False, "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(21, 14))
    grid = fig.add_gridspec(4, 5, height_ratios=(1.15, 1, 1, 1))
    xmax = max(100., max(p["relative_flops_percent"] for p in rows) * 1.04)
    top_ylim = limits(means)
    row_ylims = {s: limits([p for p in rows if p["scale"] == s]) for s in SCALES}

    def panel(ax, points, title, ylim):
        for method, style in STYLES.items():
            points_by_budget = {p["budget"]: p for p in points if p["method"] == method}
            # NaN separates missing budgets; it never interpolates an unmeasured point.
            x = [points_by_budget[n]["relative_flops_percent"] if n in points_by_budget else float("nan")
                 for n in BUDGETS[method]]
            y = [points_by_budget[n]["recovery_percent"] if n in points_by_budget else float("nan")
                 for n in BUDGETS[method]]
            ax.plot(x, y, color=style["color"], linewidth=2.1, zorder=2)
            for n, point in points_by_budget.items():
                ax.scatter(point["relative_flops_percent"], point["recovery_percent"],
                           s=46, marker=MARKERS[n], color=style["color"], edgecolors="white", linewidths=.5, zorder=3)
        ax.axhline(0, color="#CFCECE", linewidth=.8, zorder=0)
        ax.axhline(100, color="#767676", linewidth=.8, linestyle=":", zorder=0)
        ax.scatter(0, 0, marker="X", s=38, color="#4D4D4D", clip_on=False, zorder=4)
        ax.scatter(100, 100, marker="*", s=72, color="#4D4D4D", clip_on=False, zorder=4)
        ax.set(title=title, xlim=(0, xmax), ylim=ylim, xticks=(0, 25, 50, 75, 100))
        ax.tick_params(direction="out", length=3.5, width=1, labelsize=10)
        if len(points) < sum(map(len, BUDGETS.values())):
            ax.text(.97, .04, "H budgets pending", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, color="#767676")

    top_axes = [fig.add_subplot(grid[0, :2])] + [fig.add_subplot(grid[0, i]) for i in (2, 3, 4)]
    for ax, scale in zip(top_axes, ("all", *SCALES)):
        group = [p for p in means if p["scale"] == scale]
        title = "Overall · equal mean of all 15 edges" if scale == "all" else f"{SCALES[scale]} · 5-edge mean"
        panel(ax, group, title, top_ylim)
    top_axes[0].set_ylabel("Gap recovered (%)", fontsize=12)
    for r, (scale, label) in enumerate(SCALES.items(), 1):
        for c, edge in enumerate(EDGES):
            ax = fig.add_subplot(grid[r, c])
            group = [p for p in rows if (p["scale"], p["edge"]) == (scale, edge)]
            panel(ax, group, edge.replace("_to_", " → ").upper(), row_ylims[scale])
            if c == 0:
                ax.set_ylabel(f"{label}\nGap recovered (%)", fontsize=12)
            if r == 3:
                ax.set_xlabel("Extra compute (%)", fontsize=11)
    state = "complete" if not missing else f"PARTIAL · H {sum(p['method'] == 'history_conditioned' for p in rows)}/45 points"
    fig.suptitle(f"Read correction: quality–cost trade-off — {state}", fontsize=19, y=.98)
    handles = [Line2D([], [], color=s["color"], marker="o", linewidth=2, label=s["label"])
               for s in STYLES.values()]
    handles += [Line2D([], [], color="#4D4D4D", marker=m, linestyle="none", markersize=8, label=t)
                for m, t in (("X", "Reuse (0, 0)"), ("*", "Full (100, 100)"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .079), ncol=4, fontsize=12)
    fig.legend(handles=[Line2D([], [], color="#555555", marker=MARKERS[n], linestyle="none", label=f"C{n}")
                        for n in MARKERS], loc="lower center", bbox_to_anchor=(.5, .052), ncol=5, fontsize=10)
    fig.text(.5, .034, "X: (calibration + additional inference) / (Full − Reuse) FLOPs. "
             "Y: (correction AUC − Reuse AUC) / (Full AUC − Reuse AUC).", ha="center", fontsize=10)
    fig.text(.5, .017, "3,000 fixed development users per edge; all 15 edges retained. "
             "Means weight edges equally; Y-ranges are shared within each row. No clipping or extrapolation.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(.006, .115, .994, .955), h_pad=2.1, w_pad=1.5)
    return fig, {"xlim": [0., xmax], "mean_ylim": top_ylim, "edge_ylim_by_scale": row_ylims}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=RESULTS / "motivation_final")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/out/read_correction_2026_09/motivation_final")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows, missing, sources, evidence = collect(args.input_root)
    if args.require_complete and missing:
        raise ValueError(f"Need Q45 + H45 points; {len(missing)} points are still missing")
    means = equal_edge_means(rows)
    fig, axes = draw(rows, means, missing)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "partial" if missing else "complete", "measurement_points": len(rows),
        "expected_points": 90, "counts": {m: sum(p["method"] == m for p in rows) for m in BUDGETS},
        "missing_points": [{"method": m, "scale": s, "edge": e, "budget": b} for m, s, e, b in sorted(missing)],
        "budgets": BUDGETS, "equal_edge_means": means, "axes": axes, "sources": sources, "evidence": evidence,
        "generator_path": str(Path(__file__).resolve()), "generator_sha256": sha256(__file__),
        "method_sources": {
            "query_only": {"revision": "v5", "variant": "cross_phi_joint8", "root": str(RESULTS / "v5/population_run")},
            "history_conditioned": {"revision": "v4", "variant": "map_all", "c128_root": str(RESULTS / "v4/population_run"),
                                    "additional_budgets_root": str(args.input_root.resolve())}},
        "paired_controls_checked": [*PAIRED_INTEGERS, "full_auc", "reuse_auc"],
        "evaluation_role": "outcome-conditioned development exploration",
        "aggregation": "Arithmetic mean of per-edge percentages at each calibration budget; 15 edges overall, 5 per scale. Incomplete groups omitted.",
        "cost_convention": "(complete calibration + additional inference) / (Full-history recompute - rolling Reuse append)",
        "plots": ["overview.png", "overview.pdf"], "data": ["points.csv", "equal_edge_means.csv"]}
    with tempfile.TemporaryDirectory(prefix=".motivation_plot_", dir=args.output_dir) as temporary:
        stage = Path(temporary)
        for suffix in ("png", "pdf"):
            fig.savefig(stage / f"overview.{suffix}", dpi=300)
        plt.close(fig)
        write_csv(rows, stage / "points.csv")
        write_csv(means, stage / "equal_edge_means.csv")
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        table = "\n".join(f"| {p['method']} | {p['budget']} | {p['relative_flops_percent']:.2f}% | {p['recovery_percent']:.2f}% |"
                          for p in means if p["scale"] == "all")
        (stage / "README.md").write_text(f"""# Two-probe motivation overview

Status: **{manifest['status']}**. Q: {manifest['counts']['query_only']}/45 points; H: {manifest['counts']['history_conditioned']}/45 points.

[One large figure](overview.png) · [Vector PDF](overview.pdf) · [Raw point table](points.csv) · [Equal-edge means](equal_edge_means.csv) · [Evidence manifest](manifest.json)

The 19 panels contain the overall mean, three scale means, and all 15 individual edges.
Q uses the retained v5 nonlinear-feature correction (`cross_phi_joint8`) at C128/256/512. H uses the nonlinear v4 all-token map at C64/128/256; C128 is reused from the completed v4 population run. Q's 45 points come from `results/read_correction_2026_09/v5/population_run`; all 45 H points are unchanged from the earlier complete H curve.
Every calibration budget is shown, without picking the best budget per edge. Lines connect actual measurements within each probe. Missing budgets are not filled or extrapolated.

| Probe | Calibration users | Overall compute | Overall gap recovered |
|---|---:|---:|---:|
{table}

Each mean weights edges equally. Recovery is averaged after normalizing each edge by its own Full−Reuse AUC gap; it is not the recovery calculated from pooled predictions or the ratio of averaged AUC differences. The AUC columns in the mean CSV are descriptive edge means.
Incomplete budget groups are omitted from means until all 15 (overall) or five (one scale) edges are present.

Compute includes the complete calibration cost and additional inference FLOPs, divided by Full-history recompute minus rolling Reuse-append FLOPs. Reading an existing fitted artifact does not make its calibration cost zero.
Reuse (0,0) and Full (100,100) are normalization references and are not extra correction measurements.
Negative and above-100% recovery are retained; the top row shares a Y-range and each scale row shares its own Y-range.
All points use the frozen 3,000-user per-edge development panels selected in the earlier exploration. These are outcome-conditioned development results, not an independent population qualification.

Reproduce from saved results:

```bash
python figures/src/read_correction_motivation_2026_09.py --input-root results/read_correction_2026_09/motivation_final --output-dir figures/out/read_correction_2026_09/motivation_final --require-complete
```

The generator reads saved results only, verifies paired Full/Reuse controls and compute denominators, checks every retained Q and H score hash, and records source-summary hashes in the CSV and manifest.
""")
        for path in stage.iterdir():
            path.replace(args.output_dir / path.name)
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"],
                      "output": str(args.output_dir.resolve()), "overall": [p for p in means if p["scale"] == "all"]}, indent=2))


if __name__ == "__main__":
    main()
