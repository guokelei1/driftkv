"""Selected Design development evidence; reads completed reports, never runs models.

All three releases are shown. AUC is absolute and never clipped or normalized.
The cost stack uses the fixed real30k event ledger and the once-Rebuild-minus-
Reuse denominator; the quality sample does not determine cost amortization.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parents[2]
SELECTED = "fullitem8_local_kv32_joint16_support_lazy"
RESULTS = ROOT / "results/design_system_2026_10/lightweight_read" / SELECTED
OUTPUT = ROOT / "figures/out/design_system_development"
RELEASES = ("v3", "v4", "v5")
COLORS = ("#0F4D92", "#E7B65A", "#42949E", "#9A4D8E")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    summary_path = RESULTS / "development4096/summary.json"
    calibration_path = RESULTS / "calibration_summary.json"
    summary = json.loads(summary_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    assert summary["status"] == calibration["status"] == "complete"
    assert set(summary["quality"]["by_release"]) == set(RELEASES)
    cost = summary["cost"]
    assert set(cost["by_release"]) == set(RELEASES)

    auc = {
        key: [summary["quality"]["by_release"][v]["metrics"][key]["ROC_AUC"]
              for v in RELEASES]
        for key in ("reuse", "first_request_rebuild", "lightweight")
    }
    components = {
        "Shared preparation": cost["calibration_flops"],
        "First-read code init": sum(
            cost["by_release"][v]["components_flops"]["first_read_initialization"]
            for v in RELEASES),
        "Append maintenance": sum(
            cost["by_release"][v]["components_flops"]["post_initialization_append_encoding"]
            for v in RELEASES),
        "Query increment": sum(cost["by_release"][v]["extra_query_flops"] for v in RELEASES),
    }
    # The row mask is performed with initial/append code preparation, never
    # at query time. Keep all encoding work and split its actual mask work.
    for v in RELEASES:
        stage = cost["by_release"][v]
        cfg = summary["inputs"]["calibration"][v]["config"]
        scalars = cfg["num_layers"] * cfg["heads"] * cfg["rank"]
        initial = scalars * stage["encoded_initial_tokens"]
        appended = scalars * stage["encoded_append_tokens"]
        assert initial + appended == stage["components_flops"]["row_support_mask"]
        components["First-read code init"] += initial
        components["Append maintenance"] += appended
    # Preserve every executed query term, including Key projection/score work.
    for v in RELEASES:
        stage = cost["by_release"][v]
        query = sum(n for k, n in stage["components_flops"].items() if k.startswith("query_"))
        assert query == stage["extra_query_flops"]
        assert sum(stage["components_flops"].values()) == stage["variable_flops"]
    denominator = cost["first_request_rebuild_flops"]
    assert denominator == calibration["first_request_rebuild_flops"]
    assert components["Shared preparation"] == calibration["total_shared_calibration_flops"]
    assert sum(components.values()) == cost["complete_added_flops"]
    assert sum(list(components.values())[1:]) == calibration["fixed_real_trace_total_service_flops"]
    total = 100 * sum(components.values()) / denominator
    assert math.isclose(total, cost["complete_added_percent"], abs_tol=1e-10)
    assert math.isclose(total, calibration["complete_added_percent"], abs_tol=1e-10)

    data = {
        "selected_candidate": SELECTED,
        "releases": list(RELEASES),
        "auc": auc,
        "quality_users": summary["users"],
        "quality_requests": summary["requests"],
        "quality_primary_recovery_percent": summary["quality"]["primary_recovery_percent"],
        "quality_aggregation": summary["quality"]["aggregation"],
        "cost_population_users": cost["population_users"],
        "cost_requests": cost["requests"],
        "cost_denominator": cost["denominator"],
        "cost_denominator_flops": denominator,
        "cost_components_flops": components,
        "cost_components_percent": {k: 100 * n / denominator for k, n in components.items()},
        "complete_added_percent": total,
        "cost_role": cost["role"],
        "source_reconstruction_sensitivity": cost["source_reconstruction_sensitivity"],
    }
    sources = [{"path": str(p), "sha256": digest(p)}
               for p in (summary_path, calibration_path)]
    return data, sources


def draw(data):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": .6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.fontsize": 7.5, "legend.frameon": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(7, 2.05))
    left = fig.add_axes([.078, .245, .355, .56])
    right = fig.add_axes([.565, .245, .408, .40])

    series = (
        ("reuse", "Reuse", "#767676", "s", "--", 1.05),
        ("first_request_rebuild", "Rebuild", "#B64342", "^", "-.", 1.05),
        ("lightweight", "CacheTide", "#0F4D92", "o", "-", 1.5),
    )
    for key, label, color, marker, style, width in series:
        left.plot(range(3), data["auc"][key], color=color, marker=marker,
                  linestyle=style, linewidth=width, markersize=3.6,
                  markeredgewidth=.55, label=label, zorder=3)
    left.set_xticks(range(3), [v.upper() for v in RELEASES])
    values = [v for row in data["auc"].values() for v in row]
    ymin = math.floor((min(values) - .003) * 100) / 100
    ymax = math.ceil((max(values) + .003) * 100) / 100
    left.set(xlim=(-.15, 2.15), ylim=(ymin, ymax), ylabel="AUC", xlabel="Model release")
    left.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    left.grid(axis="y", color="#E3E5E7", linewidth=.5, zorder=0)
    left.legend(loc="lower left", bbox_to_anchor=(-.035, 1.015), ncol=3,
                handlelength=1.6, columnspacing=1.05, handletextpad=.4, borderaxespad=0)

    percentages = list(data["cost_components_percent"].values())
    right.barh(.77, 100, height=.23, color="#E7E8EA", edgecolor="#A5A8AC", linewidth=.5)
    start = 0.
    for value, color in zip(percentages, COLORS):
        right.barh(.22, value, left=start, height=.25, color=color, linewidth=0)
        start += value
    right.text(99, .77, "100%", ha="right", va="center", fontsize=7.5, color="#50545A")
    right.text(start + 3.0, .22, f"{start:.2f}%", va="center", color="#0F4D92",
               fontsize=8.5, fontweight="bold")
    right.axvline(20, color="#B64342", linewidth=.85, linestyle=(0, (3, 2)), zorder=0)
    right.text(21.5, 1.065, "20% budget", color="#B64342", fontsize=7.5, va="center")
    right.set(xlim=(0, 102), ylim=(-.15, 1.16), xlabel="Added FLOPs (% of Rebuild − Reuse)")
    right.set_xticks((0, 20, 40, 60, 80, 100))
    right.set_yticks((.77, .22), ("Rebuild", "CacheTide"))
    right.spines["left"].set_visible(False)
    right.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=color, edgecolor="none") for color in COLORS]
    short_labels = ("Shared prep.", "First-read init", "Append maintenance", "Query")
    labels = [f"{name} ({value:.2f}%)" for name, value in zip(short_labels, percentages)]
    right.legend(handles, labels, loc="lower left", bbox_to_anchor=(-.035, 1.32),
                 ncol=2, columnspacing=.7, handletextpad=.3, handlelength=.75,
                 borderaxespad=0, labelspacing=.5, fontsize=7.5)

    for ax in (left, right):
        ax.tick_params(direction="out", length=2.5, width=.55, pad=2)
    fig.text(.078, .975, "(a) Recommendation quality", va="top", fontsize=8.5, fontweight="bold")
    fig.text(.50, .975, "(b) Complete added compute", va="top", fontsize=8.5, fontweight="bold")
    # Export the requested physical canvas unchanged, rather than auto-cropping.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_sizes = []
    clipped = []
    for item in fig.findobj(Text):
        if not item.get_visible() or not item.get_text():
            continue
        text_sizes.append(item.get_fontsize())
        box = item.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.x1 + .5 or box.y1 > fig.bbox.y1 + .5:
            clipped.append(item.get_text())
    assert min(text_sizes) >= 7.5
    assert not clipped, f"Text outside canvas: {clipped}"
    return fig, {"minimum_font_pt": min(text_sizes), "text_outside_canvas": clipped,
                 "canvas_inches": list(fig.get_size_inches())}


def main():
    data, sources = collect()
    fig, validation = draw(data)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"development_quality_cost.{suffix}", dpi=300)
    plt.close(fig)
    (OUTPUT / "points.json").write_text(json.dumps(data, indent=2) + "\n")
    manifest = {
        "generator": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))},
        "sources": sources,
        "selected_candidate": SELECTED,
        "selection": "Explicit selected development candidate; no automatic endpoint, release or cohort selection.",
        "display": "All three releases; absolute AUC without clipping; cost stack on common 0–100% once-Rebuild-minus-Reuse scale.",
        "cost_scope": "Shared preparation plus first-read initialization, append maintenance and added query work on the fixed real30k ledger. Existing native KV is the input; input-reconstruction sensitivity remains in points.json.",
        "caption": "Development results on the Medium V2–V5 chain. (a) Absolute AUC for Reuse, first-request Rebuild, and CacheTide over all three releases. (b) Complete added FLOPs, including shared preparation and code lifecycle work, normalized to the fixed workload's Rebuild-minus-Reuse cost. The dashed line marks the 20% budget.",
        "validation": validation,
        "outputs": [{"path": str(OUTPUT / f"development_quality_cost.{s}"),
                     "sha256": digest(OUTPUT / f"development_quality_cost.{s}")}
                    for s in ("png", "pdf", "svg")],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "cost_percent": data["complete_added_percent"],
                      "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
