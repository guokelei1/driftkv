"""Sample fitting, two correction inputs, and one common read update.

The right panel is explicitly a read-level residual view. Q predicts a read
residual; H's equivalent residual is its mapped-history read minus the original
read. The H implementation directly returns the mapped-history read: the
addition shown here is not an additional measured runtime operation.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from compact_recompute_v6 import draw_recompute

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "figures/out/motivation_branches_2026_09_30/v6"
INK = "#272727"
MUTED = "#68707A"
OLD = "#DEE3E9"
BLUE = "#0F4D92"
PALE = "#EDF4FC"
TEAL = "#236B74"


def text(ax, x, y, s, fs=7, c=INK, ha="center", bold=False):
    return ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va="center",
                   weight="bold" if bold else "normal", zorder=9,
                   linespacing=1.1)


def arrow(ax, points, c=MUTED, lw=.7):
    if len(points) > 2:
        ax.plot(*zip(*points[:-1]), color=c, lw=lw, zorder=3,
                solid_capstyle="round", solid_joinstyle="round")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                 mutation_scale=4.1, linewidth=lw, color=c,
                                 linestyle="solid", shrinkA=0, shrinkB=0, zorder=4))


def draw_correction(ax):
    text(ax, 581.5, 155, "(b) Read correction", 8.7, bold=True)

    # Shared few-user calibration above the two input choices. The compact
    # right panel is aligned with the single row of recomputation strategies.
    # Explicit labels replace ambiguous miniature avatars. Fitting produces a
    # shared correction model, not a standalone unnamed weight vector.
    text(ax, 509, 138, "Few users", 7.2, INK, bold=True)
    text(ax, 509, 122, "Reuse / Full reads", 6.2, INK)
    arrow(ax, [(555, 130), (575, 130)], BLUE, 1.1)
    text(ax, 565, 141, "fit", 6.5, BLUE, bold=True)
    ax.add_patch(FancyBboxPatch((579, 118), 115, 27,
        boxstyle="round,pad=0,rounding_size=4", fc=PALE,
        ec=BLUE, lw=.9, zorder=4))
    text(ax, 636.5, 131.5, "Correction model", 7.1, BLUE, bold=True)
    text(ax, 636.5, 109, "shared across users", 6.1, BLUE)

    ax.plot([469, 694], [102, 102], color="#DCE2E8", lw=.6)
    text(ax, 581.5, 96, "Apply to other users", 6.5, bold=True)

    # Two input choices, stated inside spacious parallel lanes. g_H denotes
    # the effective read correction, including its mapped-history read; it
    # does not imply a separately trained direct-residual MLP for H.
    for x, w in ((472, 94), (582, 112)):
        ax.add_patch(FancyBboxPatch((x, 58), w, 32,
            boxstyle="round,pad=0,rounding_size=3", fc="#F6F9FD",
            ec="#CFDBEA", lw=.55, zorder=0))
    text(ax, 519, 82, "Q: query only", 6.9, BLUE, bold=True)
    text(ax, 519, 67, r"$g_{\theta_Q}(q)$", 8.8, BLUE)
    text(ax, 638, 82, "H: + history", 6.9, TEAL, bold=True)
    text(ax, 638, 67, r"$g_{\theta_H}(q,K,V)$", 8.8, TEAL)
    text(ax, 574, 75, "or", 6.3, MUTED)

    # The branches denote alternatives, not two corrections to be summed.
    ax.plot([519, 519, 638, 638], [58, 50, 50, 58],
            color=BLUE, lw=1.05, solid_joinstyle="round", zorder=3)
    arrow(ax, [(578, 50), (578, 31)], BLUE, 1.2)
    text(ax, 594, 42, r"$\Delta r$", 8.8, BLUE)

    # A single visible read backbone: inherited K/V -> old read -> corrected
    # read. The plus is an algebraic read-level view for the H alternative.
    ax.add_patch(FancyBboxPatch((470, 13), 224, 32,
        boxstyle="round,pad=0,rounding_size=4",fc="#F0F5FB",ec="none",zorder=0))
    text(ax, 500, 38, "Old K/V read", 6.6, INK, bold=True)
    text(ax, 500, 24, r"$r_{\mathrm{old}}$", 10, INK)
    arrow(ax, [(521, 24), (571, 24)], INK, 1.2)
    ax.add_patch(Circle((578, 24), 6.5, fc="white", ec=BLUE,
                        lw=1.2, zorder=5))
    text(ax, 578, 24, "+", 10.5, BLUE)
    arrow(ax, [(585, 24), (644, 24)], BLUE, 1.4)
    text(ax, 669, 24, r"$r_{\mathrm{new}}$", 10, BLUE)
    text(ax, 669, 38, "New read", 6.6, BLUE, bold=True)
    text(ax, 578, 7, "Cache unchanged", 6.0, MUTED)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUT)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans", "font.size": 7,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": None})
    fig = plt.figure(figsize=(7, 1.65))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 700), ylim=(0, 165))
    ax.set_axis_off()
    draw_recompute(ax)
    ax.plot([458, 458], [9, 160], color="#CED5DE", lw=.65)
    draw_correction(ax)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.output_dir / f"motivation_branches.{ext}", dpi=400,
                    facecolor="white", bbox_inches=None, pad_inches=0)
    fig.savefig(args.output_dir / "print_preview.png", dpi=150,
                facecolor="white", bbox_inches=None, pad_inches=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overflow = []
    for t in ax.texts:
        b = t.get_window_extent(renderer)
        if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.width or b.y1 > fig.bbox.height:
            overflow.append(t.get_text())
    metadata = {"width_inches": 7, "height_inches": 1.65,
        "width_mm": 177.8, "height_mm": 41.91,
        "selector_layout": "One row: Layer, Tail, Deviation-guided, Query-guided.",
        "selector_width_units": [70, 70, 134, 142],
        "alignment": "Family headings and shared recomputation operation centered within their panels.",
        "selection_scores": "Layer: minimum Full-logit error; Tail: recency; Deviation: first-layer K/V change; Query: first-layer attention magnitude.",
        "font_min_pt": min(t.get_fontsize() for t in ax.texts),
        "text_outside_canvas": overflow,
        "scope": "Illustrative selection scores and read-level correction, not an operator/cost graph.",
        "calibration": "Few-user Reuse/Full pairs; independent Q/H parameters reused across users.",
        "q": "g_Q is a read residual from fixed nonlinear query features followed by an affine map.",
        "h": "g_H = Read(q,T_H(K,V)) - r_old; execution directly returns the mapped-history read.",
        "cost": "No extra original read/subtraction/addition is implied for measured H execution."}
    (args.output_dir / "layout.json").write_text(json.dumps(metadata, indent=2)+"\n")
    caption = (
        "Selective recomputation rebuilds positions chosen by each strategy. "
        "Layer selects a calibrated contiguous span by Full-logit error; Tail selects recent events. "
        "Deviation and Query rank positions by first-layer K/V change and attention magnitude, respectively. "
        "Read correction fits shared correction models on sampled-user Reuse/Full pairs "
        "and applies query-only or history-aware corrections to other users. "
        "The right panel shows the equivalent read-level update r_new = r_old + delta_r. "
        "For H, delta_r denotes the change induced by mapping inherited K/V; "
        "execution directly reads the mapped K/V, without an extra residual-computation pass. "
        "Cache contents stay unchanged. Selection scores and masks are schematic.\n"
    )
    (args.output_dir / "caption.txt").write_text(caption)
    print(json.dumps(metadata, indent=2))
    plt.close(fig)
    if overflow:
        raise RuntimeError(f"Text outside canvas: {overflow}")


if __name__ == "__main__":
    main()
