"""Motivation Figure 2: simulated recompute and reuse card-hours.

Each row is one representative deployment configuration, rather than a cell
in a model-scale-by-cache-length Cartesian product. The numbers are
illustrative placeholders and should be replaced by measured single-GPU
card-hours after the cost experiment is run.

Run from the repository root:

    python figures/src/motivation2.py

Outputs:
    figures/pic/jpg/motivation2.jpg
    figures/pic/pdf/motivation2.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
JPG_OUT = ROOT / "figures" / "pic" / "jpg" / "motivation2.jpg"
PDF_OUT = ROOT / "figures" / "pic" / "pdf" / "motivation2.pdf"

# One row = one representative model/cache configuration. The values are
# deliberately simple placeholders for future measured card-hour data.
CONFIGURATIONS = [
    ("8L · H=256 · KV=1K", 1.0, 10.0),
    ("8L · H=384 · KV=2K", 2.0, 40.0),
    ("24L · H=512 · KV=4K", 5.0, 250.0),
    ("24L · H=1024 · KV=8K", 10.0, 1_200.0),
    ("24L · H=1024 · KV=16K", 20.0, 6_000.0),
    ("32L · H=4096 · KV=16K", 30.0, 18_000.0),
    ("32L · H=4096 · KV=64K", 40.0, 40_000.0),
]


def draw() -> None:
    """Create a paper-style table of simulated card-hour costs."""
    figure, axis = plt.subplots(figsize=(9.2, 4.5), dpi=180)
    axis.axis("off")

    rows = []
    for configuration, reuse_hours, recompute_hours in CONFIGURATIONS:
        rows.append(
            [
                configuration,
                f"{reuse_hours:,.0f}",
                f"{recompute_hours:,.0f}",
                f"×{recompute_hours / reuse_hours:,.0f}",
            ]
        )

    table = axis.table(
        cellText=rows,
        colLabels=[
            "Model / persistent KV configuration",
            "Reuse\n(card-hours)",
            "Recompute\n(card-hours)",
            "Recompute / reuse",
        ],
        colWidths=[0.52, 0.14, 0.17, 0.17],
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.12, 0.96, 0.74],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#bdbdbd")
        cell.set_linewidth(0.7)
        cell.set_height(0.095 if row == 0 else 0.082)
        if row == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold", color="#222222")
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color="#222222")
            if column == 0:
                cell.get_text().set_ha("left")
            if column == 3:
                cell.set_text_props(weight="bold", color="#9c2f22")

    axis.set_title(
        "Recompute incurs substantially higher single-GPU card-hour cost",
        fontsize=15,
        pad=18,
    )
    figure.text(
        0.5,
        0.055,
        "Illustrative values; replace with measured card-hours",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#666666",
    )

    JPG_OUT.parent.mkdir(parents=True, exist_ok=True)
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(JPG_OUT, format="jpg", dpi=300, bbox_inches="tight")
    figure.savefig(PDF_OUT, format="pdf", bbox_inches="tight")
    plt.close(figure)

    print(f"wrote {JPG_OUT}")
    print(f"wrote {PDF_OUT}")


if __name__ == "__main__":
    draw()
