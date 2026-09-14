"""Render retained Design 2 detection statistics without model execution."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(path, output):
    metrics=json.loads((path/"summary.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for score,item in metrics["scores"].items():
        cc=item["acceptance"]
        axes[0].plot([v["development_coverage"] for v in cc],[v["severe_rate"] for v in cc],marker=".",label=score)
        bc=item["budget"]
        axes[1].plot([v["actual_cost_fraction"] for v in bc],[v["severe_coverage"] for v in bc],marker=".",label=score)
    axes[0].axhline(metrics["severe_rate"],color="gray",linestyle="--")
    axes[0].set(xlabel="Development retained fraction",ylabel="Severe residual rate",title="Calibration-fixed acceptance thresholds")
    axes[1].set(xlabel="Fraction of hypothetical rebuild FLOPs",ylabel="Severe states covered",title="Offline score / rebuild cost ordering")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/"detection.png",dpi=160)
    fig.savefig(output/"detection.pdf")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("analysis",type=Path)
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[3]/"figures/pic/design2/detection_01")
    args=p.parse_args()
    main(args.analysis,args.output)
