"""Read retained conditional-shuffle curves; no model execution."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(path,output):
    s=json.loads((path/"summary.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    budget=np.array(s["budgets"])
    original=np.array([r["recall"][0] for r in s["original"]])
    cheap=np.array([r["recall"][0] for r in s["cost_only"]])
    axes[0].plot(budget,original,"o-",label="geometry / rebuild cost")
    axes[0].plot(budget,cheap,"o-",label="rebuild cost only")
    for name,label,color in (("within_length_shuffle","within-length shuffle","tab:green"),
                             ("uid_profile_shuffle","UID-profile shuffle","tab:red")):
        rows=s["permutations"][name]["curve"]
        avg=np.array([r["recall_mean"][0] for r in rows])
        lo=np.array([r["recall_p025"][0] for r in rows])
        hi=np.array([r["recall_p975"][0] for r in rows])
        axes[0].plot(budget,avg,".-",label=label,color=color)
        axes[0].fill_between(budget,lo,hi,color=color,alpha=.12)
    overhead=s["cost"]["check_all_over_exact"]+s["cost"]["shared_preparation_over_exact"]
    axes[1].plot(budget+overhead,original,"o-",label="geometry + checks + prep lower bound")
    axes[1].plot(budget,cheap,"o-",label="rebuild cost only")
    axes[1].axvline(overhead,linestyle="--",color="gray",label="all-state check + prep floor")
    axes[0].set(xlabel="Rebuild-only budget / Exact-All",title="Conditional shuffle controls")
    axes[1].set(xlabel="Arithmetic cost lower bound / Exact-All",title="Charge checks for every ranked state")
    for ax in axes:
        ax.set_ylabel("Weighted severe-state coverage (error >= 0.5)")
        ax.set_ylim(0,1.03)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/"budget_audit.png",dpi=160)
    fig.savefig(output/"budget_audit.pdf")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis",type=Path)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[3]/"figures/pic/design2/budget_audit_01")
    args=parser.parse_args()
    main(args.analysis,args.output)
