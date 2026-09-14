"""Exact source-conditioned detector costs from retained evidence only."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"figures/pic/design2/conditional225_01"


def main():
    s=json.loads((ROOT/"results/design2/analysis/conditional225_01/summary.json").read_text())
    c=s["roles"]["development"]; entries=s["candidate_counts"]
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    x=np.arange(3)
    online=np.array([c["accurate_original_flops"],c["formula_only_flops"],c["online_flops"]])/1e12
    inv=np.array([0,0,c["inverse_preparation_flops"]])/1e12
    common=np.full(3,c["shared_old_H_preparation"])/1e12
    axes[0].bar(x,online,color=["#9b9eaa","#699558","#128a94"],label="Online checks, all fallbacks included")
    axes[0].bar(x,inv,bottom=online,color="#aa719c",label="New inverse preparation")
    axes[0].bar(x,common,bottom=online+inv,color="#d6b16a",label="Shared original H preparation (leading)")
    for i,v in enumerate(online+inv+common):
        axes[0].text(i,v+.15,f"{v:.2f}",ha="center")
    axes[0].set_xticks(x,["Original\naccurate","Formula shortcut\n+ accurate","Exact source\ncontraction"])
    axes[0].set(ylabel="Ordinary arithmetic (TFLOPs)",title="A. Full development panel, detector accounting")
    axes[0].legend(fontsize=8,loc="upper right")
    axes[0].set_ylim(0,max(online+inv+common)*1.3)
    axes[0].grid(axis="y",alpha=.15)
    q=np.array([r["queries"] for r in entries]); xpos=np.arange(len(q)); width=.24
    lines=[("exact_panel_flops","Accurate per state","#9b9eaa"),
           ("panel_flops","Conditional per state","#128a94")]
    for offset,(key,label,color) in zip((-.25,0),lines):
        axes[1].bar(xpos+offset,[r[key]/1e9 for r in entries],width,color=color,label=label)
    compute=c["states"]-c["stages"].get("formula",0)
    amort=[r["development_no_fallback_total_flops"]/compute/1e9 for r in entries]
    axes[1].bar(xpos+.25,amort,width,color="#aa719c",label="Conditional + new prep amortized")
    axes[1].set_xticks(xpos,[str(v) for v in q])
    axes[1].set(title="B. Reuse across candidates changes the cost",xlabel="Candidates on the same state",
                ylabel="GFLOPs per non-shortcut state")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y",alpha=.15)
    axes[1].text(.03,.5,"1 / 2 / 4 candidates: cost formulas only.\n16 candidates: full numerical/decision audit.",
                 fontsize=8,transform=axes[1].transAxes)
    fig.suptitle("Exact 225D state-conditioned quadratic | Same frozen acceptance set\nCommon C/teacher/source dependencies remain additional; no system-wide cost or quality claim.",fontsize=12)
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/"conditional225.png",dpi=180)
    fig.savefig(OUT/"conditional225.pdf")
    plt.close(fig)


if __name__=="__main__": main()
