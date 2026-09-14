"""Fixed32 tiered-check diagnostic plots from retained summary only."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"figures/pic/design2/tiered32_01"


def main():
    s=json.loads((ROOT/"results/design2/analysis/tiered32_01/summary.json").read_text())
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
    roles=("residual_calibration","development")
    y=np.arange(2); left=np.zeros(2)
    for stage,label,color in (("formula","Formula shortcut","#678f55"),("bound_accept","Bound accepts","#13989d"),
            ("bound_reject","Bound rejects","#d8a052"),("exact_fallback","Full solve required","#9c9eaa")):
        width=np.array([100*s["roles"][r]["stage_fractions"][stage] for r in roles])
        axes[0].barh(y,width,left=left,color=color,label=label,height=.4)
        if stage=="exact_fallback":
            for i,w in enumerate(width):
                axes[0].text(left[i]+w/2,i,f"{w:.1f}%",ha="center",va="center")
        left+=width
    axes[0].set_yticks(y,["512 calibration UIDs","1024 development UIDs"])
    axes[0].set(xlabel="Share of states (%)",xlim=(0,100),ylim=(-.55,2.05),title="A. Which level decides?")
    axes[0].legend(loc="upper left",fontsize=8,ncol=2)
    axes[0].text(.02,.03,"Frozen detector decisions: zero mismatches\nExisting false negatives remain unchanged.",
        transform=axes[0].transAxes,fontsize=9)
    r=s["roles"]["development"]; spec=s["spectrum"]
    exact=np.array([r["exact_check_baseline_flops"],r["formula_only_flops"],r["fallback_flops"],r["fallback_flops"],r["fallback_flops"]])/1e12
    coarse=np.array([0,0,r["coarse_check_flops"],r["stages"]["bound_accept"]+r["stages"]["bound_reject"]+r["stages"]["exact_fallback"],r["coarse_check_flops"]],dtype=float)
    # Lower-bound column includes only projection GEMMs, not scalar allowance.
    coarse[3]*=r["coarse_panel_projection_flops"]
    coarse/=1e12
    prep=np.array([0,0,0,spec["verified_preparation_gemm_flops"],r["spectral_preparation_model_flops"]])/1e12
    x=np.arange(5)
    axes[1].bar(x,exact,color="#9c9eaa",label="Accurate solves")
    axes[1].bar(x,coarse,bottom=exact,color="#13989d",label="Coarse checks")
    axes[1].bar(x,prep,bottom=exact+coarse,color="#a46b9f",label="New spectral preparation")
    axes[1].axhline(r["exact_check_baseline_flops"]/1e12,color="#753c36",linestyle=":")
    axes[1].set_xticks(x,["Original\naccurate","Formula\nonly","Tiered\nonline","Tiered + prep\nverified lower","Tiered + prep\ncharge model"],fontsize=8)
    for i,v in enumerate(exact+coarse+prep):
        axes[1].text(i,v+.15,f"{v:.2f}",ha="center",fontsize=9)
    axes[1].set(ylabel="Ordinary arithmetic (TFLOPs)",title="B. Development panel: checking and new preparation")
    axes[1].legend(fontsize=8,loc="lower right")
    axes[1].grid(axis="y",alpha=.15)
    fig.suptitle("Fixed 32 inverse directions | Same threshold and acceptance set\nLower bound excludes positive eigensolver work; shared old preparation and input replay remain additional.",fontsize=12)
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/"tiered32.png",dpi=180)
    fig.savefig(OUT/"tiered32.pdf")
    plt.close(fig)


if __name__=="__main__":
    main()
