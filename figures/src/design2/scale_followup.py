"""Read-only overview of moderate-scale quality, stress cases and total cost."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]

def main():
    s=json.loads((ROOT/'results/design2/analysis/scale_followup_01/summary.json').read_text())
    b=json.loads((ROOT/'results/design2/analysis/scale_followup_01/benchmark.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');axes=axes.ravel()
    policies=('risk','guard','hash30');colors=('#235789','#59a5a0','#d39d2a')
    for j,(p,color) in enumerate(zip(policies,colors)):
        r=next(r for r in s['reports'] if r['cap']==1024 and r['cohort']=='all' and r['policy']==p)
        y=[100*(r['per_edge'][str(t)]['design2']['ROC_AUC']-r['per_edge'][str(t)]['design1']['ROC_AUC']) for t in (1,3,4,5)]
        axes[0].bar(np.arange(4)+(j-1)*.24,y,.24,label=p,color=color)
    axes[0].set(xticks=np.arange(4),xticklabels=['M1','M3','M4','M5'],ylabel='AUC difference vs D1 (percentage points)',title='Natural workload: 2048 users')
    axes[0].axhline(0,color='gray',lw=.6);axes[0].legend(fontsize=8)
    for j,cap in enumerate((1024,32,128)):
        r=next(r for r in s['reports'] if r['policy']=='risk' and r['cap']==cap and r['cohort']==('all' if cap==1024 else 'stress'))
        y=100*(r['means']['design2']-r['means']['design1'])
        ci=np.array(next(v['interval'] for v in r['uid_intervals'] if v['target']=='equal_edge'))*100
        axes[1].plot([j,j],ci,color='#235789',lw=2);axes[1].scatter(j,y,color='#235789')
    axes[1].set(xticks=range(3),xticklabels=['Natural\n2048 UID','Cap32\n512 UID','Cap128\n512 UID'],title='Quality-only: mean and UID 95% interval',ylabel='AUC difference vs D1 (pp)')
    axes[1].axhline(0,color='gray',lw=.6)
    for p,color in zip(policies,colors):
        rows=[r for r in s['scales'] if r['policy']==p]
        axes[2].plot([r['users'] for r in rows],[100*r['tiled_preparation_ratio'] for r in rows],'-o',label=p,color=color)
    axes[2].set(xscale='log',xlabel='Users (workload extrapolation)',ylabel='Total / Exact closure (%)',title='Validated action + theoretical preparation')
    axes[2].scatter([30000,100000,1000000],[40,23,15],marker='x',color='#b33b46',label='Working targets')
    axes[2].legend(fontsize=8);axes[2].grid(alpha=.15)
    for j,(p,color) in enumerate(zip(policies,colors)):
        r=next(r for r in b['normal'] if r['policy']==p and r['cohort']=='all')
        y=[100*(r['metrics'][str(t)]['design2']['ROC_AUC']-r['metrics'][str(t)]['design1']['ROC_AUC']) for t in (1,3,4,5)]
        axes[3].bar(np.arange(4)+(j-1)*.24,y,.24,label=p,color=color)
    axes[3].set(xticks=np.arange(4),xticklabels=['M1','M3','M4','M5'],ylabel='AUC difference vs D1 (pp)',title='Routine subgroup: M4 regression remains')
    axes[3].axhline(0,color='gray',lw=.6);axes[3].axhline(-.1,color='#b33b46',ls='--',lw=.8,label='Noninferiority margin');axes[3].legend(fontsize=8)
    dest=ROOT/'figures/pic/design2/scale_followup_01';dest.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(dest/f'scale_followup.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
