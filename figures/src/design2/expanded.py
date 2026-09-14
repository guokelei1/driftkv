"""Expanded cohort evidence, no model calls."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
rows=json.loads((ROOT/'results/design2/scan_30k_01/analysis/summary.json').read_text())['results']
out=ROOT/'results/design2/scan_30k_01/figures';out.mkdir(exist_ok=True)
colors=['#267b72','#bd7445'];labels=['One-shot renewal','Reusable observation']
with plt.rc_context({'font.size':8}):
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.75),layout='constrained')
    for j,policy in enumerate(['frozen','witness']):
        r=next(x for x in rows if x['policy']==policy and x['cohort']=='enriched')
        groups=['short','old_lineage','mixed','rare_activity'];values=[]
        for g in groups:
            x=r['residuals'][g][1];values.append(x['design2']['excess']/x['design1']['excess'])
        bars=ax[0].bar(np.arange(4)+(j-.5)*.35,values,width=.35,color=colors[j],label=labels[j])
        for bar,y in zip(bars,values):ax[0].text(bar.get_x()+bar.get_width()/2,y+.018,f'{y:.2f}',ha='center',fontsize=7)
        r=next(x for x in rows if x['policy']==policy and x['cohort']=='natural');ns=np.geomspace(30000,1e6,200)
        ratio=(r['shared_preparation']+r['costs']['design2']*ns/r['users'])/(r['costs']['exact']*ns/r['users'])
        ax[1].plot(ns,100*ratio,color=colors[j],label=labels[j])
    ax[0].axhline(1,color='black',lw=.8);ax[0].set(ylim=(0,1.08),xticks=range(4),xticklabels=['Short','Old lineage','Mixed','Rare activity'],ylabel='Excess residual / Design 1',title='(a) Enriched cohort (2,034 users)');ax[0].tick_params(axis='x',labelsize=7);ax[0].legend(fontsize=7)
    ax[1].plot([30000,100000,1000000],[40,20,15],'kx',ms=5,label='Working cost envelope')
    ax[1].set(xscale='log',xlabel='Users (workload extrapolation)',ylabel='D1 + D2 FLOPs / Exact (%)',title='(b) Random natural cohort (2,048)');ax[1].legend(fontsize=6);ax[1].grid(alpha=.2)
    for ext in ['pdf','png']:fig.savefig(out/f'expanded.{ext}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.65),layout='constrained')
    for ax,group in zip(axes,['short','old_lineage']):
        for j,policy in enumerate(['frozen','witness']):
            r=next(x for x in rows if x['policy']==policy and x['cohort']=='enriched');q=[x for x in r['quality'] if x['stratum']==group]
            y=np.array([x['auc_difference']*100 for x in q]);ci=np.array([x['auc_interval'] for x in q])*100
            ax.errorbar(np.arange(len(q))+(j-.5)*.10,y,yerr=[y-ci[:,0],ci[:,1]-y],fmt='o',ms=3,color=colors[j],label=labels[j],capsize=2)
        ax.axhline(0,color='black',lw=.8);ax.set(xticks=range(len(q)),xticklabels=[f"M{x['target']} (n={x['uids']})" for x in q],ylabel='AUC change vs Design 1 (pp)',title=group.replace('_',' ').title());ax.tick_params(axis='x',labelsize=6);ax.legend(fontsize=6)
    for ext in ['pdf','png']:fig.savefig(out/f'special_auc.{ext}',dpi=220)
