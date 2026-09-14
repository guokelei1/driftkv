"""Read completed lifecycle evidence; never execute models or select policies."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
s=json.loads((ROOT/'results/design2/analysis/lifecycle_variants_01/summary.json').read_text())
out=ROOT/'figures/pic/design2/lifecycle_01';out.mkdir(parents=True,exist_ok=True)
colors=dict(immediate='#087f8c',cost_only='#a67452',deferred='#9a4c91')
names=dict(immediate='Immediate rebuild',cost_only='Cost-only control',deferred='Deferred rebuild')
fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
targets=[1,3,4,5]
for m in colors:
    y=[100*(s['per_edge'][str(t)][m]['ROC_AUC']-s['per_edge'][str(t)]['design1']['ROC_AUC']) for t in targets]
    ax[0].plot(range(4),y,'o-',label=names[m],color=colors[m],lw=1.6)
ax[0].axhline(0,color='gray',lw=.8);ax[0].set_xticks(range(4),[f'M{t}' for t in targets])
ax[0].set(ylabel='AUC difference vs Design1 (percentage points)',title='A. Real chronological feedback')
ax[0].legend(fontsize=8)
for i,m in enumerate(colors):
    delta=100*(s['equal_edge_auc'][m]-s['equal_edge_auc']['design1'])
    r=next(r for r in s['bootstrap']['records'] if r['target']=='equal_edge' and r['left']==m and r['right']=='design1')
    lo,hi=np.array(r['interval'])*100
    ax[1].errorbar(i,delta,yerr=[[delta-lo],[hi-delta]],fmt='o',color=colors[m],capsize=4)
ax[1].axhline(0,color='gray',lw=.8);ax[1].set_xticks(range(3),['Immediate','Cost only','Deferred'])
ax[1].set(title='B. Equal-edge mean; paired UID95% interval',ylabel='AUC difference (percentage points)')
for m in colors:
    ax[2].plot([r['users'] for r in s['scales']],[100*r['methods'][m]['ratio_closed'] for r in s['scales']],
        'o-',color=colors[m],label=names[m])
ax[2].axhline(100,color='gray',ls='--',lw=1);ax[2].set_xscale('log');ax[2].set_yscale('log')
ax[2].set(title='C. Shared preparation + actual lifetime cost',xlabel='Users (same workload mix; extrapolation)',ylabel='% of cheaper Exact KV closure')
for a in ax:
    a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.15)
fig.suptitle('Frozen detector; original1024 developmentUID | Two follow-up policies are development exploration, not confirmation',fontsize=11)
fig.savefig(out/'lifecycle.png',dpi=170);fig.savefig(out/'lifecycle.pdf')
