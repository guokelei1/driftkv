"""Mechanism comparison from retained full replay; no model execution."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'results/design2/bounded_01';OUT=P/'figures';OUT.mkdir(exist_ok=True)
new=json.loads((P/'analysis/summary.json').read_text())['results'];old=json.loads((ROOT/'results/design2/scan_30k_01/analysis/summary.json').read_text())['results']
def row(policy,cohort):return next(r for r in (old if policy=='witness' else new) if r['policy']==policy and r['cohort']==cohort)
names=['witness','bounded_immediate','post','bounded'];labels=['Unbounded','Bounded','Keep response\nunbounded','Keep response\nbounded']
with plt.rc_context({'font.size':8}):
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.9),layout='constrained')
    for j,(cohort,color) in enumerate([('natural','#267b72'),('enriched','#b16f43')]):
        ys=[]
        for policy in names:
            v=row(policy,cohort)['residuals']['all'][1];ys.append(v['design2']['excess']/v['design1']['excess'])
        bars=ax[0].bar(np.arange(4)+(j-.5)*.34,ys,width=.34,color=color,label=cohort.title())
        for b,y in zip(bars,ys):ax[0].text(b.get_x()+b.get_width()/2,y+.015,f'{y:.2f}',ha='center',fontsize=6)
    ax[0].axhline(1,color='black',lw=.8);ax[0].set(ylim=(0,1.17),xticks=range(4),xticklabels=labels,ylabel='Excess residual / Design 1',title='(a) Every real request retained');ax[0].tick_params(axis='x',labelsize=6);ax[0].legend(fontsize=7)
    for policy,color,label in [('witness','#b16f43','Unbounded observation'),('bounded_immediate','#267b72','Bounded observation')]:
        r=row(policy,'natural');ns=np.geomspace(30000,1e6,180);ratio=(r['shared_preparation']+r['costs']['design2']*ns/r['users'])/(r['costs']['exact']*ns/r['users'])
        ax[1].plot(ns,ratio*100,label=label,color=color)
    ax[1].plot([30000,100000,1e6],[40,20,15],'kx',ms=5,label='Working envelope')
    ax[1].set(xscale='log',xlabel='Users (natural workload extrapolation)',ylabel='D1 + D2 FLOPs / Exact (%)',title='(b) Preparation + actual service');ax[1].legend(fontsize=6);ax[1].grid(alpha=.2)
    for ext in ['png','pdf']:fig.savefig(OUT/f'mechanisms.{ext}',dpi=220)
    plt.close(fig)
