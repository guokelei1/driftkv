"""Read retained evidence only: screening, tail recovery, full population costs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]

def main():
    frame=pd.read_parquet(ROOT/'results/design2/evidence_01/analysis/evaluate.parquet')
    records=json.loads((ROOT/'results/design2/sparse_01/analysis/summary.json').read_text())['results']
    modes=['firstusematched','excursionbg','excursion','excursionfull','exactdemand']
    rows=[next(r for r in records if r['cap']==1024 and r['users']==2048 and r['mode']==m) for m in modes]
    full=json.loads((ROOT/'results/design2/sparse_01/analysis/full_reference.json').read_text())['results']
    residual={r['mode']:r for r in full if r['cap']==1024}
    fig,ax=plt.subplots(1,3,figsize=(13.2,3.7),layout='constrained')
    weights=1/frame.groupby('uid').uid.transform('size').to_numpy();bad=frame.max_abs_error.to_numpy()>.5
    for name,label,color in [('source_bound','Source','#497595'),('observed_bound','Source + response','#308769'),('full_score','Full geometry','#ac7044'),('background','Group background','#888888')]:
        order=np.argsort(-frame[name+'_cal'].to_numpy(),kind='stable');x=np.cumsum(weights[order])/weights.sum();y=np.cumsum((weights*bad)[order])/(weights*bad).sum()
        ax[0].plot(np.r_[0,x],np.r_[0,y],label=label,color=color)
    ax[0].set(xlim=(0,.25),ylim=(0,1),xlabel='UID-weighted states screened',ylabel='Severe-state mass captured',title='Held-out development snapshots')
    ax[0].legend(fontsize=7,loc='lower right');ax[0].grid(alpha=.2)
    labels={'firstusematched':'First use (matched)','excursionbg':'Background trigger','excursion':'Renewal (225)','excursionfull':'Full evidence (1281)','exactdemand':'Exact on first use'}
    colors={'firstusematched':'#78a5b9','excursion':'#308769','excursionfull':'#ac7044','excursionbg':'#999999','exactdemand':'#7c6699'}
    base=residual['excursion']['failures'][1]['design1']['excess']
    for i,r in enumerate(rows):
        ax[1].bar(i,residual[r['mode']]['failures'][1]['design2']['excess']/base,color=colors[r['mode']])
    ax[1].axhline(1,color='black',lw=1,label='Design 1');ax[1].axhline(.7,color='#aa5555',ls='--',label='30% excess reduction')
    ax[1].set(xticks=range(len(rows)),xticklabels=[labels[r['mode']] for r in rows],ylabel='Primary excess residual / Design 1',title='Real lifecycle: tail protection')
    ax[1].tick_params(axis='x',labelrotation=35,labelsize=7);ax[1].legend(fontsize=7)
    ns=np.geomspace(30000,1e6,100)
    for r in rows:
        ratio=(r['shared_preparation']+r['costs']['design2']*ns/2048)/(r['costs']['exact']*ns/2048)
        ax[2].plot(ns,100*ratio,label=labels[r['mode']],color=colors[r['mode']])
    ax[2].plot([30000,100000,1000000],[40,20,15],'kx',label='Working envelope')
    ax[2].set(xscale='log',xlabel='Users (workload extrapolation)',ylabel='Full incremental FLOPs / Exact (%)',title='Preparation + actual service')
    ax[2].legend(fontsize=6);ax[2].grid(alpha=.2)
    out=ROOT/'figures/pic/design2/sparse_01';out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:fig.savefig(out/f'renewal.{ext}',dpi=200)
    plt.close(fig)
    # Two-panel paper artifact at final publication width, without the
    # separate snapshot population or unreadably small three-panel labels.
    with plt.rc_context({'font.size':8}):
        fig,axes=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained')
        for i,r in enumerate(rows):
            y=residual[r['mode']]['failures'][1]['design2']['excess']/base
            axes[0].bar(i,y,color=colors[r['mode']]);axes[0].text(i,y+.018,f'{y:.2f}',ha='center',fontsize=7)
            ratio=(r['shared_preparation']+r['costs']['design2']*ns/2048)/(r['costs']['exact']*ns/2048)
            axes[1].plot(ns,100*ratio,label=labels[r['mode']],color=colors[r['mode']])
        axes[0].axhline(1,color='black',lw=.8)
        axes[0].set(ylim=(0,1.12),xticks=range(5),xticklabels=['First use','Background','Renewal','Full','Exact on use'],ylabel='Excess residual / Design 1',title='(a) All real requests, 2,048 users')
        axes[0].tick_params(axis='x',rotation=30,labelsize=7)
        axes[1].plot([30000,100000,1000000],[40,20,15],'kx',ms=5,label='Cost envelope')
        axes[1].set(xscale='log',xlabel='Users (extrapolation)',ylabel='Total incremental FLOPs / Exact (%)',title='(b) Preparation + service')
        axes[1].legend(fontsize=6,loc='upper right');axes[1].grid(alpha=.2)
        for ext in ['pdf','png']:fig.savefig(out/f'renewal_lifecycle.{ext}',dpi=220)
        plt.close(fig)

if __name__=='__main__':main()
