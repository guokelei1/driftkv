"""Cost envelope and measured service arithmetic; never launches experiments."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'figures/pic/design2/benchmark_01'


def main():
    s=json.loads((ROOT/'results/design2/analysis/benchmark_01/summary.json').read_text())
    life=json.loads((ROOT/'results/design2/analysis/lifecycle_variants_01/summary.json').read_text())
    c=s['cost_curves'];n=np.geomspace(30000,10000000,200)
    k=life['costs']['deferred']['shared_FLOPs']
    service=life['costs']['deferred']['population_FLOPs']/1024
    exact=service/c[0]['service_asymptote']
    saving=s['opportunities']['dependency_closed_rebuild_saving_TF']*1e12/1024
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    ax=axes[0]
    ax.plot(n,100*(k/n+service)/exact,label='Current deferred D1 + D2',color='#235789')
    ax.plot(n,100*(k/n+service-saving)/exact,'--',label='Closed rebuild: unverified what-if',color='#8e6c88')
    ax.scatter([r['users'] for r in c],[100*r['target'] for r in c],marker='x',s=65,color='#ba3b46',label='Benchmark cost targets')
    ax.axhline(100*service/exact,color='#235789',linestyle=':',alpha=.6,label='Current service floor')
    ax.set(xscale='log',xlabel='Users (same workload mix)',ylabel='Total incremental / Exact closure (%)',ylim=(0,58))
    ax.grid(alpha=.18);ax.legend(fontsize=8,loc='upper right')
    ax.set_title('Shared preparation amortizes; service remains')
    components=life['variants']['deferred']['components']
    vals=[components['rebuild_literal'],components['decision_geometry'],components['decision_C_read']]
    vals.append(sum(components.values())-sum(vals))
    labels=['Real rebuild','Geometry','Decision reads','Other D1 + D2 service']
    axes[1].barh(labels,np.array(vals)/1e12,color=['#235789','#59a5a0','#e9b44c','#8e6c88'])
    for i,v in enumerate(vals):axes[1].text(v/1e12+.025,i,f'{v/1e12:.3f}',va='center',fontsize=9)
    axes[1].invert_yaxis();axes[1].set(xlabel='TFLOPs per 1024 users',xlim=(0,2.08))
    axes[1].set_title('Measured service cost: 2.915 TFLOPs')
    axes[1].text(0,-.20,'Shared preparation: 137.710 TFLOPs, included in left panel',transform=axes[1].transAxes,fontsize=8)
    OUT.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf'):fig.savefig(OUT/f'benchmark.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
