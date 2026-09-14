"""Read-only computation mechanism comparison, retaining budget deferrals."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
def main():
    s=json.loads((ROOT/'results/design2/analysis/compute_01/summary.json').read_text())['results']
    fig,ax=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    modes=['readshare','reserve','mincost','fifo'];rows=[next(r for r in s if r['mode']==m and r['users']==512 and r['cap']==1024) for m in modes]
    bottom=np.zeros(len(rows))
    for label,keys,color in [('Geometry',['decision_geometry'],'#3a6b94'),('Rebuild',['rebuild_tiled','rebuild_summary'],'#dda53b'),('Other',None,'#80b2a0')]:
        values=np.array([sum(r['components'].get(k,0) for k in keys) if keys else r['costs']['design2']-sum(r['components'].get(k,0) for k in ['decision_geometry','rebuild_tiled','rebuild_summary']) for r in rows])/1e12
        ax[0].bar(modes,values,bottom=bottom,label=label,color=color);bottom+=values
    ax[0].axhline(rows[0]['old_risk_service']/1e12,ls='--',color='#ad3d3d',label='Old risk')
    ax[0].set(title='Same 512 users: service computation',ylabel='Incremental TFLOPs');ax[0].legend(fontsize=8)
    r=max((r for r in s if r['mode']=='reserve' and r['cap']==1024),key=lambda r:r['users'])
    ax[1].plot([x['users'] for x in r['scales']],[100*x['total_ratio'] for x in r['scales']],'-o',label='Reserve, original preparation',color='#3a6b94')
    prep=json.loads((ROOT/'results/design2/lifecycle_01/preparation/summary.json').read_text())['design2_shared']
    ns=np.array([x['users'] for x in r['scales']]);old=(prep+r['old_risk_service']*ns/r['users'])/(r['costs']['exact']*ns/r['users'])
    ax[1].plot(ns,old*100,'--o',color='#ad3d3d',label='Old risk, same cohort/preparation')
    ax[1].set(xscale='log',xlabel='Users (extrapolation)',ylabel='Total / Exact closure (%)',title=f'Full cost: {r["users"]}-UID workload');ax[1].legend(fontsize=8)
    fractions=[]
    for t in [1,3,4,5]:
        a=[a for a in r['actions'] if a['target']==t];active=sum(a['states'] for a in a if a['reason']!='no_request')
        denied=sum(a['states'] for a in a if a['reason']=='budget_deferred');fractions.append(denied/active)
    ax[2].bar(['M1','M3','M4','M5'],np.array(fractions)*100,color='#dda53b')
    ax[2].set(ylabel='Actual first arrivals deferred (%)',title='The budget has a coverage cost')
    out=ROOT/'figures/pic/design2/compute_01';out.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/f'compute.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
