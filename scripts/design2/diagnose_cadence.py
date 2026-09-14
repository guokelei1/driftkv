"""Known-failure diagnostic only: did the first-request check miss a later query?"""
import json
import pandas as pd
import torch
from design2 import run_scale_followup as r
from design2.audit_benchmark import save
from design.report_native_flops import read_cost
from hstu_kvcache.design2.geometry import score_layer

OLD_REPLAY=r.base.replay_user;OLD_SCORE=r.base.corrected_score
CURRENT={};FACTORS={};RECORDS=[]
FITTED=json.loads((r.ROOT/'configs/design2/evidence_01_fitted.json').read_text())['candidates']['full_score']

def replay(model,chain,history,uid,qs,start,stop,target,adapter,ledger,charges,*args):
    CURRENT.update(uid=uid,target=target,state=id(chain['design1']))
    return OLD_REPLAY(model,chain,history,uid,qs,start,stop,target,adapter,ledger,charges,*args)

def corrected(model,state,adapter,panel,ledger,charges):
    if id(state)!=CURRENT['state']:return OLD_SCORE(model,state,adapter,panel,ledger,charges)
    p=r.base.prepare(adapter,state,ledger,charges,'service_view')
    z,trace,obs,_=adapter.read(model,state.cache,panel,p,[0]);q=panel[0].shape[1]
    charges['service_correction']+=read_cost(q,1)
    t=CURRENT['target']
    if t not in FACTORS:FACTORS[t]=torch.load(r.ROOT/f'results/design2/geometry_01/cholesky_m{t}.pt',map_location=state.cache.k.device,weights_only=True)
    u=torch.stack([score_layer(p['latent'],trace.queries[l],obs[l],p['counts'],p['active'][:,l],adapter.parameters[l],FACTORS[t][l])[0].amax(0) for l in range(6)]).amax(0)
    group=next(g for g in FITTED['params'] if g.startswith(f'm{t}_') and int(g.split('n')[1].split('-')[0])<=state.cache.seq_len<=int(g.split('-')[-1]))
    a,b=[FITTED['params'][group][k] for k in ['a','b']]
    stamp=int(state.writer.events[-1][3])+int(panel[1][0])
    for j in range(q):RECORDS.append(dict(uid=CURRENT['uid'],target=t,timestamp=stamp,item_idx=int(panel[0][0,j]),u=float(u[j]),estimate=b+a*float(u[j]),threshold=FITTED['threshold'],logit=float(z[0,j]),writes=state.writes_since_release))
    return z

if __name__=='__main__':
    from types import SimpleNamespace
    out=r.ROOT/'results/design2/sparse_01/cadence_diagnostic_v2';out.mkdir(exist_ok=False)
    failures=pd.read_parquet(r.ROOT/'results/design2/analysis/compute_01/reserve_cap1024_2048_errors.parquet')
    uids=sorted(failures.loc[failures.err_design1>.5,'uid'].unique().tolist())
    save(out/'uids.json',dict(original=uids,extension=[],scope='all seven known natural primary-failure UIDs; diagnostic, never detection success-rate evidence'))
    r.UIDS=out/'uids.json';r.base.replay_user=replay;r.base.corrected_score=corrected
    r.decide=lambda model,state,adapter,packs,factors,fitted,h,uid,t,stamp,ledger,charges,policy:dict(uid=uid,target=t,action='continue',reason='diagnostic_only')
    cli=SimpleNamespace(run_id='sparse_01/cadence_diagnostic_v2/replay',users=len(uids),offset=0,cap=1024,device='cuda:0',canary=False,policy='hash30')
    r.main(cli);pd.DataFrame(RECORDS).to_parquet(out/'actual_query_scores.parquet',index=False)
