"""One paid observation at the first actual-query evidence excursion."""
import argparse,json,math
from collections import Counter
import numpy as np
import pandas as pd
import torch
from design2 import run_scale_followup as r
from design2 import replay_requests as replay
from design2.audit_benchmark import save
from design.data import prefix
from design.report_native_flops import read_cost,value,query_ops
from hstu_kvcache.design2.geometry import score_layer
from hstu_kvcache.design2.evidence import prepare,observed_score
from hstu_kvcache.design2.conditional import arithmetic
from hstu_kvcache.design2.rebuild import current_kv,tiled_flops
from hstu_kvcache.adaptation.reader import score

MODE=None;PARAMS=None;FACTORS={};BANDS={};STATES={};CHECKS=[];CANARY=False
OLD_LOAD=torch.load
FIRST_USE=set()
PROBE_HOOK=None
COMMIT_FILTER=None
OBSERVATION_HOOK=None
INSTALL_HOOK=None

def load(path,*args,**kwargs):
    if 'conditional225_01/inverse/blocks_' in str(path):return None
    return OLD_LOAD(path,*args,**kwargs)

def hook(model,state,adapter,panel,history,uid,target,stamp,ledger,charges):
    key=(uid,target)
    if state.anchored_native:return None
    charges['excursion_control']+=128
    rec=STATES.setdefault(key,dict(uid=uid,target=target,action='continue',reason='no_excursion',checks=0,evidence_candidates=0,screened=False,u=None))
    if rec.get('attempted'):return None
    n=state.cache.seq_len;q=panel[0].shape[1]
    spec=PARAMS['full_score' if MODE=='excursionfull' else ('background' if MODE=='excursionbg' else 'observed_bound')]
    group=next(g for g in spec['params'] if g.startswith(f'm{target}_') and int(g.split('n')[1].split('-')[0])<=n<=int(g.split('-')[-1]))
    coef=spec['params'][group];a=0 if MODE=='excursionbg' else coef['a'];b=coef['background'] if MODE=='excursionbg' else coef['b']
    rec.update(count=n,group=group,timestamp=stamp,screen_threshold=spec['threshold'])
    if a==0 and b<=spec['threshold']:return None
    p=r.base.prepare(adapter,state,ledger,charges,'service_view')
    z,trace,obs,_=adapter.read(model,state.cache,panel,p,[0]);charges['service_correction']+=read_cost(q,1)
    u=0.
    if a!=0:
        if target not in FACTORS:FACTORS[target]=OLD_LOAD(r.ROOT/f'results/design2/geometry_01/cholesky_m{target}.pt',map_location=state.cache.k.device,weights_only=True)
        if MODE=='excursionfull':
            u=float(torch.stack([score_layer(p['latent'],trace.queries[l],obs[l],p['counts'],p['active'][:,l],adapter.parameters[l],FACTORS[target][l])[0].max() for l in range(6)]).max())
            charges['full_read_evidence']+=arithmetic(q)['exact_panel_flops']
        else:
            if target not in BANDS:BANDS[target]=prepare(FACTORS[target])
            u=float(observed_score(p['latent'],obs,adapter.parameters,p['counts'].double(),p['active'],BANDS[target]).max())
            charges['observed_evidence']+=q*(36*(225**2+3*225+3)+6*192*2+12)
        rec['checks']+=1;rec['evidence_candidates']+=q
    assert math.isfinite(u)
    estimate=b+a*u;rec.update(screen_estimate=estimate,u=u if MODE=='excursionfull' else None)
    CHECKS.append(dict(uid=uid,target=target,timestamp=stamp,queries=q,score=estimate))
    if CANARY:
        original=r.base.read_input(model,state.cache,panel,p['b'],p['a'],adapter.parameters,p['counts'],p['active'],capture=False)[0]
        torch.testing.assert_close(z,original,rtol=0,atol=0)
    if estimate<=spec['threshold']:return z
    i,act,delta,ts=prefix(history,uid,stamp,state.cache.k.device);fresh=current_kv(model,i,act,delta)
    assert np.array_equal(ts,[e[3] for e in state.writer.events])
    new=score(model,fresh,*panel)[0];error=float((z-new).abs().max());commit=error>.5
    if OBSERVATION_HOOK is not None:OBSERVATION_HOOK(uid,target,stamp,panel,z,new,'construction',state)
    charges['rebuild_tiled']+=tiled_flops(n);charges['observation_extra_read']+=value(query_ops(n,q))
    charges['observation_control']+=3*q+128
    if commit and COMMIT_FILTER is not None:
        commit=COMMIT_FILTER(model,state,adapter,p,history,uid,target,stamp,fresh,z,new,charges,rec)
    rec.update(attempted=True,screened=True,committed=commit,commit_error=error,actual_queries=q,renewal_writes=state.writes_since_release,
        action='rebuild' if commit else 'retain_after_paid_probe',reason='observed_commit' if commit else 'observed_retain')
    if commit:
        if INSTALL_HOOK is not None:
            INSTALL_HOOK(model,state,fresh,ts,stamp,key,charges)
            return z
        state.rebuild(fresh,ts,stamp);charges['rebuild_summary']+=r.base.summary_build_cost(n)
        return new
    if PROBE_HOOK is not None:PROBE_HOOK(model,state,fresh,ts,stamp,key,charges)
    return z

def main(cli):
    global MODE,PARAMS,CANARY
    MODE=cli.mode;CANARY=cli.canary
    PARAMS=json.loads((r.ROOT/'configs/design2/evidence_01_fitted.json').read_text())['candidates']
    def first_use_only(*args):
        key=(args[5],args[6])
        if key in FIRST_USE:
            args[9]['excursion_control']+=128
            return None
        FIRST_USE.add(key)
        return hook(*args)
    r.base.replay_user=replay.replay_user;replay.HOOK=first_use_only if MODE=='firstusematched' else hook;torch.load=load
    r.decide=lambda model,state,adapter,packs,factors,fitted,h,uid,t,stamp,ledger,charges,policy:dict(uid=uid,target=t,action='continue',reason='armed')
    cli.policy='risk';r.main(cli)
    out=r.ROOT/'results/design2'/cli.run_id
    d=pd.read_parquet(out/'decisions.parquet')
    save(out/'release_schedule.json',d.to_dict('records'))
    pd.DataFrame([STATES.get((int(row['uid']),int(row['target'])),row) for row in d.to_dict('records')]).to_parquet(out/'decisions.parquet',index=False)
    pd.DataFrame(CHECKS).to_parquet(out/'evidence_reads.parquet',index=False)
    if MODE=='firstusematched':assert not pd.DataFrame(CHECKS).duplicated(['uid','target']).any()
    cfg=json.loads((out/'configuration.json').read_text());cfg.update(sparse_mode=MODE,excursion_protocol=json.loads((r.ROOT/'configs/design2/excursion_01.json').read_text()),
        excursion_sources={str(p.relative_to(r.ROOT)):r.base.sha(p) for p in [r.ROOT/'scripts/design2/run_excursion.py',r.ROOT/'scripts/design2/replay_requests.py']})
    save(out/'configuration.json',cfg)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['excursion','excursionfull','excursionbg','firstusematched'],default='excursion');p.add_argument('--run-id',required=True)
    p.add_argument('--users',type=int,default=8);p.add_argument('--offset',type=int,default=0);p.add_argument('--cap',type=int,default=1024);p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');main(p.parse_args())
