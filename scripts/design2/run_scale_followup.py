"""Reuse the frozen lifecycle replay; demand-time decisions and capped workloads."""
import argparse
import hashlib
import json
import time
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

from design import run as original
from design.data import ROOT,histories,frozen_model,quality_requests,prefix
from design.report_native_flops import prefix_literal,rebuild,value,query_ops,read_cost,append_ops
from design2 import run_lifecycle as base
from design2.common import FrozenC,make_panel,TARGETS
from design2.audit_benchmark import save
from hstu_kvcache.design2.rebuild import current_kv,tiled_flops
from hstu_kvcache.design2.conditional import condition_state,evaluate_state,arithmetic
from hstu_kvcache.design2.geometry import score_layer
from hstu_kvcache.models.state_transition import append_with_rolling_cap
from hstu_kvcache.adaptation.reader import score

CONFIG=ROOT/'configs/design2/scale_followup_01.json'
UIDS=ROOT/'configs/design2/scale_followup_01_uids.json'


def freeze():
    split=json.loads((ROOT/'data/manifests/evokv_design_medium_v0/split.json').read_text())
    old=json.loads((ROOT/'configs/design2/detection_01.json').read_text())
    excluded=set().union(*[set(v) for v in old['groups'].values()])
    for path,key in [('results/design/v15_query_view64_01/configuration.json','calibration_uids'),
                     ('results/design/mechanism_aggregate_factorial192_01/configuration.json','fitting_uids')]:
        excluded.update(json.loads((ROOT/path).read_text())[key])
    for key in ('confirmation','reserved_legacy_confirmation'):excluded.update(split[key])
    available=set(split['calibration'])-excluded
    new=sorted(available,key=lambda u:hashlib.sha256(f'd2-benchmark-extension:17:{u}'.encode()).digest())[:1024]
    assert len(new)==1024 and not set(new)&excluded and not UIDS.exists()
    save(UIDS,dict(original=old['groups']['development'],extension=new,
        note='new to Design2; mature pre-release eligible pool; not formal confirmation or universally unseen model development data'))


class Capped:
    def __init__(self,history,cap):self.history=history;self.rows=history.rows;self.cap=cap
    def prefix(self,u,t,n):return self.history.prefix(u,t,min(n,self.cap))


def install_cap(cap):
    band=original.append_with_rolling_band
    original.append_with_rolling_band=lambda m,c,i,a,d,_:band(m,c,i,a,d,cap)
    append_many=original.append_events_many
    base.append_events_many=lambda m,s,e,l,ch,b:append_many(m,s,e,l,min(ch,cap),b)
    def ops(n,m):
        result=Counter()
        for i in range(0,m,cap):
            step=min(cap,m-i);result.update(append_ops(n,step));n=min(cap,n+step)
        return dict(result)
    base.append_ops=ops


def rebuild_state(model,state,history,uid,stamp,charges):
    i,a,d,times=prefix(history,uid,stamp,next(model.parameters()).device)
    cache=current_kv(model,i,a,d)
    n=cache.seq_len
    assert np.array_equal(times,[e[3] for e in state.writer.events])
    state.rebuild(cache,times,stamp)
    charges['rebuild_tiled']+=tiled_flops(n)
    charges['rebuild_summary']+=base.summary_build_cost(n)


def decide(model,state,adapter,packs,factors,fitted,history,uid,t,stamp,ledger,charges,policy):
    n=state.cache.seq_len;tau=fitted['thresholds']['calibrated']['0.8']
    group=next(g['group'] for g in fitted['groups'] if g['target']==t and g['lower']<=n<=g['upper'])
    a,b=[fitted['final_fit'][group][k] for k in ('a','b')]
    check=base.preparation_cost(state,t)+value(query_ops(n,16))+read_cost(16,1)+arithmetic(16)['panel_flops']
    cost=tiled_flops(n)+base.summary_build_cost(n)
    u=None;fallback=False
    if policy=='hash30':
        reject=int.from_bytes(hashlib.sha256(f'd2-hash30:17:{uid}:{t}'.encode()).digest()[:8],'big')/2**64<.3
        reason='hash_rebuild' if reject else 'hash_continue'
    elif b>tau+1e-10:reject=True;reason='formula_reject'
    elif a==0 and abs(b-tau)>1e-10:reject=False;reason='formula_accept'
    elif policy=='demand' and check>=cost:reject=True;reason='cost_rebuild'
    else:
        p=base.prepare(adapter,state,ledger,charges,'decision_prepare')
        panel=make_panel(history,[SimpleNamespace(uid=uid,state=state)],[0],stamp,state.cache.k.device)
        _,trace,obs,_=adapter.read(model,state.cache,panel,p,[0])
        charges['decision_C_read']+=value(query_ops(n,16))+read_cost(16,1)
        points=[];lows=[];highs=[];valid=True
        for l in range(6):
            matrix=condition_state(p['latent'],packs[l])
            point,lo,hi,good=evaluate_state(matrix,trace.queries[l],obs[l],adapter.parameters[l],packs[l])
            active=p['active'][:,l,None,None].bool();scale=p['counts'].double()[:,None,None]
            for values,dest in [(point,points),(lo,lows),(hi,highs)]:dest.append(torch.where(active,values.sqrt()*scale,0).max())
            valid=valid and bool((good|~active).all())
        charges['decision_geometry']+=arithmetic(16)['panel_flops']
        u=float(torch.stack(points).max());lo=float(torch.stack(lows).max());hi=float(torch.stack(highs).max())
        reject=valid and b+a*lo>tau+1e-10;accept=valid and b+a*hi<=tau-1e-10
        if not (accept or reject):
            fallback=True
            u=float(torch.stack([score_layer(p['latent'],trace.queries[l],obs[l],p['counts'],p['active'][:,l],adapter.parameters[l],factors[l])[0] for l in range(6)]).max())
            charges['decision_fallback']+=arithmetic(16)['exact_panel_flops'];reject=b+a*u>tau
        reason='geometry_reject' if reject else 'geometry_accept'
    if reject:rebuild_state(model,state,history,uid,stamp,charges)
    return dict(uid=uid,target=t,timestamp=stamp,count=n,group=group,u=u,estimate=None if u is None else b+a*u,
        reason=reason,action='rebuild' if reject else 'continue',fallback=fallback,check_cost=check,rebuild_cost=cost)


@torch.no_grad()
def main(cli):
    out=ROOT/'results/design2'/cli.run_id;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_device(cli.device);torch.backends.cuda.matmul.allow_tf32=False
    protocol=json.loads(UIDS.read_text());allusers=protocol['original']+protocol['extension']
    users=allusers[cli.offset:cli.offset+cli.users];assert len(users)==cli.users
    save(out/'configuration.json',dict(vars(cli),uids=users,protocol=json.loads(CONFIG.read_text()),
        hashes={str(p.relative_to(ROOT)):base.sha(p) for p in [CONFIG,UIDS,Path(__file__),ROOT/'src/hstu_kvcache/design2/rebuild.py',ROOT/'configs/design2/scale_calibration_01_fitted.json']}))
    starttime=time.monotonic();ledger=defaultdict(float);raw=[];costs=[];decisions=[];states=[];canaries=[]
    install_cap(cli.cap)
    h=Capped(histories(users,301),cli.cap)
    fitted=json.loads((ROOT/'configs/design2/scale_calibration_01_fitted.json').read_text())
    model=frozen_model(0,cli.device)
    initial=original.cache_at_many(model,h,[(u,231*base.DAY) for u in users],8)
    chains={u:dict(reuse=c,exact=c,design1=base.LifecycleState(c,ts,0),design2=base.LifecycleState(c,ts,0)) for u,(c,ts) in zip(users,initial)}
    del initial,model
    for t in range(1,6):
        start=base.CUTOVER_DAYS[t-1]*base.DAY;stop=(base.CUTOVER_DAYS[t] if t<5 else 301)*base.DAY
        model=frozen_model(t,cli.device);adapter=FrozenC(t,cli.device) if t in TARGETS else None
        packs=factors=None
        if adapter and cli.policy!='hash30':
            packs=torch.load(ROOT/f'results/design2/conditional225_01/inverse/blocks_m{t}.pt',map_location=cli.device,weights_only=True)
            factors=torch.load(ROOT/f'results/design2/geometry_01/cholesky_m{t}.pt',map_location=cli.device,weights_only=True)
        requests=quality_requests(users,start,stop)
        for uid in users:
            chain=chains[uid];charges={m:Counter() for m in base.METHODS};qs=requests.get(uid,[])
            for m in ('design1','design2'):
                chain[m].release(t)
                if t==1:charges[m]['initial_summary']+=base.summary_build_cost(chain[m].cache.seq_len)
            if adapter:
                c,ts=original.cache_at(model,h,uid,start);chain['exact']=c
                charges['exact']['rebuild_closed']+=value(rebuild(c.seq_len))
                if cli.canary:
                    i,a,d,_=prefix(h,uid,start,cli.device);test=current_kv(model,i,a,d)
                    torch.testing.assert_close(test.k,c.k,atol=2e-5,rtol=2e-5);torch.testing.assert_close(test.v,c.v,atol=2e-5,rtol=2e-5)
                    canaries.append(dict(uid=uid,target=t,check='tiled_exact',max_delta=float((test.k-c.k).abs().max())))
                if qs:
                    first=min(int(r['query_timestamp']) for r in qs)
                    _,before=base.replay_user(model,chain,h,uid,[],start,first,t,adapter,ledger,charges,False)
                    decisions.append(decide(model,chain['design2'],adapter,packs,factors,fitted,h,uid,t,first,ledger,charges['design2'],cli.policy))
                    rows,record=base.replay_user(model,chain,h,uid,qs,first,stop,t,adapter,ledger,charges,False)
                    record['events']+=before['events'];record['append_calls']+=before['append_calls']
                    for k,v in before['common_flops'].items():record['common_flops'][k]=record['common_flops'].get(k,0)+v
                else:
                    decisions.append(dict(uid=uid,target=t,action='unused',reason='no_request'))
                    rows,record=base.replay_user(model,chain,h,uid,qs,start,stop,t,adapter,ledger,charges,False)
            else:rows,record=base.replay_user(model,chain,h,uid,qs,start,stop,t,adapter,ledger,charges,False)
            if cli.canary and uid==users[0]:
                # The shared replay has already checked exact writer membership.
                assert all((v.cache if hasattr(v,'cache') else v).seq_len<=cli.cap for v in chain.values())
                canaries.append(dict(uid=uid,target=t,check='cap_and_writer_membership',cap=cli.cap))
            raw.extend(rows);states.append(record);costs.extend(dict(uid=uid,target=t,method=m,**dict(c)) for m,c in charges.items())
        print(json.dumps(dict(target=t,seconds=time.monotonic()-starttime)),flush=True)
        del model,adapter,packs,factors
    pd.DataFrame(raw).to_parquet(out/'quality_raw.parquet',index=False)
    pd.DataFrame(costs).fillna(0).to_parquet(out/'costs.parquet',index=False)
    pd.DataFrame(decisions).to_parquet(out/'decisions.parquet',index=False)
    save(out/'state_counts.json',states)
    save(out/'summary.json',dict(status='complete',seconds=time.monotonic()-starttime,users=len(users),canary_checks=canaries,requests=len(raw),policy=cli.policy,cap=cli.cap))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--run-id');p.add_argument('--users',type=int,default=8);p.add_argument('--offset',type=int,default=0);p.add_argument('--cap',type=int,default=1024);p.add_argument('--policy',choices=['demand','risk','hash30'],default='risk');p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true')
    cli=p.parse_args();freeze() if cli.freeze else main(cli)
