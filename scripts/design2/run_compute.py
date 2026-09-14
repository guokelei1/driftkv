"""First-service read sharing and prepaid renewal; reuse the frozen replay."""
import argparse,json,math,time
from collections import Counter
import numpy as np
import torch
from design.data import ROOT
from design.report_native_flops import rebuild,value,query_ops,read_cost
from design2 import run_scale_followup as r
from design2.audit_benchmark import save
from hstu_kvcache.design2.geometry import score_layer
from hstu_kvcache.design2.conditional import arithmetic

CONFIG=ROOT/'configs/design2/compute_01.json'
old_requests=r.quality_requests;old_history=r.histories;old_score=r.base.corrected_score;old_load=torch.load
HISTORY=None;MODE=None;CAP=None;CREDIT=0.;SPENT=0.;REQUESTS={};MEMO={};RECORDS=[];CHECKS=[]

def history(*args):
    global HISTORY
    HISTORY=old_history(*args);return HISTORY

def requests(users,start,stop):
    global CREDIT
    rows=old_requests(users,start,stop);REQUESTS.clear();REQUESTS.update(rows)
    target=r.base.CUTOVER_DAYS.index(start//r.base.DAY)+1
    if target in r.TARGETS:
        grant=.06*sum(value(rebuild(len(HISTORY.prefix(u,start,CAP)[0]))) for u in users)
        CREDIT+=grant
        RECORDS.append(dict(target=target,grant=grant,cumulative_credit=CREDIT,spent_before=SPENT))
    # Independent users are executed in order of their first arrival. Later
    # requests do not spend the shared wallet, so full per-user replay commutes.
    users.sort(key=lambda u:(min((int(q['query_timestamp']) for q in rows.get(u,[])),default=math.inf),u))
    return rows

def load(path,*args,**kwargs):
    if 'conditional225_01/inverse/blocks_' in str(path):return None
    if MODE=='fifo' and 'geometry_01/cholesky_' in str(path):return None
    return old_load(path,*args,**kwargs)

def corrected(model,state,adapter,panel,ledger,charges):
    saved=MEMO.pop(id(state),None)
    if saved is None:return old_score(model,state,adapter,panel,ledger,charges)
    z,items,dt,revision=saved
    assert revision==state.revision and torch.equal(items,panel[0]) and torch.equal(dt,panel[1])
    CHECKS.append('first_read_reused')
    return z

def decide(model,state,adapter,packs,factors,fitted,h,uid,t,stamp,ledger,charges,policy):
    global SPENT
    assert not MEMO
    n=state.cache.seq_len;tau=fitted['thresholds']['calibrated']['0.8']
    group=next(g['group'] for g in fitted['groups'] if g['target']==t and g['lower']<=n<=g['upper'])
    a,b=[fitted['final_fit'][group][k] for k in ('a','b')]
    rows=[q for q in REQUESTS[uid] if int(q['query_timestamp'])==stamp];q=len(rows);assert q
    charges['controller']+=128
    action_cost=r.tiled_flops(n)+r.base.summary_build_cost(n)
    direct=MODE=='fifo' or b>tau
    accept=MODE!='fifo' and a==0 and b<=tau
    geometry=0 if direct or accept else arithmetic(q)['exact_panel_flops']
    duplicate=0 if direct or accept else value(query_ops(n,q))
    reserve=0 if accept else geometry+action_cost+duplicate
    before=SPENT;u=None;reject=False;reason='formula_accept';cached=None
    if MODE!='readshare' and SPENT+reserve>CREDIT:
        reason='budget_deferred'
    elif direct:
        reject=True;reason='fifo_rebuild' if MODE=='fifo' else 'formula_reject'
    elif not accept:
        p=r.base.prepare(adapter,state,ledger,charges,'service_view')
        panel=(r.original.tensor([[int(v['item_idx']) for v in rows]],state.cache.k.device),
               r.original.tensor([stamp-int(state.writer.events[-1][3])],state.cache.k.device,floating=True))
        z,trace,obs,_=adapter.read(model,state.cache,panel,p,[0])
        charges['service_correction']+=read_cost(q,1)
        u=float(torch.stack([score_layer(p['latent'],trace.queries[l],obs[l],p['counts'],p['active'][:,l],adapter.parameters[l],factors[l])[0].max() for l in range(6)]).max())
        assert math.isfinite(u)
        charges['decision_geometry']+=geometry;SPENT+=geometry
        reject=b+a*u>tau;reason='actual_reject' if reject else 'actual_accept'
        cached=(z,panel[0],panel[1],state.revision)
        if getattr(decide,'canary',False):
            reference=r.base.read_input(model,state.cache,panel,p['b'],p['a'],adapter.parameters,p['counts'],p['active'],capture=False)[0]
            torch.testing.assert_close(z,reference,rtol=0,atol=0);CHECKS.append('capture_same_service_logits')
    if reject:
        r.rebuild_state(model,state,h,uid,stamp,charges);SPENT+=action_cost
        if cached is not None:
            charges['rejected_read_duplicate']+=duplicate;SPENT+=duplicate
    elif cached is not None:MEMO[id(state)]=cached
    if MODE!='readshare':assert SPENT<=CREDIT+1e-3 and SPENT-before<=reserve+1e-3
    return dict(uid=uid,target=t,timestamp=stamp,count=n,group=group,actual_queries=q,u=u,
        estimate=None if u is None else b+a*u,action='rebuild' if reject else 'continue',reason=reason,
        credit=CREDIT,spent=SPENT,charged=SPENT-before,reserved=reserve,unused_reserve=reserve-(SPENT-before) if reason!='budget_deferred' else 0,
        fallback=False)

def main(cli):
    global MODE,CAP
    MODE=cli.mode;CAP=cli.cap;decide.canary=cli.canary
    r.histories=history;r.quality_requests=requests;r.decide=decide;r.base.corrected_score=corrected;torch.load=load
    cli.policy='risk';r.main(cli);assert not MEMO
    out=ROOT/'results/design2'/cli.run_id
    cfg=json.loads((out/'configuration.json').read_text());cfg.update(compute_mode=MODE,compute_config=json.loads(CONFIG.read_text()),
        compute_sha=r.base.sha(__file__),protocol_sha=r.base.sha(CONFIG))
    save(out/'configuration.json',cfg)
    save(out/'wallet.json',dict(mode=MODE,credit=CREDIT,spent=SPENT,unspent=CREDIT-SPENT if MODE!='readshare' else None,
        grants=RECORDS,checks=dict(Counter(CHECKS)),budget_enforced=MODE!='readshare',cell_users=cli.users))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['readshare','reserve','fifo'],required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--users',type=int,default=8);p.add_argument('--offset',type=int,default=0);p.add_argument('--cap',type=int,default=1024)
    p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');main(p.parse_args())
