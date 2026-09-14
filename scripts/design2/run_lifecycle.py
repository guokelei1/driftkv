"""Four independent M0--M5 native-write lifetimes with frozen release decisions."""

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from design.data import ROOT, DAY, diagnostic_admissions, frozen_model, histories, quality_requests
from design.diagnose_native_input import read_input
from design.run import append_events, append_events_many, cache_at, cache_at_many, event_range, tensor, timed, write_json
from design.report_native_flops import V, append_ops, prefix_literal, rebuild, query_ops, value, read_cost, prepare_cost
from design2.common import FrozenC, TARGETS, make_panel
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.design2.lifecycle import LifecycleState
from hstu_kvcache.design2.conditional import condition_state, evaluate_state, arithmetic
from hstu_kvcache.design2.geometry import score_layer
from insight_two.common import CUTOVER_DAYS

CONFIG=ROOT/'configs/design2/lifecycle_01.json'
METHODS=('reuse','design1','exact','design2')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def preparation_cost(state, target):
    ops=prepare_cost(target,len(state.writer.segments))
    dim=(target+1)*(V+1)+1
    # FrozenC also computes its diagnostic source norm and total count.
    return sum(ops[k] for k in ('pack','features','pca','view'))+2*dim-1+len(state.writer.segments)-1


def summary_build_cost(n):
    # Literal SummaryWriter.from_cache pads its single slot to1024, even for short history.
    return V*1023 if n else 0


def prepare(adapter,state,ledger,charges,label):
    if state.prepared is None:
        charges[label]+=preparation_cost(state,adapter.target)
        scene=SimpleNamespace(state=state)
        state.prepared=timed(lambda:adapter.prepare([scene]),ledger,label)
        state.prepared_revision=state.revision
    assert state.prepared_revision==state.revision
    return state.prepared


def current_rebuild(model,state,history,uid,cutover,ledger,charges,label):
    n=state.cache.seq_len
    cache,times=timed(lambda:cache_at(model,history,uid,cutover),ledger,label)
    assert len(times)==n and np.array_equal(times,[e[3] for e in state.writer.events])
    charges['rebuild_literal']+=value(prefix_literal(n))
    charges['rebuild_dependency_closed']+=value(rebuild(n))  # alternate denominator; do not sum twice
    charges['rebuild_summary']+=summary_build_cost(n)
    timed(lambda:state.rebuild(cache,times,cutover),ledger,'rebuild_summary')
    return cache,times


def decide(model,state,adapter,packs,factors,fitted,history,uid,target,cutover,ledger,charges,canary):
    n=state.cache.seq_len
    group=next(g['group'] for g in fitted['groups'] if g['target']==target and g['lower']<=n<=g['upper'])
    a,b=[fitted['final_fit'][group][k] for k in ('a','b')]
    tau=fitted['thresholds']['calibrated']['0.8']
    check=preparation_cost(state,target)+value(query_ops(n,16))+read_cost(16,1)+arithmetic(16)['panel_flops']
    rebuild_cost=value(prefix_literal(n))+summary_build_cost(n)
    row=dict(uid=uid,target=target,count=n,group=group,a=a,b=b,check_remaining=check,
        rebuild_remaining=rebuild_cost,revision=state.revision,release_age=state.writes_since_release,
        previous_rebuild=None if state.last_rebuild is None else state.last_rebuild['target'],
        producer_counts={str(k):v for k,v in Counter(state.writer.segments[e[0]]['producer'] for e in state.writer.events).items()},
        u=None,estimate=None,numerical_fallback=False,accurate_max_relative_delta=None)
    if b>tau+1e-10:
        row.update(action='rebuild',reason='formula_reject')
    elif a==0 and abs(b-tau)>1e-10:
        row.update(action='continue',reason='formula_accept')
    elif check>=rebuild_cost:
        row.update(action='rebuild',reason='cost_before_check')
    else:
        p=prepare(adapter,state,ledger,charges,'decision_prepare')
        scene=SimpleNamespace(uid=uid,state=state)
        panel=make_panel(history,[scene],[0],cutover,state.cache.k.device)
        _,trace,obs,_=timed(lambda:adapter.read(model,state.cache,panel,p,[0]),ledger,'decision_C_read')
        charges['decision_C_read']+=value(query_ops(n,16))+read_cost(16,1)
        points,lows,highs,bad=[],[],[],[]
        for layer in range(6):
            matrix=condition_state(p['latent'],packs[layer])
            point,lo,hi,valid=evaluate_state(matrix,trace.queries[layer],obs[layer],adapter.parameters[layer],packs[layer])
            active=p['active'][:,layer,None,None].bool()
            scale=p['counts'].double()[:,None,None]
            points.append(torch.where(active,point.sqrt()*scale,0).amax(1))
            lows.append(torch.where(active,lo.sqrt()*scale,0).amax(1))
            highs.append(torch.where(active,hi.sqrt()*scale,0).amax(1))
            bad.append(((~valid)&active).any())
        charges['decision_geometry']+=arithmetic(16)['panel_flops']
        u=float(torch.stack(points).max()); lo=float(torch.stack(lows).max()); hi=float(torch.stack(highs).max())
        valid=not bool(torch.stack(bad).any())
        accept=valid and b+a*hi<=tau-1e-10
        reject=valid and b+a*lo>tau+1e-10
        uncertain=not (accept or reject)
        need_solve=uncertain and arithmetic(16)['exact_panel_flops']<rebuild_cost
        if canary or need_solve:
            exact=torch.stack([score_layer(p['latent'],trace.queries[l],obs[l],p['counts'],p['active'][:,l],adapter.parameters[l],factors[l])[0] for l in range(6)]).amax(0)
            exact_u=float(exact.max())
            if valid:
                torch.testing.assert_close(torch.stack(points).amax(0),exact,rtol=1e-8,atol=1e-8)
                assert lo<=exact_u<=hi
                row['accurate_max_relative_delta']=abs(u-exact_u)/max(1,exact_u)
            if need_solve:
                charges['decision_accurate_fallback']+=arithmetic(16)['exact_panel_flops']
                u=exact_u; accept=b+a*u<=tau; reject=not accept
            else:
                charges['validation_extra_geometry']+=arithmetic(16)['exact_panel_flops']
        row.update(u=u if np.isfinite(u) else None,estimate=b+a*u if np.isfinite(u) else None,
            numerical_fallback=uncertain,action='continue' if accept else 'rebuild',
            reason='geometry_accept' if accept else ('geometry_reject' if reject else 'cost_before_fallback'))
    if row['action']=='rebuild':
        current_rebuild(model,state,history,uid,cutover,ledger,charges,'design2_rebuild')
    row.update(rebuilds=state.rebuilds,post_revision=state.revision,anchored_native=state.anchored_native)
    return row


def corrected_score(model,state,adapter,panel,ledger,charges):
    p=prepare(adapter,state,ledger,charges,'service_view')
    q=panel[0].numel()
    charges['service_correction']+=read_cost(q,1)
    return read_input(model,state.cache,panel,p['b'],p['a'],adapter.parameters,p['counts'],p['active'],capture=False)[0]


def state_check(state):
    assert len(state.writer.events)==state.cache.seq_len
    assert sum(sum(s['count']) for s in state.writer.segments.values())==state.cache.seq_len
    assert state.prepared is None or state.prepared_revision==state.revision


def replay_user(model,chain,history,uid,requests,start,stop,target,adapter,ledger,charges,canary):
    events=event_range(history,uid,start,stop)
    ev,req=defaultdict(list),defaultdict(list)
    for e in events:ev[e[0]].append(e)
    for r in requests:req[int(r['query_timestamp'])].append(r)
    last=int(chain['design1'].writer.events[-1][3]); pending=[]; raw=[]
    reads=0; appends=0; append_calls=0; shared_flops=Counter()
    def flush():
        nonlocal last,appends,append_calls
        for offset in range(0,len(pending),128):
            chunk=pending[offset:offset+128]; m=len(chunk); n=chain['reuse'].seq_len
            before={name:chain[name].counts['evictions'] for name in ('design1','design2')}
            inputs=[chain[name] for name in METHODS]
            # Each branch has its own KV. Only the kernel launch is batched.
            output=timed(lambda:append_events_many(model,inputs,[chunk]*4,[last]*4,128,4),ledger,'four_independent_appends')
            chain.update(zip(METHODS,output))
            for name in ('design1','design2'):
                state=chain[name]; removed=state.counts['evictions']-before[name]
                charges[name]['summary_maintenance']+=V*(m+removed)
                if canary:state_check(state)
            lengths=[(c.cache if isinstance(c,LifecycleState) else c).seq_len for c in chain.values()]
            assert len(set(lengths))==1
            shared_flops['native_append']+=value(append_ops(n,m))
            last=chunk[-1][0]; appends+=m; append_calls+=1
        pending.clear()
    for timestamp in sorted(set(ev)|set(req)):
        rows=req[timestamp]
        if rows:
            flush(); assert last<timestamp
            device=chain['reuse'].k.device
            panel=(tensor([[int(r['item_idx']) for r in rows]],device),tensor([timestamp-last],device,floating=True))
            values={}; revision={name:chain[name].revision for name in ('design1','design2')}
            for name in METHODS:
                state=chain[name]
                if name in ('design1','design2') and target in TARGETS and not (name=='design2' and state.anchored_native):
                    z=timed(lambda:corrected_score(model,state,adapter,panel,ledger,charges[name]),ledger,name+'_read')
                else:
                    cache=state.cache if isinstance(state,LifecycleState) else state
                    z=timed(lambda:score(model,cache,*panel)[0],ledger,name+'_read')
                values[name]=z.cpu().numpy()[0]
            assert all(chain[name].revision==revision[name] for name in revision)
            shared_flops['native_read']+=value(query_ops(chain['reuse'].seq_len,len(rows)))
            for j,r in enumerate(rows):
                raw.append(dict(uid=uid,target=target,request_id=r['request_id'],timestamp=timestamp,item_idx=r['item_idx'],label=int(r['label']),
                    count=chain['reuse'].seq_len,writes_since_release=chain['design2'].writes_since_release,
                    anchored_native=chain['design2'].anchored_native,rebuilds=chain['design2'].rebuilds,
                    last_rebuild_target=None if chain['design2'].last_rebuild is None else chain['design2'].last_rebuild['target'],
                    **{name:float(v[j]) for name,v in values.items()}))
            reads+=1
        pending.extend(ev[timestamp])
    flush()
    assert appends==len(events) and len(raw)==len(requests)
    for name in ('design1','design2'):state_check(chain[name])
    return raw,dict(uid=uid,target=target,events=appends,append_calls=append_calls,requests=len(raw),request_groups=reads,
        common_flops=dict(shared_flops),counts={name:dict(chain[name].counts) for name in ('design1','design2')},
        design2_rebuilds=chain['design2'].rebuilds,design2_anchor=chain['design2'].native_anchor,
        end_kv_max_delta_d2_reuse=float((chain['design2'].cache.k-chain['reuse'].k).abs().max()))


@torch.no_grad()
def main(cli):
    out=ROOT/'results/design2'/cli.run_id
    out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(CONFIG.read_text()); protocol=json.loads((ROOT/'configs/design2/detection_01.json').read_text())
    uids=protocol['groups']['development'][cli.offset:cli.offset+cli.users]; assert len(uids)==cli.users
    device=torch.device(cli.device); torch.cuda.set_device(device); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.manual_seed(17)
    frozen=ROOT/'configs/design2/scale_calibration_01_fitted.json'; fitted=json.loads(frozen.read_text())
    assert cfg['threshold']==fitted['thresholds']['calibrated']['0.8']
    paths=[CONFIG,frozen,Path(__file__),ROOT/'src/hstu_kvcache/design2/lifecycle.py',ROOT/'src/hstu_kvcache/design2/conditional.py']
    write_json(out/'configuration.json',dict(vars(cli),uids=uids,protocol=cfg,hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},admissions=diagnostic_admissions(5)))
    started=time.perf_counter(); ledger=defaultdict(float); raw=[]; records=[]; decisions=[]; allcharges=[]; canary_checks=[]
    history=timed(lambda:histories(uids,301),ledger,'history_load')
    initial_model=frozen_model(0,device)
    initial=timed(lambda:cache_at_many(initial_model,history,[(u,CUTOVER_DAYS[0]*DAY) for u in uids],8),ledger,'common_M0')
    chains={}; initial_charge={}
    for uid,(cache,times) in zip(uids,initial):
        chains[uid]=dict(reuse=cache,exact=cache,design1=LifecycleState(cache,times,0),design2=LifecycleState(cache,times,0))
        initial_charge[uid]=summary_build_cost(cache.seq_len)
    del initial,initial_model
    for target in range(1,6):
        start=CUTOVER_DAYS[target-1]*DAY; stop=(CUTOVER_DAYS[target] if target<5 else 301)*DAY
        model=frozen_model(target,device)
        adapter=FrozenC(target,device) if target in TARGETS else None
        factors=packs=None
        if adapter:
            factors=torch.load(ROOT/f'results/design2/geometry_01/cholesky_m{target}.pt',map_location=device,weights_only=True)
            packs=torch.load(ROOT/f'results/design2/conditional225_01/inverse/blocks_m{target}.pt',map_location=device,weights_only=True)
        requests=quality_requests(uids,start,stop)
        for index,uid in enumerate(uids):
            chain=chains[uid]; charges={name:Counter() for name in METHODS}
            for name in ('design1','design2'):
                previous=chain[name].native_anchor
                chain[name].release(target)
                assert not chain[name].anchored_native and chain[name].prepared is None and chain[name].writes_since_release==0
                if target==1:charges[name]['initial_summary']+=initial_charge[uid]
                if cli.canary:canary_checks.append(dict(uid=uid,target=target,check='release_invalidation',previous_anchor=previous))
            if adapter:
                n=chain['exact'].seq_len
                exact,times=timed(lambda:cache_at(model,history,uid,start),ledger,'exact_release_rebuild')
                chain['exact']=exact
                charges['exact']['rebuild_literal']+=value(prefix_literal(n))
                charges['exact']['rebuild_dependency_closed']+=value(rebuild(n))
                decision=decide(model,chain['design2'],adapter,packs,factors,fitted,history,uid,target,start,ledger,charges['design2'],cli.canary)
                decisions.append(decision)
                if cli.canary and index==0:
                    # Independent intervention only for action semantics; never affects the policy branch.
                    forced=LifecycleState(chain['design1'].cache,[e[3] for e in chain['design1'].writer.events],target-1)
                    forced.release(target); forced.prepared={'stale':True}
                    forced.rebuild(exact,times,start)
                    assert forced.prepared is None and forced.anchored_native
                    panel=make_panel(history,[SimpleNamespace(uid=uid,state=forced)],[0],start,device)
                    torch.testing.assert_close(score(model,forced.cache,*panel)[0],score(model,exact,*panel)[0],rtol=0,atol=0)
                    sample=event_range(history,uid,start,stop)[:130]
                    if sample:
                        reference=append_events(model,exact,sample,int(times[-1]),128)
                        append_events(model,forced,sample,int(times[-1]),128)
                        torch.testing.assert_close(forced.cache.k,reference.k,atol=1e-6,rtol=1e-6)
                        torch.testing.assert_close(forced.cache.v,reference.v,atol=1e-6,rtol=1e-6)
                        assert forced.anchored_native
                        expected=list(times)+[e[0] for e in sample]
                        assert [e[3] for e in forced.writer.events]==expected[-1024:]
                        state_check(forced)
                    forced.release(target+1)
                    assert not forced.anchored_native and forced.prepared is None
                    canary_checks.append(dict(uid=uid,target=target,check='forced_rebuild_immediate_append_evict_next_release',events=len(sample)))
                if chain['design2'].anchored_native:
                    torch.testing.assert_close(chain['design2'].cache.k,exact.k,rtol=0,atol=0)
                    torch.testing.assert_close(chain['design2'].cache.v,exact.v,rtol=0,atol=0)
            rows,record=replay_user(model,chain,history,uid,requests.get(uid,[]),start,stop,target,adapter,ledger,charges,cli.canary)
            raw.extend(rows); records.append(record)
            allcharges.extend(dict(uid=uid,target=target,method=name,**dict(c)) for name,c in charges.items())
        edge=pd.DataFrame([r for r in raw if r['target']==target])
        edge.to_parquet(out/f'quality_m{target}.parquet',index=False)
        print(json.dumps(dict(target=target,users=len(uids),requests=len(edge),elapsed_seconds=time.perf_counter()-started,
            decisions=dict(Counter(d['reason'] for d in decisions if d['target']==target)))),flush=True)
        del adapter,packs,factors,model
    frame=pd.DataFrame(raw)
    assert not frame.duplicated(['target','request_id']).any()
    assert np.isfinite(frame[list(METHODS)]).all().all()
    frame.to_parquet(out/'quality_raw.parquet',index=False)
    pd.DataFrame(decisions).to_parquet(out/'decisions.parquet',index=False)
    pd.DataFrame(allcharges).fillna(0).to_parquet(out/'costs.parquet',index=False)
    write_json(out/'state_counts.json',records)
    write_json(out/'summary.json',dict(status='complete',users=len(uids),requests=len(raw),elapsed_seconds=time.perf_counter()-started,
        ledger_seconds=dict(ledger),canary_checks=canary_checks,peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20,
        no_new_fit=True,confirmation_read=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True); parser.add_argument('--users',type=int,required=True)
    parser.add_argument('--offset',type=int,default=0); parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--canary',action='store_true'); parser.add_argument('--estimate-seconds',type=int,required=True)
    main(parser.parse_args())
