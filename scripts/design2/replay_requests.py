"""Existing independent-KV event loop with an explicit pre-score action hook.

Derived from run_lifecycle.replay_user. The hook runs after causal writes are
flushed and before the read-only revision assertion; no model code is cloned.
"""
from collections import defaultdict,Counter
from design2 import run_lifecycle as b

HOOK=None
POST_SCORE_HOOK=None

def replay_user(model,chain,history,uid,requests,start,stop,target,adapter,ledger,charges,canary):
    events=b.event_range(history,uid,start,stop);ev,req=defaultdict(list),defaultdict(list)
    for e in events:ev[e[0]].append(e)
    for r in requests:req[int(r['query_timestamp'])].append(r)
    last=int(chain['design1'].writer.events[-1][3]);pending=[];raw=[]
    reads=appends=append_calls=0;shared_flops=Counter()
    def flush():
        nonlocal last,appends,append_calls
        for offset in range(0,len(pending),128):
            chunk=pending[offset:offset+128];m=len(chunk);n=chain['reuse'].seq_len
            before={name:chain[name].counts['evictions'] for name in ('design1','design2')}
            inputs=[chain[name] for name in b.METHODS]
            output=b.timed(lambda:b.append_events_many(model,inputs,[chunk]*4,[last]*4,128,4),ledger,'four_independent_appends')
            chain.update(zip(b.METHODS,output))
            for name in ('design1','design2'):
                state=chain[name];removed=state.counts['evictions']-before[name]
                charges[name]['summary_maintenance']+=b.V*(m+removed)
                if canary:b.state_check(state)
            assert len({(c.cache if isinstance(c,b.LifecycleState) else c).seq_len for c in chain.values()})==1
            shared_flops['native_append']+=b.value(b.append_ops(n,m));last=chunk[-1][0];appends+=m;append_calls+=1
        pending.clear()
    for timestamp in sorted(set(ev)|set(req)):
        rows=req[timestamp]
        if rows:
            flush();assert last<timestamp
            device=chain['reuse'].k.device
            panel=(b.tensor([[int(r['item_idx']) for r in rows]],device),b.tensor([timestamp-last],device,floating=True))
            cached=HOOK(model,chain['design2'],adapter,panel,history,uid,target,timestamp,ledger,charges['design2']) if adapter is not None else None
            values={};revision={name:chain[name].revision for name in ('design1','design2')}
            for name in b.METHODS:
                state=chain[name]
                if name=='design2' and cached is not None:z=cached
                elif name in ('design1','design2') and target in b.TARGETS and not(name=='design2' and state.anchored_native):
                    z=b.timed(lambda:b.corrected_score(model,state,adapter,panel,ledger,charges[name]),ledger,name+'_read')
                else:
                    cache=state.cache if isinstance(state,b.LifecycleState) else state
                    z=b.timed(lambda:b.score(model,cache,*panel)[0],ledger,name+'_read')
                values[name]=z.cpu().numpy()[0]
            assert all(chain[name].revision==revision[name] for name in revision)
            shared_flops['native_read']+=b.value(b.query_ops(chain['reuse'].seq_len,len(rows)))
            for j,r in enumerate(rows):
                raw.append(dict(uid=uid,target=target,request_id=r['request_id'],timestamp=timestamp,item_idx=r['item_idx'],label=int(r['label']),
                    count=chain['reuse'].seq_len,writes_since_release=chain['design2'].writes_since_release,
                    anchored_native=chain['design2'].anchored_native,rebuilds=chain['design2'].rebuilds,
                    last_rebuild_target=None if chain['design2'].last_rebuild is None else chain['design2'].last_rebuild['target'],
                    **{name:float(v[j]) for name,v in values.items()}))
            reads+=1
            if POST_SCORE_HOOK is not None:POST_SCORE_HOOK(model,chain['design2'],uid,target,timestamp,charges['design2'])
        pending.extend(ev[timestamp])
    flush();assert appends==len(events) and len(raw)==len(requests)
    for name in ('design1','design2'):b.state_check(chain[name])
    return raw,dict(uid=uid,target=target,events=appends,append_calls=append_calls,requests=len(raw),request_groups=reads,
        common_flops=dict(shared_flops),counts={name:dict(chain[name].counts) for name in ('design1','design2')},
        design2_rebuilds=chain['design2'].rebuilds,design2_anchor=chain['design2'].native_anchor,
        end_kv_max_delta_d2_reuse=float((chain['design2'].cache.k-chain['reuse'].k).abs().max()))
