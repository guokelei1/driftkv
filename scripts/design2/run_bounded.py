"""Post-response installation and one fixed FLOP allowance; original replay."""
import argparse,json
import numpy as np,pandas as pd,torch
from design2 import run_excursion as e,run_scale_followup as r,replay_requests as replay
from design2.audit_benchmark import save
from design.report_native_flops import value,query_ops
from hstu_kvcache.design2.rebuild import tiled_flops
from hstu_kvcache.adaptation.reader import score

MODE=None;OBS=[];MEM=[];PENDING={};LIVE={};CLOSED=[]
BASE_STATE=r.base.LifecycleState

def nbytes(cache):return (cache.k.numel()*cache.k.element_size()+cache.v.numel()*cache.v.element_size())
def mem(w,stamp,size,reason):
    MEM.append(dict(uid=w['uid'],target=w['target'],timestamp=int(stamp),bytes=int(size),reason=reason))
def close(state,stamp,reason):
    w=state.witness;mem(w,stamp,0,reason);LIVE.pop((w['uid'],w['target']),None)
    CLOSED.append(dict(uid=w['uid'],target=w['target'],created=w['created'],ended=int(stamp),reason=reason,allowance=w['allowance'],spent=w['spent'],reads=w['reads']))
    del state.witness

class State(BASE_STATE):
    def release(self,target,translator=None):
        if hasattr(self,'witness') and self.witness['target']!=target:
            close(self,r.base.CUTOVER_DAYS[target-1]*r.base.DAY,'model_change')
        super().release(target,translator)

def observation(uid,target,stamp,panel,z,w,kind,state):
    for j,(a,b,item) in enumerate(zip(z.reshape(-1).tolist(),w.reshape(-1).tolist(),panel[0].reshape(-1).tolist())):
        OBS.append(dict(uid=uid,target=target,timestamp=stamp,candidate=j,item_idx=int(item),service=a,witness=b,kind=kind,
            witness_created=stamp if kind=='construction' else state.witness['created'],writes=state.writes_since_release))

def retain(model,state,cache,ts,stamp,key,charges):
    allowance=.25*tiled_flops(cache.seq_len) if MODE=='bounded' else float('inf')
    state.witness=dict(uid=key[0],target=key[1],cache=cache,since=stamp,last=int(ts[-1]),created=stamp,allowance=allowance,spent=0.,reads=0)
    LIVE[key]=state;mem(state.witness,stamp,nbytes(cache),'create')
    charges['bounded_control']+=32

def schedule(model,state,cache,ts,stamp,key,charges):
    assert key not in PENDING
    PENDING[key]=dict(cache=cache,ts=ts,stamp=stamp,revision=state.revision)
    charges['boundary_control']+=32

def post_score(model,state,uid,target,stamp,charges):
    p=PENDING.pop((uid,target),None)
    if p is None:return
    assert p['stamp']==stamp and p['revision']==state.revision
    if hasattr(state,'witness'):close(state,stamp,'install')
    state.rebuild(p['cache'],p['ts'],stamp);charges['rebuild_summary']+=r.base.summary_build_cost(state.cache.seq_len)
    assert state.anchored_native

def hook(model,state,adapter,panel,history,uid,target,stamp,ledger,charges):
    if not hasattr(state,'witness'):return e.hook(model,state,adapter,panel,history,uid,target,stamp,ledger,charges)
    w=state.witness;assert target==w['target'] and not state.anchored_native
    events=r.base.event_range(history,uid,w['since'],stamp);n=w['cache'].seq_len
    append_cost=0
    for i in range(0,len(events),128):
        m=len(events[i:i+128]);append_cost+=value(r.base.append_ops(n,m));n=min(1024,n+m)
    read_cost=value(query_ops(n,panel[0].shape[1]));control=192+3*panel[0].shape[1]+16*((len(events)+127)//128)
    need=append_cost+read_cost+control
    charges['bounded_admission']+=32+16*((len(events)+127)//128)
    if need>w['allowance']-w['spent']:
        rec=e.STATES[uid,target];rec.update(expired=True,observation_spent=w['spent'],observation_allowance=w['allowance'],expiry_timestamp=stamp)
        close(state,stamp,'budget');return None
    w['spent']+=need;assert w['spent']<=w['allowance'];w['reads']+=1
    for i in range(0,len(events),128):
        chunk=events[i:i+128];before=w['cache'];last=w['last']
        w['cache']=r.base.append_events_many(model,[before],[chunk],[last],128,1)[0]
        if e.CANARY and len(chunk)<=32:
            other=r.base.append_events_many(model,[before],[chunk],[last],1,1)[0]
            torch.testing.assert_close(w['cache'].k,other.k,rtol=2e-5,atol=2e-5)
        w['last']=chunk[-1][0]
    if events:mem(w,stamp,nbytes(w['cache']),'advance')
    w['since']=stamp;assert w['last']<stamp and w['cache'].seq_len==state.cache.seq_len
    z=r.base.corrected_score(model,state,adapter,panel,ledger,charges)
    target_z=score(model,w['cache'],*panel)[0];error=float((z-target_z).abs().max())
    observation(uid,target,stamp,panel,z,target_z,'continued',state)
    charges['witness_native_append']+=append_cost;charges['witness_extra_read']+=read_cost;charges['witness_control']+=control
    if error>.5:
        ts=history.prefix(uid,stamp,1024)[2];assert np.array_equal(ts,[x[3] for x in state.writer.events])
        e.STATES[uid,target].update(action='rebuild',reason='witness_later_commit',timestamp=stamp,commit_error=error,committed=True,renewal_writes=state.writes_since_release)
        schedule(model,state,w['cache'],ts,stamp,(uid,target),charges)
    return z

def main(c):
    global MODE
    MODE=c.variant;r.UIDS=r.ROOT/(c.uids_file or 'results/design2/scan_30k_01/uids.json');r.CONFIG=r.ROOT/'configs/design2/bounded_01.json'
    r.base.LifecycleState=State;e.PROBE_HOOK=retain;e.OBSERVATION_HOOK=observation;e.INSTALL_HOOK=schedule
    replay.POST_SCORE_HOOK=post_score
    # e.main binds replay.HOOK to e.hook; retain original before replacing.
    original=e.hook
    def dispatch(*args):
        if hasattr(args[1],'witness'):return hook(*args)
        return original(*args)
    e.hook=dispatch;c.mode='excursion';e.main(c);assert not PENDING
    for state in list(LIVE.values()):close(state,301*r.base.DAY,'end_of_observation')
    out=r.ROOT/'results/design2'/c.run_id
    for name,rows in [('witness_reads',OBS),('candidate_memory',MEM),('candidate_lifetimes',CLOSED)]:pd.DataFrame(rows).to_parquet(out/f'{name}.parquet',index=False)
    cfg=json.loads((out/'configuration.json').read_text());cfg.update(variant=MODE,source=r.base.sha(__file__),post_response=True);save(out/'configuration.json',cfg)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['post','bounded'],required=True);p.add_argument('--run-id',required=True);p.add_argument('--users',type=int,default=8);p.add_argument('--offset',type=int,default=0);p.add_argument('--cap',type=int,default=1024);p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');p.add_argument('--uids-file');main(p.parse_args())
