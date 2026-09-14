"""Expanded fixed-cohort replay and one reusable target-observation candidate."""
import argparse,json
import numpy as np
import torch
from design2 import run_excursion as e
from design2 import run_scale_followup as r
from design2.audit_benchmark import save
from design.report_native_flops import value,query_ops
from hstu_kvcache.adaptation.reader import score

ROOT=r.ROOT;ORIGINAL_HOOK=e.hook

class WitnessState(r.base.LifecycleState):
    def release(self,target,translator=None):
        if hasattr(self,'witness') and self.witness['target']!=target:del self.witness
        super().release(target,translator)

def retain(model,state,cache,ts,stamp,key,charges):
    state.witness=dict(cache=cache,target=key[1],since=stamp,last=int(ts[-1]))
    rec=e.STATES[key];rec.update(witness_created=stamp,witness_reads=0,witness_events=0,witness_bytes=cache.k.numel()*cache.k.element_size()+cache.v.numel()*cache.v.element_size())

def witness_hook(model,state,adapter,panel,history,uid,target,stamp,ledger,charges):
    if not hasattr(state,'witness'):return ORIGINAL_HOOK(model,state,adapter,panel,history,uid,target,stamp,ledger,charges)
    w=state.witness;assert w['target']==target and not state.anchored_native
    events=r.base.event_range(history,uid,w['since'],stamp)
    for start in range(0,len(events),128):
        chunk=events[start:start+128];n=w['cache'].seq_len
        before=w['cache'];last=w['last']
        w['cache']=r.base.append_events_many(model,[w['cache']],[chunk],[w['last']],128,1)[0]
        if e.CANARY and len(chunk)<=32:
            other=r.base.append_events_many(model,[before],[chunk],[last],1,1)[0]
            torch.testing.assert_close(w['cache'].k,other.k,rtol=2e-5,atol=2e-5)
            torch.testing.assert_close(w['cache'].v,other.v,rtol=2e-5,atol=2e-5)
        charges['witness_native_append']+=value(r.base.append_ops(n,len(chunk)));w['last']=chunk[-1][0]
    w['since']=stamp;assert w['last']<stamp and w['cache'].seq_len==state.cache.seq_len
    z=r.base.corrected_score(model,state,adapter,panel,ledger,charges)
    target_z=score(model,w['cache'],*panel)[0];error=float((z-target_z).abs().max())
    charges['witness_extra_read']+=value(query_ops(w['cache'].seq_len,panel[0].shape[1]));charges['witness_control']+=128+3*panel[0].shape[1]
    rec=e.STATES[uid,target];rec['witness_reads']+=1;rec['witness_events']+=len(events)
    if error>.5:
        ts=history.prefix(uid,stamp,1024)[2];assert np.array_equal(ts,[x[3] for x in state.writer.events])
        cache=w['cache'];del state.witness;state.rebuild(cache,ts,stamp);charges['rebuild_summary']+=r.base.summary_build_cost(cache.seq_len)
        rec.update(action='rebuild',reason='witness_later_commit',timestamp=stamp,commit_error=error,committed=True,renewal_writes=state.writes_since_release)
        if e.CANARY:torch.testing.assert_close(score(model,state.cache,*panel)[0],target_z,rtol=0,atol=0)
        return target_z
    return z

def main(c):
    r.UIDS=ROOT/(c.uids_file or 'results/design2/scan_30k_01/uids.json');r.CONFIG=ROOT/'configs/design2/scan_30k_01.json'
    policy=c.policy
    if policy=='witness':r.base.LifecycleState=WitnessState;e.PROBE_HOOK=retain;e.hook=witness_hook
    c.mode='excursion';e.main(c)
    out=ROOT/'results/design2'/c.run_id;cfg=json.loads((out/'configuration.json').read_text())
    cfg.update(scan_policy=policy,scan_runner_sha256=r.base.sha(__file__),reference_semantics='Witness is Current-exact at construction, then a native target trajectory; subsequent discrepancies are not FreshCurrent errors')
    save(out/'configuration.json',cfg)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',choices=['frozen','witness'],required=True);p.add_argument('--run-id',required=True);p.add_argument('--users',type=int,default=8);p.add_argument('--offset',type=int,default=0);p.add_argument('--cap',type=int,default=1024);p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');p.add_argument('--uids-file');main(p.parse_args())
