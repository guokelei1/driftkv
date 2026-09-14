"""One target reconstruction provides both paid evidence and the new state."""
from collections import defaultdict
import argparse,json
import numpy as np
import torch
from design2 import run_sparse as s
from design2.audit_benchmark import save
from design.data import prefix
from design.report_native_flops import value,query_ops
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.design2.rebuild import current_kv,tiled_flops

r=s.r
REQUESTS={};ADAPTERS={};MEMO={};NATIVE={};ATTEMPTS={};HITS=[]
OLD_FACTORY=r.FrozenC;OLD_REQUESTS=r.quality_requests;OLD_DECIDE=s.decide
OLD_CORRECTED=r.base.corrected_score;OLD_SCORE=r.base.score;OLD_LOAD=torch.load
OLD_MAIN=r.main

def factory(t,d):
    a=OLD_FACTORY(t,d);ADAPTERS[t]=a;return a

def requests(*args):
    rows=OLD_REQUESTS(*args);REQUESTS.clear();REQUESTS.update(rows);return rows

def corrected(model,state,adapter,panel,ledger,charges):
    record=MEMO.pop(id(state),None)
    if record is None:return OLD_CORRECTED(model,state,adapter,panel,ledger,charges)
    z,item,dt,rev=record
    assert state.revision==rev and torch.equal(item,panel[0]) and torch.equal(dt,panel[1])
    HITS.append('adapted');return z

def native(model,cache,items,dt,*args,**kwargs):
    record=NATIVE.pop(id(cache),None)
    if record is None:return OLD_SCORE(model,cache,items,dt,*args,**kwargs)
    z,item,delta=record
    assert torch.equal(item,items) and torch.equal(delta,dt)
    HITS.append('renewed');return z,None

def rebuild(model,state,history,uid,stamp,charges):
    assert not MEMO and not NATIVE
    rows=[q for q in REQUESTS[uid] if int(q['query_timestamp'])==stamp];assert rows
    device=state.cache.k.device;i,a,d,ts=prefix(history,uid,stamp,device)
    fresh=current_kv(model,i,a,d)
    assert np.array_equal(ts,[e[3] for e in state.writer.events])
    panel=(r.original.tensor([[int(q['item_idx']) for q in rows]],device),r.original.tensor([stamp-int(ts[-1])],device,floating=True))
    old=OLD_CORRECTED(model,state,ADAPTERS[state.target],panel,defaultdict(float),charges)
    new=score(model,fresh,*panel)[0]
    charges['rebuild_tiled']+=tiled_flops(fresh.seq_len)
    # Two native reads here, one replaces the ordinary request read below.
    charges['observation_extra_read']+=value(query_ops(fresh.seq_len,len(rows)))
    tau=json.loads((r.ROOT/'configs/design2/scale_calibration_01_fitted.json').read_text())['thresholds']['calibrated']['0.8']
    error=float((old-new).abs().max());commit=error>tau
    if commit:
        state.rebuild(fresh,ts,stamp);charges['rebuild_summary']+=r.base.summary_build_cost(fresh.seq_len)
        NATIVE[id(state.cache)]=(new,panel[0],panel[1])
    else:MEMO[id(state)]=(old,panel[0],panel[1],state.revision)
    charges['observation_control']+=3*len(rows)+128
    ATTEMPTS[uid,state.target]=dict(committed=commit,commit_error=error,actual_queries=len(rows),renewal_writes=state.writes_since_release)

def decide(*args,**kwargs):
    row=OLD_DECIDE(*args,**kwargs);key=row['uid'],row['target']
    if key in ATTEMPTS:
        row.update(ATTEMPTS.pop(key));row['reason']='observed_commit' if row['committed'] else 'observed_retain'
        if not row['committed']:row['action']='retain_after_paid_probe'
    return row

def load(path,*args,**kwargs):
    if 'conditional225_01/inverse/blocks_' in str(path):return None
    return OLD_LOAD(path,*args,**kwargs)

def background_main(cli):
    p=s.PARAMS['background']
    s.PARAMS['source_bound']=dict(threshold=p['threshold'],params={g:dict(a=0.,b=v['background']) for g,v in p['params'].items()})
    return OLD_MAIN(cli)

def main(cli):
    variant=cli.mode
    s.decide=decide;r.rebuild_state=rebuild;r.FrozenC=factory;r.quality_requests=requests
    r.base.corrected_score=corrected;r.base.score=native;torch.load=load
    if variant=='bgobserve':r.main=background_main
    cli.mode='direct';s.main(cli)
    assert not MEMO and not NATIVE and not ATTEMPTS
    out=r.ROOT/'results/design2'/cli.run_id
    cfg=json.loads((out/'configuration.json').read_text());cfg.update(sparse_mode=variant,observe_protocol=json.loads((r.ROOT/'configs/design2/sparse_observe_01.json').read_text()),observe_source=r.base.sha(__file__))
    save(out/'configuration.json',cfg);save(out/'observation.json',dict(shared_reads=len(HITS),committed_reads=HITS.count('renewed'),retained_reads=HITS.count('adapted')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',default='observe');p.add_argument('--run-id',required=True);p.add_argument('--users',type=int,default=8)
    p.add_argument('--offset',type=int,default=0);p.add_argument('--cap',type=int,default=1024);p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');main(p.parse_args())
