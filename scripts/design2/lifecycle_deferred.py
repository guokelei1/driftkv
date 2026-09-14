"""Defer a frozen release rebuild intent until the first causal consumer."""

import argparse
import json
from pathlib import Path

from design.run import write_json
from design2 import run_lifecycle as base
from design2.lifecycle_cost_only import queue

CONFIG=base.ROOT/'configs/design2/lifecycle_deferred_01.json'
OUT=base.ROOT/'results/design2/lifecycle_deferred_01'
OriginalState=base.LifecycleState
immediate_rebuild=base.current_rebuild
original_decide=base.decide
original_replay=base.replay_user


class DeferredState(OriginalState):
    def __init__(self,*args):
        super().__init__(*args)
        self.pending_rebuild=None

    def release(self,target,translator=None):
        self.pending_rebuild=None
        super().release(target,translator)


def defer_rebuild(model,state,history,uid,cutover,ledger,charges,label):
    state.pending_rebuild=dict(target=state.target,uid=uid,decided_at=cutover)


def decide(*args,**kwargs):
    row=original_decide(*args,**kwargs)
    if row['action']=='rebuild':row['action']='queue_rebuild'
    return row


def replay_user(model,chain,history,uid,requests,start,stop,target,adapter,ledger,charges,canary):
    pending=chain['design2'].pending_rebuild
    if pending is None or not requests:
        return original_replay(model,chain,history,uid,requests,start,stop,target,adapter,ledger,charges,canary)
    assert pending['target']==target and pending['uid']==uid
    first=min(int(r['query_timestamp']) for r in requests)
    # Split exactly at the already-existing first-request flush boundary. No
    # future label is inspected; all ordinary branches retain their same writes.
    empty,before=original_replay(model,chain,history,uid,[],start,first,target,adapter,ledger,charges,canary)
    assert not empty
    immediate_rebuild(model,chain['design2'],history,uid,first,ledger,charges['design2'],'deferred_rebuild')
    chain['design2'].pending_rebuild=None
    raw,after=original_replay(model,chain,history,uid,requests,first,stop,target,adapter,ledger,charges,canary)
    for field in ('events','append_calls'):after[field]+=before[field]
    for k,v in before['common_flops'].items():after['common_flops'][k]=after['common_flops'].get(k,0)+v
    after.update(deferred_rebuild_executed=True,decision_timestamp=pending['decided_at'],rebuild_timestamp=first)
    return raw,after


def replay(cli):
    base.CONFIG=CONFIG; base.LifecycleState=DeferredState
    base.current_rebuild=defer_rebuild; base.decide=decide; base.replay_user=replay_user
    base.main(cli)
    p=base.ROOT/'results/design2'/cli.run_id/'configuration.json'; cfg=json.loads(p.read_text())
    cfg.update(variant='deferred',variant_source_sha256=base.sha(Path(__file__)),
        artifact_column_note='design2 column is deferred policy here; separate from primary immediateD2')
    write_json(p,cfg)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['replay','queue'],required=True)
    p.add_argument('--run-id');p.add_argument('--users',type=int);p.add_argument('--offset',type=int,default=0)
    p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');p.add_argument('--estimate-seconds',type=int,default=30)
    cli=p.parse_args()
    queue(OUT,Path(__file__),CONFIG) if cli.mode=='queue' else replay(cli)
