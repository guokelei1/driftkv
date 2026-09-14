"""Lazy Exact control: rebuild on first actual use of each target release."""
import argparse,json
from design2 import run_scale_followup as r
from design2.audit_benchmark import save

def decide(model,state,adapter,packs,factors,fitted,history,uid,target,stamp,ledger,charges,policy):
    n=state.cache.seq_len;r.rebuild_state(model,state,history,uid,stamp,charges)
    return dict(uid=uid,target=target,timestamp=stamp,count=n,action='rebuild',reason='exact_on_use',screened=True,u=None)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',default='exactdemand');p.add_argument('--run-id',required=True);p.add_argument('--users',type=int,default=8)
    p.add_argument('--offset',type=int,default=0);p.add_argument('--cap',type=int,default=1024);p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');c=p.parse_args()
    r.decide=decide;c.policy='hash30';r.main(c)
    out=r.ROOT/'results/design2'/c.run_id;cfg=json.loads((out/'configuration.json').read_text())
    cfg.update(baseline='Exact-on-first-use each admitted release',baseline_source=r.base.sha(__file__),
        cost_scope='cache-only baseline needs no summaries, translator or detector; wrapper summary tracking is evaluator instrumentation, charge rebuild_tiled only as incremental algorithm work')
    save(out/'configuration.json',cfg)
