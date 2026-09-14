"""Cost-dominant execution paths within the existing prepaid wallet."""
import argparse,json
from design2 import run_compute as c
from design2.audit_benchmark import save
from design.data import ROOT
old=c.decide
CONFIG=ROOT/'configs/design2/compute_min_01.json'

def decide(model,state,adapter,packs,factors,fitted,h,uid,t,stamp,ledger,charges,policy):
    n=state.cache.seq_len;tau=fitted['thresholds']['calibrated']['0.8']
    group=next(g['group'] for g in fitted['groups'] if g['target']==t and g['lower']<=n<=g['upper'])
    a,b=[fitted['final_fit'][group][k] for k in ('a','b')]
    q=sum(int(r['query_timestamp'])==stamp for r in c.REQUESTS[uid])
    cheap=c.arithmetic(q)['exact_panel_flops']>=c.r.tiled_flops(n)+c.r.base.summary_build_cost(n)
    direct=cheap and not(a==0 and b<=tau)
    c.MODE='fifo' if direct else 'reserve'
    try:row=old(model,state,adapter,packs,factors,fitted,h,uid,t,stamp,ledger,charges,policy)
    finally:c.MODE='reserve'
    row['cost_dominant']=direct
    if row['reason']=='fifo_rebuild':row['reason']='cost_dominant_rebuild'
    return row

def main(cli):
    c.decide=decide;cli.mode='reserve';c.main(cli)
    out=ROOT/'results/design2'/cli.run_id
    cfg=json.loads((out/'configuration.json').read_text());cfg.update(compute_mode='mincost',min_config=json.loads(CONFIG.read_text()),min_sha=c.r.base.sha(__file__))
    save(out/'configuration.json',cfg)
    w=json.loads((out/'wallet.json').read_text());w['mode']='mincost';save(out/'wallet.json',w)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',default='mincost');p.add_argument('--run-id',required=True);p.add_argument('--users',type=int,default=8)
    p.add_argument('--offset',type=int,default=0);p.add_argument('--cap',type=int,default=1024);p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');main(p.parse_args())
