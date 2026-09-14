"""One fixed label-free pairwise-margin gate on already constructed target KV."""
import argparse,json
from types import SimpleNamespace
import torch
from design2 import run_excursion as e
from design2 import run_scan as scan
from design2.common import make_panel
from design.report_native_flops import read_cost,value,query_ops
from hstu_kvcache.adaptation.reader import score

def gate(model,state,adapter,p,history,uid,target,stamp,fresh,z,new,charges,rec):
    panel=make_panel(history,[SimpleNamespace(uid=uid,state=state)],[0],stamp,state.cache.k.device)
    old=adapter.read(model,state.cache,panel,p,[0])[0]
    current=score(model,fresh,*panel)[0]
    delta=torch.cat([(current-old).reshape(-1),(new-z).reshape(-1)])
    spread=float(delta.max()-delta.min())
    charges['margin_extra_reads']+=2*value(query_ops(state.cache.seq_len,16))+read_cost(16,1)
    charges['margin_control']+=3*delta.numel()
    rec.update(margin_spread=spread,margin_pass=spread>.5)
    if e.CANARY:
        pairwise=(delta[:,None]-delta[None,:]).abs().max()
        torch.testing.assert_close(pairwise,delta.max()-delta.min(),rtol=0,atol=0)
    return spread>.5

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--users',type=int,default=8);p.add_argument('--offset',type=int,default=0);p.add_argument('--cap',type=int,default=1024);p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');p.add_argument('--uids-file',required=True)
    c=p.parse_args();c.policy='frozen';e.COMMIT_FILTER=gate;scan.main(c)
    path=scan.ROOT/'results/design2'/c.run_id/'configuration.json'
    cfg=json.loads(path.read_text());cfg.update(scan_policy='margin_gate',margin_protocol=json.loads((scan.ROOT/'configs/design2/scan_margin_gate_01.json').read_text()),margin_runner_sha256=scan.r.base.sha(__file__))
    scan.save(path,cfg)
