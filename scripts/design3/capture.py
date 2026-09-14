"""Small unchanged-policy replay, retaining real screen inputs for execution probes."""
import argparse,json
from pathlib import Path
import torch
from design2 import run_bounded_candidate as c
from design2.common import FrozenC


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--users',type=int,default=8);p.add_argument('--offset',type=int,default=0);p.add_argument('--cap',type=int,default=1024);p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');p.add_argument('--uids-file',default='results/design2/scan_30k_01/canary_existing_uids.json');a=p.parse_args();a.variant='bounded'
    out=Path('results/design3/initial_01/inputs');out.mkdir(parents=True,exist_ok=True)
    original=FrozenC.read;counts={};last={}
    def capture(self,model,cache,panel,prepared,indices):
        target=self.target;n=counts.get(target,0)
        signature=(cache.k.data_ptr(),float(prepared['latent'].sum()))
        if n<8 and last.get(target)!=signature:
            torch.save(dict(target=target,k=cache.k.cpu(),v=cache.v.cpu(),panel=tuple(v.cpu() for v in panel),prepared={k:prepared[k].cpu() for k in ('latent','counts','active','b','a')}),out/f'm{target}_{n:02d}.pt')
            counts[target]=n+1;last[target]=signature
        return original(self,model,cache,panel,prepared,indices)
    FrozenC.read=capture
    c.b.observation=c.capture;c.b.replay.replay_user=c.replay_with_candidate;c.b.main(a)
    (out/'capture.json').write_text(json.dumps(dict(source_run=a.run_id,counts=counts,scope='actual armed-request inputs; first eight distinct cache snapshots per target; not a population sample'),indent=2))
