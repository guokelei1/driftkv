"""Recover frozen calibration scenes; assess cheap nested observation bounds."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from design2 import conditional_check as old
from design2.tiered_check import old_run
from design2.audit_benchmark import save
from hstu_kvcache.design2.evidence import prepare,source_score,observed_score

def evaluate(model,history,scenes,adapter,stamp,out,target,cli,ledger,fitted):
    p=adapter.prepare(scenes)
    factors=torch.load(old.ROOT/f'results/design2/geometry_01/cholesky_m{target}.pt',map_location=cli.device,weights_only=True)
    packs=prepare(factors)
    meta=pd.DataFrame(old.metadata(scenes,p,target,cli.role,stamp))
    ref=pd.read_parquet(old.ROOT/'results/design2'/old_run(cli.role,cli.start)/f'states_m{target}.parquet')
    ref=ref.set_index(['uid','state_ordinal']).loc[list(zip(meta.uid,meta.state_ordinal))]
    meta['source_bound']=source_score(p['latent'],p['counts'].double(),p['active'],packs).cpu().numpy()
    vals=np.zeros(len(meta));queries=[]
    for indices in old.batches(scenes):
        cache=old.batch_cache([scenes[i].state.cache for i in indices]);panel=old.make_panel(history,scenes,indices,stamp,cli.device)
        z,trace,obs,_=adapter.read(model,cache,panel,p,indices)
        small=observed_score(p['latent'][indices],obs,adapter.parameters,p['counts'][indices].double(),p['active'][indices],packs)
        vals[indices]=small.amax(-1).cpu().numpy()
        for j,i in enumerate(indices):
            queries.extend(dict(uid=scenes[i].uid,state_ordinal=scenes[i].ordinal,target=target,query_index=k,observed_bound=float(v)) for k,v in enumerate(small[j]))
    meta['observed_bound']=vals;meta['full_score']=ref.detection.to_numpy();meta['error']=ref.max_abs_error.to_numpy()
    assert np.all(meta.source_bound<=meta.observed_bound+1e-5)
    assert np.all(meta.observed_bound<=meta.full_score*(1+1e-7)+1e-5)
    meta.to_parquet(out/f'states_m{target}.parquet',index=False);pd.DataFrame(queries).to_parquet(out/f'queries_m{target}.parquet',index=False)
    return dict(target=target,states=len(meta),nested_bounds=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',default='residual_calibration');p.add_argument('--start',type=int,default=0);p.add_argument('--users',type=int,default=16)
    p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');cli=p.parse_args()
    old.OUT=old.ROOT/'results/design2/evidence_01';old.evaluate=evaluate;old.replay(cli)
    out=old.OUT/'chunks'/f'{cli.role}_{cli.start:04d}'
    (out/'evidence_provenance.json').write_text(json.dumps({str(p.relative_to(old.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),old.ROOT/'src/hstu_kvcache/design2/evidence.py']},indent=2)+'\n')
