"""FreshCurrent at fixed real-request checkpoints; never feeds a serving policy."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from design.data import ROOT, histories, frozen_model
from design.run import cache_at_many, cache_at, tensor
from design.report_native_flops import prefix_literal, query_ops, value
from design2.audit_benchmark import save
from hstu_kvcache.adaptation.reader import score

OUT=ROOT/'results/design2/benchmark_reference_01'
RAW=ROOT/'results/design2/analysis/lifecycle_variants_01/quality_raw.parquet'
TARGETS=(1,3,4,5)


class PreparedHistory:
    def __init__(self,cases):
        self.cases={(r['uid'],r['timestamp']):r for r in cases}

    def prefix(self,uid,timestamp,cap):
        r=self.cases[uid,timestamp]
        assert cap==1024
        return r['items'],r['actions'],r['times']


def prepare():
    OUT.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    raw=pd.read_parquet(RAW).reset_index(names='row_index')
    users=json.loads((ROOT/'configs/design2/detection_01.json').read_text())['groups']['development']
    selected=[]
    for (uid,target),g in raw[raw.target.isin(TARGETS)].groupby(['uid','target']):
        g=g.sort_values(['timestamp','item_idx','request_id'])
        stamps={int(g.timestamp.iloc[0])}
        for k in (1,128,1024,6144):
            eligible=g[g.writes_since_release>=k]
            if len(eligible): stamps.add(int(eligible.timestamp.iloc[0]))
        for stamp in sorted(stamps):
            rows=g[g.timestamp==stamp]
            selected.append(dict(uid=int(uid),target=int(target),timestamp=stamp,
                row_indices=rows.row_index.to_numpy(),queries=rows.item_idx.to_numpy(),
                writes_since_release=int(rows.writes_since_release.iloc[0])))
    history=histories(users,301)
    for r in selected:
        items,actions,times=history.prefix(r['uid'],r['timestamp'],1024)
        assert len(items)>0 and times[-1]<r['timestamp']
        r.update(items=items,actions=actions,times=times,query_dt=int(r['timestamp']-times[-1]))
    torch.save(selected,OUT/'inputs.pt')
    pd.DataFrame([{k:r[k] for k in ('uid','target','timestamp','writes_since_release')} for r in selected]).to_parquet(OUT/'checkpoints.parquet',index=False)
    save(OUT/'preparation.json',dict(status='complete',checkpoints=len(selected),
        rows=sum(len(r['row_indices']) for r in selected),users=users,
        protocol='First actual request group and first request after writes1/128/1024/6144 per UID/target, deduplicated; no future label selection',
        estimated_population_seconds=600,estimate_basis='Prior1024 full four-branch lifecycle234s; only Current references here, allow600s including loading; refine with8UID canary',
        seconds=time.monotonic()-started,confirmation_read=False))
    print(json.dumps(dict(stage='prepared',checkpoints=len(selected),seconds=time.monotonic()-started)),flush=True)


@torch.no_grad()
def evaluate(cli):
    torch.set_num_threads(4);torch.cuda.set_device(cli.device)
    torch.backends.cuda.matmul.allow_tf32=False
    allcases=torch.load(OUT/'inputs.pt',weights_only=False)
    users=json.loads((OUT/'preparation.json').read_text())['users']
    cases=[r for r in allcases if (cli.target is None or r['target']==cli.target)
           and (not cli.canary or r['uid'] in users[:8])]
    history=PreparedHistory(cases); results=[]; checks=[]; charge=0
    started=time.monotonic()
    for t in TARGETS:
        part=[r for r in cases if r['target']==t]
        if not part: continue
        model=frozen_model(t,cli.device)
        for start in range(0,len(part),8):
            batch=part[start:start+8]
            caches=cache_at_many(model,history,[(r['uid'],r['timestamp']) for r in batch],8)
            for r,(cache,_) in zip(batch,caches,strict=True):
                panel=(tensor([r['queries']],cli.device),tensor([r['query_dt']],cli.device,floating=True))
                z=score(model,cache,*panel)[0].cpu().numpy()[0]
                charge+=value(prefix_literal(cache.seq_len))+value(query_ops(cache.seq_len,len(z)))
                if cli.canary:
                    single,_=cache_at(model,history,r['uid'],r['timestamp'])
                    torch.testing.assert_close(cache.k,single.k,atol=2e-5,rtol=2e-5)
                    torch.testing.assert_close(cache.v,single.v,atol=2e-5,rtol=2e-5)
                    direct=score(model,single,*panel)[0].cpu().numpy()[0]
                    np.testing.assert_allclose(z,direct,atol=2e-5,rtol=2e-5)
                    checks.append(dict(uid=r['uid'],target=t,count=cache.seq_len,
                        query_max_delta=float(np.max(np.abs(z-direct))),causal=True))
                results.extend(dict(row_index=int(i),fresh_current=float(v)) for i,v in zip(r['row_indices'],z,strict=True))
        del model
    name='canary8' if cli.canary else f'm{cli.target}'
    pd.DataFrame(results).to_parquet(OUT/f'{name}.parquet',index=False)
    save(OUT/f'{name}.json',dict(status='complete',checkpoints=len(cases),rows=len(results),
        seconds=time.monotonic()-started,checks=checks,evaluation_teacher_FLOPs=charge,
        scope='Evaluation-only teacher; not added to serving-method costs or available to decisions'))
    print(json.dumps(dict(stage=name,seconds=time.monotonic()-started,checkpoints=len(cases))),flush=True)


def queue():
    probe=json.loads((OUT/'canary8.json').read_text())
    assert probe['status']=='complete' and probe['checks']
    prep=json.loads((OUT/'preparation.json').read_text())
    estimate=probe['seconds']*prep['checkpoints']/probe['checkpoints']/4
    save(OUT/'queue_configuration.json',dict(estimated_seconds=estimate,gpus=[0,1,2,3],
        canary_seconds=probe['seconds'],canary_checkpoints=probe['checkpoints'],
        scope='Original1024 developmentUID fixed evaluation references, no new policies/fitting/confirmation',
        authorization='User asks benchmark-first evaluation and current-system assessment; ordinary frozen-model evaluation'))
    assert estimate<1800, 'Use detached execution with the required long-job procedure instead'
    started=time.monotonic()
    def run(pair):
        gpu,target=pair
        with (OUT/f'm{target}.log').open('x') as log:
            code=subprocess.call([sys.executable,'-u',__file__,'--mode','evaluate','--target',str(target),'--device',f'cuda:{gpu}'],
                cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:scripts'),stdout=log,stderr=subprocess.STDOUT)
        (OUT/f'm{target}.exit').write_text(str(code)+'\n')
        return dict(target=target,exit=code)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs=list(pool.map(run,enumerate(TARGETS)))
    save(OUT/'queue.json',dict(status='complete' if all(j['exit']==0 for j in jobs) else 'failed',jobs=jobs,seconds=time.monotonic()-started))
    print(json.dumps(jobs),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['prepare','evaluate','queue'],required=True)
    p.add_argument('--target',type=int);p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true')
    cli=p.parse_args()
    if cli.mode=='prepare': prepare()
    elif cli.mode=='queue': queue()
    else: evaluate(cli)
