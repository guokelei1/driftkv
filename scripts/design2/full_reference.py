"""All real-query FreshCurrent references, using the existing evaluator."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import torch
from design2 import benchmark_reference as b
from design2.report_scale_followup import load
from design2.audit_benchmark import save
from design.data import histories

ROOT=b.ROOT;BASE=ROOT/'results/design2/sparse_01/full_references'

def prepare():
    BASE.mkdir(exist_ok=False)
    ids=json.loads((ROOT/'configs/design2/scale_followup_01_uids.json').read_text())
    for cap in [1024,32,128]:
        f=load('risk',cap)[0].reset_index(names='row_index');cases=[]
        for (uid,t,stamp),g in f[f.target.isin(b.TARGETS)].groupby(['uid','target','timestamp'],sort=True):
            cases.append(dict(uid=int(uid),target=int(t),timestamp=int(stamp),row_indices=g.row_index.to_numpy(),queries=g.item_idx.to_numpy()))
        out=BASE/f'cap{cap}';out.mkdir()
        torch.save(cases,out/'inputs.pt')
        users=ids['original']+ids['extension'] if cap==1024 else ids['original'][:512]
        save(out/'preparation.json',dict(status='complete',users=users,checkpoints=len(cases),rows=sum(len(c['queries']) for c in cases),protocol=json.loads((ROOT/'configs/design2/full_reference_01.json').read_text())))
        print(cap,len(cases),sum(len(c['queries']) for c in cases),flush=True)

def evaluate(c):
    b.OUT=BASE/f'cap{c.cap}';old=torch.load
    users=json.loads((b.OUT/'preparation.json').read_text())['users']
    def materialize(path,*args,**kwargs):
        data=old(path,*args,**kwargs)
        if str(path)!=str(b.OUT/'inputs.pt'):return data
        selected=[r for r in data if (c.target is None or r['target']==c.target) and (not c.canary or r['uid'] in users[:8])]
        h=histories(sorted({r['uid'] for r in selected}),301)
        for r in selected:
            i,a,ts=h.prefix(r['uid'],r['timestamp'],c.cap)
            assert len(i) and ts[-1]<r['timestamp']
            r.update(items=i,actions=a,times=ts,query_dt=int(r['timestamp']-ts[-1]))
        return selected
    torch.load=materialize
    # Use the canonical single-prefix primitive. The expanded batch8 canary
    # exceeded its old KV equality tolerance on16 entries; do not loosen it.
    b.cache_at_many=lambda model,h,requests,size:[b.cache_at(model,h,u,t) for u,t in requests]
    b.evaluate(c)

def queue():
    p=json.loads((BASE/'cap1024/canary8.json').read_text());assert p['status']=='complete'
    total=sum(json.loads((BASE/f'cap{c}/preparation.json').read_text())['checkpoints'] for c in [1024,32,128])
    estimate=p['seconds']*total/p['checkpoints']/4
    assert estimate<1800
    save(BASE/'queue_configuration.json',dict(estimated_seconds=estimate,canary_seconds=p['seconds'],canary_groups=p['checkpoints'],
        note='Same evaluator and prior short-cap canaries; current canary validates compact materialization on all requests. User authorized complete benchmark.'))
    started=time.monotonic()
    def worker(pair):
        gpu,t=pair;records=[]
        for cap in [1024,32,128]:
            out=BASE/f'cap{cap}'
            with (out/f'm{t}.log').open('x') as log:
                code=subprocess.call([sys.executable,__file__,'evaluate','--target',str(t),'--cap',str(cap),'--device',f'cuda:{gpu}'],
                    cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:scripts',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
            (out/f'm{t}.exit').write_text(str(code)+'\n');records.append(dict(target=t,cap=cap,exit=code));print(records[-1],flush=True)
            if code:break
        return records
    with ThreadPoolExecutor(4) as pool:records=[r for part in pool.map(worker,enumerate(b.TARGETS)) for r in part]
    good=len(records)==12 and all(r['exit']==0 for r in records)
    save(BASE/'queue.json',dict(status='complete' if good else 'failed',jobs=records,seconds=time.monotonic()-started));assert good

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','evaluate','queue']);p.add_argument('--cap',type=int,default=1024);p.add_argument('--target',type=int);p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');c=p.parse_args()
    prepare() if c.mode=='prepare' else (queue() if c.mode=='queue' else evaluate(c))
