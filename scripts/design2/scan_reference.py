"""All real-request references for the expanded cohort; no policy feedback."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd,torch
from design.data import ROOT
from design2.report_sparse import load
from design2 import full_reference as full
from design2.audit_benchmark import save

RUN=ROOT/'results/design2/scan_30k_01';BASE=RUN/'references'

def prepare():
    BASE.mkdir(exist_ok=False);out=BASE/'cap1024';out.mkdir()
    if not (RUN/'frozen/summary.json').exists():
        jobs=[j for j in json.loads((RUN/'queue_configuration.json').read_text())['jobs'] if j['policy']=='frozen']
        for j in jobs:
            assert (RUN/f'{j["name"]}.exit').read_text().strip()=='0'
            assert json.loads((RUN/j['name']/'summary.json').read_text())['status']=='complete'
        save(RUN/'frozen/summary.json',dict(status='complete',jobs=[dict(j,name=j['name'].split('/')[-1],exit=0) for j in jobs]))
    f=load(RUN/'frozen')[0].reset_index(names='row_index');cases=[]
    for (uid,t,stamp),g in f[f.target.isin([1,3,4,5])].groupby(['uid','target','timestamp']):
        cases.append(dict(uid=int(uid),target=int(t),timestamp=int(stamp),row_indices=g.row_index.to_numpy(),queries=g.item_idx.to_numpy()))
    torch.save(cases,out/'inputs.pt');ids=json.loads((RUN/'uids.json').read_text())
    save(out/'preparation.json',dict(status='complete',users=ids['original']+ids['extension'],checkpoints=len(cases),rows=sum(len(c['queries']) for c in cases),scope='All requests, fixed cohorts; separate evaluation-only teacher'))

def queue():
    probe=json.loads((BASE/'cap1024/canary8.json').read_text());assert probe['status']=='complete'
    n=json.loads((BASE/'cap1024/preparation.json').read_text())['checkpoints']
    estimate=200*n/44300*1.5+60;assert estimate<1800
    save(BASE/'resources.json',dict(estimated_seconds=estimate,basis='Prior all-request44300-group reference queue200s onfourGPUs,1.5x allowance plus60s; canonical primitive retained',canary_seconds=probe['seconds']))
    started=time.monotonic();jobs=[]
    with ThreadPoolExecutor(4) as pool:
        def run(pair):
            gpu,t=pair;out=BASE/'cap1024'
            with (out/f'm{t}.log').open('x') as log:
                code=subprocess.call([sys.executable,__file__,'evaluate','--target',str(t),'--device',f'cuda:{gpu}'],cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:scripts',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
            (out/f'm{t}.exit').write_text(str(code)+'\n');return dict(target=t,exit=code)
        jobs=list(pool.map(run,enumerate([1,3,4,5])))
    good=all(j['exit']==0 for j in jobs);save(BASE/'summary.json',dict(status='complete' if good else 'failed',jobs=jobs,seconds=time.monotonic()-started));assert good

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','evaluate','queue']);p.add_argument('--target',type=int);p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');c=p.parse_args();c.cap=1024
    full.BASE=BASE
    prepare() if c.mode=='prepare' else (queue() if c.mode=='queue' else full.evaluate(c))
