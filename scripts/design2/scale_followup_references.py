"""Fixed FreshCurrent checkpoints and causal strata for new frozen lifetimes."""
import argparse,json,time,concurrent.futures,os,subprocess,sys
import numpy as np
import pandas as pd
import torch
from design.data import ROOT,DAY,histories
from design2.audit_benchmark import save
from design2.report_scale_followup import load,OUT,TARGETS
from design2 import benchmark_reference as reference


def prepare():
    ids=json.loads((ROOT/'configs/design2/scale_followup_01_uids.json').read_text())
    allusers=ids['original']+ids['extension'];h=histories(allusers,301)
    bounds=json.loads((ROOT/'results/design2/analysis/benchmark_01/calibration_bounds.json').read_text())
    cuts=np.array([231,245,259,273,287])*DAY;members=[]
    for cap in (1024,32,128):
        f,_,_,_=load('risk',cap);f=f.reset_index(names='row_index');cases=[]
        users=allusers if cap==1024 else ids['original'][:512]
        for u in users:
            times=h.rows[u][0]
            for t in TARGETS:
                stamp=cuts[t-1];end=np.searchsorted(times,stamp,side='left');kept=times[max(0,end-cap):end]
                producer=np.searchsorted(cuts,kept,side='right')
                prior=int(end-np.searchsorted(times,stamp-14*DAY,side='left'));idle=float((stamp-kept[-1])/DAY)
                rare=any((prior if b['feature']=='prior_writes' else idle)<b['lower'] or (prior if b['feature']=='prior_writes' else idle)>b['upper'] for b in bounds if b['target']==t)
                short=len(kept)<=128;old=t>=3 and np.mean(producer<=t-2)>=.25;mixed=len(np.unique(producer))>=3
                members.append(dict(uid=u,target=t,cap=cap,short=short,old_lineage=old,mixed=mixed,rare_activity=rare,
                    challenge=short or old or mixed or rare,routine=not(short or old or mixed or rare),retained_count=len(kept),
                    cohort='original' if u in set(ids['original']) else 'extension'))
        for (uid,t),g in f[f.target.isin(TARGETS)].groupby(['uid','target']):
            g=g.sort_values(['timestamp','item_idx','request_id']);stamps={int(g.timestamp.iloc[0])}
            for n in (1,128,1024,6144):
                eligible=g[g.writes_since_release>=n]
                if len(eligible):stamps.add(int(eligible.timestamp.iloc[0]))
            for stamp in sorted(stamps):
                q=g[g.timestamp==stamp];i,a,ts=h.prefix(uid,stamp,cap)
                cases.append(dict(uid=int(uid),target=int(t),timestamp=stamp,items=i,actions=a,times=ts,
                    row_indices=q.row_index.to_numpy(),queries=q.item_idx.to_numpy(),query_dt=int(stamp-ts[-1]),
                    writes_since_release=int(q.writes_since_release.iloc[0])))
        dest=ROOT/f'results/design2/scale_followup_01/references_cap{cap}';dest.mkdir(exist_ok=False)
        torch.save(cases,dest/'inputs.pt')
        save(dest/'preparation.json',dict(status='complete',users=users,checkpoints=len(cases),rows=sum(len(r['row_indices']) for r in cases),
            protocol='same first request plus writes1/128/1024/6144 checkpoints; first512 stress users, all2048 natural; evaluation only'))
    OUT.mkdir(exist_ok=True,parents=True);pd.DataFrame(members).to_parquet(OUT/'members.parquet',index=False)


def run(cli):
    reference.OUT=ROOT/f'results/design2/scale_followup_01/references_cap{cli.cap}'
    if cli.mode=='canary':
        cli.canary=True;cli.target=None;reference.evaluate(cli)
    elif cli.mode=='evaluate':
        cli.canary=False;reference.evaluate(cli)


def queue():
    root=ROOT/'results/design2/scale_followup_01';started=time.monotonic()
    for cap in (1024,32,128):
        assert json.loads((root/f'references_cap{cap}/canary8.json').read_text())['status']=='complete'
    save(root/'reference_queue_configuration.json',dict(gpus=[0,1,2,3],targets=[1,3,4,5],caps=[1024,32,128],
        estimated_seconds=300,basis='All three8UID canaries passed; prior2911-checkpoint four-GPU reference12.38s; allow300s for larger inputs and three caps'))
    def worker(pair):
        gpu,target=pair;rows=[]
        for cap in (1024,32,128):
            dest=root/f'references_cap{cap}'
            with (dest/f'm{target}.log').open('x') as log:
                code=subprocess.call([sys.executable,'-u',__file__,'--mode','evaluate','--cap',str(cap),'--target',str(target),'--device',f'cuda:{gpu}'],
                    cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:scripts'),stdout=log,stderr=subprocess.STDOUT)
            (dest/f'm{target}.exit').write_text(str(code)+'\n');rows.append(dict(target=target,cap=cap,exit=code))
            if code:break
        return rows
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=[r for batch in pool.map(worker,enumerate(TARGETS)) for r in batch]
    good=len(rows)==12 and all(r['exit']==0 for r in rows)
    save(root/'reference_queue.json',dict(status='complete' if good else 'failed',jobs=rows,seconds=time.monotonic()-started));assert good


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','canary','evaluate','queue'],required=True);p.add_argument('--cap',type=int,default=1024);p.add_argument('--target',type=int);p.add_argument('--device',default='cuda:0')
    cli=p.parse_args();prepare() if cli.mode=='prepare' else (queue() if cli.mode=='queue' else run(cli))
