"""A paid Current observation gates committing a proposed real state rebuild."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from design.data import ROOT,prefix
from design.report_native_flops import query_ops,value
from design2 import run_scale_followup as r
from design2.audit_benchmark import save
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.design2.rebuild import current_kv,tiled_flops

CONFIG=ROOT/'configs/design2/scale_guard_01.json'
ADAPTERS={};REQUESTS={};ATTEMPTS={}
old_factory=r.FrozenC;old_requests=r.quality_requests;old_decide=r.decide


def factory(t,device):
    a=old_factory(t,device);ADAPTERS[t]=a;return a


def requests(*args):
    rows=old_requests(*args);REQUESTS.clear();REQUESTS.update(rows);return rows


def guarded_rebuild(model,state,history,uid,stamp,charges):
    adapter=ADAPTERS[state.target]
    actual=[q for q in REQUESTS[uid] if int(q['query_timestamp'])==stamp]
    assert actual
    device=next(model.parameters()).device
    i,a,d,ts=prefix(history,uid,stamp,device);new=current_kv(model,i,a,d);n=new.seq_len
    assert np.array_equal(ts,[e[3] for e in state.writer.events])
    panel=(r.original.tensor([[int(q['item_idx']) for q in actual]],device),r.original.tensor([stamp-int(ts[-1])],device,floating=True))
    old=r.base.corrected_score(model,state,adapter,panel,defaultdict(float),charges)
    new_z=score(model,new,*panel)[0]
    charges['rebuild_tiled']+=tiled_flops(n)
    charges['commit_probe_reads']+=2*value(query_ops(n,len(actual)))
    tau=json.loads((ROOT/'configs/design2/scale_calibration_01_fitted.json').read_text())['thresholds']['calibrated']['0.8']
    error=float((old-new_z).abs().max());commit=error>tau
    if commit:
        state.rebuild(new,ts,stamp);charges['rebuild_summary']+=r.base.summary_build_cost(n)
    ATTEMPTS[uid,state.target]=dict(commit_error=error,committed=commit,actual_queries=len(actual),attempted=True)


def decide(*args,**kwargs):
    row=old_decide(*args,**kwargs)
    key=row['uid'],row['target']
    if key in ATTEMPTS:
        row.update(ATTEMPTS.pop(key))
        if not row['committed']:row['action']='retain_after_paid_probe'
    return row


def replay(cli):
    r.FrozenC=factory;r.quality_requests=requests;r.rebuild_state=guarded_rebuild;r.decide=decide
    cli.policy='risk';r.main(cli)
    path=ROOT/'results/design2'/cli.run_id/'configuration.json';cfg=json.loads(path.read_text())
    cfg.update(variant='guard',guard_config=json.loads(CONFIG.read_text()),guard_sha256=r.base.sha(Path(__file__)))
    save(path,cfg)


def queue():
    root=ROOT/'results/design2/scale_followup_01'
    c=json.loads((root/'canary_guard/summary.json').read_text());assert c['status']=='complete'
    out=root/'guard_cap1024';out.mkdir(exist_ok=False)
    jobs=[dict(offset=i,users=128,name=f'shard_{i:04d}') for i in range(0,2048,128)]
    save(out/'configuration.json',dict(jobs=jobs,estimated_seconds=c['seconds']*2048/8/4,config=json.loads(CONFIG.read_text())))
    started=time.monotonic()
    def worker(gpu):
        rows=[]
        for j in jobs[gpu::4]:
            with (out/f'{j["name"]}.log').open('x') as log:
                code=subprocess.call([sys.executable,'-u',__file__,'--run-id',f'scale_followup_01/guard_cap1024/{j["name"]}',
                    '--users','128','--offset',str(j['offset']),'--device',f'cuda:{gpu}'],cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:scripts'),stdout=log,stderr=subprocess.STDOUT)
            (out/f'{j["name"]}.exit').write_text(str(code)+'\n');rows.append(dict(**j,exit=code));print(json.dumps(rows[-1]),flush=True)
            if code:break
        return rows
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=[j for part in pool.map(worker,range(4)) for j in part]
    good=len(rows)==16 and all(j['exit']==0 for j in rows)
    save(out/'summary.json',dict(status='complete' if good else 'failed',jobs=rows,seconds=time.monotonic()-started));assert good


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',action='store_true');p.add_argument('--run-id');p.add_argument('--users',type=int,default=8);p.add_argument('--offset',type=int,default=0);p.add_argument('--device',default='cuda:0');p.add_argument('--cap',type=int,default=1024);p.add_argument('--canary',action='store_true')
    cli=p.parse_args();queue() if cli.queue else replay(cli)
