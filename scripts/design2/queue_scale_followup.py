"""One bounded four-GPU population per invocation, after the focused canary."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
from design.data import ROOT
from design2.audit_benchmark import save

def main(cli):
    root=ROOT/'results/design2/scale_followup_01'
    probe='canary_short'+str(cli.cap) if cli.cap<1024 else 'canary_'+cli.policy
    c=json.loads((root/probe/'summary.json').read_text());assert c['status']=='complete'
    out=root/f'{cli.policy}_cap{cli.cap}';out.mkdir(parents=True,exist_ok=False)
    users=2048 if cli.cap==1024 else 512
    jobs=[dict(offset=i,users=min(128,users-i),name=f'shard_{i:04d}') for i in range(0,users,128)]
    estimate=c['seconds']*users/8/4
    assert estimate<1800
    save(out/'configuration.json',dict(policy=cli.policy,cap=cli.cap,users=users,jobs=jobs,estimated_seconds=estimate,
        source='8UID same-cap full five-window canary, four-GPU extrapolation; loading overhead conservative',
        scope='fixed protocol scale_followup_01; no training/confirmation'))
    started=time.monotonic()
    def worker(gpu):
        results=[]
        for j in jobs[gpu::4]:
            run_id=f'scale_followup_01/{out.name}/{j["name"]}'
            with (out/f'{j["name"]}.log').open('x') as log:
                code=subprocess.call([sys.executable,'-u','scripts/design2/run_scale_followup.py','--run-id',run_id,
                    '--users',str(j['users']),'--offset',str(j['offset']),'--policy',cli.policy,'--cap',str(cli.cap),'--device',f'cuda:{gpu}'],
                    cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:scripts'),stdout=log,stderr=subprocess.STDOUT)
            (out/f'{j["name"]}.exit').write_text(str(code)+'\n')
            results.append(dict(**j,exit=code));print(json.dumps(results[-1]),flush=True)
            if code:break
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: results=[r for group in pool.map(worker,range(4)) for r in group]
    good=len(results)==len(jobs) and all(r['exit']==0 for r in results)
    save(out/'summary.json',dict(status='complete' if good else 'failed',jobs=results,seconds=time.monotonic()-started))
    assert good

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',choices=['risk','demand','hash30'],default='risk');p.add_argument('--cap',type=int,default=1024)
    main(p.parse_args())
