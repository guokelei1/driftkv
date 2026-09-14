"""Bounded compute-mechanism queues; cells persist independently of GPU order."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from design.data import ROOT
from design2.audit_benchmark import save

def main(c):
    root=ROOT/'results/design2/compute_01'
    probe=json.loads((root/f'canary_{c.mode}/summary.json').read_text());assert probe['status']=='complete'
    out=root/f'{c.mode}_cap{c.cap}_{c.users}';out.mkdir(exist_ok=False)
    assert c.users%128==0
    jobs=[dict(offset=i,users=128,name=f'shard_{i:04d}') for i in range(0,c.users,128)]
    estimate=probe['seconds']*c.users/8/4;assert estimate<1800
    save(out/'configuration.json',dict(vars(c),jobs=jobs,estimated_seconds=estimate,
        rule='same frozen128UID cells; no training, user authorized mechanism experiments'))
    start=time.monotonic()
    def worker(gpu):
        rows=[]
        for j in jobs[gpu::4]:
            prior=root/f'{c.mode}_cap{c.cap}_512'
            if c.users>512 and j['offset']<512 and (prior/'summary.json').exists():
                s=json.loads((prior/'summary.json').read_text());assert s['status']=='complete'
                (out/j['name']).symlink_to((prior/j['name']).resolve(),target_is_directory=True)
                rows.append(dict(**j,exit=0,reused_from=str(prior/j['name'])));continue
            with (out/f'{j["name"]}.log').open('x') as f:
                runner='scripts/design2/run_compute_min.py' if c.mode=='mincost' else 'scripts/design2/run_compute.py'
                code=subprocess.call([sys.executable,'-u',runner,'--mode',c.mode,
                    '--run-id',f'compute_01/{out.name}/{j["name"]}','--users','128','--offset',str(j['offset']),
                    '--cap',str(c.cap),'--device',f'cuda:{gpu}'],cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:scripts',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT)
            (out/f'{j["name"]}.exit').write_text(str(code)+'\n');rows.append(dict(**j,exit=code));print(json.dumps(rows[-1]),flush=True)
            if code:break
        return rows
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=[r for part in pool.map(worker,range(4)) for r in part]
    good=len(rows)==len(jobs) and all(r['exit']==0 for r in rows)
    save(out/'summary.json',dict(status='complete' if good else 'failed',jobs=rows,seconds=time.monotonic()-start));assert good

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['readshare','reserve','fifo','mincost'],required=True);p.add_argument('--users',type=int,default=512);p.add_argument('--cap',type=int,default=1024);main(p.parse_args())
