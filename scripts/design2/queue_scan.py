"""User-authorized expanded development replay; fixed policies, four GPUs."""
import json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from design.data import ROOT
from design2.audit_benchmark import save

OUT=ROOT/'results/design2/scan_30k_01'

def main():
    ids=json.loads((OUT/'uids.json').read_text());n=len(ids['original'])+len(ids['extension'])
    probes={p:json.loads((OUT/f'canary_{p}/summary.json').read_text()) for p in ['frozen','witness']}
    assert all(s['status']=='complete' for s in probes.values())
    jobs=[dict(policy=p,offset=i,users=min(128,n-i),name=f'{p}/shard_{i:04d}') for p in ['frozen','witness'] for i in range(0,n,128)]
    config=dict(users=n,jobs=jobs,estimated_seconds=2*((n+127)//128)*150/4,
        basis='Prior128UID shards80-100s; allow150s each and two fixed policies. Detachedtmux because conservative total40min.',
        canary_seconds={p:s['seconds'] for p,s in probes.items()},authorization='User asked broad30k scenario scan and continued expanded experiments; no training/confirmation/new releases')
    save(OUT/'queue_configuration.json',config);started=time.monotonic()
    for p in ['frozen','witness']:(OUT/p).mkdir(exist_ok=False)
    def worker(gpu):
        done=[]
        for j in jobs[gpu::4]:
            with (OUT/f'{j["name"]}.log').open('x') as log:
                code=subprocess.call([sys.executable,'scripts/design2/run_scan.py','--policy',j['policy'],'--run-id',f'scan_30k_01/{j["name"]}',
                    '--users',str(j['users']),'--offset',str(j['offset']),'--device',f'cuda:{gpu}'],cwd=ROOT,
                    env=dict(os.environ,PYTHONPATH='src:scripts',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
            (OUT/f'{j["name"]}.exit').write_text(str(code)+'\n');row=dict(**j,exit=code);done.append(row);print(json.dumps(row),flush=True)
            if code:break
        return done
    with ThreadPoolExecutor(4) as pool:done=[j for part in pool.map(worker,range(4)) for j in part]
    good=len(done)==len(jobs) and all(j['exit']==0 for j in done)
    save(OUT/'queue_summary.json',dict(status='complete' if good else 'failed',jobs=done,seconds=time.monotonic()-started));assert good
    for p in ['frozen','witness']:save(OUT/p/'summary.json',dict(status='complete',jobs=[dict(j,name=j['name'].split('/')[-1]) for j in done if j['policy']==p]))

if __name__=='__main__':main()
