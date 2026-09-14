"""Authorized two-policy mechanism experiment; old 4082 UID cohort frozen."""
import json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from design.data import ROOT
from design2.audit_benchmark import save

OUT=ROOT/'results/design2/bounded_01'
def main():
    checks={k:json.loads((OUT/f'canary_{k}/summary.json').read_text()) for k in ['post','bounded']}
    assert all(v['status']=='complete' for v in checks.values())
    probe=json.loads((OUT/'probe_bounded/summary.json').read_text());assert probe['status']=='complete'
    jobs=[dict(policy=p,offset=i,users=min(128,4082-i),name=f'{p}/shard_{i:04d}') for p in ['post','bounded'] for i in range(0,4082,128)]
    estimate=64*probe['seconds']*1.6/4
    save(OUT/'queue_configuration.json',dict(jobs=jobs,estimated_seconds=estimate,probe_seconds=probe['seconds'],basis='64 shards, four GPUs,1.6 workload allowance; detachedtmux; user authorized this mechanism validation; no training'))
    for p in ['post','bounded']:(OUT/p).mkdir(exist_ok=False)
    start=time.monotonic()
    def worker(gpu):
        done=[]
        for j in jobs[gpu::4]:
            with (OUT/f'{j["name"]}.log').open('x') as log:
                code=subprocess.call([sys.executable,'scripts/design2/run_bounded.py','--variant',j['policy'],'--run-id',f'bounded_01/{j["name"]}',
                    '--users',str(j['users']),'--offset',str(j['offset']),'--device',f'cuda:{gpu}'],cwd=ROOT,
                    env=dict(os.environ,PYTHONPATH='src:scripts',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
            (OUT/f'{j["name"]}.exit').write_text(str(code)+'\n');done.append(dict(**j,exit=code));print(json.dumps(done[-1]),flush=True)
            if code:break
        return done
    with ThreadPoolExecutor(4) as pool:done=[j for part in pool.map(worker,range(4)) for j in part]
    good=len(done)==64 and all(j['exit']==0 for j in done)
    save(OUT/'queue_summary.json',dict(status='complete' if good else 'failed',jobs=done,seconds=time.monotonic()-start));assert good
    for p in ['post','bounded']:save(OUT/p/'summary.json',dict(status='complete',jobs=[dict(j,name=j['name'].split('/')[-1]) for j in done if j['policy']==p]))
if __name__=='__main__':main()
