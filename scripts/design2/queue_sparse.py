"""Short four-GPU frozen replay queue; no training or confirmation access."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from design.data import ROOT
from design2.audit_benchmark import save

def main(c):
    root=ROOT/'results/design2/sparse_01'
    name=('canary_'+c.mode) if c.mode in ['readobserve','fullreadobserve','excursion','excursionfull'] else ('canary_excursion' if c.mode=='excursionbg' else ('canary_observe' if c.mode in ['observe','bgobserve'] else 'canary_source'))
    if c.mode=='exactdemand':name='canary_exactdemand'
    if c.mode=='firstusematched':name='canary_firstusematched'
    canary=json.loads((root/name/'summary.json').read_text());assert canary['status']=='complete'
    out=root/f'{c.mode}_cap{c.cap}_{c.users}';out.mkdir(exist_ok=False)
    estimate=canary['seconds']*c.users/8/4
    assert estimate<1800
    jobs=[dict(offset=i,users=128,name=f'shard_{i:04d}') for i in range(0,c.users,128)]
    save(out/'configuration.json',dict(vars(c),jobs=jobs,estimate_seconds=estimate,protocol_sha256=__import__('hashlib').sha256((ROOT/'configs/design2/sparse_01.json').read_bytes()).hexdigest()))
    started=time.monotonic()
    def worker(g):
        rows=[]
        for j in jobs[g::4]:
            prior=root/f'{c.mode}_cap{c.cap}_512'/j['name']
            if c.users>512 and j['offset']<512 and (prior/'summary.json').exists():
                assert json.loads((prior/'summary.json').read_text())['status']=='complete'
                (out/j['name']).symlink_to(prior.resolve(),target_is_directory=True);code=0
            else:
                with (out/f'{j["name"]}.log').open('x') as log:
                    runner='scripts/design2/run_sparse_observe.py' if c.mode in ['observe','bgobserve'] else 'scripts/design2/run_sparse.py'
                    if c.mode in ['readobserve','fullreadobserve']:runner='scripts/design2/run_read_observe.py'
                    if c.mode.startswith('excursion') or c.mode=='firstusematched':runner='scripts/design2/run_excursion.py'
                    if c.mode=='exactdemand':runner='scripts/design2/run_exact_demand.py'
                    code=subprocess.call([sys.executable,runner,'--mode',c.mode,'--run-id',f'sparse_01/{out.name}/{j["name"]}',
                        '--users','128','--offset',str(j['offset']),'--cap',str(c.cap),'--device',f'cuda:{g}'],
                        cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:scripts',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
            (out/f'{j["name"]}.exit').write_text(str(code)+'\n');rows.append(dict(**j,exit=code));print(json.dumps(rows[-1]),flush=True)
            if code:break
        return rows
    with ThreadPoolExecutor(4) as pool:rows=[r for part in pool.map(worker,range(4)) for r in part]
    good=len(rows)==len(jobs) and all(r['exit']==0 for r in rows)
    save(out/'summary.json',dict(status='complete' if good else 'failed',seconds=time.monotonic()-started,jobs=rows));assert good

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['source','direct','background','observe','bgobserve','readobserve','fullreadobserve','excursion','excursionfull','excursionbg','exactdemand','firstusematched'],required=True);p.add_argument('--cap',type=int,default=1024);p.add_argument('--users',type=int,default=512);main(p.parse_args())
