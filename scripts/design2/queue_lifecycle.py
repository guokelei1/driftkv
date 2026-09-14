"""Original1024UID, four GPU queues; the eight-user canary is not duplicated in analysis."""

import concurrent.futures
import json
import os
import subprocess
import sys
import time

from design.data import ROOT
from design.run import write_json

OUT=ROOT/'results/design2/lifecycle_01'


def main():
    canary=json.loads((OUT/'canary8_02/summary.json').read_text())
    assert canary['status']=='complete' and len(canary['canary_checks'])==84
    out=OUT/'population'; out.mkdir(exist_ok=False)
    jobs=[dict(offset=i,users=64,name=f'shard_{i:04d}') for i in range(0,1024,64)]
    write_json(out/'configuration.json',dict(jobs=jobs,gpus=[0,1,2,3],estimated_wall_seconds=900,
        basis='8UID complete lifetime21.41s including repeated model loading and full canary references; four GPUs1024/8*21.41/4=685s, 900s allowance',
        peak_gpu_mib_estimate=10000,authorization='2026-09-09 user requests original1024UID real continuous closed loop; canary passed; no long training',
        cost_corrections='Before population only: charge FrozenC source norm and literal padded summary reduction; same detector/actions. Canary quality excluded from population aggregate.'))
    start=time.perf_counter()
    def queue(gpu):
        records=[]
        for j in jobs[gpu::4]:
            cmd=[sys.executable,'-u','scripts/design2/run_lifecycle.py','--run-id',f'lifecycle_01/{j["name"]}',
                '--offset',str(j['offset']),'--users',str(j['users']),'--device',f'cuda:{gpu}','--estimate-seconds','225']
            with (OUT/f'{j["name"]}.log').open('x') as log:
                code=subprocess.call(cmd,cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:scripts'),stdout=log,stderr=subprocess.STDOUT)
            (OUT/f'{j["name"]}.exit').write_text(str(code)+'\n')
            records.append(dict(**j,gpu=gpu,exit=code)); print(json.dumps(records[-1]),flush=True)
            if code:break
        return records
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records=[r for part in pool.map(queue,range(4)) for r in part]
    status='complete' if len(records)==16 and all(r['exit']==0 for r in records) else 'failed'
    write_json(out/'summary.json',dict(status=status,jobs=records,elapsed_seconds=time.perf_counter()-start))
    assert status=='complete'


if __name__=='__main__':main()
