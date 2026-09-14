"""One explicit policy substitution in the unchanged four-branch replay harness."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from design.run import write_json
from design.report_native_flops import value, prefix_literal, query_ops, read_cost
from design2 import run_lifecycle as base
from hstu_kvcache.design2.conditional import arithmetic

ROOT=base.ROOT
CONFIG=ROOT/'configs/design2/lifecycle_cost_only_01.json'
OUT=ROOT/'results/design2/lifecycle_cost_only_01'


def cost_only(model,state,adapter,packs,factors,fitted,history,uid,target,cutover,ledger,charges,canary):
    n=state.cache.seq_len
    check=base.preparation_cost(state,target)+value(query_ops(n,16))+read_cost(16,1)+arithmetic(16)['panel_flops']
    rebuild=value(prefix_literal(n))+base.summary_build_cost(n)
    row=dict(uid=uid,target=target,count=n,group='cost_only',a=None,b=None,check_remaining=check,rebuild_remaining=rebuild,
        revision=state.revision,release_age=state.writes_since_release,
        previous_rebuild=None if state.last_rebuild is None else state.last_rebuild['target'],
        producer_counts={str(k):v for k,v in Counter(state.writer.segments[e[0]]['producer'] for e in state.writer.events).items()},
        u=None,estimate=None,numerical_fallback=False,accurate_max_relative_delta=None,
        action='rebuild' if check>=rebuild else 'continue',reason='cost_before_check' if check>=rebuild else 'cost_continue')
    if row['action']=='rebuild':base.current_rebuild(model,state,history,uid,cutover,ledger,charges,'cost_only_rebuild')
    row.update(rebuilds=state.rebuilds,post_revision=state.revision,anchored_native=state.anchored_native)
    return row


def replay(cli):
    # Research harness substitution is explicit and recorded. Native replay,
    # state lifecycle, feedback, comparisons and costs are exactly the same code.
    base.CONFIG=CONFIG
    base.decide=cost_only
    base.main(cli)
    p=ROOT/'results/design2'/cli.run_id/'configuration.json'
    c=json.loads(p.read_text());c.update(variant='cost_only',variant_source_sha256=base.sha(Path(__file__)),
        artifact_column_note='design2 output column is cost_only policy in this run; analysis renames it')
    write_json(p,c)


def queue(out=OUT, runner=Path(__file__), config=CONFIG):
    canary=json.loads((out/'canary8/summary.json').read_text());assert canary['status']=='complete'
    launch=out/'population';launch.mkdir(exist_ok=False)
    jobs=[dict(offset=i,users=64,name=f'shard_{i:04d}') for i in range(0,1024,64)]
    write_json(launch/'configuration.json',dict(jobs=jobs,gpus=[0,1,2,3],expected_wall_seconds=400,
        basis='Primary same1024 workload fourGPU234.30s; no extra geometry in this control; canary passed',
        config_sha256=base.sha(config),source_sha256=base.sha(runner)))
    start=time.perf_counter()
    def worker(gpu):
        rows=[]
        for j in jobs[gpu::4]:
            cmd=[sys.executable,'-u',str(runner),'--mode','replay','--run-id',f'{out.name}/{j["name"]}',
                 '--offset',str(j['offset']),'--users',str(j['users']),'--device',f'cuda:{gpu}','--estimate-seconds','100']
            with (out/f'{j["name"]}.log').open('x') as log:
                code=subprocess.call(cmd,cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:scripts'),stdout=log,stderr=subprocess.STDOUT)
            (out/f'{j["name"]}.exit').write_text(str(code)+'\n')
            rows.append(dict(**j,gpu=gpu,exit=code));print(json.dumps(rows[-1]),flush=True)
            if code:break
        return rows
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows=[r for group in pool.map(worker,range(4)) for r in group]
    status='complete' if len(rows)==16 and all(r['exit']==0 for r in rows) else 'failed'
    write_json(launch/'summary.json',dict(status=status,jobs=rows,elapsed_seconds=time.perf_counter()-start))
    assert status=='complete'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['replay','queue'],required=True)
    p.add_argument('--run-id');p.add_argument('--users',type=int);p.add_argument('--offset',type=int,default=0)
    p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');p.add_argument('--estimate-seconds',type=int,default=30)
    cli=p.parse_args()
    queue() if cli.mode=='queue' else replay(cli)
