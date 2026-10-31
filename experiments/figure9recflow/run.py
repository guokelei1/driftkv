"""Explicit preparation, bounded benchmark, or separately invoked formal run.

Examples use PYTHONPATH=src:. python -m experiments.figure9recflow.run ...
There are no automatic experiment successors and no content fingerprints.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import HERE, configuration, policies, write_json
from .inputs import RecFlowInputs


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def semantic_plan(path, value):
    """Resume requires the same declared inputs/settings; no file hashing."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Existing run has different declared settings: {path}')
    else:
        write_json(path, value)


def identity(payload):
    return payload


def prepare(output):
    inputs = RecFlowInputs()
    cfg = configuration()
    rows = []
    for edge in inputs.edges:
        record = dataclasses.asdict(edge)
        record['source_checkpoint'] = str(edge.source_checkpoint)
        record['target_checkpoint'] = str(edge.target_checkpoint)
        record['coverage'] = inputs.day(edge.evaluation_day).coverage()
        lengths = [len(s.events) for s in inputs.fit_snapshots(edge.evaluation_day)]
        record['fit_users'] = len(lengths)
        record['fit_history_rows'] = sum(lengths)
        record['fit_empty_users'] = lengths.count(0)
        rows.append(record)
    write_json(output / 'preparation.json', dict(config=cfg, policies=policies(cfg),
        edges=rows, fit_uids=inputs.fit_uids.tolist(), eval_uids=inputs.eval_uids.tolist(),
        formal_started=False, prepared_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print(json.dumps(dict(prepared=str(output), edges=len(rows), configs=len(policies(cfg)),
                         users=len(inputs.eval_uids), formal_started=False)), flush=True)


def fit_maps(inputs, edge, old, current, config, policy_rows, device, output):
    from .alignment import fit_cache_pairs
    from .cache_ops import encode_cache
    from .cost import RecFlowCostModel
    from .engine import tensors
    costs = RecFlowCostModel.from_model(current, attention_backend=config['backend'])
    map_policies = [p for p in policy_rows if p['method']=='kv_map']
    actual_policies = [{**p, 'fit_users':min(p['fit_users'],config['fit_limit'])} for p in map_policies]
    snapshots = inputs.fit_snapshots(edge.evaluation_day)[:config['fit_limit']]
    teacher = np.cumsum([2*costs.full_cache(len(s.events)) for s in snapshots]).tolist()
    def pairs():
        with torch.no_grad():
            for index, snapshot in enumerate(snapshots):
                args = tensors(snapshot.features, snapshot.behaviors, snapshot.deltas, device)
                yield encode_cache(old, *args), encode_cache(current, *args)
                if (index+1)%16==0 or index+1==len(snapshots):
                    write_json(output/'fit_progress.json', dict(stage='capture',users=index+1,total=len(snapshots)))
    def progress(tag, profile):
        print(json.dumps(dict(stage='fit',edge=edge.id,tag=tag,seconds=profile['seconds'],
                              steps=profile['steps'])),flush=True)
        write_json(output/'fit_progress.json',dict(stage='fit',last=tag,profile=profile))
    result = fit_cache_pairs(pairs(),actual_policies,device=device,
        num_heads=current.backbone.cfg.num_heads,
        settings={**config['fit'],'epochs':config['fit_epochs']},seed=config['seed'],
        teacher_flops=teacher,progress=progress)
    result['actual_policies'] = actual_policies
    result['diagnostic_only'] = config['fit_limit'] !=256 or config['fit_epochs']!=8
    result['edge_id'] = edge.id
    atomic_torch_save(output/'maps.pt', result)
    write_json(output/'fit_summary.json', {key:value for key,value in result.items() if key!='maps'})
    return result


def worker(args):
    from .engine import load_model, ReleaseEngine
    inputs = RecFlowInputs()
    edge = inputs.edges[args.edge-1]
    plan = json.loads((args.output/'plan.json').read_text())
    config, policy_rows = plan['config'], plan['policies']
    out = args.output/edge.id
    out.mkdir(parents=True,exist_ok=True)
    # One ordinary output lock prevents accidental duplicate execution.
    lock = (out/f'rank_{args.rank}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(args.threads)
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.reset_peak_memory_stats(device)
    begin = time.perf_counter()
    old = load_model(edge.source_checkpoint,inputs.data,device,backend=config['backend'])
    current = load_model(edge.target_checkpoint,inputs.data,device,backend=config['backend'])
    for model in (old,current):
        if not model.checkpoint_metadata['days'] or max(model.checkpoint_metadata['days'])>=edge.evaluation_day:
            raise ValueError('Checkpoint training reaches into the evaluation day')
        if model.checkpoint_metadata['scale']!='max':
            raise ValueError('The current Figure9d population plan requires actual Max checkpoints')
    if old.backbone.cfg != current.backbone.cfg:
        raise ValueError('Release endpoints have different backbone geometry')
    torch.cuda.synchronize(device)
    loaded = time.perf_counter()
    print(json.dumps(dict(stage='models_loaded',edge=edge.id,rank=args.rank,seconds=loaded-begin)),flush=True)
    maps_path = out/'maps.pt'
    if args.rank==0 and not maps_path.exists():
        fitted = fit_maps(inputs,edge,old,current,config,policy_rows,device,out)
    else:
        while not maps_path.exists():
            if (args.output/'failure.json').exists():
                raise RuntimeError('Coordinator reported worker failure')
            time.sleep(2)
        fitted = torch.load(maps_path,map_location='cpu',weights_only=False)
    maps = {name:weight.to(device) for name,weight in fitted['maps'].items()}
    engine = ReleaseEngine(old,current,policy_rows,maps,candidate_chunk=config['candidate_chunk'],
        backend=config['backend'],append_backend=config['append_backend'],cuda_graphs=config.get('cuda_graphs',False))
    all_uids = inputs.eval_uids[:args.users]
    uids = all_uids[args.rank::args.world]
    user_dir = out/'users'
    user_dir.mkdir(exist_ok=True)
    remaining = [int(uid) for uid in uids if not (user_dir/f'{uid}.pt').exists()]
    dataset = inputs.day(edge.evaluation_day,uids=remaining)
    kwargs = dict(batch_size=None,num_workers=args.prefetch_workers,collate_fn=identity)
    if args.prefetch_workers:
        kwargs.update(multiprocessing_context='spawn',prefetch_factor=2,persistent_workers=True)
    loader = DataLoader(dataset,**kwargs)
    scoring_start = time.perf_counter()
    for index, payload in enumerate(loader):
        result = engine.run_user(payload,save_scores=args.full_scores)
        atomic_torch_save(user_dir/f'{payload.uid}.pt',result)
        progress = dict(edge=edge.id,rank=args.rank,completed=index+1,remaining_total=len(dataset),
            uid=payload.uid,requests=result['ledger']['requests'],
            last_user_seconds=result['ledger']['total_seconds'],
            elapsed_seconds=time.perf_counter()-scoring_start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        write_json(out/f'progress_rank{args.rank}.json',progress)
        print(json.dumps(progress),flush=True)
    write_json(out/f'done_rank{args.rank}.json',dict(rank=args.rank,uids=uids.tolist(),
        model_load_seconds=loaded-begin,worker_seconds=time.perf_counter()-begin,
        scoring_seconds=time.perf_counter()-scoring_start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),completed=True))


def summarize_edge(inputs, edge, output, config, policy_rows, users):
    from .analysis import summarize
    uids = inputs.eval_uids[:users]
    folder=output/edge.id
    records,ledgers=[],[]
    for uid in uids:
        result=torch.load(folder/'users'/f'{int(uid)}.pt',map_location='cpu',weights_only=False)
        if result['uid']!=int(uid):
            raise ValueError('UID output identity differs')
        records.extend(result['records'])
        ledgers.append(result['ledger'])
    stats=summarize(records,uids,policy_rows,bootstrap=config['bootstrap_replicates'],seed=config['bootstrap_seed'])
    fitted=json.loads((folder/'fit_summary.json').read_text())
    rebuild=sum(x['once_rebuild_flops'] for x in ledgers)
    floprows=[]
    for i,p in enumerate(policy_rows):
        fitting=fitted['costs'].get(p['tag'],0)
        repair=sum(x['repair_flops'][i] for x in ledgers)
        floprows.append(dict(tag=p['tag'],fit_flops=fitting,repair_flops=repair,
            additional_flops=fitting+repair,once_rebuild_flops=rebuild,
            percent_of_once_rebuild=100*(fitting+repair)/rebuild if rebuild else None,
            common_source_flops=sum(x['source_cache_flops'] for x in ledgers),
            ordinary_append_flops=sum(x['ordinary_append_flops_per_policy'] for x in ledgers),
            ordinary_read_flops=sum(x['ordinary_read_flops_per_policy'] for x in ledgers)))
        stats['results'][i]['cost_percent']=floprows[-1]['percent_of_once_rebuild']
    stats.update(edge_id=edge.id,source=edge.source,target=edge.target,evaluation_day=edge.evaluation_day,
        costs=floprows,execution=dict(user_seconds=sum(x['total_seconds'] for x in ledgers),
            initialization_seconds=sum(x['initialization_seconds'] for x in ledgers),
            maintenance_and_read_seconds=sum(x['maintenance_and_read_seconds'] for x in ledgers),
            requests=sum(x['requests'] for x in ledgers),candidates=sum(x['candidates'] for x in ledgers),
            append_events=sum(x['append_events'] for x in ledgers),
            post_last_query_append_events=sum(x['post_last_query_append_events'] for x in ledgers),
            no_query_append_events=sum(x['no_query_append_events'] for x in ledgers),
            append_device_seconds=sum(x.get('append_device_seconds',0) for x in ledgers),
            read_device_seconds=sum(x.get('read_device_seconds',0) for x in ledgers)),
        scope='bounded_benchmark' if users<30000 else 'full_population',diagnostic_maps=fitted['diagnostic_only'])
    write_json(folder/'summary.json',stats)
    return stats


def controller(args):
    inputs=RecFlowInputs()
    cfg=configuration()
    cfg.update(backend=args.backend,append_backend=args.append_backend or args.backend,
        fit_limit=args.fit_users,fit_epochs=args.fit_epochs,candidate_chunk=args.chunk,cuda_graphs=args.cuda_graphs)
    if args.command=='formal':
        if args.users!=30000 or args.fit_users!=256 or args.fit_epochs!=8:
            raise ValueError('Formal requires train30000 and full prerelease fitting recipe')
    elif args.users>64:
        raise ValueError('Benchmark is bounded to at most64 fixed users per edge')
    if args.users<1 or not 1<=args.fit_users<=256 or args.fit_epochs<1:
        raise ValueError('Invalid bounded run dimensions')
    if len(set(args.gpus))!=len(args.gpus):
        raise ValueError('Use distinct physical GPUs')
    plan=dict(config=cfg,policies=policies(cfg),mode=args.command,edges=args.edges,
        users=args.users,gpus=args.gpus,full_scores=args.full_scores,
        eval_uids=inputs.eval_uids[:args.users].tolist(),fit_uids=inputs.fit_uids[:args.fit_users].tolist(),
        selected_edges=[dict(id=inputs.edges[e-1].id,
            source_checkpoint=str(inputs.edges[e-1].source_checkpoint),
            target_checkpoint=str(inputs.edges[e-1].target_checkpoint),
            evaluation_day=inputs.edges[e-1].evaluation_day) for e in args.edges])
    args.output.mkdir(parents=True,exist_ok=True)
    semantic_plan(args.output/'plan.json',plan)
    lock=(args.output/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (args.output/'failure.json').exists():
        # A retry is explicit, with old failure preserved under a timestamp.
        (args.output/'failure.json').rename(args.output/f'failure_previous_{time.time_ns()}.json')
    begin=time.perf_counter()
    summaries=[]
    for edge_no in args.edges:
        children,logs=[],[]
        try:
            for rank,gpu in enumerate(args.gpus):
                log=(args.output/f'edge{edge_no}_rank{rank}.log').open('a',buffering=1)
                logs.append(log)
                command=[sys.executable,'-m','experiments.figure9recflow.run','worker',
                    '--output',str(args.output),'--edge',str(edge_no),'--rank',str(rank),
                    '--world',str(len(args.gpus)),'--gpu',str(gpu),'--users',str(args.users),
                    '--threads',str(args.threads),'--prefetch-workers',str(args.prefetch_workers)]
                if args.full_scores: command.append('--full-scores')
                children.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
            write_json(args.output/'live.json',dict(edge=edge_no,pids=[p.pid for p in children],gpus=args.gpus))
            while any(p.poll() is None for p in children):
                failures=[p.returncode for p in children if p.poll() is not None and p.returncode!=0]
                if failures: raise RuntimeError(f'Worker failures: {failures}; inspect per-rank logs')
                time.sleep(2)
            if any(p.returncode for p in children): raise RuntimeError('Worker failed')
        except BaseException:
            write_json(args.output/'failure.json',dict(edge=edge_no,error=traceback.format_exc()))
            for p in children:
                if p.poll() is None: p.terminate()
            for p in children:
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired: p.kill(); p.wait()
            raise
        finally:
            for log in logs: log.close()
        summary=summarize_edge(inputs,inputs.edges[edge_no-1],args.output,cfg,plan['policies'],args.users)
        summaries.append(summary)
        print(json.dumps(dict(edge=edge_no,complete=True,execution=summary['execution'])),flush=True)
    from .analysis import aggregate_edges
    result=aggregate_edges(summaries,expected_edges=[inputs.edges[e-1].id for e in args.edges])
    result.update(mode=args.command,wall_seconds=time.perf_counter()-begin,formal_started=args.command=='formal',
                  automatic_successors=False)
    write_json(args.output/'summary.json',result)


def main():
    # Turn coordinator termination into cleanup, so a stopped tmux controller
    # does not leave scoring workers behind. Atomic committed users survive.
    def stop_controller(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    pre=sub.add_parser('prepare')
    pre.add_argument('--output',type=Path,default=HERE/'runs/preparation')
    for name in ('benchmark','formal'):
        p=sub.add_parser(name)
        p.add_argument('--output',type=Path,required=True)
        p.add_argument('--edges',type=int,nargs='+',default=list(range(1,8)))
        p.add_argument('--gpus',type=int,nargs='+',default=[0])
        p.add_argument('--users',type=int,default=8 if name=='benchmark' else 30000)
        p.add_argument('--fit-users',type=int,default=8 if name=='benchmark' else 256)
        p.add_argument('--fit-epochs',type=int,default=1 if name=='benchmark' else 8)
        p.add_argument('--chunk',type=int,default=128)
        p.add_argument('--backend',choices=['torch','triton'],default='torch')
        p.add_argument('--append-backend',choices=['torch','triton'])
        p.add_argument('--threads',type=int,default=4)
        p.add_argument('--prefetch-workers',type=int,default=2)
        p.add_argument('--full-scores',action='store_true')
        p.add_argument('--cuda-graphs',action='store_true')
    p=sub.add_parser('worker')
    p.add_argument('--output',type=Path,required=True)
    for name in ('edge','rank','world','gpu','users'):
        p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--prefetch-workers',type=int,default=2)
    p.add_argument('--full-scores',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.output)
    elif args.command=='worker': worker(args)
    else:
        signal.signal(signal.SIGTERM,stop_controller)
        controller(args)


if __name__=='__main__':
    main()
