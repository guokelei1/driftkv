#!/usr/bin/env python3
"""Cached-prefix-gap sensitivity of the frozen fixed-cutoff lineage scorer.

Every cache, including Rebuild, stays fixed. Query gaps use the final event
strictly before cutoff, shared across methods. No timestamp, weight, label,
or cache content changes from the original raw-history-gap study. This is not a
native streaming replay or a deployment cost measurement.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from experiments.ch3_motivation.code.common import load_edge, write_json
from experiments.ch3_motivation.code.state_lineage_probe import load_bank, load_native, native_snapshot
from experiments.ch3_motivation.code.simple_kv_maps import head_rows, rows_to_cache
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.common.artifacts import sha256

RUN=ROOT/'experiments/ch3_motivation/runs/lineage_candidate/attempt_1'
SCHEMA=pa.schema([('uid',pa.uint64()),('query_timestamp',pa.int64()),('item_idx',pa.int64()),
    ('label',pa.int8()),('method',pa.string()),('score',pa.float64()),('appended_events',pa.int64()),
    ('history_rows',pa.int64()),('query_gap',pa.int64()),('kv_mse',pa.float64()),
    ('k_mse',pa.float64()),('v_mse',pa.float64()),('initial_kv_mse',pa.float64())])


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def cached_prefix_gaps(events,times,cutoff):
    """Gap to the actual retained prefix end; ties at cutoff are excluded."""
    count=int(np.searchsorted(events[0],int(cutoff),side='left'))
    if count==0:return [0 for _ in times]
    last=int(events[0][count-1])
    return [int(t)-last for t in times]


@torch.no_grad()
def project_final(original,bank,target,heads=10):
    """Compute only chain intermediate states and the requested final outputs."""
    source=head_rows(original,heads);previous=source
    for version in range(2,target):
        previous=bank[version]['chain'](previous,previous)
    outputs={'direct':bank[target]['direct'](source,source),'chain':bank[target]['chain'](previous,previous)}
    return {kind:rows_to_cache(rows,original,heads) for kind,rows in outputs.items()}


@torch.no_grad()
def score_user(source_model,target_model,bank,events,requests,uid,target,cutoff,check=False):
    original=native_snapshot(source_model,1,events,cutoff)
    teacher=native_snapshot(target_model,target,events,cutoff)
    original_versions=(original.k._version,original.v._version)
    mapped=project_final(original,bank,target,target_model.cfg.num_heads)
    if original_versions!=(original.k._version,original.v._version):raise ValueError('Map mutated native source')
    max_cache=0.
    if check:
        heads=target_model.cfg.num_heads
        native_rows=head_rows(original,heads)
        reference={'direct':rows_to_cache(bank[target]['direct'](native_rows,native_rows),original,heads)}
        previous_cache=original
        for version in range(2,target+1):
            previous_rows=head_rows(previous_cache,heads)
            previous_cache=rows_to_cache(bank[version]['chain'](previous_rows,previous_rows),previous_cache,heads)
        reference['chain']=previous_cache
        for name,cache in mapped.items():
            for field in ('k','v'):
                a,b=getattr(cache,field),getattr(reference[name],field)
                max_cache=max(max_cache,float((a-b).abs().max()))
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    caches={'reuse':original,**mapped,'rebuild':teacher};methods=list(caches)
    packed=HSTUKVCache(torch.cat([c.k for c in caches.values()],1),
        torch.cat([c.v for c in caches.values()],1),original.seq_len)
    errors={}
    for name,c in caches.items():
        k=float((c.k.double()-teacher.k.double()).square().mean())
        v=float((c.v.double()-teacher.v.double()).square().mean())
        errors[name]=dict(k_mse=k,v_mse=v,kv_mse=(k+v)/2)
    times=sorted(int(t) for t,r in requests.items() if r)
    raw=[row for t in times for row in requests[t]];device=original.k.device
    items=torch.tensor([[int(r['item_idx']) for r in raw]],device=device,dtype=torch.long)
    if any(t<cutoff for t in times):raise ValueError('Request before fixed snapshot cutoff')
    gaps=cached_prefix_gaps(events,times,cutoff)
    deltas=torch.tensor(gaps,device=device,dtype=torch.float32)
    offset=0;rows=[];max_score=0.
    for ti,t in enumerate(times):
        for start in range(0,len(requests[t]),16):
            selected=requests[t][start:start+16];candidate=items[:,offset+start:offset+start+len(selected)]
            delta=deltas[ti:ti+1]
            values=native_score(target_model,packed,candidate.expand(len(methods),-1),delta.expand(len(methods)))[0]
            if not bool(torch.isfinite(values).all()):raise ValueError('Nonfinite candidate logit')
            if check:
                expected=torch.cat([native_score(target_model,c,candidate,delta)[0] for c in caches.values()],0)
                max_score=max(max_score,float((values-expected).abs().max()))
                torch.testing.assert_close(values,expected,atol=1e-6,rtol=1e-5)
            for name,scores in zip(methods,values.cpu().tolist(),strict=True):
                for request,value in zip(selected,scores,strict=True):
                    if int(request['label']) not in (0,1):raise ValueError('Observed binary labels required')
                    rows.append(dict(uid=int(uid),query_timestamp=t,item_idx=int(request['item_idx']),label=int(request['label']),
                        method=name,score=float(value),appended_events=0,history_rows=original.seq_len,query_gap=gaps[ti],
                        **errors[name],initial_kv_mse=errors[name]['kv_mse']))
        offset+=len(requests[t])
    stats=dict(uid=int(uid),history_rows=original.seq_len,feedback_events=len(raw),request_timestamps=len(times),
        method_names=methods,initial_mse=errors,max_cache_abs_error=max_cache if check else None,
        max_score_abs_error=max_score if check else None,native_prefills=2,appended_events=0,
        cache_scope='Fixed final-target cutoff for every method, including Rebuild',
        query_gap_protocol='cached_prefix_gap',query_gap_scope='Real request timestamp minus last event strictly before fixed cache cutoff; empty prefix0; no clipping')
    return rows,stats


def run(args):
    if args.gpu!=1:raise ValueError('Physical GPU1 only')
    population=json.loads(args.population.read_text());fixed=population['users'];edge=str(args.final_target)
    if len(set(fixed))!=len(fixed) or len(fixed)<512:raise ValueError('Fixed>=512 resident population required')
    entry=population['edges'][edge]
    for ref in entry['inputs'].values():
        if sha256(Path(ref['path']))!=ref['sha256']:raise ValueError('Frozen input changed')
    folder=Path(entry['inputs']['binding']['path']).parent
    binding,histories,requests=load_edge(folder);cutoff=int(binding['settings']['release_day'])*86400
    users=[uid for uid in fixed if requests.get(uid)][:args.users] if args.diagnostic else fixed
    eligible=[uid for uid in users if requests.get(uid)]
    if args.diagnostic and len(users)!=args.users:raise ValueError('Insufficient diagnostic feedback UIDs')
    if not args.diagnostic and len(eligible)<512:raise ValueError('At least512 feedback UIDs for quality')
    metadata=json.loads((args.checkpoint.parent/'summary.json').read_text())
    if sha256(args.checkpoint)!=metadata['checkpoint_sha256']:raise ValueError('Fit checkpoint changed')
    identity=metadata['identity']
    if (identity['final_target']!=args.final_target or identity['cutoff']!=cutoff
            or identity['population_sha256']!=sha256(args.population)
            or set(fixed)&set(identity['fit_uids']+identity['validation_users'])):
        raise ValueError('Checkpoint snapshot/population/role mismatch')
    os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['EVOKV_ATTENTION_BACKEND']='triton'
    torch.cuda.set_device(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    source_model,source_path=load_native(1,device);target_model,target_path=load_native(args.final_target,device)
    for p in (source_path,target_path):
        if sha256(p)!=identity['models'][str(p.relative_to(ROOT))]:raise ValueError('Native model changed')
    bank,payload=load_bank(args.checkpoint,device)
    if payload['identity']!=identity:raise ValueError('Checkpoint identity changed')
    methods=['reuse','direct','chain','rebuild']
    sources={str(p.resolve()):sha256(p) for p in (Path(__file__),Path(__file__).with_name('state_lineage_probe.py'),
        Path(__file__).with_name('simple_kv_maps.py'),Path(__file__).with_name('common.py'),
        Path(__file__).with_name('score_state_lineage.py'),
        ROOT/'src/hstu_kvcache/adaptation/reader.py',ROOT/'src/hstu_kvcache/evaluation/trajectory.py',
        ROOT/'src/hstu_kvcache/models/hstu.py',ROOT/'src/hstu_kvcache/models/state_transition.py',
        ROOT/'src/hstu_kvcache/models/attention.py')}
    config=dict(format='state_lineage_cached_gap_quality_v1',edge=args.final_target,final_target=args.final_target,
        users=users,fixed_resident_users=fixed,population=str(args.population.resolve()),population_sha256=sha256(args.population),
        input_binding_sha256=sha256(Path(entry['inputs']['binding']['path'])),models=identity['models'],
        checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha256(args.checkpoint),cutoff=cutoff,
        method_names=methods,method_order=methods,settings=binding['settings'],sources=sources,physical_gpu=1,
        diagnostic=args.diagnostic,check_independent=args.check_independent,block_users=args.block_users,
        torch_version=str(torch.__version__),snapshot_scope='Every method including Rebuild uses fixed final cutoff cache; no post-cutoff writes',
        source_version=1,migration_steps={'direct':1,'chain':args.final_target-1},
        unused_fit_branches='Archived checkpoint contains already completed unused anchored/short fits; they are not scored or compared here',
        query_gap_protocol='cached_prefix_gap',
        query_scope='Real observed candidates/labels at unchanged timestamps; shared query_ts minus last event strictly before cache cutoff, no clipping',
        derived_from_frozen_scorer_sha256=sha256(Path(__file__).with_name('score_state_lineage.py')),
        comparison_scope='Counterfactual fixed-cache mechanism study, not a full serving replay, deployment cost, or final Design evaluation')
    args.output.mkdir(parents=True,exist_ok=True);cp=args.output/'configuration.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Frozen scoring configuration changed')
    write_json(cp,config);frozen=digest(config);artifacts=[];scored=score_rows=0;all_stats=[]
    for offset in range(0,len(users),args.block_users):
        selected=users[offset:offset+args.block_users];block=args.output/f'blocks/part_{offset:06d}.parquet';mp=block.with_suffix('.json')
        if mp.exists():
            record=json.loads(mp.read_text())
            if record['configuration_sha256']!=frozen or record['uids']!=selected or sha256(block)!=record['score_sha256']:
                raise ValueError('Committed block changed')
        else:
            rows=[];stats=[]
            for uid in selected:
                if not requests.get(uid):continue
                values,detail=score_user(source_model,target_model,bank,histories.rows[uid],requests[uid],uid,
                    args.final_target,cutoff,args.check_independent)
                if detail['method_names']!=methods:raise ValueError('Unplanned branch list')
                rows.extend(values);stats.append(detail)
            if any(sha256(Path(p))!=h for p,h in sources.items()):raise ValueError('Numerical source changed')
            block.parent.mkdir(exist_ok=True);temp=block.with_suffix('.parquet.writing')
            pq.write_table(pa.Table.from_pylist(rows,schema=SCHEMA),temp);temp.replace(block)
            record=dict(configuration_sha256=frozen,uids=selected,score_sha256=sha256(block),stats=stats,
                scored_users=len(stats),score_rows=len(rows));write_json(mp,record)
        scored+=record['scored_users'];score_rows+=record['score_rows'];all_stats.extend(record['stats'])
        artifacts.append(dict(path=str(block.relative_to(args.output)),sha256=sha256(block),
            manifest=str(mp.relative_to(args.output)),manifest_sha256=sha256(mp)))
        progress=dict(status='running',completed_residents=offset+len(selected),residents=len(users),feedback_users=scored,
            seconds=time.perf_counter()-started);write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    if scored!=len(eligible):raise ValueError('Incomplete feedback UID coverage')
    summary=dict(status='complete',edge=args.final_target,users=len(users),feedback_users=scored,score_rows=score_rows,
        method_names=methods,method_order=methods,artifacts=artifacts,physical_gpu=1,
        diagnostic=args.diagnostic,quality_use_permitted=not args.diagnostic and scored>=512,
        seconds=time.perf_counter()-started,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        snapshot_scope=config['comparison_scope'],query_gap_protocol='cached_prefix_gap',independent_check_statistics=all_stats if args.diagnostic else None)
    write_json(args.output/'summary.json',summary);write_json(args.output/'progress.json',dict(progress,status='complete'))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--final-target',type=int,choices=(3,4),required=True)
    p.add_argument('--population',type=Path,default=RUN/'population.json')
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1);p.add_argument('--block-users',type=int,default=64)
    p.add_argument('--diagnostic',action='store_true');p.add_argument('--users',type=int,default=8)
    p.add_argument('--check-independent',action='store_true');args=p.parse_args()
    try:run(args);write_json(args.output/'exit.json',dict(exit_code=0))
    except BaseException as error:write_json(args.output/'exit.json',dict(exit_code=1,error=str(error)));raise
