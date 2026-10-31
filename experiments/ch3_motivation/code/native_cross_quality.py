#!/usr/bin/env python3
"""Current-reader native K/V crosses; request-local diagnostics, no learned F.

Reuse and once-Rebuild append independently through the current native model.
Crossed K/V are constructed only for a read and are never persistent states.
The tested batched helper retains one duplicate Reuse member (research-only
redundancy); its identity adapter is not a trained correction or a fit.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.common import load_edge, load_models, write_json
from experiments.ch3_motivation.code.kv_evolution import clone_native_trajectory
from experiments.ch3_motivation.code.real_quality_paths import (
    BatchedQualityPaths, _member, _pack, _check_numeric, CACHE_NUMERIC_ATOL, NUMERIC_ATOL,
)
from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.common.artifacts import sha256
import hashlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

RUN=ROOT/'experiments/ch3_motivation/runs/real_quality_angles/attempt_1'
METHODS=('oldKoldV','teacherKoldV','oldKteacherV','teacherKteacherV')
SCHEMA=pa.schema([('uid',pa.uint64()),('query_timestamp',pa.int64()),('item_idx',pa.int64()),
    ('label',pa.int8()),('method',pa.string()),('score',pa.float64()),
    ('appended_events',pa.int64()),('history_rows',pa.int64()),('query_gap',pa.int64())])


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class _Identity:
    def prepare_cache(self,model,cache,ids):
        return cache


def crossed_cache(reuse,teacher):
    if reuse.seq_len!=teacher.seq_len or reuse.k.shape!=teacher.k.shape:
        raise ValueError('Native teacher and reuse event prefixes must align')
    return HSTUKVCache(torch.cat((reuse.k,teacher.k,reuse.k,teacher.k),1),
        torch.cat((reuse.v,reuse.v,teacher.v,teacher.v),1),reuse.seq_len)


class NativeCross:
    @torch.no_grad()
    def __init__(self,parent,current,events,cfg,check_independent=False):
        self.paths=BatchedQualityPaths(parent,current,_Identity(),events,cfg,
            history_cap=int(cfg['history_length']),structural_controls=False)
        self.model=current;self.check=bool(check_independent)
        self.max_cache_abs_error=self.max_score_abs_error=0.
        self.references=None
        if self.check:
            source=NativeTrajectory(parent,int(cfg['edge'])-1,events,int(cfg['release_day'])*86400,
                max_length=int(cfg['history_length']),append_band_size=int(cfg['append_band_size']))
            reuse,teacher=clone_native_trajectory(source),clone_native_trajectory(source)
            reuse.switch(current,int(cfg['edge']),source.time)
            teacher.switch(current,int(cfg['edge']),source.time,rebuild=True)
            self.references=(reuse,teacher)

    def caches(self):
        return tuple(_member(self.paths.cache,self.paths.index[name]) for name in ('reuse','rebuild'))

    @torch.no_grad()
    def advance(self,timestamp):
        self.paths.advance(timestamp)
        if self.check:
            for native in self.references:native.advance(timestamp)
            actual=_pack(self.caches());expected=_pack([n.cache for n in self.references])
            self.max_cache_abs_error=max(self.max_cache_abs_error,
                float((actual.k-expected.k).abs().max()),float((actual.v-expected.v).abs().max()))
            _check_numeric(actual.k,expected.k,'native_cross_key',('reuse','rebuild'),atol=CACHE_NUMERIC_ATOL)
            _check_numeric(actual.v,expected.v,'native_cross_value',('reuse','rebuild'),atol=CACHE_NUMERIC_ATOL)

    def snapshot(self):
        return crossed_cache(*self.caches())

    @torch.no_grad()
    def score(self,items,delta,cache=None):
        cache=self.snapshot() if cache is None else cache
        versions=self.paths.cache.k._version,self.paths.cache.v._version
        values=native_score(self.model,cache,items.expand(4,-1),delta.expand(4))[0]
        if not bool(torch.isfinite(values).all()):raise ValueError('Nonfinite native cross score')
        if versions!=(self.paths.cache.k._version,self.paths.cache.v._version):
            raise ValueError('Request-local crosses modified persistent trajectories')
        if self.check:
            r,t=[n.cache for n in self.references]
            independent=(r,HSTUKVCache(t.k,r.v,r.seq_len),HSTUKVCache(r.k,t.v,r.seq_len),t)
            expected=torch.cat([native_score(self.model,c,items,delta)[0] for c in independent],0)
            self.max_score_abs_error=max(self.max_score_abs_error,float((values-expected).abs().max()))
            _check_numeric(values,expected,'native_cross_score',METHODS,atol=NUMERIC_ATOL)
        return values


@torch.no_grad()
def replay_user(parent,current,events,requests,uid,cfg,*,check_independent=False,panel=None):
    state=NativeCross(parent,current,events,cfg,check_independent)
    paths=state.paths;device=paths.device;candidate=None
    if panel is not None:
        gap=0 if paths.last_timestamp is None else paths.time-paths.last_timestamp
        if int(panel['query_delta'])!=gap or int(panel['history_length'])!=paths.initial_length:
            raise ValueError('Original candidate panel cutover history/time binding differs')
        pool=torch.as_tensor(panel['pool_ids'],dtype=torch.long,device=device)[None]
        delta=torch.tensor([gap],dtype=torch.float32,device=device)
        cache=state.snapshot()
        scores=torch.cat([state.score(pool[:,i:i+16],delta,cache) for i in range(0,64,16)],1)
        candidate=dict(uid=np.uint64(uid),history_length=np.int32(paths.initial_length),
            query_delta=np.int64(gap),pool_ids=np.asarray(panel['pool_ids'],dtype=np.int64),
            panel_indices=np.asarray(panel['panel_indices'],dtype=np.int16),
            scores=scores.cpu().numpy().astype(np.float32,copy=False))
    times=sorted(int(t) for t,rows in requests.items() if rows)
    if any(not paths.time<=t<paths.end_time for t in times):raise ValueError('Request outside release window')
    raw=[row for t in times for row in requests[t]]
    items=torch.tensor([[int(r['item_idx']) for r in raw]],dtype=torch.long,device=device)
    gaps=[0 if (n:=int(np.searchsorted(events[0],t,side='left')))==0 else t-int(events[0][n-1]) for t in times]
    deltas=torch.tensor(gaps,dtype=torch.float32,device=device)
    rows=[];origin=0
    for ti,timestamp in enumerate(times):
        state.advance(timestamp);cache=state.snapshot();request=requests[timestamp]
        for start in range(0,len(request),16):
            selected=request[start:start+16]
            values=state.score(items[:,origin+start:origin+start+len(selected)],deltas[ti:ti+1],cache).cpu().tolist()
            for method,scores in zip(METHODS,values,strict=True):
                for r,value in zip(selected,scores,strict=True):
                    if int(r['label']) not in (0,1):raise ValueError('Binary feedback required')
                    rows.append(dict(uid=int(uid),query_timestamp=timestamp,item_idx=int(r['item_idx']),
                        label=int(r['label']),method=method,score=float(value),
                        appended_events=paths.appended_tokens,history_rows=paths.cache.seq_len,query_gap=int(gaps[ti])))
        origin+=len(request)
    stats=dict(uid=int(uid),feedback_events=len(raw),request_timestamps=len(times),
        initial_history_length=paths.initial_length,appended_events=paths.appended_tokens,
        history_rows=paths.cache.seq_len,evicted_events=paths.evicted_tokens,
        max_cache_abs_error=state.max_cache_abs_error if check_independent else None,
        max_score_abs_error=state.max_score_abs_error if check_independent else None,
        candidate_panel=panel is not None)
    native_ledger=paths.ledger()
    native_ledger.pop('f_materializations')
    ledger=dict(native_execution=native_ledger,learned_f_used=False,identity_duplicate_reuse_member=True,
        persistent_states='Native Reuse and once-Rebuild, plus redundant duplicate Reuse batch member',
        cross_scope='Request-local K/V crosses only; no crossed writes; current model reader for all4',
        deployment_cost_claim=False,independent_reference_prefills=2 if check_independent else 0)
    return rows,stats,ledger,candidate


def load_panels(edge):
    folder=ROOT/f'experiments/ch3_motivation/runs/insight_extensions/attempt_1/candidate_edge{edge}'
    config=json.loads((folder/'configuration.json').read_text());frozen=digest(config)
    panels={};artifacts=[]
    for path in sorted((folder/'main').glob('*.npz')):
        manifest=path.with_suffix('.json');record=json.loads(manifest.read_text())
        if record['configuration_sha256']!=frozen or record['npz_sha256']!=sha256(path):
            raise ValueError('Original candidate input block changed')
        with np.load(path,allow_pickle=False) as data:
            if data['uid'].tolist()!=record['uids']:raise ValueError('Original panel UID binding changed')
            for i,uid in enumerate(data['uid']):
                panels[int(uid)]={k:data[k][i] for k in ('history_length','query_delta','pool_ids','panel_indices')}
        artifacts.append(dict(path=str(path.resolve()),sha256=sha256(path),manifest=str(manifest.resolve()),manifest_sha256=sha256(manifest)))
    if list(panels)!=config['roles']['main']:raise ValueError('Original main2000 panel order differs')
    return panels,dict(path=str(folder.resolve()),configuration_sha256=sha256(folder/'configuration.json'),
                      models=config['models'],main_inputs=config['inputs']['main'],artifacts=artifacts)


def run(args):
    if args.gpu!=1:raise ValueError('Only physical GPU1 authorized')
    os.environ['CUDA_VISIBLE_DEVICES']='1';torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    population=json.loads(args.population.read_text());fixed=population['users']
    if len(fixed)!=4000 or len(set(fixed))!=4000:raise ValueError('Fixed4000 population required')
    directory=ROOT/f'experiments/ch3_motivation/inputs/edge_{args.edge}'
    binding,histories,requests=load_edge(directory)
    for key in ('history','requests'):
        if sha256(directory/binding[key]['path'])!=binding[key]['sha256']:raise ValueError('Frozen input changed')
    if binding['population_sha256']!=population['source_population']['sha256']:raise ValueError('Original population binding changed')
    users=[u for u in fixed if requests.get(u)][:args.users] if args.diagnostic else fixed
    eligible=[u for u in users if requests.get(u)]
    if args.diagnostic and len(users)!=args.users:raise ValueError('Insufficient numeric users')
    if not args.diagnostic and (args.users!=4000 or len(eligible)<512):raise ValueError('Requires full4000 and >=512 feedback UIDs')
    cfg=dict(binding['settings'])
    if cfg['history_length']!=1024:raise ValueError('This bounded study uses original cap1024 only')
    panels,panel_source=load_panels(args.edge)
    if (panel_source['main_inputs']['binding_sha256']!=sha256(directory/'inputs.json')
            or panel_source['main_inputs']['history_sha256']!=binding['history']['sha256']):
        raise ValueError('Original candidate panel raw history/input binding differs')
    parent,current,models=load_models(args.edge,torch.device('cuda:0'))
    if models!=panel_source['models']:raise ValueError('Candidate panel checkpoint binding differs')
    sources={str(p.resolve()):sha256(p) for p in [Path(__file__),*[Path(__file__).with_name(n) for n in
        ('real_quality_paths.py','kv_evolution.py','common.py')],
        ROOT/'src/hstu_kvcache/evaluation/trajectory.py',ROOT/'src/hstu_kvcache/models/state_transition.py',
        ROOT/'src/hstu_kvcache/models/hstu.py',ROOT/'src/hstu_kvcache/models/attention.py',
        ROOT/'src/hstu_kvcache/adaptation/reader.py']}
    config=dict(format='native_cross_quality_v1',edge=args.edge,settings=cfg,users=users,
        fixed_resident_users=fixed,feedback_users=eligible,population=str(args.population.resolve()),
        population_sha256=sha256(args.population),input_binding_sha256=sha256(directory/'inputs.json'),
        models=models,sources=sources,physical_gpu=1,diagnostic=args.diagnostic,
        check_independent=args.check_independent,method_names=list(METHODS),candidate_method_names=list(METHODS),
        candidate_users=[u for u in users if u in panels and requests.get(u)],
        fixed_candidate_users=[u for u in fixed if u in panels and requests.get(u)],candidate_source=panel_source,
        reader='Current full model for all4 states; old denotes Reuse trajectory, not every row still being physically old',
        trajectory='Old-model native cutover prefix; teacher rebuilt once under current model; both append natively and independently',
        cross_scope='K/V cross only on same-timestamp reads, no persistent crossed writes; all4 are research diagnostic controls',
        candidate_scope='Original main with real feedback only; original unlabeled pool64/cutover/gap; not true feedback AUC. Original near panel is teacher-conditioned.',
        learned_f_used=False,identity_duplicate_reuse_member=True,deployment_cost_claim=False,block_users=args.block_users)
    args.output.mkdir(parents=True,exist_ok=True);cp=args.output/'configuration.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Use a new output attempt after changing frozen configuration')
    write_json(cp,config);frozen=digest(config);artifacts=[];candidate_artifacts=[];stats_all=[]
    started=time.perf_counter();completed=score_rows=0
    for offset in range(0,len(users),args.block_users):
        selected=users[offset:offset+args.block_users]
        path=args.output/f'blocks/part_{offset:06d}.parquet';manifest=path.with_suffix('.json')
        if manifest.exists():
            record=json.loads(manifest.read_text())
            if record['configuration_sha256']!=frozen or record['uids']!=selected or record['score_sha256']!=sha256(path):
                raise ValueError('Committed native-cross score block changed')
            if record['candidate_artifact'] and sha256(args.output/record['candidate_artifact']['path'])!=record['candidate_artifact']['sha256']:
                raise ValueError('Committed candidate scores changed')
        else:
            values=[];stats=[];ledgers=[];candidate=[]
            for uid in selected:
                if not requests.get(uid):continue
                rows,detail,ledger,panel=replay_user(parent,current,histories.rows[uid],requests[uid],uid,cfg,
                    check_independent=args.check_independent,panel=panels.get(uid))
                values.extend(rows);stats.append(detail);ledgers.append(dict(uid=uid,operations=ledger))
                if panel is not None:candidate.append(panel)
            if any(sha256(Path(p))!=h for p,h in sources.items()):raise ValueError('Numerical source changed before commit')
            path.parent.mkdir(exist_ok=True);temp=path.with_suffix('.parquet.writing')
            pq.write_table(pa.Table.from_pylist(values,schema=SCHEMA),temp);temp.replace(path)
            candidate_artifact=None
            if candidate:
                pp=args.output/f'candidate_blocks/part_{offset:06d}.npz';pp.parent.mkdir(exist_ok=True)
                temporary=pp.with_suffix('.npz.writing')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream,**{key:np.stack([c[key] for c in candidate]) for key in candidate[0]})
                temporary.replace(pp)
                candidate_artifact=dict(path=str(pp.relative_to(args.output)),sha256=sha256(pp),users=len(candidate))
            record=dict(configuration_sha256=frozen,uids=selected,scored_users=len(stats),score_rows=len(values),
                stats=stats,ledgers=ledgers,score_sha256=sha256(path),candidate_artifact=candidate_artifact)
            write_json(manifest,record)
        stats_all.extend(record['stats']);completed+=record['scored_users'];score_rows+=record['score_rows']
        reference=dict(path=str(path.relative_to(args.output)),sha256=sha256(path),
            manifest=str(manifest.relative_to(args.output)),manifest_sha256=sha256(manifest))
        artifacts.append(reference)
        if record['candidate_artifact']:candidate_artifacts.append(dict(record['candidate_artifact'],
            manifest=reference['manifest'],manifest_sha256=reference['manifest_sha256']))
        progress=dict(status='running',edge=args.edge,completed_residents=offset+len(selected),residents=len(users),
            completed_feedback_users=completed,feedback_users=len(eligible),seconds=time.perf_counter()-started)
        write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    summary=dict(status='complete',edge=args.edge,users=len(users),fixed_residents=4000,feedback_users=completed,
        score_rows=score_rows,artifacts=artifacts,candidate_artifacts=candidate_artifacts,
        candidate_users=sum(a['users'] for a in candidate_artifacts),diagnostic=args.diagnostic,
        quality_use_permitted=not args.diagnostic and completed>=512,physical_gpu=1,
        method_names=list(METHODS),candidate_method_names=list(METHODS),seconds=time.perf_counter()-started,
        max_cache_abs_error=max((s['max_cache_abs_error'] or 0 for s in stats_all),default=0),
        max_score_abs_error=max((s['max_score_abs_error'] or 0 for s in stats_all),default=0),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        deployment_cost_claim=False,learned_f_used=False,identity_duplicate_reuse_member=True)
    if completed!=len(eligible) or summary['candidate_users']!=len(config['candidate_users']):raise ValueError('Incomplete native/candidate coverage')
    write_json(args.output/'summary.json',summary);write_json(args.output/'progress.json',dict(progress,status='complete'))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edge',type=int,choices=(2,3),required=True)
    p.add_argument('--population',type=Path,default=RUN/'population.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--block-users',type=int,choices=range(1,129),default=64)
    p.add_argument('--users',type=int,default=4000)
    p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--check-independent',action='store_true')
    args=p.parse_args()
    try:
        run(args);write_json(args.output/'exit.json',dict(exit_code=0))
    except BaseException as error:
        write_json(args.output/'exit.json',dict(exit_code=1,error=str(error)));raise
