#!/usr/bin/env python3
"""Read-only strengths of one frozen inherited-KV correction on shared writes.

One native Reuse trajectory consumes every event causally. A cutover-only map
defines a fixed delta on inherited old rows. Request-local lambda reads never
write back, and never modify newly written current-native rows.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
from collections import Counter
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
from experiments.ch3_motivation.code.common import load_edge, load_models, write_json
from experiments.ch3_motivation.code.fit_candidate_robust_maps import load_robust_maps
from experiments.ch3_motivation.code.kv_evolution import clone_native_trajectory
from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.evaluation.trajectory import NativeTrajectory, MAX_DELTA_SECONDS, _short_prefix_backend
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.models.state_transition import append_with_rolling_band
from hstu_kvcache.common.artifacts import sha256

ROBUST=ROOT/'experiments/ch3_motivation/runs/candidate_robustness/attempt_1'
GRID=(('lambda0',0.),('lambda025',.25),('lambda05',.5),('lambda075',.75),('lambda1',1.),
      ('lambda125',1.25),('lambda15',1.5),('lambda175',1.75),('lambda2',2.))
SCHEMA=pa.schema([('uid',pa.uint64()),('query_timestamp',pa.int64()),('item_idx',pa.int64()),
    ('label',pa.int8()),('method',pa.string()),('lambda_value',pa.float64()),('score',pa.float64()),
    ('appended_events',pa.int64()),('history_rows',pa.int64()),('query_gap',pa.int64()),
    ('inherited_rows',pa.int64()),('evicted_events',pa.int64())])


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class ReadonlyStrengthTrajectory:
    @torch.no_grad()
    def __init__(self,parent,current,linear,events,cfg,check_independent=False):
        self.model=current;self.device=next(current.parameters()).device
        self.cap=int(cfg['history_length']);self.band=int(cfg['append_band_size'])
        self.time=int(cfg['release_day'])*86400;self.end=int(cfg['end_day'])*86400
        source=NativeTrajectory(parent,int(cfg['edge'])-1,events,self.time,max_length=self.cap,append_band_size=self.band)
        self.events=source.events;self.cursor=self.origin=source.cursor
        self.original=source.cache;self.cache=source.cache;self.initial_length=self.cache.seq_len
        before=(self.original.k._version,self.original.v._version)
        self.mapped=linear(self.original)
        if before!=(self.original.k._version,self.original.v._version):raise ValueError('Frozen map mutated source')
        if (self.mapped.k.shape!=self.original.k.shape or self.mapped.v.shape!=self.original.v.shape
                or self.mapped.seq_len!=self.initial_length):raise ValueError('Map geometry differs')
        self.dk=self.mapped.k-self.original.k;self.dv=self.mapped.v-self.original.v
        self.appended=self.evicted=0;self.check=check_independent
        self.reference=clone_native_trajectory(source) if self.check else None
        if self.reference is not None:self.reference.switch(current,int(cfg['edge']),self.time)
        stop=int(np.searchsorted(self.events[0],self.end,side='left'))
        times=self.events[0][self.origin:stop];gaps=np.empty(len(times),dtype=np.float32)
        if len(times):
            gaps[0]=0 if source.last_timestamp is None else int(times[0])-source.last_timestamp
            gaps[1:]=np.diff(times);np.clip(gaps,0,MAX_DELTA_SECONDS,out=gaps)
        self.items=torch.as_tensor(np.asarray(self.events[1][self.origin:stop],dtype=np.int64),device=self.device)[None]
        self.behaviors=torch.as_tensor(np.asarray(self.events[2][self.origin:stop],dtype=np.int64),device=self.device)[None]
        self.deltas=torch.as_tensor(gaps,device=self.device)[None]
        self.lambdas=torch.tensor([a for _,a in GRID],device=self.device,dtype=self.cache.k.dtype)[None,:,None,None]
        self.max_cache_abs_error=self.max_score_abs_error=0.;self.append_shapes=Counter();self.read_shapes=Counter()

    @property
    def inherited(self):return max(0,self.initial_length-self.evicted)

    @torch.no_grad()
    def advance(self,until):
        until=int(until)
        if not self.time<=until<self.end:raise ValueError('Backward/out-of-window request')
        stop=int(np.searchsorted(self.events[0],until,side='left'))
        while self.cursor<stop:
            width=1<<(min(self.band,stop-self.cursor).bit_length()-1)
            offset=self.cursor-self.origin;n=self.cache.seq_len
            with _short_prefix_backend(self.model,n,self.cap):
                self.cache=append_with_rolling_band(self.model,self.cache,self.items[:,offset:offset+width],
                    self.behaviors[:,offset:offset+width],self.deltas[:,offset:offset+width],self.cap)
            self.cursor+=width;self.appended+=width;self.evicted+=max(0,n+width-self.cap)
            self.append_shapes[n,width]+=1
        self.time=until
        if self.check:
            self.reference.advance(until)
            for field in ('k','v'):
                a,b=getattr(self.cache,field),getattr(self.reference.cache,field)
                self.max_cache_abs_error=max(self.max_cache_abs_error,float((a-b).abs().max()))
                torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
            if (self.cursor,self.appended,self.evicted)!=(self.reference.cursor,self.reference.appended_tokens,self.reference.evicted_tokens):
                raise ValueError('Native causal/FIFO counters differ')

    @torch.no_grad()
    def read_cache(self):
        """Ephemeral batch only; inherited endpoints are copied without arithmetic."""
        n=self.inherited
        if not n:return self.cache
        start=self.initial_length-n
        k=self.cache.k.expand(-1,len(GRID),-1,-1).clone()
        v=self.cache.v.expand(-1,len(GRID),-1,-1).clone()
        k[:,:,:n]=self.original.k[:,:,start:]+self.lambdas*self.dk[:,:,start:]
        v[:,:,:n]=self.original.v[:,:,start:]+self.lambdas*self.dv[:,:,start:]
        k[:,0:1,:n]=self.cache.k[:,:,:n];v[:,0:1,:n]=self.cache.v[:,:,:n]
        k[:,4:5,:n]=self.mapped.k[:,:,start:];v[:,4:5,:n]=self.mapped.v[:,:,start:]
        return HSTUKVCache(k,v,self.cache.seq_len)

    @torch.no_grad()
    def score(self,candidates,delta):
        versions=(self.cache.k._version,self.cache.v._version)
        read=self.read_cache();count=len(GRID) if self.inherited else 1
        result=native_score(self.model,read,candidates.expand(count,-1),delta.expand(count))[0]
        if count==1:result=result.expand(len(GRID),-1)  # Guaranteed flat after all inherited rows disappear.
        if not bool(torch.isfinite(result).all()):raise ValueError('Nonfinite real-feedback score')
        if self.check:
            expected=[]
            for index in range(len(GRID)):
                cache=(HSTUKVCache(read.k[:,index:index+1],read.v[:,index:index+1],read.seq_len)
                    if count>1 else self.cache)
                expected.append(native_score(self.model,cache,candidates,delta)[0])
            expected=torch.cat(expected,0)
            self.max_score_abs_error=max(self.max_score_abs_error,float((result-expected).abs().max()))
            torch.testing.assert_close(result,expected,atol=1e-6,rtol=1e-5)
        if versions!=(self.cache.k._version,self.cache.v._version):raise ValueError('Read-only probe wrote back')
        self.read_shapes[self.cache.seq_len,candidates.shape[1],count]+=1
        return result


@torch.no_grad()
def replay_user(parent,current,linear,events,requests,uid,cfg,check=False):
    state=ReadonlyStrengthTrajectory(parent,current,linear,events,cfg,check)
    times=sorted(int(t) for t,rows in requests.items() if rows);raw=[r for t in times for r in requests[t]]
    items=torch.tensor([[int(r['item_idx']) for r in raw]],device=state.device,dtype=torch.long)
    gaps=[0 if (i:=int(np.searchsorted(state.events[0],t,side='left')))==0 else t-int(state.events[0][i-1]) for t in times]
    deltas=torch.tensor(gaps,device=state.device,dtype=torch.float32)
    rows=[];offset=flat_events=0
    for ti,t in enumerate(times):
        state.advance(t);selected=requests[t]
        if state.inherited==0:flat_events+=len(selected)
        for start in range(0,len(selected),int(cfg.get('query_batch',16))):
            part=selected[start:start+int(cfg.get('query_batch',16))]
            values=state.score(items[:,offset+start:offset+start+len(part)],deltas[ti:ti+1]).cpu().tolist()
            for (name,strength),scores in zip(GRID,values,strict=True):
                for request,value in zip(part,scores,strict=True):
                    if int(request['label']) not in (0,1):raise ValueError('Observed binary labels required')
                    rows.append(dict(uid=int(uid),query_timestamp=t,item_idx=int(request['item_idx']),label=int(request['label']),
                        method=name,lambda_value=strength,score=float(value),appended_events=state.appended,
                        history_rows=state.cache.seq_len,query_gap=gaps[ti],inherited_rows=state.inherited,evicted_events=state.evicted))
        offset+=len(selected)
    stats=dict(uid=int(uid),initial_history_length=state.initial_length,feedback_events=len(raw),
        request_timestamps=len(times),appended_events=state.appended,evicted_events=state.evicted,
        flat_no_inherited_events=flat_events,max_cache_abs_error=state.max_cache_abs_error if check else None,
        max_score_abs_error=state.max_score_abs_error if check else None,source_prefills=1,frozen_map_materializations=1,
        persistent_native_write_paths=1,request_local_read_paths=len(GRID),
        append_shapes=[dict(before=n,width=w,calls=c) for (n,w),c in sorted(state.append_shapes.items())],
        read_shapes=[dict(history=n,queries=q,batch=b,calls=c) for (n,q,b),c in sorted(state.read_shapes.items())])
    return rows,stats


def run(args):
    if args.gpu!=1:raise ValueError('Physical GPU1 only')
    population=json.loads(args.population.read_text());fixed=population['users']
    if len(fixed)<512 or len(set(fixed))!=len(fixed):raise ValueError('Fixed unique>=512 resident panel required')
    source=ROOT/'experiments/ch3_motivation/inputs';roles=json.loads((source/'population.json').read_text())
    if set(fixed)&set(roles['fit']+roles['validation']+roles['development']+roles['confirmation']):raise ValueError('Reserved role overlap')
    folder=source/f'edge_{args.edge}';binding,histories,requests=load_edge(folder)
    if binding['population_sha256']!=sha256(source/'population.json'):raise ValueError('Input role binding changed')
    for key in ('history','requests'):
        if sha256(folder/binding[key]['path'])!=binding[key]['sha256']:raise ValueError('Frozen native input changed')
    if not set(fixed)<=set(histories.rows):raise ValueError('Panel missing from original native history pack')
    users=[u for u in fixed if requests.get(u)][:args.users] if args.diagnostic else fixed
    eligible=[u for u in users if requests.get(u)]
    if args.diagnostic and len(users)!=args.users:raise ValueError('Insufficient diagnostic users')
    if not args.diagnostic and len(eligible)<512:raise ValueError('At least512 feedback UIDs required')
    os.environ['CUDA_VISIBLE_DEVICES']='1';torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    device=torch.device('cuda:0');parent,current,models=load_models(args.edge,device)
    checkpoint=ROBUST/f'fitting/edge_{args.edge}/robust_maps.pt';metadata=json.loads((checkpoint.parent/'summary.json').read_text())
    identity=metadata['identity']
    if (sha256(checkpoint)!=metadata['checkpoint_sha256'] or identity['edge']!=args.edge or identity['models']!=models
            or identity['fit_uids']!=roles['fit'][:256] or identity['validation_users']!=roles['validation']):
        raise ValueError('Frozen linear256_rows/model/roles changed')
    linear=load_robust_maps(checkpoint,device)['linear256_rows'];cfg=dict(binding['settings'],history_length=1024)
    paths=[Path(__file__),*[Path(__file__).with_name(p) for p in ('fit_candidate_robust_maps.py','simple_kv_maps.py','common.py','kv_evolution.py')],
        ROOT/'src/hstu_kvcache/evaluation/trajectory.py',ROOT/'src/hstu_kvcache/models/state_transition.py',
        ROOT/'src/hstu_kvcache/models/hstu.py',ROOT/'src/hstu_kvcache/models/attention.py',ROOT/'src/hstu_kvcache/adaptation/reader.py']
    sources={str(p.resolve()):sha256(p) for p in paths}
    config=dict(format='readonly_candidate_strength_v1',edge=args.edge,users=users,fixed_resident_users=fixed,
        population=str(args.population.resolve()),population_sha256=sha256(args.population),models=models,
        input_binding_sha256=sha256(folder/'inputs.json'),maps=str(checkpoint.resolve()),maps_sha256=sha256(checkpoint),
        base_map_name='linear256_rows',method_names=[n for n,_ in GRID],method_order=[n for n,_ in GRID],
        method_strengths=dict(GRID),settings=cfg,sources=sources,physical_gpu=1,diagnostic=args.diagnostic,
        check_independent=args.check_independent,block_users=args.block_users,
        plan=str(args.plan.resolve()) if args.plan else None,plan_sha256=sha256(args.plan) if args.plan else None,
        write_scope='One shared causal Reuse trajectory; current-native appends; all raw events strictly before query; original band/FIFO/gap rules',
        correction_scope='One frozen cutover map delta on old inherited rows only; surviving old rows interpolated read-only; new native rows untouched',
        endpoints='lambda0 exact native cache; lambda1 exact mapped inherited rows plus shared native new rows; no writeback',
        flat_scope='After inherited rows vanish, score once and broadcast all lambdas; item loss ties remain unassigned',
        distinction='Read-only lambda1 is not the previous persistent mapped-trajectory F1',
        analysis_scope='Observed-label item-wise loss sensitivity only; no oracle-selected overall AUC or deployment claim')
    args.output.mkdir(parents=True,exist_ok=True);cp=args.output/'configuration.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Frozen run changed')
    write_json(cp,config);frozen=digest(config);artifacts=[];all_stats=[];scored=score_rows=0;started=time.perf_counter()
    for offset in range(0,len(users),args.block_users):
        selected=users[offset:offset+args.block_users];block=args.output/f'blocks/part_{offset:06d}.parquet';mp=block.with_suffix('.json')
        if mp.exists():
            record=json.loads(mp.read_text())
            if record['configuration_sha256']!=frozen or record['uids']!=selected or sha256(block)!=record['score_sha256']:raise ValueError('Committed block changed')
        else:
            rows=[];stats=[]
            for uid in selected:
                if not requests.get(uid):continue
                values,detail=replay_user(parent,current,linear,histories.rows[uid],requests[uid],uid,cfg,args.check_independent)
                rows.extend(values);stats.append(detail)
            if any(sha256(Path(p))!=h for p,h in sources.items()):raise ValueError('Numerical source changed')
            block.parent.mkdir(exist_ok=True);temp=block.with_suffix('.parquet.writing');pq.write_table(pa.Table.from_pylist(rows,schema=SCHEMA),temp);temp.replace(block)
            record=dict(configuration_sha256=frozen,uids=selected,score_sha256=sha256(block),stats=stats,scored_users=len(stats),score_rows=len(rows));write_json(mp,record)
        scored+=record['scored_users'];score_rows+=record['score_rows'];all_stats.extend(record['stats'])
        artifacts.append(dict(path=str(block.relative_to(args.output)),sha256=sha256(block),manifest=str(mp.relative_to(args.output)),manifest_sha256=sha256(mp)))
        progress=dict(status='running',edge=args.edge,completed_residents=offset+len(selected),residents=len(users),feedback_users=scored,seconds=time.perf_counter()-started)
        write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    if scored!=len(eligible):raise ValueError('Incomplete feedback population')
    summary=dict(status='complete',edge=args.edge,users=len(users),feedback_users=scored,score_rows=score_rows,
        method_names=config['method_names'],method_order=config['method_names'],artifacts=artifacts,physical_gpu=1,
        diagnostic=args.diagnostic,quality_use_permitted=not args.diagnostic and scored>=512,
        seconds=time.perf_counter()-started,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        flat_no_inherited_events=sum(s['flat_no_inherited_events'] for s in all_stats),
        independent_check_statistics=all_stats if args.diagnostic else None)
    write_json(args.output/'summary.json',summary);write_json(args.output/'progress.json',dict(progress,status='complete'));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--edge',type=int,choices=(2,3),required=True)
    p.add_argument('--population',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path)
    p.add_argument('--gpu',type=int,default=1);p.add_argument('--block-users',type=int,default=64)
    p.add_argument('--diagnostic',action='store_true');p.add_argument('--users',type=int,default=8);p.add_argument('--check-independent',action='store_true')
    args=p.parse_args()
    try:run(args);write_json(args.output/'exit.json',dict(exit_code=0))
    except BaseException as error:write_json(args.output/'exit.json',dict(exit_code=1,error=str(error)));raise
