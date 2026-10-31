"""Batched native replay after once-per-user static cache transformations.

Callbacks run in dict insertion order as transform(source_cache, item_ids).
F2 may reuse F1's callback-local result. There are exactly len(maps)+2 native
persistent members: Reuse, each requested map, and once-Rebuild. No oracle,
crossed K/V path, duplicate F computation, or unused cache member is added.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from experiments.ch3_motivation.code.kv_evolution import clone_native_trajectory
from experiments.ch3_motivation.code.real_quality_paths import (
    _pack, _check_numeric, CACHE_NUMERIC_ATOL, NUMERIC_ATOL, NUMERIC_RTOL,
)
from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.evaluation.trajectory import NativeTrajectory, MAX_DELTA_SECONDS, _short_prefix_backend
from hstu_kvcache.models.state_transition import append_with_rolling_band


class StaticPaths:
    @torch.no_grad()
    def __init__(self,parent,current,maps,events,cfg,check_independent=False):
        if parent.training or current.training:raise ValueError('Frozen eval-mode models required')
        if any(not isinstance(n,str) or n in ('reuse','rebuild') for n in maps):
            raise ValueError('Static map names must be strings distinct from native anchors')
        self.model=current;self.device=next(current.parameters()).device
        self.cap=int(cfg['history_length']);self.band=int(cfg['append_band_size'])
        self.time=int(cfg['release_day'])*86400;self.end_time=int(cfg['end_day'])*86400
        source=NativeTrajectory(parent,int(cfg['edge'])-1,events,self.time,
            max_length=self.cap,append_band_size=self.band)
        self.events=source.events;self.cursor=self.origin=source.cursor
        self.initial_length=source.cache.seq_len
        ids=torch.as_tensor(source.retained_events[1],dtype=torch.long,device=self.device)[None]
        caches=[source.cache]
        for name,transform in maps.items():
            versions=source.cache.k._version,source.cache.v._version,ids._version
            result=transform(source.cache,ids)
            if versions!=(source.cache.k._version,source.cache.v._version,ids._version):
                raise ValueError(f'{name}: static map mutated its source cache or item IDs')
            if (result.seq_len!=source.cache.seq_len or result.k.shape!=source.cache.k.shape
                    or result.v.shape!=source.cache.v.shape or result.k.dtype!=source.cache.k.dtype
                    or result.v.dtype!=source.cache.v.dtype or result.k.device!=self.device
                    or result.v.device!=self.device):
                raise ValueError(f'{name}: transformed cache geometry/dtype/device differs from source')
            caches.append(result)
        caches.append(source.fresh_cache(current))
        self.methods=['reuse',*maps,'rebuild']
        self.cache=_pack(caches)
        self.check=bool(check_independent);self.references=None
        self.max_cache_abs_error=self.max_score_abs_error=0.
        if self.check:
            self.references=[]
            for cache in caches:
                native=clone_native_trajectory(source)
                native.switch(current,int(cfg['edge']),source.time)
                native.cache=cache
                self.references.append(native)
        self.appended=self.evicted=0
        self.append_shapes=Counter();self.query_shapes=Counter()
        self.prefetch_end=int(np.searchsorted(self.events[0],self.end_time,side='left'))
        times=self.events[0][self.origin:self.prefetch_end]
        gaps=np.empty(len(times),dtype=np.float32)
        if len(times):
            gaps[0]=0 if source.last_timestamp is None else int(times[0])-source.last_timestamp
            gaps[1:]=np.diff(times);np.clip(gaps,0,MAX_DELTA_SECONDS,out=gaps)
        self.items=torch.as_tensor(np.asarray(self.events[1][self.origin:self.prefetch_end],dtype=np.int64),device=self.device)[None]
        self.behaviors=torch.as_tensor(np.asarray(self.events[2][self.origin:self.prefetch_end],dtype=np.int64),device=self.device)[None]
        self.deltas=torch.as_tensor(gaps,device=self.device)[None]

    @torch.no_grad()
    def advance(self,until):
        until=int(until)
        if not self.time<=until<self.end_time:raise ValueError('Request time is backward/outside release window')
        stop=int(np.searchsorted(self.events[0],until,side='left'))
        count=len(self.methods)
        while self.cursor<stop:
            width=1<<(min(self.band,stop-self.cursor).bit_length()-1)
            offset=self.cursor-self.origin;before=self.cache.seq_len
            inputs=(self.items[:,offset:offset+width],self.behaviors[:,offset:offset+width],self.deltas[:,offset:offset+width])
            with _short_prefix_backend(self.model,before,self.cap):
                self.cache=append_with_rolling_band(self.model,self.cache,*(x.expand(count,-1) for x in inputs),self.cap)
            self.cursor+=width;self.appended+=width;self.evicted+=max(0,before+width-self.cap)
            self.append_shapes[before,width]+=1
        self.time=until
        if self.check:
            for native in self.references:native.advance(until)
            expected=_pack([n.cache for n in self.references])
            self.max_cache_abs_error=max(self.max_cache_abs_error,float((self.cache.k-expected.k).abs().max()),
                                        float((self.cache.v-expected.v).abs().max()))
            _check_numeric(self.cache.k,expected.k,'static_key',self.methods,atol=CACHE_NUMERIC_ATOL)
            _check_numeric(self.cache.v,expected.v,'static_value',self.methods,atol=CACHE_NUMERIC_ATOL)
            if any(n.cursor!=self.cursor or n.appended_tokens!=self.appended or n.evicted_tokens!=self.evicted for n in self.references):
                raise ValueError('Batched and independent chronological/FIFO counters differ')

    @torch.no_grad()
    def score(self,candidates,delta):
        count=len(self.methods)
        scores=native_score(self.model,self.cache,candidates.expand(count,-1),delta.expand(count))[0]
        if not bool(torch.isfinite(scores).all()):raise ValueError('Nonfinite real feedback score')
        if self.check:
            expected=torch.cat([native_score(self.model,n.cache,candidates,delta)[0] for n in self.references],0)
            self.max_score_abs_error=max(self.max_score_abs_error,float((scores-expected).abs().max()))
            _check_numeric(scores,expected,'static_score',self.methods,atol=NUMERIC_ATOL)
        self.query_shapes[self.cache.seq_len,candidates.shape[1]]+=1
        return scores

    def ledger(self):
        return dict(source_native_prefills=1,current_teacher_prefills=1,
            once_static_transforms={n:1 for n in self.methods[1:-1]},persistent_methods=self.methods,
            append_shapes=[dict(before_length=n,width=w,calls=c,batch=len(self.methods))
                           for (n,w),c in sorted(self.append_shapes.items())],
            query_shapes=[dict(history_length=n,queries=q,calls=c,batch=len(self.methods))
                          for (n,q),c in sorted(self.query_shapes.items())],
            event_inputs=dict(rows=self.prefetch_end-self.origin,array_transfers=3 if self.device.type=='cuda' else 0,
                transient_bytes=sum(t.numel()*t.element_size() for t in (self.items,self.behaviors,self.deltas)),
                per_append_scalar_tensor_constructions=0),
            scope='Actual shared research calls; each requested static cache has its own subsequent native writes. No oracle or unused persistent members.')


@torch.no_grad()
def replay_user(parent,current,maps,events,requests,uid,cfg,check_independent=False):
    """Return (long real-feedback rows, per-UID stats, native execution ledger).

    Maps are evaluated exactly once each, only on the native cutover source.
    All event and request inputs are prefetched once, consumed causally. Legacy
    KV-error columns are nullable: this scorer computes real-label logits only.
    """
    state=StaticPaths(parent,current,maps,events,cfg,check_independent)
    times=sorted(int(t) for t,rows in requests.items() if rows)
    if any(not state.time<=t<state.end_time for t in times):raise ValueError('Request outside release window')
    raw=[r for t in times for r in requests[t]]
    if any(int(r['label']) not in (0,1) for r in raw):raise ValueError('Binary observed labels required')
    items=torch.tensor([[int(r['item_idx']) for r in raw]],dtype=torch.long,device=state.device)
    gaps=[0 if (n:=int(np.searchsorted(state.events[0],t,side='left')))==0 else t-int(state.events[0][n-1]) for t in times]
    deltas=torch.tensor(gaps,dtype=torch.float32,device=state.device)
    rows=[];origin=0;query_batch=int(cfg.get('query_batch',16))
    for ti,timestamp in enumerate(times):
        state.advance(timestamp);request=requests[timestamp]
        for start in range(0,len(request),query_batch):
            selected=request[start:start+query_batch]
            values=state.score(items[:,origin+start:origin+start+len(selected)],deltas[ti:ti+1]).cpu().tolist()
            for method,scores in zip(state.methods,values,strict=True):
                for r,value in zip(selected,scores,strict=True):
                    rows.append(dict(uid=int(uid),query_timestamp=timestamp,item_idx=int(r['item_idx']),
                        label=int(r['label']),method=method,score=float(value),appended_events=state.appended,
                        history_rows=state.cache.seq_len,query_gap=int(gaps[ti]),
                        kv_mse=None,k_mse=None,v_mse=None,initial_kv_mse=None))
        origin+=len(request)
    stats=dict(uid=int(uid),feedback_events=len(raw),request_timestamps=len(times),
        initial_history_length=state.initial_length,appended_events=state.appended,
        evicted_events=state.evicted,history_rows=state.cache.seq_len,method_names=state.methods,
        max_cache_abs_error=state.max_cache_abs_error if check_independent else None,
        max_score_abs_error=state.max_score_abs_error if check_independent else None,
        max_batch_abs_error=max(state.max_cache_abs_error,state.max_score_abs_error) if check_independent else None,
        numeric_tolerances=dict(cache_atol=CACHE_NUMERIC_ATOL,score_atol=NUMERIC_ATOL,rtol=NUMERIC_RTOL),
        prediction_input_prefetch=dict(candidate_rows=len(raw),query_gaps=len(times),
            array_transfers=2 if state.device.type=='cuda' else 0,
            transient_bytes=items.numel()*items.element_size()+deltas.numel()*deltas.element_size()),
        kv_error_metrics_computed=False)
    return rows,stats,state.ledger()
