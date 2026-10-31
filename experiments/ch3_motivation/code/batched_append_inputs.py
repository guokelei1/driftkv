"""Batch upload scalar append inputs; preserve the original causal replay.

Each user uploads post-release item, behavior and clamped delta arrays once.
The unchanged sequential append loop consumes only its current event views.
No K/V, model arithmetic, candidate reads or persistent method state changes.
"""
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from experiments.ch3_motivation.code import probe_projected_fast as original
from hstu_kvcache.evaluation.trajectory import MAX_DELTA_SECONDS
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.common.artifacts import sha256


def binding():
    files=[Path(__file__),Path(original.__file__),Path(original.__file__).with_name("probe_fast.py")]
    return dict(format="batched_append_input_upload_v1",source_sha256={str(p):sha256(p) for p in files},
        scope="transient per-user input transfer only; original per-event causal compute and FLOPs")


class BatchedAppendInputsTrajectory(original.ProjectedFastProbeTrajectory):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.append_input_origin=self.native.cursor
        end=int(np.searchsorted(self.native.events[0],int(self.cfg["end_day"])*86400,side="left"))
        start=self.append_input_origin
        times=self.native.events[0][start:end]
        gaps=np.empty(len(times),dtype=np.float32)
        if len(times):
            gaps[0]=0 if self.native.last_timestamp is None else int(times[0])-self.native.last_timestamp
            gaps[1:]=np.diff(times)
            np.clip(gaps,0,MAX_DELTA_SECONDS,out=gaps)
        self.append_items=torch.as_tensor(np.asarray(self.native.events[1][start:end],dtype=np.int64),device=self.device)[None]
        self.append_behaviors=torch.as_tensor(np.asarray(self.native.events[2][start:end],dtype=np.int64),device=self.device)[None]
        self.append_deltas=torch.as_tensor(gaps,device=self.device)[None]
        self.append_input_rows=end-start

    @torch.no_grad()
    def advance(self,until):
        until=int(until)
        if until<self.native.time:
            raise ValueError("probe trajectory cannot move backward")
        stop=int(np.searchsorted(self.native.events[0],until,side="left"))
        if self.native.cursor<stop:
            self._prepare()
        while self.native.cursor<stop:
            if self.probe.stage=="f" or self.source.seq_len==0:
                self.padded=None
                return super().advance(until)
            position=self.native.cursor
            timestamp=int(self.native.events[0][position])
            index=position-self.append_input_origin
            item=self.append_items[:,index:index+1]
            behavior=self.append_behaviors[:,index:index+1]
            delta=self.append_deltas[:,index:index+1]
            before,inherited=self.cache.seq_len,self.source.seq_len
            if before==self.native.max_length:
                self._ensure_padded()
                graph=self.pool.get("append",1)
                k,v,self.padded=graph.run(self.cache,self.padded,inherited,item,delta,behavior)
                self.native.cache=HSTUKVCache(k,v,before)
                self._slice_source(1)
                self.padded_shapes["behavior",before-1,self.source.seq_len,1]+=1
                self.counts["fixed_graph_append_calls"]+=1
            else:
                # A short user is an ordinary required execution case. Keep
                # its exact shape until it reaches the fixed serving window.
                cache,source,ids,_=self.probe.append(self.model,self.cache,self.source,self.source_ids,
                    item,behavior,delta,max_length=self.native.max_length,read_features=self.features)
                dropped=inherited-source.seq_len
                if self.features is not None:
                    self.features=tuple(rows[:,:,dropped:] for rows in self.features)
                self.source,self.source_ids,self.native.cache=source,ids,cache
                self.padded=None
                self.counts["eager_corrected_append_calls"]+=1
            self.native.cursor+=1
            self.native.producers=np.r_[self.native.producers,self.native.version][-self.native.max_length:]
            self.native.ledger["append",self.native.version,before,1]+=1
            self.native.appended_tokens+=1
            self.native.evicted_tokens+=max(0,before+1-self.native.max_length)
            n=min(before,self.native.max_length-1)
            self.read_shapes["behavior",n,self.source.seq_len,1]+=1
            self.append_shapes[before,1,self.source.seq_len]+=1
            self.counts["corrected_append_calls"]+=1
            self.counts["appended_rows"]+=1
        self.native.time=until

    def report(self):
        value=super().report()
        value["append_input_execution"]=dict(input_rows_prefetched=self.append_input_rows,
            input_tensor_bytes=sum(t.numel()*t.element_size() for t in (self.append_items,self.append_behaviors,self.append_deltas)),
            host_to_device_array_transfers=3,per_event_scalar_tensor_constructions=0,
            scope="transient quality-run input arrays including unconsumed window tail; not method persistent state or controlled timing")
        return value


@contextmanager
def batched_append_inputs():
    previous=original.ProjectedFastProbeTrajectory
    if previous is not BatchedAppendInputsTrajectory.__bases__[0]:
        raise ValueError("nested or unknown projected trajectory substitution")
    original.ProjectedFastProbeTrajectory=BatchedAppendInputsTrajectory
    try:
        yield
    finally:
        original.ProjectedFastProbeTrajectory=previous
