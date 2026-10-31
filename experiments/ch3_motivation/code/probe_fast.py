"""Fixed-window CUDA graphs for the frozen Chapter 3 probe replay.

The reference in probe_replay remains the oracle. A GPU prefix mask selects
inherited rows without changing graph shape. h is nevertheless evaluated on
ALL padded rows, and that additional arithmetic is reported explicitly.
"""
from __future__ import annotations

from collections import Counter
import time

import numpy as np
import torch

from experiments.ch3_motivation.code.probe_replay import ProbeTrajectory
from hstu_kvcache.adaptation.reader import history_read, read_embedded, score as native_score
from hstu_kvcache.evaluation.trajectory import MAX_DELTA_SECONDS
from hstu_kvcache.models import HSTUKVCache


def pad_features(features, length):
    """Normalized inherited row features followed by dummy native-row inputs."""
    values=torch.stack(tuple(features))
    if values.shape[-2] > length:
        raise ValueError("row features exceed the fixed cache window")
    if values.shape[-2] == length:
        return values
    zeros=values.new_zeros(*values.shape[:-2],length-values.shape[-2],values.shape[-1])
    return torch.cat((values,zeros),dim=-2)


class PaddedProbeGraph:
    """One fixed model/probe/operation/query shape; outputs are borrowed buffers.

    Replaying overwrites the previous output. A sequential trajectory can keep
    the append output as its state: it is copied into graph input buffers before
    the next replay overwrites it. Candidate outputs never become cache state.
    CPU runs use the identical tensor arithmetic without CUDA capture.
    """
    def __init__(self, model, probe, *, operation, window=1024, queries=16, capture=True):
        if operation not in ("score","append") or (operation=="append" and queries!=1):
            raise ValueError("fixed graph operation must be score or one-behavior append")
        if window < 2:
            raise ValueError("fixed-window graph requires at least two rows")
        if any(block.attn.position_bias is not None for block in model.blocks):
            raise ValueError("Chapter 3 graphs require the actual no-position-bias model")
        self.model,self.probe,self.operation=model,probe,operation
        self.window,self.queries=window,queries
        self.length=window if operation=="score" else window-1
        self.graph=None
        self.outputs=None
        self.capture_seconds=0.
        self.warmup_forwards=self.capture_forwards=self.replays=0
        c=probe.cfg
        parameter=next(model.parameters())
        device,dtype=parameter.device,parameter.dtype
        self.k=torch.zeros(c.num_layers,1,self.length,c.num_heads*c.head_dim,device=device,dtype=dtype)
        self.v=torch.zeros_like(self.k)
        self.rows=torch.zeros(c.num_layers,1,c.num_heads,self.length,c.row_dim,device=device,dtype=dtype)
        self.count=torch.zeros((),device=device,dtype=torch.long)
        self.positions=torch.arange(self.length,device=device)
        self.ids=torch.zeros(1,queries,device=device,dtype=torch.long)
        self.behavior=torch.zeros_like(self.ids)
        self.delta=torch.zeros(1,queries,device=device,dtype=dtype)
        self.row_padding=self.rows.new_zeros(c.num_layers,1,c.num_heads,1,c.row_dim)
        self._active_mask=None
        if capture and device.type=="cuda":
            self.capture()

    def _read(self, layer, attention, q, k, v):
        if self.probe.stage=="f":
            return history_read(attention,q,k,v)
        kh,vh=self.probe._heads(k),self.probe._heads(v)
        gk,gv=self.probe.query_correction(layer,q).chunk(2,-1)
        shift=(q*gk).sum(-1,keepdim=True)
        hv=None
        if self.probe.h is not None:
            hk,hv=self.probe.interaction_correction(layer,q,self.rows[layer]).chunk(2,-1)
            shift=shift+(q[:,:,:,None]*hk).sum(-1)
        dots=q@kh.transpose(-2,-1)+shift*self._active_mask
        weights=attention._activate(dots*attention.scale)
        if attention.block_variant=="hstu_reference":
            weights=weights/attention.cfg.max_seq_len
        weights=attention.attn_dropout(weights)
        inherited_weights=weights*self._active_mask
        result=weights@vh+inherited_weights.sum(-1,keepdim=True)*gv
        if hv is not None:
            result=result+(inherited_weights[...,None]*hv).sum(-2)
        return result

    def _forward(self):
        self._active_mask=(self.positions<self.count).to(self.k.dtype)[None,None,None]
        cache=HSTUKVCache(self.k,self.v,self.length)
        if self.operation=="score":
            values,_=native_score(self.model,cache,self.ids,self.delta,history_read_fn=self._read)
            return (values,)
        result=read_embedded(self.model,cache,self.model.embed_inputs(self.ids,self.behavior,self.delta),
                             history_read_fn=self._read)
        return (torch.cat((self.k,result.new_kv.k),dim=2),
                torch.cat((self.v,result.new_kv.v),dim=2),
                torch.cat((self.rows,self.row_padding),dim=-2))

    @torch.no_grad()
    def capture(self):
        if self.k.device.type!="cuda" or self.graph is not None:
            return
        self.model.eval()
        self.probe.eval()
        started=time.perf_counter()
        current=torch.cuda.current_stream(self.k.device)
        warmup=torch.cuda.Stream(device=self.k.device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup):
            for _ in range(2):
                values=self._forward()
                self.warmup_forwards+=1
                del values
        current.wait_stream(warmup)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs=self._forward()
        self.capture_forwards=1
        torch.cuda.synchronize(self.k.device)
        self.capture_seconds=time.perf_counter()-started

    @torch.no_grad()
    def run(self, cache, features, inherited_count, candidates, delta, behavior=None):
        if cache.seq_len!=self.window or candidates.shape!=(1,self.queries):
            raise ValueError("fixed graph requires one full-window user and its declared query shape")
        if not 0<=inherited_count<=self.window:
            raise ValueError("inherited count must be inside the physical window")
        if self.operation=="append":
            self.k.copy_(cache.k[:,:,1:])
            self.v.copy_(cache.v[:,:,1:])
            self.rows.copy_(features[...,1:,:])
            self.count.fill_(max(0,inherited_count-1))
            if behavior is None:
                raise ValueError("behavior append requires the actual behavior IDs")
            self.behavior.copy_(behavior)
        else:
            self.k.copy_(cache.k)
            self.v.copy_(cache.v)
            self.rows.copy_(features)
            self.count.fill_(inherited_count)
        self.ids.copy_(candidates)
        self.delta.copy_(delta)
        if self.graph is None:
            self.outputs=self._forward()
        else:
            self.graph.replay()
        self.replays+=1
        return self.outputs

    def report(self):
        c=self.probe.cfg
        # No credit for masked native suffix rows: all physical h rows execute.
        f=self.probe.forward_flops(self.length,self.queries)
        cached_normalization=(2*c.row_dim*self.length*c.num_heads*c.num_layers
                              if self.probe.h is not None and c.normalize else 0)
        return dict(operation=self.operation,window=self.window,physical_read_rows=self.length,
            queries=self.queries,stage=self.probe.stage,cuda_graph=self.graph is not None,
            capture_seconds=self.capture_seconds,warmup_forward_calls=self.warmup_forwards,
            capture_forward_calls=self.capture_forwards,replays=self.replays,
            correction_read_flops_estimate_per_replay=f["read_extra_flops"]-cached_normalization,
            cached_row_normalization_omitted_from_read_flops=cached_normalization,
            prefix_mask_multiply_flops_per_replay=(2*c.num_heads*self.queries*self.length*c.num_layers
                if self.probe.stage!="f" else 0),
            actual_h_mlp_row_pairs_per_replay=(c.num_layers*c.num_heads*self.queries*self.length
                if self.probe.h is not None else 0),
            feature_input_copy_bytes_per_replay=self.rows.numel()*self.rows.element_size(),
            kv_input_copy_bytes_per_replay=2*self.k.numel()*self.k.element_size(),
            scope="All padded rows execute; borrowed graph outputs; shared setup charged separately")


class GraphProbePool:
    """One frozen model/probe pool; inherited count is never part of its key."""
    def __init__(self,model,probe,*,window=1024,capture=True):
        self.model,self.probe,self.window,self.capture_enabled=model,probe,window,capture
        self.graphs={}

    def get(self,operation,queries):
        key=(operation,int(queries))
        if key not in self.graphs:
            self.graphs[key]=PaddedProbeGraph(self.model,self.probe,operation=operation,
                window=self.window,queries=int(queries),capture=self.capture_enabled)
        return self.graphs[key]

    def report(self):
        return [graph.report() for _,graph in sorted(self.graphs.items())]


class FastProbeTrajectory(ProbeTrajectory):
    """Full-window graphs with the reference path for actual short histories."""
    def __init__(self,parent,current,probe,events,cfg,*,pool=None,initial=None):
        super().__init__(parent,current,probe,events,cfg,initial=initial)
        self.pool=pool or GraphProbePool(current,probe,window=self.native.max_length)
        if self.pool.model is not current or self.pool.probe is not probe:
            raise ValueError("graph pool belongs to another frozen model or probe")
        self.padded=None
        self.padded_shapes=Counter()

    def _ensure_padded(self):
        if self.padded is None:
            self.padded=pad_features(self.features,self.native.max_length)

    @torch.no_grad()
    def score(self,candidates,query_delta):
        self._prepare()
        if self.cache.seq_len!=self.native.max_length or self.source.seq_len==0:
            self.counts["eager_or_native_score_batches"]+=1
            return super().score(candidates,query_delta)
        self._ensure_padded()
        q=candidates.shape[1]
        graph=self.pool.get("score",q)
        values,=graph.run(self.cache,self.padded,self.source.seq_len,candidates,query_delta)
        self.read_shapes["query",self.cache.seq_len,self.source.seq_len,q]+=1
        self.padded_shapes["query",self.cache.seq_len,self.source.seq_len,q]+=1
        self.counts["candidate_reads"]+=candidates.numel()
        self.counts["candidate_batches"]+=1
        self.counts["fixed_graph_score_batches"]+=1
        return values

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
            item=torch.tensor([[int(self.native.events[1][position])]],device=self.device)
            behavior=torch.tensor([[int(self.native.events[2][position])]],device=self.device)
            gap=0 if self.native.last_timestamp is None else timestamp-self.native.last_timestamp
            delta=torch.tensor([[min(MAX_DELTA_SECONDS,max(0,gap))]],device=self.device,dtype=torch.float32)
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
        result=super().report()
        result["padded_read_shapes"]=[dict(kind=kind,physical_read_rows=n,inherited_rows=c,queries=q,calls=count)
            for (kind,n,c,q),count in sorted(self.padded_shapes.items())]
        result["graph_pool_scope"]="Pool setup/replay totals are shared across sequential users; report pool.report() once"
        return result


@torch.no_grad()
def replay_fast_probe_user(parent,current,probe,events,requests_by_timestamp,uid,cfg,*,pool,tag=None,
                           stop_after_last_query=True):
    requests_by_timestamp={timestamp:rows for timestamp,rows in requests_by_timestamp.items() if rows}
    if stop_after_last_query and not requests_by_timestamp:
        # This independent edge has no scored output for the user. Deployment
        # costs still come from the driver's COMPLETE CPU event ledger.
        return [],dict(counts=dict(quality_skipped_no_feedback_users=1),initial_shapes=[],prepare_shapes=[],
            read_shapes=[],append_shapes=[],padded_read_shapes=[],
            quality_shortcut=dict(stop_after_last_query=True,no_feedback_gpu_skipped=True),
            cost_scope="GPU quality work only; use full CPU ledger for deployment costs")
    trajectory=FastProbeTrajectory(parent,current,probe,events,cfg,pool=pool)
    scores=[]
    at,end=int(cfg["release_day"])*86400,int(cfg["end_day"])*86400
    for timestamp,requests in sorted(requests_by_timestamp.items()):
        if not at<=int(timestamp)<end:
            raise ValueError("request outside the frozen release evaluation window")
        trajectory.advance(timestamp)
        for start in range(0,len(requests),int(cfg.get("query_batch",16))):
            batch=requests[start:start+int(cfg.get("query_batch",16))]
            ids=torch.tensor([[int(row["item_idx"]) for row in batch]],device=trajectory.device)
            gap=0 if trajectory.native.last_timestamp is None else timestamp-trajectory.native.last_timestamp
            delta=torch.tensor([gap],device=trajectory.device,dtype=torch.float32)
            values=trajectory.score(ids,delta)[0].cpu().tolist()
            scores.extend(dict(row,uid=int(uid),method=tag or probe.stage,score=float(value))
                          for row,value in zip(batch,values,strict=True))
    if not stop_after_last_query:
        trajectory.advance(end)
    report=trajectory.report()
    report["quality_shortcut"]=dict(stop_after_last_query=bool(stop_after_last_query),
        no_feedback_gpu_skipped=False,last_executed_timestamp=trajectory.native.time,
        cost_scope="Full deployment CPU ledger includes every user and post-last-feedback event")
    return scores,report


@torch.no_grad()
def evaluate_fast_probe(parent,current,probe,histories,by_user,uids,cfg,device=None,*,tag=None,
                        on_user=None,pool=None,stop_after_last_query=True):
    """Same shard interface as evaluate_probe, with shared fixed-shape graphs."""
    pool=pool or GraphProbePool(current,probe,window=int(cfg.get("history_length",1024)))
    scores,counts=[],Counter()
    shapes={name:Counter() for name in ("initial_shapes","prepare_shapes","read_shapes",
                                        "append_shapes","padded_read_shapes")}
    for index,uid in enumerate(uids):
        values,report=replay_fast_probe_user(parent,current,probe,histories.rows[uid],by_user.get(uid,{}),
            uid,cfg,pool=pool,tag=tag,stop_after_last_query=stop_after_last_query)
        scores.extend(values)
        counts.update(report["counts"])
        for name,counter in shapes.items():
            for row in report[name]:
                key=tuple((field,value) for field,value in row.items() if field!="calls")
                counter[key]+=row["calls"]
        if on_user is not None:
            on_user(index+1,len(uids),uid,report)
    return scores,dict(counts=dict(counts),
        **{name:[dict(key,calls=calls) for key,calls in sorted(counter.items())]
           for name,counter in shapes.items()},
        graph_pool=pool.report(),
        quality_shortcut=dict(stop_after_last_query=bool(stop_after_last_query),
            skipped_no_feedback_users=counts.get("quality_skipped_no_feedback_users",0)),
        cost_scope="Use physical padded rows for every graph h call; graph setup charged once per shared pool")
