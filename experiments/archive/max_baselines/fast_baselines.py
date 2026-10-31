"""Batch independent policies, share identical native writes, cache CUDA graphs.

This accelerates offline quality evaluation of one independent release edge.
All policy states remain separate. Q-only variants share Reuse's native cache
because their defined writers are identical. An optional tail shortcut stops
GPU work after the last scored query; the complete deployment event ledger is
still charged, including idle users and the unobserved suffix.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]

from experiments.archive.max_baselines.baselines import (
    DAY, prepare_policy, raw_tensors, query_behavior_scores,
)
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.methods.baselines.layer_recompute.core import _capture_boundaries
from hstu_kvcache.methods.baselines import layer_recompute as lr
from hstu_kvcache.methods.baselines.deviation_recompute import layer0_deviation_scores
from hstu_kvcache.methods.baselines.query_recompute import query_attention_scores
from hstu_kvcache.evaluation.trajectory import NativeTrajectory, _short_prefix_backend
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.models.state_transition import append_with_rolling_band
from hstu_kvcache.methods.cachetide.kernels.reader import launch_read
from hstu_kvcache.common.artifacts import sha256, write_json


def native_read(index,attention,q,k,v):
    del index
    return launch_read(q,k,v,k[:,:0,:1],q,q,q,None,(0,0,0,0),
        code_count=0,scale=float(attention.scale))


class PackedQuery:
    """Same cross-head feature maps, one batched matmul for the three policies."""
    def __init__(self,policies):
        self.policies=policies
        self.layers=[]
        for i in range(len(policies[0]["modules"])):
            modules=[p["modules"][i] for p in policies]
            if any(m.feature_mode!="cross_phi" for m in modules):
                raise ValueError("this evaluation uses the selected cross_phi baseline")
            self.layers.append(tuple(torch.stack([getattr(m,name) for m in modules]).detach()
                for name in ("input_mean","input_scale","weight","bias")))

    def reader(self,n):
        def read(index,attention,q,k,v):
            native=native_read(index,attention,q,k,v)
            mean,scale,weight,bias=self.layers[index]
            flat=q.transpose(1,2).flatten(2)
            features=torch.cat((flat,torch.nn.functional.elu(flat)),-1)
            output=torch.bmm((features-mean[:,None])/scale[:,None],weight)+bias[:,None]
            rate=output.reshape(q.shape[0],q.shape[2],q.shape[1],q.shape[3]).transpose(1,2)
            return native+rate*n
        return read


class PackedHistory:
    """Batch the three independent nonlinear maps across all layers."""
    def __init__(self,policies):
        self.policies=policies
        self.layers=len(policies[0]["modules"])
        modules=[p["modules"][layer] for layer in range(self.layers) for p in policies]
        if any(m.query_correction is not None for m in modules):
            raise ValueError("persistent mapping uses the selected no-query-affine control")
        self.mean=torch.stack([m.input_mean for m in modules])[:,None]
        self.scale=torch.stack([m.input_scale for m in modules])[:,None]
        self.weight=torch.stack([m.map_weight for m in modules])
        self.bias=torch.stack([m.map_bias for m in modules])[:,None]
        self.encoder=torch.stack([m.encoder.weight.T for m in modules])
        self.encoder_bias=torch.stack([m.encoder.bias for m in modules])[:,None]
        self.output=torch.stack([m.output.weight.T for m in modules])
        self.output_bias=torch.stack([m.output.bias for m in modules])[:,None]
        self.units=torch.stack([m.residual_scale for m in modules])[:,None]

    def apply(self,cache):
        p=len(self.policies)
        joined=torch.cat((cache.k,cache.v),-1).expand(-1,p,-1,-1)
        shape=joined.shape
        flat=joined.reshape(self.layers*p,cache.seq_len,-1)
        normalized=(flat-self.mean)/self.scale
        affine=torch.bmm(normalized,self.weight)+self.bias
        hidden=torch.nn.functional.silu(torch.bmm(normalized,self.encoder)+self.encoder_bias)
        nonlinear=(torch.bmm(hidden,self.output)+self.output_bias)*self.units
        mapped=(flat+(affine+nonlinear)*torch.ones_like(flat[...,:1])).reshape(shape)
        k,v=mapped.split(cache.k.shape[-1],-1)
        return {policy["tag"]:HSTUKVCache(k[:,i:i+1],v[:,i:i+1],cache.seq_len)
                for i,policy in enumerate(self.policies)}


class NativeGraphs:
    """Only fixed full-window shapes; inference buffers never become user state."""
    def __init__(self,enabled=True):
        self.enabled=enabled
        self.entries={}
        self.inputs={}
        self.counts=Counter()
        self.prepare_seconds=0.

    def run(self,kind,model,cache,ids,delta,*,behavior=None,reader=None,reader_key="native"):
        def compute(cache,ids,delta,behavior):
            if kind=="append":
                return append_with_rolling_band(model,cache,ids,behavior,delta,1024)
            return score(model,cache,ids,delta,history_read_fn=reader or native_read)[0]
        supported=(self.enabled and cache.seq_len==1024
            and ((kind=="append" and ids.shape[1] in (1,2,4,8,16,32,64,128,256))
                 or (kind=="score" and ids.shape[1]<=64)))
        if not supported:
            self.counts["eager_"+kind]+=1
            with _short_prefix_backend(model,cache.seq_len,1024):
                return compute(cache,ids,delta,behavior)
        key=(kind,id(model),reader_key,ids.shape[0],ids.shape[1])
        entry=self.entries.get(key)
        if entry is None:
            started=time.perf_counter()
            bank_key=(cache.k.shape,cache.k.dtype)
            if bank_key not in self.inputs:
                self.inputs[bank_key]=HSTUKVCache(torch.empty_like(cache.k),torch.empty_like(cache.v),1024)
            staged=self.inputs[bank_key]
            staged.k.copy_(cache.k);staged.v.copy_(cache.v)
            x,dt=ids.clone(),delta.clone()
            bh=None if behavior is None else behavior.clone()
            stream=torch.cuda.Stream(device=ids.device)
            stream.wait_stream(torch.cuda.current_stream(ids.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    compute(staged,x,dt,bh)
            torch.cuda.current_stream(ids.device).wait_stream(stream)
            torch.cuda.synchronize(ids.device)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                output=compute(staged,x,dt,bh)
            torch.cuda.synchronize(ids.device)
            # Graphs retain raw pointers, not Python closures. Keep packed
            # weights alive across later sample blocks and graph replays.
            entry=dict(graph=graph,cache=staged,ids=x,delta=dt,behavior=bh,output=output,reader=reader)
            self.entries[key]=entry
            self.prepare_seconds+=time.perf_counter()-started
            self.counts["captures"]+=1
        else:
            entry["cache"].k.copy_(cache.k);entry["cache"].v.copy_(cache.v)
            entry["ids"].copy_(ids);entry["delta"].copy_(delta)
            if behavior is not None:entry["behavior"].copy_(behavior)
        entry["graph"].replay()
        self.counts["graph_"+kind]+=1
        return entry["output"]

    def describe(self):
        return dict(counts=dict(self.counts),entries=len(self.entries),prepare_seconds=self.prepare_seconds,
            input_bytes=sum((c.k.numel()+c.v.numel())*c.k.element_size() for c in self.inputs.values()))


@torch.no_grad()
def evaluate_batched(parent,current,histories,by_user,uids,policies,cfg,device,*,graphs=None,
                     stop_after_last_query=False,progress=None,shared_preparation=True):
    graphs=graphs or NativeGraphs(enabled=False)
    native=[p for p in policies if p["method"]!="query_only"]
    queries=[p for p in policies if p["method"]=="query_only"]
    packed=PackedQuery(queries) if queries else None
    history_policies=[p for p in native if p["method"]=="history_conditioned"]
    history=PackedHistory(history_policies) if shared_preparation and history_policies else None
    reuse_index=next(i for i,p in enumerate(native) if p["method"]=="reuse")
    scores=[]
    ledger=Counter()
    at,end=cfg["release_day"]*DAY,cfg["end_day"]*DAY
    count=len(native)
    stats=Counter()
    per_user=[]
    started=time.perf_counter()
    for user_index,uid in enumerate(uids):
        user_started=time.perf_counter()
        previous_bands=stats["native_append_launches"]
        rounds=by_user.get(uid,{})
        if stop_after_last_query and not rounds:
            stats["no_quality_request_users_skipped"]+=1
            per_user.append(dict(uid=uid,seconds=time.perf_counter()-user_started,requests=0,append_bands=0))
            continue
        events=histories.rows[uid]
        with _capture_boundaries(parent) as boundaries:
            base=NativeTrajectory(parent,cfg["edge"]-1,events,at,
                max_length=cfg["history_length"],append_band_size=cfg["append_band_size"])
        cursor=base.cursor
        last=base.last_timestamp
        prepared=False
        cache=None

        def prepare(consumer):
            nonlocal prepared,cache
            if prepared:return
            prepared=True
            values=[]
            shared={}
            selectors={}
            n=base.cache.seq_len
            if shared_preparation and n:
                raw=raw_tensors(base.retained_events,device)
                if history is not None:shared.update(history.apply(base.cache))
                layer_policies=[p for p in native if p["method"]=="layer"]
                if layer_policies and all(p["interval"][0]==0 for p in layer_policies):
                    widest=max(p["interval"][1] for p in layer_policies)
                    with _short_prefix_backend(current,n,cfg["history_length"]):
                        prefix=lr.recompute_interval(current,lr.LayerRecomputeState(base.cache,boundaries),*raw,(0,widest)).cache
                    for p in layer_policies:
                        mask=(torch.arange(len(current.blocks),device=device)<=p["interval"][1])[:,None,None,None]
                        shared[p["tag"]]=HSTUKVCache(torch.where(mask,prefix.k,base.cache.k),
                            torch.where(mask,prefix.v,base.cache.v),n)
                if any(p["method"]=="deviation" for p in native):
                    selectors["deviation"]=layer0_deviation_scores(current,base.cache,*raw)
                if any(p["method"]=="query" for p in native):
                    selectors["query"]=(query_attention_scores(current,base.cache,consumer["items"],consumer["delta"])
                        if consumer["kind"]=="query" else query_behavior_scores(current,base.cache,
                            consumer["items"],consumer["behavior"],consumer["delta"]))
            # Every repair starts from the identical immutable old native state.
            for policy in native:
                if policy["tag"] in shared:
                    values.append(shared[policy["tag"]])
                    continue
                trajectory=object.__new__(NativeTrajectory)
                trajectory.__dict__.update(base.__dict__)
                trajectory.model=current
                prepare_policy(trajectory,boundaries,policy,current,consumer,cfg,device,
                    selection_scores=selectors.get(policy["method"]))
                values.append(trajectory.cache)
            cache=HSTUKVCache(torch.cat([c.k for c in values],1),torch.cat([c.v for c in values],1),base.cache.seq_len)
            stats["prepared_users"]+=1

        def advance(until):
            nonlocal cache,cursor,last
            stop=int(np.searchsorted(events[0],until,side="left"))
            if cursor<stop and not prepared:
                prepare(dict(kind="behavior",items=torch.tensor([[int(events[1][cursor])]],device=device),
                    behavior=torch.tensor([[int(events[2][cursor])]],device=device),
                    delta=torch.tensor([[min(7*DAY,int(events[0][cursor])-last)]],device=device,dtype=torch.float32)))
            while cursor<stop:
                width=1<<(min(cfg["append_band_size"],stop-cursor).bit_length()-1)
                stop_band=cursor+width
                t=events[0][cursor:stop_band]
                dt=np.empty(width,dtype=np.float32)
                dt[0]=int(t[0])-last
                dt[1:]=np.diff(t)
                np.clip(dt,0,7*DAY,out=dt)
                ids=torch.as_tensor(events[1][cursor:stop_band],device=device,dtype=torch.long)[None].expand(count,-1)
                bh=torch.as_tensor(events[2][cursor:stop_band],device=device,dtype=torch.long)[None].expand(count,-1)
                delta=torch.as_tensor(dt,device=device)[None].expand(count,-1)
                before=cache.seq_len
                cache=graphs.run("append",current,cache,ids,delta,behavior=bh)
                ledger[before,width,before==cfg["history_length"]]+=count
                stats["native_append_launches"]+=1
                cursor=stop_band
                last=int(t[-1])
        for timestamp,requests in sorted(rounds.items()):
            advance(timestamp)
            selection=requests[:cfg.get("repair_query_batch",16)]
            prepare(dict(kind="query",items=torch.tensor([[r["item_idx"] for r in selection]],device=device),
                delta=torch.tensor([timestamp-last],device=device,dtype=torch.float32)))
            for pos in range(0,len(requests),cfg["query_batch"]):
                batch=requests[pos:pos+cfg["query_batch"]]
                ids=torch.tensor([[r["item_idx"] for r in batch]],device=device)
                delta=torch.tensor([timestamp-last],device=device,dtype=torch.float32)
                values=graphs.run("score",current,cache,ids.expand(count,-1),delta.expand(count))
                results=values.cpu().tolist()
                for policy,row_scores in zip(native,results,strict=True):
                    for row,value in zip(batch,row_scores,strict=True):
                        scores.append(dict(row,method=policy["tag"],score=value))
                if packed is not None:
                    q_count=len(queries)
                    shared=HSTUKVCache(cache.k[:,reuse_index:reuse_index+1].expand(-1,q_count,-1,-1),
                        cache.v[:,reuse_index:reuse_index+1].expand(-1,q_count,-1,-1),cache.seq_len)
                    q_values=graphs.run("score",current,shared,ids.expand(q_count,-1),delta.expand(q_count),
                        reader=packed.reader(cache.seq_len),reader_key="packed_query")
                    for policy,row_scores in zip(queries,q_values.cpu().tolist(),strict=True):
                        for row,value in zip(batch,row_scores,strict=True):
                            scores.append(dict(row,method=policy["tag"],score=value))
                stats["candidate_batches"]+=1
                stats["feedback_events"]+=len(batch)
        if not stop_after_last_query:
            advance(end)
        per_user.append(dict(uid=uid,seconds=time.perf_counter()-user_started,
            requests=sum(len(r) for r in rounds.values()),append_bands=stats["native_append_launches"]-previous_bands,
            source_rows=base.cache.seq_len))
        if progress is not None and (user_index+1)%8==0:
            progress(dict(users=user_index+1,total=len(uids),elapsed_seconds=time.perf_counter()-started,
                scores=len(scores)))
    torch.cuda.synchronize(device)
    return scores,dict(stats=dict(stats),per_user=per_user,seconds=time.perf_counter()-started,graphs=graphs.describe(),
        optimized_append_shapes=[dict(shape=list(key),count=value) for key,value in ledger.items()],
        stop_after_last_query=stop_after_last_query,
        cost_scope="Complete CPU deployment ledger still includes the unobserved tail and users without feedback; this shortcut accelerates offline quality computation only")


def evaluate_checkpointed(parent,current,histories,by_user,uids,policies,cfg,device,output,*,identity,progress=None,
                          assigned_offsets=None,retain_scores=True):
    """Commit independent user blocks; only a complete parquet+marker is resumed."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    folder=output/"quality_shards"
    folder.mkdir(exist_ok=True)
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    engine=NativeGraphs()
    size=cfg.get("evaluation_checkpoint_users",4096)
    all_scores=[]
    records=[]
    totals=Counter()
    started=time.perf_counter()
    for offset in range(0,len(uids),size):
        if assigned_offsets is not None and offset not in assigned_offsets:continue
        block=uids[offset:offset+size]
        block_sha=hashlib.sha256(json.dumps(block).encode()).hexdigest()
        shard=folder/f"part_{offset:06d}.parquet"
        marker=folder/f"part_{offset:06d}.json"
        if marker.exists():
            record=json.loads(marker.read_text())
            assert record["fingerprint"]==fingerprint and record["uids_sha256"]==block_sha,"checkpoint settings/users differ"
            assert record["scores_sha256"]==sha256(shard),"checkpoint scores differ"
            rows=pq.read_table(shard).to_pylist() if retain_scores else []
            resumed=True
        else:
            rows,stats=evaluate_batched(parent,current,histories,by_user,block,policies,cfg,device,
                graphs=engine,stop_after_last_query=cfg["stop_after_last_query"])
            # An empty-feedback block still commits an empty file and its UIDs.
            schema=pa.schema([("uid",pa.uint64()),("query_timestamp",pa.int64()),("raw_item_id",pa.uint64()),
                ("item_idx",pa.int64()),("label",pa.int64()),("method",pa.string()),("score",pa.float64())])
            staged=shard.with_suffix(".parquet.tmp")
            pq.write_table(pa.Table.from_pylist(rows,schema=schema),staged)
            with staged.open("rb") as handle:os.fsync(handle.fileno())
            staged.replace(shard)
            record=dict(fingerprint=fingerprint,uids_sha256=block_sha,offset=offset,users=len(block),
                scores=len(rows),scores_sha256=sha256(shard),
                execution=dict(stats=stats["stats"],seconds=stats["seconds"]),completed_users=offset+len(block))
            staged_marker=marker.with_suffix(".json.tmp")
            write_json(staged_marker,record)
            with staged_marker.open("rb") as handle:os.fsync(handle.fileno())
            staged_marker.replace(marker)
            resumed=False
        if retain_scores:all_scores.extend(rows)
        totals.update(record["execution"]["stats"])
        records.append(dict(offset=offset,users=len(block),resumed=resumed,
            seconds=record["execution"]["seconds"],scores=record["scores"]))
        if progress is not None:
            local_total=(len(uids) if assigned_offsets is None else
                sum(min(size,len(uids)-o) for o in assigned_offsets))
            progress(dict(users=sum(r["users"] for r in records),total=local_total,
                scores=sum(r["scores"] for r in records),global_offset_end=offset+len(block),global_users=len(uids),
                elapsed_seconds=time.perf_counter()-started,checkpoint=str(marker),resumed=resumed))
    return all_scores,dict(stats=dict(totals),blocks=records,seconds_this_invocation=time.perf_counter()-started,
        checkpoint_users=size,fingerprint=fingerprint,graphs=engine.describe(),
        cost_scope="all deployment events charged independently; shards retain scored quality only")
