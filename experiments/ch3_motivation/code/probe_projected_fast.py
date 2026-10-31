"""Equivalent frozen-h execution with its fixed first-layer row map cached.

h remains the SAME ordinary nonlinear MLP with the SAME weights and width.
Only its first affine map is split as q@Wq + row@Wr + bias. This is an
inference execution rewrite, with no rank restriction or parameter fitting.
The projected row table can exceed native K/V memory and is reported as such.
"""
from __future__ import annotations

from collections import Counter

import torch
from torch.nn import functional as F

from experiments.ch3_motivation.code.probe_fast import (
    PaddedProbeGraph, GraphProbePool, FastProbeTrajectory, pad_features,
)
from experiments.ch3_motivation.code.probe_cost import padded_read_extra_flops
from hstu_kvcache.adaptation.reader import history_read


@torch.no_grad()
def project_row_features(probe,features):
    """Cache row@Wr WITHOUT bias from fixed normalized [B,H,N,row_dim] inputs.

    Return one [B,H,N,hidden_width] tensor/layer. f/g-only graph inputs have
    width zero, since neither reader uses row features after f preparation.
    Call once per active inherited row set and slice/shift on FIFO eviction.
    """
    if probe.h is None:
        return tuple(rows[...,:0] for rows in features)
    d=probe.cfg.head_dim
    return tuple(torch.einsum("bhni,hiw->bhnw",rows,probe.h[layer].weights[0][:,d:])
                 for layer,rows in enumerate(features))


def projected_interaction_correction(probe,layer,q,projected_rows):
    """Same h(q,row), [B,H,Q,N,2D]; preserve query norm, biases and ALL layers."""
    if probe.h is None:
        raise ValueError("projected interaction requires the original h MLP")
    c=probe.cfg
    network=probe.h[layer]
    normalized=q/probe.query_scale[layer][None,:,None] if c.normalize else q
    query=torch.einsum("bhqi,hiw->bhqw",normalized,network.weights[0][:,:c.head_dim])
    value=query[:,:,:,None]+projected_rows[:,:,None]
    # The cached row projection owns NO bias. Add the original first bias
    # once, after combining q/row contributions and before the first SiLU.
    value=F.silu(value+network.biases[0][None,:,None,None])
    for index in range(1,len(network.weights)):
        value=torch.einsum("bh...i,hio->bh...o",value,network.weights[index])
        value=value+network.biases[index][None,:,None,None]
        if index+1!=len(network.weights):
            value=F.silu(value)
    return value*probe._output_scale(layer,value.ndim)


def projected_row_preparation_flops(probe,rows,*,batch=1):
    c=probe.cfg
    return (int(2*batch*int(rows)*c.num_layers*c.num_heads*c.row_dim*c.hidden_width)
            if probe.h is not None else 0)


def projected_read_extra_flops(probe,rows,queries):
    """Single physical block, including both masks and every padded h pair."""
    if probe.stage=="f":
        return 0
    c,n,q=probe.cfg,int(rows),int(queries)
    lh,d,w=c.num_layers*c.num_heads,c.head_dim,c.hidden_width
    norm=d if c.normalize else 0
    widths=[d]+[w]*c.depth+[2*d]
    g_mlp=sum(2*i*o+o for i,o in zip(widths[:-1],widths[1:],strict=True))+2*c.depth*w
    g=q*lh*(norm+g_mlp+6*d+4*n)
    h=0
    if probe.h is not None:
        remaining=[w]*c.depth+[2*d]
        rest=sum(2*i*o+o for i,o in zip(remaining[:-1],remaining[1:],strict=True))+2*(c.depth-1)*w
        # Query projection once/query. Each pair still combines the two
        # hidden vectors, adds the first bias, runs SiLU and the entire tail.
        h=q*lh*(norm+2*d*w+n*(4*w+rest+6*d+1)+d)
    return int(g+h)


def _tensor_bytes(value):
    return value.numel()*value.element_size()


def _storage_bytes(values):
    stores={}
    for value in values:
        storage=value.untyped_storage()
        stores[str(value.device),storage.data_ptr()]=storage.nbytes()
    return sum(stores.values())


class ProjectedPaddedGraph(PaddedProbeGraph):
    """Same fixed-window reader/writer graph with projected row input buffers."""
    def __init__(self,model,probe,*,operation,window=1024,queries=16,capture=True):
        # Allocate projected buffers directly, avoiding a transient full raw
        # row table merely to replace the base class's input allocation.
        if operation not in ("score","append") or (operation=="append" and queries!=1):
            raise ValueError("fixed graph operation must be score or one-behavior append")
        if window<2:
            raise ValueError("fixed-window graph requires at least two rows")
        if any(block.attn.position_bias is not None for block in model.blocks):
            raise ValueError("Chapter 3 graphs require the actual no-position-bias model")
        self.model,self.probe,self.operation=model,probe,operation
        self.window,self.queries=window,queries
        self.length=window if operation=="score" else window-1
        self.graph=self.outputs=None
        self.capture_seconds=0.
        self.warmup_forwards=self.capture_forwards=self.replays=0
        c=probe.cfg
        parameter=next(model.parameters())
        device,dtype=parameter.device,parameter.dtype
        self.feature_width=c.hidden_width if probe.h is not None else 0
        self.k=torch.zeros(c.num_layers,1,self.length,c.num_heads*c.head_dim,device=device,dtype=dtype)
        self.v=torch.zeros_like(self.k)
        self.rows=torch.zeros(c.num_layers,1,c.num_heads,self.length,self.feature_width,device=device,dtype=dtype)
        self.count=torch.zeros((),device=device,dtype=torch.long)
        self.positions=torch.arange(self.length,device=device)
        self.ids=torch.zeros(1,queries,device=device,dtype=torch.long)
        self.behavior=torch.zeros_like(self.ids)
        self.delta=torch.zeros(1,queries,device=device,dtype=dtype)
        self.row_padding=self.rows.new_zeros(c.num_layers,1,c.num_heads,1,self.feature_width)
        self._active_mask=None
        if capture and device.type=="cuda":
            self.capture()

    def _read(self,layer,attention,q,k,v):
        if self.probe.stage=="f":
            return history_read(attention,q,k,v)
        kh,vh=self.probe._heads(k),self.probe._heads(v)
        gk,gv=self.probe.query_correction(layer,q).chunk(2,-1)
        shift=(q*gk).sum(-1,keepdim=True)
        hv=None
        if self.probe.h is not None:
            hk,hv=projected_interaction_correction(self.probe,layer,q,self.rows[layer]).chunk(2,-1)
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

    def report(self):
        value=super().report()
        c=self.probe.cfg
        width=self.feature_width
        tensors=[self.k,self.v,self.rows,self.count,self.positions,self.ids,self.behavior,self.delta,self.row_padding]
        if self.outputs is not None:
            tensors.extend(self.outputs)
        value.update(execution="fixed first-layer row projection; identical h weights, width and nonlinear tail",
            projected_feature_width=width,raw_row_feature_width=c.row_dim,
            raw_padded_read_extra_flops_per_replay=padded_read_extra_flops(self.probe,self.length,self.queries),
            projected_read_extra_flops_per_replay=projected_read_extra_flops(self.probe,self.length,self.queries),
            correction_read_flops_estimate_per_replay=projected_read_extra_flops(self.probe,self.length,self.queries),
            cached_row_normalization_omitted_from_read_flops=0,
            arithmetic_scope="projected read extra includes prefix masks; prefix_mask_multiply field is a component, not another add",
            visible_input_output_buffer_storage_bytes=_storage_bytes(tensors),
            row_projection_input_bytes_per_replay=_tensor_bytes(self.rows),
            one_layer_first_pair_hidden_tensor_bytes=c.num_heads*self.queries*self.length*width*self.k.element_size(),
            one_layer_query_projection_tensor_bytes=c.num_heads*self.queries*width*self.k.element_size(),
            one_layer_raw_concat_eliminated_bytes=(c.num_heads*self.queries*self.length*(c.head_dim+c.row_dim)*self.k.element_size()
                if self.probe.h is not None else 0),
            memory_scope="visible shared graph input/output buffers only; CUDA graph intermediate pool/peak memory measured separately; per-layer tensor fields are shapes, not an additive peak")
        return value


class ProjectedGraphPool(GraphProbePool):
    def get(self,operation,queries):
        key=operation,int(queries)
        if key not in self.graphs:
            self.graphs[key]=ProjectedPaddedGraph(self.model,self.probe,operation=operation,
                window=self.window,queries=int(queries),capture=self.capture_enabled)
        return self.graphs[key]


class ProjectedFastProbeTrajectory(FastProbeTrajectory):
    """Original causal FIFO lifecycle, with immutable h row projections cached."""
    def __init__(self,parent,current,probe,events,cfg,*,pool=None,initial=None):
        pool=pool or ProjectedGraphPool(current,probe,window=int(cfg.get("history_length",1024)))
        if not isinstance(pool,ProjectedGraphPool):
            raise ValueError("projected trajectory requires a projected graph pool")
        super().__init__(parent,current,probe,events,cfg,pool=pool,initial=initial)
        self.projected_preparation_flops=0

    def _ensure_padded(self):
        if self.padded is not None:
            return
        rows=project_row_features(self.probe,self.features)
        self.padded=pad_features(rows,self.native.max_length)
        self.projected_preparation_flops+=projected_row_preparation_flops(self.probe,self.source.seq_len)
        if self.probe.h is not None:
            self.counts["projected_row_prepare_calls"]+=1
            self.counts["projected_row_prepare_rows"]+=self.source.seq_len

    def report(self):
        result=super().report()
        features=tuple(self.features or ())
        tensors=[self.cache.k,self.cache.v,self.source.k,self.source.v,self.source_ids,*features]
        if self.padded is not None:
            tensors.append(self.padded)
        c=self.probe.cfg
        element=self.cache.k.element_size()
        result["projected_execution"]=dict(extra_row_preparation_flops=self.projected_preparation_flops,
            materialized_cache_logical_bytes=_tensor_bytes(self.cache.k)+_tensor_bytes(self.cache.v),
            immutable_source_cache_logical_bytes=_tensor_bytes(self.source.k)+_tensor_bytes(self.source.v),
            immutable_raw_feature_logical_bytes=sum(_tensor_bytes(value) for value in features),
            padded_projected_table_logical_bytes=_tensor_bytes(self.padded) if self.padded is not None else 0,
            source_item_ids_logical_bytes=_tensor_bytes(self.source_ids),
            retained_tensor_storage_bytes=_storage_bytes(tensors),
            full_window_projected_table_bytes=(c.num_layers*c.num_heads*self.native.max_length*c.hidden_width*element
                if self.probe.h is not None else 0),
            full_window_native_kv_bytes=2*c.num_layers*c.num_heads*self.native.max_length*c.head_dim*element,
            projection_table_to_native_kv_ratio=c.hidden_width/(2*c.head_dim) if self.probe.h is not None else 0.,
            scope="per active user; raw source/features retained for exact short-window fallback and FIFO; borrowed projected graph output may overlap pool buffers; logical bytes exclude evicted rows but storage bytes include backing allocations")
        return result


@torch.no_grad()
def replay_projected_probe_user(parent,current,probe,events,requests_by_timestamp,uid,cfg,*,pool,tag=None,
                                stop_after_last_query=True):
    requests_by_timestamp={timestamp:rows for timestamp,rows in requests_by_timestamp.items() if rows}
    if stop_after_last_query and not requests_by_timestamp:
        return [],dict(counts=dict(quality_skipped_no_feedback_users=1),initial_shapes=[],prepare_shapes=[],
            read_shapes=[],append_shapes=[],padded_read_shapes=[],projected_execution=dict(extra_row_preparation_flops=0),
            quality_shortcut=dict(stop_after_last_query=True,no_feedback_gpu_skipped=True))
    trajectory=ProjectedFastProbeTrajectory(parent,current,probe,events,cfg,pool=pool)
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
        last_executed_timestamp=trajectory.native.time,no_feedback_gpu_skipped=False)
    return scores,report


@torch.no_grad()
def evaluate_projected_probe(parent,current,probe,histories,by_user,uids,cfg,device=None,*,tag=None,
                             on_user=None,pool=None,stop_after_last_query=True):
    """evaluate_fast_probe's shard interface, with extra row-prep/memory records."""
    pool=pool or ProjectedGraphPool(current,probe,window=int(cfg.get("history_length",1024)))
    scores,counts=[],Counter()
    shapes={name:Counter() for name in ("initial_shapes","prepare_shapes","read_shapes","append_shapes","padded_read_shapes")}
    preparation=0
    memory_maxima=Counter()
    for index,uid in enumerate(uids):
        values,report=replay_projected_probe_user(parent,current,probe,histories.rows[uid],by_user.get(uid,{}),
            uid,cfg,pool=pool,tag=tag,stop_after_last_query=stop_after_last_query)
        scores.extend(values)
        counts.update(report["counts"])
        for name,counter in shapes.items():
            for row in report[name]:
                key=tuple((field,value) for field,value in row.items() if field!="calls")
                counter[key]+=row["calls"]
        state=report["projected_execution"]
        preparation+=state["extra_row_preparation_flops"]
        for name,value in state.items():
            if name.endswith("bytes"):
                memory_maxima[name]=max(memory_maxima[name],value)
        if on_user is not None:
            on_user(index+1,len(uids),uid,report)
    return scores,dict(counts=dict(counts),
        **{name:[dict(key,calls=calls) for key,calls in sorted(counter.items())] for name,counter in shapes.items()},
        graph_pool=pool.report(),projected_execution=dict(extra_row_preparation_flops=preparation,
            max_observed_end_state_bytes=dict(memory_maxima),
            cost_scope="GPU quality execution only; full deployment still requires complete CPU event ledger and projected preparation/read/startup costs"),
        quality_shortcut=dict(stop_after_last_query=bool(stop_after_last_query),
            skipped_no_feedback_users=counts.get("quality_skipped_no_feedback_users",0)))
