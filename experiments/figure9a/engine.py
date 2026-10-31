"""User/policy batching, shared repair prefixes and fused sparse attention."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
from collections import Counter, defaultdict
from contextlib import contextmanager
import gc
import time

from experiments.figure9a.common import DAY, ROOT
import numpy as np
import torch
from hstu_kvcache.evaluation.yambda import load_model
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.models.state_transition import append_with_rolling_band
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.methods.baselines.layer_recompute.core import LayerRecomputeState, recompute_interval
from hstu_kvcache.methods.baselines.tail_recompute import recompute_tail
from hstu_kvcache.methods.baselines.deviation_recompute import layer0_deviation_scores
from hstu_kvcache.methods.baselines.sparse_recompute import recompute_positions, _block_output
from hstu_kvcache.evaluation.trajectory import _short_prefix_backend
from hstu_kvcache.methods.cachetide.kernels.reader import launch_read
from hstu_kvcache.evaluation.fast_baselines import native_read
from hstu_kvcache.methods.kv_map.large import fit_maps as fit_mlp_maps, apply_map, apply_map_reference

def load_models(cfg,edge,device):
    models=[]
    for version in (edge-1,edge):
        model,payload=load_model(ROOT/cfg["checkpoints"]/f"v{version}/checkpoint_100.pt",device)
        assert model.cfg.num_layers==10 and model.cfg.hidden_size==320
        model.eval().requires_grad_(False)
        models.append(model)
        del payload
        gc.collect()
    return models

def raw_batch(histories,uids,at,device):
    prefixes=[]
    cursors=[]
    for uid in uids:
        events=histories.rows[uid]
        stop=int(np.searchsorted(events[0],at,side="left"))
        n=min(stop,1024)
        cursors.append(stop)
        prefixes.append(tuple(a[stop-n:stop] for a in events))
    assert len({len(p[0]) for p in prefixes})==1
    times=np.stack([p[0] for p in prefixes])
    dt=np.zeros(times.shape,dtype=np.float32)
    dt[:,1:]=np.diff(times,axis=1)
    tensors=(torch.as_tensor(np.stack([p[1] for p in prefixes]),device=device,dtype=torch.long),
        torch.as_tensor(np.stack([p[2] for p in prefixes]),device=device,dtype=torch.long),
        torch.as_tensor(dt,device=device))
    return tensors,cursors,[int(p[0][-1]) for p in prefixes]

@contextmanager
def capture_needed(model,starts):
    bounds={}
    handles=[model.blocks[l].norm.register_forward_pre_hook(
        lambda module,args,l=l:bounds.__setitem__(l,args[0].detach().clone())) for l in starts]
    try:
        yield bounds
    finally:
        for h in handles:
            h.remove()

def fit_maps(parent,current,histories,fit_uids,cfg,rows,device):
    return fit_mlp_maps(parent,current,histories,fit_uids,cfg,rows,device,raw_batch)

@torch.no_grad()
def fused_positions(current,cache,raw,positions):
    """Same sparse replay, fused QK/ELU/mask/AV; no dense attention weights.

    The full-window budgets are multiples of 64: visited tiles equal the dense
    M*N cost convention. Short histories use the unmodified dense reference.
    """
    n,m=cache.seq_len,positions.shape[1]
    if n!=1024 or m==n:
        return recompute_positions(current,cache,*raw,positions,query_chunk_size=64)
    x=current.embed_inputs(*(a.gather(1,positions) for a in raw))
    mask=(torch.arange(n,device=x.device)[None,None,None,:]<=positions[:,None,:,None])
    keys,values=[],[]
    batch=positions.shape[0]
    for l,block in enumerate(current.blocks):
        residual,normalized=x,block.norm(x)
        attn=block.attn
        assert attn.position_bias is None and attn.causal_diagonal=="inclusive" and block.block_variant=="legacy"
        q,sk,sv=attn._project(normalized)
        sk=sk.transpose(1,2).reshape(batch,m,320)
        sv=sv.transpose(1,2).reshape(batch,m,320)
        index=positions[:,:,None].expand(-1,-1,320)
        k=cache.k[l].scatter(1,index,sk)
        v=cache.v[l].scatter(1,index,sv)
        mask_strides=(mask.stride(0),0,mask.stride(2),mask.stride(3))
        out=launch_read(q,k,v,k[:,:0,:1],q,q,q,mask,mask_strides,code_count=0,scale=float(attn.scale))
        x=_block_output(block,residual,normalized,attn._finish(out))
        keys.append(k);values.append(v)
    return HSTUKVCache(torch.stack(keys),torch.stack(values),n)

@torch.no_grad()
def prepare(parent,current,raw,rows,maps,*,reference=False):
    n=raw[0].shape[1]
    starts={p["interval"][0] for p in rows if p["method"]=="layer" and p["interval"][0]}
    with _short_prefix_backend(parent,n,1024),capture_needed(parent,starts) as boundaries:
        old=parent.compute_kv(*raw)
    outputs={}
    with _short_prefix_backend(current,n,1024):
        if not reference:
            fresh=current.compute_kv(*raw)
            # The same current prefix is exact for every layer-0 prefix point.
            for p in rows:
                if p["method"]=="rebuild":
                    outputs[p["tag"]]=fresh
                elif p["method"]=="layer" and p["interval"][0]==0:
                    mask=(torch.arange(10,device=old.k.device)<=p["interval"][1])[:,None,None,None]
                    outputs[p["tag"]]=HSTUKVCache(torch.where(mask,fresh.k,old.k),torch.where(mask,fresh.v,old.v),n)
        rank=None
        if not reference and any(p["method"]=="deviation" and p["tokens"]<n for p in rows):
            importance=layer0_deviation_scores(current,old,*raw)
            rank=torch.argsort(importance,dim=1,descending=True,stable=True)
        for p in rows:
            tag,method=p["tag"],p["method"]
            if tag in outputs:
                continue
            if method=="reuse":
                outputs[tag]=old
            elif method=="rebuild":
                outputs[tag]=current.compute_kv(*raw)
            elif method=="layer":
                outputs[tag]=recompute_interval(current,LayerRecomputeState(old,boundaries),*raw,p["interval"]).cache
            elif method=="tail":
                outputs[tag]=recompute_tail(current,old,*raw,min(n,p["tokens"]))
            elif method=="deviation":
                m=min(n,p["tokens"])
                if m==n:
                    outputs[tag]=current.compute_kv(*raw)
                else:
                    ranked=(torch.argsort(layer0_deviation_scores(current,old,*raw),dim=1,descending=True,stable=True)
                        if reference else rank)
                    positions=ranked[:,:m].sort(dim=1).values
                    outputs[tag]=(recompute_positions(current,old,*raw,positions,query_chunk_size=64)
                        if reference else fused_positions(current,old,raw,positions))
            elif method=="kv_map":
                outputs[tag]=(apply_map_reference(old,maps[tag]) if reference else apply_map(old,maps[tag]))
            else:
                raise ValueError(method)
    # [L, users * policies, N, D], policy varies fastest.
    return HSTUKVCache(torch.stack([outputs[p["tag"]].k for p in rows],2).flatten(1,2),
        torch.stack([outputs[p["tag"]].v for p in rows],2).flatten(1,2),n)

class Graphs:
    """Shared graph pool and input bank; outputs are consumed before replay.

    Every append output is copied into the separate resident state immediately;
    score output is copied to CPU immediately. Pool sharing therefore never
    destroys a live state belonging to another user or graph.
    """
    def __init__(self,enabled=True):
        self.enabled=enabled
        self.entries={}
        self.banks={}
        self.outputs={}
        self.residents={}
        self.pool=torch.cuda.graph_pool_handle() if enabled else None
        self.stream=torch.cuda.Stream() if enabled else None
        self.stats=Counter()
        self.capture_seconds=0.
        self.preparation_entry=None

    def load_state(self,cache):
        key=tuple(cache.k.shape)
        if key not in self.residents:
            self.residents[key]=HSTUKVCache(torch.empty_like(cache.k),torch.empty_like(cache.v),cache.seq_len)
        resident=self.residents[key]
        if resident.k.data_ptr()!=cache.k.data_ptr():
            resident.k.copy_(cache.k);resident.v.copy_(cache.v)
        return resident

    def prepare(self,parent,current,raw,rows,maps):
        """Capture the repeated full-window preparation into the resident bank."""
        if self.preparation_entry is None:
            started=time.perf_counter()
            x,b,t=(a.clone() for a in raw)
            stream=self.stream
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                warm=prepare(parent,current,(x,b,t),rows,maps)
            torch.cuda.current_stream().wait_stream(stream)
            resident=self.load_state(warm)
            del warm
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            def compute_to_resident():
                value=prepare(parent,current,(x,b,t),rows,maps)
                resident.k.copy_(value.k);resident.v.copy_(value.v)
            with torch.cuda.graph(graph,pool=self.pool,stream=stream):
                compute_to_resident()
            self.preparation_entry=(graph,x,b,t,resident,maps)
            self.capture_seconds+=time.perf_counter()-started
        graph,x,b,t,resident,_=self.preparation_entry
        for target,value in zip((x,b,t),raw,strict=True):
            target.copy_(value)
        graph.replay()
        return resident

    def run(self,kind,model,cache,ids,dt,bh=None,*,direct=False):
        def compute(c,x,t,b):
            if kind=="append":
                return append_with_rolling_band(model,c,x,b,t,1024)
            return score(model,c,x,t,history_read_fn=native_read)[0]
        if not self.enabled or cache.seq_len!=1024:
            with _short_prefix_backend(model,cache.seq_len,1024):
                return compute(cache,ids,dt,bh)
        key=(kind,ids.shape,direct)
        if key not in self.entries:
            started=time.perf_counter()
            bank_key=tuple(cache.k.shape)
            if direct:
                staged=cache
            else:
                if bank_key not in self.banks:
                    self.banks[bank_key]=HSTUKVCache(torch.empty_like(cache.k),torch.empty_like(cache.v),1024)
                staged=self.banks[bank_key]
                staged.k.copy_(cache.k);staged.v.copy_(cache.v)
            x,t=ids.clone(),dt.clone()
            b=None if bh is None else bh.clone()
            stream=self.stream
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    compute(staged,x,t,b)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            if kind=="append":
                if direct:
                    output=staged
                    backup=(staged.k.clone(),staged.v.clone())
                else:
                    if bank_key not in self.outputs:
                        self.outputs[bank_key]=HSTUKVCache(torch.empty_like(cache.k),torch.empty_like(cache.v),1024)
                    output=self.outputs[bank_key]
            else:
                output=torch.empty_like(ids,dtype=cache.k.dtype)
            def compute_to_output():
                result=compute(staged,x,t,b)
                if kind=="append":
                    output.k.copy_(result.k);output.v.copy_(result.v)
                else:
                    output.copy_(result)
                # Temporary result tensors die at this function boundary, so
                # captures of other shapes can reuse their graph-pool storage.
            with torch.cuda.graph(graph,pool=self.pool,stream=stream):
                compute_to_output()
            if direct and kind=="append":
                staged.k.copy_(backup[0]);staged.v.copy_(backup[1])
            self.entries[key]=(graph,staged,x,t,b,output)
            self.capture_seconds+=time.perf_counter()-started
        graph,staged,x,t,b,output=self.entries[key]
        if not direct:
            staged.k.copy_(cache.k);staged.v.copy_(cache.v)
        x.copy_(ids);t.copy_(dt)
        if b is not None:
            b.copy_(bh)
        graph.replay()
        self.stats[kind]+=1
        return output

    def describe(self):
        return dict(entries=len(self.entries),preparation_graph=self.preparation_entry is not None,
            replays=dict(self.stats),capture_seconds=self.capture_seconds)

def waves(inputs,uids,batch,at):
    """Equal-length cohorts only; no padding changes causal histories."""
    groups=defaultdict(list)
    for uid in uids:
        if not inputs["by_user"].get(uid):
            yield [uid]
            continue
        n=min(1024,int(np.searchsorted(inputs["histories"].rows[uid][0],at,side="left")))
        groups[n].append(uid)
        if len(groups[n])==(batch if n==1024 else 1):
            yield groups.pop(n)
    for group in groups.values():
        yield group

@torch.no_grad()
def evaluate_wave(parent,current,inputs,uids,cfg,rows,maps,device,graphs,*,reference=False):
    """Group ready users by append width and query width, preserving each clock."""
    active=[u for u in uids if inputs["by_user"].get(u)]
    p=len(rows)
    if not active:
        return [],dict(users=len(uids),requests=0,append_groups=0,query_groups=0)
    raw,cursors,last=raw_batch(inputs["histories"],active,cfg["release_day"]*DAY,device)
    if graphs.enabled and raw[0].shape==(1,1024) and not reference:
        cache=graphs.prepare(parent,current,raw,rows,maps)
    else:
        cache=prepare(parent,current,raw,rows,maps,reference=reference)
    n=cache.seq_len
    total=len(active)
    direct=graphs.enabled and total==1 and n==1024
    if direct:
        cache=graphs.load_state(cache)
    k=cache.k.reshape(10,total,p,n,320).contiguous()
    v=cache.v.reshape(10,total,p,n,320).contiguous()
    rounds=[list(sorted(inputs["by_user"][u].items())) for u in active]
    current_round=[0]*total
    labels=[[] for _ in active]
    scores=[[] for _ in active]
    stats=Counter(users=len(uids))

    def take(indices):
        if indices==list(range(total)):
            return HSTUKVCache(k.flatten(1,2),v.flatten(1,2),n)
        index=torch.tensor(indices,device=device)
        return HSTUKVCache(k.index_select(1,index).flatten(1,2),v.index_select(1,index).flatten(1,2),n)

    def execute(kind,indices,ids,dt,bh=None):
        # Shape buckets bound graph storage to 1+2+4+8 user slots instead of
        # 1+2+...+8. Extra slots duplicate an independent user and are discarded.
        count=len(indices)
        padded=1<<(count-1).bit_length() if n==1024 else count
        if padded>count:
            extra=(padded-count)*p
            ids=torch.cat((ids,ids[-p:].repeat(padded-count,1)))
            dt=torch.cat((dt,dt[-p:].repeat((padded-count,)+(1,)*(dt.ndim-1))))
            if bh is not None:
                bh=torch.cat((bh,bh[-p:].repeat(padded-count,1)))
            assert len(ids)==count*p+extra
        result=graphs.run(kind,current,take(indices+[indices[-1]]*(padded-count)),ids,dt,bh,direct=direct)
        if kind=="append":
            return HSTUKVCache(result.k[:,:count*p],result.v[:,:count*p],result.seq_len)
        return result[:count*p]

    def advance(ready):
        nonlocal k,v,n
        targets={i:int(np.searchsorted(inputs["histories"].rows[active[i]][0],rounds[i][current_round[i]][0],side="left")) for i in ready}
        while True:
            groups=defaultdict(list)
            for i,stop in targets.items():
                if cursors[i]<stop:
                    w=1<<(min(128,stop-cursors[i]).bit_length()-1)
                    groups[w].append(i)
            if not groups:
                break
            for width,indices in groups.items():
                ids,bh,dt=[],[],[]
                for i in indices:
                    events=inputs["histories"].rows[active[i]]
                    start=cursors[i]
                    times=events[0][start:start+width]
                    delta=np.empty(width,np.float32)
                    delta[0]=int(times[0])-last[i]
                    delta[1:]=np.diff(times)
                    np.clip(delta,0,7*DAY,out=delta)
                    ids.append(events[1][start:start+width]);bh.append(events[2][start:start+width]);dt.append(delta)
                    cursors[i]+=width
                    last[i]=int(times[-1])
                values=[torch.as_tensor(np.stack(a),device=device,dtype=dtype).repeat_interleave(p,dim=0)
                    for a,dtype in ((ids,torch.long),(dt,torch.float32),(bh,torch.long))]
                result=execute("append",indices,*values)
                next_n=result.seq_len
                index=torch.tensor(indices,device=device)
                if next_n!=n:
                    # Short cohorts use scalar users: their growing histories
                    # cannot share a resident tensor with a different length.
                    assert total==1
                    k=result.k.reshape(10,total,p,next_n,320).clone()
                    v=result.v.reshape(10,total,p,next_n,320).clone()
                    n=next_n
                else:
                    if not direct:
                        k.index_copy_(1,index,result.k.reshape(10,len(indices),p,n,320))
                        v.index_copy_(1,index,result.v.reshape(10,len(indices),p,n,320))
                stats["append_groups"]+=1
    while True:
        ready=[i for i in range(total) if current_round[i]<len(rounds[i])]
        if not ready:
            break
        advance(ready)
        batches=defaultdict(list)
        for i in ready:
            timestamp,requests=rounds[i][current_round[i]]
            for offset in range(0,len(requests),cfg["query_batch"]):
                subset=requests[offset:offset+cfg["query_batch"]]
                batches[(offset,len(subset))].append((i,timestamp,subset))
        for _,group in sorted(batches.items()):
            indices=[a[0] for a in group]
            ids=torch.tensor([[r["item_idx"] for r in a[2]] for a in group],device=device).repeat_interleave(p,dim=0)
            dt=torch.tensor([a[1]-last[a[0]] for a in group],device=device,dtype=torch.float32).repeat_interleave(p)
            actual_width=ids.shape[1]
            padded_width=1<<(actual_width-1).bit_length()
            if padded_width>actual_width:
                ids=torch.cat((ids,ids[:,-1:].expand(-1,padded_width-actual_width)),1)
            result=execute("score",indices,ids,dt)[:,:actual_width].cpu().numpy().reshape(len(group),p,-1)
            for j,(i,_,subset) in enumerate(group):
                labels[i].extend(r["label"] for r in subset)
                scores[i].append(result[j].T.copy())
                stats["requests"]+=len(subset)
            stats["query_groups"]+=1
        for i in ready:
            current_round[i]+=1
    return [(uid,np.asarray(labels[i],dtype=np.uint8),np.concatenate(scores[i])) for i,uid in enumerate(active)],dict(stats)
