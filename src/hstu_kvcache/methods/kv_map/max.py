"""Actual iterative head-local residual MLPs and their executed fit ledger."""
import gc
import math
import time
import torch
import torch.nn.functional as F
from hstu_kvcache.common.artifacts import DAY
from hstu_kvcache.methods.kv_map.cost_max import fitting_flops
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.evaluation.trajectory import _short_prefix_backend
from hstu_kvcache.evaluation.cost import CostModel

def normalized_forward(x,w1,b1,w2,b2):
    return torch.bmm(F.silu(torch.bmm(x,w1)+b1),w2)+b2

@torch.no_grad()
def normalize_inplace(x,floor):
    mean=x.mean(1,keepdim=True)
    x.sub_(mean)
    scale=x.square().mean(1,keepdim=True).sqrt().clamp_min(floor)
    x.div_(scale)
    return mean,scale

def fit_one(x,y,width,settings,seed):
    """All fixed epochs; no future feedback, validation search or early stop."""
    assert not x.requires_grad and not y.requires_grad
    channels,observations,dim=x.shape
    with torch.no_grad():
        mx,sx=normalize_inplace(x,settings["normalization_floor"])
        my,sy=normalize_inplace(y,settings["normalization_floor"])
        generator=torch.Generator(device=x.device).manual_seed(seed)
        w1=torch.nn.Parameter(torch.randn(channels,dim,width,device=x.device,generator=generator)*math.sqrt(2/dim))
        b1=torch.nn.Parameter(torch.zeros(channels,1,width,device=x.device))
        w2=torch.nn.Parameter(torch.zeros(channels,width,dim,device=x.device))
        b2=torch.nn.Parameter(torch.zeros(channels,1,dim,device=x.device))
        probe_x=x[:,:min(2048,observations)]
        probe_y=y[:,:min(2048,observations)]
        initial=float(F.mse_loss(normalized_forward(probe_x,w1,b1,w2,b2),probe_y))
    params=(w1,b1,w2,b2)
    optimizer=torch.optim.AdamW(params,lr=settings["learning_rate"],weight_decay=settings["weight_decay"],
        betas=(settings["beta1"],settings["beta2"]),eps=settings["epsilon"],foreach=False,fused=False)
    steps=0
    with torch.enable_grad():
        for _ in range(settings["epochs"]):
            order=torch.randperm(observations,device=x.device,generator=generator)
            for start in range(0,observations,settings["batch_rows"]):
                index=order[start:start+settings["batch_rows"]]
                bx=x.index_select(1,index)
                by=y.index_select(1,index)
                optimizer.zero_grad(set_to_none=True)
                loss=F.mse_loss(normalized_forward(bx,*params),by)
                loss.backward()
                optimizer.step()
                steps+=1
    with torch.no_grad():
        final=float(F.mse_loss(normalized_forward(probe_x,*params),probe_y))
    assert math.isfinite(final),"nonfinite prerelease teacher fit"
    weights=tuple(a.detach() for a in (mx,sx,w1,b1,w2,b2,my,sy))
    return weights,dict(steps=steps,parameters=sum(a.numel() for a in params),
        normalized_teacher_probe_mse_before=initial,normalized_teacher_probe_mse_after=final,
        diagnostic_scope="Same fixed first <=2048 prerelease fit rows; not held-out quality or checkpoint selection")

@torch.no_grad()
def fit_maps(parent,current,histories,fit_uids,cfg,rows,device,make_raw):
    """Capture teachers once, independently train each of the six real maps."""
    started=time.perf_counter()
    source_x,source_y,counts,teacher_costs=[],[],[],[]
    observations=capture=0
    cm=CostModel(320,16,10,"triton")
    for uid in fit_uids:
        raw,_,_=make_raw(histories,[uid],cfg["release_day"]*DAY,device)
        n=raw[0].shape[1]
        with _short_prefix_backend(parent,n,1024),_short_prefix_backend(current,n,1024):
            old=parent.compute_kv(*raw)
            teacher=current.compute_kv(*raw)
        x=torch.stack((old.k[:,0],old.v[:,0])).reshape(2,16,n,10,32).permute(0,1,3,2,4).reshape(320,n,32)
        y=torch.stack((teacher.k[:,0]-old.k[:,0],teacher.v[:,0]-old.v[:,0])).reshape(2,16,n,10,32).permute(0,1,3,2,4).reshape(320,n,32)
        source_x.append(x.cpu());source_y.append(y.cpu())
        observations+=n
        capture+=2*(cm if n==1024 else CostModel(320,16,10,"torch")).full_cache(n)
        counts.append(observations);teacher_costs.append(capture)
    del old,teacher,x,y,raw
    maps,costs,profiles={}, {}, {}
    for p in rows:
        if p["method"]!="kv_map":
            continue
        begin=time.perf_counter()
        users=p["fit_users"]
        x=torch.cat(source_x[:users],1).to(device)
        y=torch.cat(source_y[:users],1).to(device)
        weights,profile=fit_one(x,y,p["width"],cfg["kv_training"],cfg["seed"]+1000*users+p["width"])
        audit=fitting_flops(counts[users-1],p["width"],cfg["kv_training"]["epochs"],
            cfg["kv_training"]["batch_rows"],teacher_costs[users-1])
        assert audit["steps"]==profile["steps"] and audit["parameters"]==profile["parameters"]
        maps[p["tag"]]=weights
        costs[p["tag"]]=audit["total_flops"]
        profiles[p["tag"]]=dict(**profile,flop_breakdown=audit,seconds=time.perf_counter()-begin)
        print(__import__("json").dumps(dict(stage="mlp_map_fitted",method=p["tag"],**profiles[p["tag"]])),flush=True)
        del x,y
        gc.collect()
    del source_x,source_y
    torch.cuda.synchronize(device)
    return maps,costs,time.perf_counter()-started,profiles

@torch.no_grad()
def apply_map(cache,weights):
    mx,sx,w1,b1,w2,b2,my,sy=weights
    n,batch=cache.seq_len,cache.k.shape[1]
    x=torch.stack((cache.k,cache.v)).reshape(2,16,batch,n,10,32).permute(0,1,4,2,3,5).reshape(320,batch*n,32)
    mapped=x+(normalized_forward((x-mx)/sx,w1,b1,w2,b2)*sy+my)
    shaped=mapped.reshape(2,16,10,batch,n,32).permute(0,1,3,4,2,5).reshape(2,16,batch,n,320)
    return HSTUKVCache(shaped[0],shaped[1],n)

@torch.no_grad()
def apply_map_reference(cache,weights):
    """Independent unbatched linear calls check channel layout and weights."""
    mx,sx,w1,b1,w2,b2,my,sy=weights
    outputs=[]
    for kind,source in enumerate((cache.k,cache.v)):
        layers=[]
        for layer in range(16):
            heads=[]
            for head in range(10):
                index=kind*160+layer*10+head
                x=source[layer,:,:,head*32:(head+1)*32]
                normalized=(x-mx[index])/sx[index]
                hidden=F.silu(F.linear(normalized,w1[index].T,b1[index,0]))
                delta=F.linear(hidden,w2[index].T,b2[index,0])
                heads.append(x+(delta*sy[index]+my[index]))
            layers.append(torch.cat(heads,-1))
        outputs.append(torch.stack(layers))
    return HSTUKVCache(outputs[0],outputs[1],cache.seq_len)
