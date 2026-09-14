"""Cooperative boundaries in the existing cache-only dependency-closed rebuild.

Generator yields after actual GPU work submission, not midway through a kernel.
Its return value is the same target KV; scheduling does not change the action.
"""
import torch
import torch.nn.functional as F
from hstu_kvcache.models import HSTUKVCache


def rebuild_steps(model,items,actions,deltas,tile=64):
    assert not model.training and model.cfg.block_variant=='legacy'
    x=model.embed_inputs(items,actions,deltas);batch,n,width=x.shape;layers=[]
    yield 'embedding'
    for l,block in enumerate(model.blocks):
        norm=block.norm(x);att=block.attn
        assert att.position_bias is None and att.causal_diagonal=='inclusive' and block.gating=='silu_gate'
        if l==len(model.blocks)-1:
            layers.append(att.project_kv(norm));yield 'final_kv';break
        q,k,v=att._project(norm);pieces=[]
        yield 'projection'
        for lo in range(0,n,tile):
            hi=min(n,lo+tile)
            w=att._activate((q[:,:,lo:hi]@k[:,:,:hi].transpose(-2,-1))*att.scale)
            mask=torch.arange(hi,device=x.device)[None,:]<=torch.arange(lo,hi,device=x.device)[:,None]
            pieces.append((w*mask)@v[:,:,:hi]);yield 'tile'
        out=torch.cat(pieces,dim=2).transpose(1,2).reshape(batch,n,width)
        x=x+att.out_proj(out)*F.silu(block.gate_proj(norm))
        layers.append((k.transpose(1,2).reshape(batch,n,width),v.transpose(1,2).reshape(batch,n,width)))
        yield 'layer_finish'
    return HSTUKVCache.from_layer_list(layers,n)


def finish(generator,between=None):
    while True:
        try:stage=next(generator)
        except StopIteration as done:return done.value
        if between is not None:between(stage)
