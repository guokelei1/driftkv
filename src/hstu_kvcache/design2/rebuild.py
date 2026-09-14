"""Exact cache-only computation for frozen legacy HSTU; tiled causal attention."""
import torch
import torch.nn.functional as F
from hstu_kvcache.models import HSTUKVCache


@torch.no_grad()
def current_kv(model, items, actions, deltas, tile=64):
    assert not model.training and model.cfg.block_variant=='legacy'
    x=model.embed_inputs(items,actions,deltas); batch,n,width=x.shape; layers=[]
    for l,block in enumerate(model.blocks):
        norm=block.norm(x);att=block.attn
        assert att.position_bias is None and att.causal_diagonal=='inclusive' and block.gating=='silu_gate'
        if l==len(model.blocks)-1:
            layers.append(att.project_kv(norm));break
        q,k,v=att._project(norm)
        pieces=[]
        for lo in range(0,n,tile):
            hi=min(n,lo+tile)
            w=att._activate((q[:,:,lo:hi]@k[:,:,:hi].transpose(-2,-1))*att.scale)
            mask=torch.arange(hi,device=x.device)[None,:]<=torch.arange(lo,hi,device=x.device)[:,None]
            pieces.append((w*mask)@v[:,:,:hi])
        out=torch.cat(pieces,dim=2).transpose(1,2).reshape(batch,n,width)
        x=x+att.out_proj(out)*F.silu(block.gate_proj(norm))
        layers.append((k.transpose(1,2).reshape(batch,n,width),v.transpose(1,2).reshape(batch,n,width)))
    return HSTUKVCache.from_layer_list(layers,n)


def tiled_flops(n,tile=64):
    # Cache-only five blocks and final norm/KV. Include unused upper entries
    # inside each causal tile; no claim that a rectangular GEMM is triangular.
    from design.report_native_flops import rebuild,value
    pairs=sum(min(tile,n-lo)*min(n,lo+tile) for lo in range(0,n,tile))
    extra=pairs-n*(n+1)//2
    return value(rebuild(n))+extra*(4*5*192+3*5*6)
