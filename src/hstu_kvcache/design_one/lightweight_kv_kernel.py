"""Inference-only single-scan Key/Value correction with one shared narrow code.

The native cache stays intact. Query projection into the code space corrects
attention scores; that same attention aggregates native V and codes, followed
by the Value decoder. No expanded K/V or history attention matrix is stored.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _read(Q,K,V,Z,DK,DV,Out,
          Q0:tl.constexpr,Q1:tl.constexpr,Q2:tl.constexpr,
          K0:tl.constexpr,K2:tl.constexpr,V0:tl.constexpr,V2:tl.constexpr,
          Z0:tl.constexpr,Z1:tl.constexpr,Z2:tl.constexpr,
          H:tl.constexpr,M:tl.constexpr,N:tl.constexpr,SCALE:tl.constexpr,BN:tl.constexpr):
    query,bh=tl.program_id(0),tl.program_id(1)
    batch,head=bh//H,bh%H
    rows,dim,rank=tl.arange(0,BN),tl.arange(0,32),tl.arange(0,8)
    q=tl.load(Q+batch*Q0+head*Q1+query*Q2+dim)
    dk=tl.load(DK+head*8*32+rank[:,None]*32+dim[None,:])
    qcode=tl.sum(dk*q[None,:],axis=1)
    k=tl.load(K+batch*K0+head*32+rows[:,None]*K2+dim[None,:],rows[:,None]<N,0.)
    z=tl.load(Z+batch*Z0+head*Z1+rows[:,None]*Z2+rank[None,:],rows[:,None]<N,0.)
    scores=(tl.sum(k*q[None,:],axis=1)+tl.sum(z*qcode[None,:],axis=1))*SCALE
    elu=tl.where(scores>0,scores,tl.exp(scores)-1.)
    weights=tl.where(rows<N,elu+1.,0.)
    v=tl.load(V+batch*V0+head*32+rows[:,None]*V2+dim[None,:],rows[:,None]<N,0.)
    native=tl.sum(weights[:,None]*v,axis=0)
    latent=tl.sum(weights[:,None]*z,axis=0)
    dv=tl.load(DV+head*8*32+rank[:,None]*32+dim[None,:])
    result=native+tl.sum(latent[:,None]*dv,axis=0)
    tl.store(Out+(bh*M+query)*32+dim,result)


def fused_kv_read(q,k,v,codes,key_decoder,value_decoder,*,scale=1.):
    """FP32 CUDA [B,6,Q<=16,32], native history N<=1024, shared rank8."""
    if torch.is_grad_enabled():raise ValueError("KV kernel is inference-only")
    batch,heads,queries,dim=q.shape
    n=k.shape[1]
    if (q.device.type!="cuda" or q.dtype!=torch.float32 or heads!=6 or dim!=32
            or not 1<=queries<=16 or not 1<=n<=1024
            or k.shape!=(batch,n,heads*dim) or v.shape!=k.shape
            or codes.shape!=(batch,n,heads,8)
            or key_decoder.shape!=(heads,8,dim) or value_decoder.shape!=key_decoder.shape):
        raise ValueError("KV kernel requires bounded Medium FP32 rank8 shapes")
    tensors=(k,v,codes,key_decoder,value_decoder)
    if (q.stride(-1)!=1 or any(x.dtype!=q.dtype or x.device!=q.device or x.stride(-1)!=1 for x in tensors)
            or not key_decoder.is_contiguous() or not value_decoder.is_contiguous()):
        raise ValueError("KV kernel requires matching native layout and contiguous decoders")
    out=torch.empty_like(q,memory_format=torch.contiguous_format)
    _read[(queries,batch*heads)](q,k,v,codes,key_decoder,value_decoder,out,
        *q.stride()[:3],k.stride(0),k.stride(1),v.stride(0),v.stride(1),
        codes.stride(0),codes.stride(2),codes.stride(1),H=heads,M=queries,N=n,SCALE=float(scale),
        BN=triton.next_power_of_2(n),num_warps=8,num_stages=1,enable_fp_fusion=False)
    return out


def make_history_read(adapter,state):
    def read(index,attention,q,k,v):
        if (attention.training or attention.position_bias is not None
                or attention.block_variant!="legacy" or attention.cfg.activation!="elu_plus1"):
            raise ValueError("KV kernel implements frozen legacy ELU+1 history reads")
        if state.seq_len!=k.shape[1]:raise ValueError("KV codes and history rows differ")
        return fused_kv_read(q,k,v,state.codes[index],adapter.layers[index].key_decoder,
                             adapter.layers[index].decoder,scale=attention.scale)
    return read
