"""History scan with query-code projection inside the same CTA.

Derived from implementation_2026_10.reader_kernel; native attention, activation,
mask and aggregation retain their original order. Gate inputs remain dynamic
and are prepared once per operation; no full K/V rewriting or new fit occurs.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "COUNT", "K0", "V0", "Z0", "M0", "M1", "M2", "M3"])
def _read_vector(Q,K,V,Z,P,G,DV,DK,Mask,Out,
          Q0:tl.constexpr,Q1:tl.constexpr,Q2:tl.constexpr,
          K0,K2:tl.constexpr,V0,V2:tl.constexpr,Z0,Z2:tl.constexpr,
          P0:tl.constexpr,P1:tl.constexpr,P2:tl.constexpr,
          G0:tl.constexpr,G1:tl.constexpr,G2:tl.constexpr,
          M0,M1,M2,M3,
          H:tl.constexpr,M:tl.constexpr,N,COUNT,SCALE:tl.constexpr,
          HAS_CODE:tl.constexpr,HAS_MASK:tl.constexpr,COUNT_POINTER:tl.constexpr,BN:tl.constexpr):
    """One Query/head CTA: FP32 products/reductions, no Query-tile padding."""
    query,bh=tl.program_id(0),tl.program_id(1)
    batch,head=bh//H,bh%H
    count=tl.load(COUNT) if COUNT_POINTER else COUNT
    dims=tl.arange(0,32)
    ranks=tl.arange(0,8)
    q=tl.load(Q+batch*Q0+head*Q1+query*Q2+dims)
    native=tl.zeros((32,),tl.float32)
    latent=tl.zeros((8,),tl.float32)
    if HAS_CODE:
        gk=tl.load(P+batch*P0+head*P1+query*P2+ranks,ranks<6,0)
        dk=tl.load(DK+head*6*32+ranks[:,None]*32+dims[None,:],ranks[:,None]<6,0)
        p=tl.sum(dk*q[None,:],axis=1)*gk
        coord=tl.where(ranks<3,ranks,12+head*6+(ranks-3))
    for start in range(0,N,BN):
        cols=start+tl.arange(0,BN)
        valid=cols<N
        k=tl.load(K+batch*K0+cols[:,None]*K2+head*32+dims[None,:],valid[:,None],0)
        v=tl.load(V+batch*V0+cols[:,None]*V2+head*32+dims[None,:],valid[:,None],0)
        scores=tl.sum(k*q[None,:],axis=1)
        if HAS_CODE:
            z=tl.zeros((BN,8),tl.float32)
            if start<count:
                z=tl.load(Z+batch*Z0+cols[:,None]*Z2+coord[None,:],
                          (cols[:,None]<count)&(ranks[None,:]<6),0)
                scores=scores+tl.sum(z*p[None,:],axis=1)
        scaled=scores*SCALE
        elu=tl.where(scaled>0,scaled,tl.exp(scaled)-1.)
        weights=elu+1.
        if HAS_MASK:
            mv=tl.load(Mask+batch*M0+head*M1+query*M2+cols*M3,valid,0)
            weights=weights*mv.to(tl.float32)
        weights=tl.where(valid,weights,0.)
        native=native+tl.sum(weights[:,None]*v,axis=0)
        if HAS_CODE:
            if start<count:
                latent=latent+tl.sum(weights[:,None]*z,axis=0)
    if HAS_CODE:
        g=tl.load(G+batch*G0+head*G1+query*G2+ranks,ranks<6,0)
        dv=tl.load(DV+head*6*32+ranks[:,None]*32+dims[None,:],ranks[:,None]<6,0)
        native=native+tl.sum((latent*g)[:,None]*dv,axis=0)
    tl.store(Out+(bh*M+query)*32+dims,native)


@triton.jit(do_not_specialize=["N", "COUNT", "K0", "V0", "Z0", "M0", "M1", "M2", "M3"])
def _read(Q,K,V,Z,P,G,DV,DK,Mask,Out,
          Q0:tl.constexpr,Q1:tl.constexpr,Q2:tl.constexpr,
          K0,K2:tl.constexpr,V0,V2:tl.constexpr,Z0,Z2:tl.constexpr,
          P0:tl.constexpr,P1:tl.constexpr,P2:tl.constexpr,
          G0:tl.constexpr,G1:tl.constexpr,G2:tl.constexpr,
          M0,M1,M2,M3,
          H:tl.constexpr,M:tl.constexpr,N,COUNT,SCALE:tl.constexpr,
          HAS_CODE:tl.constexpr,HAS_MASK:tl.constexpr,COUNT_POINTER:tl.constexpr,
          BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr,BR:tl.constexpr):
    block,bh=tl.program_id(0),tl.program_id(1)
    batch,head=bh//H,bh%H
    count=tl.load(COUNT) if COUNT_POINTER else COUNT
    rows=block*BM+tl.arange(0,BM)
    dims=tl.arange(0,BD)
    ranks=tl.arange(0,BR)
    q=tl.load(Q+batch*Q0+head*Q1+rows[:,None]*Q2+dims[None,:],rows[:,None]<M,0)
    native=tl.zeros((BM,BD),tl.float32)
    latent=tl.zeros((BM,BR),tl.float32)
    if HAS_CODE:
        gk=tl.load(P+batch*P0+head*P1+rows[:,None]*P2+ranks[None,:],
                   (rows[:,None]<M)&(ranks[None,:]<6),0)
        dk=tl.load(DK+head*6*32+ranks[:,None]*32+dims[None,:],ranks[:,None]<6,0)
        p=tl.dot(q,tl.trans(dk),input_precision="tf32x3")*gk
        coord=tl.where(ranks<3,ranks,12+head*6+(ranks-3))
    for start in range(0,N,BN):
        cols=start+tl.arange(0,BN)
        kv_mask=cols[:,None]<N
        k=tl.load(K+batch*K0+cols[:,None]*K2+head*32+dims[None,:],kv_mask,0)
        v=tl.load(V+batch*V0+cols[:,None]*V2+head*32+dims[None,:],kv_mask,0)
        scores=tl.dot(q,tl.trans(k),input_precision="tf32x3")
        if HAS_CODE:
            z=tl.zeros((BN,BR),tl.float32)
            if start<count:
                z=tl.load(Z+batch*Z0+cols[:,None]*Z2+coord[None,:],
                          (cols[:,None]<count)&(ranks[None,:]<6),0)
                scores=scores+tl.dot(p,tl.trans(z),input_precision="tf32x3")
        scaled=scores*SCALE
        elu=tl.where(scaled>0,scaled,tl.exp(scaled)-1.)
        weights=elu+1.
        keep=(rows[:,None]<M)&(cols[None,:]<N)
        if HAS_MASK:
            mv=tl.load(Mask+batch*M0+head*M1+rows[:,None]*M2+cols[None,:]*M3,keep,0)
            weights=weights*mv.to(tl.float32)
        weights=tl.where(keep,weights,0.)
        native=tl.dot(weights,v,native,input_precision="tf32x3")
        if HAS_CODE:
            if start<count:
                latent=tl.dot(weights,z,latent,input_precision="tf32x3")
    if HAS_CODE:
        g=tl.load(G+batch*G0+head*G1+rows[:,None]*G2+ranks[None,:],
                  (rows[:,None]<M)&(ranks[None,:]<6),0)
        dv=tl.load(DV+head*6*32+ranks[:,None]*32+dims[None,:],ranks[:,None]<6,0)
        native=native+tl.dot(latent*g,dv,input_precision="tf32x3")
    tl.store(Out+(bh*M+rows[:,None])*32+dims[None,:],native,rows[:,None]<M)


def launch_read(q,k,v,codes,key_gate,value_gate,key_decoder,value_decoder,mask,mask_strides,*,code_count,scale,count_tensor=None):
    b,h,m,d=q.shape
    out=torch.empty_like(q,memory_format=torch.contiguous_format)
    launch_args=(q,k,v,codes,key_gate,value_gate,value_decoder,key_decoder,
        q if mask is None else mask,out,*q.stride()[:3],
        k.stride(0),k.stride(1),v.stride(0),v.stride(1),codes.stride(0),codes.stride(1),
        *key_gate.stride()[:3],*value_gate.stride()[:3],*mask_strides)
    meta=dict(H=h,M=m,N=k.shape[1],COUNT=code_count if count_tensor is None else count_tensor,SCALE=scale,
              HAS_CODE=bool(code_count),HAS_MASK=mask is not None,COUNT_POINTER=count_tensor is not None)
    if m<=4:
        _read_vector[(m,b*h)](*launch_args,**meta,BN=64,
            num_warps=4,num_stages=1,enable_fp_fusion=False)
    else:
        _read[(triton.cdiv(m,16),b*h)](*launch_args,**meta,
            BM=16,BN=64,BD=32,BR=16,num_warps=4,num_stages=1,enable_fp_fusion=False)
    return out
