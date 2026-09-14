"""Nested observation constraints for the SAME ridge quadratic.

For H SPD and selected coordinates I, min_{g:g[I]=f[I]} g.T H^-1 g
= f[I].T H[I,I]^-1 f[I]. This is a lower bound, never a safety bound.
"""
import torch

def prepare(cholesky, include_observed=True):
    out=[]
    for L in cholesky:
        source=torch.arange(33,device=L.device)*33
        joint=torch.cat((source,torch.arange(1089,1281,device=L.device)))
        packs={}
        coordinates=[('source',source)]+([('observed',joint)] if include_observed else [])
        for name,index in coordinates:
            rows=L[:,index,:]
            gram=rows@rows.transpose(-1,-2)
            packs[name]=torch.linalg.cholesky(gram)
        out.append(packs)
    return out

def quadratic(f,chol):
    # f: batch,head,query,dimension
    b,h,q,p=f.shape
    rhs=f.permute(1,3,0,2).reshape(h,p,b*q)
    y=torch.linalg.solve_triangular(chol,rhs,upper=False)
    return y.square().sum(1).reshape(h,b,q).permute(1,0,2).sqrt()

def source_score(x,counts,active,packs):
    f=x[:,None,None,:].expand(-1,6,1,-1)
    return torch.stack([quadratic(f,p['source'])*counts[:,None,None]*active[:,l,None,None]
                        for l,p in enumerate(packs)]).amax((0,2,3))

def observed_score(x,observed,parameters,counts,active,packs):
    values=[]
    for l,p in enumerate(packs):
        o=(observed[l].double()-parameters[l]['read_center'])/parameters[l]['read_scale']
        f=torch.cat((x[:,None,:].expand(-1,o.shape[1],-1),o),-1)[:,None].expand(-1,6,-1,-1)
        values.append(quadratic(f,p['observed'])*counts[:,None,None]*active[:,l,None,None])
    return torch.stack(values).amax((0,2))
