"""Unequal-length users exercise means and all weighted ridge statistics."""
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.fit_candidate_robust_maps import add_scaled_moments
from experiments.ch3_motivation.code.fit_simple_kv_maps import RowMoments, fit_from_moments


def test_user_weighted_ridge_matches_independent_weighted_algebra():
    x1=torch.tensor([[[1.,2.,-1.,0.],[3.,1.,0.,4.]],[[2.,-1.,1.,0.],[5.,0.,2.,1.]]])
    x2=torch.tensor([[[5.,0.,2.,1.],[2.,3.,1.,-2.],[4.,1.,0.,1.]],[[1.,3.,-1.,2.],[2.,4.,3.,0.],[0.,2.,1.,5.]]])
    y1=x1.square()/5+x1.flip(-1);y2=x2.square()/5+x2.flip(-1)
    moment=RowMoments(2,4,'cpu')
    for x,y in [(x1,y1),(x2,y2)]:
        user=RowMoments(2,4,'cpu');user.add(x,y)
        add_scaled_moments(moment,user,1/x.shape[1])
    x=torch.cat([x1,x2],1).double();y=torch.cat([y1,y2],1).double()
    weights=torch.tensor([.5,.5,1/3,1/3,1/3],dtype=torch.float64)
    mass=weights.sum();mean=(x*weights[None,:,None]).sum(1)/mass
    target=(y*weights[None,:,None]).sum(1)/mass
    for name,wanted in [('sx',(x*weights[None,:,None]).sum(1)),('sy',(y*weights[None,:,None]).sum(1)),
        ('yy',(y.square()*weights[None,:,None]).sum(1)),('xx',x.transpose(1,2).bmm(x*weights[None,:,None])),
        ('xy',x.transpose(1,2).bmm(y*weights[None,:,None]))]:
        torch.testing.assert_close(getattr(moment,name),wanted)
    assert moment.rows==2.
    scale=((x-mean[:,None]).square()*weights[None,:,None]).sum(1).div(mass).sqrt()
    normalized=(x-mean[:,None])/scale[:,None]
    gram=normalized.transpose(1,2).bmm(normalized*weights[None,:,None])/mass
    rhs=normalized.transpose(1,2).bmm((y-target[:,None])*weights[None,:,None])/mass
    for regularization,tensors in fit_from_moments(moment).items():
        wanted=torch.linalg.solve(gram+regularization*torch.eye(4)[None],rhs).float()
        torch.testing.assert_close(tensors['linear_weight'],wanted,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(tensors['xmean'],mean.float())
        torch.testing.assert_close(tensors['xscale'],scale.float())
        torch.testing.assert_close(tensors['ymean'],target.float())
