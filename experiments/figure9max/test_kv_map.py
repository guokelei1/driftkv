"""Check analytical training GEMMs against actual autograd, plus real capacity."""
from pathlib import Path
import sys
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from experiments.figure9max.common import configuration
from hstu_kvcache.methods.kv_map.cost_max import fitting_flops, step_flops
from hstu_kvcache.methods.kv_map.max import normalized_forward, fit_one, apply_map, apply_map_reference
from hstu_kvcache.models import HSTUKVCache

class MatrixTrace(TorchDispatchMode):
    def __init__(self):
        self.operations=[]

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        if func==torch.ops.aten.bmm.default:
            a,b=args[:2]
            self.operations.append(dict(a=list(a.shape),b=list(b.shape),flops=2*a.shape[0]*a.shape[1]*a.shape[2]*b.shape[2]))
        return func(*args,**(kwargs or {}))

def test_backward_has_three_gemms_for_frozen_inputs():
    torch.manual_seed(17)
    g,n,d,w=2,7,4,6
    x=torch.randn(g,n,d)
    params=[torch.nn.Parameter(a) for a in (torch.randn(g,d,w),torch.zeros(g,1,w),torch.randn(g,w,d),torch.zeros(g,1,d))]
    with MatrixTrace() as forward:
        loss=torch.nn.functional.mse_loss(normalized_forward(x,*params),torch.randn_like(x))
    with MatrixTrace() as backward:
        loss.backward()
    expected=step_flops(n,w,g,d)
    assert len(forward.operations)==2 and len(backward.operations)==3
    assert sum(a["flops"] for a in forward.operations)==expected["forward_gemm"]
    assert sum(a["flops"] for a in backward.operations)==expected["backward_gemm"]
    assert x.grad is None

def test_all_epochs_partial_batches_and_diagnostics_match_actual_gemms():
    torch.manual_seed(17)
    g,n,d,w=2,15,4,6
    x=torch.randn(g,n,d)
    y=x.square()+.3*x
    settings=dict(configuration()["kv_training"],epochs=2,batch_rows=7)
    with MatrixTrace() as trace:
        _,profile=fit_one(x,y,w,settings,17)
    audit=fitting_flops(n,w,2,7,0,g,d)
    assert profile["steps"]==audit["steps"]==6
    actual=sum(a["flops"] for a in trace.operations)
    diagnostic_gemms=2*g*min(2048,n)*4*d*w
    assert actual==audit["parts"]["forward_gemm"]+audit["parts"]["backward_gemm"]+diagnostic_gemms
    assert profile["parameters"]==audit["parameters"]

def test_training_learns_nonlinear_teacher_without_backbone_gradients():
    torch.manual_seed(17)
    x=torch.randn(2,31,4)
    y=.4*x.square()+x
    settings=dict(configuration()["kv_training"],epochs=8,batch_rows=16,learning_rate=.02)
    _,profile=fit_one(x,y,8,settings,17)
    assert profile["normalized_teacher_probe_mse_after"]<profile["normalized_teacher_probe_mse_before"]

def test_channel_packing_matches_independent_linear_calls():
    torch.manual_seed(17)
    cache=HSTUKVCache(torch.randn(16,2,7,320,dtype=torch.float64),torch.randn(16,2,7,320,dtype=torch.float64),7)
    weights=(torch.randn(320,1,32,dtype=torch.float64),torch.rand(320,1,32,dtype=torch.float64)+.5,
        torch.randn(320,32,8,dtype=torch.float64)*.1,torch.randn(320,1,8,dtype=torch.float64)*.1,
        torch.randn(320,8,32,dtype=torch.float64)*.1,torch.randn(320,1,32,dtype=torch.float64)*.1,
        torch.randn(320,1,32,dtype=torch.float64),torch.rand(320,1,32,dtype=torch.float64)+.5)
    actual=apply_map(cache,weights)
    reference=apply_map_reference(cache,weights)
    torch.testing.assert_close(actual.k,reference.k,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(actual.v,reference.v,atol=1e-12,rtol=1e-12)

def test_hidden_expansion_is_nonlinear_capacity_not_redundant_linear_rank():
    x=torch.tensor([[[.3,.7]]])
    step=torch.tensor([[[.2,-.1]]])
    w1=torch.tensor([[[1.,-1.,.5],[.3,.7,1.]]])
    w2=torch.tensor([[[1.,.2],[.4,-.7],[.2,.8]]])
    b1=torch.zeros(1,1,3);b2=torch.zeros(1,1,2)
    curvature=normalized_forward(x+step,w1,b1,w2,b2)+normalized_forward(x-step,w1,b1,w2,b2)-2*normalized_forward(x,w1,b1,w2,b2)
    assert curvature.abs().max()>1e-4
