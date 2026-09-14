"""The cheap statistic must be the minimum quadratic, not an inverse subblock."""
import torch
from hstu_kvcache.design2.evidence import quadratic

def test_known_coordinates_minimize_original_quadratic():
    torch.manual_seed(17)
    a=torch.randn(19,19,dtype=torch.float64)
    h=a@a.T+torch.eye(19,dtype=torch.float64)*.3
    i=torch.tensor([0,3,7,13]);v=torch.randn(4,5,dtype=torch.float64)
    hii=h[i][:,i];completion=h[:,i]@torch.linalg.solve(hii,v)
    torch.testing.assert_close(completion[i],v)
    floor=quadratic(v.T[None,None],torch.linalg.cholesky(hii)[None])[0,0].square()
    actual=(completion*torch.linalg.solve(h,completion)).sum(0)
    torch.testing.assert_close(floor,actual,rtol=1e-12,atol=1e-12)
    other=completion+torch.randn_like(completion);other[i]=v
    assert torch.all((other*torch.linalg.solve(h,other)).sum(0)>=floor)
