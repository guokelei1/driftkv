import torch

from design.diagnose_native_input import fit_joint
from hstu_kvcache.design2.geometry import factorize, features, gram_sum, score_layer


def test_frozen_objective_and_aggregate_geometry():
    g = torch.Generator().manual_seed(170908)
    def r(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64)
    x, q, o, y = r(5,3), r(5,2,4,3), r(5,4,4), r(5,2,4,3)
    x[:,0] = 1
    n = torch.tensor([16,32,128,512,1024])
    mu = 700000.
    p,_ = fit_joint(x,q,y,o,n,count_square_mean=mu)
    f = features(x,q,o,p)
    gram = gram_sum(x,q,o,n,p,mu)
    explicit = torch.einsum("shqa,shqb,s->hab",f,f,n.double().square()/mu)/4
    torch.testing.assert_close(gram,explicit,atol=1e-10,rtol=1e-10)
    system = gram+.05*torch.eye(f.shape[-1])
    rhs = torch.einsum("shqa,shqd,s->had",f,y,n.double().square()/mu)/4
    theta = torch.cat((p["weights"].transpose(0,1).flatten(1,2),p["read_weights"]),1)
    torch.testing.assert_close(system@theta,rhs,atol=1e-9,rtol=1e-9)
    chol,_ = factorize(gram,5)
    active = torch.tensor([1,0,1,1,1])
    aggregate,rate = score_layer(x,q,o,n,active,p,chol)
    direct = torch.einsum("shqa,hab,shqb->shq",f,torch.linalg.inv(system/5),f).sqrt().amax(1)
    torch.testing.assert_close(rate,direct)
    torch.testing.assert_close(aggregate,direct*(n*active)[:,None])
    assert aggregate[1].eq(0).all()
