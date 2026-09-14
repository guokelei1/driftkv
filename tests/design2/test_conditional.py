import torch

from hstu_kvcache.design2.conditional import prepare_inverse, condition_state, evaluate_state, arithmetic


def test_source_contraction_matches_full_joint_and_solve():
    torch.manual_seed(170916)
    r, j, n, heads = 3, 4, 5, 2
    p = r*j+n
    a = torch.randn(heads, p, p, dtype=torch.float64)
    l = torch.linalg.cholesky(a@a.transpose(-2,-1)+.01*torch.eye(p))
    pack = prepare_inverse([l], r, j)[0][0]
    x = torch.randn(3, r, dtype=torch.float64)
    query = torch.randn(3, heads, 7, j-1, dtype=torch.float64)
    obs = torch.randn(3, 7, n, dtype=torch.float64)
    params = dict(query_center=torch.randn(heads, j-1, dtype=torch.float64)[:,None],
        query_scale=torch.ones(heads,1,j-1,dtype=torch.float64)*2,
        read_center=torch.randn(n,dtype=torch.float64), read_scale=torch.ones(n,dtype=torch.float64)*3)
    state = condition_state(x, pack)
    point, lo, hi, valid = evaluate_state(state, query, obs, params, pack)
    from hstu_kvcache.design2.geometry import features
    f = features(x, query, obs, params)
    rhs = f.permute(1,3,0,2).reshape(heads,p,-1)
    exact = torch.linalg.solve_triangular(l,rhs,upper=False).square().sum(1).reshape(heads,3,7).permute(1,0,2)
    assert valid.all()
    torch.testing.assert_close(point,exact,rtol=1e-11,atol=1e-11)
    assert torch.all(lo<=exact) and torch.all(exact<=hi)
    # All directions are retained; different x must produce a different matrix.
    assert not torch.equal(state['matrix'][0],state['matrix'][1])


def test_uncertain_quadratic_is_not_clipped_into_valid_score():
    pack = prepare_inverse([torch.eye(17,dtype=torch.float64)[None]],3,4)[0][0]
    x = torch.ones(1,3,dtype=torch.float64)
    state = condition_state(x,pack)
    state['matrix'].neg_()  # Synthetic numerical-invalid path, not an experiment perturbation.
    params=dict(query_center=torch.zeros(1,1,3),query_scale=torch.ones(1,1,3),
                read_center=torch.zeros(5),read_scale=torch.ones(5))
    point,_,hi,valid=evaluate_state(state,torch.ones(1,1,2,3),torch.ones(1,2,5),params,pack)
    assert not valid.any() and torch.isnan(point).all() and torch.isinf(hi).all()
    assert arithmetic(1)['panel_flops']>arithmetic(1)['exact_panel_flops']
    assert arithmetic(16)['panel_flops']<arithmetic(16)['exact_panel_flops']/5
