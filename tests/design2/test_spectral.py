import torch

from hstu_kvcache.design2.spectral import prepare_factor, quadratic_bounds, stack_specs


def test_partial_bounds_include_exact_with_nonzero_remainder():
    torch.manual_seed(170915)
    p, k = 48, 8
    q, _ = torch.linalg.qr(torch.randn(p, p, dtype=torch.float64))
    h = (q*torch.logspace(-2, 2, p, dtype=torch.float64))@q.T
    l = torch.linalg.cholesky(h)
    item = prepare_factor(l, k)
    # Includes selected-space, orthogonal-space and arbitrary directions.
    f = torch.cat([q.T, torch.randn(32, p, dtype=torch.float64)], 0)[None, None]
    lo, hi = quadratic_bounds(f, stack_specs([item]))
    exact = torch.linalg.solve_triangular(l, f[0, 0].T, upper=False).square().sum(0)
    assert torch.all(lo[0, 0] <= exact) and torch.all(exact <= hi[0, 0])
    assert lo[0, 0, -1] > 0
    assert hi[0, 0, k+5] > 0  # Discarded projection is not treated as zero.


def test_ridge_degenerate_spectrum_and_zero_vectors():
    l = torch.eye(40, dtype=torch.float64)*.1
    spec = stack_specs([prepare_factor(l, 32)])
    f = torch.zeros(1, 1, 2, 40, dtype=torch.float64)
    f[0, 0, 1, -1] = 1
    lo, hi = quadratic_bounds(f, spec)
    assert lo[0, 0, 0] == 0 and hi[0, 0, 0] >= 0
    assert lo[0, 0, 1] <= 100 <= hi[0, 0, 1]
