"""FP64 partial-spectrum quadratic bounds, including eigensystem residuals.

The reference matrix is the exact real product of the stored Cholesky L L^T.
Bounds use the standard gamma_n rounding model; near-threshold decisions defer
to the existing accurate implementation. This is not a bound on model error.
"""

import torch


def gamma(n):
    u = torch.finfo(torch.float64).eps / 2
    return n*u/(1-n*u)


def prepare_factor(l, rank=32):
    p = l.shape[0]
    h = l @ l.T
    d, q = torch.linalg.eigh(h)
    orth = torch.linalg.matrix_norm(q.T@q-torch.eye(p, dtype=l.dtype), ord="fro")
    residual = torch.linalg.matrix_norm(h-(q*d)@q.T, ord="fro")
    # Conservative gamma budgets for reconstruction, certificate GEMMs,
    # norm reductions and the measured residual. Not fitted approximation errors.
    g = gamma(8*p)
    qnorm2 = q.square().sum()/(1-g)
    eta = (orth + g*(qnorm2+p))/(1-g)
    eps_h = (residual + g*(l.square().sum()/(1-g)+qnorm2*d[-1]+torch.linalg.matrix_norm(h, ord="fro")))/(1-g)
    # Polar Qbar is orthogonal, ||Q-Qbar|| <= eta/(1+sqrt(1-eta)).
    if eta >= 1 or d[0] <= 0:
        raise ArithmeticError("No positive spectral certificate")
    delta_q = eta/(1+torch.sqrt(1-eta))
    eps_h = eps_h + delta_q*(2+delta_q)*d[-1]
    if eps_h >= d[0]:
        raise ArithmeticError("Residual budget cannot certify positive inverse")
    inv_error = eps_h/(d[0]*(d[0]-eps_h)) + delta_q*(2+delta_q)/d[0]
    return dict(vectors=q[:, :rank].contiguous(), inverse=1/d[:rank],
        dmin=d[0], remainder_min=d[rank], dmax=d[-1], eta=eta, inverse_error=inv_error,
        metadata=dict(p=p, rank=rank, dmin=float(d[0]), d33=float(d[rank]), dmax=float(d[-1]),
            orthogonality_fro=float(orth), reconstruction_fro=float(residual),
            orthogonality_budget=float(eta), matrix_error_budget=float(eps_h), inverse_error_budget=float(inv_error)))


def quadratic_bounds(f, spec):
    """f: [batch,head,query,p]; spec tensors stacked over head."""
    p, k = f.shape[-1], spec["vectors"].shape[-1]
    norm = f.square().sum(-1)
    tlo, thi = norm/(1+gamma(2*p)), norm/(1-gamma(2*p))
    z = torch.einsum("bhqp,hpk->bhqk", f, spec["vectors"])
    energy = z.square().sum(-1)
    weighted = (z.square()*spec["inverse"][None, :, None]).sum(-1)
    eta = spec["eta"][None, :, None]
    # ||computed V^T f - V^T f||_2 <= gamma_p ||V||_F ||f||_2.
    dz = gamma(2*p)*torch.sqrt(k*(1+eta)*thi)
    de = 2*torch.sqrt(energy/(1-gamma(2*k)))*dz + dz.square() + gamma(2*k)*energy/(1-gamma(2*k))
    dw = de/spec["dmin"][None, :, None] + gamma(4*k)*weighted/(1-gamma(4*k))
    rlo = torch.clamp((1-eta)*tlo-(energy+de), min=0)
    rhi = torch.clamp((1+eta)*thi-torch.clamp(energy-de, min=0), min=0)
    correction = spec["inverse_error"][None, :, None]*thi
    lo = torch.clamp(weighted-dw+rlo/spec["dmax"][None, :, None]-correction, min=0)
    hi = weighted+dw+rhi/spec["remainder_min"][None, :, None]+correction
    # Final scalar operations get an outward budget too.
    pad = gamma(64)*(hi+weighted+correction+1)
    return torch.clamp(lo-pad, min=0), hi+pad


def stack_specs(items, device="cpu"):
    keys = ("vectors", "inverse", "dmin", "remainder_min", "dmax", "eta", "inverse_error")
    return {key: torch.stack([x[key] for x in items]).to(device) for key in keys}
