"""Exact source-conditioned225D quadratic, with FP64 numerical fallback bounds."""

import torch

from .geometry import coordinates
from .spectral import gamma


def prepare_inverse(layers, rank=33, query_dim=33):
    """Prepare shared inverse blocks without changing the frozen Cholesky matrix."""
    packed, checks = [], []
    for layer, l in enumerate(layers):
        heads, p, _ = l.shape
        eye = torch.eye(p, dtype=l.dtype, device=l.device).expand(heads, -1, -1)
        t = torch.linalg.solve_triangular(l, eye, upper=False)
        inverse = t.transpose(-2, -1)@t
        residual = l@t-eye
        # Exact real L and stored T satisfy LT=I+E. Bound E from computed
        # residual, product roundoff and norm reductions; rho<1 certifies T.
        g = gamma(8*p)
        tn = t.square().sum((-2, -1)).sqrt()/(1-g)
        ln = l.square().sum((-2, -1)).sqrt()/(1-g)
        rho = (residual.square().sum((-2, -1)).sqrt()+g*(ln*tn+p))/(1-g)
        if not torch.all(rho < 1):
            raise ArithmeticError("Cannot certify inverse preparation")
        product_error = g*tn.square()
        inv_norm = inverse.square().sum((-2,-1)).sqrt()/(1-g)
        # T=L^-1(I+E), hence H^-1=(I+E)^-T (T.T T) (I+E)^-1.
        # Use the norm of the computed Gram rather than the much looser ||T||F².
        inv_error = (inv_norm+product_error)*rho*(2-rho)/(1-rho).square()+product_error
        m = rank*query_dim
        aa = inverse[:, :m, :m]
        ao = inverse[:, :m, m:]
        oo = inverse[:, m:, m:]
        # Source-major feature order: [r,j,s,k] -> [r,s,head,j,k].
        aa_packed = aa.reshape(heads, rank, query_dim, rank, query_dim).permute(1, 3, 0, 2, 4)
        ao_packed = ao.reshape(heads, rank, query_dim, p-m).permute(1, 0, 2, 3)
        packed.append(dict(aa=aa_packed.reshape(rank*rank, heads*query_dim*query_dim).contiguous(),
            ao=ao_packed.reshape(rank, heads*query_dim*(p-m)).contiguous(), oo=oo.contiguous(),
            aa_norm=aa.square().sum((-2,-1)).sqrt()/(1-g),
            ao_norm=ao.square().sum((-2,-1)).sqrt()/(1-g),
            inverse_norm=inv_norm, inverse_error=inv_error,
            rank=rank, query_dim=query_dim))
        checks.extend(dict(layer=layer, head=h, residual_fro=float(residual[h].norm()),
            residual_budget=float(rho[h]), inverse_error_budget=float(inv_error[h]),
            inverse_fro=float(packed[-1]["inverse_norm"][h])) for h in range(heads))
    return packed, checks


def condition_state(x, pack):
    """Temporary matrices for one batch/layer, reused by all query candidates."""
    r, j = pack["rank"], pack["query_dim"]
    h, native, _ = pack["oo"].shape
    outer = (x[:, :, None]*x[:, None, :]).flatten(1)
    qq = (outer@pack["aa"]).reshape(len(x), h, j, j)
    qo = (x@pack["ao"]).reshape(len(x), h, j, native)
    oo = pack["oo"][None].expand(len(x), -1, -1, -1)
    matrix = torch.cat((torch.cat((qq, qo), -1), torch.cat((qo.transpose(-2,-1), oo), -1)), -2)
    x2 = x.square().sum(-1)/(1-gamma(2*r))
    qq_error = gamma(4*r*r)*x2[:, None]*pack["aa_norm"][None]
    qo_error = gamma(4*r)*x2.sqrt()[:, None]*pack["ao_norm"][None]
    norm = pack["inverse_norm"][None]*torch.maximum(x2[:, None], torch.ones_like(x2[:, None]))+qq_error+2*qo_error
    return dict(matrix=matrix, x2=x2, qq_error=qq_error, qo_error=qo_error, matrix_norm_bound=norm)


def evaluate_state(state, query, observed_rate, parameters, pack):
    q, o = coordinates(query, observed_rate, parameters)
    o = o[:, None].expand(-1, q.shape[1], -1, -1)
    z = torch.cat((q, o), -1)
    value = ((z@state["matrix"])*z).sum(-1)
    q2 = q.square().sum(-1)/(1-gamma(2*q.shape[-1]))
    o2 = o.square().sum(-1)/(1-gamma(2*o.shape[-1]))
    f2 = state["x2"][:, None, None]*q2+o2
    delta = pack["inverse_error"][None, :, None]*f2
    delta = delta+state["qq_error"][:, :, None]*q2+2*state["qo_error"][:, :, None]*(q2*o2).sqrt()
    delta = delta+gamma(8*z.shape[-1])*state["matrix_norm_bound"][:, :, None]*(q2+o2)
    delta = delta+gamma(64)*(value.abs()+delta+1)
    # A nonpositive/sign-uncertain point must fall back. Only the mathematical
    # lower endpoint may be bounded by zero; never clip the point estimate.
    valid = torch.isfinite(value) & torch.isfinite(delta) & (value > delta)
    lower = torch.where(valid, value-delta, torch.zeros_like(value))
    upper = torch.where(valid, value+delta, torch.full_like(value, torch.inf))
    point = torch.where(valid, value, torch.full_like(value, torch.nan))
    return point, lower, upper, valid


def arithmetic(queries):
    r, j, native, heads, layers = 33, 33, 192, 6, 6
    k, p = j+native, r*j+native
    state_gemm = layers*heads*(2*r*r*j*j+2*r*j*native)
    # x outer/norms and all state certificate scalar work, conservatively charged
    # per layer because the prototype recomputes them at that granularity.
    state_scalar = layers*(r*r+2*r+64*heads)
    query_gemm = queries*layers*heads*2*k*k
    # q/o normalization, quadratic dot and norm reductions, error/decision
    # guards and aggregation. 256 scalar allowance/unit covers remaining setup.
    query_scalar = queries*layers*(heads*(2*32+2*k-1+2*j-1+2*native-1+256)+2*native)
    return dict(state_gemm=state_gemm, state_scalar_allowance=state_scalar,
        query_gemm=query_gemm, query_scalar_allowance=query_scalar,
        panel_flops=state_gemm+state_scalar+query_gemm+query_scalar+2,
        exact_panel_flops=queries*layers*(heads*p*p+heads*(2*32+r*j+2*p-1+1)+2*native)+layers+2,
        state_matrix_bytes=layers*heads*k*k*8, layer_batch4_matrix_bytes=4*heads*k*k*8,
        scalar_scope="Conservative explicit scalar allowance; sqrt/comparisons/casts/memory traffic separate")
