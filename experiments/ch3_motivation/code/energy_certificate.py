"""Draft, CPU-tested energy certificate; not imported by frozen E6 runners.

For orthonormal Q and B=Q.T@M, C_r=sum(s(B)[:r]**2), R=||B||_F**2,
T=||M||_F**2 satisfy C_r <= exact_top_r(M) <= C_r + T - R.
Thus C_r/(C_r+T-R) lower-bounds the existing E6 captured-energy ratio.

The numerical version below is conditional on explicit scalar energy error
and orthogonality error bounds. It is a conservative numerical guard, not an
interval-arithmetic proof about arbitrary floating-point SVD implementations.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def sketch_energy_certificate(total_energy, all_sketch_singular_values, *,
        ranks=(2, 6, 16), relative_energy_tolerance=1e-5,
        orthogonality_defect=0., threshold=.98):
    """Return batched certificates without a per-matrix CPU synchronization.

    total_energy has shape [...]; all_sketch_singular_values has shape [...,k]
    and contains the ENTIRE small SVD spectrum, before truncation to rank16.
    Accumulation of its squares happens in float64 on these tiny tensors.

    Let a=relative_energy_tolerance*T. We assume the supplied total energy,
    each prefix energy C_r, and full sketch energy R each differ from their
    exact represented-matrix values by <=a. orthogonality_defect must upper-
    bound ||Q.T@Q-I||_2, including any error in its measurement; zero assumes
    exact orthogonality. These assumptions must be checked at integration.

    With delta<1, the true orthogonal projection has top-r energy at most
    (C_r+a)/(1-delta), and residual energy at most
    (T+a)-max(R-a,0)/(1+delta). The returned ratio uses max(C_r-a,0) over
    the sum of those upper bounds. For zero error and delta=0 it reduces to
    C_r/(C_r+max(T-R,0)). Invalid inputs fail the certificate rather than pass.

    The target is the E6 reported sketch singular-energy / exact top-r energy,
    not a bound on downstream logit/AUC error. Exact zero matrices have no
    nonzero target energy and receive the explicit ratio convention 1.
    """
    if (not isinstance(total_energy, torch.Tensor)
            or not isinstance(all_sketch_singular_values, torch.Tensor)
            or all_sketch_singular_values.ndim < 1
            or all_sketch_singular_values.shape[:-1] != total_energy.shape
            or total_energy.device != all_sketch_singular_values.device
            or all_sketch_singular_values.shape[-1] < 1):
        raise ValueError("energy and complete sketch spectrum must have matching batch shapes")
    if (not ranks or any(not isinstance(rank, int) or rank <= 0 for rank in ranks)
            or relative_energy_tolerance < 0 or not 0 < threshold <= 1):
        raise ValueError("positive ranks, nonnegative energy tolerance and valid threshold required")
    total = total_energy.to(torch.float64)
    singular = all_sketch_singular_values.to(torch.float64)
    delta = torch.as_tensor(orthogonality_defect, device=total.device, dtype=torch.float64)
    delta = torch.broadcast_to(delta, total.shape)
    squared = singular.square()
    cumulative = squared.cumsum(-1)
    range_energy = cumulative[..., -1]
    slack = relative_energy_tolerance * total.abs()
    safe_delta = delta.clamp(0, 1 - torch.finfo(torch.float64).eps)
    residual_upper = (total + slack - (range_energy - slack).clamp_min(0) / (1 + safe_delta)).clamp_min(0)
    valid = (torch.isfinite(total) & (total >= 0)
        & torch.isfinite(singular).all(-1) & (singular >= 0).all(-1)
        & (singular[..., 1:] <= singular[..., :-1]).all(-1)
        & torch.isfinite(delta) & (delta >= 0) & (delta < 1)
        & (range_energy <= (1 + safe_delta) * (total + slack) + slack))
    zero = (total == 0) & (range_energy == 0)
    captured = torch.stack([cumulative[..., min(rank, singular.shape[-1]) - 1] for rank in ranks], -1)
    numerator = (captured - slack[..., None]).clamp_min(0)
    denominator = (captured + slack[..., None]) / (1 - safe_delta[..., None]) + residual_upper[..., None]
    tiny = torch.finfo(torch.float64).tiny
    bound = numerator / denominator.clamp_min(tiny)
    bound = torch.where(zero[..., None], torch.ones_like(bound), bound)
    bound = torch.where(valid[..., None], bound.clamp(0, 1), torch.zeros_like(bound))
    passes = valid[..., None] & (bound >= threshold)
    return dict(ranks=tuple(ranks), range_energy=range_energy,
        residual_range_energy_estimate=(total - range_energy).clamp_min(0),
        residual_range_energy_upper_bound=residual_upper, energy_error_allowance=slack,
        captured_energy=captured, captured_energy_ratio_lower_bound=bound,
        passes=passes, needs_refinement=~passes.all(-1), inputs_valid=valid,
        exact_zero_matrix=zero & valid, threshold=threshold,
        relative_energy_tolerance=relative_energy_tolerance,
        orthogonality_defect_upper_bound=delta)
