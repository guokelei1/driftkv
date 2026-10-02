"""Per-head affine correction of native historical attention responses."""

from __future__ import annotations

import torch
from torch import nn


class QueryCorrection(nn.Module):
    """Predict response/N from q, then scale by the current history length.

    q is [batch, heads, queries, head_dim]. K/V are intentionally unused.
    The normalization is fixed from calibration, so this remains affine in q.
    """

    kind = "query_only"

    def __init__(self, heads: int, head_dim: int):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.register_buffer("q_mean", torch.zeros(heads, head_dim))
        self.register_buffer("q_scale", torch.ones(heads, head_dim))
        self.weight = nn.Parameter(torch.zeros(heads, head_dim, head_dim))
        self.bias = nn.Parameter(torch.zeros(heads, head_dim))

    def get_config(self):
        return {"heads": self.heads, "head_dim": self.head_dim}

    def normalize_query(self, query):
        return (query - self.q_mean[None, :, None]) / self.q_scale[None, :, None]

    def rate(self, query, keys=None, values=None, counts=None):
        normalized = self.normalize_query(query)
        return torch.einsum("bhqd,hde->bhqe", normalized, self.weight) + self.bias[None, :, None]

    def forward(self, query, keys, values, counts):
        return self.rate(query, keys, values, counts) * counts.to(query.dtype)[:, None, None, None]


def fit_query(query, target_rate, counts=None, ridge: float = 0.01):
    """FP64 ridge with an unpenalized intercept and equal query weights.

    Targets already equal (teacher history read - reused history read)/N.
    ``counts`` may remove empty histories; it never divides the target again.
    Objective: mean squared residual per output + ridge * squared slopes.
    Returned module has the input's device/dtype; statistics are plain scalars.
    """
    if query.shape != target_rate.shape or query.ndim != 4:
        raise ValueError("query and target_rate must both have shape [B,H,Q,d]")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    if counts is not None:
        valid = counts > 0
        query, target_rate = query[valid], target_rate[valid]
    batch, heads, queries, dim = query.shape
    if batch * queries == 0:
        raise ValueError("ridge fit requires nonempty histories and queries")
    x = query.detach().double().permute(1, 0, 2, 3).reshape(heads, -1, dim)
    y = target_rate.detach().double().permute(1, 0, 2, 3).reshape(heads, -1, dim)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("nonfinite calibration inputs")
    mean = x.mean(1)
    scale = x.std(1, correction=0).clamp_min(1e-6)
    normalized = (x - mean[:, None]) / scale[:, None]
    design = torch.cat([normalized, torch.ones_like(normalized[..., :1])], dim=-1)
    gram = design.transpose(-2, -1) @ design / design.shape[1]
    penalty = torch.eye(dim + 1, dtype=x.dtype, device=x.device) * ridge
    penalty[-1, -1] = 0
    rhs = design.transpose(-2, -1) @ y / design.shape[1]
    if ridge == 0:
        fitted = torch.linalg.lstsq(design, y).solution
    else:
        fitted = torch.linalg.solve(gram + penalty, rhs)
    module = QueryCorrection(heads, dim).to(device=query.device, dtype=query.dtype)
    with torch.no_grad():
        module.q_mean.copy_(mean)
        module.q_scale.copy_(scale)
        module.weight.copy_(fitted[:, :-1])
        module.bias.copy_(fitted[:, -1])
    residual = design @ fitted - y
    stats = {
        "ridge": float(ridge), "users": batch, "queries": batch * queries,
        "rate_mse": float(residual.square().mean()),
        "zero_rate_mse": float(y.square().mean()),
        "q_scale_min": float(scale.min()), "q_scale_max": float(scale.max()),
        "ridge_objective": "mean_query_squared_error_plus_ridge_slopes_only",
        "solve_dtype": "float64",
    }
    return module, stats
