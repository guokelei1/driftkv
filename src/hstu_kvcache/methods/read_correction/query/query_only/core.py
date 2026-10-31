"""Small query-only nonlinear feature bases with closed-form affine fitting.

ELU(query) is a feature hypothesis, not a factorization of native ELU(q @ k).
Neither candidate reads historical keys, values, or the native read response.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QueryFeatureCorrection(nn.Module):
    kind = "query_features_v5"

    def __init__(self, heads: int, head_dim: int, feature_mode: str = "head_phi"):
        super().__init__()
        if feature_mode not in ("head_phi", "cross_phi"):
            raise ValueError("feature_mode must be head_phi or cross_phi")
        self.heads, self.head_dim, self.feature_mode = heads, head_dim, feature_mode
        width = heads * head_dim
        feature_shape = (heads, head_dim) if feature_mode == "head_phi" else (2 * width,)
        weight_shape = (heads, head_dim, head_dim) if feature_mode == "head_phi" else (2 * width, width)
        output_shape = (heads, head_dim) if feature_mode == "head_phi" else (width,)
        self.register_buffer("input_mean", torch.zeros(feature_shape))
        self.register_buffer("input_scale", torch.ones(feature_shape))
        self.weight = nn.Parameter(torch.zeros(weight_shape))
        self.bias = nn.Parameter(torch.zeros(output_shape))

    def get_config(self):
        return {"heads": self.heads, "head_dim": self.head_dim, "feature_mode": self.feature_mode}

    def features(self, query):
        if self.feature_mode == "head_phi":
            return F.elu(query) + 1
        flat = query.transpose(1, 2).flatten(2)
        return torch.cat((flat, F.elu(flat)), dim=-1)

    def rate(self, query, keys=None, values=None, counts=None):
        features = self.features(query)
        if self.feature_mode == "head_phi":
            normalized = (features - self.input_mean[None, :, None]) / self.input_scale[None, :, None]
            return torch.einsum("bhqi,hio->bhqo", normalized, self.weight) + self.bias[None, :, None]
        normalized = (features - self.input_mean) / self.input_scale
        output = normalized @ self.weight + self.bias
        return output.reshape(query.shape[0], query.shape[2], self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query, keys, values, counts):
        return self.rate(query) * counts.to(query.dtype)[:, None, None, None]


@torch.no_grad()
def fit_feature_rule(query, target_rate, counts=None, *, feature_mode="head_phi", ridge=.01):
    """Fit train-only standardization and centered FP64 ridge, bias unpenalized."""
    if query.ndim != 4 or target_rate.shape != query.shape or ridge < 0:
        raise ValueError("matching [B,H,Q,d] inputs and nonnegative ridge required")
    if counts is not None:
        valid = counts > 0
        query, target_rate = query[valid], target_rate[valid]
    batch, heads, queries, dim = query.shape
    if batch * queries == 0:
        raise ValueError("nonempty fitting queries required")
    module = QueryFeatureCorrection(heads, dim, feature_mode).to(query)
    features = module.features(query.detach().double())
    target = target_rate.detach().double()
    if feature_mode == "head_phi":
        x = features.permute(1, 0, 2, 3).reshape(heads, -1, dim)
        y = target.permute(1, 0, 2, 3).reshape(heads, -1, dim)
    else:
        x = features.reshape(1, -1, 2 * heads * dim)
        y = target.transpose(1, 2).reshape(1, -1, heads * dim)
    if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise ValueError("nonfinite fitting inputs")
    mean, scale = x.mean(1), x.std(1, correction=0).clamp_min(1e-6)
    normalized = (x - mean[:, None]) / scale[:, None]
    bias = y.mean(1)
    centered = y - bias[:, None]
    if ridge == 0:
        weight = torch.linalg.lstsq(normalized, centered).solution
    else:
        gram = normalized.transpose(-2, -1) @ normalized / x.shape[1]
        penalty = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype) * ridge
        rhs = normalized.transpose(-2, -1) @ centered / x.shape[1]
        weight = torch.linalg.solve(gram + penalty, rhs)
    module.input_mean.copy_(mean.reshape_as(module.input_mean))
    module.input_scale.copy_(scale.reshape_as(module.input_scale))
    module.weight.copy_(weight.reshape_as(module.weight))
    module.bias.copy_(bias.reshape_as(module.bias))
    residual = normalized @ weight + bias[:, None] - y
    return module, {"feature_mode": feature_mode, "ridge": float(ridge), "users": batch,
        "queries": batch * queries, "feature_width": x.shape[-1], "output_width": y.shape[-1],
        "rate_mse": float(residual.square().mean()), "zero_rate_mse": float(y.square().mean()),
        "feature_scale_min": float(scale.min()), "feature_scale_max": float(scale.max()),
        "ridge_objective": "mean_query_squared_error_plus_ridge_slopes_only",
        "normalization_source": "fitting users only", "solve_dtype": "float64"}
