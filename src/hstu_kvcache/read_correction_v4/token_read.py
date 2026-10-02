"""Transient, cross-head affine token mapping followed by the native read.

The mapped K/V exist only during this request. An optional old-producer prefix
gate leaves later Current-produced cache rows untouched; those rows are not
claimed to equal an exact full-history recomputation.
"""

from __future__ import annotations

import torch
from torch import nn

from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.read_correction.query_only import QueryCorrection


class TokenReadCorrection(nn.Module):
    """[Khat,Vhat] = [K,V] + affine(normalize([K,V])), per history token.

    Mapping mixes all heads. The optional query affine rule is an additional
    response/N residual after reading the mapped tokens; it starts at zero.
    """

    kind = "token_read_v4"

    def __init__(self, heads: int, head_dim: int, query_affine: bool = False):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        width = 2 * heads * head_dim
        self.register_buffer("input_mean", torch.zeros(width))
        self.register_buffer("input_scale", torch.ones(width))
        self.map_weight = nn.Parameter(torch.zeros(width, width))
        self.map_bias = nn.Parameter(torch.zeros(width))
        self.query_correction = QueryCorrection(heads, head_dim) if query_affine else None

    def get_config(self):
        return {"heads": self.heads, "head_dim": self.head_dim,
                "query_affine": self.query_correction is not None}

    def token_delta(self, normalized_x):
        """Token residual hook; stronger isolated candidates may override it."""
        return normalized_x @ self.map_weight + self.map_bias

    def map_tokens(self, keys, values, counts, old_counts=None):
        """Dense mapping, with a valid-prefix/producer gate on its residual.

        K/V are [B,N,D]; counts and old_counts are [B]. The first counts rows
        are valid, and the first old_counts valid rows are eligible for mapping.
        All N padded rows still execute the affine arithmetic before gating.
        """
        joined = torch.cat((keys, values), dim=-1)
        normalized = (joined - self.input_mean) / self.input_scale
        residual = self.token_delta(normalized)
        positions = torch.arange(keys.shape[1], device=keys.device)[None]
        eligible = positions < counts[:, None]
        if old_counts is not None:
            eligible = eligible & (positions < old_counts[:, None])
        mapped = joined + residual * eligible[..., None].to(residual.dtype)
        return mapped.split(self.heads * self.head_dim, dim=-1)

    def forward_new_read(self, query, keys, values, counts, attention, old_counts=None):
        """One new history read; never recompute the supplied native old read."""
        if attention.block_variant != "legacy" or attention.activation != "elu_plus1":
            raise ValueError("v4 probe currently supports the retained legacy ELU+1 checkpoints")
        mapped_k, mapped_v = self.map_tokens(keys, values, counts, old_counts)
        valid = torch.arange(keys.shape[1], device=keys.device)[None] < counts[:, None]
        # A mapped padding value could otherwise contribute under ELU+1.
        # This also makes padding harmless when callers supply nonzero padding.
        heads = history_read(attention, query, mapped_k, mapped_v, count=valid)
        if self.query_correction is not None:
            heads = heads + self.query_correction(query, keys, values, counts)
        return heads


def fit_affine_tokens(x, y, ridge: float = 1e-3):
    """Fit teacher-minus-old token residuals using centered FP64 ridge.

    x and y have shape [M,2D]. Mean/scale use these fitting rows only. The
    objective is mean squared residual per output + ridge * squared normalized
    slopes; centering leaves the intercept unpenalized. Parameters are returned
    on the input device/dtype and can be copied directly into the module.
    """
    if x.ndim != 2 or x.shape != y.shape or not x.shape[0]:
        raise ValueError("x/y must have the same nonempty [M,2D] shape")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    xd, yd = x.detach().double(), y.detach().double()
    if not torch.isfinite(xd).all() or not torch.isfinite(yd).all():
        raise ValueError("nonfinite token calibration inputs")
    mean = xd.mean(0)
    scale = xd.std(0, correction=0).clamp_min(1e-6)
    normalized = (xd - mean) / scale
    bias = yd.mean(0)
    centered_y = yd - bias
    if ridge == 0:
        weight = torch.linalg.lstsq(normalized, centered_y).solution
    else:
        gram = normalized.T @ normalized / len(xd)
        penalty = torch.eye(xd.shape[1], dtype=xd.dtype, device=xd.device) * ridge
        weight = torch.linalg.solve(gram + penalty, normalized.T @ centered_y / len(xd))
    error = normalized @ weight + bias - yd
    params = {"input_mean": mean.to(x), "input_scale": scale.to(x),
              "map_weight": weight.to(x), "map_bias": bias.to(x)}
    stats = {"rows": len(xd), "input_width": x.shape[1], "ridge": float(ridge),
             "solve_dtype": "float64", "residual_mse": float(error.square().mean()),
             "zero_residual_mse": float(yd.square().mean()),
             "input_scale_min": float(scale.min()), "input_scale_max": float(scale.max()),
             "objective": "mean squared token-residual error plus ridge on normalized slopes; centered unpenalized intercept"}
    return params, stats


def score_token_corrected(model, cache, candidates, query_delta, modules, counts,
                          old_counts=None, *, trace=False):
    """Return (logits, ReadResult), replacing only each transient history read.

    The native reader supplies the old response to the override. We calculate
    one mapped response and return it directly; native history is not read a
    second time. Traced corrections subtract the already supplied native heads.
    None modules retain their native read, useful for sequential calibration.
    """
    if len(modules) != len(model.blocks):
        raise ValueError("one token correction or None is required per layer")
    corrections = [] if trace else None

    def override(layer, query, native_history):
        module = modules[layer]
        if module is None:
            if trace:
                corrections.append(torch.zeros_like(native_history))
            return native_history
        new_read = module.forward_new_read(query, cache.k[layer], cache.v[layer], counts,
                                           model.blocks[layer].attn, old_counts)
        if trace:
            corrections.append(new_read - native_history)
        return new_read

    logits, result = score(model, cache, candidates, query_delta, trace=trace, history_override=override)
    if trace:
        result.corrections = tuple(corrections)
    return logits, result
