from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no bias, no mean-subtraction).

    HSTU uses RMSNorm in place of LayerNorm. Eps is applied outside the sqrt
    following the common HSTU/LLaMA convention so renormalisation is exact.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute in float32 for stability, cast back
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x.to(dtype)) * self.weight
