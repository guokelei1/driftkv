"""Draft nonlinear residual over the frozen cross-head token ridge map."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from hstu_kvcache.read_correction_v4.token_read import TokenReadCorrection


class NonlinearTokenReadCorrection(TokenReadCorrection):
    kind = "token_read_nonlinear_v4"

    def __init__(self, heads, head_dim, query_affine=False, nonlinear_width=None):
        super().__init__(heads, head_dim, query_affine=query_affine)
        width = heads * head_dim
        self.nonlinear_width = max(1, width // 2) if nonlinear_width is None else int(nonlinear_width)
        self.encoder = nn.Linear(2 * width, self.nonlinear_width)
        self.output = nn.Linear(self.nonlinear_width, 2 * width)
        self.register_buffer("residual_scale", torch.ones(2 * width))
        self.map_weight.requires_grad_(False)
        self.map_bias.requires_grad_(False)
        if self.query_correction is not None:
            self.query_correction.requires_grad_(False)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def initialize_base(self, base):
        self.load_state_dict(base.state_dict(), strict=False)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()
        return self

    def get_config(self):
        return dict(super().get_config(), nonlinear_width=self.nonlinear_width)

    def residual_delta(self, normalized_x):
        return self.output(F.silu(self.encoder(normalized_x))) * self.residual_scale

    def token_delta(self, normalized_x):
        return super().token_delta(normalized_x) + self.residual_delta(normalized_x)

    @torch.no_grad()
    def set_residual_scale(self, value):
        value = torch.as_tensor(value, device=self.residual_scale.device, dtype=self.residual_scale.dtype)
        if value.shape != self.residual_scale.shape or not torch.isfinite(value).all() or not (value > 0).all():
            raise ValueError("one finite positive residual unit per K/V coordinate is required")
        self.residual_scale.copy_(value)
