"""Full-rank K/V read view for the next Design 1 probe.

Fitting supplies a centered, standardized-input ridge residual with its
intercept unpenalized. Parameters below are already in physical K/V units.
This shares the causal row-context and read-view lifecycle of ContextKV32.
"""

import torch
from torch import nn

from hstu_kvcache.design_one.nonlinear import ContextKVReadViewAdapter


class AffineKVViewLayer(nn.Module):
    """[K,V] + normalize([K,V,is_parent,birth_fraction]) @ weight + bias."""

    def __init__(self, heads, head_dim):
        super().__init__()
        self.heads, self.head_dim = int(heads), int(head_dim)
        self.width = self.heads * self.head_dim
        inputs, outputs = 2 * self.width + 2, 2 * self.width
        self.register_buffer("input_center", torch.zeros(inputs))
        self.register_buffer("input_scale", torch.ones(inputs))
        self.register_buffer("map_weight", torch.zeros(inputs, outputs))
        self.register_buffer("map_bias", torch.zeros(outputs))

    def get_config(self):
        return dict(heads=self.heads, head_dim=self.head_dim)

    @torch.no_grad()
    def set_parameters(self, center, scale, weight, bias):
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("input normalization scales must be finite and positive")
        self.input_center.copy_(center)
        self.input_scale.copy_(scale)
        self.map_weight.copy_(weight)
        self.map_bias.copy_(bias)

    def map_tokens(self, k, v, context):
        source = torch.cat((k, v), dim=-1)
        features = torch.cat((source, context.to(source)), dim=-1)
        normalized = (features.to(self.input_center.dtype) - self.input_center) / self.input_scale
        residual = normalized @ self.map_weight + self.map_bias
        mapped = source + residual.to(source.dtype)
        return mapped.split(self.width, dim=-1)

    def token_forward_flops(self, *, batch=1, length=1):
        """Normalization, full dense map, bias and residual add; multiply-add=2.

        The fitted output units have already been folded into map_weight/bias.
        Context bookkeeping, teacher capture and the ridge solve are separate.
        """
        inputs, outputs = self.map_weight.shape
        return int(batch * length * (2 * inputs + 2 * inputs * outputs + 2 * outputs))


class AffineKVReadViewAdapter(ContextKVReadViewAdapter):
    """Full-rank affine maps with the existing immutable-context read lifecycle.

    map_cache, make_history_override, estimate_flops, export_state and
    from_state_dict share the contextual adapter's implementations. Old
    neural adapters and their checkpoint layouts are unchanged.
    """

    kind = "design_one_affine_kv_view_v1"
    layer_class = AffineKVViewLayer

    def __init__(self, num_layers, heads, head_dim, max_length=1024):
        nn.Module.__init__(self)
        self.num_layers, self.heads, self.head_dim = int(num_layers), int(heads), int(head_dim)
        self.max_length = int(max_length)
        self.layers = nn.ModuleList([
            self.layer_class(heads, head_dim) for _ in range(num_layers)
        ])

    def get_config(self):
        return dict(num_layers=self.num_layers, heads=self.heads,
                    head_dim=self.head_dim, max_length=self.max_length)
