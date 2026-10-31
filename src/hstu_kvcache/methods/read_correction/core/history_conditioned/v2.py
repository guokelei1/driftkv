"""Full-history correction with cross-head, query-conditioned token interaction.

This is a more expressive exploratory branch, separate from the retained v1
encoder. Each candidate explicitly re-encodes its complete legal history; no
state is persisted and the supplied K/V cache is never changed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .core import HistoryCorrection


class HistoryCorrectionV2(nn.Module):
    """Keep a fitted v1 rule and add a query-conditioned history Transformer.

    The extra encoder mixes all K/V heads, incorporates position before SiLU,
    and applies one bidirectional self-attention/FFN block per query. All its
    tokens precede the current request, so bidirectional history access is
    causal with respect to the request. A masked mean feeds a zero-initialized
    response-rate projection, preserving the supplied base at initialization.
    """

    kind = "history_conditioned_v2"

    def __init__(self, heads: int, head_dim: int, encoder_width: int | None = None,
                 attention_heads: int = 4, query_chunk: int = 1,
                 base_width: int = 32, base_token_chunk: int = 256,
                 base_query_chunk: int = 4, freeze_base: bool = True):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.width = heads * head_dim
        self.encoder_width = encoder_width or max(1, self.width // 2)
        self.attention_heads = max(a for a in range(1, min(attention_heads, self.encoder_width) + 1)
                                   if self.encoder_width % a == 0)
        if query_chunk < 1:
            raise ValueError("query_chunk must be positive")
        self.query_chunk, self.freeze_base = query_chunk, freeze_base
        self.base = HistoryCorrection(heads, head_dim, width=base_width,
                                      query_chunk=base_query_chunk, token_chunk=base_token_chunk)
        self.base.requires_grad_(not freeze_base)
        r = self.encoder_width
        self.key_projection = nn.Linear(self.width, r)
        self.value_projection = nn.Linear(self.width, r)
        self.query_projection = nn.Linear(self.width, r)
        self.position_projection = nn.Linear(2, r)
        self.attention_norm = nn.LayerNorm(r)
        self.qkv_projection = nn.Linear(r, 3 * r)
        self.attention_output = nn.Linear(r, r)
        self.feedforward_norm = nn.LayerNorm(r)
        self.feedforward_in = nn.Linear(r, 2 * r)
        self.feedforward_out = nn.Linear(2 * r, r)
        self.output = nn.Linear(r, self.width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("output_scale", torch.ones(heads, head_dim))

    def get_config(self):
        return {
            "heads": self.heads, "head_dim": self.head_dim,
            "encoder_width": self.encoder_width, "attention_heads": self.attention_heads,
            "query_chunk": self.query_chunk, "base_width": self.base.width,
            "base_token_chunk": self.base.token_chunk, "base_query_chunk": self.base.query_chunk,
            "freeze_base": self.freeze_base,
        }

    @torch.no_grad()
    def initialize_history(self, module: HistoryCorrection):
        """Copy a fitted v1 model; its parameters and buffers remain untouched."""
        self.base.load_state_dict(module.state_dict())
        self.output.weight.zero_()
        self.output.bias.zero_()
        return self

    @torch.no_grad()
    def initialize_query(self, module):
        self.base.initialize_query(module)
        self.output.weight.zero_()
        self.output.bias.zero_()
        return self

    @torch.no_grad()
    def set_history_normalization(self, keys, values, counts=None):
        self.base.set_history_normalization(keys, values, counts)
        return self

    @torch.no_grad()
    def set_output_scale(self, scale):
        """Calibration residual units; loss can operate in normalized units.

        Accept [heads,head_dim] or one RMS per head. This scale stays in the
        artifact and is applied to the extra branch only, leaving v1 unchanged.
        """
        scale = torch.as_tensor(scale, device=self.output_scale.device, dtype=self.output_scale.dtype)
        if scale.ndim == 1 and scale.numel() == self.heads:
            scale = scale[:, None].expand(-1, self.head_dim)
        if scale.shape != self.output_scale.shape or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("output_scale must contain finite positive calibration units")
        self.output_scale.copy_(scale)
        return self

    def _encode(self, tokens, valid):
        """[user*query,history,r], boolean valid history keys."""
        batch, length, width = tokens.shape
        projected = self.qkv_projection(self.attention_norm(tokens))
        q, k, v = projected.reshape(batch, length, 3, self.attention_heads,
                                    width // self.attention_heads).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :],
                                                  dropout_p=0., is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.attention_output(attended)
        tokens = tokens + self.feedforward_out(F.gelu(self.feedforward_in(self.feedforward_norm(tokens))))
        return tokens

    def residual_rate(self, query, keys, values, counts=None):
        """Extra branch alone, in physical response/N units."""
        batch, _, queries, _ = query.shape
        length = keys.shape[1]
        if length == 0 or queries == 0:
            return torch.zeros_like(query)
        if counts is None:
            counts = torch.full((batch,), length, device=query.device)
        index = torch.arange(length, device=query.device)
        valid = index[None] < counts[:, None]
        kn = (keys - self.base.k_mean.reshape(1, 1, -1)) / self.base.k_scale.reshape(1, 1, -1)
        vn = (values - self.base.v_mean.reshape(1, 1, -1)) / self.base.v_scale.reshape(1, 1, -1)
        kn = kn.masked_fill(~valid[..., None], 0.)
        vn = vn.masked_fill(~valid[..., None], 0.)
        # Oldest=0/newest=1; position refers only to this retained prefix.
        position = index.to(query.dtype)[None] / (counts - 1).clamp_min(1).to(query.dtype)[:, None]
        position = position.masked_fill(~valid, 0.)
        position_input = torch.stack((position, position.square()), dim=-1)
        history_projection = (self.key_projection(kn) + self.value_projection(vn)
                              + self.position_projection(position_input))
        qn = self.base.normalize_query(query).transpose(1, 2).reshape(batch, queries, self.width)
        query_projection = self.query_projection(qn)
        outputs = []
        for start in range(0, queries, self.query_chunk):
            qp = query_projection[:, start:start + self.query_chunk]
            chunk = qp.shape[1]
            tokens = F.silu(history_projection[:, None] + qp[:, :, None])
            flat_valid = valid[:, None].expand(batch, chunk, length).reshape(batch * chunk, length)
            encoded = self._encode(tokens.reshape(batch * chunk, length, self.encoder_width), flat_valid)
            encoded = encoded.reshape(batch, chunk, length, self.encoder_width)
            pooled = encoded.masked_fill(~valid[:, None, :, None], 0.).sum(2)
            pooled = pooled / counts.clamp_min(1).to(query.dtype)[:, None, None]
            rate = self.output(pooled).reshape(batch, chunk, self.heads, self.head_dim).transpose(1, 2)
            rate = rate * self.output_scale[None, :, None]
            outputs.append(rate.masked_fill((counts == 0)[:, None, None, None], 0.))
        return torch.cat(outputs, dim=2)

    def rate(self, query, keys, values, counts=None):
        return self.base.rate(query, keys, values, counts) + self.residual_rate(query, keys, values, counts)

    def forward(self, query, keys, values, counts):
        return self.rate(query, keys, values, counts) * counts.to(query.dtype)[:, None, None, None]
