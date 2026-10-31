"""Explicit extra per-query access to the complete retained user history."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..query_only import QueryCorrection


class HistoryCorrection(QueryCorrection):
    """rate = affine(q) + B mean_i SiLU(Wq q + Wk K_i + Wv V_i + c).

    Parameters are head-local and shared across users and history positions.
    No summary is persisted. Token/query chunks bound inference intermediates;
    training still retains activations for backward and needs bounded minibatches.
    """

    kind = "history_conditioned"

    def __init__(self, heads: int, head_dim: int, width: int = 32,
                 query_chunk: int = 8, token_chunk: int = 64):
        super().__init__(heads, head_dim)
        self.width, self.query_chunk, self.token_chunk = width, query_chunk, token_chunk
        self.register_buffer("k_mean", torch.zeros(heads, head_dim))
        self.register_buffer("k_scale", torch.ones(heads, head_dim))
        self.register_buffer("v_mean", torch.zeros(heads, head_dim))
        self.register_buffer("v_scale", torch.ones(heads, head_dim))
        self.query_projection = nn.Parameter(torch.empty(heads, head_dim, width))
        self.key_projection = nn.Parameter(torch.empty(heads, head_dim, width))
        self.value_projection = nn.Parameter(torch.empty(heads, head_dim, width))
        self.encoder_bias = nn.Parameter(torch.zeros(heads, width))
        self.history_weight = nn.Parameter(torch.zeros(heads, width, head_dim))
        for parameter in (self.query_projection, self.key_projection, self.value_projection):
            nn.init.normal_(parameter, std=1 / math.sqrt(3 * head_dim))

    def get_config(self):
        return dict(super().get_config(), width=self.width,
                    query_chunk=self.query_chunk, token_chunk=self.token_chunk)

    @torch.no_grad()
    def initialize_query(self, query_module):
        for name in ("q_mean", "q_scale", "weight", "bias"):
            getattr(self, name).copy_(getattr(query_module, name))
        self.history_weight.zero_()
        return self

    def _heads(self, tensor):
        return tensor.reshape(tensor.shape[0], tensor.shape[1], self.heads, self.head_dim).transpose(1, 2)

    @torch.no_grad()
    def set_history_normalization(self, keys, values, counts=None):
        """Fit normalization only from calibration histories; padding is excluded."""
        n = keys.shape[1]
        if counts is None:
            counts = torch.full((keys.shape[0],), n, device=keys.device)
        valid = torch.arange(n, device=keys.device)[None, :] < counts[:, None]
        if not bool(valid.any()):
            raise ValueError("normalization requires a nonempty calibration history")
        for name, tensor in (("k", keys), ("v", values)):
            # Select [valid_tokens,heads,dim], avoiding the padding contribution.
            selected = self._heads(tensor).transpose(1, 2)[valid].double()
            getattr(self, name + "_mean").copy_(selected.mean(0))
            getattr(self, name + "_scale").copy_(selected.std(0, correction=0).clamp_min(1e-6))
        return self

    def rate(self, query, keys, values, counts=None):
        base = super().rate(query)
        batch, _, num_queries, _ = query.shape
        tokens = keys.shape[1]
        if tokens == 0 or num_queries == 0:
            return base
        if counts is None:
            counts = torch.full((batch,), tokens, device=query.device)
        valid = torch.arange(tokens, device=query.device)[None, :] < counts[:, None]
        k = (self._heads(keys) - self.k_mean[None, :, None]) / self.k_scale[None, :, None]
        v = (self._heads(values) - self.v_mean[None, :, None]) / self.v_scale[None, :, None]
        projected = (torch.einsum("bhnd,hdw->bhnw", k, self.key_projection)
                     + torch.einsum("bhnd,hdw->bhnw", v, self.value_projection))
        projected_q = (torch.einsum("bhqd,hdw->bhqw", self.normalize_query(query), self.query_projection)
                       + self.encoder_bias[None, :, None])
        chunks = []
        for q_start in range(0, num_queries, self.query_chunk):
            q_part = projected_q[:, :, q_start:q_start + self.query_chunk]
            total = torch.zeros_like(q_part)
            for t_start in range(0, tokens, self.token_chunk):
                t_end = t_start + self.token_chunk
                features = F.silu(q_part.unsqueeze(-2) + projected[:, :, None, t_start:t_end])
                mask = valid[:, None, None, t_start:t_end, None]
                total = total + (features * mask).sum(-2)
            mean = total / counts.clamp_min(1).to(query.dtype)[:, None, None, None]
            chunks.append(torch.einsum("bhqw,hwd->bhqd", mean, self.history_weight))
        return base + torch.cat(chunks, dim=2)
