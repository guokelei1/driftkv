from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import BehaviorEncoder, ItemEmbedding, TemporalEncoder
from .kv_cache import HSTUKVCache


@dataclass
class DenseHSTUV2Config:
    num_items: int
    num_behaviors: int
    num_prediction_items: int | None = None
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 2
    head_dim: int | None = None
    max_seq_len: int = 2048
    temporal_num_freqs: int = 16
    temporal_max_period: float = 86400.0
    input_dropout: float = 0.1
    output_dropout: float = 0.0
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.num_items < 1:
            raise ValueError("num_items must be positive")
        if self.num_prediction_items is None:
            self.num_prediction_items = self.num_items
        if not 1 <= self.num_prediction_items <= self.num_items:
            raise ValueError("num_prediction_items must be in [1, num_items]")
        if self.hidden_size < 1 or self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("model dimensions must be positive")
        if self.head_dim is None:
            if self.hidden_size % self.num_heads:
                raise ValueError("hidden_size must be divisible by num_heads")
            self.head_dim = self.hidden_size // self.num_heads
        if self.num_heads * self.head_dim != self.hidden_size:
            raise ValueError("dense_hstu_v2 requires num_heads * head_dim == hidden_size")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")


class DenseHSTUV2Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        output_dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.attn_alpha = head_dim**-0.5
        self.input_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.uvqk = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.output_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.output = nn.Linear(3 * hidden_size, hidden_size, bias=False)
        self.output_dropout = nn.Dropout(output_dropout)
        nn.init.xavier_uniform_(self.uvqk.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def _project(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u, v, q, k = self.uvqk(self.input_norm(x)).chunk(4, dim=-1)
        return F.silu(u), v, q, k

    def _shape(self, value: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = value.shape
        return value.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

    def _finish(
        self,
        residual: torch.Tensor,
        u: torch.Tensor,
        attention: torch.Tensor,
    ) -> torch.Tensor:
        product = u * self.output_norm(attention)
        update = self.output(torch.cat((u, attention, product), dim=-1))
        return residual + self.output_dropout(update)

    def forward(
        self,
        x: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, sequence, _ = x.shape
        u, v_flat, q_flat, k_flat = self._project(x)
        q = self._shape(q_flat)
        k = self._shape(k_flat)
        v = self._shape(v_flat)
        weights = F.silu(torch.matmul(q, k.transpose(-2, -1)) * self.attn_alpha)
        keep = torch.ones(sequence, sequence, device=x.device, dtype=torch.bool).tril()
        keep = keep.view(1, 1, sequence, sequence)
        if valid is not None:
            keep = keep & valid.view(batch, 1, 1, sequence)
            keep = keep & valid.view(batch, 1, sequence, 1)
        weights = weights * keep.to(weights.dtype) / self.max_seq_len
        attention = torch.matmul(weights, v)
        attention = attention.transpose(1, 2).reshape(batch, sequence, self.hidden_size)
        output = self._finish(x, u, attention)
        if valid is not None:
            output = output * valid.unsqueeze(-1)
            k_flat = k_flat * valid.unsqueeze(-1)
            v_flat = v_flat * valid.unsqueeze(-1)
        return output, (k_flat, v_flat)

    def forward_with_cache(
        self,
        x_new: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, suffix, _ = x_new.shape
        prefix = cached_k.shape[1]
        u, v_new_flat, q_flat, k_new_flat = self._project(x_new)
        q = self._shape(q_flat)
        k_new = self._shape(k_new_flat)
        v_new = self._shape(v_new_flat)
        k_cached = self._shape(cached_k)
        v_cached = self._shape(cached_v)
        k_all = torch.cat((k_cached, k_new), dim=2)
        v_all = torch.cat((v_cached, v_new), dim=2)
        weights = F.silu(torch.matmul(q, k_all.transpose(-2, -1)) * self.attn_alpha)
        keep = torch.ones(suffix, prefix + suffix, device=x_new.device, dtype=torch.bool)
        keep[:, prefix:] = torch.ones(
            suffix,
            suffix,
            device=x_new.device,
            dtype=torch.bool,
        ).tril()
        weights = weights * keep.view(1, 1, suffix, prefix + suffix) / self.max_seq_len
        attention = torch.matmul(weights, v_all)
        attention = attention.transpose(1, 2).reshape(batch, suffix, self.hidden_size)
        output = self._finish(x_new, u, attention)
        return output, (
            k_all.transpose(1, 2).reshape(batch, prefix + suffix, self.hidden_size),
            v_all.transpose(1, 2).reshape(batch, prefix + suffix, self.hidden_size),
        )


class DenseHSTUV2(nn.Module):
    def __init__(self, cfg: DenseHSTUV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden_size
        assert cfg.head_dim is not None
        self.item_emb = ItemEmbedding(cfg.num_items, hidden)
        self.behavior_emb = BehaviorEncoder(cfg.num_behaviors, hidden)
        self.temporal_enc = TemporalEncoder(
            hidden,
            cfg.temporal_num_freqs,
            cfg.temporal_max_period,
        )
        self.position_emb = nn.Embedding(cfg.max_seq_len, hidden)
        nn.init.normal_(self.position_emb.weight, std=0.02)
        self.in_proj = nn.Linear(hidden, hidden, bias=False)
        self.input_dropout = nn.Dropout(cfg.input_dropout)
        self.blocks = nn.ModuleList(
            [
                DenseHSTUV2Block(
                    hidden,
                    cfg.num_heads,
                    cfg.head_dim,
                    cfg.max_seq_len,
                    cfg.output_dropout,
                    cfg.norm_eps,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden, eps=cfg.norm_eps)

    def embed_inputs(
        self,
        item_ids: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        if item_ids.shape != behaviors.shape or item_ids.shape != time_deltas.shape:
            raise ValueError("input feature shapes differ")
        sequence = item_ids.shape[1]
        if position_offset < 0 or position_offset + sequence > self.cfg.max_seq_len:
            raise ValueError("position range exceeds max_seq_len")
        positions = torch.arange(
            position_offset,
            position_offset + sequence,
            device=item_ids.device,
        )
        x = (
            self.item_emb(item_ids)
            + self.behavior_emb(behaviors)
            + self.temporal_enc(time_deltas)
            + self.position_emb(positions).unsqueeze(0)
        )
        return self.input_dropout(self.in_proj(x))

    def forward_embedded(
        self,
        x: torch.Tensor,
        return_kv: bool = False,
        return_hidden: bool = True,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, HSTUKVCache | None]:
        if x.ndim != 3 or x.shape[-1] != self.cfg.hidden_size:
            raise ValueError("embedded inputs must have shape [batch, sequence, hidden]")
        sequence = x.shape[1]
        valid = None
        if lengths is not None:
            if lengths.shape != (x.shape[0],):
                raise ValueError("lengths and batch dimension differ")
            lengths = lengths.to(x.device)
            valid = torch.arange(sequence, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            x = x * valid.unsqueeze(-1)
        kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.blocks:
            x, kv = block(x, valid)
            kvs.append(kv)
        x = self.final_norm(x)
        if valid is not None:
            x = x * valid.unsqueeze(-1)
        cache = HSTUKVCache.from_layer_list(kvs, seq_len=sequence) if return_kv else None
        if not return_hidden:
            x = torch.empty(0, device=x.device)
        return x, cache

    def forward(
        self,
        item_ids: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
        return_kv: bool = False,
        return_hidden: bool = True,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, HSTUKVCache | None]:
        return self.forward_embedded(
            self.embed_inputs(item_ids, behaviors, time_deltas),
            return_kv=return_kv,
            return_hidden=return_hidden,
            lengths=lengths,
        )

    @torch.no_grad()
    def compute_kv(
        self,
        item_ids: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> HSTUKVCache:
        was_training = self.training
        self.eval()
        try:
            _, cache = self.forward(
                item_ids,
                behaviors,
                time_deltas,
                return_kv=True,
                return_hidden=False,
                lengths=lengths,
            )
        finally:
            if was_training:
                self.train()
        assert cache is not None
        return cache

    def last_hidden(
        self,
        hidden: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lengths is None:
            return hidden[:, -1]
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, (lengths.to(hidden.device) - 1).clamp_min(0)]

    def score_candidates(
        self,
        hidden: torch.Tensor,
        candidate_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        normalize: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        last = self.last_hidden(hidden, lengths)
        candidates = self.item_emb.weight[candidate_ids]
        if normalize:
            last = F.normalize(last, dim=-1)
            candidates = F.normalize(candidates, dim=-1)
        return torch.einsum("...h,...ch->...c", last, candidates) / temperature

    @torch.no_grad()
    def forward_with_cache_embedded(
        self,
        cached_kv: HSTUKVCache,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, HSTUKVCache]:
        if x.ndim != 3 or x.shape[-1] != self.cfg.hidden_size:
            raise ValueError("embedded suffix must have shape [batch, sequence, hidden]")
        if cached_kv.k.shape[1] != x.shape[0]:
            raise ValueError("cached K/V and suffix batch dimensions differ")
        kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, block in enumerate(self.blocks):
            x, kv = block.forward_with_cache(x, cached_kv.k[layer], cached_kv.v[layer])
            kvs.append(kv)
        x = self.final_norm(x)
        return x, HSTUKVCache.from_layer_list(
            kvs,
            seq_len=cached_kv.seq_len + x.shape[1],
        )

    @torch.no_grad()
    def forward_with_cache(
        self,
        cached_kv: HSTUKVCache,
        new_item_ids: torch.Tensor,
        new_behaviors: torch.Tensor,
        new_time_deltas: torch.Tensor,
    ) -> tuple[torch.Tensor, HSTUKVCache]:
        was_training = self.training
        self.eval()
        try:
            embedded = self.embed_inputs(
                new_item_ids,
                new_behaviors,
                new_time_deltas,
                position_offset=cached_kv.seq_len,
            )
            return self.forward_with_cache_embedded(cached_kv, embedded)
        finally:
            if was_training:
                self.train()
