from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kv_cache import HSTUKVCache


@dataclass
class TargetAwareKVConfig:
    num_items: int
    hidden_size: int = 64
    temperature: float = 0.1
    input_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.num_items < 1 or self.hidden_size < 1 or self.temperature <= 0:
            raise ValueError("target-aware K/V dimensions differ")


class TargetAwareKV(nn.Module):
    def __init__(self, cfg: TargetAwareKVConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(cfg.num_items + 1, cfg.hidden_size, padding_idx=0)
        self.outcome_emb = nn.Embedding(2, cfg.hidden_size)
        self.query_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.key_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.value_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.context_norm = nn.LayerNorm(cfg.hidden_size)
        self.output = nn.Linear(cfg.hidden_size, 1)
        self.dropout = nn.Dropout(cfg.input_dropout)
        nn.init.normal_(self.item_emb.weight, std=cfg.hidden_size**-0.5)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        nn.init.eye_(self.query_proj.weight)
        nn.init.eye_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.output.bias)

    def _query(self, item_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_proj(self.dropout(self.item_emb(item_ids))), dim=-1)

    def _kv(self, item_ids: torch.Tensor, labels: torch.Tensor):
        item = self.dropout(self.item_emb(item_ids))
        key = F.normalize(self.key_proj(item), dim=-1)
        value = self.value_proj(item + self.outcome_emb(labels.long()))
        return key, value

    def compute_kv(
        self,
        item_ids: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> HSTUKVCache:
        key, value = self._kv(item_ids, labels)
        if lengths is not None:
            valid = torch.arange(item_ids.shape[1], device=item_ids.device).unsqueeze(0) < lengths.to(
                item_ids.device
            ).unsqueeze(1)
            key = key * valid.unsqueeze(-1)
            value = value * valid.unsqueeze(-1)
        return HSTUKVCache(key.unsqueeze(0), value.unsqueeze(0), item_ids.shape[1])

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        keep: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.cfg.temperature
        masked = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
        weights = torch.softmax(masked, dim=-1) * keep.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        context = torch.matmul(weights, value)
        return self.output(self.context_norm(context)).squeeze(-1)

    def forward(
        self,
        item_ids: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor | None = None,
        return_kv: bool = False,
    ):
        batch, sequence = item_ids.shape
        query = self._query(item_ids)
        key, value = self._kv(item_ids, labels)
        positions = torch.arange(sequence, device=item_ids.device)
        keep = positions.view(1, sequence, 1) > positions.view(1, 1, sequence)
        keep = keep.expand(batch, -1, -1)
        if lengths is not None:
            valid = positions.unsqueeze(0) < lengths.to(item_ids.device).unsqueeze(1)
            keep = keep & valid.unsqueeze(1) & valid.unsqueeze(2)
            key = key * valid.unsqueeze(-1)
            value = value * valid.unsqueeze(-1)
        logits = self._attend(query, key, value, keep)
        cache = HSTUKVCache(key.unsqueeze(0), value.unsqueeze(0), sequence) if return_kv else None
        return logits, cache

    def forward_with_cache(
        self,
        cache: HSTUKVCache,
        item_ids: torch.Tensor,
        labels: torch.Tensor,
    ):
        batch, suffix = item_ids.shape
        prefix = cache.seq_len
        query = self._query(item_ids)
        new_key, new_value = self._kv(item_ids, labels)
        key = torch.cat((cache.k[0], new_key), dim=1)
        value = torch.cat((cache.v[0], new_value), dim=1)
        keep = torch.ones(batch, suffix, prefix + suffix, dtype=torch.bool, device=item_ids.device)
        suffix_positions = torch.arange(suffix, device=item_ids.device)
        keep[:, :, prefix:] = suffix_positions.view(1, suffix, 1) > suffix_positions.view(
            1, 1, suffix
        )
        logits = self._attend(query, key, value, keep)
        return logits, HSTUKVCache(key.unsqueeze(0), value.unsqueeze(0), prefix + suffix)

    def empty_cache(self, batch: int, device: torch.device) -> HSTUKVCache:
        empty = torch.empty(1, batch, 0, self.cfg.hidden_size, device=device)
        return HSTUKVCache(empty, empty.clone(), 0)


@dataclass
class FeatureCrossKVConfig:
    num_items: int
    hidden_size: int = 32
    input_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.num_items < 1 or self.hidden_size < 1:
            raise ValueError("feature-cross K/V dimensions differ")


class FeatureCrossKV(nn.Module):
    def __init__(self, cfg: FeatureCrossKVConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.value_item_emb = nn.Embedding(cfg.num_items + 1, cfg.hidden_size, padding_idx=0)
        self.outcome_emb = nn.Embedding(2, cfg.hidden_size)
        self.value_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.output = nn.Sequential(
            nn.LayerNorm(2 * cfg.hidden_size + 2),
            nn.Linear(2 * cfg.hidden_size + 2, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, 1),
        )
        self.dropout = nn.Dropout(cfg.input_dropout)
        nn.init.normal_(self.value_item_emb.weight, std=cfg.hidden_size**-0.5)
        with torch.no_grad():
            self.value_item_emb.weight[0].zero_()
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.output[-1].bias)

    def _kv(self, item_ids: torch.Tensor, labels: torch.Tensor):
        key = item_ids.unsqueeze(-1).to(torch.float32)
        value = self.value_proj(
            self.dropout(self.value_item_emb(item_ids)) + self.outcome_emb(labels.long())
        )
        return key, value

    def compute_kv(
        self,
        item_ids: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> HSTUKVCache:
        key, value = self._kv(item_ids, labels)
        if lengths is not None:
            valid = torch.arange(item_ids.shape[1], device=item_ids.device).unsqueeze(0) < lengths.to(
                item_ids.device
            ).unsqueeze(1)
            key = key * valid.unsqueeze(-1)
            value = value * valid.unsqueeze(-1)
        return HSTUKVCache(key.unsqueeze(0), value.unsqueeze(0), item_ids.shape[1])

    def _score(
        self,
        query_items: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        keep: torch.Tensor,
    ) -> torch.Tensor:
        same = query_items.unsqueeze(-1) == key.squeeze(-1).unsqueeze(1).long()
        matched = keep & same
        global_count = keep.sum(dim=-1, keepdim=True)
        matched_count = matched.sum(dim=-1, keepdim=True)
        global_context = torch.matmul(keep.to(value.dtype), value) / global_count.clamp_min(1)
        matched_context = torch.matmul(matched.to(value.dtype), value) / matched_count.clamp_min(1)
        features = torch.cat(
            (
                matched_context,
                global_context,
                torch.log1p(matched_count.to(value.dtype)),
                torch.log1p(global_count.to(value.dtype)),
            ),
            dim=-1,
        )
        return self.output(features).squeeze(-1)

    def forward(
        self,
        item_ids: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor | None = None,
        return_kv: bool = False,
    ):
        batch, sequence = item_ids.shape
        key, value = self._kv(item_ids, labels)
        positions = torch.arange(sequence, device=item_ids.device)
        keep = positions.view(1, sequence, 1) > positions.view(1, 1, sequence)
        keep = keep.expand(batch, -1, -1)
        if lengths is not None:
            valid = positions.unsqueeze(0) < lengths.to(item_ids.device).unsqueeze(1)
            keep = keep & valid.unsqueeze(1) & valid.unsqueeze(2)
            key = key * valid.unsqueeze(-1)
            value = value * valid.unsqueeze(-1)
        logits = self._score(item_ids, key, value, keep)
        cache = HSTUKVCache(key.unsqueeze(0), value.unsqueeze(0), sequence) if return_kv else None
        return logits, cache

    def forward_with_cache(
        self,
        cache: HSTUKVCache,
        item_ids: torch.Tensor,
        labels: torch.Tensor,
    ):
        batch, suffix = item_ids.shape
        prefix = cache.seq_len
        new_key, new_value = self._kv(item_ids, labels)
        key = torch.cat((cache.k[0], new_key), dim=1)
        value = torch.cat((cache.v[0], new_value), dim=1)
        keep = torch.ones(batch, suffix, prefix + suffix, dtype=torch.bool, device=item_ids.device)
        positions = torch.arange(suffix, device=item_ids.device)
        keep[:, :, prefix:] = positions.view(1, suffix, 1) > positions.view(1, 1, suffix)
        logits = self._score(item_ids, key, value, keep)
        return logits, HSTUKVCache(key.unsqueeze(0), value.unsqueeze(0), prefix + suffix)

    def empty_cache(self, batch: int, device: torch.device) -> HSTUKVCache:
        key = torch.empty(1, batch, 0, 1, device=device)
        value = torch.empty(1, batch, 0, self.cfg.hidden_size, device=device)
        return HSTUKVCache(key, value, 0)
