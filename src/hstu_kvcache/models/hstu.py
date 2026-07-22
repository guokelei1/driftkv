from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .block import HSTUBlock, HSTUBlockConfig
from .embeddings import BehaviorEncoder, ItemEmbedding, TemporalEncoder
from .kv_cache import HSTUKVCache


@dataclass
class HSTUConfig:
    num_items: int
    num_behaviors: int
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 2
    head_dim: int | None = None
    max_seq_len: int = 2048
    temporal_num_freqs: int = 16
    temporal_max_period: float = 86400.0
    gating: str = "silu_gate"
    qk_scale: float = 1.0
    attn_dropout: float = 0.0
    activation: str = "elu_plus1"
    norm_eps: float = 1e-6
    input_dropout: float = 0.1
    tie_item_embeddings: bool = True


class HSTU(nn.Module):
    """Hierarchical Sequential Transductive Unit (Zhai et al., ICML'24).

    Faithful re-implementation of the defining features, modular for research:
      * Pointwise aggregated attention (elu+1, unnormalised) - in PointwiseAttention.
      * RMSNorm, no absolute positional embedding (temporal time-deltas instead).
      * FFN merged into attention via pointwise gating - in HSTUBlock.
      * Tied item-embedding output head.

    ``forward`` returns hidden states; ``compute_kv`` returns the batched
    derived prefix K/V cache F(theta, x), which can be captured under one
    model version and migrated for use by another.
    """

    def __init__(self, cfg: HSTUConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_size

        self.item_emb = ItemEmbedding(cfg.num_items, h)
        self.behavior_emb = BehaviorEncoder(cfg.num_behaviors, h)
        self.temporal_enc = TemporalEncoder(h, cfg.temporal_num_freqs, cfg.temporal_max_period)
        self.in_proj = nn.Linear(h, h, bias=False)
        self.input_dropout = nn.Dropout(cfg.input_dropout)

        self.blocks = nn.ModuleList(
            [self._make_block(cfg, h) for _ in range(cfg.num_layers)]
        )
        self.final_norm = nn.LayerNorm(h) if False else _RMSNormOrLn(h, cfg.norm_eps)

    @staticmethod
    def _make_block(cfg: HSTUConfig, h: int) -> HSTUBlock:
        from .attention import PointwiseAttentionConfig

        attn_cfg = PointwiseAttentionConfig(
            hidden_size=h,
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            qk_scale=cfg.qk_scale,
            attn_dropout=cfg.attn_dropout,
            activation=cfg.activation,
        )
        return HSTUBlock(HSTUBlockConfig(attn=attn_cfg, gating=cfg.gating, norm_eps=cfg.norm_eps))

    def embed_inputs(
        self,
        item_ids: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
    ) -> torch.Tensor:
        x = self.item_emb(item_ids) + self.behavior_emb(behaviors) + self.temporal_enc(time_deltas)
        x = self.in_proj(x)
        return self.input_dropout(x)

    def forward(
        self,
        item_ids: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
        return_kv: bool = False,
        return_hidden: bool = True,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, HSTUKVCache | None]:
        """Run the transducer.

        Args:
            item_ids/behaviors: [B, L] long, padded with 0.
            time_deltas: [B, L] float seconds, 0 for first event / padding.
            return_kv: also return the derived KV cache F(theta, x_u).
            return_hidden: return final hidden states (set False to save memory
                when only KV is needed).

        Returns (hidden_or_zeros, kv_cache_or_none).
        """
        x = self.embed_inputs(item_ids, behaviors, time_deltas)
        L = x.shape[1]
        valid = None
        if lengths is not None:
            lengths = lengths.to(x.device)
            valid = torch.arange(L, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            x = x * valid.unsqueeze(-1)
        kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for blk in self.blocks:
            if return_kv:
                x, (k, v) = blk(x, attn_mask=None, return_kv=True)
                kvs.append((k, v))
            else:
                x = blk(x, attn_mask=None, return_kv=False)
            if valid is not None:
                x = x * valid.unsqueeze(-1)
        x = self.final_norm(x)
        if valid is not None:
            x = x * valid.unsqueeze(-1)
        kv = None
        if return_kv:
            kv = HSTUKVCache.from_layer_list(kvs, seq_len=L)
        if not return_hidden:
            x = torch.empty(0, device=x.device)
        return x, kv

    @torch.no_grad()
    def compute_kv(
        self,
        item_ids: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> HSTUKVCache:
        """Convenience: return only the derived KV cache F(theta, x_u).

        Deterministic: forces eval mode so dropout does not perturb the cached
        representation (KV must be a pure function of theta and x_u).
        """
        was_training = self.training
        self.eval()
        try:
            _, kv = self.forward(
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
        assert kv is not None
        return kv

    def last_hidden(
        self,
        hidden: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lengths is None:
            return hidden[:, -1, :]
        lengths = lengths.to(hidden.device)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, (lengths - 1).clamp_min(0), :]

    def score_candidates(
        self,
        hidden: torch.Tensor,
        candidate_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """hidden [B, L, H] (use last position) -> scores [B, C]."""
        last = self.last_hidden(hidden, lengths)
        return self.item_emb.score(last, candidate_ids)

    @torch.no_grad()
    def forward_with_cache(
        self,
        cached_kv: HSTUKVCache,
        new_item_ids: torch.Tensor,
        new_behaviors: torch.Tensor,
        new_time_deltas: torch.Tensor,
    ) -> tuple[torch.Tensor, HSTUKVCache]:
        """Incremental inference: reuse prefix KV (possibly from theta_old) + new suffix tokens.

        This is the core KV-cache-reuse operation:
          * Prefix [1..n] KV was cached (maybe by theta_old) -> reused as-is.
          * Suffix [n+1..n+m] is embedded + processed with the CURRENT model (theta_new).
          * At each layer, Q/gating/norm use theta_new; prefix K,V stay from the cache.

        Args:
            cached_kv: HSTUKVCache with K,V [num_layers, B, n, inner] from a prior forward.
            new_item_ids/behaviors/time_deltas: [B, m] for the new suffix tokens.

        Returns: hidden [B, m, H] for new positions, updated_kv HSTUKVCache [B, n+m, inner].
        """
        was_training = self.training
        self.eval()
        try:
            B, m = new_item_ids.shape
            x = self.embed_inputs(new_item_ids, new_behaviors, new_time_deltas)  # [B, m, H]
            new_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
            for li, blk in enumerate(self.blocks):
                cached_k = cached_kv.k[li]  # [B, n, inner]
                cached_v = cached_kv.v[li]
                x, (k_all, v_all) = blk.forward_with_cache(x, cached_k, cached_v)
                new_kvs.append((k_all, v_all))
            x = self.final_norm(x)
            updated_kv = HSTUKVCache.from_layer_list(new_kvs, seq_len=cached_kv.seq_len + m)
            return x, updated_kv
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def forward_stale_kv(
        self,
        item_ids: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
        stale_kv: HSTUKVCache,
    ) -> torch.Tensor:
        """Full forward with entirely stale K,V (from theta_old).

        Most extreme staleness: Q/gating/norm/embeddings from current model,
        but K,V at EVERY layer from stale_kv (cached under theta_old).
        """
        was_training = self.training
        self.eval()
        try:
            x = self.embed_inputs(item_ids, behaviors, time_deltas)
            for li, blk in enumerate(self.blocks):
                x = blk.forward_stale_kv(x, stale_kv.k[li], stale_kv.v[li])
            x = self.final_norm(x)
            return x
        finally:
            if was_training:
                self.train()


def _RMSNormOrLn(h: int, eps: float) -> nn.Module:
    from .rmsnorm import RMSNorm

    return RMSNorm(h, eps=eps)
