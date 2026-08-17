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
    num_prediction_items: int | None = None
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
    block_variant: str = "legacy"
    relative_position_bias: bool = False
    causal_diagonal: str = "inclusive"

    def __post_init__(self) -> None:
        if self.num_items < 1:
            raise ValueError("num_items must be positive")
        if self.num_prediction_items is None:
            self.num_prediction_items = self.num_items
        if not 1 <= self.num_prediction_items <= self.num_items:
            raise ValueError("num_prediction_items must be in [1, num_items]")
        if self.block_variant not in ("legacy", "hstu_reference"):
            raise ValueError("block_variant differs")
        if self.block_variant == "hstu_reference" and (
            self.gating != "silu_gate" or self.activation != "silu"
        ):
            raise ValueError("hstu_reference block configuration differs")
        if self.causal_diagonal not in ("inclusive", "exclusive"):
            raise ValueError("causal_diagonal differs")


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
        self.output_emb = (
            None
            if cfg.tie_item_embeddings
            else ItemEmbedding(int(cfg.num_prediction_items), h)
        )
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
            max_seq_len=cfg.max_seq_len,
            block_variant=cfg.block_variant,
            relative_position_bias=cfg.relative_position_bias,
            causal_diagonal=cfg.causal_diagonal,
        )
        return HSTUBlock(
            HSTUBlockConfig(
                attn=attn_cfg,
                gating=cfg.gating,
                norm_eps=cfg.norm_eps,
                block_variant=cfg.block_variant,
            )
        )

    def embed_inputs(
        self,
        item_ids: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
    ) -> torch.Tensor:
        return self.combine_input_features(
            self.lookup_item_embeddings(item_ids),
            behaviors,
            time_deltas,
        )

    def lookup_item_embeddings(
        self,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.item_emb(item_ids)

    def combine_input_features(
        self,
        item_vectors: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
    ) -> torch.Tensor:
        if item_vectors.shape[:-1] != behaviors.shape:
            raise ValueError("item vectors and behaviors shapes differ")
        if time_deltas.shape != behaviors.shape:
            raise ValueError("time deltas and behaviors shapes differ")
        if item_vectors.shape[-1] != self.cfg.hidden_size:
            raise ValueError("item vector width differs from model hidden size")
        x = item_vectors + self.behavior_emb(behaviors) + self.temporal_enc(time_deltas)
        x = self.in_proj(x)
        return self.input_dropout(x)

    def forward_embedded(
        self,
        x: torch.Tensor,
        return_kv: bool = False,
        return_hidden: bool = True,
        lengths: torch.Tensor | None = None,
        first_layer_residual_reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, HSTUKVCache | None]:
        if x.ndim != 3 or x.shape[-1] != self.cfg.hidden_size:
            raise ValueError("embedded inputs must have shape [batch, sequence, hidden]")
        L = x.shape[1]
        valid = None
        if lengths is not None:
            if lengths.shape != (x.shape[0],):
                raise ValueError("lengths and embedded batch dimension differ")
            lengths = lengths.to(x.device)
            valid = torch.arange(L, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            x = x * valid.unsqueeze(-1)
        reset_residual = None
        if first_layer_residual_reset_mask is not None:
            if first_layer_residual_reset_mask.shape != x.shape[:2]:
                raise ValueError("first-layer residual reset mask shape differs")
            reset_residual = x * first_layer_residual_reset_mask.to(
                device=x.device, dtype=x.dtype
            ).unsqueeze(-1)
        kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, blk in enumerate(self.blocks):
            if return_kv:
                x, (k, v) = blk(x, attn_mask=None, return_kv=True)
                kvs.append((k, v))
            else:
                x = blk(x, attn_mask=None, return_kv=False)
            if layer == 0 and reset_residual is not None:
                x = x - reset_residual
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

    @torch.no_grad()
    def compute_kv_from_item_embeddings(
        self,
        item_vectors: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> HSTUKVCache:
        was_training = self.training
        self.eval()
        try:
            _, kv = self.forward_embedded(
                self.combine_input_features(
                    item_vectors,
                    behaviors,
                    time_deltas,
                ),
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
        return self.score_hidden(last, candidate_ids)

    @property
    def prediction_item_weight(self) -> torch.Tensor:
        if self.output_emb is None:
            return self.item_emb.weight
        return self.output_emb.weight

    def score_hidden(
        self,
        hidden: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.output_emb is None:
            return self.item_emb.score(hidden, candidate_ids)
        return self.output_emb.score(hidden, candidate_ids)

    @torch.no_grad()
    def forward_with_cache_embedded(
        self,
        cached_kv: HSTUKVCache,
        x: torch.Tensor,
        first_layer_residual_reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, HSTUKVCache]:
        if x.ndim != 3 or x.shape[-1] != self.cfg.hidden_size:
            raise ValueError("embedded suffix must have shape [batch, sequence, hidden]")
        if cached_kv.k.shape[1] != x.shape[0]:
            raise ValueError("cached K/V and embedded suffix batch dimensions differ")
        reset_residual = None
        if first_layer_residual_reset_mask is not None:
            if first_layer_residual_reset_mask.shape != x.shape[:2]:
                raise ValueError("first-layer residual reset mask shape differs")
            reset_residual = x * first_layer_residual_reset_mask.to(
                device=x.device, dtype=x.dtype
            ).unsqueeze(-1)
        m = x.shape[1]
        new_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for li, blk in enumerate(self.blocks):
            cached_k = cached_kv.k[li]
            cached_v = cached_kv.v[li]
            x, (k_all, v_all) = blk.forward_with_cache(x, cached_k, cached_v)
            if li == 0 and reset_residual is not None:
                x = x - reset_residual
            new_kvs.append((k_all, v_all))
        x = self.final_norm(x)
        updated_kv = HSTUKVCache.from_layer_list(
            new_kvs,
            seq_len=cached_kv.seq_len + m,
        )
        return x, updated_kv

    @torch.no_grad()
    def forward_with_cache_embedded_new_kv(
        self,
        cached_kv: HSTUKVCache,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, HSTUKVCache]:
        if x.ndim != 3 or x.shape[-1] != self.cfg.hidden_size:
            raise ValueError("embedded suffix must have shape [batch, sequence, hidden]")
        if cached_kv.k.shape[1] != x.shape[0]:
            raise ValueError("cached K/V and embedded suffix batch dimensions differ")
        new_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, block in enumerate(self.blocks):
            x, new_kv = block.forward_with_cache_new_kv(
                x,
                cached_kv.k[layer],
                cached_kv.v[layer],
            )
            new_kvs.append(new_kv)
        x = self.final_norm(x)
        return x, HSTUKVCache.from_layer_list(
            new_kvs,
            seq_len=x.shape[1],
        )

    @torch.no_grad()
    def forward_with_cache_from_item_embeddings(
        self,
        cached_kv: HSTUKVCache,
        new_item_vectors: torch.Tensor,
        new_behaviors: torch.Tensor,
        new_time_deltas: torch.Tensor,
    ) -> tuple[torch.Tensor, HSTUKVCache]:
        was_training = self.training
        self.eval()
        try:
            return self.forward_with_cache_embedded(
                cached_kv,
                self.combine_input_features(
                    new_item_vectors,
                    new_behaviors,
                    new_time_deltas,
                ),
            )
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def forward_with_cache_from_item_embeddings_new_kv(
        self,
        cached_kv: HSTUKVCache,
        new_item_vectors: torch.Tensor,
        new_behaviors: torch.Tensor,
        new_time_deltas: torch.Tensor,
    ) -> tuple[torch.Tensor, HSTUKVCache]:
        was_training = self.training
        self.eval()
        try:
            return self.forward_with_cache_embedded_new_kv(
                cached_kv,
                self.combine_input_features(
                    new_item_vectors,
                    new_behaviors,
                    new_time_deltas,
                ),
            )
        finally:
            if was_training:
                self.train()

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
            return self.forward_with_cache_embedded(
                cached_kv,
                self.embed_inputs(
                    new_item_ids,
                    new_behaviors,
                    new_time_deltas,
                ),
            )
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
