from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .block import HSTUBlock, HSTUBlockConfig
from .embeddings import BehaviorEncoder, ItemEmbedding, QueryTokenEncoder, TemporalEncoder
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
    # CC query fields are independent from behavior/PAD/MASK embeddings.
    num_query_types: int = 1
    num_query_actions: int = 1
    query_type_id: int = 0
    query_action_id: int = 0

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
        if self.num_query_types < 1 or not 0 <= self.query_type_id < self.num_query_types:
            raise ValueError("query_type_id must index the query type table")
        if self.num_query_actions < 1 or not 0 <= self.query_action_id < self.num_query_actions:
            raise ValueError("query_action_id must index the reserved query action table")


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
        self.query_encoder = QueryTokenEncoder(
            h,
            num_query_types=cfg.num_query_types,
            num_query_actions=cfg.num_query_actions,
        )
        self.temporal_enc = TemporalEncoder(h, cfg.temporal_num_freqs, cfg.temporal_max_period)
        self.in_proj = nn.Linear(h, h, bias=False)
        self.input_dropout = nn.Dropout(cfg.input_dropout)
        # CC scores must not recover candidate identity from a second lookup
        # at the output head. The candidate enters through the transient query
        # token; this shared scalar head makes the item-embedding ablation a
        # meaningful shortcut audit.
        self.cc_score_head = nn.Linear(h, 1, bias=True)

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

    def embed_query_tokens(
        self,
        candidate_ids: torch.Tensor,
        query_time_deltas: torch.Tensor,
        query_type_ids: torch.Tensor | None = None,
        query_action_ids: torch.Tensor | None = None,
        item_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build transient CC query tokens for ``[batch, candidates]``.

        The candidate item, shared query type, reserved query action and query
        time are the only inputs. No future label, organic/played-ratio
        feature, or persistent-cache write is performed here.
        """
        if candidate_ids.ndim != 2:
            raise ValueError("candidate_ids must have shape [batch, candidates]")
        batch, candidates = candidate_ids.shape
        if item_vectors is None:
            item_vectors = self.lookup_item_embeddings(candidate_ids)
        if item_vectors.shape != (batch, candidates, self.cfg.hidden_size):
            raise ValueError("item_vectors must have shape [B, C, hidden]")

        def _expand_ids(
            values: torch.Tensor | None,
            default: int,
            name: str,
        ) -> torch.Tensor:
            if values is None:
                return torch.full(
                    (batch, candidates),
                    default,
                    dtype=torch.long,
                    device=candidate_ids.device,
                )
            values = values.to(device=candidate_ids.device, dtype=torch.long)
            if values.shape == (batch,):
                return values[:, None].expand(batch, candidates)
            if values.shape == (batch, candidates):
                return values
            raise ValueError(f"{name} must have shape [B] or [B, C]")

        query_time_deltas = query_time_deltas.to(
            device=candidate_ids.device,
            dtype=item_vectors.dtype,
        )
        if query_time_deltas.shape == (batch,):
            query_time_deltas = query_time_deltas[:, None].expand(batch, candidates)
        elif query_time_deltas.shape != (batch, candidates):
            raise ValueError("query_time_deltas must have shape [B] or [B, C]")
        query_type_ids = _expand_ids(query_type_ids, self.cfg.query_type_id, "query_type_ids")
        query_action_ids = _expand_ids(
            query_action_ids,
            self.cfg.query_action_id,
            "query_action_ids",
        )
        x = self.query_encoder(
            item_vectors,
            query_type_ids,
            query_action_ids,
            self.temporal_enc(query_time_deltas),
        )
        return self.input_dropout(self.in_proj(x))

    def forward_embedded(
        self,
        x: torch.Tensor,
        return_kv: bool = False,
        return_hidden: bool = True,
        lengths: torch.Tensor | None = None,
        first_layer_residual_reset_mask: torch.Tensor | None = None,
        residual_scale: float = 1.0,
        attention_scale: float = 1.0,
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
                x, (k, v) = blk(
                    x,
                    attn_mask=None,
                    return_kv=True,
                    residual_scale=residual_scale,
                    attention_scale=attention_scale,
                )
                kvs.append((k, v))
            else:
                x = blk(
                    x,
                    attn_mask=None,
                    return_kv=False,
                    residual_scale=residual_scale,
                    attention_scale=attention_scale,
                )
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
        residual_scale: float = 1.0,
        attention_scale: float = 1.0,
    ) -> tuple[torch.Tensor, HSTUKVCache | None]:
        return self.forward_embedded(
            self.embed_inputs(item_ids, behaviors, time_deltas),
            return_kv=return_kv,
            return_hidden=return_hidden,
            lengths=lengths,
            residual_scale=residual_scale,
            attention_scale=attention_scale,
        )

    @torch.no_grad()
    def compute_kv(
        self,
        item_ids: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
        lengths: torch.Tensor | None = None,
        residual_scale: float = 1.0,
        attention_scale: float = 1.0,
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
                residual_scale=residual_scale,
                attention_scale=attention_scale,
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

    @staticmethod
    def _cc_lengths(
        lengths: torch.Tensor | None,
        batch: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        if lengths is None:
            return torch.full((batch,), width, dtype=torch.long, device=device)
        lengths = lengths.to(device=device, dtype=torch.long)
        if lengths.shape != (batch,):
            raise ValueError("prefix lengths must have shape [B]")
        if bool((lengths < 0).any()) or bool((lengths > width).any()):
            raise ValueError("prefix lengths must be between zero and the padded width")
        return lengths

    def _observe_cc_from_prefix_cache(
        self,
        prefix_kv: HSTUKVCache,
        candidate_ids: torch.Tensor,
        query_time_deltas: torch.Tensor,
        prefix_lengths: torch.Tensor | None = None,
        query_type_ids: torch.Tensor | None = None,
        query_action_ids: torch.Tensor | None = None,
        candidate_item_vectors: torch.Tensor | None = None,
        *,
        training_append: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return scores and pre-head query readouts for a prefix cache."""
        if candidate_ids.ndim != 2:
            raise ValueError("candidate_ids must have shape [B, C]")
        batch, candidates = candidate_ids.shape
        if candidates < 1:
            raise ValueError("candidate_ids must contain at least one candidate")
        if prefix_kv.k.ndim != 4 or prefix_kv.v.shape != prefix_kv.k.shape:
            raise ValueError("prefix_kv must contain [layers, B, L, width] K/V tensors")
        if prefix_kv.k.shape[1] != batch:
            raise ValueError("prefix cache and candidates have different batch sizes")
        width = prefix_kv.k.shape[2]
        lengths = self._cc_lengths(prefix_lengths, batch, width, candidate_ids.device)
        if prefix_kv.seq_len < width:
            raise ValueError("prefix cache seq_len is smaller than its tensor width")
        if candidate_item_vectors is not None:
            if candidate_item_vectors.shape != (
                batch,
                candidates,
                self.cfg.hidden_size,
            ):
                raise ValueError("candidate_item_vectors must have shape [B, C, hidden]")
            candidate_item_vectors = candidate_item_vectors.to(
                device=candidate_ids.device,
                dtype=prefix_kv.k.dtype,
            )

        # Normalise once so every grouped append uses exactly the same query
        # timestamp contract. ``embed_query_tokens`` performs the detailed
        # [B]/[B,C] validation and broadcasting.
        query_time_deltas = query_time_deltas.to(device=candidate_ids.device)
        if query_time_deltas.shape == (batch,):
            query_time_deltas = query_time_deltas[:, None].expand(batch, candidates)
        elif query_time_deltas.shape != (batch, candidates):
            raise ValueError("query_time_deltas must have shape [B] or [B, C]")

        # Grouping by true prefix length avoids changing relative-position
        # distances for right-padded users. Within each group, [B,C] is
        # flattened to B*C one-token sequences, so candidates never attend to
        # one another.
        scores = None
        readouts = None
        unique_lengths = torch.unique(lengths, sorted=True).tolist()
        for length_value in unique_lengths:
            length = int(length_value)
            rows = torch.nonzero(lengths == length, as_tuple=False).flatten()
            group_candidates = candidate_ids.index_select(0, rows)
            group_times = query_time_deltas.index_select(0, rows)
            group_types = (
                None
                if query_type_ids is None
                else query_type_ids.index_select(0, rows)
            )
            group_actions = (
                None
                if query_action_ids is None
                else query_action_ids.index_select(0, rows)
            )
            group_item_vectors = (
                None
                if candidate_item_vectors is None
                else candidate_item_vectors.index_select(0, rows)
            )
            group_k = prefix_kv.k.index_select(1, rows)[:, :, :length, :]
            group_v = prefix_kv.v.index_select(1, rows)[:, :, :length, :]
            group_cache = HSTUKVCache(
                k=group_k.repeat_interleave(candidates, dim=1),
                v=group_v.repeat_interleave(candidates, dim=1),
                seq_len=length,
            )
            query = self.embed_query_tokens(
                group_candidates,
                group_times,
                query_type_ids=group_types,
                query_action_ids=group_actions,
                item_vectors=group_item_vectors,
            ).reshape(-1, 1, self.cfg.hidden_size)
            if training_append:
                hidden, _ = self.forward_with_cache_embedded_grad(group_cache, query)
            else:
                hidden, _ = self.forward_with_cache_embedded(group_cache, query)
            group_scores = self.cc_score_head(hidden[:, 0, :]).squeeze(-1)
            group_scores = group_scores.reshape(rows.numel(), candidates)
            group_readouts = hidden[:, 0, :].reshape(
                rows.numel(), candidates, self.cfg.hidden_size
            )
            if scores is None:
                scores = group_scores.new_empty((batch, candidates))
                readouts = group_readouts.new_empty(
                    (batch, candidates, self.cfg.hidden_size)
                )
            scores = scores.index_copy(0, rows, group_scores)
            readouts = readouts.index_copy(0, rows, group_readouts)
        assert scores is not None and readouts is not None
        return scores, readouts

    def _score_cc_from_prefix_cache(
        self,
        prefix_kv: HSTUKVCache,
        candidate_ids: torch.Tensor,
        query_time_deltas: torch.Tensor,
        prefix_lengths: torch.Tensor | None = None,
        query_type_ids: torch.Tensor | None = None,
        query_action_ids: torch.Tensor | None = None,
        candidate_item_vectors: torch.Tensor | None = None,
        *,
        training_append: bool,
    ) -> torch.Tensor:
        scores, _ = self._observe_cc_from_prefix_cache(
            prefix_kv,
            candidate_ids,
            query_time_deltas,
            prefix_lengths=prefix_lengths,
            query_type_ids=query_type_ids,
            query_action_ids=query_action_ids,
            candidate_item_vectors=candidate_item_vectors,
            training_append=training_append,
        )
        return scores

    def score_cc_full(
        self,
        prefix_item_ids: torch.Tensor,
        prefix_behaviors: torch.Tensor,
        prefix_time_deltas: torch.Tensor,
        candidate_ids: torch.Tensor,
        query_time_deltas: torch.Tensor,
        lengths: torch.Tensor | None = None,
        query_type_ids: torch.Tensor | None = None,
        query_action_ids: torch.Tensor | None = None,
        candidate_item_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Current-Full CC conditional reranking scores ``[B, C]``.

        The current model computes each user's prefix KV once. Each candidate
        is then appended as a one-token transient query and the resulting KV
        is discarded, so the persistent prefix remains unchanged.
        """
        if prefix_item_ids.ndim != 2:
            raise ValueError("prefix_item_ids must have shape [B, L]")
        if prefix_behaviors.shape != prefix_item_ids.shape:
            raise ValueError("prefix behaviors and items have different shapes")
        if prefix_time_deltas.shape != prefix_item_ids.shape:
            raise ValueError("prefix time deltas and items have different shapes")
        if candidate_ids.shape[0] != prefix_item_ids.shape[0]:
            raise ValueError("prefix and candidate batches differ")
        _, current_prefix = self.forward(
            prefix_item_ids,
            prefix_behaviors,
            prefix_time_deltas,
            return_kv=True,
            return_hidden=False,
            lengths=lengths,
        )
        assert current_prefix is not None
        return self._score_cc_from_prefix_cache(
            current_prefix,
            candidate_ids,
            query_time_deltas,
            prefix_lengths=lengths,
            query_type_ids=query_type_ids,
            query_action_ids=query_action_ids,
            candidate_item_vectors=candidate_item_vectors,
            training_append=torch.is_grad_enabled(),
        )

    def score_cc_full_chunked(
        self,
        prefix_item_ids: torch.Tensor,
        prefix_behaviors: torch.Tensor,
        prefix_time_deltas: torch.Tensor,
        candidate_ids: torch.Tensor,
        query_time_deltas: torch.Tensor,
        *,
        chunk_size: int,
        lengths: torch.Tensor | None = None,
        query_type_ids: torch.Tensor | None = None,
        query_action_ids: torch.Tensor | None = None,
        candidate_item_vectors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Score a complete candidate universe in exact independent chunks.

        The prefix is encoded exactly once. Returned chunks preserve candidate
        order and can be concatenated or passed to the streaming listwise-loss
        helper. The method never caps or samples candidates.
        """
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if prefix_item_ids.ndim != 2:
            raise ValueError("prefix_item_ids must have shape [B, L]")
        if prefix_behaviors.shape != prefix_item_ids.shape:
            raise ValueError("prefix behaviors and items have different shapes")
        if prefix_time_deltas.shape != prefix_item_ids.shape:
            raise ValueError("prefix time deltas and items have different shapes")
        if candidate_ids.ndim != 2 or candidate_ids.shape[0] != prefix_item_ids.shape[0]:
            raise ValueError("candidate_ids must have shape [B, C]")
        if candidate_ids.shape[1] < 1:
            raise ValueError("candidate_ids must contain at least one candidate")
        _, current_prefix = self.forward(
            prefix_item_ids,
            prefix_behaviors,
            prefix_time_deltas,
            return_kv=True,
            return_hidden=False,
            lengths=lengths,
        )
        assert current_prefix is not None

        def sliced(values: torch.Tensor | None, start: int, end: int) -> torch.Tensor | None:
            if values is None or values.ndim == 1:
                return values
            if values.ndim >= 2 and values.shape[1] == candidate_ids.shape[1]:
                return values[:, start:end]
            return values

        output = []
        for start in range(0, candidate_ids.shape[1], chunk_size):
            end = min(start + chunk_size, candidate_ids.shape[1])
            output.append(
                self._score_cc_from_prefix_cache(
                    current_prefix,
                    candidate_ids[:, start:end],
                    sliced(query_time_deltas, start, end),
                    prefix_lengths=lengths,
                    query_type_ids=sliced(query_type_ids, start, end),
                    query_action_ids=sliced(query_action_ids, start, end),
                    candidate_item_vectors=sliced(candidate_item_vectors, start, end),
                    training_append=torch.is_grad_enabled(),
                )
            )
        return tuple(output)

    def score_cc_reuse(
        self,
        parent_prefix_kv: HSTUKVCache,
        candidate_ids: torch.Tensor,
        query_time_deltas: torch.Tensor,
        prefix_lengths: torch.Tensor | None = None,
        query_type_ids: torch.Tensor | None = None,
        query_action_ids: torch.Tensor | None = None,
        candidate_item_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reuse parent-prefix KV and score transient current-model queries."""
        return self._score_cc_from_prefix_cache(
            parent_prefix_kv,
            candidate_ids,
            query_time_deltas,
            prefix_lengths=prefix_lengths,
            query_type_ids=query_type_ids,
            query_action_ids=query_action_ids,
            candidate_item_vectors=candidate_item_vectors,
            training_append=torch.is_grad_enabled(),
        )

    def observe_cc_reuse(
        self,
        parent_prefix_kv: HSTUKVCache,
        candidate_ids: torch.Tensor,
        query_time_deltas: torch.Tensor,
        prefix_lengths: torch.Tensor | None = None,
        query_type_ids: torch.Tensor | None = None,
        query_action_ids: torch.Tensor | None = None,
        candidate_item_vectors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a reused prefix and expose the pre-head readout without mutation."""
        return self._observe_cc_from_prefix_cache(
            parent_prefix_kv,
            candidate_ids,
            query_time_deltas,
            prefix_lengths=prefix_lengths,
            query_type_ids=query_type_ids,
            query_action_ids=query_action_ids,
            candidate_item_vectors=candidate_item_vectors,
            training_append=torch.is_grad_enabled(),
        )

    def observe_cc_full(
        self,
        prefix_item_ids: torch.Tensor,
        prefix_behaviors: torch.Tensor,
        prefix_time_deltas: torch.Tensor,
        candidate_ids: torch.Tensor,
        query_time_deltas: torch.Tensor,
        lengths: torch.Tensor | None = None,
        query_type_ids: torch.Tensor | None = None,
        query_action_ids: torch.Tensor | None = None,
        candidate_item_vectors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Current-Full scores and pre-head query readouts."""
        _, current_prefix = self.forward(
            prefix_item_ids,
            prefix_behaviors,
            prefix_time_deltas,
            return_kv=True,
            return_hidden=False,
            lengths=lengths,
        )
        assert current_prefix is not None
        return self._observe_cc_from_prefix_cache(
            current_prefix,
            candidate_ids,
            query_time_deltas,
            prefix_lengths=lengths,
            query_type_ids=query_type_ids,
            query_action_ids=query_action_ids,
            candidate_item_vectors=candidate_item_vectors,
            training_append=torch.is_grad_enabled(),
        )

    # Explicit aliases make the two protocol paths easy to find without
    # changing the pre-existing hidden-state ``score_candidates`` API.
    def score_candidates_full(self, *args, **kwargs) -> torch.Tensor:
        return self.score_cc_full(*args, **kwargs)

    def score_candidates_reuse(self, *args, **kwargs) -> torch.Tensor:
        return self.score_cc_reuse(*args, **kwargs)

    def _forward_with_cache_embedded_impl(
        self,
        cached_kv: HSTUKVCache,
        x: torch.Tensor,
        first_layer_residual_reset_mask: torch.Tensor | None = None,
        residual_scale: float = 1.0,
        attention_scale: float = 1.0,
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
            x, (k_all, v_all) = blk.forward_with_cache(
                x,
                cached_k,
                cached_v,
                residual_scale=residual_scale,
                attention_scale=attention_scale,
            )
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
    def forward_with_cache_embedded(
        self,
        cached_kv: HSTUKVCache,
        x: torch.Tensor,
        first_layer_residual_reset_mask: torch.Tensor | None = None,
        residual_scale: float = 1.0,
        attention_scale: float = 1.0,
    ) -> tuple[torch.Tensor, HSTUKVCache]:
        return self._forward_with_cache_embedded_impl(
            cached_kv,
            x,
            first_layer_residual_reset_mask=first_layer_residual_reset_mask,
            residual_scale=residual_scale,
            attention_scale=attention_scale,
        )

    def forward_with_cache_embedded_grad(
        self,
        cached_kv: HSTUKVCache,
        x: torch.Tensor,
        first_layer_residual_reset_mask: torch.Tensor | None = None,
        residual_scale: float = 1.0,
        attention_scale: float = 1.0,
    ) -> tuple[torch.Tensor, HSTUKVCache]:
        """Training-capable transient append used by CC qualification.

        The public inference append remains no-grad. CC theta-0 training uses
        this explicit variant so gradients can flow through both the current
        prefix and the candidate query token.
        """
        return self._forward_with_cache_embedded_impl(
            cached_kv,
            x,
            first_layer_residual_reset_mask=first_layer_residual_reset_mask,
            residual_scale=residual_scale,
            attention_scale=attention_scale,
        )

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
        residual_scale: float = 1.0,
        attention_scale: float = 1.0,
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
                residual_scale=residual_scale,
                attention_scale=attention_scale,
            )
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def forward_with_cache_new_kv(
        self,
        cached_kv: HSTUKVCache,
        new_item_ids: torch.Tensor,
        new_behaviors: torch.Tensor,
        new_time_deltas: torch.Tensor,
    ) -> tuple[torch.Tensor, HSTUKVCache]:
        """Append new tokens while returning only their K/V rows.

        For a one-token suffix this avoids copying the retained prefix K/V,
        matching append-only/paged serving state rather than a dense-cache
        reconstruction benchmark.
        """
        was_training = self.training
        self.eval()
        try:
            return self.forward_with_cache_embedded_new_kv(
                cached_kv,
                self.embed_inputs(new_item_ids, new_behaviors, new_time_deltas),
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
