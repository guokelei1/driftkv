from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """Continuous time-delta embedding via sinusoidal/RBF features + MLP.

    HSTU uses time-since-last-event rather than absolute positional embeddings,
    because the model is designed for non-stationary streaming data where
    absolute position carries no meaning. This is a research-friendly stub:
    swap in RBF kernels or learned bucketisation without touching the model.
    """

    def __init__(self, hidden_size: int, num_freqs: int = 16, max_period: float = 86400.0) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        self.max_period = max_period
        self.proj = nn.Linear(2 * num_freqs, hidden_size, bias=False)

    def forward(self, time_delta: torch.Tensor) -> torch.Tensor:
        # time_delta: [B, L] in seconds (>=0). 0 for the first event.
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(self.num_freqs, device=time_delta.device) / self.num_freqs
        )  # [num_freqs]
        phases = time_delta.unsqueeze(-1) * freqs  # [B, L, num_freqs]
        emb = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)  # [B, L, 2*num_freqs]
        return self.proj(emb)


class BehaviorEncoder(nn.Module):
    """Embedding for discrete behavior types (click/like/watch/...)."""

    def __init__(self, num_behaviors: int, hidden_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_behaviors + 1, hidden_size, padding_idx=0)

    def forward(self, behavior: torch.Tensor) -> torch.Tensor:
        return self.embed(behavior)


class QueryTokenEncoder(nn.Module):
    """Encode a transient candidate-conditioned CC query token.

    Query tokens deliberately use tables separate from ``BehaviorEncoder``.
    In particular, ``action_embedding`` has no padding index: the configured
    action ID is a reserved query action, not PAD or MASK.  The encoder does
    not retain state; callers may append its output to a prefix for scoring
    and discard the resulting one-token cache.
    """

    def __init__(
        self,
        hidden_size: int,
        num_query_types: int = 1,
        num_query_actions: int = 1,
    ) -> None:
        super().__init__()
        if num_query_types < 1 or num_query_actions < 1:
            raise ValueError("query embedding table sizes must be positive")
        self.type_embedding = nn.Embedding(num_query_types, hidden_size)
        # Do not set padding_idx: RESERVED_QUERY_ACTION is independent of PAD/MASK.
        self.action_embedding = nn.Embedding(num_query_actions, hidden_size)

    def forward(
        self,
        item_vectors: torch.Tensor,
        query_type_ids: torch.Tensor,
        query_action_ids: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if item_vectors.shape != time_embedding.shape:
            raise ValueError("query item and time embeddings must have the same shape")
        if item_vectors.shape[:-1] != query_type_ids.shape:
            raise ValueError("query type IDs and query items have incompatible shapes")
        if item_vectors.shape[:-1] != query_action_ids.shape:
            raise ValueError("query action IDs and query items have incompatible shapes")
        return (
            item_vectors
            + self.type_embedding(query_type_ids)
            + self.action_embedding(query_action_ids)
            + time_embedding
        )


class ItemEmbedding(nn.Module):
    """Shared item embedding table used both as input feature and as output head.

    Tied weights between the input item embedding lookup and the output scoring
    logits is the HSTU convention (dot-product retrieval). We expose the table
    directly so versioned evaluation uses the same current scoring head.
    """

    def __init__(self, num_items: int, hidden_size: int, padding_idx: int = 0) -> None:
        super().__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.zeros(num_items + 1, hidden_size))
        self.sparse_gradient = False
        nn.init.normal_(self.weight, std=0.02)
        with torch.no_grad():
            self.weight[padding_idx].fill_(0.0)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            item_ids,
            self.weight,
            sparse=self.sparse_gradient,
        )

    def score(self, hidden: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """Dot-product scoring: hidden [.., H] x candidates [.., C, H] -> [.., C]."""
        cand = F.embedding(
            candidate_ids,
            self.weight,
            sparse=self.sparse_gradient,
        )
        return torch.einsum("...h,...ch->...c", hidden, cand)
