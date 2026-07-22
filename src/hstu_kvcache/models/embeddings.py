from __future__ import annotations

import math

import torch
import torch.nn as nn


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
        nn.init.normal_(self.weight, std=0.02)
        with torch.no_grad():
            self.weight[padding_idx].fill_(0.0)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[item_ids]

    def score(self, hidden: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """Dot-product scoring: hidden [.., H] x candidates [.., C, H] -> [.., C]."""
        cand = self.weight[candidate_ids]  # [.., C, H]
        return torch.einsum("...h,...ch->...c", hidden, cand)
