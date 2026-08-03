from __future__ import annotations

import math

import torch

from .xp_projected_edge import TrainableProjectedModuloEmbedding


def lookup_multifield_projected(
    embedding: TrainableProjectedModuloEmbedding,
    feature_ids: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    if (
        feature_ids.ndim != 3
        or lengths.shape != (feature_ids.shape[0],)
        or feature_ids.shape[2] < 1
        or feature_ids.device != lengths.device
    ):
        raise ValueError("multi-field projected lookup shape differs")
    batch, width, fields = feature_ids.shape
    if bool(torch.any(lengths < 0)) or bool(torch.any(lengths > width)):
        raise ValueError("multi-field projected lookup lengths differ")
    flattened = feature_ids.reshape(batch, width * fields)
    vectors = embedding(flattened, lengths * fields)
    return vectors.reshape(batch, width, fields, embedding.hidden_size).sum(dim=2) / math.sqrt(
        fields
    )
