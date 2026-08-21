"""Training primitives for candidate-conditioned reranking."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class FrozenLinearBaseRanker(nn.Module):
    """A fitted linear base scorer that cannot change with the CC path.

    Feature construction and coefficient fitting deliberately live outside
    the HSTU model.  Parameters are registered as buffers, so optimizers used
    for the residual model cannot silently refit or rescale the base score.
    """

    def __init__(
        self,
        coefficients: torch.Tensor,
        *,
        intercept: float = 0.0,
        feature_mean: torch.Tensor | None = None,
        feature_scale: torch.Tensor | None = None,
        feature_clip_low: torch.Tensor | None = None,
        feature_clip_high: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        coefficients = torch.as_tensor(coefficients).detach().float()
        if coefficients.ndim != 1 or coefficients.numel() == 0:
            raise ValueError("coefficients must be a non-empty rank-one tensor")
        if not torch.isfinite(coefficients).all():
            raise ValueError("coefficients must be finite")
        mean = (
            torch.zeros_like(coefficients)
            if feature_mean is None
            else torch.as_tensor(feature_mean).detach().float()
        )
        scale = (
            torch.ones_like(coefficients)
            if feature_scale is None
            else torch.as_tensor(feature_scale).detach().float()
        )
        clip_low = (
            torch.full_like(coefficients, -1.0e30)
            if feature_clip_low is None
            else torch.as_tensor(feature_clip_low).detach().float()
        )
        clip_high = (
            torch.full_like(coefficients, 1.0e30)
            if feature_clip_high is None
            else torch.as_tensor(feature_clip_high).detach().float()
        )
        if any(
            value.shape != coefficients.shape
            for value in (mean, scale, clip_low, clip_high)
        ):
            raise ValueError("feature normalization must align with coefficients")
        if not all(torch.isfinite(value).all() for value in (mean, scale, clip_low, clip_high)):
            raise ValueError("feature normalization must be finite")
        if bool((scale <= 0).any()) or bool((clip_low > clip_high).any()):
            raise ValueError("feature_scale and clip bounds are invalid")
        if not torch.isfinite(torch.tensor(float(intercept))):
            raise ValueError("intercept must be finite")
        self.register_buffer("coefficients", coefficients)
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale)
        self.register_buffer("feature_clip_low", clip_low)
        self.register_buffer("feature_clip_high", clip_high)
        self.register_buffer("intercept", torch.tensor(float(intercept), dtype=torch.float32))

    @property
    def num_features(self) -> int:
        return int(self.coefficients.numel())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2 or features.shape[-1] != self.num_features:
            raise ValueError("features must end in the frozen base feature dimension")
        if not torch.isfinite(features).all():
            raise ValueError("base features must be finite")
        coefficients = self.coefficients.to(device=features.device, dtype=features.dtype)
        mean = self.feature_mean.to(device=features.device, dtype=features.dtype)
        scale = self.feature_scale.to(device=features.device, dtype=features.dtype)
        clip_low = self.feature_clip_low.to(device=features.device, dtype=features.dtype)
        clip_high = self.feature_clip_high.to(device=features.device, dtype=features.dtype)
        intercept = self.intercept.to(device=features.device, dtype=features.dtype)
        clipped = torch.clamp(features, min=clip_low, max=clip_high)
        return ((clipped - mean) / scale * coefficients).sum(dim=-1) + intercept

    @classmethod
    def from_frozen_artifact(cls, artifact: dict) -> FrozenLinearBaseRanker:
        """Construct the non-trainable scorer from a sealed P7 Base artifact."""
        scaler = artifact["scaler"]
        return cls(
            torch.as_tensor(artifact["coefficients"]),
            intercept=float(artifact["intercept"]),
            feature_mean=torch.as_tensor(scaler["mean"]),
            feature_scale=torch.as_tensor(scaler["scale"]),
            feature_clip_low=torch.as_tensor(scaler["clip_low"]),
            feature_clip_high=torch.as_tensor(scaler["clip_high"]),
        )


def combine_base_and_cc_residual(
    base_logits: torch.Tensor,
    residual_logits: torch.Tensor,
) -> torch.Tensor:
    """Return the frozen deployment score ``base + residual``.

    No coefficient is accepted by design.  Both tensors must already refer to
    the same materialized candidate universe and ordering.
    """
    if base_logits.shape != residual_logits.shape:
        raise ValueError("base and residual logits must have identical shapes")
    if base_logits.device != residual_logits.device:
        raise ValueError("base and residual logits must be on the same device")
    if base_logits.dtype != residual_logits.dtype:
        raise ValueError("base and residual logits must have the same dtype")
    if not torch.isfinite(base_logits).all() or not torch.isfinite(residual_logits).all():
        raise ValueError("base and residual logits must be finite")
    return base_logits + residual_logits


def masked_listwise_cross_entropy(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_indices: torch.Tensor,
    example_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Request-mean CE for padded variable-size candidate sets.

    Padding is set to ``-inf`` before the softmax. Candidate count therefore
    never changes a request's weight, and a padded target is rejected.
    """
    if scores.ndim != 2 or candidate_mask.shape != scores.shape:
        raise ValueError("scores and candidate_mask must have shape [B, C]")
    mask = candidate_mask.to(device=scores.device, dtype=torch.bool)
    if bool((~mask).all(dim=1).any()):
        raise ValueError("every request must contain at least one candidate")
    positive_indices = positive_indices.to(device=scores.device, dtype=torch.long)
    if positive_indices.shape != (scores.shape[0],):
        raise ValueError("positive_indices must have shape [B]")
    if bool((positive_indices < 0).any()) or bool((positive_indices >= scores.shape[1]).any()):
        raise ValueError("positive index is outside the padded candidate tensor")
    if not bool(mask.gather(1, positive_indices[:, None]).all()):
        raise ValueError("positive candidates must not be padding")
    masked_scores = scores.masked_fill(~mask, -torch.inf)
    per_request = -masked_scores.gather(1, positive_indices[:, None]).squeeze(1)
    per_request = per_request + torch.logsumexp(masked_scores, dim=1)
    if example_weights is None:
        return per_request.mean()
    weights = example_weights.to(device=scores.device, dtype=scores.dtype)
    if weights.shape != per_request.shape:
        raise ValueError("example_weights must have shape [B]")
    if not torch.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("example_weights must be finite and non-negative")
    if not bool(weights.sum() > 0):
        raise ValueError("example_weights must have positive total weight")
    return (per_request * weights).sum() / weights.sum()


def exact_chunked_listwise_cross_entropy(
    score_chunks: Sequence[torch.Tensor],
    positive_index: int,
) -> torch.Tensor:
    """Exact one-request listwise CE via cross-chunk log-sum-exp.

    Chunks may use any positive sizes. Their order is the candidate order, and
    gradients flow through every chunk without concatenating all logits.
    """
    if not score_chunks:
        raise ValueError("score_chunks must not be empty")
    flattened = []
    for chunk in score_chunks:
        if chunk.ndim == 2 and chunk.shape[0] == 1:
            chunk = chunk.squeeze(0)
        if chunk.ndim != 1 or chunk.numel() == 0:
            raise ValueError("each score chunk must contain one request and at least one candidate")
        if not torch.isfinite(chunk).all():
            raise ValueError("score chunks must be finite")
        flattened.append(chunk)
    total = sum(chunk.numel() for chunk in flattened)
    if not 0 <= int(positive_index) < total:
        raise ValueError("positive_index is outside the chunked candidate universe")
    log_normalizer = torch.logsumexp(flattened[0], dim=0)
    for chunk in flattened[1:]:
        log_normalizer = torch.logaddexp(log_normalizer, torch.logsumexp(chunk, dim=0))
    offset = 0
    positive_score = None
    for chunk in flattened:
        if offset <= positive_index < offset + chunk.numel():
            positive_score = chunk[int(positive_index) - offset]
            break
        offset += chunk.numel()
    assert positive_score is not None
    return log_normalizer - positive_score


def conditional_reranking_loss(
    scores: torch.Tensor,
    positive_indices: torch.Tensor,
    example_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy within each frozen candidate panel.

    This is conditional reranking, not a full-catalog probability objective.
    Proposal metadata (rank, weight and ``log_q_main``) remains reportable
    alongside the example; it is not mixed into model scores or used to fit a
    target KV. Optional example weights only weight complete training rows.
    """
    if scores.ndim != 2:
        raise ValueError("scores must have shape [B, C]")
    positive_indices = positive_indices.to(device=scores.device, dtype=torch.long)
    if positive_indices.shape != (scores.shape[0],):
        raise ValueError("positive_indices must have shape [B]")
    if bool((positive_indices < 0).any()) or bool((positive_indices >= scores.shape[1]).any()):
        raise ValueError("positive index is outside the candidate panel")
    per_example = F.cross_entropy(scores, positive_indices, reduction="none")
    if example_weights is None:
        return per_example.mean()
    example_weights = example_weights.to(device=scores.device, dtype=per_example.dtype)
    if example_weights.shape != per_example.shape:
        raise ValueError("example_weights must have shape [B]")
    if not torch.isfinite(example_weights).all() or bool((example_weights < 0).any()):
        raise ValueError("example_weights must be finite and non-negative")
    normalizer = example_weights.sum()
    if float(normalizer) <= 0:
        raise ValueError("example_weights must contain a positive total weight")
    return (per_example * example_weights).sum() / normalizer
