"""Fit-free diagnostics for affine cones in HSTU ELU+1 attention.

For legacy HSTU without relative-position bias, one attention head reads

    sum_i (ELU(scale * <q, k_i>) + 1) v_i.

Inside a fixed sign cone, the non-negative half of this response is exactly
affine in the query (not a Taylor approximation):

    B_P + scale * q M_P,
    B_P = sum_{i in P} v_i,
    M_P = sum_{i in P} k_i outer v_i.

The negative half is ``sum_{i not in P} exp(scale * <q,k_i>) v_i``.  This
module exposes the two pieces and statistics for testing whether candidate
queries from one user actually share a cone.  It performs no fitting,
clustering, label access, or cache mutation.

Tensor convention is ``q=[B,H,Q,Dk]``, ``k=[B,H,N,Dk]`` and
``v=[B,H,N,Dv]``.  The scale is the already-resolved
``PointwiseAttention.scale`` value, including any configured head scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionConeMaskStatistics:
    """Per-user, per-head stability statistics, each shaped ``[B,H]``."""

    positive_fraction: torch.Tensor
    shared_positive_fraction: torch.Tensor
    shared_negative_fraction: torch.Tensor
    unanimous_fraction: torch.Tensor
    majority_agreement: torch.Tensor
    pairwise_sign_agreement: torch.Tensor
    pairwise_positive_jaccard: torch.Tensor


@dataclass(frozen=True)
class AffineConeMoments:
    """Exact positive-half moments for one fixed attention cone.

    ``base`` is ``B=[B,H,Dv]`` and ``linear`` is
    ``M=[B,H,Dk,Dv]``. ``positive_mask`` is retained for auditability.
    """

    base: torch.Tensor
    linear: torch.Tensor
    positive_mask: torch.Tensor


@dataclass(frozen=True)
class SignedAffineConeMoments:
    """Current-minus-Parent affine moments with both cones kept explicit."""

    base: torch.Tensor
    linear: torch.Tensor
    current_positive_mask: torch.Tensor
    parent_positive_mask: torch.Tensor


@dataclass(frozen=True)
class ELUPlusOneResponseDecomposition:
    """Native response split and query-local negative-residual diagnostics."""

    logits: torch.Tensor
    positive_mask: torch.Tensor
    positive_response: torch.Tensor
    negative_response: torch.Tensor
    full_response: torch.Tensor
    negative_activation_fraction: torch.Tensor
    negative_response_fraction: torch.Tensor
    negative_to_positive_response_ratio: torch.Tensor


def _validate_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shapes [B,H,Q,D], [B,H,N,D], [B,H,N,Dv]")
    batch, heads, queries, key_width = q.shape
    if k.shape[:2] != (batch, heads) or k.shape[3] != key_width:
        raise ValueError("q and k batch/head/key widths differ")
    if v.shape[:3] != k.shape[:3]:
        raise ValueError("k and v batch/head/history axes differ")
    if not (q.is_floating_point() and k.is_floating_point() and v.is_floating_point()):
        raise ValueError("q, k, and v must be floating point")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if queries < 1 or k.shape[2] < 1:
        raise ValueError("query and history axes must be non-empty")
    return batch, heads, queries, key_width, v.shape[3]


def scaled_qk_logits(q: torch.Tensor, k: torch.Tensor, *, scale: float) -> torch.Tensor:
    """Return the exact pre-activation HSTU logits ``scale * q @ k.T``."""

    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shapes [B,H,Q,D] and [B,H,N,D]")
    if q.shape[:2] != k.shape[:2] or q.shape[3] != k.shape[3]:
        raise ValueError("q and k batch/head/key widths differ")
    if q.device != k.device or q.dtype != k.dtype:
        raise ValueError("q and k must share device and dtype")
    if not (q.is_floating_point() and k.is_floating_point()):
        raise ValueError("q and k must be floating point")
    return torch.matmul(q, k.transpose(-2, -1)) * float(scale)


def attention_cone_mask_statistics(
    positive_mask: torch.Tensor,
) -> AttentionConeMaskStatistics:
    """Measure how consistently a user's candidate queries partition history.

    Args:
        positive_mask: Boolean ``[B,H,Q,N]`` mask from scaled QK logits.

    Pairwise metrics average over unordered query pairs.  With one query they
    are defined as one.  Positive Jaccard also equals one when both queries
    have an empty positive set.
    """

    if positive_mask.ndim != 4 or positive_mask.dtype != torch.bool:
        raise ValueError("positive_mask must be boolean with shape [B,H,Q,N]")
    if positive_mask.shape[2] < 1 or positive_mask.shape[3] < 1:
        raise ValueError("query and history axes must be non-empty")

    mask = positive_mask
    queries = mask.shape[2]
    positive_fraction = mask.to(torch.float32).mean(dim=(-2, -1))
    all_positive = mask.all(dim=2)
    all_negative = (~mask).all(dim=2)
    shared_positive_fraction = all_positive.to(torch.float32).mean(dim=-1)
    shared_negative_fraction = all_negative.to(torch.float32).mean(dim=-1)
    unanimous_fraction = (all_positive | all_negative).to(torch.float32).mean(dim=-1)

    positive_votes = mask.sum(dim=2)
    majority = torch.maximum(positive_votes, queries - positive_votes)
    majority_agreement = (majority.to(torch.float32) / queries).mean(dim=-1)

    if queries == 1:
        ones = torch.ones_like(positive_fraction)
        pairwise_sign_agreement = ones
        pairwise_positive_jaccard = ones
    else:
        pair_indices = torch.triu_indices(
            queries, queries, offset=1, device=mask.device
        )
        left = mask.index_select(2, pair_indices[0])
        right = mask.index_select(2, pair_indices[1])
        pairwise_sign_agreement = (left == right).to(torch.float32).mean(dim=(-2, -1))
        intersection = (left & right).sum(dim=-1).to(torch.float32)
        union = (left | right).sum(dim=-1).to(torch.float32)
        jaccard = torch.where(union > 0, intersection / union, torch.ones_like(union))
        pairwise_positive_jaccard = jaccard.mean(dim=-1)

    return AttentionConeMaskStatistics(
        positive_fraction=positive_fraction,
        shared_positive_fraction=shared_positive_fraction,
        shared_negative_fraction=shared_negative_fraction,
        unanimous_fraction=unanimous_fraction,
        majority_agreement=majority_agreement,
        pairwise_sign_agreement=pairwise_sign_agreement,
        pairwise_positive_jaccard=pairwise_positive_jaccard,
    )


def build_positive_affine_moments(
    k: torch.Tensor,
    v: torch.Tensor,
    positive_mask: torch.Tensor,
) -> AffineConeMoments:
    """Construct ``B`` and ``M`` for a fixed, query-shared positive mask."""

    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("k and v must have shapes [B,H,N,Dk] and [B,H,N,Dv]")
    if k.shape[:3] != v.shape[:3]:
        raise ValueError("k and v batch/head/history axes differ")
    if positive_mask.shape != k.shape[:3] or positive_mask.dtype != torch.bool:
        raise ValueError("positive_mask must be boolean with shape [B,H,N]")
    if k.device != v.device or positive_mask.device != k.device:
        raise ValueError("k, v, and positive_mask must share a device")
    if k.dtype != v.dtype or not (k.is_floating_point() and v.is_floating_point()):
        raise ValueError("k and v must share a floating-point dtype")

    weight = positive_mask.to(dtype=v.dtype)
    base = torch.einsum("bhn,bhnv->bhv", weight, v)
    linear = torch.einsum("bhn,bhnk,bhnv->bhkv", weight, k, v)
    return AffineConeMoments(
        base=base,
        linear=linear,
        positive_mask=positive_mask.detach(),
    )


def read_positive_affine_moments(
    q: torch.Tensor,
    moments: AffineConeMoments,
    *,
    scale: float,
) -> torch.Tensor:
    """Read exact positive-half response for queries inside ``moments``' cone."""

    if q.ndim != 4:
        raise ValueError("q must have shape [B,H,Q,Dk]")
    if q.shape[:2] != moments.base.shape[:2]:
        raise ValueError("query and moment batch/head axes differ")
    if q.shape[3] != moments.linear.shape[2]:
        raise ValueError("query width differs from moment key width")
    if moments.base.shape != moments.linear.shape[:2] + moments.linear.shape[3:]:
        raise ValueError("base and linear value widths differ")
    if q.device != moments.base.device or q.device != moments.linear.device:
        raise ValueError("query and moments must share a device")
    if q.dtype != moments.base.dtype or q.dtype != moments.linear.dtype:
        raise ValueError("query and moments must share a dtype")
    linear_read = torch.einsum("bhqk,bhkv->bhqv", q, moments.linear)
    return moments.base.unsqueeze(2) + float(scale) * linear_read


def subtract_affine_moments(
    current: AffineConeMoments,
    parent: AffineConeMoments,
) -> SignedAffineConeMoments:
    """Return Current-minus-Parent moments without fitting either response."""

    if current.base.shape != parent.base.shape or current.linear.shape != parent.linear.shape:
        raise ValueError("Current and Parent moment shapes differ")
    if current.base.device != parent.base.device or current.base.dtype != parent.base.dtype:
        raise ValueError("Current and Parent moments must share device and dtype")
    # The two readers may occupy different cones.  Their individual masks are
    # not collapsed into a fictitious shared mask.
    return SignedAffineConeMoments(
        base=current.base - parent.base,
        linear=current.linear - parent.linear,
        current_positive_mask=current.positive_mask,
        parent_positive_mask=parent.positive_mask,
    )


def decompose_elu_plus_one_response(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    eps: float = 1e-12,
) -> ELUPlusOneResponseDecomposition:
    """Split the native ELU+1 response into positive affine and negative parts.

    Fractions are query-local ``[B,H,Q]`` quantities.  Activation fraction is
    the negative branch's scalar weight mass divided by total ELU+1 weight
    mass.  Response fraction is its vector L2 norm divided by the full response
    norm, so it can exceed one under vector cancellation.
    """

    _validate_qkv(q, k, v)
    if eps <= 0:
        raise ValueError("eps must be positive")
    logits = scaled_qk_logits(q, k, scale=scale)
    positive_mask = logits >= 0
    positive_weights = torch.where(positive_mask, logits + 1.0, torch.zeros_like(logits))
    negative_weights = torch.where(positive_mask, torch.zeros_like(logits), torch.exp(logits))
    positive_response = torch.matmul(positive_weights, v)
    negative_response = torch.matmul(negative_weights, v)
    full_response = positive_response + negative_response

    positive_mass = positive_weights.sum(dim=-1)
    negative_mass = negative_weights.sum(dim=-1)
    negative_activation_fraction = negative_mass / (positive_mass + negative_mass).clamp_min(eps)
    negative_norm = torch.linalg.vector_norm(negative_response, dim=-1)
    positive_norm = torch.linalg.vector_norm(positive_response, dim=-1)
    full_norm = torch.linalg.vector_norm(full_response, dim=-1)
    return ELUPlusOneResponseDecomposition(
        logits=logits,
        positive_mask=positive_mask,
        positive_response=positive_response,
        negative_response=negative_response,
        full_response=full_response,
        negative_activation_fraction=negative_activation_fraction,
        negative_response_fraction=negative_norm / full_norm.clamp_min(eps),
        negative_to_positive_response_ratio=negative_norm / positive_norm.clamp_min(eps),
    )
