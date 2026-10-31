"""Observed feedback loss plus explicitly weak candidate-list supervision."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def ranking_loss(logits, labels, candidate_mask, *, listwise_weight=0.1,
                 global_active_requests=None, ddp_world_size=1, validate=True):
    """Request-mean BCE + weight * multi-positive listwise cross entropy.

    labels: -1 unobserved, 0 exposed ineffective, 1 exposed effective.
    The listwise denominator covers the COMPLETE valid candidate list. Unknown
    candidates are contrastive distractors, never observed BCE negatives.
    Concatenate logits from exact candidate chunks before calling this function.

    DDP averages gradients. Supply the total active requests across all ranks
    (and all microbatches in the accumulation window) to compute local_sum *
    world_size / global_count. Do not divide this loss by accumulation steps
    again. Empty local ranks return a connected zero, not an exception. A
    globally empty step must skip optimizer.step on every rank, including decay.
    Metrics must aggregate *_sum and active_requests, not local-rank means.
    validate=False is allowed after CPU batch validation to avoid CUDA syncs.
    """
    if logits.ndim != 2 or logits.shape != labels.shape or logits.shape != candidate_mask.shape:
        raise ValueError("Expected matching [requests, candidates] tensors")
    if listwise_weight < 0 or not math.isfinite(listwise_weight):
        raise ValueError("Invalid listwise weight")
    if not isinstance(ddp_world_size, int) or ddp_world_size < 1:
        raise ValueError("ddp_world_size must be a positive integer")
    if ddp_world_size != 1 and global_active_requests is None:
        raise ValueError("Distributed loss requires the global active request count")
    if candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be boolean")
    if validate and not torch.isin(labels[candidate_mask], labels.new_tensor([-1, 0, 1])).all():
        raise ValueError("Invalid label states")
    if validate and not torch.isfinite(logits[candidate_mask]).all():
        raise ValueError("Nonfinite valid logits")
    # Mask before arithmetic so ignored/padded -inf scores cannot create NaNs.
    scores = torch.where(candidate_mask, logits.float(), 0.0)
    observed = candidate_mask & (labels >= 0)
    positive = candidate_mask & (labels == 1)
    active = observed.any(dim=1)
    targets = torch.where(observed, labels, 0).float()
    pointwise = F.binary_cross_entropy_with_logits(scores, targets, reduction="none")
    bce = (pointwise * observed).sum(dim=1) / observed.sum(dim=1).clamp_min(1)
    # Empty padding-only rows get a finite dummy denominator; they have no loss.
    denominator_scores = scores.masked_fill(~candidate_mask, -torch.inf)
    denominator_scores = torch.where(candidate_mask.any(dim=1, keepdim=True), denominator_scores, 0.0)
    log_denominator = torch.logsumexp(denominator_scores, dim=1)
    log_denominator = torch.where(candidate_mask.any(dim=1), log_denominator, 0.0)
    positive_mean = (scores * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    listwise = torch.where(positive.any(dim=1), log_denominator - positive_mean, 0.0)
    count = active.sum()
    bce_sum = (bce * active).sum()
    listwise_sum = (listwise * active).sum()
    denominator = count if global_active_requests is None else torch.as_tensor(
        global_active_requests, dtype=torch.float32, device=logits.device).detach()
    if denominator.ndim != 0 or (validate and (not torch.isfinite(denominator) or denominator < count)):
        raise ValueError("Global request count must be a scalar at least the local count")
    total = (bce_sum + listwise_weight * listwise_sum) * ddp_world_size / denominator.clamp_min(1)
    return total, {"bce": bce_sum / count.clamp_min(1),
                   "listwise": listwise_sum / count.clamp_min(1),
                   "bce_sum": bce_sum, "listwise_sum": listwise_sum, "active_requests": count}
