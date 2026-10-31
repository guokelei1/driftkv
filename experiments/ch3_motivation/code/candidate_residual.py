"""Describe candidate variation of read errors at one fixed cache snapshot.

The mapped path supplies the query in every layer.  Reading teacher K/V with
that SAME query isolates local read error from changes in lower-layer queries.
The result describes a rich-query residual; it does not establish that a
candidate-independent cache is insufficient or that item-only gates are needed.
"""
from __future__ import annotations

import torch

from hstu_kvcache.adaptation.reader import history_read, score as native_score


def _energy_ratios(centered, total):
    """Return null for an exactly zero residual, whose ratio is undefined."""
    return [None if float(t) == 0. else min(1., max(0., float(c / t)))
            for c, t in zip(centered, total, strict=True)]


@torch.no_grad()
def candidate_residual(model, mapped_cache, teacher_cache, candidates, query_delta):
    """Return JSON-safe local residual statistics for one user and Q16.

    ``trace.queries[layer]`` and ``trace.history_heads[layer]`` both have shape
    [1, heads, 16, head_dim].  Residuals are teacher read minus mapped read.
    Length normalization divides BOTH reads by max(1, history_length); no
    evaluation-user moments or labels fit a scale.  Layer/head arrays use [L,H],
    and the per-candidate array uses [L,H,16].  ``summary`` is this user's
    equal-layer/head MSE; its variation ratio pools residual energies, rather
    than averaging ratios from heads with different or zero residual energy.

    The native scoring path may include time, behavior and history context in
    q.  Candidate-varying residuals therefore imply neither item-only gating
    benefits nor that no fixed K/V correction could reproduce these reads.
    """
    if model.training:
        raise ValueError("candidate residuals require the frozen model in eval mode")
    if tuple(candidates.shape) != (1, 16):
        raise ValueError("candidate residuals use one fixed user's Q16 snapshot")
    if mapped_cache.seq_len != teacher_cache.seq_len:
        raise ValueError("mapped and teacher histories must have aligned lengths")

    scores, trace = native_score(model, mapped_cache, candidates, query_delta, trace=True)
    residuals = []
    length = max(1, mapped_cache.seq_len)
    for layer, (q, actual) in enumerate(zip(trace.queries, trace.history_heads, strict=True)):
        wanted = history_read(model.blocks[layer].attn, q,
                              teacher_cache.k[layer], teacher_cache.v[layer])
        residuals.append(wanted - actual)
    residual = torch.stack(residuals).double()[:, 0]  # [L,H,Q,D]
    energy = residual.square().sum((2, 3))
    centered = residual - residual.mean(2, keepdim=True)
    variation = centered.square().sum((2, 3))
    mse = residual.square().mean((2, 3))
    # One compact statistics transfer avoids per-head/layer CUDA synchronizes.
    payload = torch.cat((mse[..., None], energy[..., None], variation[..., None],
                         residual.square().mean(3)), -1).cpu()
    raw = payload[..., 0]
    ratios = [_energy_ratios(c, t) for c, t in zip(payload[..., 2], payload[..., 1], strict=True)]
    total_energy, centered_energy = float(payload[..., 1].sum()), float(payload[..., 2].sum())
    mean_raw = float(raw.mean())
    pooled_ratio = (None if total_energy == 0. else
                    min(1., max(0., centered_energy / total_energy)))
    return dict(
        format="fixed_snapshot_same_query_candidate_residual_v1",
        history_length=int(mapped_cache.seq_len), candidate_count=16,
        layers=raw.shape[0], heads=raw.shape[1],
        raw_mse_by_layer_head=raw.tolist(),
        length_normalized_mse_by_layer_head=(raw / (length * length)).tolist(),
        centered_energy_ratio_by_layer_head=ratios,
        candidate_raw_mse_by_layer_head=payload[..., 3:].tolist(),
        mapped_scores=scores[0].cpu().tolist(),
        summary=dict(raw_read_mse=mean_raw,
                     length_normalized_read_mse=mean_raw / (length * length),
                     centered_energy_ratio=pooled_ratio),
        normalization="divide both historical reads by max(1, aligned history length); no fitted evaluation scale",
        scope="descriptive residual at the mapped path's same queries; no proof of gate necessity, item-only conditioning benefit, or impossibility of a fixed K/V correction",
    )
