"""Diagnostic query-conditioned low-rank functional corrections.

Targets are derived from Current-Exact versus Current-Reuse anchor traces, so
these routines are oracle representation tests.  They do not define an
executable estimator and never return or mutate a KV cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from insight.candidate_shared_causal import _block_update, _cached_prefix_heads
from insight.reader_compatibility_correction import _self_heads
from hstu_kvcache.models import HSTUKVCache


LAYERED_STAGES = ("av_aggregation", "u_gated_update", "layer_hidden")


@dataclass(frozen=True)
class LowRankResult:
    scores: torch.Tensor
    diagnostics: tuple[dict[str, torch.Tensor | float | int | str], ...]
    storage_values_fp32_per_user: int


def _rank_at_energy(singular_values: torch.Tensor, threshold: float) -> int:
    energy = singular_values.float().square()
    if float(energy.sum()) <= 1e-20:
        return 0
    cumulative = torch.cumsum(energy, dim=0) / energy.sum()
    return int(torch.searchsorted(cumulative, threshold).item() + 1)


def fit_predict_low_rank(
    anchor_features: torch.Tensor,
    anchor_targets: torch.Tensor,
    heldout_features: torch.Tensor,
    *,
    rank: int,
    ridge_relative: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, tuple[dict[str, float | int], ...], int]:
    """Fit a per-user reduced-rank ridge map and return anchor/heldout targets.

    Inputs use ``[B,C,D]``.  Rank 0 is exactly the anchor target mean and stores
    only that output vector.  Positive ranks store an input mean, output mean,
    an input-to-coefficient map and an output basis.
    """
    if anchor_features.ndim != 3 or anchor_targets.ndim != 3:
        raise ValueError("anchor features and targets must have shape [B,C,D]")
    if heldout_features.ndim != 3:
        raise ValueError("held-out features must have shape [B,C,D]")
    if anchor_features.shape[:2] != anchor_targets.shape[:2]:
        raise ValueError("anchor feature and target candidate axes differ")
    if anchor_features.shape[0] != heldout_features.shape[0]:
        raise ValueError("anchor and held-out batches differ")
    if anchor_features.shape[2] != heldout_features.shape[2]:
        raise ValueError("anchor and held-out feature widths differ")
    if rank < 0:
        raise ValueError("rank must be non-negative")

    batch, anchors, input_width = anchor_features.shape
    output_width = anchor_targets.shape[2]
    anchor_predictions = []
    heldout_predictions = []
    diagnostics: list[dict[str, float | int]] = []
    storage = output_width if rank == 0 else (
        input_width + output_width + rank * (input_width + output_width)
    )
    for index in range(batch):
        x = anchor_features[index].float()
        y = anchor_targets[index].float()
        x_heldout = heldout_features[index].float()
        x_mean = x.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        x_centered = x - x_mean
        y_centered = y - y_mean
        singular_values = torch.linalg.svdvals(y_centered)
        total_energy = singular_values.square().sum().clamp_min(1e-20)
        if rank == 0 or float(total_energy) <= 1e-20:
            anchor_prediction = y_mean.expand_as(y)
            heldout_prediction = y_mean.expand(x_heldout.shape[0], output_width)
            retained = 0.0
            effective_rank = 0
        else:
            effective_rank = min(rank, anchors, output_width)
            _, _, vh = torch.linalg.svd(y_centered, full_matrices=False)
            basis = vh[:effective_rank]
            coefficients = y_centered @ basis.T
            gram = x_centered @ x_centered.T
            scale = float(torch.trace(gram) / max(anchors, 1))
            regularizer = max(ridge_relative * scale, 1e-8)
            alpha = torch.linalg.solve(
                gram + regularizer * torch.eye(anchors, device=gram.device),
                coefficients,
            )
            anchor_coefficients = gram @ alpha
            heldout_coefficients = (x_heldout - x_mean) @ x_centered.T @ alpha
            anchor_prediction = y_mean + anchor_coefficients @ basis
            heldout_prediction = y_mean + heldout_coefficients @ basis
            retained = float(singular_values[:effective_rank].square().sum() / total_energy)
        relative_error = float(
            torch.linalg.vector_norm(anchor_prediction - y)
            / torch.linalg.vector_norm(y).clamp_min(1e-20)
        )
        diagnostics.append(
            {
                "target_rank90": _rank_at_energy(singular_values, 0.90),
                "target_rank95": _rank_at_energy(singular_values, 0.95),
                "rank_retained_centered_energy": retained,
                "anchor_fit_relative_l2": relative_error,
                "effective_rank": effective_rank,
            }
        )
        anchor_predictions.append(anchor_prediction.to(anchor_targets.dtype))
        heldout_predictions.append(heldout_prediction.to(anchor_targets.dtype))
    return (
        torch.stack(anchor_predictions),
        torch.stack(heldout_predictions),
        tuple(diagnostics),
        storage,
    )


def _layer_values(
    model,
    cache: HSTUKVCache,
    x: torch.Tensor,
    layer: int,
    candidates: int,
) -> dict[str, torch.Tensor]:
    block = model.blocks[layer]
    batch = x.shape[0] // candidates
    residual = x
    x_norm = block.norm(x)
    q, k_new, v_new = block.attn._project(x_norm)
    prefix = _cached_prefix_heads(block.attn, q, cache, layer, candidates)
    self_heads = _self_heads(block, q, k_new, v_new).reshape(
        batch, candidates, block.attn.num_heads, 1, block.attn.head_dim
    )
    av = prefix + self_heads
    update = _block_update(
        block,
        x_norm,
        av.reshape(batch * candidates, block.attn.num_heads, 1, block.attn.head_dim),
    )
    hidden = residual + update
    return {
        "q": q.squeeze(2).reshape(batch, candidates, -1),
        "av_aggregation": av.squeeze(3).reshape(batch, candidates, -1),
        "u_gated_update": update.reshape(batch, candidates, -1),
        "layer_hidden": hidden.reshape(batch, candidates, -1),
        "residual": residual,
        "x_norm": x_norm,
        "av_heads": av,
        "update": update,
        "hidden": hidden,
    }


def _advance_with_prediction(
    model,
    values: dict[str, torch.Tensor],
    prediction: torch.Tensor,
    *,
    layer: int,
    candidates: int,
    stage: str,
) -> torch.Tensor:
    block = model.blocks[layer]
    batch = prediction.shape[0]
    if stage == "av_aggregation":
        corrected = values["av_heads"] + prediction.reshape(
            batch, candidates, block.attn.num_heads, 1, block.attn.head_dim
        )
        update = _block_update(
            block,
            values["x_norm"],
            corrected.reshape(
                batch * candidates, block.attn.num_heads, 1, block.attn.head_dim
            ),
        )
        return values["residual"] + update
    if stage == "u_gated_update":
        return values["residual"] + values["update"] + prediction.reshape(
            batch * candidates, 1, -1
        )
    if stage == "layer_hidden":
        return values["hidden"] + prediction.reshape(batch * candidates, 1, -1)
    raise ValueError(f"unsupported layered stage: {stage}")


@torch.inference_mode()
def low_rank_layered_correction(
    model,
    exact_cache: HSTUKVCache,
    reuse_cache: HSTUKVCache,
    anchor_candidates: torch.Tensor,
    heldout_candidates: torch.Tensor,
    query_time_deltas: torch.Tensor,
    *,
    stage: str,
    rank: int,
) -> LowRankResult:
    if stage not in LAYERED_STAGES:
        raise ValueError(f"unsupported layered stage: {stage}")
    if model.cfg.relative_position_bias:
        raise ValueError("functional-boundary adapter requires no relative-position bias")
    batch, anchors = anchor_candidates.shape
    if heldout_candidates.shape[0] != batch:
        raise ValueError("anchor and held-out candidate batches differ")
    heldout = heldout_candidates.shape[1]
    anchor_x = model.embed_query_tokens(anchor_candidates, query_time_deltas).reshape(
        batch * anchors, 1, model.cfg.hidden_size
    )
    heldout_x = model.embed_query_tokens(heldout_candidates, query_time_deltas).reshape(
        batch * heldout, 1, model.cfg.hidden_size
    )
    diagnostics: list[dict[str, torch.Tensor | float | int | str]] = []
    storage_total = 0

    for layer in range(len(model.blocks)):
        anchor_exact = _layer_values(model, exact_cache, anchor_x, layer, anchors)
        anchor_reuse = _layer_values(model, reuse_cache, anchor_x, layer, anchors)
        heldout_reuse = _layer_values(model, reuse_cache, heldout_x, layer, heldout)
        target = anchor_exact[stage] - anchor_reuse[stage]
        anchor_prediction, heldout_prediction, fit, storage = fit_predict_low_rank(
            anchor_reuse["q"], target, heldout_reuse["q"], rank=rank
        )
        storage_total += storage
        for batch_index, row in enumerate(fit):
            diagnostics.append(
                {
                    "stage": stage,
                    "layer": layer,
                    "rank": rank,
                    "batch_index": batch_index,
                    "storage_values_fp32": storage,
                    **row,
                }
            )
        anchor_x = _advance_with_prediction(
            model,
            anchor_reuse,
            anchor_prediction,
            layer=layer,
            candidates=anchors,
            stage=stage,
        )
        heldout_x = _advance_with_prediction(
            model,
            heldout_reuse,
            heldout_prediction,
            layer=layer,
            candidates=heldout,
            stage=stage,
        )

    readout = model.final_norm(heldout_x).reshape(
        batch, heldout, model.cfg.hidden_size
    )
    scores = model.cc_score_head(readout).squeeze(-1)
    return LowRankResult(scores, tuple(diagnostics), storage_total)


@torch.inference_mode()
def low_rank_final_representation(
    model,
    exact_cache: HSTUKVCache,
    reuse_cache: HSTUKVCache,
    anchor_candidates: torch.Tensor,
    heldout_candidates: torch.Tensor,
    query_time_deltas: torch.Tensor,
    *,
    rank: int,
) -> LowRankResult:
    _, anchor_exact = model.observe_cc_reuse(
        exact_cache, anchor_candidates, query_time_deltas
    )
    _, anchor_reuse = model.observe_cc_reuse(
        reuse_cache, anchor_candidates, query_time_deltas
    )
    _, heldout_reuse = model.observe_cc_reuse(
        reuse_cache, heldout_candidates, query_time_deltas
    )
    _, prediction, fit, storage = fit_predict_low_rank(
        anchor_reuse,
        anchor_exact - anchor_reuse,
        heldout_reuse,
        rank=rank,
    )
    corrected = heldout_reuse + prediction
    diagnostics = tuple(
        {
            "stage": "final_readout",
            "layer": -1,
            "rank": rank,
            "batch_index": batch_index,
            "storage_values_fp32": storage,
            **row,
        }
        for batch_index, row in enumerate(fit)
    )
    return LowRankResult(
        model.cc_score_head(corrected).squeeze(-1), diagnostics, storage
    )

