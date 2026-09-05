"""Exact producer-state/read-version commutator diagnostics.

For a reader version ``r`` and a cache producer version ``p``, define

``F(r, p) = reader_r(cache_p, query_r)``.

The four exact endpoints expose whether changing the cache producer and
changing the reader approximately commute.  If they did, the cross endpoint

``F(C,P) + F(P,C) - F(P,P)``

would recover ``F(C,C)``.  Its error is exactly the mixed finite difference

``F(C,C) - F(C,P) - F(P,C) + F(P,P)``.

This module is diagnostic-only.  In particular, ``F(P,C)`` reads exact
Current K/V and all score-level arithmetic is request/candidate dependent; it
is not an executable cache-migration action.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hstu_kvcache.models import HSTUKVCache
from insight_two.cone_response_memory import (
    _block_update,
    _native_prefix_heads,
    _native_self_heads,
)


@dataclass(frozen=True)
class ReaderProducerTrace:
    """One coherent reader path and its exact S4 tensors."""

    scores: torch.Tensor
    readout: torch.Tensor
    layer_s4: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class FiniteDifferenceDiagnostic:
    """Mixed finite difference relative to the Current-reader state effect."""

    mixed: torch.Tensor
    current_state_effect: torch.Tensor
    parent_state_effect: torch.Tensor
    mixed_over_current_state_l2: float
    parent_over_current_state_l2: float
    state_effect_cosine: float
    l2_recovery: float


def _norm(values: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(values.float())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = _norm(left) * _norm(right)
    if float(denominator) <= 1e-20:
        return 0.0
    return float(torch.dot(left.float().reshape(-1), right.float().reshape(-1)) / denominator)


def _validate_four(
    current_current: torch.Tensor,
    current_parent: torch.Tensor,
    parent_current: torch.Tensor,
    parent_parent: torch.Tensor,
) -> None:
    shape = current_current.shape
    if not shape:
        raise ValueError("commutator endpoints must not be scalar tensors")
    if any(
        endpoint.shape != shape
        for endpoint in (current_parent, parent_current, parent_parent)
    ):
        raise ValueError("commutator endpoint shapes differ")


def commuted_endpoint(
    current_parent: torch.Tensor,
    parent_current: torch.Tensor,
    parent_parent: torch.Tensor,
) -> torch.Tensor:
    """Return ``F(C,P) + F(P,C) - F(P,P)`` without fitting."""

    if (
        current_parent.shape != parent_current.shape
        or current_parent.shape != parent_parent.shape
    ):
        raise ValueError("commuted endpoint shapes differ")
    return current_parent + parent_current - parent_parent


def finite_difference_diagnostic(
    current_current: torch.Tensor,
    current_parent: torch.Tensor,
    parent_current: torch.Tensor,
    parent_parent: torch.Tensor,
) -> FiniteDifferenceDiagnostic:
    """Measure the exact reader-version by producer-state interaction."""

    _validate_four(current_current, current_parent, parent_current, parent_parent)
    current_effect = current_current - current_parent
    parent_effect = parent_current - parent_parent
    mixed = current_effect - parent_effect
    current_norm = _norm(current_effect)
    mixed_ratio = float(_norm(mixed) / current_norm.clamp_min(1e-20))
    parent_ratio = float(_norm(parent_effect) / current_norm.clamp_min(1e-20))
    return FiniteDifferenceDiagnostic(
        mixed=mixed,
        current_state_effect=current_effect,
        parent_state_effect=parent_effect,
        mixed_over_current_state_l2=mixed_ratio,
        parent_over_current_state_l2=parent_ratio,
        state_effect_cosine=_cosine(current_effect, parent_effect),
        l2_recovery=1.0 - mixed_ratio,
    )


def _validate_trace_inputs(
    model,
    cache: HSTUKVCache,
    candidate_ids: torch.Tensor,
) -> None:
    if model.training:
        raise ValueError("reader/producer trace requires model.eval()")
    if cache.k.ndim != 4 or cache.k.shape != cache.v.shape:
        raise ValueError("cache must contain matching [L,B,N,W] K/V")
    if cache.k.shape[0] != len(model.blocks):
        raise ValueError("cache and reader layer counts differ")
    if cache.k.shape[1] != 1:
        raise ValueError("commutator preflight supports one user")
    if cache.k.shape[2] != cache.seq_len:
        raise ValueError("cache tensor width and seq_len differ")
    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != 1:
        raise ValueError("candidate_ids must have shape [1,C]")
    if candidate_ids.shape[1] < 1:
        raise ValueError("at least one candidate is required")
    for block in model.blocks:
        if block.attn.position_bias is not None:
            raise ValueError("trace does not implement relative-position bias")


@torch.inference_mode()
def trace_reader_producer(
    model,
    cache: HSTUKVCache,
    candidate_ids: torch.Tensor,
    query_time_deltas: torch.Tensor,
    *,
    query_type_ids: torch.Tensor | None = None,
    query_action_ids: torch.Tensor | None = None,
    candidate_item_vectors: torch.Tensor | None = None,
) -> ReaderProducerTrace:
    """Trace one exact ``F(reader, cache producer)`` path.

    Query embeddings, Q projections, gates, residuals, final normalization and
    score head all come from ``model``.  Only the persistent prefix K/V comes
    from ``cache``.  Thus a Parent and Current call each use the proper query
    semantics of their reader version.
    """

    _validate_trace_inputs(model, cache, candidate_ids)
    candidates = candidate_ids.shape[1]
    x = model.embed_query_tokens(
        candidate_ids,
        query_time_deltas,
        query_type_ids=query_type_ids,
        query_action_ids=query_action_ids,
        item_vectors=candidate_item_vectors,
    ).reshape(candidates, 1, model.cfg.hidden_size)
    layer_s4: list[torch.Tensor] = []
    for layer_index, block in enumerate(model.blocks):
        residual = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        prefix_heads = _native_prefix_heads(
            block.attn,
            q,
            cache.k[layer_index],
            cache.v[layer_index],
        )
        self_heads = _native_self_heads(block.attn, q, k_new, v_new)
        s4_heads = prefix_heads + self_heads
        layer_s4.append(
            s4_heads.squeeze(2).reshape(
                1,
                candidates,
                block.attn.num_heads * block.attn.head_dim,
            )
        )
        x = residual + _block_update(block, x_norm, s4_heads)

    readout = model.final_norm(x).reshape(1, candidates, model.cfg.hidden_size)
    scores = model.cc_score_head(readout).squeeze(-1)
    return ReaderProducerTrace(
        scores=scores,
        readout=readout,
        layer_s4=tuple(layer_s4),
    )
