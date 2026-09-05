"""All-history associative prefix-moment diagnostics for the legacy reader.

This module tests a representation/closure hypothesis, not a migration design:
all history positions participate in dense additive reductions, while a small
fixed set of *historical queries* defines a shared activation region.  Inside
the positive branch of legacy ELU+1 attention, prefix response is exactly
``B_i + scale * q_i @ M_i`` with associative prefix summaries

``B_i = sum_{j<=i} m_j v_j`` and
``M_i = sum_{j<=i} m_j k_j outer v_j``.

The shared mask ``m`` is approximate; the negative ELU branch is omitted.
Exact Current/Parent traces are oracle diagnostic inputs only.  This module
does not claim that dense closure meets EvoKV's 20% release-time budget.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HistoryLayerTrace:
    """One exact or approximate full-history layer in head layout."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    response_heads: torch.Tensor
    hidden_in: torch.Tensor
    hidden_out: torch.Tensor


@dataclass(frozen=True)
class HistoricalProbeRegion:
    """One causal, shared key-region estimate from fixed history queries."""

    query_positions: torch.Tensor
    positive_mask: torch.Tensor
    eligible_probe_counts: torch.Tensor
    positive_probe_counts: torch.Tensor


@dataclass(frozen=True)
class PrefixAffineMoments:
    """Cumulative positive-branch affine summaries for every prefix."""

    base: torch.Tensor
    linear: torch.Tensor
    positive_mask: torch.Tensor
    causal_diagonal: str


@dataclass(frozen=True)
class AffineMomentSummary:
    """Associative summary of one contiguous history segment."""

    base: torch.Tensor
    linear: torch.Tensor


@dataclass(frozen=True)
class PrefixMomentRollout:
    """Closed dense rollout using its own historical-query regions."""

    layers: tuple[HistoryLayerTrace, ...]
    regions: tuple[HistoricalProbeRegion, ...]
    final_hidden: torch.Tensor


@dataclass(frozen=True)
class HistoricalPrefixPairDiagnostic:
    """Complete per-layer evidence for one Parent/Current history pair."""

    layer_head_records: tuple[dict[str, Any], ...]
    cost: dict[str, int | float | str]


def _validate_legacy_model(model) -> None:
    if model.training:
        raise ValueError("historical prefix-moment diagnostics require model.eval()")
    if not model.blocks:
        raise ValueError("model must contain at least one block")
    for block in model.blocks:
        if block.attn.activation != "elu_plus1":
            raise ValueError("prefix affine moments require legacy ELU+1 attention")
        if block.attn.block_variant != "legacy":
            raise ValueError("prefix affine moments require the legacy block")
        if block.attn.position_bias is not None:
            raise ValueError("prefix affine moments require no position bias")
        if block.attn.causal_diagonal not in {"inclusive", "exclusive"}:
            raise ValueError("unsupported causal diagonal")


def _validate_heads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q/K/V must share [B,H,N,D] layout")
    if not q.is_floating_point() or not k.is_floating_point() or not v.is_floating_point():
        raise ValueError("Q/K/V must be floating point")


def _causal_keep(length: int, diagonal: str, device: torch.device) -> torch.Tensor:
    if diagonal not in {"inclusive", "exclusive"}:
        raise ValueError("causal diagonal must be inclusive or exclusive")
    offset = 0 if diagonal == "inclusive" else -1
    return torch.ones(length, length, dtype=torch.bool, device=device).tril(offset)


def fixed_historical_query_positions(
    history_length: int,
    probe_count: int,
    *,
    layout: str = "equal_width_endpoints",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return label-free historical query rows, always including the last row."""

    if not 1 <= probe_count <= history_length:
        raise ValueError("probe_count must be in [1, history_length]")
    if layout == "equal_width_endpoints":
        positions = (
            torch.arange(1, probe_count + 1, device=device, dtype=torch.long)
            * history_length
            // probe_count
            - 1
        )
    elif layout == "recent_tail":
        positions = torch.arange(
            history_length - probe_count,
            history_length,
            device=device,
            dtype=torch.long,
        )
    else:
        raise ValueError("probe layout must be equal_width_endpoints or recent_tail")
    if torch.unique(positions).numel() != probe_count or positions[-1] != history_length - 1:
        raise RuntimeError("historical probe positions are not unique or tail-covering")
    return positions


def historical_probe_region(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    scale: float,
    probe_count: int,
    layout: str = "equal_width_endpoints",
    causal_diagonal: str = "inclusive",
) -> HistoricalProbeRegion:
    """Estimate one key mask per head using only causally eligible probe rows."""

    if q.shape != k.shape or q.ndim != 4:
        raise ValueError("Q/K must share [B,H,N,D] layout")
    if causal_diagonal != "inclusive":
        # The Medium path is inclusive.  Supporting exclusive attention would
        # require a convention for the last key, which no historical query can
        # legally classify; reject it rather than silently inventing one.
        raise ValueError("historical probe region currently requires inclusive causality")
    length = q.shape[2]
    positions = fixed_historical_query_positions(
        length, probe_count, layout=layout, device=q.device
    )
    probe_q = q.index_select(2, positions)
    logits = torch.matmul(probe_q, k.transpose(-2, -1)) * scale
    keys = torch.arange(length, device=q.device)
    eligible = keys.unsqueeze(0) <= positions.unsqueeze(1)
    eligible_counts = eligible.sum(dim=0)
    if bool((eligible_counts < 1).any()):
        raise RuntimeError("tail query must classify every historical key")
    positives = ((logits >= 0) & eligible[None, None]).sum(dim=2)
    mask = 2 * positives >= eligible_counts[None, None]
    return HistoricalProbeRegion(
        query_positions=positions.detach(),
        positive_mask=mask.detach(),
        eligible_probe_counts=eligible_counts.detach(),
        positive_probe_counts=positives.detach(),
    )


def activation_region_statistics(
    q: torch.Tensor,
    k: torch.Tensor,
    region: HistoricalProbeRegion,
    *,
    scale: float,
    causal_diagonal: str = "inclusive",
) -> dict[str, torch.Tensor]:
    """Measure shared-region agreement against all historical queries."""

    if q.shape != k.shape or q.ndim != 4:
        raise ValueError("Q/K must share [B,H,N,D] layout")
    batch, heads, length, _ = q.shape
    if region.positive_mask.shape != (batch, heads, length):
        raise ValueError("region mask shape differs from Q/K")
    causal = _causal_keep(length, causal_diagonal, q.device)
    logits = torch.matmul(q, k.transpose(-2, -1)) * scale
    signs = logits >= 0
    shared = region.positive_mask.unsqueeze(2).expand_as(signs)
    keep = causal[None, None].expand(batch, heads, -1, -1)
    denominator = keep.sum(dim=(0, 2, 3)).clamp_min(1)
    agreement = ((signs == shared) & keep).sum(dim=(0, 2, 3)) / denominator
    negative = ((~signs) & keep).sum(dim=(0, 2, 3)) / denominator

    # Full-history majority is an oracle diagnostic, never the executable mask.
    full_positive_votes = (signs & keep).sum(dim=2)
    full_eligible = keep.sum(dim=2)
    full_majority = 2 * full_positive_votes >= full_eligible
    majority_agreement = (full_majority == region.positive_mask).float().mean(dim=(0, 2))
    unanimous = (full_positive_votes == 0) | (full_positive_votes == full_eligible)
    unanimous_fraction = unanimous.float().mean(dim=(0, 2))

    by_query_denominator = keep.sum(dim=3).clamp_min(1)
    by_query_agreement = ((signs == shared) & keep).sum(dim=3) / by_query_denominator
    return {
        "shared_region_pair_agreement": agreement.float(),
        "shared_region_query_p10_agreement": torch.quantile(
            by_query_agreement.float().permute(1, 0, 2).reshape(heads, -1),
            0.10,
            dim=1,
        ),
        "probe_vs_full_majority_key_agreement": majority_agreement.float(),
        "unanimous_key_fraction": unanimous_fraction.float(),
        "negative_logit_pair_fraction": negative.float(),
    }


def build_prefix_affine_moments(
    k: torch.Tensor,
    v: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    causal_diagonal: str = "inclusive",
) -> PrefixAffineMoments:
    """Build all-prefix affine summaries with dense associative scans."""

    if k.shape != v.shape or k.ndim != 4:
        raise ValueError("K/V must share [B,H,N,D] layout")
    if positive_mask.shape != k.shape[:3]:
        raise ValueError("positive mask must have shape [B,H,N]")
    masked_v = v * positive_mask.unsqueeze(-1).to(dtype=v.dtype)
    base = torch.cumsum(masked_v, dim=2)
    atoms = torch.einsum("bhnd,bhne->bhnde", k, masked_v)
    linear = torch.cumsum(atoms, dim=2)
    if causal_diagonal == "exclusive":
        base = torch.cat((torch.zeros_like(base[:, :, :1]), base[:, :, :-1]), dim=2)
        linear = torch.cat((torch.zeros_like(linear[:, :, :1]), linear[:, :, :-1]), dim=2)
    elif causal_diagonal != "inclusive":
        raise ValueError("causal diagonal must be inclusive or exclusive")
    return PrefixAffineMoments(
        base=base,
        linear=linear,
        positive_mask=positive_mask,
        causal_diagonal=causal_diagonal,
    )


def read_prefix_affine_moments(
    q: torch.Tensor,
    moments: PrefixAffineMoments,
    *,
    scale: float,
) -> torch.Tensor:
    """Read all causal prefixes in ``O(N H D^2)`` rather than ``O(N^2 H D)``."""

    if q.ndim != 4 or moments.base.shape != q.shape:
        raise ValueError("Q and prefix-base layouts differ")
    if moments.linear.shape != (*q.shape, q.shape[-1]):
        raise ValueError("prefix-linear layout differs")
    return moments.base + scale * torch.einsum("bhnd,bhnde->bhne", q, moments.linear)


def summarize_affine_segment(
    k: torch.Tensor,
    v: torch.Tensor,
    positive_mask: torch.Tensor,
) -> AffineMomentSummary:
    """Reduce one segment to an associative ``(B,M)`` pair."""

    if k.shape != v.shape or k.ndim != 4 or positive_mask.shape != k.shape[:3]:
        raise ValueError("segment K/V/mask layouts differ")
    masked_v = v * positive_mask.unsqueeze(-1).to(dtype=v.dtype)
    return AffineMomentSummary(
        base=masked_v.sum(dim=2),
        linear=torch.einsum("bhnd,bhne->bhde", k, masked_v),
    )


def combine_affine_summaries(
    left: AffineMomentSummary,
    right: AffineMomentSummary,
) -> AffineMomentSummary:
    """Associative concatenation operator for two adjacent summaries."""

    if left.base.shape != right.base.shape or left.linear.shape != right.linear.shape:
        raise ValueError("affine summary layouts differ")
    return AffineMomentSummary(
        base=left.base + right.base,
        linear=left.linear + right.linear,
    )


def exact_history_response_heads(
    attention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Return the model-native causal history response before output projection."""

    _validate_heads(q, k, v)
    length = q.shape[2]
    positions = torch.arange(length, device=q.device)
    keep = _causal_keep(length, attention.causal_diagonal, q.device)
    return attention._aggregate(
        q,
        k,
        v,
        positions,
        positions,
        keep[None, None].to(dtype=q.dtype),
    )


def exact_positive_branch_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal_diagonal: str = "inclusive",
) -> torch.Tensor:
    """Return the exact query-specific positive ELU+1 branch."""

    _validate_heads(q, k, v)
    length = q.shape[2]
    logits = torch.matmul(q, k.transpose(-2, -1)) * scale
    keep = _causal_keep(length, causal_diagonal, q.device)
    weights = (1.0 + logits) * (logits >= 0) * keep[None, None]
    return torch.matmul(weights.to(dtype=v.dtype), v)


def _block_update(block, x_norm: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
    attention_out = block.attn._finish(heads)
    if block.gating == "silu_gate":
        return attention_out * F.silu(block.gate_proj(x_norm))
    if block.gating == "glu":
        return attention_out * torch.sigmoid(block.gate_proj(x_norm))
    if block.gating == "ffn":
        return block.fc2(F.silu(block.fc1(x_norm)) * block.fc3(x_norm))
    return attention_out


@torch.inference_mode()
def trace_exact_history_layers(
    model,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
) -> tuple[HistoryLayerTrace, ...]:
    """Trace exact full-history Q/K/V and response heads at every layer."""

    _validate_legacy_model(model)
    x = model.embed_inputs(item_ids, behaviors, time_deltas)
    traces: list[HistoryLayerTrace] = []
    for block in model.blocks:
        hidden_in = x
        x_norm = block.norm(x)
        q, k, v = block.attn._project(x_norm)
        heads = exact_history_response_heads(block.attn, q, k, v)
        x = hidden_in + _block_update(block, x_norm, heads)
        traces.append(
            HistoryLayerTrace(
                q=q.detach(),
                k=k.detach(),
                v=v.detach(),
                response_heads=heads.detach(),
                hidden_in=hidden_in.detach(),
                hidden_out=x.detach(),
            )
        )
    return tuple(traces)


@torch.inference_mode()
def rollout_dense_prefix_moments(
    model,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    *,
    probe_count: int,
    probe_layout: str = "equal_width_endpoints",
) -> PrefixMomentRollout:
    """Close the approximate reader over all history tokens layer by layer."""

    _validate_legacy_model(model)
    x = model.embed_inputs(item_ids, behaviors, time_deltas)
    traces: list[HistoryLayerTrace] = []
    regions: list[HistoricalProbeRegion] = []
    for block in model.blocks:
        hidden_in = x
        x_norm = block.norm(x)
        q, k, v = block.attn._project(x_norm)
        region = historical_probe_region(
            q,
            k,
            scale=block.attn.scale,
            probe_count=probe_count,
            layout=probe_layout,
            causal_diagonal=block.attn.causal_diagonal,
        )
        moments = build_prefix_affine_moments(
            k,
            v,
            region.positive_mask,
            causal_diagonal=block.attn.causal_diagonal,
        )
        heads = read_prefix_affine_moments(q, moments, scale=block.attn.scale)
        x = hidden_in + _block_update(block, x_norm, heads)
        traces.append(
            HistoryLayerTrace(
                q=q.detach(),
                k=k.detach(),
                v=v.detach(),
                response_heads=heads.detach(),
                hidden_in=hidden_in.detach(),
                hidden_out=x.detach(),
            )
        )
        regions.append(region)
    return PrefixMomentRollout(
        layers=tuple(traces), regions=tuple(regions), final_hidden=x.detach()
    )


def _relative_recovery(
    reference: torch.Tensor,
    observed: torch.Tensor,
    *,
    dimensions: tuple[int, ...],
    eps: float = 1e-20,
) -> torch.Tensor:
    error = torch.linalg.vector_norm(reference - observed, dim=dimensions)
    norm = torch.linalg.vector_norm(reference, dim=dimensions).clamp_min(eps)
    return 1.0 - error / norm


@torch.inference_mode()
def paired_teacher_forced_diagnostics(
    model,
    current: tuple[HistoryLayerTrace, ...],
    parent: tuple[HistoryLayerTrace, ...],
    *,
    probe_count: int,
    probe_layout: str = "equal_width_endpoints",
) -> list[dict[str, Any]]:
    """Measure paired Current-minus-Parent response recovery per layer/head.

    The Current exact historical Q is teacher-forced into both K/V arms.  This
    isolates functional state representability from rollout error.
    """

    _validate_legacy_model(model)
    if len(current) != len(model.blocks) or len(parent) != len(model.blocks):
        raise ValueError("trace and model layer counts differ")
    records: list[dict[str, Any]] = []
    for layer, (block, current_layer, parent_layer) in enumerate(
        zip(model.blocks, current, parent, strict=True)
    ):
        q = current_layer.q
        current_region = historical_probe_region(
            q,
            current_layer.k,
            scale=block.attn.scale,
            probe_count=probe_count,
            layout=probe_layout,
            causal_diagonal=block.attn.causal_diagonal,
        )
        parent_region = historical_probe_region(
            q,
            parent_layer.k,
            scale=block.attn.scale,
            probe_count=probe_count,
            layout=probe_layout,
            causal_diagonal=block.attn.causal_diagonal,
        )
        current_moments = build_prefix_affine_moments(
            current_layer.k,
            current_layer.v,
            current_region.positive_mask,
            causal_diagonal=block.attn.causal_diagonal,
        )
        parent_moments = build_prefix_affine_moments(
            parent_layer.k,
            parent_layer.v,
            parent_region.positive_mask,
            causal_diagonal=block.attn.causal_diagonal,
        )
        current_exact = exact_history_response_heads(
            block.attn, q, current_layer.k, current_layer.v
        )
        parent_exact = exact_history_response_heads(block.attn, q, parent_layer.k, parent_layer.v)
        exact_delta = current_exact - parent_exact
        approximate_delta = read_prefix_affine_moments(
            q, current_moments, scale=block.attn.scale
        ) - read_prefix_affine_moments(q, parent_moments, scale=block.attn.scale)
        delta_recovery = _relative_recovery(exact_delta, approximate_delta, dimensions=(0, 2, 3))
        delta_cosine = F.cosine_similarity(
            exact_delta.permute(1, 0, 2, 3).reshape(q.shape[1], -1),
            approximate_delta.permute(1, 0, 2, 3).reshape(q.shape[1], -1),
            dim=1,
            eps=1e-20,
        )
        current_positive = exact_positive_branch_heads(
            q,
            current_layer.k,
            current_layer.v,
            scale=block.attn.scale,
            causal_diagonal=block.attn.causal_diagonal,
        )
        negative_fraction = torch.linalg.vector_norm(
            current_exact - current_positive, dim=(0, 2, 3)
        ) / torch.linalg.vector_norm(current_exact, dim=(0, 2, 3)).clamp_min(1e-20)
        current_stats = activation_region_statistics(
            q,
            current_layer.k,
            current_region,
            scale=block.attn.scale,
            causal_diagonal=block.attn.causal_diagonal,
        )
        parent_stats = activation_region_statistics(
            q,
            parent_layer.k,
            parent_region,
            scale=block.attn.scale,
            causal_diagonal=block.attn.causal_diagonal,
        )
        cross_version_region_agreement = (
            (current_region.positive_mask == parent_region.positive_mask).float().mean(dim=(0, 2))
        )
        for head in range(q.shape[1]):
            records.append(
                {
                    "layer": layer,
                    "head": head,
                    "probe_count": probe_count,
                    "probe_layout": probe_layout,
                    "teacher_forced_paired_response_recovery": float(delta_recovery[head]),
                    "teacher_forced_paired_response_cosine": float(delta_cosine[head]),
                    "Current_negative_response_norm_fraction": float(negative_fraction[head]),
                    "Current_Parent_probe_region_agreement": float(
                        cross_version_region_agreement[head]
                    ),
                    **{
                        f"Current_{name}": float(values[head])
                        for name, values in current_stats.items()
                    },
                    **{
                        f"Parent_{name}": float(values[head])
                        for name, values in parent_stats.items()
                    },
                }
            )
    return records


def exact_to_rollout_layer_diagnostics(
    exact: tuple[HistoryLayerTrace, ...],
    rollout: PrefixMomentRollout,
) -> list[dict[str, float | int]]:
    """Report error accumulation after each closed approximate layer."""

    if len(exact) != len(rollout.layers):
        raise ValueError("exact and rollout layer counts differ")
    rows: list[dict[str, float | int]] = []
    for layer, (reference, observed) in enumerate(zip(exact, rollout.layers, strict=True)):
        response_recovery = _relative_recovery(
            reference.response_heads,
            observed.response_heads,
            dimensions=(0, 2, 3),
        )
        key_recovery = _relative_recovery(reference.k, observed.k, dimensions=(0, 2, 3))
        hidden_recovery = _relative_recovery(
            reference.hidden_out, observed.hidden_out, dimensions=(0, 1, 2)
        )
        for head in range(reference.q.shape[1]):
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "closed_response_recovery": float(response_recovery[head]),
                    "closed_key_recovery": float(key_recovery[head]),
                    "closed_hidden_recovery": float(hidden_recovery),
                }
            )
    return rows


@torch.inference_mode()
def diagnose_model_pair(
    parent_model,
    current_model,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    *,
    probe_count: int,
    probe_layout: str = "equal_width_endpoints",
) -> HistoricalPrefixPairDiagnostic:
    """Run the minimal teacher-forced plus closed-rollout diagnostic."""

    _validate_legacy_model(parent_model)
    _validate_legacy_model(current_model)
    if len(parent_model.blocks) != len(current_model.blocks):
        raise ValueError("Parent and Current layer counts differ")
    if parent_model.cfg.hidden_size != current_model.cfg.hidden_size:
        raise ValueError("Parent and Current hidden widths differ")
    if parent_model.cfg.num_heads != current_model.cfg.num_heads:
        raise ValueError("Parent and Current head counts differ")
    current = trace_exact_history_layers(current_model, item_ids, behaviors, time_deltas)
    parent = trace_exact_history_layers(parent_model, item_ids, behaviors, time_deltas)
    teacher = paired_teacher_forced_diagnostics(
        current_model,
        current,
        parent,
        probe_count=probe_count,
        probe_layout=probe_layout,
    )
    rollout = rollout_dense_prefix_moments(
        current_model,
        item_ids,
        behaviors,
        time_deltas,
        probe_count=probe_count,
        probe_layout=probe_layout,
    )
    closed = exact_to_rollout_layer_diagnostics(current, rollout)
    closed_by_key = {(int(row["layer"]), int(row["head"])): row for row in closed}
    combined: list[dict[str, Any]] = []
    for row in teacher:
        key = (int(row["layer"]), int(row["head"]))
        companion = closed_by_key[key]
        combined.append(
            {
                **row,
                **{
                    name: value
                    for name, value in companion.items()
                    if name not in {"layer", "head"}
                },
            }
        )
    cost = dense_prefix_moment_cost(
        layers=len(current_model.blocks),
        hidden=current_model.cfg.hidden_size,
        heads=current_model.cfg.num_heads,
        context=item_ids.shape[1],
        probes=probe_count,
        temporal_freqs=current_model.cfg.temporal_num_freqs,
    )
    return HistoricalPrefixPairDiagnostic(layer_head_records=tuple(combined), cost=cost)


def dense_prefix_moment_cost(
    *,
    layers: int,
    hidden: int,
    heads: int,
    context: int,
    probes: int,
    temporal_freqs: int = 16,
) -> dict[str, int | float | str]:
    """Strict matmul/add FLOPs for paired all-history dense closure.

    The cost includes full-token block transforms, two-arm probe classification,
    Current/Parent prefix-moment construction and two-arm prefix reads.  It
    excludes Exact oracle attention, labels, and candidate evaluation.  ELU,
    comparisons and indexing are reported as non-matmul operations rather than
    silently counted as zero-cost evidence.
    """

    if min(layers, hidden, heads, context, probes, temporal_freqs) < 1:
        raise ValueError("architecture and diagnostic sizes must be positive")
    if hidden % heads:
        raise ValueError("hidden size must be divisible by heads")
    if probes > context:
        raise ValueError("probe count cannot exceed context")
    head_dim = hidden // heads

    def input_projection(tokens: int) -> int:
        return 2 * tokens * (2 * temporal_freqs) * hidden + 2 * tokens * hidden * hidden

    def block_linear(tokens: int) -> int:
        # Q/K/V, output projection and gate projection.
        return 2 * tokens * (5 * hidden * hidden)

    def exact_attention(pairs: int) -> int:
        return 4 * pairs * hidden

    causal_pairs = context * (context + 1) // 2
    exact = input_projection(context) + layers * (
        block_linear(context) + exact_attention(causal_pairs)
    )
    dense_transform_floor = input_projection(context) + layers * block_linear(context)

    # Two K arms (Current and Parent) are classified by the same historical Q.
    probe_qk = 4 * probes * context * hidden
    # Per arm: mask V + base scan (2NH), K outer V + matrix scan (2NHd).
    two_arm_moment_build = 4 * context * hidden + 4 * context * heads * head_dim**2
    # Two qM reads plus base additions and Current-minus-Parent subtraction.
    two_arm_reads = 4 * context * heads * head_dim**2 + 3 * context * hidden
    prefix_per_layer = probe_qk + two_arm_moment_build + two_arm_reads
    total = dense_transform_floor + layers * prefix_per_layer
    comparisons = 2 * layers * heads * probes * context
    persistent_scalars = layers * heads * (head_dim + head_dim**2)
    materialized_prefix_scalars = 2 * heads * context * (head_dim + head_dim**2)
    fused_scan_accumulator_scalars = 2 * heads * (head_dim + head_dim**2)
    return {
        "layers": layers,
        "hidden": hidden,
        "heads": heads,
        "head_dim": head_dim,
        "context": context,
        "historical_probes": probes,
        "full_exact_recompute_flops_per_user": exact,
        "dense_token_transform_floor_flops_per_user": dense_transform_floor,
        "historical_probe_qk_flops_per_layer": probe_qk,
        "two_arm_prefix_moment_build_flops_per_layer": two_arm_moment_build,
        "two_arm_prefix_read_flops_per_layer": two_arm_reads,
        "prefix_reduction_flops_per_layer": prefix_per_layer,
        "total_dense_prefix_closure_flops_per_user": total,
        "dense_transform_floor_over_exact": dense_transform_floor / exact,
        "total_over_exact": total / exact,
        "activation_sign_comparisons_not_counted_as_flops": comparisons,
        "persistent_delta_moment_scalars": persistent_scalars,
        "persistent_delta_moment_ratio_to_full_Current_KV": persistent_scalars
        / (2 * layers * context * hidden),
        "materialized_two_arm_all_prefix_summary_scalars_one_layer": (materialized_prefix_scalars),
        "fused_two_arm_scan_accumulator_scalars": fused_scan_accumulator_scalars,
        "support_semantics": "all_history_dense_reduction_no_token_subset",
        "cost_semantics": (
            "paired_dense_prefix_closure_includes_full_token_transforms_and_"
            "two_arm_reduction_excludes_oracle_attention"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-medium-cost",
        action="store_true",
        help="print the static 6L/H192/N1024 P8/P32 cost audit",
    )
    args = parser.parse_args()
    if not args.print_medium_cost:
        parser.error("this module exposes tensor diagnostics; use --print-medium-cost")
    payload = {
        f"P{probes}": dense_prefix_moment_cost(
            layers=6,
            hidden=192,
            heads=6,
            context=1024,
            probes=probes,
        )
        for probes in (8, 32)
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
