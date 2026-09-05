"""Cross-version attention activation-boundary diagnostics and replay.

The legacy Medium checkpoints use pointwise ``ELU(z)+1`` attention.  Zero is
therefore a real computational boundary: positive interactions are affine in
``z`` while negative interactions are exponential.  This module asks whether
the *change set* of that interaction graph is a useful migration object.

Three evidence levels are kept separate:

* ``trace_exact_endpoint_graphs`` is an Exact-Parent/Exact-Current oracle.  It
  measures sign agreement and partitions the finite response delta into pairs
  which keep or cross the ELU activation boundary.
* ``intervene_serving_boundary_delta`` is a Current-cache oracle.  It tests
  whether correcting only crossing pairs is causally sufficient for serving.
* ``build_no_target_boundary_replay_cache`` is a no-target recurrence.  It
  reads raw history and Parent K/V, forms Current historical queries, reuses
  Parent contributions on same-region pairs and uses recursively generated
  Current contributions on crossing pairs.

The last path is executable algebraically, but its dense interaction-graph
discovery cost is audited independently.  No rank, fitted map, sampled token,
label, candidate anchor, or Current Exact target enters that constructor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch

from hstu_kvcache.models import HSTUKVCache
from insight_two.cone_response_memory import _block_update, _native_self_heads

BoundaryServingMode = Literal[
    "parent",
    "current",
    "crossing_delta_only",
    "unchanged_delta_only",
]


@dataclass(frozen=True)
class InteractionGraphLayerMetrics:
    """One historical layer's exact endpoint graph comparison."""

    layer: int
    causal_pairs_per_head: int
    activation_region_agreement: float
    activation_region_crossing_fraction: float
    current_activation_mass_on_crossings: float
    parent_activation_mass_on_crossings: float
    activation_change_l1_on_crossings: float
    current_response_norm_on_crossings_over_full: float
    finite_response_delta_crossing_over_joint: float
    finite_response_delta_unchanged_over_joint: float
    crossing_delta_response_gap_recovery: float
    unchanged_delta_response_gap_recovery: float
    decomposition_relative_l2_error: float


@dataclass(frozen=True)
class ExactEndpointGraphTrace:
    """Exact caches plus per-layer interaction-graph diagnostics."""

    parent_cache: HSTUKVCache
    current_cache: HSTUKVCache
    layer_metrics: tuple[InteractionGraphLayerMetrics, ...]


@dataclass(frozen=True)
class BoundaryReplayResult:
    """No-target recursively generated cache and its graph statistics."""

    cache: HSTUKVCache
    crossing_fraction_by_active_layer: tuple[float, ...]


@dataclass(frozen=True)
class BoundaryServingIntervention:
    scores: torch.Tensor
    crossing_fraction_by_layer: tuple[float, ...]


@dataclass(frozen=True)
class ActivationBoundaryCostAudit:
    """Generous Medium lower bounds for exact boundary discovery."""

    context: int
    hidden: int
    heads: int
    layers: int
    active_layers: int
    exact_all_flops: int
    causal_pairs_per_layer_per_head: int
    current_graph_qk_floor_flops: int
    current_graph_qk_floor_over_exact: float
    no_target_two_graph_one_value_floor_flops: int
    no_target_two_graph_one_value_floor_over_exact: float
    parent_graph_bits_if_persisted: int
    parent_graph_bytes_if_persisted: int
    parent_graph_over_parent_kv_fp32: float
    within_twenty_percent_before_response_or_projection: bool
    verdict: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


def _validate_legacy_model(model) -> None:
    if model.training:
        raise ValueError("activation-boundary diagnostics require model.eval()")
    if not model.blocks:
        raise ValueError("model must contain at least one block")
    for block in model.blocks:
        attention = block.attn
        if attention.activation != "elu_plus1":
            raise ValueError("activation boundary is defined for ELU+1 checkpoints")
        if attention.block_variant != "legacy":
            raise ValueError("this diagnostic is fixed to the legacy Medium block")
        if attention.position_bias is not None:
            raise ValueError("relative position bias changes the audited boundary")
        if attention.causal_diagonal != "inclusive":
            raise ValueError("the Medium workload requires inclusive causal attention")


def _validate_history(
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
) -> None:
    if item_ids.ndim != 2 or item_ids.shape[0] != 1:
        raise ValueError("single-user history must have shape [1,N]")
    if behaviors.shape != item_ids.shape or time_deltas.shape != item_ids.shape:
        raise ValueError("history fields have different shapes")
    if item_ids.shape[1] < 1:
        raise ValueError("history must be non-empty")


def _heads_to_cache(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 4:
        raise ValueError("head tensor must have shape [B,H,N,D]")
    batch, heads, length, width = values.shape
    return values.transpose(1, 2).reshape(batch, length, heads * width)


def _cache_to_heads(attention, values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or values.shape[-1] != attention.inner:
        raise ValueError("cache layer must have shape [B,N,attention.inner]")
    batch, length, _ = values.shape
    return values.view(batch, length, attention.num_heads, attention.head_dim).transpose(1, 2)


def _causal_keep(
    length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.ones(length, length, device=device, dtype=dtype).tril()[None, None]


def _scaled_logits(attention, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.matmul(q, k.transpose(-2, -1)) * attention.scale


def _history_heads(
    attention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    length = q.shape[2]
    keep = _causal_keep(length, device=q.device, dtype=q.dtype)
    weights = attention._activate(_scaled_logits(attention, q, k)) * keep
    return torch.matmul(attention.attn_dropout(weights), v)


def _relative_norm(values: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(values.float())
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-20)
    return float(numerator / denominator)


@torch.inference_mode()
def compare_endpoint_interaction_graphs(
    parent_attention,
    current_attention,
    parent_q: torch.Tensor,
    parent_k: torch.Tensor,
    parent_v: torch.Tensor,
    current_q: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    *,
    layer: int,
) -> InteractionGraphLayerMetrics:
    """Partition an exact finite endpoint response delta by ELU branch change."""

    shape = parent_q.shape
    if shape != current_q.shape or parent_k.shape != current_k.shape:
        raise ValueError("Parent and Current Q/K shapes differ")
    if parent_v.shape != current_v.shape or parent_v.shape != parent_k.shape:
        raise ValueError("Parent and Current K/V shapes differ")
    if shape[0] != 1 or shape[2] != parent_k.shape[2]:
        raise ValueError("historical endpoint trace must be one square sequence")
    length = shape[2]
    keep = _causal_keep(length, device=parent_q.device, dtype=parent_q.dtype)
    valid = keep.to(dtype=torch.bool).expand(1, parent_attention.num_heads, length, length)

    parent_logits = _scaled_logits(parent_attention, parent_q, parent_k)
    current_logits = _scaled_logits(current_attention, current_q, current_k)
    same = ((parent_logits >= 0) == (current_logits >= 0)) & valid
    crossing = (~same) & valid
    parent_weights = parent_attention._activate(parent_logits) * keep
    current_weights = current_attention._activate(current_logits) * keep

    same_float = same.to(dtype=parent_weights.dtype)
    crossing_float = crossing.to(dtype=parent_weights.dtype)
    parent_same = torch.matmul(parent_weights * same_float, parent_v)
    parent_crossing = torch.matmul(parent_weights * crossing_float, parent_v)
    current_same = torch.matmul(current_weights * same_float, current_v)
    current_crossing = torch.matmul(current_weights * crossing_float, current_v)
    parent_full = parent_same + parent_crossing
    current_full = current_same + current_crossing
    joint_delta = current_full - parent_full
    unchanged_delta = current_same - parent_same
    crossing_delta = current_crossing - parent_crossing
    decomposition_error = joint_delta - unchanged_delta - crossing_delta

    valid_count = valid.sum().clamp_min(1)
    current_mass = current_weights.masked_select(valid).sum().clamp_min(1e-20)
    parent_mass = parent_weights.masked_select(valid).sum().clamp_min(1e-20)
    activation_change = torch.abs(current_weights - parent_weights)
    activation_change_total = activation_change.masked_select(valid).sum().clamp_min(1e-20)
    return InteractionGraphLayerMetrics(
        layer=layer,
        causal_pairs_per_head=length * (length + 1) // 2,
        activation_region_agreement=float(same.sum() / valid_count),
        activation_region_crossing_fraction=float(crossing.sum() / valid_count),
        current_activation_mass_on_crossings=float(
            current_weights.masked_select(crossing).sum() / current_mass
        ),
        parent_activation_mass_on_crossings=float(
            parent_weights.masked_select(crossing).sum() / parent_mass
        ),
        activation_change_l1_on_crossings=float(
            activation_change.masked_select(crossing).sum() / activation_change_total
        ),
        current_response_norm_on_crossings_over_full=_relative_norm(current_crossing, current_full),
        finite_response_delta_crossing_over_joint=_relative_norm(crossing_delta, joint_delta),
        finite_response_delta_unchanged_over_joint=_relative_norm(unchanged_delta, joint_delta),
        crossing_delta_response_gap_recovery=1.0 - _relative_norm(unchanged_delta, joint_delta),
        unchanged_delta_response_gap_recovery=1.0 - _relative_norm(crossing_delta, joint_delta),
        decomposition_relative_l2_error=_relative_norm(decomposition_error, joint_delta),
    )


@torch.inference_mode()
def trace_exact_endpoint_graphs(
    parent_model,
    current_model,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
) -> ExactEndpointGraphTrace:
    """Trace exact Parent/Current histories and compare every causal pair."""

    _validate_legacy_model(parent_model)
    _validate_legacy_model(current_model)
    _validate_history(item_ids, behaviors, time_deltas)
    if len(parent_model.blocks) != len(current_model.blocks):
        raise ValueError("Parent and Current layer counts differ")
    parent_x = parent_model.embed_inputs(item_ids, behaviors, time_deltas)
    current_x = current_model.embed_inputs(item_ids, behaviors, time_deltas)
    parent_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
    current_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
    metrics: list[InteractionGraphLayerMetrics] = []

    for layer, (parent_block, current_block) in enumerate(
        zip(parent_model.blocks, current_model.blocks, strict=True)
    ):
        parent_residual = parent_x
        current_residual = current_x
        parent_norm = parent_block.norm(parent_x)
        current_norm = current_block.norm(current_x)
        parent_q, parent_k, parent_v = parent_block.attn._project(parent_norm)
        current_q, current_k, current_v = current_block.attn._project(current_norm)
        parent_kv.append((_heads_to_cache(parent_k), _heads_to_cache(parent_v)))
        current_kv.append((_heads_to_cache(current_k), _heads_to_cache(current_v)))
        metrics.append(
            compare_endpoint_interaction_graphs(
                parent_block.attn,
                current_block.attn,
                parent_q,
                parent_k,
                parent_v,
                current_q,
                current_k,
                current_v,
                layer=layer,
            )
        )
        parent_heads = _history_heads(parent_block.attn, parent_q, parent_k, parent_v)
        current_heads = _history_heads(current_block.attn, current_q, current_k, current_v)
        parent_x = parent_residual + _block_update(parent_block, parent_norm, parent_heads)
        current_x = current_residual + _block_update(current_block, current_norm, current_heads)

    length = item_ids.shape[1]
    return ExactEndpointGraphTrace(
        parent_cache=HSTUKVCache.from_layer_list(parent_kv, seq_len=length),
        current_cache=HSTUKVCache.from_layer_list(current_kv, seq_len=length),
        layer_metrics=tuple(metrics),
    )


def _boundary_partitioned_prefix(
    attention,
    q: torch.Tensor,
    parent_k: torch.Tensor,
    parent_v: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    *,
    mode: BoundaryServingMode,
) -> tuple[torch.Tensor, float]:
    """Read a serving prefix after partitioning matched-query interactions."""

    parent_keys = _cache_to_heads(attention, parent_k).expand(q.shape[0], -1, -1, -1)
    parent_values = _cache_to_heads(attention, parent_v).expand(q.shape[0], -1, -1, -1)
    current_keys = _cache_to_heads(attention, current_k).expand(q.shape[0], -1, -1, -1)
    current_values = _cache_to_heads(attention, current_v).expand(q.shape[0], -1, -1, -1)
    parent_logits = _scaled_logits(attention, q, parent_keys)
    current_logits = _scaled_logits(attention, q, current_keys)
    crossing = (parent_logits >= 0) != (current_logits >= 0)
    same = ~crossing
    parent_weights = attention._activate(parent_logits)
    current_weights = attention._activate(current_logits)

    if mode == "parent":
        heads = torch.matmul(parent_weights, parent_values)
    elif mode == "current":
        heads = torch.matmul(current_weights, current_values)
    elif mode == "crossing_delta_only":
        heads = torch.matmul(parent_weights * same, parent_values) + torch.matmul(
            current_weights * crossing, current_values
        )
    elif mode == "unchanged_delta_only":
        heads = torch.matmul(current_weights * same, current_values) + torch.matmul(
            parent_weights * crossing, parent_values
        )
    else:
        raise ValueError(f"unsupported boundary serving mode: {mode}")
    return attention.attn_dropout(heads), float(crossing.float().mean())


@torch.inference_mode()
def intervene_serving_boundary_delta(
    model,
    current_cache: HSTUKVCache,
    parent_cache: HSTUKVCache,
    candidate_ids: torch.Tensor,
    query_time_deltas: torch.Tensor,
    *,
    mode: BoundaryServingMode,
) -> BoundaryServingIntervention:
    """Causally test the crossing-only delta with Exact Current K/V as oracle."""

    _validate_legacy_model(model)
    if current_cache.k.shape != parent_cache.k.shape:
        raise ValueError("Parent and Current cache shapes differ")
    if current_cache.v.shape != parent_cache.v.shape:
        raise ValueError("Parent and Current cache shapes differ")
    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != 1:
        raise ValueError("single-user candidate IDs must have shape [1,C]")
    candidates = candidate_ids.shape[1]
    x = model.embed_query_tokens(candidate_ids, query_time_deltas).reshape(
        candidates, 1, model.cfg.hidden_size
    )
    crossing_fractions: list[float] = []
    for layer, block in enumerate(model.blocks):
        residual = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        prefix, fraction = _boundary_partitioned_prefix(
            block.attn,
            q,
            parent_cache.k[layer],
            parent_cache.v[layer],
            current_cache.k[layer],
            current_cache.v[layer],
            mode=mode,
        )
        crossing_fractions.append(fraction)
        x = residual + _block_update(
            block,
            x_norm,
            prefix + _native_self_heads(block.attn, q, k_new, v_new),
        )
    readout = model.final_norm(x).reshape(1, candidates, model.cfg.hidden_size)
    return BoundaryServingIntervention(
        scores=model.cc_score_head(readout).squeeze(-1),
        crossing_fraction_by_layer=tuple(crossing_fractions),
    )


@torch.inference_mode()
def build_no_target_boundary_replay_cache(
    model,
    parent_cache: HSTUKVCache,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
) -> BoundaryReplayResult:
    """Recursively replay only matched-query activation-boundary crossings.

    At each non-terminal layer the Current historical query compares its
    recursively generated key against the stored Parent key.  Same-region
    pairs use the Parent K/V contribution; crossing pairs use the generated
    Current contribution.  The terminal layer only emits K/V because its
    historical attention output is not needed by the serving cache.
    """

    _validate_legacy_model(model)
    _validate_history(item_ids, behaviors, time_deltas)
    if parent_cache.k.shape[0] != len(model.blocks):
        raise ValueError("Parent cache layer count differs from model")
    if parent_cache.k.shape != parent_cache.v.shape:
        raise ValueError("Parent K/V shapes differ")
    if parent_cache.k.shape[1:3] != (1, item_ids.shape[1]):
        raise ValueError("Parent cache and raw history shapes differ")

    x = model.embed_inputs(item_ids, behaviors, time_deltas)
    layer_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
    crossing_fractions: list[float] = []
    length = item_ids.shape[1]
    keep = _causal_keep(length, device=x.device, dtype=x.dtype)
    valid = keep.to(torch.bool).expand(1, model.cfg.num_heads, length, length)

    for layer, block in enumerate(model.blocks):
        residual = x
        x_norm = block.norm(x)
        q, current_k, current_v = block.attn._project(x_norm)
        layer_kv.append((_heads_to_cache(current_k), _heads_to_cache(current_v)))
        if layer == len(model.blocks) - 1:
            break

        parent_k = _cache_to_heads(block.attn, parent_cache.k[layer])
        parent_v = _cache_to_heads(block.attn, parent_cache.v[layer])
        parent_logits = _scaled_logits(block.attn, q, parent_k)
        current_logits = _scaled_logits(block.attn, q, current_k)
        crossing = ((parent_logits >= 0) != (current_logits >= 0)) & valid
        same = (~crossing) & valid
        crossing_fractions.append(float(crossing.sum() / valid.sum().clamp_min(1)))
        parent_weights = block.attn._activate(parent_logits) * keep
        current_weights = block.attn._activate(current_logits) * keep
        heads = torch.matmul(parent_weights * same, parent_v) + torch.matmul(
            current_weights * crossing, current_v
        )
        x = residual + _block_update(block, x_norm, heads)

    return BoundaryReplayResult(
        cache=HSTUKVCache.from_layer_list(layer_kv, seq_len=length),
        crossing_fraction_by_active_layer=tuple(crossing_fractions),
    )


def medium_activation_boundary_cost_audit() -> ActivationBoundaryCostAudit:
    """Return a lower bound which grants all non-attention work for free.

    Even if the Parent sign graph were materialized at cache creation, finding
    the exact Current graph requires one Current QK contraction on all causal
    pairs in each non-terminal layer.  That operation alone exceeds 20% of
    Exact-All.  The no-target recurrence actually needs two QK contractions
    (Current-query/Parent-key and Current-query/Current-key) plus at least one
    mixed weighted-value reduction.  Projections, normalization, gates,
    output projections, mask materialization, and writes are all omitted from
    the lower bound, making the rejection conservative.
    """

    context = 1024
    hidden = 192
    heads = 6
    layers = 6
    active_layers = layers - 1
    exact_all = 4_771_282_944
    pairs = context * (context + 1) // 2
    one_graph_qk = active_layers * 2 * pairs * hidden
    boundary_replay_floor = 3 * one_graph_qk
    graph_bits = active_layers * heads * pairs
    graph_bytes = (graph_bits + 7) // 8
    parent_kv_bytes = 2 * layers * context * hidden * 4
    return ActivationBoundaryCostAudit(
        context=context,
        hidden=hidden,
        heads=heads,
        layers=layers,
        active_layers=active_layers,
        exact_all_flops=exact_all,
        causal_pairs_per_layer_per_head=pairs,
        current_graph_qk_floor_flops=one_graph_qk,
        current_graph_qk_floor_over_exact=one_graph_qk / exact_all,
        no_target_two_graph_one_value_floor_flops=boundary_replay_floor,
        no_target_two_graph_one_value_floor_over_exact=boundary_replay_floor / exact_all,
        parent_graph_bits_if_persisted=graph_bits,
        parent_graph_bytes_if_persisted=graph_bytes,
        parent_graph_over_parent_kv_fp32=graph_bytes / parent_kv_bytes,
        within_twenty_percent_before_response_or_projection=(one_graph_qk / exact_all <= 0.20),
        verdict=(
            "NO_GO: exact activation-boundary discovery already exceeds the "
            "20% migration budget before response evaluation or projections"
        ),
    )
