"""Paired functional-delta primitives for the legacy pointwise reader.

This module separates three evidence levels which must not be conflated:

* an Exact-cache paired-delta oracle;
* a legal Parent-conditioned carrier lower bound; and
* a future recursive functional-closure algorithm (not implemented here).

The paired estimators subtract Current and Parent contributions at the same
sampled history positions before aggregation.  They therefore do not repeat
the failed ``sampled Current total - complete Parent total`` construction.
The legal carrier path receives no Current Exact cache: it projects layer-0
K/V directly from raw events and obtains every upper-layer carrier by running
the Current token over the causally preceding Parent cache.

The affine moments are specific to the repository's legacy ELU+1/no-bias
reader.  They are not a theorem about faithful SiLU HSTU or softmax attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.models.state_transition import truncate_cache
from insight_two.address_response_memory import (
    AddressLandmarkSelection,
    _farthest_first_order,
    _voronoi_assignment,
)
from insight_two.attention_cone_moments import (
    build_positive_affine_moments,
    read_positive_affine_moments,
    scaled_qk_logits,
)
from insight_two.cone_response_memory import (
    ConeLayerResponseMoment,
    ConeResponseMemory,
    _block_update,
    _native_prefix_heads,
    _native_self_heads,
)


SUPPORTED_PROBE_COUNTS = (8, 32)
SUPPORTED_CARRIER_COUNTS = (64, 128)


@dataclass(frozen=True)
class CurrentLayer0Projection:
    """Dependency-free Current layer-0 K/V for every raw history event."""

    k: torch.Tensor
    v: torch.Tensor

    @property
    def source_length(self) -> int:
        return int(self.k.shape[1])


@dataclass(frozen=True)
class RegionMomentDisagreement:
    """Label-free nested-ledger disagreement read by fixed history probes."""

    relative_l2: float
    cosine: float
    maximum_absolute_difference: float
    reference_l2: float


def fixed_history_probe_positions(
    history_length: int,
    probe_count: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Equal-width lower midpoints used as label-free history-item probes."""

    if probe_count not in SUPPORTED_PROBE_COUNTS:
        raise ValueError(f"probe_count must be one of {SUPPORTED_PROBE_COUNTS}")
    if history_length < probe_count or history_length % probe_count:
        raise ValueError("history length must be a positive multiple of probe count")
    width = history_length // probe_count
    return torch.arange(probe_count, device=device, dtype=torch.long) * width + (
        width - 1
    ) // 2


def _validate_model(model) -> None:
    if model.training:
        raise ValueError("paired region delta requires model.eval()")
    if not model.blocks:
        raise ValueError("model must contain at least one block")
    for block in model.blocks:
        if block.attn.activation != "elu_plus1":
            raise ValueError("paired region delta requires legacy ELU+1 attention")
        if block.attn.block_variant != "legacy":
            raise ValueError("paired region delta requires the legacy block variant")
        if block.attn.position_bias is not None:
            raise ValueError("paired region delta requires no relative-position bias")


def _validate_parent_cache(model, cache: HSTUKVCache) -> None:
    if cache.k.ndim != 4 or cache.k.shape != cache.v.shape:
        raise ValueError("Parent cache must contain matching [L,B,N,W] K/V")
    if cache.k.shape[0] != len(model.blocks) or cache.k.shape[1] != 1:
        raise ValueError("Parent cache must contain one user and all model layers")
    if cache.k.shape[2] != cache.seq_len:
        raise ValueError("Parent cache tensor width and seq_len differ")
    if cache.k.shape[3] != model.blocks[0].attn.inner:
        raise ValueError("Parent cache width differs from model attention width")


def _validate_raw_history(
    parent_cache: HSTUKVCache,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
) -> None:
    if item_ids.ndim != 2 or item_ids.shape[0] != 1:
        raise ValueError("raw history must have shape [1,N]")
    if behaviors.shape != item_ids.shape or time_deltas.shape != item_ids.shape:
        raise ValueError("raw item, behavior, and time tensors differ")
    if item_ids.shape[1] != parent_cache.seq_len:
        raise ValueError("raw history width differs from Parent cache")


@torch.inference_mode()
def trace_history_item_region_queries(
    model,
    parent_cache: HSTUKVCache,
    history_item_ids: torch.Tensor,
    query_time_deltas: torch.Tensor,
    *,
    probe_count: int,
) -> tuple[torch.Tensor, ...]:
    """Trace Current recommendation-query Q using IDs drawn from user history.

    The probe identities come from fixed raw-history positions, not a release
    candidate panel.  They still pass through the native query-token encoder,
    because the state being certified will be read by recommendation queries.
    """

    _validate_model(model)
    _validate_parent_cache(model, parent_cache)
    if history_item_ids.shape != (1, parent_cache.seq_len):
        raise ValueError("history_item_ids must have shape [1,Parent length]")
    positions = fixed_history_probe_positions(
        parent_cache.seq_len, probe_count, device=history_item_ids.device
    )
    probes = history_item_ids.index_select(1, positions)
    x = model.embed_query_tokens(probes, query_time_deltas).reshape(
        probe_count, 1, model.cfg.hidden_size
    )
    queries: list[torch.Tensor] = []
    for layer, block in enumerate(model.blocks):
        residual = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        queries.append(q.detach())
        prefix = _native_prefix_heads(
            block.attn, q, parent_cache.k[layer], parent_cache.v[layer]
        )
        self_heads = _native_self_heads(block.attn, q, k_new, v_new)
        x = residual + _block_update(block, x_norm, prefix + self_heads)
    return tuple(queries)


@torch.inference_mode()
def project_full_current_layer0(
    model,
    parent_cache: HSTUKVCache,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
) -> CurrentLayer0Projection:
    """Project all raw events to exact Current layer-0 K/V without attention."""

    _validate_model(model)
    _validate_parent_cache(model, parent_cache)
    _validate_raw_history(parent_cache, item_ids, behaviors, time_deltas)
    embedded = model.embed_inputs(item_ids, behaviors, time_deltas)
    normalized = model.blocks[0].norm(embedded)
    k, v = model.blocks[0].attn.project_kv(normalized)
    return CurrentLayer0Projection(k=k.detach(), v=v.detach())


def select_legal_layer0_address_landmarks(
    current_layer0: CurrentLayer0Projection,
    parent_cache: HSTUKVCache,
    *,
    sample_count: int,
) -> AddressLandmarkSelection:
    """Select nested landmarks using legal Current layer-0 and Parent keys."""

    if sample_count not in SUPPORTED_CARRIER_COUNTS:
        raise ValueError(
            f"sample_count must be one of {SUPPORTED_CARRIER_COUNTS}"
        )
    if current_layer0.k.ndim != 3 or current_layer0.k.shape[0] != 1:
        raise ValueError("Current layer-0 K must have shape [1,N,W]")
    if current_layer0.k.shape != current_layer0.v.shape:
        raise ValueError("Current layer-0 K/V shapes differ")
    if current_layer0.source_length != parent_cache.seq_len:
        raise ValueError("Current layer-0 and Parent history lengths differ")
    if current_layer0.k.shape[-1] != parent_cache.k.shape[-1]:
        raise ValueError("Current layer-0 and Parent widths differ")
    features = torch.cat(
        (current_layer0.k[0].float(), parent_cache.k[0, 0].float()), dim=-1
    )
    features = torch.nn.functional.normalize(features, p=2.0, dim=-1, eps=1e-12)
    selected = _farthest_first_order(features, sample_count)
    assignments, masses = _voronoi_assignment(features, selected)
    return AddressLandmarkSelection(
        source_length=parent_cache.seq_len,
        selected_positions=selected,
        cluster_masses=masses,
        assignments=assignments,
    )


@torch.inference_mode()
def replay_parent_conditioned_current_carriers(
    model,
    parent_cache: HSTUKVCache,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    positions: torch.Tensor,
) -> HSTUKVCache:
    """Replay real Current events over exactly their causal Parent prefixes.

    Each carrier at source position ``i`` sees Parent positions ``[0,i)`` and
    its own Current self state.  Carriers do not see one another and no Exact
    Current upper-layer state is an input; recursive functional closure is a
    separate, later experiment.
    """

    _validate_model(model)
    _validate_parent_cache(model, parent_cache)
    _validate_raw_history(parent_cache, item_ids, behaviors, time_deltas)
    positions = positions.to(device=item_ids.device, dtype=torch.long)
    if positions.ndim != 1 or positions.numel() < 1:
        raise ValueError("carrier positions must be a non-empty vector")
    if torch.unique(positions).numel() != positions.numel():
        raise ValueError("carrier positions must be unique")
    if bool((positions < 0).any()) or bool((positions >= parent_cache.seq_len).any()):
        raise ValueError("carrier position is outside Parent history")

    layer_k: list[list[torch.Tensor]] = [[] for _ in model.blocks]
    layer_v: list[list[torch.Tensor]] = [[] for _ in model.blocks]
    for position_value in positions.tolist():
        position = int(position_value)
        prefix = truncate_cache(parent_cache, position)
        _, new = model.forward_with_cache_new_kv(
            prefix,
            item_ids[:, position : position + 1],
            behaviors[:, position : position + 1],
            time_deltas[:, position : position + 1],
        )
        if new.seq_len != 1:
            raise RuntimeError("one carrier replay returned more than one K/V row")
        for layer in range(len(model.blocks)):
            layer_k[layer].append(new.k[layer])
            layer_v[layer].append(new.v[layer])
    return HSTUKVCache(
        k=torch.stack([torch.cat(rows, dim=1) for rows in layer_k], dim=0),
        v=torch.stack([torch.cat(rows, dim=1) for rows in layer_v], dim=0),
        seq_len=int(positions.numel()),
    )


def _cache_heads(attention, values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != attention.inner:
        raise ValueError("sample cache layer must have shape [1,R,inner]")
    return values.view(
        1, values.shape[1], attention.num_heads, attention.head_dim
    ).transpose(1, 2)


def majority_positive_mask(
    attention,
    probe_q: torch.Tensor,
    sample_k: torch.Tensor,
) -> torch.Tensor:
    """Majority positive mask for any preregistered probe count."""

    if probe_q.ndim != 4 or probe_q.shape[2] != 1:
        raise ValueError("probe_q must have shape [P,H,1,D]")
    if probe_q.shape[1:] != (
        attention.num_heads,
        1,
        attention.head_dim,
    ):
        raise ValueError("probe query head layout differs")
    probes = probe_q.shape[0]
    if probes not in SUPPORTED_PROBE_COUNTS:
        raise ValueError(f"probe count must be one of {SUPPORTED_PROBE_COUNTS}")
    keys = _cache_heads(attention, sample_k).expand(probes, -1, -1, -1)
    positive_votes = (
        scaled_qk_logits(probe_q, keys, scale=attention.scale) >= 0
    ).sum(dim=0).squeeze(1)
    return (2 * positive_votes >= probes).unsqueeze(0)


def _weighted_moments(
    attention,
    k: torch.Tensor,
    v: torch.Tensor,
    positive_mask: torch.Tensor,
    weights: torch.Tensor,
):
    keys = _cache_heads(attention, k)
    values = _cache_heads(attention, v)
    weights = weights.to(device=values.device, dtype=values.dtype)
    if weights.shape != (values.shape[2],):
        raise ValueError("quadrature weights differ from sample width")
    return build_positive_affine_moments(
        keys,
        values * weights.view(1, 1, -1, 1),
        positive_mask,
    )


def build_paired_region_delta_memory(
    model,
    current_samples: HSTUKVCache,
    parent_cache: HSTUKVCache,
    probe_queries: tuple[torch.Tensor, ...],
    source_positions: torch.Tensor,
    weights: torch.Tensor,
) -> ConeResponseMemory:
    """Build paired sampled Current-minus-Parent affine-bulk moments."""

    _validate_model(model)
    _validate_parent_cache(model, parent_cache)
    if current_samples.k.ndim != 4 or current_samples.k.shape != current_samples.v.shape:
        raise ValueError("Current samples must contain matching [L,1,R,W] K/V")
    if current_samples.k.shape[:2] != (len(model.blocks), 1):
        raise ValueError("Current sample layer/batch layout differs")
    source_positions = source_positions.to(
        device=parent_cache.k.device, dtype=torch.long
    )
    if source_positions.shape != (current_samples.seq_len,):
        raise ValueError("source positions differ from Current sample width")
    if torch.unique(source_positions).numel() != source_positions.numel():
        raise ValueError("source positions must be unique")
    if bool((source_positions < 0).any()) or bool(
        (source_positions >= parent_cache.seq_len).any()
    ):
        raise ValueError("source position is outside Parent history")
    if len(probe_queries) != len(model.blocks):
        raise ValueError("probe query layer count differs")

    layers: list[ConeLayerResponseMoment] = []
    weights = weights.to(device=parent_cache.v.device, dtype=parent_cache.v.dtype)
    for layer, block in enumerate(model.blocks):
        current_k = current_samples.k[layer]
        current_v = current_samples.v[layer]
        parent_k = parent_cache.k[layer].index_select(1, source_positions)
        parent_v = parent_cache.v[layer].index_select(1, source_positions)
        current_mask = majority_positive_mask(
            block.attn, probe_queries[layer], current_k
        )
        parent_mask = majority_positive_mask(
            block.attn, probe_queries[layer], parent_k
        )
        current = _weighted_moments(
            block.attn, current_k, current_v, current_mask, weights
        )
        parent = _weighted_moments(
            block.attn, parent_k, parent_v, parent_mask, weights
        )
        layers.append(
            ConeLayerResponseMoment(
                base=(current.base - parent.base).detach(),
                linear=(current.linear - parent.linear).detach(),
                current_positive_mask=current_mask.detach(),
                parent_positive_mask=parent_mask.detach(),
                current_sample_positions=source_positions.detach(),
                current_sample_weights=weights.detach(),
                source_length=parent_cache.seq_len,
            )
        )
    return ConeResponseMemory(
        layers=tuple(layers),
        source_length=parent_cache.seq_len,
        anchor_count=int(probe_queries[0].shape[0]),
        source_kv_scalars=parent_cache.k.numel() + parent_cache.v.numel(),
    )


def exact_cache_samples(
    exact_cache: HSTUKVCache,
    positions: torch.Tensor,
) -> HSTUKVCache:
    """Extract oracle Current samples while retaining explicit oracle typing."""

    positions = positions.to(device=exact_cache.k.device, dtype=torch.long)
    return HSTUKVCache(
        k=exact_cache.k.index_select(2, positions),
        v=exact_cache.v.index_select(2, positions),
        seq_len=int(positions.numel()),
    )


def certify_nested_moment_disagreement(
    model,
    probe_queries: tuple[torch.Tensor, ...],
    coarse: ConeResponseMemory,
    fine: ConeResponseMemory,
    *,
    eps: float = 1e-12,
) -> RegionMomentDisagreement:
    """Compare two paired ledgers without labels or an Exact target response."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    if len(coarse.layers) != len(model.blocks) or len(fine.layers) != len(model.blocks):
        raise ValueError("memory and model layer counts differ")
    coarse_reads: list[torch.Tensor] = []
    fine_reads: list[torch.Tensor] = []
    for layer, block in enumerate(model.blocks):
        q = probe_queries[layer]

        def read(moment: ConeLayerResponseMoment) -> torch.Tensor:
            base = moment.base.expand(q.shape[0], -1, -1)
            linear = moment.linear.expand(q.shape[0], -1, -1, -1)
            return base.unsqueeze(2) + block.attn.scale * torch.einsum(
                "bhqk,bhkv->bhqv", q, linear
            )

        coarse_reads.append(read(coarse.layers[layer]).float().reshape(-1))
        fine_reads.append(read(fine.layers[layer]).float().reshape(-1))
    coarse_flat = torch.cat(coarse_reads)
    fine_flat = torch.cat(fine_reads)
    difference = fine_flat - coarse_flat
    fine_norm = torch.linalg.vector_norm(fine_flat)
    coarse_norm = torch.linalg.vector_norm(coarse_flat)
    cosine = torch.nn.functional.cosine_similarity(
        fine_flat.unsqueeze(0), coarse_flat.unsqueeze(0), dim=1, eps=eps
    )[0]
    return RegionMomentDisagreement(
        relative_l2=float((torch.linalg.vector_norm(difference) / fine_norm.clamp_min(eps)).detach()),
        cosine=float(cosine.detach()),
        maximum_absolute_difference=float(difference.abs().max().detach()),
        reference_l2=float(fine_norm.detach()),
    )


def _address_selection_operation_counts(
    *,
    context: int,
    hidden: int,
    carriers: int,
) -> dict[str, int]:
    """Arithmetic counts for the executable layer-0 address selector."""

    # Address feature width is 2H: Current layer-0 K concatenated with Parent
    # layer-0 K. For a W-vector, squared Euclidean distance performs W
    # subtractions, W squares, and W-1 reduction adds: 3W-1 FLOPs.
    address_width = 2 * hidden
    distance_flops = context * (3 * address_width - 1)
    normalization = 3 * context * address_width
    mean = context * address_width
    # mean-distance + initial-center distance + R-1 incremental distances.
    farthest_first = (carriers + 1) * distance_flops
    voronoi = carriers * distance_flops
    total = normalization + mean + farthest_first + voronoi
    # This deliberately excludes argsort, whose comparison count depends on
    # the backend. It is a transparent lower bound for the non-FLOP selection
    # work, not part of the FLOP numerator.
    comparison_lower_bound = (
        context  # normalization clamp
        + (context - 1)  # first-center argmax
        + (carriers - 1) * (2 * context - 1)  # argmax + running minimum
        + context * (carriers - 1)  # Voronoi argmin
    )
    return {
        "address_width": address_width,
        "normalization": normalization,
        "mean": mean,
        "farthest_first": farthest_first,
        "voronoi": voronoi,
        "total": total,
        "comparison_lower_bound": comparison_lower_bound,
    }


def paired_region_delta_cost(
    *,
    layers: int,
    hidden: int,
    heads: int,
    context: int,
    carriers: int,
    probes: int,
    carrier_position_sum: int | None = None,
    temporal_freqs: int = 16,
) -> dict[str, int | float | bool | str | list[int]]:
    """Audit release-time compute and storage for one paired-delta ledger.

    ``neural_generation_flops_per_user`` follows the repository's established
    multiply-add convention for dense projections and attention.  It includes
    the complete implementation cost of ``project_full_current_layer0``.  In
    particular, the current public ``PointwiseAttention.project_kv`` helper
    calls ``_project`` and therefore executes Q, K, and V projections even
    though it returns only K/V.  The discarded Q projection is exposed as its
    own field; it is never silently treated as free.

    ``address_selection_flops_per_user`` separately counts the floating-point
    arithmetic executed by L2 normalization, farthest-first traversal, and
    the final Voronoi assignment over the concatenated Current/Parent layer-0
    key address.  Tensor indexing, casts, integer reductions, comparisons,
    and sorting are not FLOPs; a lower bound on selection comparisons is
    reported separately.  ``total_generation_flops_per_user`` and the legacy
    ``over_full_fraction`` alias both include neural *and* selection FLOPs.

    When ``carrier_position_sum`` is omitted, the function reports the
    equal-coverage expectation ``R*(N-1)/2``.  That is an estimate, not a
    per-user budget certificate.  A formal run must pass the observed sum of
    its selected source positions and gate on the resulting total fraction.

    This function prices one R-carrier ledger only.  It does not pretend that
    the second ledger needed by a nested R64/R128 certificate is already
    available; certificate construction must be budgeted as another ledger
    (shared-work optimization may be added only with a matching executable).
    """

    if min(layers, hidden, heads, context, carriers, probes, temporal_freqs) < 1:
        raise ValueError("architecture and work sizes must be positive")
    if hidden % heads:
        raise ValueError("hidden size must be divisible by heads")
    if carriers not in SUPPORTED_CARRIER_COUNTS:
        raise ValueError(f"carriers must be one of {SUPPORTED_CARRIER_COUNTS}")
    if probes not in SUPPORTED_PROBE_COUNTS:
        raise ValueError(f"probes must be one of {SUPPORTED_PROBE_COUNTS}")
    if carriers > context:
        raise ValueError("carriers cannot exceed context")
    minimum_position_sum = carriers * (carriers - 1) // 2
    maximum_position_sum = carriers * (2 * context - carriers - 1) // 2
    position_sum_is_observed = carrier_position_sum is not None
    if carrier_position_sum is None:
        carrier_position_sum = carriers * (context - 1) // 2
    if not minimum_position_sum <= carrier_position_sum <= maximum_position_sum:
        raise ValueError(
            "carrier position sum is impossible for unique source positions"
        )
    head_dim = hidden // heads

    def input_projection(tokens: int) -> int:
        return 2 * tokens * (2 * temporal_freqs) * hidden + 2 * tokens * hidden * hidden

    def block_linear(tokens: int) -> int:
        return 2 * tokens * (5 * hidden * hidden)

    def attention(pairs: int) -> int:
        return 4 * pairs * hidden

    full_pairs = context * (context + 1) // 2
    full = input_projection(context) + layers * (
        block_linear(context) + attention(full_pairs)
    )

    # The executable projects full layer-0 K *and* V.  project_kv currently
    # delegates to _project, so it also computes and discards Q.  Keep the
    # intrinsic K/V work and this removable implementation overhead distinct.
    layer0_input = input_projection(context)
    layer0_kv_projection = 4 * context * hidden * hidden
    layer0_discarded_q_projection = 2 * context * hidden * hidden
    layer0_pointwise = 5 * context * hidden
    layer0_scan = (
        layer0_input
        + layer0_kv_projection
        + layer0_discarded_q_projection
        + layer0_pointwise
    )
    probe_reads = input_projection(probes) + layers * (
        block_linear(probes) + attention(probes * (context + 1))
    )
    carrier_pairs = carrier_position_sum + carriers
    carrier_reads = input_projection(carriers) + layers * (
        block_linear(carriers) + attention(carrier_pairs)
    )

    # Each version forms a P-query majority mask before accumulating moments.
    # q@k uses a multiply-add per key dimension; scaling is one extra FLOP per
    # head/logit. Boolean vote reductions are reported as non-FLOP comparisons.
    majority_mask = (
        4 * layers * probes * carriers * hidden
        + 2 * layers * probes * carriers * heads
    )
    # For each of Current and Parent: quadrature-weight V, mask/base reduce,
    # and mask/K/V outer-product reduce. The factor three counts the explicit
    # multiply(s) plus accumulation rather than treating either mask as free.
    moment_accumulation = 2 * layers * carriers * (
        3 * hidden + 3 * heads * head_dim * head_dim
    )
    moment_delta = layers * (hidden + heads * head_dim * head_dim)
    moment_build = majority_mask + moment_accumulation + moment_delta
    neural_total = (
        layer0_scan
        + probe_reads
        + carrier_reads
        + moment_build
    )

    selection = _address_selection_operation_counts(
        context=context,
        hidden=hidden,
        carriers=carriers,
    )
    address_width = selection["address_width"]
    selection_total = selection["total"]
    total = neural_total + selection_total
    optimized_neural_total = neural_total - layer0_discarded_q_projection
    optimized_total = optimized_neural_total + selection_total
    position_attention_flops = 4 * layers * hidden
    minimum_total = total + (
        minimum_position_sum - carrier_position_sum
    ) * position_attention_flops
    maximum_total = total + (
        maximum_position_sum - carrier_position_sum
    ) * position_attention_flops

    full_kv_scalars = 2 * layers * context * hidden
    moment_scalars = layers * heads * (head_dim + head_dim * head_dim)
    carrier_kv_scalars = 2 * layers * carriers * hidden
    layer0_kv_scalars = 2 * context * hidden
    address_feature_scalars = context * address_width
    address_distance_scalars = context * carriers
    return {
        "layers": layers,
        "hidden": hidden,
        "heads": heads,
        "context": context,
        "carriers": carriers,
        "probes": probes,
        "carrier_position_sum": carrier_position_sum,
        "carrier_position_sum_is_observed": position_sum_is_observed,
        "minimum_unique_carrier_position_sum": minimum_position_sum,
        "maximum_unique_carrier_position_sum": maximum_position_sum,
        "full_recompute_flops_per_user": full,
        "full_layer0_input_projection_flops_per_user": layer0_input,
        "full_layer0_kv_projection_flops_per_user": layer0_kv_projection,
        "discarded_full_layer0_q_projection_flops_per_user": (
            layer0_discarded_q_projection
        ),
        "full_layer0_pointwise_flops_per_user": layer0_pointwise,
        "full_layer0_projection_flops_per_user": layer0_scan,
        "probe_read_flops_per_user": probe_reads,
        "parent_conditioned_carrier_flops_per_user": carrier_reads,
        "majority_region_mask_flops_per_user": majority_mask,
        "paired_moment_accumulation_flops_per_user": moment_accumulation,
        "paired_moment_subtraction_flops_per_user": moment_delta,
        "paired_moment_build_flops_per_user": moment_build,
        "nested_certificate_flops_per_user": 0,
        "nested_certificate_companion_included": False,
        "neural_generation_flops_per_user": neural_total,
        "address_feature_normalization_flops_per_user": selection[
            "normalization"
        ],
        "address_feature_mean_flops_per_user": selection["mean"],
        "address_farthest_first_flops_per_user": selection["farthest_first"],
        "address_voronoi_flops_per_user": selection["voronoi"],
        "address_selection_flops_per_user": selection_total,
        "selection_flops_per_user": selection_total,
        "address_selection_nonflop_comparisons_lower_bound": (
            selection["comparison_lower_bound"]
        ),
        "total_generation_flops_per_user": total,
        "neural_over_full_fraction": neural_total / full,
        "selection_over_full_fraction": selection_total / full,
        "total_over_full_fraction": total / full,
        "minimum_total_over_full_fraction": minimum_total / full,
        "maximum_total_over_full_fraction": maximum_total / full,
        "within_20_percent_at_reported_position_sum": total / full <= 0.20,
        "within_20_percent_for_all_unique_position_sets": (
            maximum_total / full <= 0.20
        ),
        "optimized_kv_only_total_flops_per_user": optimized_total,
        "optimized_kv_only_total_over_full_fraction": optimized_total / full,
        "over_full_fraction": total / full,
        "full_current_kv_scalars": full_kv_scalars,
        "full_current_kv_bytes_fp32": 4 * full_kv_scalars,
        "persistent_moment_scalars": moment_scalars,
        "persistent_moment_bytes_fp32": 4 * moment_scalars,
        "persistent_moment_bytes_fp16": 2 * moment_scalars,
        "persistent_moment_ratio_to_full_current_kv": moment_scalars
        / full_kv_scalars,
        "current_carrier_kv_scalars_if_retained": carrier_kv_scalars,
        "current_carrier_kv_bytes_fp32_if_retained": 4 * carrier_kv_scalars,
        "persistent_moment_plus_carrier_ratio_if_retained": (
            moment_scalars + carrier_kv_scalars
        )
        / full_kv_scalars,
        "transient_full_current_layer0_kv_scalars": layer0_kv_scalars,
        "transient_full_current_layer0_kv_bytes_fp32": 4 * layer0_kv_scalars,
        "transient_address_feature_scalars_fp32": address_feature_scalars,
        "transient_address_feature_bytes_fp32": 4 * address_feature_scalars,
        "transient_voronoi_distance_scalars_fp32": address_distance_scalars,
        "transient_voronoi_distance_bytes_fp32": 4 * address_distance_scalars,
        "persistent_serving_state_semantics": (
            "moments_only_excludes_carrier_atoms_and_audit_metadata"
        ),
        "carrier_storage_semantics": (
            "construction_temporary_unless_explicitly_retained_for_refresh"
        ),
        "cost_semantics": (
            "single_non_recursive_parent_conditioned_ledger_current_implementation_"
            "includes_address_selection_not_final_closure"
        ),
    }


def causal_delta_closure_cost(
    *,
    layers: int,
    hidden: int,
    heads: int,
    context: int,
    carriers: int,
    recursive_delta: bool,
    carrier_position_sum: int | None = None,
    temporal_freqs: int = 16,
) -> dict[str, int | float | bool | str]:
    """Audit one executable independent or recursive native-delta closure.

    Both variants project the full raw-history Current layer-0 K/V, select
    addresses, and replay the chosen real events over their complete causal
    Parent prefixes. The recursive variant additionally lets carrier ``j``
    read the ``j`` earlier paired Current/Parent atoms at every layer. It does
    not run history probes or compile affine moments, so those diagnostic costs
    are intentionally absent.

    The recursive read counts two native arms for every earlier pair. Per arm,
    qK and weighted-V reduction cost ``4H`` FLOPs; logit scaling and applying
    the prefix count add two scalar multiplies per head. Current-minus-Parent
    response subtraction adds ``H`` per carrier/layer. ELU evaluation and
    comparison/index operations remain outside the repository's matmul FLOP
    convention and are not silently reclassified as neural matmuls.

    Minimal persistent storage contains only the new Current carrier K/V plus
    positions and masses; Parent K/V remains the already-existing serving
    base. The larger materialized paired object is reported separately because
    the current intervention helper may duplicate Parent atoms for convenience.
    """

    if min(layers, hidden, heads, context, carriers, temporal_freqs) < 1:
        raise ValueError("architecture and work sizes must be positive")
    if hidden % heads:
        raise ValueError("hidden size must be divisible by heads")
    if carriers not in SUPPORTED_CARRIER_COUNTS:
        raise ValueError(f"carriers must be one of {SUPPORTED_CARRIER_COUNTS}")
    if carriers > context:
        raise ValueError("carriers cannot exceed context")

    minimum_position_sum = carriers * (carriers - 1) // 2
    maximum_position_sum = carriers * (2 * context - carriers - 1) // 2
    position_sum_is_observed = carrier_position_sum is not None
    if carrier_position_sum is None:
        carrier_position_sum = carriers * (context - 1) // 2
    if not minimum_position_sum <= carrier_position_sum <= maximum_position_sum:
        raise ValueError(
            "carrier position sum is impossible for unique source positions"
        )

    def input_projection(tokens: int) -> int:
        return 2 * tokens * (2 * temporal_freqs) * hidden + 2 * tokens * hidden * hidden

    def block_linear(tokens: int) -> int:
        return 2 * tokens * (5 * hidden * hidden)

    def attention(pairs: int) -> int:
        return 4 * pairs * hidden

    full_pairs = context * (context + 1) // 2
    full = input_projection(context) + layers * (
        block_linear(context) + attention(full_pairs)
    )

    layer0_input = input_projection(context)
    layer0_kv_projection = 4 * context * hidden * hidden
    # Current PointwiseAttention.project_kv executes _project and discards Q.
    layer0_discarded_q_projection = 2 * context * hidden * hidden
    layer0_pointwise = 5 * context * hidden
    layer0_scan = (
        layer0_input
        + layer0_kv_projection
        + layer0_discarded_q_projection
        + layer0_pointwise
    )

    parent_prefix_pairs = carrier_position_sum + carriers
    carrier_replay = input_projection(carriers) + layers * (
        block_linear(carriers) + attention(parent_prefix_pairs)
    )
    earlier_pairs = carriers * (carriers - 1) // 2
    recursive_pair_reads = (
        layers * earlier_pairs * (8 * hidden + 4 * heads)
        if recursive_delta
        else 0
    )
    recursive_response_combine = (
        layers * carriers * hidden if recursive_delta else 0
    )
    recursive_read = recursive_pair_reads + recursive_response_combine
    # build_native_pair_memory multiplies both Current and Parent V by their
    # masses and negates the Parent arm. Parent indexing/concatenation is I/O.
    native_pair_materialization = 3 * layers * carriers * hidden
    neural_total = (
        layer0_scan
        + carrier_replay
        + recursive_read
        + native_pair_materialization
    )

    selection = _address_selection_operation_counts(
        context=context,
        hidden=hidden,
        carriers=carriers,
    )
    selection_total = selection["total"]
    total = neural_total + selection_total
    optimized_total = total - layer0_discarded_q_projection
    position_attention_flops = 4 * layers * hidden
    minimum_total = total + (
        minimum_position_sum - carrier_position_sum
    ) * position_attention_flops
    maximum_total = total + (
        maximum_position_sum - carrier_position_sum
    ) * position_attention_flops

    full_kv_scalars = 2 * layers * context * hidden
    incremental_carrier_scalars = 2 * layers * carriers * hidden
    materialized_pair_scalars = 4 * layers * carriers * hidden
    metadata_bytes = 2 * carriers * 8  # int64 positions and masses
    minimal_sidecar_bytes = 4 * incremental_carrier_scalars + metadata_bytes
    materialized_pair_bytes = 4 * materialized_pair_scalars + metadata_bytes
    layer0_kv_scalars = 2 * context * hidden
    address_feature_scalars = context * (2 * hidden)
    address_distance_scalars = context * carriers
    return {
        "method": (
            "recursive_causal_delta_closure"
            if recursive_delta
            else "independent_parent_conditioned_carriers"
        ),
        "layers": layers,
        "hidden": hidden,
        "heads": heads,
        "context": context,
        "carriers": carriers,
        "recursive_delta": recursive_delta,
        "carrier_position_sum": carrier_position_sum,
        "carrier_position_sum_is_observed": position_sum_is_observed,
        "minimum_unique_carrier_position_sum": minimum_position_sum,
        "maximum_unique_carrier_position_sum": maximum_position_sum,
        "full_recompute_flops_per_user": full,
        "full_layer0_input_projection_flops_per_user": layer0_input,
        "full_layer0_kv_projection_flops_per_user": layer0_kv_projection,
        "discarded_full_layer0_q_projection_flops_per_user": (
            layer0_discarded_q_projection
        ),
        "full_layer0_pointwise_flops_per_user": layer0_pointwise,
        "full_layer0_projection_flops_per_user": layer0_scan,
        "parent_conditioned_carrier_flops_per_user": carrier_replay,
        "recursive_earlier_pair_count_per_layer": earlier_pairs,
        "recursive_paired_atom_read_flops_per_user": recursive_pair_reads,
        "recursive_response_combine_flops_per_user": recursive_response_combine,
        "recursive_delta_flops_per_user": recursive_read,
        "native_pair_materialization_flops_per_user": native_pair_materialization,
        "neural_generation_flops_per_user": neural_total,
        "address_feature_normalization_flops_per_user": selection[
            "normalization"
        ],
        "address_feature_mean_flops_per_user": selection["mean"],
        "address_farthest_first_flops_per_user": selection["farthest_first"],
        "address_voronoi_flops_per_user": selection["voronoi"],
        "address_selection_nonflop_comparisons_lower_bound": selection[
            "comparison_lower_bound"
        ],
        "selection_flops_per_user": selection_total,
        "total_generation_flops_per_user": total,
        "neural_over_full_fraction": neural_total / full,
        "selection_over_full_fraction": selection_total / full,
        "total_over_full_fraction": total / full,
        "minimum_total_over_full_fraction": minimum_total / full,
        "maximum_total_over_full_fraction": maximum_total / full,
        "over_full_fraction": total / full,
        "optimized_kv_only_total_flops_per_user": optimized_total,
        "optimized_kv_only_total_over_full_fraction": optimized_total / full,
        "within_20_percent_at_reported_position_sum": total / full <= 0.20,
        "within_20_percent_for_all_unique_position_sets": (
            maximum_total / full <= 0.20
        ),
        "full_current_kv_scalars": full_kv_scalars,
        "full_current_kv_bytes_fp32": 4 * full_kv_scalars,
        "irreducible_incremental_carrier_scalars": incremental_carrier_scalars,
        "persistent_position_and_mass_bytes_int64": metadata_bytes,
        "irreducible_incremental_sidecar_bytes_fp32": minimal_sidecar_bytes,
        "irreducible_incremental_sidecar_ratio_to_full_current_kv_fp32": (
            minimal_sidecar_bytes / (4 * full_kv_scalars)
        ),
        "materialized_intervention_pair_scalars": materialized_pair_scalars,
        "materialized_intervention_pair_bytes_fp32": materialized_pair_bytes,
        "materialized_intervention_ratio_to_full_current_kv_fp32": (
            materialized_pair_bytes / (4 * full_kv_scalars)
        ),
        "transient_full_current_layer0_kv_scalars": layer0_kv_scalars,
        "transient_full_current_layer0_kv_bytes_fp32": 4 * layer0_kv_scalars,
        "transient_address_feature_scalars_fp32": address_feature_scalars,
        "transient_address_feature_bytes_fp32": 4 * address_feature_scalars,
        "transient_voronoi_distance_scalars_fp32": address_distance_scalars,
        "transient_voronoi_distance_bytes_fp32": 4 * address_distance_scalars,
        "storage_semantics": (
            "minimal_sidecar_reuses_parent_base_materialized_pair_duplicates_parent_atoms"
        ),
        "cost_semantics": (
            "native_causal_delta_closure_includes_address_selection_excludes_"
            "diagnostic_probes_and_moments"
        ),
    }
