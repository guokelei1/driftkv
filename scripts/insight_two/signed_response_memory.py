"""Oracle stratified signed-response memory for HSTU attention.

This module is a representation diagnostic, not an executable migration
estimator.  It reads both a Current-Exact cache and its Parent-Reuse control
cache.  No target is fitted and no labels are consumed.

For each layer, fixed equal-width chronological strata contribute one fixed
midpoint.  The selected Current atom is positive and the paired Reuse atom is
negative; their values are multiplied by the inverse within-stratum inclusion
probability.  A real candidate query then reads those signed atoms with the
model's native unnormalised attention kernel::

    sum_s (1 / pi_s) * [phi(q K_exact_s) V_exact_s
                        - phi(q K_reuse_s) V_reuse_s]

The complete Reuse cache is the control path.  Adding this response estimate
to its native prefix read gives the layer intervention.  When every history
position is selected (``R == N``), every ``pi_s`` is one and the intervention
numerically reconstructs the Current-Exact reader path.

Midpoint selection is deliberately deterministic.  The inverse-probability
weight is the nominal weight for one point from a stratum; this is therefore a
fixed stratified quadrature diagnostic, not a claim of randomized unbiasedness.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hstu_kvcache.models import HSTUKVCache


SUPPORTED_SAMPLE_COUNTS = (8, 16, 32, 64, 128)


@dataclass(frozen=True)
class FixedMidpointStrata:
    """Auditable fixed equal-width strata over chronological cache positions."""

    history_length: int
    sample_count: int
    starts: torch.Tensor
    stops: torch.Tensor
    midpoints: torch.Tensor
    inverse_inclusion_probabilities: torch.Tensor


@dataclass(frozen=True)
class OracleSignedResponseMemory:
    """Paired signed native-attention atoms derived from Exact and Reuse.

    ``keys`` and ``signed_values`` have shape ``[layers, batch, 2R, kv_width]``.
    The first ``R`` atoms are Current-Exact with positive values; the final
    ``R`` atoms are Parent-Reuse with negative values.  Both halves retain the
    sampled source positions so relative-position bias, when configured, is
    evaluated at the original history location.

    This object is oracle diagnostic state because construction reads the full
    Current-Exact cache.  It is not a legal release-time estimator.
    """

    keys: torch.Tensor
    signed_values: torch.Tensor
    source_positions: torch.Tensor
    sample_positions: torch.Tensor
    inverse_inclusion_probabilities: torch.Tensor
    source_length: int

    @property
    def sample_count(self) -> int:
        return int(self.sample_positions.numel())

    @property
    def atom_count(self) -> int:
        return int(self.source_positions.numel())


@dataclass(frozen=True)
class OracleSignedResponseIntervention:
    """Outputs from the coherent layer-by-layer oracle reader intervention."""

    scores: torch.Tensor
    readout: torch.Tensor
    layer_residual_heads: tuple[torch.Tensor, ...]


def fixed_midpoint_strata(
    history_length: int,
    sample_count: int,
) -> FixedMidpointStrata:
    """Return fixed lower-midpoint samples for equal-width time-order strata.

    Cache positions are chronological.  Equal widths are required rather than
    silently creating shorter end strata.  For an even stratum width, the
    lower of the two central positions is the frozen midpoint.
    """

    # R=N is an instrumentation-only exact-reconstruction control.  It is
    # intentionally allowed beyond the compact research grid (for example,
    # N=1024 in the Medium workload).
    if sample_count not in SUPPORTED_SAMPLE_COUNTS and sample_count != history_length:
        raise ValueError(
            "sample_count must be one of "
            f"{SUPPORTED_SAMPLE_COUNTS} or equal history_length, got {sample_count}"
        )
    if history_length < sample_count:
        raise ValueError("sample_count cannot exceed history_length")
    if history_length % sample_count:
        raise ValueError("equal-width strata require history_length % sample_count == 0")
    width = history_length // sample_count
    starts = torch.arange(sample_count, dtype=torch.long) * width
    stops = starts + width
    midpoints = starts + (width - 1) // 2
    inverse_probability = torch.full(
        (sample_count,), float(width), dtype=torch.float32
    )
    return FixedMidpointStrata(
        history_length=history_length,
        sample_count=sample_count,
        starts=starts,
        stops=stops,
        midpoints=midpoints,
        inverse_inclusion_probabilities=inverse_probability,
    )


def _validate_cache(cache: HSTUKVCache, name: str) -> None:
    if cache.k.ndim != 4 or cache.k.shape != cache.v.shape:
        raise ValueError(f"{name} must contain matching [layers,B,N,width] K/V")
    if cache.k.shape[2] != cache.seq_len:
        raise ValueError(f"{name} seq_len must equal its tensor history width")
    if not cache.k.is_floating_point() or not cache.v.is_floating_point():
        raise ValueError(f"{name} K/V must be floating point")


@torch.inference_mode()
def build_oracle_signed_response_memory(
    exact_cache: HSTUKVCache,
    reuse_cache: HSTUKVCache,
    *,
    sample_count: int,
) -> OracleSignedResponseMemory:
    """Build a fixed stratified signed memory with no fitting.

    The full ``exact_cache`` is intentionally an input, making the returned
    memory an oracle mechanism probe.  The function neither changes nor
    returns a migrated KV cache.
    """

    _validate_cache(exact_cache, "exact_cache")
    _validate_cache(reuse_cache, "reuse_cache")
    if exact_cache.seq_len != reuse_cache.seq_len:
        raise ValueError("Exact and Reuse cache lengths differ")
    if exact_cache.k.shape != reuse_cache.k.shape:
        raise ValueError("Exact and Reuse cache tensor shapes differ")
    if exact_cache.k.device != reuse_cache.k.device:
        raise ValueError("Exact and Reuse caches must be on the same device")
    if exact_cache.k.dtype != reuse_cache.k.dtype:
        raise ValueError("Exact and Reuse caches must have the same dtype")

    strata = fixed_midpoint_strata(exact_cache.seq_len, sample_count)
    positions = strata.midpoints.to(device=exact_cache.k.device)
    inverse_probability = strata.inverse_inclusion_probabilities.to(
        device=exact_cache.v.device,
        dtype=exact_cache.v.dtype,
    )
    exact_k = exact_cache.k.index_select(2, positions)
    exact_v = exact_cache.v.index_select(2, positions)
    reuse_k = reuse_cache.k.index_select(2, positions)
    reuse_v = reuse_cache.v.index_select(2, positions)
    value_weight = inverse_probability.view(1, 1, sample_count, 1)

    # IPW lives in V, so the real query can read the positive and negative
    # atoms in one native unnormalised-attention operation.
    keys = torch.cat((exact_k, reuse_k), dim=2).detach()
    signed_values = torch.cat(
        (exact_v * value_weight, -reuse_v * value_weight), dim=2
    ).detach()
    source_positions = torch.cat((positions, positions), dim=0)
    return OracleSignedResponseMemory(
        keys=keys,
        signed_values=signed_values,
        source_positions=source_positions,
        sample_positions=positions,
        inverse_inclusion_probabilities=inverse_probability,
        source_length=exact_cache.seq_len,
    )


def _validate_memory(
    attention,
    q: torch.Tensor,
    memory: OracleSignedResponseMemory,
    layer: int,
    candidate_count: int,
) -> tuple[int, int]:
    if attention.training:
        raise ValueError("oracle signed-response reads require eval mode")
    if q.ndim != 4 or q.shape[2] != 1:
        raise ValueError("q must have shape [B*C, heads, 1, head_dim]")
    if candidate_count < 1 or q.shape[0] % candidate_count:
        raise ValueError("candidate_count does not divide the flattened query batch")
    batch = q.shape[0] // candidate_count
    if memory.keys.ndim != 4 or memory.keys.shape != memory.signed_values.shape:
        raise ValueError("signed memory atoms must share [layers,B,2R,width] shape")
    if not 0 <= layer < memory.keys.shape[0]:
        raise ValueError("memory layer is out of range")
    if memory.keys.shape[1] != batch:
        raise ValueError("signed memory and query batches differ")
    if memory.atom_count != 2 * memory.sample_count:
        raise ValueError("signed memory must contain paired positive/negative atoms")
    if memory.keys.shape[2] != memory.atom_count:
        raise ValueError("signed memory atom axis differs from its source positions")
    if memory.keys.shape[3] != attention.inner:
        raise ValueError("signed memory KV width differs from the attention reader")
    if q.shape[1:] != (attention.num_heads, 1, attention.head_dim):
        raise ValueError("query head layout differs from the attention reader")
    return batch, memory.atom_count


def read_ipw_signed_native_residual(
    attention,
    q: torch.Tensor,
    memory: OracleSignedResponseMemory,
    *,
    layer: int,
    candidate_count: int,
) -> torch.Tensor:
    """Let real candidate ``q`` read one layer's signed response memory.

    Returns pre-output-projection heads shaped ``[B*C, heads, 1, head_dim]``.
    The activation, QK scale, optional relative-position bias, HSTU-reference
    scaling and attention dropout module are the reader's native operations.
    Eval mode is required so paired control-variate reads are deterministic.
    """

    batch, atoms = _validate_memory(
        attention, q, memory, layer, candidate_count
    )
    flat = batch * candidate_count
    keys = memory.keys[layer].to(device=q.device, dtype=q.dtype)
    values = memory.signed_values[layer].to(device=q.device, dtype=q.dtype)
    keys = keys.repeat_interleave(candidate_count, dim=0).view(
        flat, atoms, attention.num_heads, attention.head_dim
    ).transpose(1, 2)
    values = values.repeat_interleave(candidate_count, dim=0).view(
        flat, atoms, attention.num_heads, attention.head_dim
    ).transpose(1, 2)
    weights = torch.matmul(q, keys.transpose(-2, -1)) * attention.scale
    query_positions = torch.tensor([memory.source_length], device=q.device)
    key_positions = memory.source_positions.to(device=q.device)
    bias = attention._relative_position_bias(
        query_positions, key_positions, weights.dtype
    )
    if bias is not None:
        weights = weights + bias
    weights = attention._activate(weights)
    if attention.block_variant == "hstu_reference":
        weights = weights / attention.cfg.max_seq_len
    weights = attention.attn_dropout(weights)
    return torch.matmul(weights, values)


def _native_prefix_heads(
    attention,
    q: torch.Tensor,
    cache: HSTUKVCache,
    layer: int,
    candidate_count: int,
) -> torch.Tensor:
    flat = q.shape[0]
    length = cache.seq_len
    keys = cache.k[layer].to(device=q.device, dtype=q.dtype)
    values = cache.v[layer].to(device=q.device, dtype=q.dtype)
    keys = keys.repeat_interleave(candidate_count, dim=0).view(
        flat, length, attention.num_heads, attention.head_dim
    ).transpose(1, 2)
    values = values.repeat_interleave(candidate_count, dim=0).view(
        flat, length, attention.num_heads, attention.head_dim
    ).transpose(1, 2)
    weights = torch.matmul(q, keys.transpose(-2, -1)) * attention.scale
    query_positions = torch.tensor([length], device=q.device)
    key_positions = torch.arange(length, device=q.device)
    bias = attention._relative_position_bias(
        query_positions, key_positions, weights.dtype
    )
    if bias is not None:
        weights = weights + bias
    weights = attention._activate(weights)
    if attention.block_variant == "hstu_reference":
        weights = weights / attention.cfg.max_seq_len
    return torch.matmul(attention.attn_dropout(weights), values)


def _native_self_heads(
    attention,
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    query_position: int,
) -> torch.Tensor:
    if attention.causal_diagonal != "inclusive":
        return torch.zeros_like(v_new)
    weights = (q * k_new).sum(dim=-1, keepdim=True) * attention.scale
    positions = torch.tensor([query_position], device=q.device)
    bias = attention._relative_position_bias(positions, positions, weights.dtype)
    if bias is not None:
        weights = weights + bias
    weights = attention._activate(weights)
    if attention.block_variant == "hstu_reference":
        weights = weights / attention.cfg.max_seq_len
    return attention.attn_dropout(weights) * v_new


def _block_update(block, x_norm: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
    attention_out = block.attn._finish(heads)
    if block.block_variant == "hstu_reference":
        assert block.attn_output_norm is not None
        return block.attn.out_proj(
            block.attn_output_norm(attention_out) * F.silu(block.gate_proj(x_norm))
        )
    if block.gating == "silu_gate":
        return attention_out * F.silu(block.gate_proj(x_norm))
    if block.gating == "glu":
        return attention_out * torch.sigmoid(block.gate_proj(x_norm))
    if block.gating == "ffn":
        return block.fc2(F.silu(block.fc1(x_norm)) * block.fc3(x_norm))
    return attention_out


@torch.inference_mode()
def intervene_oracle_signed_response_memory(
    model,
    reuse_cache: HSTUKVCache,
    memory: OracleSignedResponseMemory,
    candidate_ids: torch.Tensor,
    query_time_deltas: torch.Tensor,
    *,
    query_type_ids: torch.Tensor | None = None,
    query_action_ids: torch.Tensor | None = None,
    candidate_item_vectors: torch.Tensor | None = None,
) -> OracleSignedResponseIntervention:
    """Run a coherent Current reader with the oracle signed memory per layer.

    Each layer first performs the ordinary full Parent-Reuse prefix read.  The
    same layer's real candidate query then reads the compact signed atoms, and
    that response is added before the native attention output transform,
    gating, and residual update.  Consequently later-layer queries reflect
    all earlier interventions rather than being replayed from a frozen trace.
    """

    _validate_cache(reuse_cache, "reuse_cache")
    if model.training:
        raise ValueError("oracle signed-response intervention requires model.eval()")
    if candidate_ids.ndim != 2 or candidate_ids.shape[1] < 1:
        raise ValueError("candidate_ids must have shape [B,C] with C >= 1")
    batch, candidates = candidate_ids.shape
    if reuse_cache.k.shape[1] != batch:
        raise ValueError("Reuse cache and candidate batches differ")
    if reuse_cache.seq_len != memory.source_length:
        raise ValueError("Reuse cache length differs from signed-memory source")
    if reuse_cache.k.shape[0] != memory.keys.shape[0]:
        raise ValueError("Reuse cache and signed memory layer counts differ")
    if len(model.blocks) != memory.keys.shape[0]:
        raise ValueError("model and signed memory layer counts differ")

    x = model.embed_query_tokens(
        candidate_ids,
        query_time_deltas,
        query_type_ids=query_type_ids,
        query_action_ids=query_action_ids,
        item_vectors=candidate_item_vectors,
    ).reshape(batch * candidates, 1, model.cfg.hidden_size)
    layer_residuals: list[torch.Tensor] = []
    for layer, block in enumerate(model.blocks):
        residual_x = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        reuse_heads = _native_prefix_heads(
            block.attn, q, reuse_cache, layer, candidates
        )
        signed_residual = read_ipw_signed_native_residual(
            block.attn,
            q,
            memory,
            layer=layer,
            candidate_count=candidates,
        )
        layer_residuals.append(
            signed_residual.reshape(
                batch,
                candidates,
                block.attn.num_heads,
                block.attn.head_dim,
            )
        )
        self_heads = _native_self_heads(
            block.attn, q, k_new, v_new, memory.source_length
        )
        x = residual_x + _block_update(
            block, x_norm, reuse_heads + signed_residual + self_heads
        )

    readout = model.final_norm(x).reshape(batch, candidates, model.cfg.hidden_size)
    return OracleSignedResponseIntervention(
        scores=model.cc_score_head(readout).squeeze(-1),
        readout=readout,
        layer_residual_heads=tuple(layer_residuals),
    )
