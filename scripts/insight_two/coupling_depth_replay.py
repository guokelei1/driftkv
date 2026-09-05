"""Depth-limited Parent coupling for finite-release mode replay.

This module is a deliberately narrow, non-formal mechanism diagnostic.  The
Current arm is replayed at fixed rank through every cache-formation layer.  A
matched Parent arm is advanced only through a preregistered prefix of layers.
Within that prefix, signed migration cores use the difference between the two
equal-resolution approximate arms.  Above the prefix, they revert to the
single-arm ``approximate Current - exact Parent`` control::

    l < d:   E_l = U0^T (Khat_l^C - Khat_l^P)
    l >= d:  E_l = U0^T (Khat_l^C - K_l^P)
    K_l^mig = K_l^P + U0 E_l

and analogously for V.  ``U0`` is always compiled from the approximate paired
layer-0 K/V defect.  Thus changing ``d`` interrupts only the *depth of matched
release coupling*; it does not change the Current arm, basis source, sidecar
rank, candidates, or labels.

No API accepts Current-Exact K/V.  Whole-history randomized range finding is a
release-time diagnostic compiler, not a claim of tokenwise causal replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from hstu_kvcache.models import HSTUKVCache
from insight_two.matrix_free_input_range import matrix_free_input_cost
from insight_two.mode_space_replay import (
    FactorizedCacheLayer,
    FactorizedReplay,
    SharedModeSplice,
    TokenModeFactors,
    _compress_token_modes,
    _factorized_legacy_block_step,
    factorized_rmsnorm,
    randomized_token_basis,
)


@dataclass(frozen=True)
class ParentPrefixReplay:
    """Matched Parent cache factors for exactly ``formation_depth`` layers."""

    layers: tuple[FactorizedCacheLayer, ...]
    block_input_factors: tuple[TokenModeFactors, ...]
    formation_depth: int
    seq_len: int

    def __post_init__(self) -> None:
        if self.formation_depth < 1:
            raise ValueError("formation_depth must be positive")
        if len(self.layers) != self.formation_depth:
            raise ValueError("Parent prefix layer count differs from depth")
        if len(self.block_input_factors) != self.formation_depth:
            raise ValueError("Parent prefix input count differs from depth")
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive")


def _validate_model(model) -> None:
    if model.training:
        raise ValueError("coupling-depth replay requires model.eval()")
    if not model.blocks:
        raise ValueError("model must contain at least one block")
    if model.cfg.block_variant != "legacy":
        raise ValueError("coupling-depth replay currently covers legacy blocks")


def _factorized_kv_only(
    block,
    input_factors: TokenModeFactors,
) -> FactorizedCacheLayer:
    """Form the terminal cache K/V without an unused Q/attention/update path."""

    normalized = factorized_rmsnorm(block.norm, input_factors)
    attention = block.attn
    if attention.k_proj.bias is not None or attention.v_proj.bias is not None:
        raise ValueError("factorized terminal K/V requires bias-free projections")
    key_core = normalized.right @ attention.k_proj.weight.transpose(0, 1)
    value_core = normalized.right @ attention.v_proj.weight.transpose(0, 1)
    return FactorizedCacheLayer(
        left=normalized.left,
        key_core=key_core,
        value_core=value_core,
    )


@torch.inference_mode()
def factorized_parent_prefix_replay(
    model,
    embedded_history: torch.Tensor,
    *,
    formation_depth: int,
    rank: int = 4,
    compression: Literal["exact_svd", "fixed_range_finder"] = "fixed_range_finder",
    sketch_oversample: int = 4,
    sketch_power_iterations: int = 1,
    sketch_seed: int = 17,
) -> ParentPrefixReplay:
    """Advance Parent only far enough to form cache layers ``[0,d)``.

    The final requested Parent layer uses a K/V-only projection because its
    post-block state has no consumer.  This is the executable reason the
    static cost scales with ``d``; no upper Parent approximate state is formed.
    """

    _validate_model(model)
    if embedded_history.ndim != 3 or not embedded_history.is_floating_point():
        raise ValueError("embedded_history must be floating [B,N,H]")
    _, length, hidden = embedded_history.shape
    if hidden != model.cfg.hidden_size:
        raise ValueError("embedded history width differs from model")
    if not 1 <= formation_depth <= len(model.blocks):
        raise ValueError("formation_depth must be within model depth")
    if not 1 <= rank <= min(length, hidden):
        raise ValueError("rank must be in [1,min(history,hidden)]")

    dense_state = embedded_history
    inputs: list[TokenModeFactors] = []
    layers: list[FactorizedCacheLayer] = []
    for layer in range(formation_depth):
        factors = _compress_token_modes(
            dense_state,
            rank=rank,
            compression=compression,
            oversample=sketch_oversample,
            power_iterations=sketch_power_iterations,
            seed=sketch_seed + layer,
        )
        inputs.append(factors)
        if layer + 1 == formation_depth:
            cache_layer = _factorized_kv_only(model.blocks[layer], factors)
        else:
            dense_state, cache_layer = _factorized_legacy_block_step(model.blocks[layer], factors)
        layers.append(cache_layer)
    return ParentPrefixReplay(
        layers=tuple(layers),
        block_input_factors=tuple(inputs),
        formation_depth=formation_depth,
        seq_len=int(length),
    )


@torch.inference_mode()
def factorized_current_rank_handoff_replay(
    model,
    embedded_history: torch.Tensor,
    *,
    handoff_depth: int = 3,
    early_rank: int = 4,
    upper_rank: int = 8,
    sketch_oversample: int = 4,
    sketch_power_iterations: int = 1,
    sketch_seed: int = 17,
) -> FactorizedReplay:
    """Merge the stopped Parent arm's mode budget into upper Current replay.

    Layers ``[0,handoff_depth)`` are bitwise the same fixed-rank Current arm as
    the depth-profile diagnostic.  At the boundary, the next compression uses
    ``upper_rank`` and the Current trajectory alone continues.  The last layer
    forms K/V only.  This function implements one preregistered structural
    handoff; the runner does not sweep its depth or ranks.
    """

    _validate_model(model)
    if embedded_history.ndim != 3 or not embedded_history.is_floating_point():
        raise ValueError("embedded_history must be floating [B,N,H]")
    _, length, hidden = embedded_history.shape
    if hidden != model.cfg.hidden_size:
        raise ValueError("embedded history width differs from model")
    if not 1 <= handoff_depth < len(model.blocks):
        raise ValueError("handoff_depth must leave at least one upper layer")
    if not 1 <= early_rank <= min(length, hidden):
        raise ValueError("early_rank is outside the token matrix")
    if not early_rank <= upper_rank <= min(length, hidden):
        raise ValueError("upper_rank must be at least early_rank and valid")

    dense_state = embedded_history
    input_factors: list[TokenModeFactors] = []
    cache_layers: list[FactorizedCacheLayer] = []
    materialized: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer, block in enumerate(model.blocks):
        rank = early_rank if layer < handoff_depth else upper_rank
        factors = _compress_token_modes(
            dense_state,
            rank=rank,
            compression="fixed_range_finder",
            oversample=sketch_oversample,
            power_iterations=sketch_power_iterations,
            seed=sketch_seed + layer,
        )
        input_factors.append(factors)
        if layer + 1 == len(model.blocks):
            cache_layer = _factorized_kv_only(block, factors)
        else:
            dense_state, cache_layer = _factorized_legacy_block_step(block, factors)
        cache_layers.append(cache_layer)
        materialized.append(cache_layer.materialize())
    return FactorizedReplay(
        cache=HSTUKVCache.from_layer_list(materialized, seq_len=int(length)),
        layers=tuple(cache_layers),
        block_input_factors=tuple(input_factors),
    )


def _validate_exact_parent_and_current(
    exact_parent: HSTUKVCache,
    current: FactorizedReplay,
) -> tuple[int, int, int, int]:
    if exact_parent.k.ndim != 4 or exact_parent.k.shape != exact_parent.v.shape:
        raise ValueError("exact Parent cache must contain matching [L,B,N,H] K/V")
    if current.cache.k.shape != exact_parent.k.shape:
        raise ValueError("Current replay and exact Parent cache shapes differ")
    if current.cache.v.shape != exact_parent.v.shape:
        raise ValueError("Current replay and exact Parent cache shapes differ")
    layers, batch, length, width = (int(value) for value in exact_parent.k.shape)
    if exact_parent.seq_len != length or current.cache.seq_len != length:
        raise ValueError("cache sequence lengths differ")
    if len(current.layers) != layers:
        raise ValueError("Current replay layer count differs")
    return layers, batch, length, width


@torch.inference_mode()
def coupling_depth_layer0_basis(
    parent_prefix: ParentPrefixReplay,
    current: FactorizedReplay,
    *,
    rank: int = 8,
    oversample: int = 4,
    power_iterations: int = 0,
    seed: int = 1017,
) -> torch.Tensor:
    """Compile U0 from paired approximate layer-0 ``Delta[K,V]`` only."""

    if not current.layers:
        raise ValueError("Current replay has no cache layers")
    if current.cache.seq_len != parent_prefix.seq_len:
        raise ValueError("Parent and Current replay sequence lengths differ")
    parent_key, parent_value = parent_prefix.layers[0].materialize()
    current_key, current_value = current.layers[0].materialize()
    if parent_key.shape != current_key.shape or parent_value.shape != current_value.shape:
        raise ValueError("paired layer-0 cache shapes differ")
    defect = torch.cat((current_key - parent_key, current_value - parent_value), dim=-1)
    return randomized_token_basis(
        defect,
        rank=rank,
        oversample=oversample,
        power_iterations=power_iterations,
        seed=seed,
    )


def _validate_orthonormal_basis(
    basis: torch.Tensor,
    *,
    batch: int,
    length: int,
) -> None:
    if basis.ndim != 3 or basis.shape[:2] != (batch, length):
        raise ValueError("basis must have shape [B,N,r]")
    gram = basis.transpose(1, 2) @ basis
    identity = torch.eye(basis.shape[2], device=basis.device, dtype=basis.dtype).expand_as(gram)
    if not torch.allclose(gram, identity, atol=2e-4, rtol=2e-4):
        raise ValueError("basis columns must be orthonormal")


@torch.inference_mode()
def splice_with_coupling_depth(
    exact_parent: HSTUKVCache,
    current: FactorizedReplay,
    parent_prefix: ParentPrefixReplay,
    basis: torch.Tensor,
) -> SharedModeSplice:
    """Apply matched subtraction below ``d`` and exact-Parent subtraction above."""

    layers, batch, length, _ = _validate_exact_parent_and_current(exact_parent, current)
    depth = parent_prefix.formation_depth
    if depth > layers or parent_prefix.seq_len != length:
        raise ValueError("Parent prefix lineage differs")
    basis = basis.to(device=exact_parent.k.device, dtype=exact_parent.k.dtype)
    _validate_orthonormal_basis(basis, batch=batch, length=length)
    transpose = basis.transpose(1, 2)

    migrated: list[tuple[torch.Tensor, torch.Tensor]] = []
    delta_keys: list[torch.Tensor] = []
    delta_values: list[torch.Tensor] = []
    for layer, current_factors in enumerate(current.layers):
        current_alignment = transpose @ current_factors.left
        current_key = current_alignment @ current_factors.key_core
        current_value = current_alignment @ current_factors.value_core
        if layer < depth:
            parent_factors = parent_prefix.layers[layer]
            parent_alignment = transpose @ parent_factors.left
            parent_key = parent_alignment @ parent_factors.key_core
            parent_value = parent_alignment @ parent_factors.value_core
        else:
            parent_key = transpose @ exact_parent.k[layer]
            parent_value = transpose @ exact_parent.v[layer]
        delta_key = current_key - parent_key
        delta_value = current_value - parent_value
        migrated.append(
            (
                exact_parent.k[layer] + basis @ delta_key,
                exact_parent.v[layer] + basis @ delta_value,
            )
        )
        delta_keys.append(delta_key.detach())
        delta_values.append(delta_value.detach())
    return SharedModeSplice(
        cache=HSTUKVCache.from_layer_list(migrated, seq_len=length),
        basis=basis.detach(),
        delta_k_cores=tuple(delta_keys),
        delta_v_cores=tuple(delta_values),
    )


def medium_coupling_depth_cost(formation_depth: int) -> dict[str, int | float | bool | str]:
    """Strict Medium release-time ledger for the frozen depth profile.

    This reproduces the repository's existing multiply-add=2 FLOP audit for
    ``N=1024,H=192,L=6,heads=6``.  Every arm uses target rank 4, oversample 4,
    power 1.  U0 uses target rank 8, oversample 4, power 0.  The terminal layer
    of each required trajectory forms RMSNorm and K/V only; its post-block
    state has no consumer.  The returned prototype upper bound additionally
    charges those unused terminal block outputs/recompressions.
    """

    layers = 6
    if not 1 <= formation_depth <= layers:
        raise ValueError("formation_depth must be in [1,6]")

    exact_all = 4_771_282_944
    raw_input = 88_489_984
    initial_compression = 12_951_382
    full_block_body = 70_206_208
    post_block_compression = 13_282_134
    terminal_kv_only = 643_840
    full_transition = full_block_body + post_block_compression
    terminal_full_extra = full_transition - terminal_kv_only

    current_arm = (
        raw_input + initial_compression + (layers - 1) * full_transition + terminal_kv_only
    )
    parent_prefix = (
        raw_input + initial_compression + (formation_depth - 1) * full_transition + terminal_kv_only
    )
    layer0_basis_and_core = 1_247_808
    paired_upper_core_per_layer = 183_296
    # rank-4 Current factor-to-U0 core plus dense exact-Parent projection.
    current_minus_exact_parent_core_per_layer = 6_384_640
    paired_upper_cores = (formation_depth - 1) * paired_upper_core_per_layer
    autonomous_upper_cores = (layers - formation_depth) * current_minus_exact_parent_core_per_layer
    migration_sufficient_total = (
        current_arm
        + parent_prefix
        + layer0_basis_and_core
        + paired_upper_cores
        + autonomous_upper_cores
    )
    prototype_full_block_total = migration_sufficient_total + 2 * terminal_full_extra
    sidecar_scalars = 1024 * 8 + 2 * layers * 8 * 192

    return {
        "formation_depth": formation_depth,
        "exact_all_flops_per_user": exact_all,
        "current_arm_flops_per_user": current_arm,
        "parent_prefix_flops_per_user": parent_prefix,
        "layer0_basis_and_core_flops_per_user": layer0_basis_and_core,
        "paired_upper_core_flops_per_user": paired_upper_cores,
        "autonomous_upper_core_flops_per_user": autonomous_upper_cores,
        "migration_sufficient_total_flops_per_user": migration_sufficient_total,
        "migration_sufficient_over_exact_all": migration_sufficient_total / exact_all,
        "within_twenty_percent": migration_sufficient_total / exact_all <= 0.20,
        "generic_full_block_prototype_flops_per_user": prototype_full_block_total,
        "generic_full_block_prototype_over_exact_all": prototype_full_block_total / exact_all,
        "sidecar_scalars": sidecar_scalars,
        "cost_semantics": (
            "release-time constructor; terminal required layer is RMSNorm+KV-only; "
            "both model-specific raw inputs, fixed range finders, nonlinear causal "
            "attention, gate/residual boundaries, U0, and signed-core formation charged"
        ),
    }


def medium_rank_handoff_cost() -> dict[str, int | float | bool | str]:
    """Strict cost of the sole ``d=3, Current 4->8`` budget handoff profile."""

    exact_all = 4_771_282_944
    raw_input = 88_489_984
    initial_rank4_compression = 12_951_382
    rank4_block_body = 70_206_208
    rank4_post_compression = 13_282_134
    rank4_to_rank8_post_compression = 20_262_336
    rank8_block_body = 133_809_664
    rank8_post_compression = 20_729_280
    rank8_terminal_kv_only = 1_363_456

    current_arm = (
        raw_input
        + initial_rank4_compression
        + 3 * rank4_block_body
        + 2 * rank4_post_compression
        + rank4_to_rank8_post_compression
        + 2 * (rank8_block_body + rank8_post_compression)
        + rank8_terminal_kv_only
    )
    parent_prefix = 269_061_890
    layer0_basis_and_core = 1_247_808
    paired_early_upper_cores = 2 * 183_296
    rank8_current_minus_exact_parent_cores = 3 * 6_474_752
    total = (
        current_arm
        + parent_prefix
        + layer0_basis_and_core
        + paired_early_upper_cores
        + rank8_current_minus_exact_parent_cores
    )

    # A like-for-like terminal-K/V specialization of the frozen single-arm
    # rank-8 control.  Its previously documented generic full-block cost is
    # retained separately so comparisons cannot mix executor semantics.
    single_arm_rank8_full_block = 1_087_985_792
    unused_rank8_terminal_output = (
        rank8_block_body + rank8_post_compression - rank8_terminal_kv_only
    )
    single_arm_rank8_kv_terminal = single_arm_rank8_full_block - unused_rank8_terminal_output
    parent_unused_terminal_output = rank4_block_body + rank4_post_compression - 643_840
    generic_full_block_total = total + unused_rank8_terminal_output + parent_unused_terminal_output
    matrix_free_initial = matrix_free_input_cost(
        history_length=1024,
        hidden_size=192,
        temporal_num_freqs=16,
        rank=4,
        oversample=4,
        power_iterations=1,
    ).flops
    dense_initial_per_arm = raw_input + initial_rank4_compression
    matrix_free_two_arm_saving = 2 * (dense_initial_per_arm - matrix_free_initial)
    matrix_free_combined_total = total - matrix_free_two_arm_saving
    return {
        "profile": "d3_parent4_current4_then_current8",
        "exact_all_flops_per_user": exact_all,
        "current_handoff_arm_flops_per_user": current_arm,
        "parent_prefix_flops_per_user": parent_prefix,
        "layer0_basis_and_core_flops_per_user": layer0_basis_and_core,
        "paired_early_upper_core_flops_per_user": paired_early_upper_cores,
        "rank8_autonomous_upper_core_flops_per_user": (rank8_current_minus_exact_parent_cores),
        "migration_sufficient_total_flops_per_user": total,
        "migration_sufficient_over_exact_all": total / exact_all,
        "within_twenty_percent": total / exact_all <= 0.20,
        "generic_full_block_prototype_flops_per_user": generic_full_block_total,
        "generic_full_block_prototype_over_exact_all": generic_full_block_total / exact_all,
        "single_arm_rank8_kv_terminal_flops_per_user": single_arm_rank8_kv_terminal,
        "single_arm_rank8_kv_terminal_over_exact_all": single_arm_rank8_kv_terminal / exact_all,
        "single_arm_rank8_frozen_full_block_flops_per_user": (single_arm_rank8_full_block),
        "single_arm_rank8_frozen_full_block_over_exact_all": (
            single_arm_rank8_full_block / exact_all
        ),
        "matrix_free_initial_factor_flops_per_arm": matrix_free_initial,
        "dense_input_plus_initial_factor_flops_per_arm": dense_initial_per_arm,
        "matrix_free_two_arm_saving_flops_per_user": matrix_free_two_arm_saving,
        "matrix_free_combined_total_flops_per_user": matrix_free_combined_total,
        "matrix_free_combined_over_exact_all": matrix_free_combined_total / exact_all,
        "matrix_free_combined_within_twenty_percent": (
            matrix_free_combined_total / exact_all <= 0.20
        ),
        "scientific_mechanism": (
            "early equal-resolution release coupling followed by fixed total-rank "
            "handoff to the autonomous Current arm"
        ),
        "executor_component": (
            "matrix-free initial randomized range finding; semantics-preserving "
            "classical operator rewrite, not the scientific mechanism"
        ),
        "sidecar_scalars": 26_624,
        "cost_semantics": (
            "one fixed rank-budget handoff after layer 2; Current final and Parent "
            "prefix terminal layers are KV-only; all rank4-to-rank8 compression and "
            "three dense exact-Parent upper core projections are charged"
        ),
    }
