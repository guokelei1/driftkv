"""Matched Parent-cache control variate for release-mode replay.

This module asks a narrow mechanism question.  A rank-restricted Current
replay has a structured approximation error.  Can we cancel that error
without executing a second Parent Transformer replay by applying the same
fixed token-axis numerical operator directly to the already-persistent Parent
K/V cache?

For each layer, the Parent cache is approximated jointly in ``[K,V]`` with a
fixed randomized range finder.  The migration sidecar stores only the shared
layer-0 defect basis and signed per-layer cores for

``approx(Current replay) - approx(Parent cache)``.

The exact Parent cache remains the serving base.  No function in this module
accepts Current-Exact K/V, labels, candidates, or future events.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hstu_kvcache.models import HSTUKVCache
from insight_two.mode_space_replay import (
    FactorizedCacheLayer,
    FactorizedReplay,
    SharedModeSplice,
    randomized_token_basis,
    randomized_token_factors,
)


@dataclass(frozen=True)
class MatchedParentCacheApproximation:
    """Fixed rank approximation of each exact Parent joint ``[K,V]`` matrix."""

    layers: tuple[FactorizedCacheLayer, ...]
    seq_len: int


def _validate_parent_cache(parent: HSTUKVCache) -> tuple[int, int, int, int]:
    if parent.k.ndim != 4 or parent.k.shape != parent.v.shape:
        raise ValueError("Parent cache must contain matching [L,B,N,D] K/V")
    layers, batch, length, width = (
        int(value) for value in parent.k.shape
    )
    if parent.seq_len != length:
        raise ValueError("Parent cache seq_len differs from its token axis")
    if min(layers, batch, length, width) < 1:
        raise ValueError("Parent cache dimensions must be positive")
    return layers, batch, length, width


def _validate_replay(
    parent: HSTUKVCache,
    replay: FactorizedReplay,
) -> tuple[int, int, int, int]:
    shape = _validate_parent_cache(parent)
    if replay.cache.k.shape != parent.k.shape or replay.cache.v.shape != parent.v.shape:
        raise ValueError("Parent cache and replay shapes differ")
    if replay.cache.seq_len != parent.seq_len or len(replay.layers) != shape[0]:
        raise ValueError("Parent cache and replay lineage differ")
    return shape


@torch.inference_mode()
def approximate_exact_parent_cache(
    parent: HSTUKVCache,
    *,
    rank: int = 4,
    oversample: int = 4,
    power_iterations: int = 1,
    seed: int = 17,
) -> MatchedParentCacheApproximation:
    """Apply one fixed joint-``[K,V]`` operator to every Parent cache layer.

    K and V deliberately share the token factor.  The seed is advanced only
    by layer index, matching the fixed per-layer convention of reduced replay;
    it is independent of user, release, and observed quality.
    """

    layers, _, length, width = _validate_parent_cache(parent)
    if not 1 <= rank <= min(length, 2 * width):
        raise ValueError("rank must be in [1,min(history,2*cache_width)]")
    approximations: list[FactorizedCacheLayer] = []
    for layer in range(layers):
        joint = torch.cat((parent.k[layer], parent.v[layer]), dim=-1)
        factors = randomized_token_factors(
            joint,
            rank=rank,
            oversample=oversample,
            power_iterations=power_iterations,
            seed=seed + layer,
        )
        approximations.append(
            FactorizedCacheLayer(
                left=factors.left,
                key_core=factors.right[:, :, :width],
                value_core=factors.right[:, :, width:],
            )
        )
    return MatchedParentCacheApproximation(
        layers=tuple(approximations),
        seq_len=parent.seq_len,
    )


def _validate_approximation(
    parent: HSTUKVCache,
    approximation: MatchedParentCacheApproximation,
) -> None:
    layers, batch, length, width = _validate_parent_cache(parent)
    if approximation.seq_len != parent.seq_len or len(approximation.layers) != layers:
        raise ValueError("Parent approximation lineage differs")
    for factors in approximation.layers:
        key, value = factors.materialize()
        if key.shape != (batch, length, width) or value.shape != key.shape:
            raise ValueError("Parent approximation shape differs")


def _orthonormal_basis_check(
    basis: torch.Tensor,
    *,
    batch: int,
    length: int,
) -> None:
    if basis.ndim != 3 or basis.shape[:2] != (batch, length):
        raise ValueError("shared basis must have shape [B,N,r]")
    gram = basis.transpose(1, 2) @ basis
    identity = torch.eye(
        basis.shape[2], device=basis.device, dtype=basis.dtype
    ).expand_as(gram)
    if not torch.allclose(gram, identity, atol=2e-4, rtol=2e-4):
        raise ValueError("shared basis columns must be orthonormal")


@torch.inference_mode()
def matched_layer0_defect_basis(
    parent: HSTUKVCache,
    current_replay: FactorizedReplay,
    parent_approximation: MatchedParentCacheApproximation,
    *,
    rank: int = 8,
    oversample: int = 4,
    power_iterations: int = 0,
    seed: int = 1017,
) -> torch.Tensor:
    """Build U0 from matched approximate Current-minus-Parent layer-0 K/V."""

    _, batch, length, _ = _validate_replay(parent, current_replay)
    _validate_approximation(parent, parent_approximation)
    current_key, current_value = current_replay.layers[0].materialize()
    parent_key, parent_value = parent_approximation.layers[0].materialize()
    defect = torch.cat(
        (current_key - parent_key, current_value - parent_value), dim=-1
    )
    if defect.shape[:2] != (batch, length):
        raise RuntimeError("layer-0 matched defect layout differs")
    return randomized_token_basis(
        defect,
        rank=rank,
        oversample=oversample,
        power_iterations=power_iterations,
        seed=seed,
    )


@torch.inference_mode()
def splice_matched_parent_cache_differential(
    parent: HSTUKVCache,
    current_replay: FactorizedReplay,
    parent_approximation: MatchedParentCacheApproximation,
    basis: torch.Tensor,
) -> SharedModeSplice:
    """Add matched approximate release-difference cores to exact Parent K/V."""

    layers, batch, length, _ = _validate_replay(parent, current_replay)
    _validate_approximation(parent, parent_approximation)
    basis = basis.to(device=parent.k.device, dtype=parent.k.dtype)
    _orthonormal_basis_check(basis, batch=batch, length=length)
    transpose = basis.transpose(1, 2)
    migrated: list[tuple[torch.Tensor, torch.Tensor]] = []
    delta_keys: list[torch.Tensor] = []
    delta_values: list[torch.Tensor] = []
    for layer in range(layers):
        current = current_replay.layers[layer]
        matched_parent = parent_approximation.layers[layer]
        current_alignment = transpose @ current.left
        parent_alignment = transpose @ matched_parent.left
        delta_key = (
            current_alignment @ current.key_core
            - parent_alignment @ matched_parent.key_core
        )
        delta_value = (
            current_alignment @ current.value_core
            - parent_alignment @ matched_parent.value_core
        )
        migrated.append(
            (
                parent.k[layer] + basis @ delta_key,
                parent.v[layer] + basis @ delta_value,
            )
        )
        delta_keys.append(delta_key.detach())
        delta_values.append(delta_value.detach())
    return SharedModeSplice(
        cache=HSTUKVCache.from_layer_list(migrated, seq_len=parent.seq_len),
        basis=basis.detach(),
        delta_k_cores=tuple(delta_keys),
        delta_v_cores=tuple(delta_values),
    )


@torch.inference_mode()
def paired_replay_layer0_defect_basis(
    parent: HSTUKVCache,
    parent_replay: FactorizedReplay,
    current_replay: FactorizedReplay,
    *,
    rank: int = 8,
    oversample: int = 4,
    power_iterations: int = 0,
    seed: int = 1017,
) -> torch.Tensor:
    """Build the comparison U0 from two equal-resolution model replays."""

    _, batch, length, _ = _validate_replay(parent, current_replay)
    _validate_replay(parent, parent_replay)
    current_key, current_value = current_replay.layers[0].materialize()
    parent_key, parent_value = parent_replay.layers[0].materialize()
    defect = torch.cat(
        (current_key - parent_key, current_value - parent_value), dim=-1
    )
    if defect.shape[:2] != (batch, length):
        raise RuntimeError("layer-0 paired-replay defect layout differs")
    return randomized_token_basis(
        defect,
        rank=rank,
        oversample=oversample,
        power_iterations=power_iterations,
        seed=seed,
    )


@torch.inference_mode()
def splice_paired_replay_differential(
    parent: HSTUKVCache,
    parent_replay: FactorizedReplay,
    current_replay: FactorizedReplay,
    basis: torch.Tensor,
) -> SharedModeSplice:
    """Comparison path: add two-replay approximate finite difference to Parent."""

    layers, batch, length, _ = _validate_replay(parent, current_replay)
    _validate_replay(parent, parent_replay)
    basis = basis.to(device=parent.k.device, dtype=parent.k.dtype)
    _orthonormal_basis_check(basis, batch=batch, length=length)
    transpose = basis.transpose(1, 2)
    migrated: list[tuple[torch.Tensor, torch.Tensor]] = []
    delta_keys: list[torch.Tensor] = []
    delta_values: list[torch.Tensor] = []
    for layer in range(layers):
        current = current_replay.layers[layer]
        reduced_parent = parent_replay.layers[layer]
        current_alignment = transpose @ current.left
        parent_alignment = transpose @ reduced_parent.left
        delta_key = (
            current_alignment @ current.key_core
            - parent_alignment @ reduced_parent.key_core
        )
        delta_value = (
            current_alignment @ current.value_core
            - parent_alignment @ reduced_parent.value_core
        )
        migrated.append(
            (
                parent.k[layer] + basis @ delta_key,
                parent.v[layer] + basis @ delta_value,
            )
        )
        delta_keys.append(delta_key.detach())
        delta_values.append(delta_value.detach())
    return SharedModeSplice(
        cache=HSTUKVCache.from_layer_list(migrated, seq_len=parent.seq_len),
        basis=basis.detach(),
        delta_k_cores=tuple(delta_keys),
        delta_v_cores=tuple(delta_values),
    )
