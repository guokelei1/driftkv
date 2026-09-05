"""Address-aware oracle signed-response memory for HSTU attention.

This module changes only the landmark geometry of the frozen signed-response
diagnostic.  Instead of choosing landmarks by chronological strata, it covers
the layer-0 cross-version attention address space.  For every history position
we concatenate its Current-Exact and Parent-Reuse layer-0 keys, L2-normalize
that vector, and construct one deterministic nested farthest-first ordering.

Each prefix of that ordering defines a Voronoi quadrature.  A selected center
represents the number of source positions assigned to it, and that positive
integer cluster mass weights both the Current atom and its paired negative
Parent atom.  The selected positions and masses are shared by every layer and
head.  Candidate queries still read the resulting Current-minus-Parent
residual through the model's native attention kernel.

This remains an oracle representation diagnostic: construction reads the full
Current-Exact cache, including upper-layer K/V at the selected positions.  It
does not fit targets, read labels, or mutate either input cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hstu_kvcache.models import HSTUKVCache
from insight_two.signed_response_memory import OracleSignedResponseMemory


@dataclass(frozen=True)
class AddressLandmarkSelection:
    """One prefix of a nested attention-address landmark ordering."""

    source_length: int
    selected_positions: torch.Tensor
    cluster_masses: torch.Tensor
    assignments: torch.Tensor

    @property
    def sample_count(self) -> int:
        return int(self.selected_positions.numel())


def _validate_cache_pair(
    exact_cache: HSTUKVCache,
    reuse_cache: HSTUKVCache,
) -> None:
    for cache, name in ((exact_cache, "exact_cache"), (reuse_cache, "reuse_cache")):
        if cache.k.ndim != 4 or cache.k.shape != cache.v.shape:
            raise ValueError(f"{name} must contain matching [layers,B,N,width] K/V")
        if cache.k.shape[2] != cache.seq_len:
            raise ValueError(f"{name} seq_len must equal its tensor history width")
        if not cache.k.is_floating_point() or not cache.v.is_floating_point():
            raise ValueError(f"{name} K/V must be floating point")
    if exact_cache.k.shape != reuse_cache.k.shape:
        raise ValueError("Exact and Reuse cache tensor shapes differ")
    if exact_cache.seq_len != reuse_cache.seq_len:
        raise ValueError("Exact and Reuse cache lengths differ")
    if exact_cache.k.device != reuse_cache.k.device:
        raise ValueError("Exact and Reuse caches must be on the same device")
    if exact_cache.k.dtype != reuse_cache.k.dtype:
        raise ValueError("Exact and Reuse caches must have the same dtype")
    if exact_cache.k.shape[1] != 1:
        raise ValueError(
            "address landmarks are selected per user and require cache batch size 1"
        )


def _address_features(
    exact_cache: HSTUKVCache,
    reuse_cache: HSTUKVCache,
) -> torch.Tensor:
    """Return normalized [N, 2*kv_width] Current+Parent layer-0 addresses."""

    # Geometry is evaluated in float32 even when cache storage uses a lower
    # precision.  This is selection metadata only; the native K/V atoms retain
    # their original dtype.
    features = torch.cat(
        (exact_cache.k[0, 0], reuse_cache.k[0, 0]), dim=-1
    ).to(dtype=torch.float32)
    if not torch.isfinite(features).all():
        raise ValueError("layer-0 address features must be finite")
    return F.normalize(features, p=2.0, dim=-1, eps=1e-12)


def _farthest_first_order(features: torch.Tensor, sample_count: int) -> torch.Tensor:
    """Return a deterministic nested farthest-first prefix.

    The first center is farthest from the population mean.  Every later center
    maximizes its minimum squared Euclidean distance to the selected set.
    ``torch.argmax`` returns the first maximum, so token-index ties resolve to
    the smallest unselected position.
    """

    if features.ndim != 2:
        raise ValueError("features must have shape [history, feature_width]")
    history_length = features.shape[0]
    if not 1 <= sample_count <= history_length:
        raise ValueError("sample_count must be in [1, history_length]")

    mean = features.mean(dim=0, keepdim=True)
    first_distances = (features - mean).square().sum(dim=-1)
    first = int(torch.argmax(first_distances).item())
    selected = torch.empty(sample_count, dtype=torch.long, device=features.device)
    selected[0] = first
    chosen = torch.zeros(history_length, dtype=torch.bool, device=features.device)
    chosen[first] = True
    min_distances = (features - features[first]).square().sum(dim=-1)

    for offset in range(1, sample_count):
        eligible_distances = min_distances.masked_fill(chosen, -1.0)
        position = int(torch.argmax(eligible_distances).item())
        selected[offset] = position
        chosen[position] = True
        distance_to_new = (features - features[position]).square().sum(dim=-1)
        min_distances = torch.minimum(min_distances, distance_to_new)
    return selected


def _voronoi_assignment(
    features: torch.Tensor,
    selected_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign source positions to nearest centers and return positive masses.

    Selected positions are fixed to their own centers before assigning the
    remaining positions.  This makes every mass positive even when two source
    address vectors are identical.  For non-center distance ties, the center
    with the smallest source token index wins.
    """

    centers = features.index_select(0, selected_positions)
    distances = (features[:, None, :] - centers[None, :, :]).square().sum(dim=-1)

    # Reorder columns only for deterministic nearest-center tie resolution.
    tie_order = torch.argsort(selected_positions)
    nearest_in_tie_order = torch.argmin(distances.index_select(1, tie_order), dim=1)
    assignments = tie_order.index_select(0, nearest_in_tie_order)
    assignments = assignments.to(dtype=torch.long)
    assignments[selected_positions] = torch.arange(
        selected_positions.numel(), device=features.device, dtype=torch.long
    )
    masses = torch.bincount(assignments, minlength=selected_positions.numel())
    if torch.any(masses <= 0):
        raise RuntimeError("every selected landmark must have positive cluster mass")
    return assignments, masses


@torch.inference_mode()
def select_address_landmarks(
    exact_cache: HSTUKVCache,
    reuse_cache: HSTUKVCache,
    *,
    sample_count: int,
) -> AddressLandmarkSelection:
    """Select one deterministic prefix and compute its Voronoi cluster masses."""

    _validate_cache_pair(exact_cache, reuse_cache)
    features = _address_features(exact_cache, reuse_cache)
    selected = _farthest_first_order(features, sample_count)
    assignments, masses = _voronoi_assignment(features, selected)
    return AddressLandmarkSelection(
        source_length=exact_cache.seq_len,
        selected_positions=selected,
        cluster_masses=masses,
        assignments=assignments,
    )


@torch.inference_mode()
def build_oracle_address_response_memory(
    exact_cache: HSTUKVCache,
    reuse_cache: HSTUKVCache,
    *,
    sample_count: int,
) -> OracleSignedResponseMemory:
    """Build paired signed native-attention atoms at address-aware landmarks."""

    selection = select_address_landmarks(
        exact_cache, reuse_cache, sample_count=sample_count
    )
    positions = selection.selected_positions
    masses = selection.cluster_masses.to(
        device=exact_cache.v.device, dtype=exact_cache.v.dtype
    )
    exact_k = exact_cache.k.index_select(2, positions)
    exact_v = exact_cache.v.index_select(2, positions)
    reuse_k = reuse_cache.k.index_select(2, positions)
    reuse_v = reuse_cache.v.index_select(2, positions)
    value_weight = masses.view(1, 1, sample_count, 1)

    return OracleSignedResponseMemory(
        keys=torch.cat((exact_k, reuse_k), dim=2).detach(),
        signed_values=torch.cat(
            (exact_v * value_weight, -reuse_v * value_weight), dim=2
        ).detach(),
        source_positions=torch.cat((positions, positions), dim=0),
        sample_positions=positions,
        # The frozen memory field stores generic quadrature weights despite its
        # historical IPW name.  Here those weights are Voronoi cluster masses.
        inverse_inclusion_probabilities=masses,
        source_length=exact_cache.seq_len,
    )
