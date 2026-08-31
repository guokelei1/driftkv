"""Self-calibrating, progressive extensions of lightweight PRO.

The mechanism remains one per-user AV sidecar.  Two fixed, history-derived
probes estimate a shared direction and layer-wise old/recent amplitudes.  Only
the amplitudes change as old positions are evicted; no translated K/V is ever
materialised and no request candidate enters construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hstu_kvcache.models import HSTUKVCache

if __package__:
    from .pro_lazy_reader import PROCarrierLayout, PROProbeComponents
else:
    from pro_lazy_reader import PROCarrierLayout, PROProbeComponents


@dataclass(frozen=True)
class ProgressiveCarrierLayout:
    nominal_positions: int
    old_positions: int
    repair_evidence: int
    carriers: int
    represented_masses: tuple[int, ...]


@dataclass(frozen=True)
class ProgressivePROSidecar:
    directions: tuple[torch.Tensor, ...]
    old_amplitudes: torch.Tensor
    recent_amplitudes: torch.Tensor
    probe_direction_cosines: torch.Tensor
    probe_norm_ratios: torch.Tensor


def fixed_probe_items(item_ids: torch.Tensor, repair_width: int = 128) -> torch.Tensor:
    """Return the frozen latest and recent-window-start probe items."""
    if item_ids.ndim != 2 or item_ids.shape[1] < repair_width:
        raise ValueError("fixed probes require a full recent repair window")
    return torch.stack((item_ids[:, -1], item_ids[:, -repair_width]), dim=1)


@torch.inference_mode()
def build_progressive_parent_conditioned_carriers(
    *,
    parent_cache: HSTUKVCache,
    current,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    repair_width: int = 128,
    carrier_count: int,
) -> tuple[HSTUKVCache, ProgressiveCarrierLayout]:
    """Build deterministic possibly-unequal chronological carrier groups."""
    if item_ids.shape != behaviors.shape or item_ids.shape != time_deltas.shape:
        raise ValueError("raw prefix tensors differ in shape")
    if item_ids.ndim != 2 or item_ids.shape[0] != parent_cache.k.shape[1]:
        raise ValueError("raw prefix and Parent cache batch dimensions differ")
    if item_ids.shape[1] != parent_cache.seq_len:
        raise ValueError("raw prefix width differs from Parent cache")
    if not 1 <= carrier_count <= repair_width <= parent_cache.seq_len:
        raise ValueError("progressive carrier layout is outside the prefix")

    boundaries = torch.div(
        torch.arange(carrier_count + 1, device=item_ids.device) * repair_width,
        carrier_count,
        rounding_mode="floor",
    )
    masses = boundaries[1:] - boundaries[:-1]
    if bool((masses < 1).any()) or int(masses.sum()) != repair_width:
        raise RuntimeError("progressive carrier partition does not cover recent evidence")
    endpoints = boundaries[1:] - 1
    recent_items = item_ids[:, -repair_width:].index_select(1, endpoints)
    recent_behaviors = behaviors[:, -repair_width:].index_select(1, endpoints)
    recent_deltas = time_deltas[:, -repair_width:].index_select(1, endpoints)
    embedded = current.embed_inputs(recent_items, recent_behaviors, recent_deltas)
    old_positions = parent_cache.seq_len - repair_width
    parent_prefix = HSTUKVCache(
        k=parent_cache.k[:, :, :old_positions],
        v=parent_cache.v[:, :, :old_positions],
        seq_len=old_positions,
    )
    _, carriers = current.forward_with_cache_embedded_new_kv(parent_prefix, embedded)
    scaled_values = carriers.v * masses.to(carriers.v.dtype).view(1, 1, -1, 1)
    carriers = HSTUKVCache(k=carriers.k, v=scaled_values, seq_len=carrier_count)
    return carriers, ProgressiveCarrierLayout(
        nominal_positions=parent_cache.seq_len,
        old_positions=old_positions,
        repair_evidence=repair_width,
        carriers=carrier_count,
        represented_masses=tuple(int(value) for value in masses.cpu()),
    )


def _flatten(value: torch.Tensor) -> torch.Tensor:
    return value.float().reshape(value.shape[0], -1)


@torch.inference_mode()
def combine_two_probe_components(
    first: PROProbeComponents,
    second: PROProbeComponents,
) -> ProgressivePROSidecar:
    """Separate a stable two-probe direction from old/recent amplitudes."""
    if len(first.corrections) != len(second.corrections):
        raise ValueError("probe layer counts differ")
    directions = []
    old_amplitudes = []
    recent_amplitudes = []
    cosines = []
    norm_ratios = []
    for layer in range(len(first.corrections)):
        one = _flatten(first.corrections[layer])
        two = _flatten(second.corrections[layer])
        one_norm = one.norm(dim=1).clamp_min(1e-12)
        two_norm = two.norm(dim=1).clamp_min(1e-12)
        one_direction = one / one_norm[:, None]
        two_direction = two / two_norm[:, None]
        direction_sum = one_direction + two_direction
        sum_norm = direction_sum.norm(dim=1)
        direction = direction_sum / sum_norm.clamp_min(1e-12)[:, None]
        fallback = sum_norm < 1e-8
        if bool(fallback.any()):
            direction[fallback] = one_direction[fallback]
        cosines.append((one_direction * two_direction).sum(dim=1))
        norm_ratios.append(two_norm / one_norm)

        old_values = torch.stack(
            (_flatten(first.old_corrections[layer]), _flatten(second.old_corrections[layer])),
            dim=1,
        )
        recent_values = torch.stack(
            (
                _flatten(first.recent_corrections[layer]),
                _flatten(second.recent_corrections[layer]),
            ),
            dim=1,
        )
        old_amplitudes.append(torch.einsum("bpd,bd->bp", old_values, direction).mean(dim=1))
        recent_amplitudes.append(
            torch.einsum("bpd,bd->bp", recent_values, direction).mean(dim=1)
        )
        directions.append(direction.reshape_as(first.corrections[layer]).to(first.corrections[layer].dtype))
    return ProgressivePROSidecar(
        directions=tuple(directions),
        old_amplitudes=torch.stack(old_amplitudes, dim=1),
        recent_amplitudes=torch.stack(recent_amplitudes, dim=1),
        probe_direction_cosines=torch.stack(cosines, dim=1),
        probe_norm_ratios=torch.stack(norm_ratios, dim=1),
    )


def segment_coverage(
    evictions: torch.Tensor,
    *,
    old_positions: int = 384,
    recent_positions: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = evictions.float().clamp_min(0)
    old = (old_positions - values).clamp(0, old_positions) / float(old_positions)
    recent_evictions = (values - old_positions).clamp_min(0)
    recent = (recent_positions - recent_evictions).clamp(0, recent_positions) / float(
        recent_positions
    )
    return old, recent


def progressive_corrections(
    sidecar: ProgressivePROSidecar,
    evictions: torch.Tensor,
    *,
    old_positions: int = 384,
    recent_positions: int = 128,
) -> tuple[torch.Tensor, ...]:
    """Update only layer scalars using segment-aware eviction coverage."""
    old_coverage, recent_coverage = segment_coverage(
        evictions, old_positions=old_positions, recent_positions=recent_positions
    )
    amplitude = (
        sidecar.old_amplitudes * old_coverage[:, None]
        + sidecar.recent_amplitudes * recent_coverage[:, None]
    )
    return tuple(
        direction * amplitude[:, layer].to(direction.dtype).view(
            direction.shape[0], *([1] * (direction.ndim - 1))
        )
        for layer, direction in enumerate(sidecar.directions)
    )


def global_coverage_corrections(
    corrections: tuple[torch.Tensor, ...], evictions: torch.Tensor, context: int = 512
) -> tuple[torch.Tensor, ...]:
    factor = (context - evictions.float()).clamp(0, context) / float(context)
    return tuple(
        value * factor.to(value.dtype).view(value.shape[0], *([1] * (value.ndim - 1)))
        for value in corrections
    )


def legacy_layout(layout: ProgressiveCarrierLayout) -> PROCarrierLayout:
    """Return the old layout type only when all groups have equal mass."""
    unique = set(layout.represented_masses)
    if len(unique) != 1:
        raise ValueError("unequal progressive groups have no scalar represented mass")
    return PROCarrierLayout(
        nominal_positions=layout.nominal_positions,
        old_positions=layout.old_positions,
        repair_evidence=layout.repair_evidence,
        carriers=layout.carriers,
        represented_mass=next(iter(unique)),
    )
