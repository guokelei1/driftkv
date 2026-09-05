"""Oracle projections for diagnosing temporal coordinates of an S4 response.

Projection coefficients use the current request's Exact functional correction.
They are diagnostic ceilings, never executable estimators.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TemporalProjection:
    correction: tuple[torch.Tensor, ...]
    coefficients: torch.Tensor
    relative_l2: torch.Tensor


def _validate(
    current: tuple[torch.Tensor, ...], frozen: tuple[torch.Tensor, ...]
) -> None:
    if not current or len(current) != len(frozen):
        raise ValueError("current and frozen correction layers differ")
    if any(left.shape != right.shape for left, right in zip(current, frozen, strict=True)):
        raise ValueError("current and frozen correction shapes differ")


def project_global_coefficient(
    current: tuple[torch.Tensor, ...], frozen: tuple[torch.Tensor, ...]
) -> TemporalProjection:
    """Project the current layered response onto one frozen user direction."""
    _validate(current, frozen)
    current_flat = torch.cat(
        [value.float().reshape(value.shape[0], -1) for value in current], dim=1
    )
    frozen_flat = torch.cat(
        [value.float().reshape(value.shape[0], -1) for value in frozen], dim=1
    ).to(current_flat.device)
    coefficient = (current_flat * frozen_flat).sum(dim=1) / frozen_flat.square().sum(
        dim=1
    ).clamp_min(1e-20)
    projected = tuple(
        reference * coefficient.to(reference.dtype).reshape(
            reference.shape[0], *([1] * (reference.ndim - 1))
        )
        for reference in frozen
    )
    projected_flat = torch.cat(
        [value.float().reshape(value.shape[0], -1) for value in projected], dim=1
    )
    relative_l2 = (current_flat - projected_flat).norm(dim=1) / current_flat.norm(
        dim=1
    ).clamp_min(1e-20)
    return TemporalProjection(projected, coefficient[:, None], relative_l2)


def project_layerwise_coefficients(
    current: tuple[torch.Tensor, ...], frozen: tuple[torch.Tensor, ...]
) -> TemporalProjection:
    """Project each layer onto its own frozen direction, yielding L scalars."""
    _validate(current, frozen)
    projected = []
    coefficients = []
    current_flat = []
    projected_flat = []
    for actual, reference in zip(current, frozen, strict=True):
        actual_value = actual.float().reshape(actual.shape[0], -1)
        reference_value = reference.float().reshape(reference.shape[0], -1).to(
            actual_value.device
        )
        coefficient = (actual_value * reference_value).sum(
            dim=1
        ) / reference_value.square().sum(dim=1).clamp_min(1e-20)
        value = reference * coefficient.to(reference.dtype).reshape(
            reference.shape[0], *([1] * (reference.ndim - 1))
        )
        projected.append(value)
        coefficients.append(coefficient)
        current_flat.append(actual_value)
        projected_flat.append(value.float().reshape(value.shape[0], -1))
    actual_all = torch.cat(current_flat, dim=1)
    projected_all = torch.cat(projected_flat, dim=1)
    relative_l2 = (actual_all - projected_all).norm(dim=1) / actual_all.norm(
        dim=1
    ).clamp_min(1e-20)
    return TemporalProjection(
        tuple(projected), torch.stack(coefficients, dim=1), relative_l2
    )
