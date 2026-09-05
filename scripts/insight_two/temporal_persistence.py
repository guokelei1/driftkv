"""Utilities for the frozen-cutover S4 persistence diagnostic.

This module deliberately contains no estimator.  The cutover correction passed
to these helpers may be an Exact oracle used to test whether a functional
boundary persists, but it must not be reported on an executable cost frontier.
"""

from __future__ import annotations

import hashlib

import torch


TIME_BUCKETS = (
    (0, 86_400, "[0d,1d)"),
    (86_400, 3 * 86_400, "[1d,3d)"),
    (3 * 86_400, 7 * 86_400, "[3d,7d)"),
    (7 * 86_400, 14 * 86_400, "[7d,14d)"),
)

APPEND_BUCKETS = (
    (0, 0, "0"),
    (1, 8, "[1,8]"),
    (9, 32, "[9,32]"),
    (33, 128, "[33,128]"),
    (129, 512, "[129,512]"),
    (513, None, ">512"),
)


def time_bucket(seconds_since_cutover: int) -> str:
    if seconds_since_cutover < 0:
        raise ValueError("persistence observation precedes cutover")
    for lower, upper, name in TIME_BUCKETS:
        if lower <= seconds_since_cutover < upper:
            return name
    return ">=14d"


def append_bucket(append_count: int) -> str:
    if append_count < 0:
        raise ValueError("append count must be non-negative")
    for lower, upper, name in APPEND_BUCKETS:
        if append_count >= lower and (upper is None or append_count <= upper):
            return name
    raise AssertionError("append bucket table is incomplete")


def remaining_parent_fraction(append_count: int, context: int) -> float:
    if append_count < 0 or context < 1:
        raise ValueError("invalid append count or context")
    return max(0, context - append_count) / context


def scale_correction(
    correction: tuple[torch.Tensor, ...], factor: float
) -> tuple[torch.Tensor, ...]:
    if factor < 0:
        raise ValueError("correction scale must be non-negative")
    return tuple(value * factor for value in correction)


def flatten_correction(correction: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if not correction:
        raise ValueError("correction must contain at least one layer")
    batches = {int(value.shape[0]) for value in correction}
    if len(batches) != 1:
        raise ValueError("correction layers have inconsistent batch dimensions")
    return torch.cat(
        [value.float().reshape(value.shape[0], -1) for value in correction], dim=1
    )


def correction_drift(
    current: tuple[torch.Tensor, ...], frozen: tuple[torch.Tensor, ...]
) -> dict[str, torch.Tensor]:
    if len(current) != len(frozen):
        raise ValueError("current and frozen correction layer counts differ")
    current_flat = flatten_correction(current)
    frozen_flat = flatten_correction(frozen).to(current_flat.device)
    if current_flat.shape != frozen_flat.shape:
        raise ValueError("current and frozen correction shapes differ")
    current_norm = current_flat.norm(dim=1)
    frozen_norm = frozen_flat.norm(dim=1)
    cosine = torch.nn.functional.cosine_similarity(current_flat, frozen_flat, dim=1)
    return {
        "direction_cosine": cosine,
        "current_norm": current_norm,
        "frozen_norm": frozen_norm,
        "current_to_frozen_norm_ratio": current_norm / frozen_norm.clamp_min(1e-20),
        "relative_l2": (current_flat - frozen_flat).norm(dim=1)
        / current_norm.clamp_min(1e-20),
    }


def correction_sha256(correction: tuple[torch.Tensor, ...]) -> str:
    """Hash shape, dtype and bytes so a runner can prove the sidecar stayed frozen."""
    digest = hashlib.sha256()
    for value in correction:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
