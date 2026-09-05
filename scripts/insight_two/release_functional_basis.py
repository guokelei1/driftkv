"""Oracle diagnostics for user-shared directions within one release edge.

Projection coefficients for evaluation users are computed from their Exact
functional targets.  The helpers therefore measure representation structure
only and must never be used as an executable migration estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OracleReleaseBasis:
    mean: np.ndarray
    layer_bases: tuple[np.ndarray, ...]
    layer_singular_values: tuple[np.ndarray, ...]

    def project(self, targets: np.ndarray, rank: int) -> np.ndarray:
        if targets.ndim != 3 or targets.shape[1:] != self.mean.shape:
            raise ValueError("release-basis targets must have shape [users,layers,width]")
        if rank < 0:
            raise ValueError("release-basis rank must be non-negative")
        centered = targets.astype(np.float64) - self.mean
        if rank == 0:
            return np.broadcast_to(self.mean, targets.shape).copy()
        projected = np.empty_like(centered)
        for layer, basis in enumerate(self.layer_bases):
            selected = basis[: min(rank, len(basis))]
            projected[:, layer] = centered[:, layer] @ selected.T @ selected
        return projected + self.mean


def fit_oracle_release_basis(targets: np.ndarray) -> OracleReleaseBasis:
    if targets.ndim != 3 or targets.shape[0] < 2:
        raise ValueError("release basis requires [users,layers,width] with two users")
    values = targets.astype(np.float64)
    mean = values.mean(axis=0)
    bases = []
    singular_values = []
    for layer in range(values.shape[1]):
        _, singular, vh = np.linalg.svd(values[:, layer] - mean[layer], full_matrices=False)
        bases.append(vh)
        singular_values.append(singular)
    return OracleReleaseBasis(
        mean=mean,
        layer_bases=tuple(bases),
        layer_singular_values=tuple(singular_values),
    )


def rank_at_energy(singular_values: np.ndarray, threshold: float) -> int:
    if not 0 < threshold <= 1:
        raise ValueError("energy threshold must be in (0,1]")
    energy = np.square(singular_values.astype(np.float64))
    if energy.sum() <= 1e-20:
        return 0
    return int(np.searchsorted(np.cumsum(energy) / energy.sum(), threshold) + 1)
