"""Small shared-reader primitives for a fixed-state, UID-disjoint Insight probe.

This module reuses the existing joint response ridge solver. It has no model,
history, release-chain or output-directory dependencies at execution time.
Summaries contain only source-cache K/V means; teachers enter ``fit_layer``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from design.diagnose_native_input import fit_joint, solver_canary
from design.diagnose_summary_objective import source_projection
from hstu_kvcache.models import HSTUKVCache


def cache_features(cache: HSTUKVCache) -> torch.Tensor:
    """Return source-only K/V means in [users, layers * 2 * width] order."""
    means = torch.stack((cache.k.float().mean(2), cache.v.float().mean(2)), dim=2)
    return means.permute(1, 0, 2, 3).flatten(1)


@dataclass
class SummaryProjection:
    center: torch.Tensor
    scale: torch.Tensor
    projection: torch.Tensor
    diagnostics: dict

    @classmethod
    def fit(cls, source_cache: HSTUKVCache, rank: int = 32) -> "SummaryProjection":
        return fit_summary(source_cache, rank=rank)

    @property
    def metadata(self) -> dict:
        return self.diagnostics

    def encode(self, cache: HSTUKVCache) -> torch.Tensor:
        """Return CPU double coordinates; move the small result for GPU reads."""
        return self.encode_features(cache_features(cache).cpu())

    def encode_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply frozen calibration-only coordinates to any source summary."""
        center, scale, projection = (
            value.to(features.device) for value in (self.center, self.scale, self.projection)
        )
        latent = (features.double() - center) / scale @ projection
        return torch.cat((torch.ones_like(latent[:, :1]), latent), dim=-1)

    def state_dict(self) -> dict:
        return {
            "center": self.center.cpu(),
            "scale": self.scale.cpu(),
            "projection": self.projection.cpu(),
            "diagnostics": self.diagnostics,
        }


def fit_summary(source_cache: HSTUKVCache, rank: int = 32) -> SummaryProjection:
    """Fit whitening only on the supplied calibration users' source caches.

    The caller must exclude evaluation users before invoking this function.
    The projection uses at most ``rank`` nonzero principal directions.
    """
    features = cache_features(source_cache).cpu()
    fit_mask = torch.ones(len(features), dtype=torch.bool, device=features.device)
    _, state, diagnostics = source_projection(features, fit_mask, rank=rank)
    return SummaryProjection(**state, diagnostics=diagnostics)


def fit_layer(latent, query, wanted, observed, counts):
    """Fit T*r + b(s) + A(s)q with the retained aggregate-response ridge .01.

    query/wanted are [users, heads, candidates, head_width]; wanted is the
    teacher-minus-native response divided by count. observed is the native
    response divided by count, in [users, candidates, full_width] order.
    latent is [users, 1+rank], with a leading one; use only that column for
    the shared rule without an explicit cache summary. All normalization is
    estimated from these calibration examples, separately for each arm.
    """
    # PyTorch 2.12 / MKL 2024.2 batched CPU LU fails in DLASWP with multiple
    # intra-op threads at the experiment's 6 x 225 x 225 system size. Keep the
    # same double-precision ridge solve, with one CPU thread only while fitting.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        parameters, stats = fit_joint(latent, query, wanted, observed, counts)
    finally:
        torch.set_num_threads(previous_threads)
    stats["cpu_solver_threads"] = 1
    return parameters, stats


def prepare_layer(latent, parameters, *, dtype=torch.float32):
    """Generate one cache-conditioned affine view, reusable across candidates."""
    device = latent.device
    p = {key: value.to(device=device, dtype=dtype) for key, value in parameters.items()}
    coefficients = torch.einsum("br,rhjd->bhjd", latent.to(dtype), p["weights"])
    slope = coefficients[:, :, 1:] / p["query_scale"].transpose(-2, -1)
    intercept = coefficients[:, :, 0] - (p["query_center"] @ slope).squeeze(-2)
    return {
        "intercept": intercept,
        "slope": slope,
        "read_weights": p["read_weights"],
        "read_center": p["read_center"],
        "read_scale": p["read_scale"],
    }


def apply_prepared_layer(query, native, counts, view):
    """Correct actual full-history response without reading any teacher cache."""
    observed = native.transpose(1, 2).flatten(2) / counts[:, None, None]
    observed = (observed - view["read_center"]) / view["read_scale"]
    delta = view["intercept"][:, :, None] + query @ view["slope"]
    delta = delta + torch.einsum("bqa,had->bhqd", observed, view["read_weights"])
    return native + delta * counts[:, None, None, None]


def apply_layer(latent, query, native, counts, parameters):
    """Convenience wrapper; use a prepared view for repeated candidate reads."""
    view = prepare_layer(latent, parameters, dtype=query.dtype)
    return apply_prepared_layer(query, native, counts, view)


def parameters_to_device(parameters, device, dtype=torch.float32):
    """Move shared parameters once before evaluating a series of GPU batches."""
    return [
        {key: value.to(device=device, dtype=dtype) for key, value in layer.items()}
        for layer in parameters
    ]


def make_history_override(parameters, latent, counts):
    """Return a callback for adaptation.reader.score(history_override=...).

    During sequential fitting, layers after the fitted prefix pass through
    unchanged. No source/teacher tensor is captured by this callback.
    """
    views = [prepare_layer(latent, parameters_for_layer) for parameters_for_layer in parameters]

    def transform(layer, query, native):
        if layer >= len(views):
            return native
        return apply_prepared_layer(query, native, counts, views[layer])

    return transform


def self_check() -> dict:
    """Tiny CPU reference comparisons for the two numerical transformations."""
    solver_canary(torch.device("cpu"))
    rng = torch.Generator().manual_seed(171920)
    k = torch.randn(2, 8, 5, 6, generator=rng)
    v = torch.randn(2, 8, 5, 6, generator=rng)
    calibration = HSTUKVCache(k[:, :6], v[:, :6], 5)
    projection = fit_summary(calibration, rank=3)
    latent = projection.encode(calibration)
    whitened = latent[:, 1:]
    torch.testing.assert_close(whitened.mean(0), torch.zeros(3, dtype=torch.double), atol=1e-12, rtol=0)
    torch.testing.assert_close(whitened.T @ whitened / 5, torch.eye(3, dtype=torch.double), atol=1e-10, rtol=1e-10)
    reordered = HSTUKVCache(k[:, :6, [4, 0, 2, 1, 3]], v[:, :6, [4, 0, 2, 1, 3]], 5)
    torch.testing.assert_close(projection.encode(reordered), latent, atol=2e-6, rtol=2e-6)

    query = torch.randn(6, 2, 9, 3, generator=rng, dtype=torch.double)
    native = torch.randn(6, 2, 9, 3, generator=rng, dtype=torch.double)
    wanted = torch.randn(6, 2, 9, 3, generator=rng, dtype=torch.double)
    counts = torch.arange(1, 7, dtype=torch.double)
    observed = native.transpose(1, 2).flatten(2) / counts[:, None, None]
    p, stats = fit_layer(latent, query, wanted, observed, counts)
    normalized_q = (query - p["query_center"]) / p["query_scale"]
    x = torch.cat((torch.ones_like(normalized_q[..., :1]), normalized_q), dim=-1)
    normalized_read = (observed - p["read_center"]) / p["read_scale"]
    prediction = torch.einsum("bhqj,br,rhjd->bhqd", x, latent, p["weights"])
    prediction += torch.einsum("bqa,had->bhqd", normalized_read, p["read_weights"])
    expected = native + prediction * counts[:, None, None, None]
    actual = apply_layer(latent, query, native, counts, p)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    callback = make_history_override([p], latent, counts.float())
    torch.testing.assert_close(callback(0, query.float(), native.float()), expected.float(), atol=3e-6, rtol=3e-6)
    assert torch.equal(callback(1, query.float(), native.float()), native.float())
    # Retain the actual no-summary system size that triggers multithreaded MKL.
    q_full = torch.randn(32, 6, 16, 32, generator=rng, dtype=torch.double)
    y_full = torch.randn(32, 6, 16, 32, generator=rng, dtype=torch.double)
    r_full = torch.randn(32, 16, 192, generator=rng, dtype=torch.double)
    before = torch.get_num_threads()
    _, actual_size_stats = fit_layer(torch.ones(32, 1, dtype=torch.double), q_full,
                                    y_full, r_full, torch.full((32,), 1024.0))
    assert torch.get_num_threads() == before
    return {
        "status": "passed",
        "ridge_reference": "existing explicit weighted normal-equation canary",
        "summary_whitening": "calibration-only covariance identity and event permutation",
        "affine_read_equivalence_max_abs": float((actual - expected).abs().max()),
        "normal_relative_residual": stats["normal_relative_residual"],
        "actual_size_normal_relative_residual": actual_size_stats["normal_relative_residual"],
        "cpu_solver_threads": actual_size_stats["cpu_solver_threads"],
    }


if __name__ == "__main__":
    import json

    torch.set_num_threads(2)
    print(json.dumps(self_check(), indent=2))
