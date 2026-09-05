"""Oracle diagnostics for low-rank release differential structure.

Two deliberately distinct ceilings are implemented:

1. a per-user, per-layer truncated SVD of the exact joint ``Delta[K,V]``;
2. projection of exact ``Delta K`` and ``Delta V`` onto left singular
   subspaces derived only from the corresponding release weight deltas.

Both paths read exact Current K/V to obtain user coefficients.  They are not
executable migration actions.  Low-rank compression is not the Design claim;
the only possible follow-up is to generate the differential by propagating it
along the model's parameter-change path without Current Exact state.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from hstu_kvcache.models import HSTUKVCache

TOKEN_SVD_RANKS = (1, 2, 4, 8, 16)
PARAMETER_SUBSPACE_RANKS = (4, 8, 16, 32)


@dataclass(frozen=True)
class JointDeltaSVDOracle:
    """One layer's exact joint-delta truncated-SVD oracle."""

    rank: int
    history_basis: torch.Tensor
    coefficients: torch.Tensor
    basis: torch.Tensor
    singular_values: torch.Tensor
    reconstructed_k: torch.Tensor
    reconstructed_v: torch.Tensor
    source_length: int
    width: int

    @property
    def stored_scalars(self) -> int:
        return self.coefficients.numel() + self.basis.numel()


@dataclass(frozen=True)
class KVParameterLeftSubspace:
    """Release-only K/V output bases from projection-weight deltas.

    ``rank`` is the requested rank cap.  A truly rank-deficient parameter
    delta stores only its numerically supported directions; arbitrary null-SVD
    completion vectors must never inflate the oracle ceiling.
    """

    rank: int
    k_basis: torch.Tensor
    v_basis: torch.Tensor
    k_singular_values: torch.Tensor
    v_singular_values: torch.Tensor

    @property
    def k_realized_rank(self) -> int:
        return int(self.k_basis.shape[1])

    @property
    def v_realized_rank(self) -> int:
        return int(self.v_basis.shape[1])

    @property
    def shared_basis_scalars(self) -> int:
        return self.k_basis.numel() + self.v_basis.numel()


@dataclass(frozen=True)
class ParameterSubspaceOracleProjection:
    """Exact user delta projected into a release-derived subspace."""

    rank: int
    k_coefficients: torch.Tensor
    v_coefficients: torch.Tensor
    reconstructed_k: torch.Tensor
    reconstructed_v: torch.Tensor
    subspace: KVParameterLeftSubspace

    @property
    def per_user_stored_scalars(self) -> int:
        return self.k_coefficients.numel() + self.v_coefficients.numel()


def _validate_layer_kv(
    parent_k: torch.Tensor,
    parent_v: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
) -> tuple[int, int]:
    expected = parent_k.shape
    if parent_k.ndim != 3 or expected[0] != 1:
        raise ValueError("layer K/V must have shape [1,N,W]")
    if any(value.shape != expected for value in (parent_v, current_k, current_v)):
        raise ValueError("Parent and Current layer K/V shapes differ")
    if not all(value.is_floating_point() for value in (parent_k, parent_v, current_k, current_v)):
        raise ValueError("K/V must be floating point")
    if not all(value.device == parent_k.device for value in (parent_v, current_k, current_v)):
        raise ValueError("Parent and Current K/V must share a device")
    return int(expected[1]), int(expected[2])


def _validate_cache_pair(
    parent_cache: HSTUKVCache,
    current_cache: HSTUKVCache,
) -> tuple[int, int, int]:
    if parent_cache.k.ndim != 4 or parent_cache.k.shape != parent_cache.v.shape:
        raise ValueError("Parent cache must contain matching [L,1,N,W] K/V")
    if (
        current_cache.k.shape != parent_cache.k.shape
        or current_cache.v.shape != parent_cache.v.shape
    ):
        raise ValueError("Parent and Current cache layouts differ")
    if parent_cache.k.shape[1] != 1:
        raise ValueError("release differential diagnostic is per user")
    if parent_cache.seq_len != current_cache.seq_len:
        raise ValueError("Parent and Current cache lengths differ")
    if parent_cache.k.shape[2] != parent_cache.seq_len:
        raise ValueError("cache tensor width differs from seq_len")
    return (
        int(parent_cache.k.shape[0]),
        int(parent_cache.k.shape[2]),
        int(parent_cache.k.shape[3]),
    )


def joint_kv_delta_matrix(
    parent_k: torch.Tensor,
    parent_v: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
) -> torch.Tensor:
    """Return exact ``Delta[K,V]`` as ``[N,2W]`` in float32."""

    _validate_layer_kv(parent_k, parent_v, current_k, current_v)
    return torch.cat(
        (
            current_k[0].float() - parent_k[0].float(),
            current_v[0].float() - parent_v[0].float(),
        ),
        dim=1,
    )


def _validate_rank(rank: int, rows: int, columns: int) -> None:
    if not 1 <= rank <= min(rows, columns):
        raise ValueError("rank must be in [1,min(rows,columns)]")


@torch.inference_mode()
def joint_delta_token_svd_oracle(
    parent_k: torch.Tensor,
    parent_v: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    *,
    rank: int,
) -> JointDeltaSVDOracle:
    """Fit and reconstruct one exact joint cache delta with truncated SVD."""

    length, width = _validate_layer_kv(parent_k, parent_v, current_k, current_v)
    delta = joint_kv_delta_matrix(parent_k, parent_v, current_k, current_v)
    _validate_rank(rank, *delta.shape)
    left, singular, right = torch.linalg.svd(delta, full_matrices=False)
    return _joint_delta_oracle_from_svd(
        parent_k,
        parent_v,
        left,
        singular,
        right,
        rank=rank,
        source_length=length,
        width=width,
    )


def _joint_delta_oracle_from_svd(
    parent_k: torch.Tensor,
    parent_v: torch.Tensor,
    left: torch.Tensor,
    singular: torch.Tensor,
    right: torch.Tensor,
    *,
    rank: int,
    source_length: int,
    width: int,
) -> JointDeltaSVDOracle:
    """Truncate already-fitted factors so a rank grid pays for one SVD."""

    coefficients = left[:, :rank] * singular[:rank]
    basis = right[:rank]
    reconstructed = coefficients @ basis
    delta_k, delta_v = reconstructed.split(width, dim=1)
    return JointDeltaSVDOracle(
        rank=rank,
        history_basis=left[:, :rank].detach(),
        coefficients=coefficients.detach(),
        basis=basis.detach(),
        singular_values=singular.detach(),
        reconstructed_k=(parent_k.float() + delta_k.unsqueeze(0)).detach(),
        reconstructed_v=(parent_v.float() + delta_v.unsqueeze(0)).detach(),
        source_length=source_length,
        width=width,
    )


@torch.inference_mode()
def joint_delta_token_svd_grid(
    parent_cache: HSTUKVCache,
    current_cache: HSTUKVCache,
    *,
    ranks: Iterable[int] = TOKEN_SVD_RANKS,
) -> dict[tuple[int, int], JointDeltaSVDOracle]:
    """Fit the preregistered token-SVD rank grid independently per layer."""

    layers, _, _ = _validate_cache_pair(parent_cache, current_cache)
    ranks = tuple(int(rank) for rank in ranks)
    if len(set(ranks)) != len(ranks):
        raise ValueError("token-SVD ranks must be unique")
    result: dict[tuple[int, int], JointDeltaSVDOracle] = {}
    for layer in range(layers):
        delta = joint_kv_delta_matrix(
            parent_cache.k[layer],
            parent_cache.v[layer],
            current_cache.k[layer],
            current_cache.v[layer],
        )
        left, singular, right = torch.linalg.svd(delta, full_matrices=False)
        for rank in ranks:
            _validate_rank(rank, *delta.shape)
            result[(layer, rank)] = _joint_delta_oracle_from_svd(
                parent_cache.k[layer],
                parent_cache.v[layer],
                left,
                singular,
                right,
                rank=rank,
                source_length=delta.shape[0],
                width=parent_cache.k.shape[-1],
            )
    return result


def reconstruct_cache_from_token_svd(
    parent_cache: HSTUKVCache,
    oracles: dict[tuple[int, int], JointDeltaSVDOracle],
    *,
    rank: int,
) -> HSTUKVCache:
    """Stack one rank's Parent-plus-delta layer reconstructions."""

    layers, length, _ = _validate_cache_pair(parent_cache, parent_cache)
    selected = [oracles[(layer, rank)] for layer in range(layers)]
    if any(oracle.source_length != length for oracle in selected):
        raise ValueError("token-SVD oracle history lengths differ")
    return HSTUKVCache(
        k=torch.stack([oracle.reconstructed_k for oracle in selected], dim=0),
        v=torch.stack([oracle.reconstructed_v for oracle in selected], dim=0),
        seq_len=length,
    )


def _projection_weight(module) -> torch.Tensor:
    if not hasattr(module, "weight") or module.weight.ndim != 2:
        raise ValueError("projection module must expose a matrix weight")
    return module.weight.detach().float()


@torch.inference_mode()
def kv_parameter_left_subspace(
    parent_attention,
    current_attention,
    *,
    rank: int,
) -> KVParameterLeftSubspace:
    """Derive separate K and V output bases from release parameter deltas."""

    parent_k = _projection_weight(parent_attention.k_proj)
    current_k = _projection_weight(current_attention.k_proj)
    parent_v = _projection_weight(parent_attention.v_proj)
    current_v = _projection_weight(current_attention.v_proj)
    if parent_k.shape != current_k.shape or parent_v.shape != current_v.shape:
        raise ValueError("Parent and Current K/V projection shapes differ")
    if parent_k.shape != parent_v.shape:
        raise ValueError("K and V projection shapes differ")
    _validate_rank(rank, *parent_k.shape)
    k_left, k_singular, v_left, v_singular = _kv_parameter_left_svd(
        parent_k,
        current_k,
        parent_v,
        current_v,
    )
    return _truncate_kv_parameter_subspace(
        k_left,
        k_singular,
        v_left,
        v_singular,
        rank=rank,
        matrix_columns=parent_k.shape[1],
    )


def _kv_parameter_left_svd(
    parent_k: torch.Tensor,
    current_k: torch.Tensor,
    parent_v: torch.Tensor,
    current_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k_left, k_singular, _ = torch.linalg.svd(
        current_k - parent_k,
        full_matrices=False,
    )
    v_left, v_singular, _ = torch.linalg.svd(
        current_v - parent_v,
        full_matrices=False,
    )
    return k_left, k_singular, v_left, v_singular


def _truncate_kv_parameter_subspace(
    k_left: torch.Tensor,
    k_singular: torch.Tensor,
    v_left: torch.Tensor,
    v_singular: torch.Tensor,
    *,
    rank: int,
    matrix_columns: int,
) -> KVParameterLeftSubspace:
    k_supported = min(
        rank,
        _numerical_rank_from_singular(
            k_singular,
            rows=k_left.shape[0],
            columns=matrix_columns,
        ),
    )
    v_supported = min(
        rank,
        _numerical_rank_from_singular(
            v_singular,
            rows=v_left.shape[0],
            columns=matrix_columns,
        ),
    )
    return KVParameterLeftSubspace(
        rank=rank,
        k_basis=k_left[:, :k_supported].detach(),
        v_basis=v_left[:, :v_supported].detach(),
        k_singular_values=k_singular.detach(),
        v_singular_values=v_singular.detach(),
    )


def _numerical_rank_from_singular(
    singular: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> int:
    if singular.numel() == 0 or float(singular[0]) == 0:
        return 0
    tolerance = max(rows, columns) * torch.finfo(singular.dtype).eps * float(singular[0])
    return int((singular > tolerance).sum())


@torch.inference_mode()
def parameter_subspace_oracle_projection(
    parent_k: torch.Tensor,
    parent_v: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    subspace: KVParameterLeftSubspace,
) -> ParameterSubspaceOracleProjection:
    """Project exact user delta onto a release-only K/V output subspace."""

    _, width = _validate_layer_kv(parent_k, parent_v, current_k, current_v)
    if (
        subspace.k_basis.ndim != 2
        or subspace.k_basis.shape[0] != width
        or subspace.k_basis.shape[1] > subspace.rank
    ):
        raise ValueError("K parameter basis and cache width differ")
    if (
        subspace.v_basis.ndim != 2
        or subspace.v_basis.shape[0] != width
        or subspace.v_basis.shape[1] > subspace.rank
    ):
        raise ValueError("V parameter basis and cache width differ")
    delta_k = current_k[0].float() - parent_k[0].float()
    delta_v = current_v[0].float() - parent_v[0].float()
    k_coefficients = delta_k @ subspace.k_basis
    v_coefficients = delta_v @ subspace.v_basis
    reconstructed_k = parent_k.float() + (
        k_coefficients @ subspace.k_basis.transpose(0, 1)
    ).unsqueeze(0)
    reconstructed_v = parent_v.float() + (
        v_coefficients @ subspace.v_basis.transpose(0, 1)
    ).unsqueeze(0)
    return ParameterSubspaceOracleProjection(
        rank=subspace.rank,
        k_coefficients=k_coefficients.detach(),
        v_coefficients=v_coefficients.detach(),
        reconstructed_k=reconstructed_k.detach(),
        reconstructed_v=reconstructed_v.detach(),
        subspace=subspace,
    )


@torch.inference_mode()
def parameter_subspace_oracle_grid(
    parent_model,
    current_model,
    parent_cache: HSTUKVCache,
    current_cache: HSTUKVCache,
    *,
    ranks: Iterable[int] = PARAMETER_SUBSPACE_RANKS,
) -> dict[tuple[int, int], ParameterSubspaceOracleProjection]:
    """Project every exact layer delta on release-derived K/V bases."""

    layers, _, _ = _validate_cache_pair(parent_cache, current_cache)
    if len(parent_model.blocks) != layers or len(current_model.blocks) != layers:
        raise ValueError("model and cache layer counts differ")
    ranks = tuple(int(rank) for rank in ranks)
    if len(set(ranks)) != len(ranks):
        raise ValueError("parameter-subspace ranks must be unique")
    result: dict[tuple[int, int], ParameterSubspaceOracleProjection] = {}
    for layer in range(layers):
        parent_attention = parent_model.blocks[layer].attn
        current_attention = current_model.blocks[layer].attn
        parent_k_weight = _projection_weight(parent_attention.k_proj)
        current_k_weight = _projection_weight(current_attention.k_proj)
        parent_v_weight = _projection_weight(parent_attention.v_proj)
        current_v_weight = _projection_weight(current_attention.v_proj)
        if (
            parent_k_weight.shape != current_k_weight.shape
            or parent_v_weight.shape != current_v_weight.shape
            or parent_k_weight.shape != parent_v_weight.shape
        ):
            raise ValueError("Parent and Current K/V projection shapes differ")
        factors = _kv_parameter_left_svd(
            parent_k_weight,
            current_k_weight,
            parent_v_weight,
            current_v_weight,
        )
        for rank in ranks:
            _validate_rank(rank, *parent_k_weight.shape)
            subspace = _truncate_kv_parameter_subspace(
                *factors,
                rank=rank,
                matrix_columns=parent_k_weight.shape[1],
            )
            result[(layer, rank)] = parameter_subspace_oracle_projection(
                parent_cache.k[layer],
                parent_cache.v[layer],
                current_cache.k[layer],
                current_cache.v[layer],
                subspace,
            )
    return result


def reconstruct_cache_from_parameter_subspace(
    parent_cache: HSTUKVCache,
    projections: dict[tuple[int, int], ParameterSubspaceOracleProjection],
    *,
    rank: int,
) -> HSTUKVCache:
    """Stack one parameter-subspace rank's layer reconstructions."""

    layers, length, _ = _validate_cache_pair(parent_cache, parent_cache)
    selected = [projections[(layer, rank)] for layer in range(layers)]
    return HSTUKVCache(
        k=torch.stack([projection.reconstructed_k for projection in selected], dim=0),
        v=torch.stack([projection.reconstructed_v for projection in selected], dim=0),
        seq_len=length,
    )


def _delta_reconstruction_metrics(
    exact_delta: torch.Tensor,
    observed_delta: torch.Tensor,
    *,
    eps: float = 1e-20,
) -> dict[str, float]:
    exact = exact_delta.float().reshape(-1)
    observed = observed_delta.float().reshape(-1)
    exact_norm = torch.linalg.vector_norm(exact)
    error_norm = torch.linalg.vector_norm(exact - observed)
    if float(exact_norm) <= eps:
        recovery = 1.0 if float(error_norm) <= eps else float("-inf")
        energy = 1.0 if float(error_norm) <= eps else float("-inf")
    else:
        recovery = float(1.0 - error_norm / exact_norm)
        energy = float(1.0 - error_norm.square() / exact_norm.square())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            exact.unsqueeze(0), observed.unsqueeze(0), dim=1, eps=eps
        )[0]
    )
    return {
        "relative_l2_recovery": recovery,
        "captured_delta_energy": energy,
        "cosine_to_exact_delta": cosine,
        "exact_delta_l2": float(exact_norm),
        "residual_delta_l2": float(error_norm),
    }


def _token_participation(
    coefficients: torch.Tensor,
    *,
    fraction: float = 0.90,
) -> dict[str, float]:
    if coefficients.ndim != 2 or not 0 < fraction <= 1:
        raise ValueError("coefficients/fraction differ")
    energy = coefficients.float().square().sum(dim=1)
    total = energy.sum()
    if float(total) == 0:
        return {
            "normalized_token_participation_ratio": 0.0,
            "fraction_tokens_covering_90pct_energy": 0.0,
            "nonzero_token_fraction": 0.0,
        }
    participation = total.square() / energy.square().sum().clamp_min(1e-20)
    sorted_energy = torch.sort(energy, descending=True).values
    covering = int(
        torch.searchsorted(
            torch.cumsum(sorted_energy, dim=0),
            fraction * total,
            right=False,
        ).item()
        + 1
    )
    return {
        "normalized_token_participation_ratio": float(participation / coefficients.shape[0]),
        "fraction_tokens_covering_90pct_energy": (covering / coefficients.shape[0]),
        "nonzero_token_fraction": float((energy > 0).float().mean()),
    }


def _prefixed_participation(
    values: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in _token_participation(values).items()}


def token_svd_oracle_metrics(
    parent_k: torch.Tensor,
    parent_v: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    oracle: JointDeltaSVDOracle,
) -> dict[str, float | int | bool | str]:
    """Quantify reconstruction and dense token participation for one oracle."""

    length, width = _validate_layer_kv(parent_k, parent_v, current_k, current_v)
    exact = joint_kv_delta_matrix(parent_k, parent_v, current_k, current_v)
    observed = torch.cat(
        (
            (oracle.reconstructed_k - parent_k.float())[0],
            (oracle.reconstructed_v - parent_v.float())[0],
        ),
        dim=1,
    )
    return {
        "rank": oracle.rank,
        "source_length": length,
        "width": width,
        "stored_scalars": oracle.stored_scalars,
        "stored_ratio_to_full_layer_KV": oracle.stored_scalars / (2 * length * width),
        "diagnostic_full_singular_value_scalars_not_in_factor_storage": (
            oracle.singular_values.numel()
        ),
        "diagnostic_history_basis_scalars_not_in_factor_storage": (oracle.history_basis.numel()),
        "oracle_coefficients_use_Current_Exact": True,
        "design_admissible": False,
        "evidence_class": "per_user_exact_delta_truncated_SVD_oracle",
        **_delta_reconstruction_metrics(exact, observed),
        **_prefixed_participation(exact, prefix="exact_delta"),
        **_prefixed_participation(
            oracle.coefficients,
            prefix="captured_component",
        ),
    }


def parameter_subspace_oracle_metrics(
    parent_k: torch.Tensor,
    parent_v: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    projection: ParameterSubspaceOracleProjection,
) -> dict[str, float | int | bool | str]:
    """Quantify a release-derived basis with Exact per-user coefficients."""

    length, width = _validate_layer_kv(parent_k, parent_v, current_k, current_v)
    exact = joint_kv_delta_matrix(parent_k, parent_v, current_k, current_v)
    observed = torch.cat(
        (
            (projection.reconstructed_k - parent_k.float())[0],
            (projection.reconstructed_v - parent_v.float())[0],
        ),
        dim=1,
    )
    joined_coefficients = torch.cat((projection.k_coefficients, projection.v_coefficients), dim=1)
    return {
        "rank_per_KV": projection.rank,
        "K_realized_release_supported_rank": (projection.subspace.k_realized_rank),
        "V_realized_release_supported_rank": (projection.subspace.v_realized_rank),
        "source_length": length,
        "width": width,
        "per_user_coefficient_scalars": projection.per_user_stored_scalars,
        "per_user_coefficient_ratio_to_full_layer_KV": (
            projection.per_user_stored_scalars / (2 * length * width)
        ),
        "release_shared_basis_scalars": projection.subspace.shared_basis_scalars,
        "diagnostic_full_parameter_singular_value_scalars_not_in_basis_storage": (
            projection.subspace.k_singular_values.numel()
            + projection.subspace.v_singular_values.numel()
        ),
        "basis_uses_only_release_parameter_delta": True,
        "oracle_coefficients_use_Current_Exact": True,
        "design_admissible": False,
        "evidence_class": "release_parameter_subspace_oracle_ceiling",
        **_delta_reconstruction_metrics(exact, observed),
        **_prefixed_participation(exact, prefix="exact_delta"),
        **_prefixed_participation(
            joined_coefficients,
            prefix="projected_component",
        ),
    }


def matrix_delta_spectrum(
    parent_weight: torch.Tensor,
    current_weight: torch.Tensor,
    *,
    ranks: Iterable[int] = PARAMETER_SUBSPACE_RANKS,
) -> dict[str, Any]:
    """Return singular-spectrum statistics for one release parameter delta."""

    if parent_weight.ndim != 2 or parent_weight.shape != current_weight.shape:
        raise ValueError("Parent and Current weights must share a matrix shape")
    delta = current_weight.detach().float() - parent_weight.detach().float()
    singular = torch.linalg.svdvals(delta)
    energy = singular.square()
    total = energy.sum()
    if float(total) == 0:
        normalized = torch.zeros_like(energy)
        effective_rank = 0.0
        stable_rank = 0.0
    else:
        normalized = energy / total
        nonzero = normalized[normalized > 0]
        effective_rank = float(torch.exp(-(nonzero * torch.log(nonzero)).sum()))
        stable_rank = float(total / energy[0].clamp_min(1e-20))
    cumulative = torch.cumsum(normalized, dim=0)

    def rank_for(threshold: float) -> int:
        if float(total) == 0:
            return 0
        return int(torch.searchsorted(cumulative, threshold).item() + 1)

    grid = {}
    for rank in ranks:
        rank = int(rank)
        _validate_rank(rank, *delta.shape)
        grid[str(rank)] = float(cumulative[rank - 1])
    numerical_rank = _numerical_rank_from_singular(
        singular,
        rows=delta.shape[0],
        columns=delta.shape[1],
    )
    return {
        "rows": int(delta.shape[0]),
        "columns": int(delta.shape[1]),
        "frobenius_norm": float(torch.linalg.vector_norm(delta)),
        "spectral_norm": float(singular[0]) if singular.numel() else 0.0,
        "stable_rank": stable_rank,
        "entropy_effective_rank": effective_rank,
        "numerical_rank": numerical_rank,
        "rank_for_90pct_energy": rank_for(0.90),
        "rank_for_95pct_energy": rank_for(0.95),
        "rank_for_99pct_energy": rank_for(0.99),
        "captured_energy_by_rank": grid,
        "singular_values": [float(value) for value in singular],
    }


def kv_projection_parameter_delta_spectra(
    parent_attention,
    current_attention,
    *,
    ranks: Iterable[int] = PARAMETER_SUBSPACE_RANKS,
) -> dict[str, dict[str, Any]]:
    """Report separate and joint K/V release-weight delta spectra."""

    parent_k = _projection_weight(parent_attention.k_proj)
    current_k = _projection_weight(current_attention.k_proj)
    parent_v = _projection_weight(parent_attention.v_proj)
    current_v = _projection_weight(current_attention.v_proj)
    return {
        "K": matrix_delta_spectrum(parent_k, current_k, ranks=ranks),
        "V": matrix_delta_spectrum(parent_v, current_v, ranks=ranks),
        "joint_stacked_KV": matrix_delta_spectrum(
            torch.cat((parent_k, parent_v), dim=0),
            torch.cat((current_k, current_v), dim=0),
            ranks=ranks,
        ),
    }


def model_kv_projection_parameter_delta_spectra(
    parent_model,
    current_model,
    *,
    ranks: Iterable[int] = PARAMETER_SUBSPACE_RANKS,
) -> list[dict[str, Any]]:
    """Report the preregistered K/V release spectra for every model layer."""

    if len(parent_model.blocks) != len(current_model.blocks):
        raise ValueError("Parent and Current model layer counts differ")
    ranks = tuple(int(rank) for rank in ranks)
    return [
        {
            "layer": layer,
            **kv_projection_parameter_delta_spectra(
                parent_model.blocks[layer].attn,
                current_model.blocks[layer].attn,
                ranks=ranks,
            ),
        }
        for layer in range(len(parent_model.blocks))
    ]


def _classical_thin_svd_flop_estimate(rows: int, columns: int) -> int:
    """Return a symmetric classical thin-SVD audit estimate.

    Backend algorithms differ, so this estimate is never presented as measured
    runtime or as an executable Design cost.  All exact multiply/add counts are
    reported separately by :func:`release_differential_oracle_cost`.
    """

    long_side = max(rows, columns)
    short_side = min(rows, columns)
    return int(4 * long_side * short_side**2 + (8 / 3) * short_side**3)


def release_differential_oracle_cost(
    *,
    layers: int,
    hidden: int,
    context: int,
    token_svd_rank: int,
    parameter_rank: int,
    heads: int = 6,
    temporal_freqs: int = 16,
) -> dict[str, int | float | bool | str]:
    """Strict storage/FLOP semantics for both Current-Exact oracle families."""

    if (
        min(
            layers,
            hidden,
            context,
            token_svd_rank,
            parameter_rank,
            heads,
            temporal_freqs,
        )
        < 1
    ):
        raise ValueError("architecture and ranks must be positive")
    if hidden % heads:
        raise ValueError("hidden size must be divisible by heads")
    _validate_rank(token_svd_rank, context, 2 * hidden)
    _validate_rank(parameter_rank, hidden, hidden)

    def input_projection(tokens: int) -> int:
        return 2 * tokens * (2 * temporal_freqs) * hidden + 2 * tokens * hidden**2

    def block_linear(tokens: int) -> int:
        return 2 * tokens * (5 * hidden**2)

    causal_pairs = context * (context + 1) // 2
    exact_all = input_projection(context) + layers * (
        block_linear(context) + 4 * causal_pairs * hidden
    )
    joint_width = 2 * hidden
    # One user-specific full SVD per layer is shared across the token rank grid.
    # This is a classical estimate; torch.linalg.svd backend work may differ.
    token_full_svd_per_layer = _classical_thin_svd_flop_estimate(
        context,
        joint_width,
    )
    token_delta_subtraction = 2 * context * hidden
    token_coefficient_scaling = context * token_svd_rank
    token_reconstruction = 4 * context * token_svd_rank * hidden
    token_parent_add = 2 * context * hidden
    token_oracle_aux = layers * (
        token_delta_subtraction
        + token_full_svd_per_layer
        + token_coefficient_scaling
        + token_reconstruction
        + token_parent_add
    )

    # Two exact delta-to-basis coefficient products, two reconstructions, and
    # Parent addition. The release-only basis SVD is separately amortized.
    parameter_delta_subtraction = 2 * context * hidden
    parameter_coefficient_and_reconstruction = 8 * context * hidden * parameter_rank
    parameter_parent_add = 2 * context * hidden
    parameter_oracle_aux = layers * (
        parameter_delta_subtraction
        + parameter_coefficient_and_reconstruction
        + parameter_parent_add
    )
    one_weight_svd = _classical_thin_svd_flop_estimate(hidden, hidden)
    release_weight_delta_subtraction = layers * 2 * hidden**2
    release_basis_svd = layers * 2 * one_weight_svd
    release_basis_construction = release_weight_delta_subtraction + release_basis_svd

    full_kv_scalars = 2 * layers * context * hidden
    token_user_scalars = layers * token_svd_rank * (context + 2 * hidden)
    parameter_user_scalars = 2 * layers * context * parameter_rank
    parameter_shared_scalars = 2 * layers * hidden * parameter_rank
    return {
        "layers": layers,
        "hidden": hidden,
        "context": context,
        "token_svd_rank": token_svd_rank,
        "parameter_rank_per_KV": parameter_rank,
        "full_Exact_All_flops_per_user": exact_all,
        "token_SVD_oracle_auxiliary_flops_per_user": token_oracle_aux,
        "token_SVD_classical_fit_flops_estimate_per_layer": (token_full_svd_per_layer),
        "token_SVD_coefficient_scaling_flops_per_user": (layers * token_coefficient_scaling),
        "token_SVD_oracle_total_including_Current_Exact": exact_all + token_oracle_aux,
        "token_SVD_oracle_total_over_Exact_All": (exact_all + token_oracle_aux) / exact_all,
        "parameter_subspace_oracle_auxiliary_flops_per_user": parameter_oracle_aux,
        "parameter_subspace_oracle_total_including_Current_Exact": (
            exact_all + parameter_oracle_aux
        ),
        "parameter_subspace_oracle_total_over_Exact_All": (exact_all + parameter_oracle_aux)
        / exact_all,
        "release_shared_KV_weight_delta_subtraction_flops": (release_weight_delta_subtraction),
        "release_shared_KV_weight_basis_SVD_flops_estimate": release_basis_svd,
        "release_shared_KV_basis_construction_flops_unamortized": (release_basis_construction),
        "parameter_subspace_oracle_total_one_user_unamortized": (
            exact_all + parameter_oracle_aux + release_basis_construction
        ),
        "parameter_subspace_oracle_total_one_user_unamortized_over_Exact_All": (
            (exact_all + parameter_oracle_aux + release_basis_construction) / exact_all
        ),
        "full_Current_KV_scalars": full_kv_scalars,
        "token_SVD_per_user_factor_scalars": token_user_scalars,
        "token_SVD_per_user_factor_ratio_to_full_KV": token_user_scalars / full_kv_scalars,
        "parameter_subspace_per_user_coefficient_scalars": parameter_user_scalars,
        "parameter_subspace_per_user_coefficient_ratio_to_full_KV": (
            parameter_user_scalars / full_kv_scalars
        ),
        "parameter_subspace_release_shared_basis_scalars": parameter_shared_scalars,
        "Current_Exact_required_for_all_user_coefficients": True,
        "Current_Exact_KV_materialization_required": True,
        "token_SVD_basis_is_per_user_Current_Exact_derived": True,
        "parameter_subspace_basis_is_release_shared_and_parameter_only": True,
        "parameter_subspace_per_user_total_excludes_release_shared_basis_fit": True,
        "rank_grid_reuses_one_full_SVD_per_matrix": True,
        "cost_model_requires_cache_width_equal_hidden": True,
        "within_20_percent_design_budget": False,
        "design_admissible": False,
        "SVD_flop_semantics": (
            "classical_thin_SVD_estimate_4mn2_plus_8over3n3_backend_actual_varies"
        ),
        "storage_semantics": (
            "Parent_KV_is_existing_base_factors_or_coefficients_are_incremental_"
            "but_oracle_Current_Exact_is_not_an_executable_constructor"
        ),
        "future_candidate_semantics": (
            "propagate_layer0_seeded_release_differential_along_parameter_path_"
            "not_low_rank_compression"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-medium-cost",
        action="store_true",
        help="print static Medium oracle cost/storage grids",
    )
    args = parser.parse_args()
    if not args.print_medium_cost:
        parser.error("tensor diagnostics are imported; use --print-medium-cost")
    payload = {
        f"tokenR{token_rank}_parameterR{parameter_rank}": (
            release_differential_oracle_cost(
                layers=6,
                hidden=192,
                context=1024,
                token_svd_rank=token_rank,
                parameter_rank=parameter_rank,
            )
        )
        for token_rank in TOKEN_SVD_RANKS
        for parameter_rank in PARAMETER_SUBSPACE_RANKS
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
