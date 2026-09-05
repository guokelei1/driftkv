from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from insight_two.release_differential_structure import (  # noqa: E402
    joint_delta_token_svd_grid,
    joint_delta_token_svd_oracle,
    kv_parameter_left_subspace,
    matrix_delta_spectrum,
    model_kv_projection_parameter_delta_spectra,
    parameter_subspace_oracle_grid,
    parameter_subspace_oracle_metrics,
    parameter_subspace_oracle_projection,
    reconstruct_cache_from_parameter_subspace,
    reconstruct_cache_from_token_svd,
    release_differential_oracle_cost,
    token_svd_oracle_metrics,
)

from hstu_kvcache.models import HSTUKVCache  # noqa: E402


def _cache(k: torch.Tensor, v: torch.Tensor) -> HSTUKVCache:
    return HSTUKVCache(k=k, v=v, seq_len=k.shape[2])


def _attention(width: int) -> SimpleNamespace:
    return SimpleNamespace(
        k_proj=torch.nn.Linear(width, width, bias=False),
        v_proj=torch.nn.Linear(width, width, bias=False),
    )


def _model(*attentions: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(blocks=[SimpleNamespace(attn=attention) for attention in attentions])


def test_joint_token_svd_reconstructs_exact_low_rank_dense_support() -> None:
    torch.manual_seed(1101)
    length, width, rank = 8, 5, 2
    parent_k = torch.randn(1, length, width)
    parent_v = torch.randn(1, length, width)
    angles = torch.arange(length).float() * (2 * torch.pi / length)
    coefficients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    basis, _ = torch.linalg.qr(torch.randn(2 * width, rank))
    delta = coefficients @ basis.transpose(0, 1)
    delta_k, delta_v = delta.split(width, dim=1)
    current_k = parent_k + delta_k.unsqueeze(0)
    current_v = parent_v + delta_v.unsqueeze(0)

    oracle = joint_delta_token_svd_oracle(
        parent_k,
        parent_v,
        current_k,
        current_v,
        rank=rank,
    )
    metrics = token_svd_oracle_metrics(
        parent_k,
        parent_v,
        current_k,
        current_v,
        oracle,
    )

    assert torch.allclose(oracle.reconstructed_k, current_k, atol=2e-5, rtol=2e-5)
    assert torch.allclose(oracle.reconstructed_v, current_v, atol=2e-5, rtol=2e-5)
    assert oracle.history_basis.shape == (length, rank)
    assert torch.allclose(
        oracle.history_basis.transpose(0, 1) @ oracle.history_basis,
        torch.eye(rank),
        atol=2e-5,
        rtol=2e-5,
    )
    assert metrics["captured_delta_energy"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["exact_delta_normalized_token_participation_ratio"] == pytest.approx(
        1.0,
        abs=1e-6,
    )
    assert metrics["exact_delta_nonzero_token_fraction"] == 1.0
    assert metrics["oracle_coefficients_use_Current_Exact"] is True
    assert metrics["design_admissible"] is False


def test_token_svd_grid_reuses_one_factorization_per_layer_and_stacks_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1111)
    layers, length, width = 2, 6, 4
    parent = _cache(
        torch.randn(layers, 1, length, width),
        torch.randn(layers, 1, length, width),
    )
    current = _cache(
        parent.k + torch.randn_like(parent.k),
        parent.v + torch.randn_like(parent.v),
    )
    original_svd = torch.linalg.svd
    calls = 0

    def counted_svd(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counted_svd)
    grid = joint_delta_token_svd_grid(parent, current, ranks=(1, 2, 6))
    reconstructed = reconstruct_cache_from_token_svd(parent, grid, rank=6)

    assert calls == layers
    assert reconstructed.seq_len == length
    assert torch.allclose(reconstructed.k, current.k, atol=2e-5, rtol=2e-5)
    assert torch.allclose(reconstructed.v, current.v, atol=2e-5, rtol=2e-5)


def test_parameter_left_subspace_recovers_only_release_supported_directions() -> None:
    torch.manual_seed(1123)
    length, width = 7, 6
    parent_attention = _attention(width)
    current_attention = _attention(width)
    with torch.no_grad():
        parent_attention.k_proj.weight.zero_()
        parent_attention.v_proj.weight.zero_()
        current_attention.k_proj.weight.zero_()
        current_attention.v_proj.weight.zero_()
        current_attention.k_proj.weight[0, 0] = 4.0
        current_attention.v_proj.weight[2, 1] = 3.0

    subspace = kv_parameter_left_subspace(
        parent_attention,
        current_attention,
        rank=4,
    )
    assert subspace.k_realized_rank == 1
    assert subspace.v_realized_rank == 1

    parent_k = torch.zeros(1, length, width)
    parent_v = torch.zeros(1, length, width)
    current_k = torch.zeros_like(parent_k)
    current_v = torch.zeros_like(parent_v)
    # K is deliberately orthogonal to the only release-supported K direction;
    # an arbitrary null-space completion at requested rank four would leak it.
    current_k[0, :, 1] = torch.linspace(1.0, 2.0, length)
    current_v[0, :, 2] = torch.linspace(-2.0, 1.0, length)

    projection = parameter_subspace_oracle_projection(
        parent_k,
        parent_v,
        current_k,
        current_v,
        subspace,
    )
    metrics = parameter_subspace_oracle_metrics(
        parent_k,
        parent_v,
        current_k,
        current_v,
        projection,
    )

    assert torch.count_nonzero(projection.reconstructed_k) == 0
    assert torch.allclose(projection.reconstructed_v, current_v)
    assert metrics["K_realized_release_supported_rank"] == 1
    assert metrics["V_realized_release_supported_rank"] == 1
    assert metrics["captured_delta_energy"] < 1.0
    assert metrics["basis_uses_only_release_parameter_delta"] is True
    assert metrics["oracle_coefficients_use_Current_Exact"] is True
    assert metrics["design_admissible"] is False


def test_parameter_subspace_uses_left_output_directions_for_rectangular_weights() -> None:
    torch.manual_seed(1147)
    length, input_width, cache_width, rank = 9, 3, 5, 2
    parent_attention = SimpleNamespace(
        k_proj=torch.nn.Linear(input_width, cache_width, bias=False),
        v_proj=torch.nn.Linear(input_width, cache_width, bias=False),
    )
    current_attention = SimpleNamespace(
        k_proj=torch.nn.Linear(input_width, cache_width, bias=False),
        v_proj=torch.nn.Linear(input_width, cache_width, bias=False),
    )
    k_left, _ = torch.linalg.qr(torch.randn(cache_width, rank))
    v_left, _ = torch.linalg.qr(torch.randn(cache_width, rank))
    k_weight_delta = k_left @ torch.randn(rank, input_width)
    v_weight_delta = v_left @ torch.randn(rank, input_width)
    with torch.no_grad():
        parent_attention.k_proj.weight.zero_()
        parent_attention.v_proj.weight.zero_()
        current_attention.k_proj.weight.copy_(k_weight_delta)
        current_attention.v_proj.weight.copy_(v_weight_delta)

    history = torch.randn(length, input_width)
    parent_k = torch.zeros(1, length, cache_width)
    parent_v = torch.zeros(1, length, cache_width)
    current_k = (history @ k_weight_delta.transpose(0, 1)).unsqueeze(0)
    current_v = (history @ v_weight_delta.transpose(0, 1)).unsqueeze(0)
    subspace = kv_parameter_left_subspace(
        parent_attention,
        current_attention,
        rank=rank,
    )
    projection = parameter_subspace_oracle_projection(
        parent_k,
        parent_v,
        current_k,
        current_v,
        subspace,
    )

    assert subspace.k_basis.shape == (cache_width, rank)
    assert subspace.v_basis.shape == (cache_width, rank)
    assert torch.allclose(projection.reconstructed_k, current_k, atol=2e-5, rtol=2e-5)
    assert torch.allclose(projection.reconstructed_v, current_v, atol=2e-5, rtol=2e-5)


def test_parameter_grid_reuses_release_svd_and_reconstructs_supported_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1151)
    layers, length, width = 2, 5, 4
    parent_attentions = [_attention(width) for _ in range(layers)]
    current_attentions = [_attention(width) for _ in range(layers)]
    parent_k = torch.randn(layers, 1, length, width)
    parent_v = torch.randn(layers, 1, length, width)
    current_k = parent_k.clone()
    current_v = parent_v.clone()
    with torch.no_grad():
        for layer, (parent_attention, current_attention) in enumerate(
            zip(parent_attentions, current_attentions, strict=True)
        ):
            parent_attention.k_proj.weight.zero_()
            parent_attention.v_proj.weight.zero_()
            current_attention.k_proj.weight.copy_(torch.eye(width))
            current_attention.v_proj.weight.copy_(torch.eye(width))
            current_k[layer] += (layer + 1) * torch.randn(1, length, width)
            current_v[layer] += (layer + 1) * torch.randn(1, length, width)

    original_svd = torch.linalg.svd
    calls = 0

    def counted_svd(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counted_svd)
    grid = parameter_subspace_oracle_grid(
        _model(*parent_attentions),
        _model(*current_attentions),
        _cache(parent_k, parent_v),
        _cache(current_k, current_v),
        ranks=(1, 2, 4),
    )
    reconstructed = reconstruct_cache_from_parameter_subspace(
        _cache(parent_k, parent_v),
        grid,
        rank=4,
    )

    assert calls == 2 * layers
    assert torch.allclose(reconstructed.k, current_k, atol=2e-5, rtol=2e-5)
    assert torch.allclose(reconstructed.v, current_v, atol=2e-5, rtol=2e-5)


def test_parameter_delta_spectrum_reports_known_energy_and_all_layers() -> None:
    parent = torch.zeros(4, 4)
    current = torch.diag(torch.tensor([4.0, 3.0, 0.0, 0.0]))
    spectrum = matrix_delta_spectrum(parent, current, ranks=(1, 2, 4))
    assert spectrum["numerical_rank"] == 2
    assert spectrum["stable_rank"] == pytest.approx(25 / 16)
    assert spectrum["rank_for_90pct_energy"] == 2
    assert spectrum["captured_energy_by_rank"]["1"] == pytest.approx(16 / 25)
    assert spectrum["captured_energy_by_rank"]["2"] == 1.0

    parent_attention = _attention(4)
    current_attention = _attention(4)
    with torch.no_grad():
        parent_attention.k_proj.weight.zero_()
        parent_attention.v_proj.weight.zero_()
        current_attention.k_proj.weight.copy_(current)
        current_attention.v_proj.weight.copy_(current)
    layers = model_kv_projection_parameter_delta_spectra(
        _model(parent_attention),
        _model(current_attention),
        ranks=(1, 2, 4),
    )
    assert len(layers) == 1
    assert layers[0]["layer"] == 0
    assert layers[0]["K"]["numerical_rank"] == 2
    assert layers[0]["joint_stacked_KV"]["rank_for_99pct_energy"] == 2


def test_medium_cost_cannot_be_misread_as_an_executable_budget_point() -> None:
    cost = release_differential_oracle_cost(
        layers=6,
        hidden=192,
        context=1024,
        token_svd_rank=16,
        parameter_rank=16,
    )
    assert cost["full_Exact_All_flops_per_user"] == 4_771_282_944
    assert cost["token_SVD_oracle_total_over_Exact_All"] > 1.0
    assert cost["parameter_subspace_oracle_total_over_Exact_All"] > 1.0
    assert (
        cost["parameter_subspace_oracle_total_one_user_unamortized_over_Exact_All"]
        > cost["parameter_subspace_oracle_total_over_Exact_All"]
    )
    assert cost["full_Current_KV_scalars"] == 2_359_296
    assert cost["Current_Exact_required_for_all_user_coefficients"] is True
    assert cost["within_20_percent_design_budget"] is False
    assert cost["design_admissible"] is False
    assert cost["future_candidate_semantics"] == (
        "propagate_layer0_seeded_release_differential_along_parameter_path_not_low_rank_compression"
    )
