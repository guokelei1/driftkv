"""Small numerical oracles for the proposed per-matrix randomized-SVD guard."""
from pathlib import Path
import sys

import pytest
import torch

from experiments.ch3_motivation.code.energy_certificate import sketch_energy_certificate


def matrix_with_spectrum(spectrum, *, dtype=torch.float64, scale=1.):
    generator = torch.Generator().manual_seed(41)
    left = torch.linalg.qr(torch.randn(72, len(spectrum), generator=generator, dtype=torch.float64)).Q
    right = torch.linalg.qr(torch.randn(48, len(spectrum), generator=generator, dtype=torch.float64)).Q
    return ((left * torch.tensor(spectrum, dtype=torch.float64)) @ right.T * scale).to(dtype)


def sketch(matrix, *, width=32, iterations=2):
    generator = torch.Generator().manual_seed(19)
    omega = torch.randn(matrix.shape[-1], width, generator=generator, dtype=matrix.dtype)
    q = torch.linalg.qr(matrix @ omega, mode="reduced").Q
    for _ in range(iterations):
        z = torch.linalg.qr(matrix.T @ q, mode="reduced").Q
        q = torch.linalg.qr(matrix @ z, mode="reduced").Q
    return q, torch.linalg.svdvals(q.T @ matrix)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1e-8, 1., 1e8])
def test_structured_certificate_never_exceeds_exact_accuracy(dtype, scale):
    matrix = matrix_with_spectrum([10. / (i + 1) for i in range(12)] + [1e-7] * 36,
                                  dtype=dtype, scale=scale)
    q, singular = sketch(matrix)
    represented = matrix.double()
    total = represented.square().sum()
    delta = torch.linalg.matrix_norm(q.double().T @ q.double() - torch.eye(q.shape[1], dtype=torch.float64), ord="fro")
    result = sketch_energy_certificate(total, singular, orthogonality_defect=delta + 1e-12)
    exact = torch.linalg.svdvals(represented)
    actual = torch.stack([singular[:rank].double().square().sum() / exact[:rank].square().sum()
                          for rank in result["ranks"]])
    assert torch.all(result["captured_energy_ratio_lower_bound"] <= actual)
    assert result["passes"].all()


def test_flat_spectrum_is_retained_as_conservative_failure():
    matrix = torch.eye(48, dtype=torch.float64)
    q, singular = sketch(matrix)
    result = sketch_energy_certificate(matrix.square().sum(), singular, relative_energy_tolerance=0.)
    expected = torch.tensor([rank / (rank + 16) for rank in (2, 6, 16)], dtype=torch.float64)
    torch.testing.assert_close(result["captured_energy_ratio_lower_bound"], expected, rtol=1e-12, atol=1e-12)
    assert not result["passes"].any()  # Exact ratio is one despite an uninformative range bound.
    assert result["needs_refinement"]


def test_zero_matrix_and_cancellation_slack():
    result = sketch_energy_certificate(torch.tensor([0., 1. - 5e-8], dtype=torch.float64),
        torch.tensor([[0., 0.], [1., 0.]], dtype=torch.float64))
    torch.testing.assert_close(result["captured_energy_ratio_lower_bound"][0], torch.ones(3, dtype=torch.float64))
    assert result["exact_zero_matrix"][0] and not result["needs_refinement"][0]
    assert result["residual_range_energy_estimate"][1] == 0
    assert result["residual_range_energy_upper_bound"][1] > 0
    assert torch.all(result["captured_energy_ratio_lower_bound"][1] < 1)


def test_full_small_spectrum_is_used_before_rank16_truncation():
    matrix = matrix_with_spectrum([1.] * 24 + [0.] * 24)
    _, singular = sketch(matrix)
    full = sketch_energy_certificate(matrix.square().sum(), singular)
    truncated = sketch_energy_certificate(matrix.square().sum(), singular[:16])
    assert full["passes"].all()
    assert not truncated["passes"].any()
    assert full["range_energy"] > truncated["range_energy"]


def test_nonorthogonal_q_correction_is_below_exact_reported_ratio():
    matrix = matrix_with_spectrum([4. / (i + 1) for i in range(48)])
    q, _ = sketch(matrix)
    q = q * torch.linspace(.98, 1.02, q.shape[-1], dtype=torch.float64)
    singular = torch.linalg.svdvals(q.T @ matrix)
    delta = torch.linalg.matrix_norm(q.T @ q - torch.eye(q.shape[1], dtype=torch.float64), ord="fro")
    result = sketch_energy_certificate(matrix.square().sum(), singular,
        orthogonality_defect=delta, relative_energy_tolerance=1e-12)
    exact = torch.linalg.svdvals(matrix)
    actual = torch.stack([singular[:rank].square().sum() / exact[:rank].square().sum() for rank in (2, 6, 16)])
    assert torch.all(result["captured_energy_ratio_lower_bound"] <= actual)


def test_invalid_energy_or_basis_cannot_pass():
    result = sketch_energy_certificate(torch.tensor([1., -1., 0., 1.]),
        torch.tensor([[1., 0.], [1., 0.], [float("nan"), 0.], [1., 0.]]),
        orthogonality_defect=torch.tensor([1., 0., 0., 0.]))
    assert result["needs_refinement"][:3].all()
    assert not result["passes"][:3].any()
    assert result["passes"][3].all()
