import pytest
import torch
from pathlib import Path
import sys
from experiments.ch3_motivation.code.sketch_svd_gram import wide_svd


@pytest.mark.parametrize('rank', [0, 4, 16])
def test_projected_svd_reconstruction_and_distinct_leading_ranks(rank):
    generator = torch.Generator().manual_seed(17)
    a = torch.randn(2, 16, 16, generator=generator, dtype=torch.float64)
    b = torch.randn(2, 96, 16, generator=generator, dtype=torch.float64)
    u = torch.linalg.qr(a).Q
    v = torch.linalg.qr(b).Q
    spectrum = torch.zeros(16, dtype=torch.float64)
    if rank:
        spectrum[:rank] = torch.logspace(0, -3, rank, dtype=torch.float64)
    matrix = ((u * spectrum) @ v.transpose(-1, -2)).float()
    lu, ls, lv = wide_svd(matrix)
    ru, rs, rv = torch.linalg.svd(matrix, full_matrices=False)
    assert torch.isfinite(ls).all() and torch.isfinite(lv).all()
    torch.testing.assert_close((lu * ls[..., None, :]) @ lv, matrix, atol=1e-6, rtol=1e-5)
    for count in (2, 6, 16):
        # A cutoff through a null space has zero reconstruction contribution.
        left = (lu[..., :count] * ls[..., None, :count]) @ lv[..., :count, :]
        right = (ru[..., :count] * rs[..., None, :count]) @ rv[..., :count, :]
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-4)
