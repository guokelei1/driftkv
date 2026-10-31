"""Numerical checks of E4 definitions and E6 compression, not quality evidence."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.analysis import (
    MismatchAccumulator, decompose_interaction, decompose_layer_heads, mismatch_rows, summarize_mismatch,
)
from hstu_kvcache.models import HSTUKVCache


def cache(k, v=None):
    return HSTUKVCache(k, torch.zeros_like(k) if v is None else v, k.shape[2])


def test_mismatch_sums_heads_and_layers_and_excludes_padding():
    # Event 0 has 8 units, event 1 has 2 units. Padding must not dominate.
    old = torch.zeros(2, 1, 3, 2, dtype=torch.float64)
    new = old.clone()
    new[0, 0, 0] = torch.tensor([2., 0.])
    new[1, 0, 0] = torch.tensor([0., 2.])
    new[0, 0, 1] = torch.tensor([1., 1.])
    new[:, :, 2] = 1000.
    row = mismatch_rows(cache(old), cache(new), uids=[71], lengths=[2])[0]
    assert row["uid"] == 71 and row["history_rows"] == 2
    assert row["selected_rows"] == 1
    assert row["absolute_total_deviation"] == 10
    assert row["absolute_top10_row_share"] == .8
    np.testing.assert_allclose(row["absolute_layer_shares"], [.6, .4])
    np.testing.assert_allclose(row["absolute_layer_evenness"], [1.2, .8])
    assert row["relative_total_deviation"] == 3.
    assert row["relative_top10_row_share"] == pytest.approx(2 / 3)


def test_relative_mismatch_removes_per_layer_norm_scale():
    new = torch.ones(2, 1, 10, 1, dtype=torch.float64)
    new[1] *= 10
    old = new / 2
    row = mismatch_rows(cache(old), cache(new))[0]
    assert row["absolute_layer_shares"] == pytest.approx([1 / 101, 100 / 101])
    assert row["relative_layer_evenness"] == pytest.approx([1, 1])
    assert row["absolute_top10_row_share"] == pytest.approx(.1)
    assert row["relative_top10_row_share"] == pytest.approx(.1)


def test_zero_difference_concentration_is_undefined():
    values = torch.ones(2, 2, 4, 2)
    rows = mismatch_rows(cache(values), cache(values))
    assert all(r["absolute_top10_row_share"] is None for r in rows)
    summary = summarize_mismatch(rows)
    assert summary["all"]["absolute"]["top10_row_share"]["count"] == 0
    assert summary["all"]["absolute"]["pooled_layer_shares"] == [None, None]


def test_streaming_pool_distinguishes_global_and_within_user_top_rows():
    # One user's mismatch dwarfs the other; global top 10% need not take one
    # row from each user. This checks that the two pooled definitions differ.
    new = torch.ones(1, 2, 10, 1, dtype=torch.float64)
    new[0, 0, :2] = torch.tensor([[10.], [9.]])
    acc = MismatchAccumulator()
    acc.update(cache(torch.zeros_like(new)), cache(new), uids=[8, 9])
    result = acc.summary()["all"]["absolute"]
    assert result["pooled_global_top10_share"] == pytest.approx(181 / 199)
    assert result["pooled_within_user_top10_share"] == pytest.approx(101 / 199)


def test_long_history_summary_uses_actual_unpadded_length():
    values = torch.ones(1, 2, 300, 1)
    rows = mismatch_rows(cache(torch.zeros_like(values)), cache(values), lengths=[255, 256])
    summary = summarize_mismatch(rows)
    assert summary["history_at_least_256"]["users"] == 1
    assert summary["history_at_least_256"]["history_rows"] == 256


def interaction_tensor():
    generator = torch.Generator().manual_seed(83)
    return torch.randn(7, 9, 8, generator=generator, dtype=torch.float64)


def test_rank_zero_preserves_both_main_effects_and_full_rank_restores_h():
    h = interaction_tensor()
    obj = decompose_interaction(h, max_rank=9)
    expected = h.mean(0, keepdim=True) + h.mean(1, keepdim=True) - h.mean((0, 1), keepdim=True)
    torch.testing.assert_close(obj.reconstruct(0), expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(obj.reconstruct(9), h, atol=1e-13, rtol=1e-13)
    assert torch.equal(obj.reconstruct(None), h)
    residual = h - expected
    torch.testing.assert_close(residual.mean(0), torch.zeros_like(residual.mean(0)), atol=1e-14, rtol=0)
    torch.testing.assert_close(residual.mean(1), torch.zeros_like(residual.mean(1)), atol=1e-14, rtol=0)


def test_truncation_matrix_orientation_and_shared_rank_factors():
    h = interaction_tensor()
    obj = decompose_interaction(h, max_rank=8)
    centered = h - h.mean(0, keepdim=True) - h.mean(1, keepdim=True) + h.mean((0, 1), keepdim=True)
    matrix = centered.permute(0, 2, 1).reshape(-1, h.shape[1])
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    for rank in (0, 1, 2, 4, 6, 8):
        low = ((u[:, :rank] * s[:rank]) @ vh[:rank]).reshape(7, 8, 9).transpose(-1, -2)
        torch.testing.assert_close(obj.reconstruct(rank), h - centered + low, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(obj.reconstruct(rank, candidates=slice(1, 4), rows=slice(2, 5)),
                                   obj.reconstruct(rank)[1:4, 2:5], atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("only", ["key", "value"])
def test_single_block_truncation_keeps_other_block_exact(only):
    h = interaction_tensor()
    obj = decompose_interaction(h, only=only, max_rank=9, normalize_kv=True)
    untouched = slice(4, 8) if only == "key" else slice(0, 4)
    assert torch.equal(obj.reconstruct(0)[..., untouched], h[..., untouched])
    torch.testing.assert_close(obj.reconstruct(9), h, atol=1e-13, rtol=1e-13)


def test_kv_normalization_restores_units_and_balances_interaction():
    h = interaction_tensor()
    h[..., :4] *= 1000
    obj = decompose_interaction(h, normalize_kv=True, max_rank=9)
    assert obj.metadata["block_scales"][0] > 100 * obj.metadata["block_scales"][1]
    torch.testing.assert_close(obj.reconstruct(9), h, atol=1e-10, rtol=1e-13)


def test_randomized_svd_is_seeded_and_recovers_known_low_rank_interaction():
    generator = torch.Generator().manual_seed(17)
    u = torch.randn(12, 6, 2, generator=generator, dtype=torch.float64)
    v = torch.randn(2, 17, generator=generator, dtype=torch.float64)
    h = (u @ v).transpose(-1, -2)
    exact = decompose_interaction(h, max_rank=4)
    approximate = decompose_interaction(h, method="randomized", max_rank=4, oversample=4, power_iterations=1)
    repeated = decompose_interaction(h, method="randomized", max_rank=4, oversample=4, power_iterations=1)
    torch.testing.assert_close(approximate.reconstruct(2), exact.reconstruct(2), atol=1e-12, rtol=1e-12)
    assert torch.equal(approximate.left, repeated.left)
    assert approximate.relative_error(2) < 1e-25
    assert approximate.metadata["method"] == "randomized"


def test_zero_interaction_and_rank_zero_only_decomposition():
    h = torch.ones(5, 7, 4, dtype=torch.float64)
    obj = decompose_interaction(h, normalize_kv=True, max_rank=0)
    assert obj.max_rank == 0
    torch.testing.assert_close(obj.reconstruct(0), h)
    assert obj.metadata["normalized_interaction_squared_norm"] == 0
    assert obj.metadata["block_scales"] == [1, 1]
    with pytest.raises(ValueError, match="rank must"):
        obj.reconstruct(1)


def test_layer_head_api_preserves_each_head_layout():
    h = torch.stack([interaction_tensor(), interaction_tensor() * 3])
    heads = decompose_layer_heads(h, max_rank=9)
    for original, obj in zip(h, heads, strict=True):
        torch.testing.assert_close(obj.reconstruct(9), original, atol=1e-12, rtol=1e-13)


def test_mismatch_checkpoint_retains_scalar_row_masses_and_rejects_changes(tmp_path):
    from experiments.ch3_motivation.code.run_mismatch import _commit_block, _restore_block
    new = torch.arange(1, 21, dtype=torch.float64).reshape(1, 2, 10, 1)
    acc = MismatchAccumulator()
    acc.update(cache(torch.zeros_like(new)), cache(new), uids=[19, 23])
    _commit_block(tmp_path, 0, acc, "frozen-input-code-model")
    restored = _restore_block(tmp_path, 0, [19, 23], "frozen-input-code-model")
    assert restored.rows == acc.rows
    assert restored.summary() == acc.summary()
    with pytest.raises(ValueError, match="identity/users"):
        _restore_block(tmp_path, 0, [19, 23], "changed-input-code-model")
    with (tmp_path / "part_000000.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        _restore_block(tmp_path, 0, [19, 23], "frozen-input-code-model")
