"""CPU contracts for native-KV structural controls and closed-form fitting."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.fit_simple_kv_maps import RowMoments, fit_from_moments, prediction_mse
from experiments.ch3_motivation.code.simple_kv_maps import (
    head_rows, load_maps, maps_from_tensors, rows_to_cache,
)
from hstu_kvcache.models import HSTUKVCache


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fixture_cache():
    # Every coordinate differs, so a head/batch/row transpose cannot pass.
    k = torch.arange(2*2*5*6, dtype=torch.float32).reshape(2, 2, 5, 6)
    return HSTUKVCache(k, k.neg()-1, 5)


def test_head_local_roundtrip_with_two_batches_and_nonmutation():
    cache = fixture_cache()
    before = cache.k.clone(), cache.v.clone()
    rows = head_rows(cache, 2)
    assert rows.shape == (4, 10, 6)
    torch.testing.assert_close(rows[2, 5, :3], cache.k[1, 1, 0, :3])
    torch.testing.assert_close(rows[3, 9, 3:], cache.v[1, 1, 4, 3:])
    rebuilt = rows_to_cache(rows, cache, 2)
    assert rebuilt.seq_len == cache.seq_len
    assert torch.equal(rebuilt.k, cache.k) and torch.equal(rebuilt.v, cache.v)
    assert torch.equal(cache.k, before[0]) and torch.equal(cache.v, before[1])


def test_shift_and_normalized_affine_are_head_local_and_loadable(tmp_path):
    cache = fixture_cache()
    x = head_rows(cache, 2)
    pairs, _, width = x.shape
    tensors = dict(shift=torch.arange(pairs*width).reshape(pairs, width).float()/10,
        xmean=torch.ones(pairs, width)*2, xscale=torch.ones(pairs, width)*3,
        linear_weight=torch.eye(width)[None].repeat(pairs, 1, 1)*2,
        ymean=torch.arange(pairs).float()[:, None].expand(pairs, width).clone())
    checkpoint = tmp_path/"maps.pt"
    torch.save(dict(format="simple_kv_maps_v1", num_heads=2, tensors=tensors,
                    metadata=dict(no_candidates=True)), checkpoint)
    maps = load_maps(checkpoint)
    actual_shift = head_rows(maps["kv_shift"](cache), 2)
    actual_linear = head_rows(maps["kv_linear"](cache), 2)
    torch.testing.assert_close(actual_shift, x+tensors["shift"][:, None])
    torch.testing.assert_close(actual_linear, (x-2)/3*2+tensors["ymean"][:, None])
    assert maps["kv_linear"].metadata == dict(no_candidates=True)
    assert maps["kv_shift"].materialization_flops(5, batch=2) == 10*pairs*width
    assert maps["kv_linear"].materialization_flops(5, batch=2) == 10*pairs*(2*width**2+3*width)
    assert torch.equal(cache.k, fixture_cache().k)


def make_data():
    generator = torch.Generator().manual_seed(23)
    x = torch.randn(4, 91, 6, generator=generator)
    coefficient = torch.randn(4, 6, 6, generator=generator)/3
    intercept = torch.randn(4, 1, 6, generator=generator)
    y = x.bmm(coefficient)+intercept
    return x, y


def test_streaming_moments_and_ridge_match_dense_solution():
    x, y = make_data()
    moments = RowMoments(4, 6, "cpu")
    for start, end in [(0, 7), (7, 60), (60, 91)]:
        moments.add(x[:, start:end], y[:, start:end])
    mean, scale, target, gram, rhs = moments.normalized()
    xd, yd = x.double(), y.double()
    torch.testing.assert_close(mean, xd.mean(1))
    torch.testing.assert_close(target, yd.mean(1))
    torch.testing.assert_close(scale, xd.std(1, unbiased=False))
    standardized = (xd-mean[:, None])/scale[:, None]
    torch.testing.assert_close(gram, standardized.transpose(1, 2).bmm(standardized)/91)
    torch.testing.assert_close(rhs, standardized.transpose(1, 2).bmm(yd-target[:, None])/91)
    candidates = fit_from_moments(moments)
    for regularization, tensors in candidates.items():
        wanted = torch.linalg.solve(gram+regularization*torch.eye(6)[None], rhs).float()
        torch.testing.assert_close(tensors["linear_weight"], wanted)
    fitted = maps_from_tensors(candidates[1e-4], 2)["kv_linear"]
    # Four independent layer/head maps over two layers, each head dimension3.
    cache = rows_to_cache(x, HSTUKVCache(torch.zeros(2, 1, 91, 6), torch.zeros(2, 1, 91, 6), 91), 2)
    prediction = head_rows(fitted(cache), 2)
    assert float((prediction-y).square().mean()) < 1e-7


@pytest.mark.parametrize("kind", ["source", "kv_shift", "kv_linear"])
def test_moment_mse_matches_explicit_affine_output(kind):
    x, y = make_data()
    moments = RowMoments(4, 6, "cpu")
    moments.add(x, y)
    tensors = fit_from_moments(moments)[.1]
    xd = x.double()
    if kind == "source":
        predicted = xd
    elif kind == "kv_shift":
        predicted = xd+tensors["shift"].double()[:, None]
    else:
        predicted = ((xd-tensors["xmean"].double()[:, None])/tensors["xscale"].double()[:, None]).bmm(
            tensors["linear_weight"].double())+tensors["ymean"].double()[:, None]
    error = (predicted-y.double()).square()
    result = prediction_mse(moments, tensors, kind)
    assert result["k_mse"] == pytest.approx(float(error[..., :3].mean()), abs=1e-12)
    assert result["v_mse"] == pytest.approx(float(error[..., 3:].mean()), abs=1e-12)
    assert result["pooled_mse"] == pytest.approx(float(error.mean()), abs=1e-12)


def test_constant_coordinates_have_finite_regularized_fit():
    x = torch.ones(2, 9, 4)*7
    y = torch.ones_like(x)*-3
    moments = RowMoments(2, 4, "cpu")
    moments.add(x, y)
    for tensors in fit_from_moments(moments).values():
        assert all(torch.isfinite(value).all() for value in tensors.values())
        assert torch.equal(tensors["xscale"], torch.full((2, 4), 1e-6))
        assert torch.count_nonzero(tensors["linear_weight"]) == 0
        assert prediction_mse(moments, tensors, "kv_linear")["pooled_mse"] == 0.


def test_empty_or_mismatched_statistics_fail_before_fitting():
    moments = RowMoments(2, 4, "cpu")
    with pytest.raises(ValueError, match="empty"):
        fit_from_moments(moments)
    with pytest.raises(ValueError, match="geometry"):
        moments.add(torch.zeros(2, 3, 4), torch.zeros(2, 2, 4))
