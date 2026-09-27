import torch

from hstu_kvcache.baselines.deviation_recompute import (
    layer0_deviation_scores,
    recompute_deviation,
)
from hstu_kvcache.baselines.sparse_recompute import recompute_positions, select_top_positions
from hstu_kvcache.models import HSTUKVCache


def test_deviation_only_reads_layer0_and_matches_current_projection(model_factory, raw_events):
    parent, current = model_factory(17), model_factory(29)
    cache = parent.compute_kv(*raw_events)
    exact = current.compute_kv(*raw_events)
    expected = ((exact.k[0] - cache.k[0]).square().mean(-1)
                + (exact.v[0] - cache.v[0]).square().mean(-1))
    # Upper layers must not execute just to discover the early-layer deviation.
    def forbid(_module, _inputs):
        raise AssertionError("selector executed an upper block")
    handles = [block.register_forward_pre_hook(forbid) for block in current.blocks[1:]]
    try:
        actual = layer0_deviation_scores(current, cache, *raw_events)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(actual, expected)
    positions = select_top_positions(actual, 2)
    selected = recompute_deviation(current, cache, *raw_events, n=2)
    reference = recompute_positions(current, cache, *raw_events, positions)
    torch.testing.assert_close(selected.k, reference.k)
    torch.testing.assert_close(selected.v, reference.v)


def test_deviation_zero_and_full_skip_selection(model_factory, raw_events, monkeypatch):
    parent, current = model_factory(17), model_factory(29)
    cache = parent.compute_kv(*raw_events)
    def forbidden(*args, **kwargs):
        raise AssertionError("endpoint incurred token-selection overhead")
    monkeypatch.setattr("hstu_kvcache.baselines.deviation_recompute.core.layer0_deviation_scores",
                        forbidden)
    assert recompute_deviation(current, cache, *raw_events, n=0) is cache
    actual = recompute_deviation(current, cache, *raw_events, n=cache.seq_len)
    full = current.compute_kv(*raw_events)
    torch.testing.assert_close(actual.k, full.k)
    torch.testing.assert_close(actual.v, full.v)


def test_roundoff_ties_ignore_gemm_noise_and_preserve_real_deviation(model_factory, raw_events):
    current = model_factory(29)
    exact = current.compute_kv(*raw_events)
    eps = torch.finfo(exact.k.dtype).eps
    # Simulate different reduction/tiling orders for mathematically identical
    # current-produced layer-0 rows. Noise rankings deliberately run backwards.
    scales = exact.k[0].square().mean(-1, keepdim=True).sqrt()
    perturbation = torch.arange(1, 7).float()[None, :, None] * eps * scales
    caches = []
    for noise in (perturbation, perturbation.flip(1)):
        k, v = exact.k.clone(), exact.v.clone()
        k[0] += noise
        caches.append(HSTUKVCache(k, v, exact.seq_len))
    scores = [layer0_deviation_scores(current, cache, *raw_events) for cache in caches]
    assert all(torch.equal(score, torch.zeros_like(score)) for score in scores)
    assert all(select_top_positions(score, 3).tolist() == [[0, 1, 2], [0, 1, 2]] for score in scores)

    # A genuine changed row remains nonzero and outranks the zero ties.
    caches[0].k[0, :, 4] += 0.02
    actual = layer0_deviation_scores(current, caches[0], *raw_events)
    expected = ((exact.k[0, :, 4] - caches[0].k[0, :, 4]).square().mean(-1)
                + (exact.v[0, :, 4] - caches[0].v[0, :, 4]).square().mean(-1))
    torch.testing.assert_close(actual[:, 4], expected)
    assert select_top_positions(actual, 2).tolist() == [[0, 4], [0, 4]]
