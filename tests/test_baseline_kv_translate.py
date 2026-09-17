import torch

from hstu_kvcache.baselines.kv_translate import (
    KVTranslator,
    fit,
    fit_affine_ridge,
    select_source_layers,
)
from hstu_kvcache.models.kv_cache import HSTUKVCache


def test_centered_ridge_matches_augmented_lstsq_with_unpenalized_intercept():
    rng = torch.Generator().manual_seed(19)
    base = torch.randn(40, 3, generator=rng, dtype=torch.float64)
    x = torch.cat((base, base[:, :1] + base[:, 1:2]), dim=1) + 12
    y = torch.randn(40, 2, generator=rng, dtype=torch.float64) + 7
    ridge = 0.07
    weight, bias = fit_affine_ridge(x, y, ridge)
    design = torch.cat((x, torch.ones(40, 1, dtype=x.dtype)), dim=1)
    penalty = torch.zeros(4, 5, dtype=x.dtype)
    penalty[:, :4] = ridge ** 0.5 * torch.eye(4, dtype=x.dtype)
    reference = torch.linalg.lstsq(
        torch.cat((design, penalty)),
        torch.cat((y, torch.zeros(4, 2, dtype=y.dtype))),
    ).solution
    torch.testing.assert_close(weight, reference[:-1], atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(bias, reference[-1], atol=1e-11, rtol=1e-11)
    # Correlated features remain finite at model-like scales in float32.
    weight32, bias32 = fit_affine_ridge((x * 1000).float(), y.float(), 0.01)
    assert weight32.dtype == bias32.dtype == torch.float32
    assert torch.isfinite(weight32).all() and torch.isfinite(bias32).all()


def _pair(seed, length):
    rng = torch.Generator().manual_seed(seed)
    keys = torch.randn(3, 2, length, 4, generator=rng, dtype=torch.float64)
    values = torch.randn(3, 2, length, 4, generator=rng, dtype=torch.float64)
    source = HSTUKVCache(keys, values, length)
    # Each target depends on two source layers, including cross-head features.
    a = 2 * torch.eye(4, dtype=keys.dtype) + 0.3 * torch.eye(4, dtype=keys.dtype).roll(2, 0)
    b = torch.eye(4, dtype=keys.dtype)
    layer_pairs = ((2, 1), (0, 2), (1, 0))
    target = HSTUKVCache(
        torch.stack([keys[first] @ a + keys[second] @ b + 3 for first, second in layer_pairs]),
        torch.stack([-values[first] @ a + values[second] @ b - 2 for first, second in layer_pairs]),
        length,
    )
    return source, target


def test_cross_layer_cross_head_fit_and_separate_key_value_maps():
    source, target = _pair(31, 192)
    check_source, check_target = _pair(47, 128)
    translator = fit(
        source, target, num_heads=2, k=2, ridge=1e-9,
        selection_source=check_source, selection_target=check_target,
    )
    assert translator.selection_evaluation == "provided_validation"
    assert translator.source_layers.tolist() == [[2, 1], [0, 2], [1, 0]]
    predicted = translator.apply(check_source)
    torch.testing.assert_close(predicted.k, check_target.k, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(predicted.v, check_target.v, atol=1e-9, rtol=1e-9)
    changed_keys = HSTUKVCache(check_source.k + 10, check_source.v, check_source.seq_len)
    torch.testing.assert_close(translator.apply(changed_keys).v, predicted.v)


def test_probe_ranking_labels_in_sample_and_handles_constant_rank_deficient_heads():
    rows = torch.arange(12, dtype=torch.float64).reshape(1, 6, 2)
    values = torch.stack((rows, rows.flip(-1)))
    # Duplicate coordinates give a rank-deficient corresponding-head fit.
    values = values[..., :1].expand(2, 1, 6, 4).clone()
    source = HSTUKVCache(values, torch.ones_like(values), 6)
    target = HSTUKVCache(values * 2 + 1, torch.ones_like(values) * 5, 6)
    selected, scores, evaluation = select_source_layers(source, target, num_heads=2, k=1)
    assert evaluation == "in_sample"
    assert selected.shape == (2, 1)
    # K is perfectly predicted; constant V contributes no selection signal.
    torch.testing.assert_close(scores, torch.full_like(scores, 0.5))


def test_two_mappings_use_actual_snapshot_with_appended_and_evicted_rows():
    keys = torch.arange(24, dtype=torch.float32).reshape(2, 1, 6, 2)
    cache = HSTUKVCache(keys, -keys, 6)
    original_k, original_v = cache.k.clone(), cache.v.clone()
    weights = torch.eye(2).repeat(2, 1, 1)
    first = KVTranslator(
        torch.tensor([[1], [0]]), weights, torch.ones(2, 2),
        weights * 2, -torch.ones(2, 2),
    )
    mapped = first.apply(cache)
    torch.testing.assert_close(mapped.k, original_k.flip(0) + 1)
    torch.testing.assert_close(mapped.v, original_v.flip(0) * 2 - 1)
    torch.testing.assert_close(cache.k, original_k)
    torch.testing.assert_close(cache.v, original_v)
    # A new current-produced row is appended and the oldest row is evicted.
    new_k = torch.full((2, 1, 1, 2), 123.0)
    new_v = torch.full_like(new_k, -45)
    live = HSTUKVCache(
        torch.cat((mapped.k[:, :, 1:], new_k), dim=2),
        torch.cat((mapped.v[:, :, 1:], new_v), dim=2),
        6,
    )
    live_k, live_v = live.k.clone(), live.v.clone()
    second = KVTranslator(
        torch.tensor([[1], [0]]), weights * 3, torch.zeros(2, 2),
        weights, torch.full((2, 2), 4.0),
    )
    final = second.apply(live)
    torch.testing.assert_close(final.k, live_k.flip(0) * 3)
    torch.testing.assert_close(final.v, live_v.flip(0) + 4)
    torch.testing.assert_close(live.k, live_k)
    torch.testing.assert_close(live.v, live_v)
    assert final.seq_len == cache.seq_len
    assert final.k.dtype == cache.k.dtype and final.k.device == cache.k.device
