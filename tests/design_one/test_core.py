"""Numerical checks for the Design 1 formula and real retained-state updates."""

import torch

from hstu_kvcache.design_one import ProducerSummary, SharedReadAdapter, SummaryProjection
from hstu_kvcache.models import HSTUKVCache


def _cache(k, v):
    return HSTUKVCache(k, v, k.shape[2])


def _adapter(features, layers=2, heads=2, width=3, *, max_length=8):
    projection = SummaryProjection.fit(features, rank=3)
    latent = projection.projection.shape[1] + 1
    parameters = [dict(
        weights=torch.randn(latent, heads, width + 1, width, dtype=torch.double) * .01,
        query_center=torch.randn(1, heads, 1, width, dtype=torch.double),
        query_scale=torch.rand(1, heads, 1, width, dtype=torch.double) + .5,
        read_weights=torch.randn(heads, heads * width, width, dtype=torch.double) * .01,
        read_center=torch.randn(heads * width, dtype=torch.double),
        read_scale=torch.rand(heads * width, dtype=torch.double) + .5,
    ) for _ in range(layers)]
    return SharedReadAdapter(projection, parameters, max_length=max_length)


def test_incremental_producer_summary_matches_retained_cache_reduction():
    torch.manual_seed(1791)
    initial = _cache(torch.randn(2, 3, 5, 6), torch.randn(2, 3, 5, 6))
    lineage = [4, 4, 5, 4, 5]
    summary = ProducerSummary.from_cache(initial, lineage, max_length=8)
    summary.evict_prefix(initial, 2)
    added = _cache(torch.randn(2, 3, 3, 6), torch.randn(2, 3, 3, 6))
    summary.add(added, 5)
    retained = _cache(torch.cat((initial.k[:, :, 2:], added.k), 2),
                      torch.cat((initial.v[:, :, 2:], added.v), 2))
    expected = ProducerSummary.from_cache(retained, lineage[2:] + [5, 5, 5], max_length=8)
    expected.native_writes.fill_(3)
    torch.testing.assert_close(summary.features(), expected.features(), atol=2e-7, rtol=2e-6)
    assert summary.counts.tolist() == [[1, 5]] * 3
    assert summary.revision == 2
    # Empty producer groups are zero even after prior floating additions.
    summary.evict_prefix(retained, retained.seq_len)
    assert torch.count_nonzero(summary.sums) == 0
    assert torch.isfinite(summary.features()).all()


def test_prepared_correction_matches_normalized_shared_rule():
    torch.manual_seed(1792)
    features = torch.randn(7, 31)
    adapter = _adapter(features, layers=1)
    counts = torch.tensor([1, 2, 3, 4, 5, 6, 8], dtype=torch.double)
    view = adapter.prepare_features(features, counts)
    q, native = torch.randn(7, 2, 5, 3, dtype=torch.double), torch.randn(7, 2, 5, 3, dtype=torch.double)
    p = adapter.parameters[0]
    normalized_q = (q - p["query_center"]) / p["query_scale"]
    augmented_q = torch.cat((torch.ones_like(q[..., :1]), normalized_q), dim=-1)
    latent = adapter.projection.encode_features(features)
    observed = native.transpose(1, 2).flatten(2) / counts[:, None, None]
    observed = (observed - p["read_center"]) / p["read_scale"]
    delta = torch.einsum("bhqi,br,rhid->bhqd", augmented_q, latent, p["weights"])
    delta += torch.einsum("bqa,had->bhqd", observed, p["read_weights"])
    expected = native + delta * counts[:, None, None, None]
    torch.testing.assert_close(view(0, q, native), expected, atol=2e-15, rtol=2e-14)
    assert view(1, q, native) is native
    restored = SharedReadAdapter.from_state_dict(adapter.state_dict())
    torch.testing.assert_close(restored.prepare_features(features, counts)(0, q, native), expected)
    # Serving uses the same actual q/r, with a retained FP32 view.
    served = adapter.to("cpu").prepare_features(features, counts.float())
    torch.testing.assert_close(served(0, q.float(), native.float()), expected.float(), atol=3e-7, rtol=2e-5)


def test_selected_owner_updates_preserve_different_producer_lineages():
    torch.manual_seed(1794)
    cache = _cache(torch.randn(2, 3, 5, 6), torch.randn(2, 3, 5, 6))
    producers = torch.tensor([[4, 4, 5, 4, 5], [4, 4, 4, 4, 4], [5, 5, 5, 4, 4]])
    summary = ProducerSummary.from_cache(cache, producers, max_length=5)
    original = summary.features().clone()
    owners = torch.tensor([0, 2])
    selected_cache = _cache(cache.k[:, owners], cache.v[:, owners])
    updated = summary.select(owners)
    updated.evict_prefix(selected_cache, 2)
    added = _cache(torch.randn(2, 2, 2, 6), torch.randn(2, 2, 2, 6))
    updated.add(added, 5)
    summary.put(owners, updated)
    torch.testing.assert_close(summary.features()[1], original[1])
    new_k, new_v = cache.k.clone(), cache.v.clone()
    new_k[:, owners] = torch.cat((selected_cache.k[:, :, 2:], added.k), dim=2)
    new_v[:, owners] = torch.cat((selected_cache.v[:, :, 2:], added.v), dim=2)
    new_producers = producers.clone()
    new_producers[owners] = torch.cat((producers[owners, 2:], torch.full((2, 2), 5)), dim=1)
    reference = ProducerSummary.from_cache(_cache(new_k, new_v), new_producers, max_length=5)
    reference.native_writes[owners] = 2
    torch.testing.assert_close(summary.features(), reference.features(), atol=2e-7, rtol=2e-6)


def test_mass_features_are_additive_and_first_native_event_has_retained_fraction():
    cache = _cache(torch.full((1, 1, 8, 1), 2.), torch.full((1, 1, 8, 1), 3.))
    summary = ProducerSummary.from_cache(cache, 4, max_length=8)
    before = summary.features("producer_mass")
    summary.evict_prefix(cache, 1)
    added = _cache(torch.tensor([[[[10.]]]]), torch.tensor([[[[20.]]]]))
    summary.add(added, 5)
    mass = summary.features("producer_mass")
    mean = summary.features("producer_mean")
    # Producer blocks have K, V, retained count/context, retained fraction.
    torch.testing.assert_close(before[0, 4:6], torch.zeros(2))
    torch.testing.assert_close(mass[0, 4:6], mean[0, 4:6] / 8)
    torch.testing.assert_close(mass[0, :2], torch.tensor([2., 3.]) * 7 / 8)
    torch.testing.assert_close(mass[0, :2] + mass[0, 4:6], torch.tensor([24., 41.]) / 8)
    torch.testing.assert_close(mass[:, [2, 3, 6, 7, 8]], mean[:, [2, 3, 6, 7, 8]])

    torch.manual_seed(1796)
    initial = _adapter(torch.randn(7, mass.shape[1]), layers=1, heads=1, width=1)
    adapter = SharedReadAdapter(initial.projection, initial.parameters,
                                max_length=8, summary_mode="producer_mass")
    restored = SharedReadAdapter.from_state_dict(adapter.state_dict()).to("cpu")
    assert restored.summary_mode == "producer_mass"
    actual = restored.prepare(summary)
    expected = restored.prepare_features(mass, summary.batch_counts())
    torch.testing.assert_close(actual.layers[0].offset, expected.layers[0].offset)
    old_checkpoint = initial.state_dict()
    old_checkpoint.pop("summary_mode")
    assert SharedReadAdapter.from_state_dict(old_checkpoint).summary_mode == "producer_mean"
