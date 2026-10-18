"""Algebra and dynamic-reader checks, independent of a quality experiment."""
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from code_structure_2026_10.structure import StructuredKVAdapter
from hstu_kvcache.models import HSTUKVCache, PointwiseAttention, PointwiseAttentionConfig
from query_relay_2026_10.adapter import ReleasePrefix, UnifiedRelayAdapter
from query_relay_2026_10.fixed_features import build_bank, load_bank


@pytest.fixture
def fixture():
    torch.manual_seed(20261014)
    source = StructuredKVAdapter(2, 6, 4, layout="shared12_private6").double().eval()
    with torch.no_grad():
        for layer in source.layers:
            layer.input_weight.normal_(std=.05)
            layer.input_bias.normal_(std=.03)
            layer.key_decoder.normal_(std=.04)
            layer.decoder.normal_(std=.04)
    cache = HSTUKVCache(torch.randn(2, 2, 5, 24, dtype=torch.float64),
                       torch.randn(2, 2, 5, 24, dtype=torch.float64), 5)
    catalog = torch.randn(13, 24, dtype=torch.float64)
    items = torch.randint(0, len(catalog), (2, 5))
    attention = PointwiseAttention(PointwiseAttentionConfig(24, 6, head_dim=4)).double()
    q = torch.randn(2, 6, 3, 4, dtype=torch.float64) * .3
    return source, cache, catalog, items, attention, q


def _random_update(adapter):
    with torch.no_grad():
        for layer in adapter.layers:
            for parameter in layer.rotation:
                parameter.normal_(std=.07)
            for parameter in layer.native_update:
                parameter.normal_(std=.1)
            layer.item_update_weight.normal_(std=.04)
            layer.item_update_bias.normal_(std=.02)


def test_zero_gate_warmstart_preserves_original_read(fixture):
    source, cache, catalog, items, attention, q = fixture
    adapter = UnifiedRelayAdapter.from_structured(source, mode="legacy")
    old = source.encode_cache(cache, catalog[items])
    new = adapter.encode_initial(cache, catalog[items])
    torch.testing.assert_close(new.codes, old.codes, atol=0, rtol=0)
    attention.eval()
    for index in range(2):
        wanted = source.layers[index].read(attention, q, cache.k[index], cache.v[index],
                                           old.codes[index])
        got = adapter.make_history_read(new)(index, attention, q,
                                            cache.k[index], cache.v[index])
        torch.testing.assert_close(got, wanted, atol=1e-12, rtol=1e-12)


def test_query_non_linearity_and_query_specific_effective_kv(fixture):
    source, cache, catalog, items, attention, q = fixture
    adapter = UnifiedRelayAdapter.from_structured(source, mode="legacy")
    state = adapter.encode_initial(cache, catalog[items])
    layer = adapter.layers[0]
    with torch.no_grad():
        layer.key_gate_weight.normal_(std=.5)
        layer.value_gate_weight.normal_(std=.5)
    z = layer.accessible_codes(state.codes[0])
    gk, gv = layer.query_gates(q)
    phi = (q @ layer.key_decoder.transpose(-2, -1)[None]) * gk
    phi2 = (2 * q @ layer.key_decoder.transpose(-2, -1)[None]) * layer.query_gates(2*q)[0]
    assert ((phi2 - 2*phi) @ z.transpose(-2, -1)).abs().max() > 1e-5

    # For each individual query one can expand KV, but the expansion varies
    # with that query. There is no single static key shift in this witness.
    dk = torch.einsum("bhnr,bhqr,hrd->bhqnd", z, gk, layer.key_decoder)
    dv = torch.einsum("bhnr,bhqr,hrd->bhqnd", z, gv, layer.decoder)
    keys = cache.k[0].reshape(2, 5, 6, 4).transpose(1, 2)
    values = cache.v[0].reshape(2, 5, 6, 4).transpose(1, 2)
    weights = attention._activate(torch.einsum("bhqd,bhqnd->bhqn", q,
                                             keys[:, :, None] + dk) * attention.scale)
    expanded = torch.einsum("bhqn,bhqnd->bhqd", weights, values[:, :, None] + dv)
    actual = layer.read(attention, q, cache.k[0], cache.v[0], state.codes[0])
    torch.testing.assert_close(actual, expanded, atol=1e-12, rtol=1e-12)
    assert (dk[:, :, 0] - dk[:, :, 1]).abs().max() > 1e-5


@pytest.mark.parametrize("mode", ["A", "B"])
def test_composition_matches_every_sequential_release(fixture, mode):
    source, cache, catalog, items, _, _ = fixture
    origin = UnifiedRelayAdapter.from_structured(source, mode=mode)
    prefix = ReleasePrefix.identity(origin, len(catalog))
    anchors = prefix.zero_anchors(cache, items)
    sequential = prefix.materialize(cache, anchors, items)
    assert sequential.storage_bytes() == cache.k.shape[0] * 2 * 5 * 48 * 8
    for _ in range(4):
        adapter = copy.deepcopy(origin)
        _random_update(adapter)
        sequential = adapter.advance_codes(cache, sequential, catalog[items])
        prefix = adapter.compile_release(prefix, catalog)
        direct = prefix.materialize(cache, anchors, items)
        torch.testing.assert_close(direct.codes, sequential.codes, atol=2e-12, rtol=2e-12)
        restored = prefix.anchorize(cache, direct, items)
        torch.testing.assert_close(restored.codes, anchors.codes, atol=3e-12, rtol=3e-12)
        for layer in adapter.layers:
            a, _ = layer.update_matrices()
            torch.testing.assert_close(a.T @ a, torch.eye(48, dtype=a.dtype),
                                       atol=1e-12, rtol=1e-12)
            # A shared coordinate has one owner; private heads cannot rotate
            # into another head or into the shared block.
            assert torch.count_nonzero(a[:12, 12:]) == 0
            assert torch.count_nonzero(a[12:18, 18:]) == 0


def test_zero_birth_anchor_and_fifo_preserve_row_semantics(fixture):
    source, cache, catalog, items, _, _ = fixture
    adapter = UnifiedRelayAdapter.from_structured(source, mode="B")
    _random_update(adapter)
    prefix = adapter.compile_release(None, catalog)
    born = prefix.zero_anchors(cache, items)
    torch.testing.assert_close(prefix.materialize(cache, born, items).codes,
                               torch.zeros_like(born.codes), atol=2e-12, rtol=0)
    following = copy.deepcopy(adapter)
    _random_update(following)
    next_prefix = following.compile_release(prefix, catalog)
    after = next_prefix.materialize(cache, born, items)
    expected = following.advance_codes(cache, torch.zeros_like(born.codes), catalog[items])
    torch.testing.assert_close(after.codes, expected.codes, atol=3e-12, rtol=3e-12)
    retained = HSTUKVCache(cache.k[:, :, 2:], cache.v[:, :, 2:], 3)
    retained_anchors = type(born)(born.codes[:, :, 2:], 3)
    got = next_prefix.materialize(retained, retained_anchors, items[:, 2:])
    torch.testing.assert_close(got.codes, after.codes[:, :, 2:], atol=3e-12, rtol=3e-12)


def test_known_zero_suffix_and_causal_mask_are_exact(fixture):
    source, cache, catalog, items, attention, q = fixture
    adapter = UnifiedRelayAdapter.from_structured(source, mode="legacy")
    state = adapter.encode_initial(cache, catalog[items])
    state.codes[:, :, 3:] = 0
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1]])
    for layer in range(2):
        full = adapter.make_history_read(state, valid_mask=mask)(
            layer, attention, q, cache.k[layer], cache.v[layer])
        prefix = adapter.make_history_read(state, valid_mask=mask, code_count=3)(
            layer, attention, q, cache.k[layer], cache.v[layer])
        torch.testing.assert_close(prefix, full, atol=1e-12, rtol=1e-12)
    no_codes = adapter.make_history_read(state, valid_mask=mask, code_count=0)(
        0, attention, q, cache.k[0], cache.v[0])
    keys = cache.k[0].reshape(2, 5, 6, 4).transpose(1, 2)
    values = cache.v[0].reshape(2, 5, 6, 4).transpose(1, 2)
    native = (attention._activate(q @ keys.transpose(-2, -1) * attention.scale)
              * mask[None, None]) @ values
    torch.testing.assert_close(no_codes, native, atol=0, rtol=0)


def test_calibration_gradients_with_zero_dropout_training(fixture):
    source, cache, catalog, items, attention, q = fixture
    adapter = UnifiedRelayAdapter.from_structured(source, mode="B")
    previous = cache.k.new_empty(2, 2, 5, 6, 8).normal_()
    state = adapter.advance_codes(cache, previous, catalog[items])
    attention.train()
    output = adapter.make_history_read(state)(0, attention, q, cache.k[0], cache.v[0])
    output.square().mean().backward()
    layer = adapter.layers[0]
    for parameter in (layer.key_gate_weight, layer.value_gate_weight,
                      layer.item_update_weight, layer.native_update[0], layer.rotation[0]):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert not layer.native_basis_0.requires_grad


def test_c_is_explicitly_noncomposable(fixture):
    source, cache, catalog, items, _, _ = fixture
    adapter = UnifiedRelayAdapter.from_structured(source, mode="C")
    initial = adapter.encode_initial(cache, catalog[items])
    updated = adapter.advance_codes(cache, initial, catalog[items])
    assert not torch.equal(initial.codes, updated.codes)
    with pytest.raises(ValueError, match="affine"):
        adapter.compile_release(None, catalog)


@pytest.mark.parametrize("read_shared", [0, 6, 12])
def test_read_width_matches_zeroed_full_reader_and_keeps_state_budget(fixture, read_shared):
    source, cache, catalog, items, attention, q = fixture
    full = UnifiedRelayAdapter.from_structured(source, mode="B")
    with torch.no_grad():
        for layer in full.layers:
            layer.key_gate_weight.normal_(std=.3)
            layer.value_gate_weight.normal_(std=.3)
    narrow = copy.deepcopy(full)
    narrow.set_read_shared(read_shared)
    oracle = copy.deepcopy(full)
    with torch.no_grad():
        for layer in oracle.layers:
            layer.key_decoder[:, read_shared:12] = 0
            layer.decoder[:, read_shared:12] = 0
    previous = cache.k.new_empty(2, 2, 5, 6, 8).normal_()
    state = narrow.advance_codes(cache, previous, catalog[items])
    full_state = full.advance_codes(cache, previous, catalog[items])
    torch.testing.assert_close(state.codes, full_state.codes, atol=0, rtol=0)
    assert state.codes.shape == (2, 2, 5, 6, 8)
    assert narrow.accessible_rank == read_shared + 6
    assert narrow.get_config()["read_shared"] == read_shared
    state.codes[:, :, 3:] = 0
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1]])
    for index in range(2):
        layer = narrow.layers[index]
        assert layer.query_gates(q)[0].shape == (2, 6, 3, read_shared + 6)
        assert layer.accessible_codes(state.codes[index]).shape == (2, 6, 5, read_shared + 6)
        assert layer.key_decoder.shape == (6, 18, 4)
        assert layer.access_indices[0, layer.read_slots].tolist() == list(range(read_shared)) + list(range(12, 18))
        actual = narrow.make_history_read(state, valid_mask=mask, code_count=3)(
            index, attention, q, cache.k[index], cache.v[index])
        expected = oracle.make_history_read(state, valid_mask=mask, code_count=3)(
            index, attention, q, cache.k[index], cache.v[index])
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("mode", ["A", "B"])
def test_origin_encoder_warmstart_keeps_item_native_cancellation(fixture, tmp_path, mode):
    source, cache, catalog, items, attention, q = fixture
    record = build_bank(source, catalog, tmp_path / "origin.pt",
                        source_bindings={"model": "frozen_toy_V3"}, chunk_size=4)
    adapter = UnifiedRelayAdapter.from_structured(source, mode=mode,
        feature_mode="origin_encoder", feature_bank_path=record["path"],
        feature_bank_sha256=record["sha256"])
    expected = source.encode_cache(cache, catalog[items])
    # Current embeddings are allowed to change; the fixed feature still uses
    # the frozen origin embedding indexed by the immutable raw item ID.
    actual = adapter.encode_initial(cache, catalog[items] + 100, item_ids=items)
    torch.testing.assert_close(actual.codes, expected.codes, atol=2e-13, rtol=2e-13)
    for index in range(2):
        eta, _ = adapter.layers[index].fixed_features(cache.k[index], cache.v[index], items)
        torch.testing.assert_close(eta, expected.codes[index].flatten(-2), atol=2e-13, rtol=2e-13)
        assert not torch.count_nonzero(adapter.layers[index].item_update_weight)
        assert not torch.count_nonzero(adapter.layers[index].item_update_bias)
        attention.eval()
        wanted = source.layers[index].read(attention, q, cache.k[index], cache.v[index],
                                           expected.codes[index])
        got = adapter.make_history_read(actual)(index, attention, q, cache.k[index], cache.v[index])
        torch.testing.assert_close(got, wanted, atol=2e-12, rtol=2e-12)
    assert actual.storage_bytes() == expected.storage_bytes()
    assert not any(key.endswith("table") for key in adapter.state_dict())
    assert adapter.get_config()["feature_bank_sha256"] == record["sha256"]
    with pytest.raises(ValueError, match="raw item IDs"):
        adapter.encode_initial(cache, catalog[items])


def test_origin_bank_reuses_one_asset_and_validates_binding(fixture, tmp_path):
    source, _, catalog, _, _, _ = fixture
    path = tmp_path / "origin.pt"
    binding = {"model": "frozen_toy_V3"}
    first = build_bank(source, catalog, path, source_bindings=binding, chunk_size=4)
    reused = build_bank(source, catalog, path, source_bindings=binding, chunk_size=4)
    assert first["created"] and not reused["created"]
    assert first["sha256"] == reused["sha256"]
    assert first["prepare_flops"] == 2 * 2 * len(catalog) * 24 * 48
    assert reused["prepare_flops"] == 0
    bank = load_bank(path, first["sha256"], device="cpu", dtype=torch.float64)
    again = load_bank(path, first["sha256"], device="cpu", dtype=torch.float64)
    assert bank is again and bank.table.data_ptr() == again.table.data_ptr()
    assert bank.table.shape == (2, len(catalog), 48)
    assert len(list(tmp_path.iterdir())) == 1
    with pytest.raises(ValueError, match="different origin"):
        build_bank(source, catalog, path, source_bindings={"model": "frozen_toy_V4"})
    with pytest.raises(ValueError, match="SHA256"):
        load_bank(path, "0" * 64, device="cpu", dtype=torch.float64)


def test_origin_encoder_composition_and_mapped_catalogue_use_raw_item_ids(fixture, tmp_path):
    source, cache, catalog, _, _, _ = fixture
    items = torch.tensor([[1, 5, 8, 5, 12], [12, 1, 8, 1, 5]])
    bank = build_bank(source, catalog, tmp_path / "origin.pt",
                      source_bindings={"model": "frozen_toy_V3"})
    origin = UnifiedRelayAdapter.from_structured(source, mode="B",
        feature_mode="origin_encoder", feature_bank_path=bank["path"],
        feature_bank_sha256=bank["sha256"])
    prefix = ReleasePrefix.identity(origin, len(catalog))
    anchors = prefix.zero_anchors(cache, items)
    sequential = prefix.materialize(cache, anchors, items)
    shared_bank = load_bank(bank["path"], bank["sha256"], device="cpu", dtype=torch.float64)
    for release in range(3):
        actor = copy.deepcopy(origin)
        _random_update(actor)
        current_catalog = catalog + torch.randn_like(catalog) * .1 * release
        sequential = actor.advance_codes(cache, sequential, current_catalog[items], item_ids=items)
        prefix = actor.compile_release(prefix, current_catalog)
        direct = prefix.materialize(cache, anchors, items)
        torch.testing.assert_close(direct.codes, sequential.codes, atol=3e-12, rtol=3e-12)
        restored = prefix.anchorize(cache, direct, items)
        torch.testing.assert_close(restored.codes, anchors.codes, atol=3e-12, rtol=3e-12)
        # deepcopy keeps only strings describing the bank, never its table.
        assert all(not hasattr(layer, "feature_bank") for layer in actor.layers)
        assert shared_bank is load_bank(actor.feature_bank_path, actor.feature_bank_sha256,
                                        device="cpu", dtype=torch.float64)
    subset = torch.unique(items)
    positions = torch.searchsorted(subset, items)
    mapped = ReleasePrefix(prefix.P, prefix.Q, prefix.D[:, subset], prefix.layers, prefix.release)
    actual = mapped.materialize(cache, anchors, items, table_item_ids=positions)
    expected = prefix.materialize(cache, anchors, items)
    torch.testing.assert_close(actual.codes, expected.codes, atol=0, rtol=0)
    zero_anchors = mapped.zero_anchors(cache, items, table_item_ids=positions)
    newborn = mapped.materialize(cache, zero_anchors, items, table_item_ids=positions)
    torch.testing.assert_close(newborn.codes, torch.zeros_like(newborn.codes), atol=3e-12, rtol=0)
