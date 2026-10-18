"""Research checks for tiled in-place conversion of the current compact state."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from code_structure_2026_10.structure import StructuredKVAdapter
from hstu_kvcache.design_one.lightweight import LightweightValueState
from hstu_kvcache.models import HSTUKVCache
from implementation_2026_10.state import InplaceReleasePrefix
from query_relay_2026_10.adapter import UnifiedRelayAdapter
from query_relay_2026_10.fixed_features import build_bank
from query_relay_2026_10.precision import StableReleasePrefix


def example(dtype=torch.float64):
    torch.manual_seed(20261015)
    source = StructuredKVAdapter(2, 6, 4, layout="shared12_private6").to(dtype).eval()
    with torch.no_grad():
        for layer in source.layers:
            layer.input_weight.normal_(std=.05)
            layer.input_bias.normal_(std=.02)
    actor = UnifiedRelayAdapter.from_structured(source, mode="A").eval()
    with torch.no_grad():
        for layer in actor.layers:
            for rotation in layer.rotation:
                rotation.normal_(std=.08)
    cache = HSTUKVCache(torch.randn(2, 2, 11, 24, dtype=dtype),
                       torch.randn(2, 2, 11, 24, dtype=dtype), 11)
    catalog = torch.randn(17, 24, dtype=dtype)
    ids = torch.arange(1, 12)[None].repeat(2, 1)
    return source, actor, cache, catalog, ids


@pytest.mark.parametrize("chunk", [1, 4, 32])
@torch.no_grad()
def test_inplace_roundtrip_matches_reference_without_changing_storage(chunk):
    _, actor, cache, catalog, ids = example()
    prefix = StableReleasePrefix.wrap(actor.compile_release(None, catalog))
    fast = InplaceReleasePrefix.wrap(prefix, row_chunk=chunk)
    state = actor.encode_initial(cache, catalog[ids])
    snapshot = state.codes.clone()
    native = (cache.k.clone(), cache.v.clone())
    pointer = state.codes.data_ptr()
    expected = prefix.anchorize(cache, state, ids)
    assert fast.anchorize(cache, state, ids) is state
    assert state.codes.data_ptr() == pointer
    torch.testing.assert_close(state.codes, expected.codes, atol=2e-14, rtol=2e-14)
    expected = prefix.materialize(cache, expected, ids)
    assert fast.materialize(cache, state, ids) is state
    assert state.codes.data_ptr() == pointer
    torch.testing.assert_close(state.codes, expected.codes, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(state.codes, snapshot, atol=2e-14, rtol=2e-14)
    assert torch.equal(cache.k, native[0]) and torch.equal(cache.v, native[1])
    assert fast.P.data_ptr() == prefix.P.data_ptr()
    assert fast.inverse_blocks[0][0].data_ptr() == prefix.inverse_blocks[0][0].data_ptr()


@torch.no_grad()
def test_newborn_zero_mixed_rows_fifo_and_later_release():
    _, actor, cache, catalog, ids = example()
    prefix = StableReleasePrefix.wrap(actor.compile_release(None, catalog))
    fast = InplaceReleasePrefix.wrap(prefix, row_chunk=3)
    current = actor.encode_initial(cache, catalog[ids])
    current.codes[:, :, -3:] = 0
    expected_anchor = prefix.anchorize(cache, current, ids)
    fast.anchorize(cache, current, ids)
    # FIFO survivors can be noncontiguous between layers/batch members.
    retained = HSTUKVCache(cache.k[:, :, 2:], cache.v[:, :, 2:], 9)
    survivor = LightweightValueState(current.codes[:, :, 2:], 9)
    expected_anchor = LightweightValueState(expected_anchor.codes[:, :, 2:], 9)
    next_prefix = StableReleasePrefix.wrap(actor.compile_release(prefix, catalog))
    want = next_prefix.materialize(retained, expected_anchor, ids[:, 2:])
    ptr = survivor.codes.data_ptr()
    InplaceReleasePrefix.wrap(next_prefix, row_chunk=4).materialize(retained, survivor, ids[:, 2:])
    assert survivor.codes.data_ptr() == ptr
    torch.testing.assert_close(survivor.codes, want.codes, atol=3e-14, rtol=3e-14)


@torch.no_grad()
def test_origin_raw_ids_distinct_from_subset_table_ids(tmp_path):
    source, _, cache, catalog, ids = example()
    bank = build_bank(source, catalog, tmp_path / "bank.pt", source_bindings={"test": "fixed"})
    actor = UnifiedRelayAdapter.from_structured(source, mode="A", feature_mode="origin_encoder",
        feature_bank_path=bank["path"], feature_bank_sha256=bank["sha256"]).eval()
    subset = catalog[1:12]
    prefix = StableReleasePrefix.wrap(actor.compile_release(None, subset))
    current = actor.encode_initial(cache, catalog[ids], item_ids=ids)
    expected = prefix.anchorize(cache, current, ids, table_item_ids=ids - 1)
    fast = InplaceReleasePrefix.wrap(prefix, row_chunk=2)
    fast.anchorize(cache, current, ids, table_item_ids=ids - 1)
    torch.testing.assert_close(current.codes, expected.codes, atol=3e-14, rtol=3e-14)
    before = current.codes.clone()
    bad = ids.clone()
    bad[0, -1] = len(catalog)
    with pytest.raises(ValueError, match="immutable feature bank"):
        fast.materialize(cache, current, bad, table_item_ids=ids - 1)
    assert torch.equal(before, current.codes)  # Invalid tail IDs cannot partly mutate the prefix.


@torch.no_grad()
def test_empty_cache_and_feature_callback():
    _, actor, cache, catalog, ids = example()
    prefix = StableReleasePrefix.wrap(actor.compile_release(None, catalog))
    calls = []
    def features(layer, k, v, raw):
        calls.append(k.shape[1])
        return layer.fixed_features(k, v, raw)[0]
    fast = InplaceReleasePrefix.wrap(prefix, row_chunk=4, feature_fn=features)
    current = actor.encode_initial(cache, catalog[ids])
    fast.anchorize(cache, current, ids)
    assert calls == [4, 4, 3] * 2
    empty = HSTUKVCache(cache.k[:, :, :0], cache.v[:, :, :0], 0)
    state = LightweightValueState(current.codes[:, :, :0], 0)
    assert fast.materialize(empty, state, ids[:, :0]) is state
    assert calls == [4, 4, 3] * 2


@torch.no_grad()
def test_invalid_geometry_and_alias_are_rejected_before_mutation():
    _, actor, cache, catalog, ids = example()
    fast = InplaceReleasePrefix.wrap(actor.compile_release(None, catalog), row_chunk=3)
    current = actor.encode_initial(cache, catalog[ids])
    before = current.codes.clone()
    bad = ids.clone()
    bad[0, -1] = len(catalog)
    with pytest.raises(ValueError, match="publication table"):
        fast.materialize(cache, current, bad)
    assert torch.equal(current.codes, before)
    with pytest.raises(ValueError, match="integer"):
        fast.materialize(cache, current, ids.double())
    expanded = LightweightValueState(current.codes[:, :1].expand_as(current.codes), 11)
    with pytest.raises(ValueError, match="non-overlapping"):
        fast.materialize(cache, expanded, ids)
    # A shared backing allocation would make later native reads depend on writes.
    backing = torch.zeros(2 * 2 * 11 * 48, dtype=cache.k.dtype)
    native_view = backing[:2 * 2 * 11 * 24].view(2, 2, 11, 24)
    alias_cache = HSTUKVCache(native_view, cache.v, 11)
    alias_state = LightweightValueState(backing.view(2, 2, 11, 6, 8), 11)
    with pytest.raises(ValueError, match="must not alias"):
        fast.materialize(alias_cache, alias_state, ids)


def test_grad_enabled_call_is_rejected():
    _, actor, cache, catalog, ids = example()
    fast = InplaceReleasePrefix.wrap(actor.compile_release(None, catalog))
    with torch.no_grad():
        current = actor.encode_initial(cache, catalog[ids])
    with pytest.raises(RuntimeError, match="inference-only"):
        fast.materialize(cache, current, ids)


@torch.no_grad()
def test_old_session_accepts_inplace_prefix_across_publish_and_fifo(monkeypatch):
    # Reuse the existing causal trajectory fixture, not a second service model.
    import importlib.util
    path = ROOT / "tests/query_relay_2026_10/test_session.py"
    spec = importlib.util.spec_from_file_location("relay_session_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from query_relay_2026_10.session import RelaySession
    models, actors, _, make = module.fixtures(monkeypatch, mode="A")
    reference, fast = RelaySession(make()), RelaySession(make())
    prefix = None
    for index, at in enumerate((40, 60, 80)):
        model, actor = models[index + 1], actors[index]
        catalog = model.lookup_item_embeddings(torch.arange(49)[None])[0]
        prefix = StableReleasePrefix.wrap(actor.compile_release(prefix, catalog))
        reference.publish(model, index + 3, at, actor, prefix=prefix)
        fast.publish(model, index + 3, at, actor,
                     prefix=InplaceReleasePrefix.wrap(prefix, row_chunk=2))
        ptr = fast.state.codes.data_ptr()
        a = reference.score(at, torch.tensor([[30, 31]]))
        b = fast.score(at, torch.tensor([[30, 31]]))
        assert fast.state.codes.data_ptr() == ptr
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
        reference.advance(at + 1)
        fast.advance(at + 1)
        torch.testing.assert_close(reference.cache.k, fast.cache.k, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(reference.cache.v, fast.cache.v, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(reference.state.codes, fast.state.codes, atol=3e-6, rtol=3e-5)
