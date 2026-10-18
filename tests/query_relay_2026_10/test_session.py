"""Causal corrected writes, code FIFO, and cold-release composition checks."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from code_structure_2026_10.structure import StructuredKVAdapter
from hstu_kvcache.design_one.lightweight import LightweightValueState
from hstu_kvcache.design_two.replay import NativeTrajectory
from hstu_kvcache.models import HSTU, HSTUConfig
from query_relay_2026_10.adapter import UnifiedRelayAdapter
from query_relay_2026_10.session import RelaySession, append_corrected_event


def fixtures(monkeypatch, mode="B", *, cold=False):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(20261014)
    cfg = HSTUConfig(num_items=48, num_behaviors=2, hidden_size=12, num_layers=3,
                     num_heads=6, max_seq_len=4, input_dropout=0., attn_dropout=0.)
    initial = HSTU(cfg).eval().requires_grad_(False)
    models = [initial]
    for _ in range(3):
        model = copy.deepcopy(models[-1])
        with torch.no_grad():
            model.item_emb.weight.add_(torch.randn_like(model.item_emb.weight) * .01)
        models.append(model)
    source = StructuredKVAdapter(3, 6, 2, max_length=4,
                                  layout="shared12_private6").eval()
    with torch.no_grad():
        for layer in source.layers:
            layer.input_weight.normal_(std=.03)
            layer.input_bias.normal_(std=.02)
            layer.decoder.normal_(std=.03)
            layer.key_decoder.normal_(std=.03)
    adapters = [UnifiedRelayAdapter.from_structured(source, mode=mode).eval()
                for _ in range(3)]
    with torch.no_grad():
        for index, adapter in enumerate(adapters):
            for layer in adapter.layers:
                layer.key_gate_weight.normal_(std=.03)
                layer.value_gate_weight.normal_(std=.03)
                if mode in ("A", "B"):
                    for rotation in layer.rotation:
                        rotation.normal_(std=.01)
                    layer.item_update_bias.add_(.01 * index)
    times = np.arange(20) * 10
    if cold:
        times[4:] += 100
    events = (times, np.arange(1, 21), np.zeros(20, dtype=np.int64))

    def trajectory():
        return NativeTrajectory(models[0], 2, events, 40, max_length=4,
                                append_band_size=4)

    return models, adapters, events, trajectory


@torch.no_grad()
def test_first_append_prepares_and_matches_differentiable_sequential_oracle(monkeypatch):
    models, adapters, events, make = fixtures(monkeypatch)
    session = RelaySession(make())
    original = session.cache.k.clone(), session.cache.v.clone()
    session.publish(models[1], 3, 40, adapters[0])
    assert session.pending_encode
    # A scoring query at the release boundary prepares codes but writes no KV.
    session.score(40, torch.tensor([[30, 31]]))
    torch.testing.assert_close(session.cache.k, original[0], rtol=0, atol=0)
    torch.testing.assert_close(session.cache.v, original[1], rtol=0, atol=0)
    cache, state = session.cache, session.state
    expected, expected_state = cache, state
    for pos in range(4, 6):
        expected, expected_state, _ = append_corrected_event(
            models[1], expected, expected_state, adapters[0],
            torch.tensor([[events[1][pos]]]), torch.tensor([[0]]),
            torch.tensor([[10.]]), 4)
    session.advance(51)
    torch.testing.assert_close(session.cache.k, expected.k, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(session.cache.v, expected.v, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(session.state.codes, expected_state.codes, rtol=0, atol=0)
    assert session.coded_rows == 2
    assert not torch.count_nonzero(session.state.codes[:, :, -2:])
    # Surviving native rows retain their original values exactly.
    torch.testing.assert_close(session.cache.k[:, :, :2], original[0][:, :, -2:],
                               rtol=0, atol=0)
    assert session.trajectory.ledger["append", 3, 4, 2] == 1
    assert session.costs["writer_dense_code_pairs"] == 8
    assert session.costs["writer_old_code_valid_pairs"] == 5
    assert all("forward_with_cache" not in block.attn.__dict__ for block in models[1].blocks)


@torch.no_grad()
def test_event_only_interval_prepares_before_write_and_zero_suffix_clears(monkeypatch):
    models, adapters, _, make = fixtures(monkeypatch)
    session = RelaySession(make())
    session.publish(models[1], 3, 40, adapters[0])
    session.advance(41)  # No ranking request; the first behavior is a consumer.
    assert session.costs["explicit_update_calls"] == 1
    assert session.events[1]["action"] == "explicit_update"
    assert session.coded_rows == 3
    session.advance(81)
    assert session.coded_rows == 0
    assert not torch.count_nonzero(session.state.codes)
    codes = session.state.codes
    candidates = torch.tensor([[30, 31]])
    session.score(81, candidates)
    assert session.state.codes is codes
    session.publish(models[2], 4, 81, adapters[1])
    session.prepare_codes()
    assert session.coded_rows == session.cache.seq_len == 4
    assert torch.count_nonzero(session.state.codes)


@torch.no_grad()
def test_corrected_write_changes_upper_native_kv_and_restores_methods(monkeypatch):
    models, adapters, _, make = fixtures(monkeypatch)
    corrected = RelaySession(make())
    native = make()
    corrected.publish(models[1], 3, 40, adapters[0])
    native.switch(models[1], 3, 40)
    corrected.advance(41)
    native.advance(41)
    torch.testing.assert_close(corrected.cache.k[0], native.cache.k[0], rtol=0, atol=0)
    assert float((corrected.cache.k[1:, :, -1:] - native.cache.k[1:, :, -1:]).abs().max()) > 1e-7


@pytest.mark.parametrize("mode", ["A", "B"])
@torch.no_grad()
def test_unread_cold_user_direct_compilation_matches_sequential_updates(monkeypatch, mode):
    models, adapters, _, make = fixtures(monkeypatch, mode, cold=True)
    explicit, compiled = RelaySession(make()), RelaySession(make())
    prefix = None
    for index, at in enumerate((40, 50, 60)):
        model, adapter = models[index + 1], adapters[index]
        catalog = model.lookup_item_embeddings(torch.arange(49)[None])[0]
        prefix = adapter.compile_release(prefix, catalog)
        explicit.publish(model, index + 3, at, adapter)
        compiled.publish(model, index + 3, at, adapter, prefix=prefix)
    assert compiled.state_format == "anchor" and compiled.costs["compiled_prepare_calls"] == 0
    a = explicit.score(70, torch.tensor([[30, 31]]))
    b = compiled.score(70, torch.tensor([[30, 31]]))
    torch.testing.assert_close(a, b, rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(explicit.state.codes, compiled.state.codes, rtol=2e-5, atol=3e-6)
    assert explicit.costs["explicit_update_calls"] == 3
    assert compiled.costs["compiled_prepare_calls"] == 1
    assert compiled.memory["compact_peak_bytes"] == compiled.state.storage_bytes()


@torch.no_grad()
def test_compiled_anchor_retirement_preserves_newborn_zero_and_later_updates(monkeypatch):
    models, adapters, _, make = fixtures(monkeypatch)
    explicit, compiled = RelaySession(make()), RelaySession(make())
    prefix = None
    for index, at in enumerate((40, 60, 80)):
        model, adapter = models[index + 1], adapters[index]
        catalog = model.lookup_item_embeddings(torch.arange(49)[None])[0]
        prefix = adapter.compile_release(prefix, catalog)
        explicit.publish(model, index + 3, at, adapter)
        compiled.publish(model, index + 3, at, adapter, prefix=prefix)
        for session in (explicit, compiled):
            session.advance(at + 1)
        assert not torch.count_nonzero(compiled.state.codes[:, :, -1:])
        torch.testing.assert_close(explicit.state.codes, compiled.state.codes, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(explicit.cache.k, compiled.cache.k, rtol=3e-5, atol=3e-6)
    assert compiled.costs["anchorize_calls"] == 2


def test_event_oracle_keeps_gradients_through_corrected_history(monkeypatch):
    models, adapters, _, make = fixtures(monkeypatch)
    session = RelaySession(make())
    session.publish(models[1], 3, 40, adapters[0])
    session.prepare_codes()
    code = session.state.codes.detach().clone().requires_grad_()
    state = LightweightValueState(code, session.cache.seq_len)
    cache, _, _ = append_corrected_event(models[1], session.cache, state, adapters[0],
                                         torch.tensor([[5]]), torch.tensor([[0]]),
                                         torch.tensor([[10.]]), 4)
    loss = cache.k[1:, :, -1].square().sum() + cache.v[1:, :, -1].square().sum()
    loss.backward()
    assert code.grad is not None and float(code.grad.abs().sum()) > 0
    assert adapters[0].layers[0].key_gate_weight.grad is not None
    assert all(parameter.grad is None for parameter in models[1].parameters())


@torch.no_grad()
def test_chunked_subset_compilation_preserves_raw_item_mapping(monkeypatch):
    from types import SimpleNamespace
    from query_relay_2026_10.evaluate import compile_prefixes

    models, adapters, events, make = fixtures(monkeypatch)
    named_models = {f"v{i + 2}": model for i, model in enumerate(models)}
    named_adapters = {f"v{i + 3}": adapter for i, adapter in enumerate(adapters)}
    releases = (("v3", 40), ("v4", 60), ("v5", 80))
    compiled, records = compile_prefixes(named_models, named_adapters,
        SimpleNamespace(rows={7: events}), [7], releases, 100, chunk_size=3)
    full = None
    for version, _ in releases:
        full = named_adapters[version].compile_release(full,
            named_models[version].lookup_item_embeddings(torch.arange(49)[None])[0])
    cache = make().cache
    anchor = LightweightValueState(cache.k.new_zeros(3, 1, 4, 6, 8), 4)
    expected = full.materialize(cache, anchor, torch.tensor([[1, 2, 3, 4]]))
    actual = compiled["v5"].materialize(cache, anchor, np.array([1, 2, 3, 4]))
    torch.testing.assert_close(actual.codes, expected.codes, rtol=2e-5, atol=3e-6)
    assert records["v5"]["items"] == 10
