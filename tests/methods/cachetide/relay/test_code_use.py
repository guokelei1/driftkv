"""Read-only code controls retain the writer's native cache and zero suffix."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]

from tests.methods.cachetide.relay.test_session import fixtures
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.methods.cachetide.compact.lightweight import LightweightValueState
from hstu_kvcache.methods.cachetide.adapter import item_gate_query
from experiments.archive.query_relay.diagnose_code_use import code_payload_state
from hstu_kvcache.methods.cachetide.session import RelaySession


def mixed_session(monkeypatch):
    models, adapters, _, make = fixtures(monkeypatch, mode="A")
    adapter = adapters[0]
    adapter.set_gate_source("item")
    session = RelaySession(make())
    session.publish(models[1], 3, 40, adapter)
    session.advance(51)
    return models[1], adapter, session


@torch.no_grad()
def test_payload_controls_are_independent_and_only_replace_inherited_prefix(monkeypatch):
    _, adapter, session = mixed_session(monkeypatch)
    live = session.state.codes.clone()
    mean = torch.arange(adapter.num_layers * 48, dtype=live.dtype).reshape(-1, 48) / 100
    control = code_payload_state(session.state, mean, session.coded_rows)
    assert session.coded_rows == 2 and control.seq_len == 4
    assert control.codes.shape == live.shape
    assert control.codes.data_ptr() != session.state.codes.data_ptr()
    for row in range(session.coded_rows):
        torch.testing.assert_close(control.codes[:, 0, row].flatten(-2), mean, atol=0, rtol=0)
    assert not torch.count_nonzero(control.codes[:, :, session.coded_rows:])
    control.codes.add_(10)
    torch.testing.assert_close(session.state.codes, live, atol=0, rtol=0)
    zero = code_payload_state(session.state)
    assert not torch.count_nonzero(zero.codes)
    assert zero.seq_len == session.state.seq_len


@torch.no_grad()
def test_zero_auxiliary_read_matches_native_on_the_same_corrected_writer_cache(monkeypatch):
    model, adapter, session = mixed_session(monkeypatch)
    candidates = torch.tensor([[30, 31, 32]])
    before = session.cache.k.clone(), session.cache.v.clone(), session.state.codes.clone()
    native = score(model, session.cache, candidates, torch.tensor([11.]))[0]
    reader = adapter.make_history_read(code_payload_state(session.state), code_count=0,
                                      gate_query=item_gate_query(model, candidates))
    auxiliary_zero = score(model, session.cache, candidates, torch.tensor([11.]),
                           history_read_fn=reader)[0]
    torch.testing.assert_close(auxiliary_zero, native, atol=0, rtol=0)
    for actual, expected in zip((session.cache.k, session.cache.v, session.state.codes), before):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@torch.no_grad()
def test_global_mean_reader_matches_literal_prefix_rows_after_fifo(monkeypatch):
    model, adapter, session = mixed_session(monkeypatch)
    candidates = torch.tensor([[30, 31, 32]])
    mean = torch.linspace(-.8, .9, adapter.num_layers * 48).reshape(-1, 48)
    gate_query = item_gate_query(model, candidates)
    literal = torch.zeros_like(session.state.codes)
    for layer in range(adapter.num_layers):
        for row in range(session.coded_rows):
            literal[layer, 0, row] = mean[layer].reshape(6, 8)
    expected = score(model, session.cache, candidates, torch.tensor([11.]),
        history_read_fn=adapter.make_history_read(LightweightValueState(literal, 4),
            code_count=2, gate_query=gate_query))[0]
    actual = score(model, session.cache, candidates, torch.tensor([11.]),
        history_read_fn=adapter.make_history_read(
            code_payload_state(session.state, mean, session.coded_rows),
            code_count=session.coded_rows, gate_query=gate_query))[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Once the inherited prefix has left, the same mean becomes a native read.
    session.advance(81)
    assert session.coded_rows == 0
    retired = code_payload_state(session.state, mean, session.coded_rows)
    assert not torch.count_nonzero(retired.codes)
    native = score(model, session.cache, candidates, torch.tensor([11.]))[0]
    actual = score(model, session.cache, candidates, torch.tensor([11.]),
        history_read_fn=adapter.make_history_read(retired, code_count=0,
                                                  gate_query=gate_query))[0]
    torch.testing.assert_close(actual, native, atol=0, rtol=0)
