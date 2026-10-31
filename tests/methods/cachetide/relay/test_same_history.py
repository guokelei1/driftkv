"""Same-candidate gate means remove within-history variation without writes."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]

from hstu_kvcache.methods.cachetide.structure import StructuredKVAdapter
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from hstu_kvcache.methods.cachetide.adapter import UnifiedRelayAdapter, item_gate_query
from experiments.archive.query_relay.diagnose_same_history import (
    WithinHistoryGates, same_set_mean_reader, score_errors,
)


def fixture(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(1715)
    model = HSTU(HSTUConfig(num_items=32, num_behaviors=2, hidden_size=24, num_layers=2,
        num_heads=6, max_seq_len=8, input_dropout=0., attn_dropout=0.)).double().eval()
    source = StructuredKVAdapter(2, 6, 4, max_length=8, layout="shared12_private6").double()
    with torch.no_grad():
        for layer in source.layers:
            layer.input_weight.normal_(std=.1)
            layer.decoder.normal_(std=.15)
            layer.key_decoder.normal_(std=.15)
    adapter = UnifiedRelayAdapter.from_structured(source, mode="legacy", read_shared=3,
        read_private=3, gate_normalization="rms").eval()
    with torch.no_grad():
        for layer in adapter.layers:
            layer.key_gate_weight.normal_(std=.3)
            layer.value_gate_weight.normal_(std=.3)
    cache = HSTUKVCache(torch.randn(2, 1, 5, 24, dtype=torch.float64),
                       torch.randn(2, 1, 5, 24, dtype=torch.float64), 5)
    state = adapter.encode_initial(cache, torch.randn(1, 5, 24, dtype=torch.float64))
    return model, adapter, cache, state


@torch.no_grad()
def test_reader_matches_literal_same_set_mean_and_single_candidate_dynamic(monkeypatch):
    model, adapter, cache, state = fixture(monkeypatch)
    q = torch.randn(1, 6, 16, 4, dtype=torch.float64)
    reader = same_set_mean_reader(adapter, state, code_count=3)
    means = tuple(value.mean(2, keepdim=True) for value in adapter.layers[0].query_gates(q))
    expected = adapter.layers[0].read(model.blocks[0].attn, q, cache.k[0], cache.v[0],
                                     state.codes[0], code_count=3, gate_constants=means)
    actual = reader(0, model.blocks[0].attn, q, cache.k[0], cache.v[0])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    k, v, codes = cache.k.clone(), cache.v.clone(), state.codes.clone()
    candidates, delta = torch.tensor([[6]]), torch.tensor([10.], dtype=torch.float64)
    dynamic = score(model, cache, candidates, delta,
                    history_read_fn=adapter.make_history_read(state, code_count=3))[0]
    mean = score(model, cache, candidates, delta, history_read_fn=reader)[0]
    torch.testing.assert_close(dynamic, mean, rtol=0, atol=1e-12)
    torch.testing.assert_close(cache.k, k, rtol=0, atol=0)
    torch.testing.assert_close(cache.v, v, rtol=0, atol=0)
    torch.testing.assert_close(state.codes, codes, rtol=0, atol=0)


@torch.no_grad()
def test_statistics_do_not_confuse_between_histories_with_within_set_queries(monkeypatch):
    _, adapter, _, _ = fixture(monkeypatch)
    adapter.layers = torch.nn.ModuleList([adapter.layers[0]])
    stats = WithinHistoryGates()
    # Two different history/time conditions, each with identical candidate Qs.
    q1 = torch.randn(1, 6, 1, 4, dtype=torch.float64).expand(-1, -1, 16, -1)
    q2 = torch.randn(1, 6, 1, 4, dtype=torch.float64).expand(-1, -1, 16, -1)
    stats.add("v3", adapter, [q1], code_rows=3)
    stats.add("v3", adapter, [q2], code_rows=3)
    stats.add("v3", adapter, [torch.randn_like(q1)], code_rows=0)
    for row in stats.result()["rows"]:
        assert row["active_sets"] == 2 and row["inactive_sets"] == 1
        assert row["rms_within_set_coordinate_std"] < 1e-15
        assert row["rms_between_set_mean_std"] > .01
        assert row["nonconstant_coordinate_set_fraction"] == 0
    errors = score_errors(torch.tensor([12., 13., 15.]), torch.tensor([2., 3., 5.]))
    assert errors["logit_mse"] == 100
    assert errors["centered_logit_mse"] == 0


@torch.no_grad()
def test_first_source_shadow_uses_this_operations_q0_and_preserves_layer_query(monkeypatch):
    model, adapter, cache, state = fixture(monkeypatch)
    adapter.set_gate_source("first")
    reader = same_set_mean_reader(adapter, state, code_count=3)
    q0 = torch.randn(1, 6, 16, 4, dtype=torch.float64)
    q1 = torch.randn_like(q0) * 10
    reader(0, model.blocks[0].attn, q0, cache.k[0], cache.v[0])
    actual = reader(1, model.blocks[1].attn, q1, cache.k[1], cache.v[1])
    means = tuple(gate.mean(2, keepdim=True) for gate in adapter.layers[1].query_gates(q0))
    wanted = adapter.layers[1].read(model.blocks[1].attn, q1, cache.k[1], cache.v[1],
        state.codes[1], code_count=3, gate_constants=means)
    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    # Reusing a reader for another request must replace its q0, not keep the
    # previous operation or a dynamic branch's transient query.
    next_q0 = torch.randn_like(q0)
    reader(0, model.blocks[0].attn, next_q0, cache.k[0], cache.v[0])
    next_actual = reader(1, model.blocks[1].attn, q1, cache.k[1], cache.v[1])
    next_means = tuple(gate.mean(2, keepdim=True) for gate in adapter.layers[1].query_gates(next_q0))
    next_wanted = adapter.layers[1].read(model.blocks[1].attn, q1, cache.k[1], cache.v[1],
        state.codes[1], code_count=3, gate_constants=next_means)
    torch.testing.assert_close(next_actual, next_wanted, rtol=0, atol=0)
    assert float((actual - next_actual).abs().max()) > 1e-5
    # Statistics must use q0 for the upper layer as well.
    stats = WithinHistoryGates()
    stats.add("v3", adapter, [q0, q1], code_rows=3)
    row = next(row for row in stats.result()["rows"] if row["layer"] == 1 and row["side"] == "key")
    expected_std = adapter.layers[1].query_gates(q0)[0][0].std(1, unbiased=False).mean()
    assert abs(row["mean_within_set_coordinate_std"] - float(expected_std)) < 1e-12


@torch.no_grad()
def test_item_source_shadow_and_statistics_use_exact_candidate_embeddings(monkeypatch):
    model, adapter, cache, state = fixture(monkeypatch)
    adapter.set_gate_source("item")
    candidates = torch.arange(1, 17)[None]
    gate_query = item_gate_query(model, candidates)
    expected = model.lookup_item_embeddings(candidates).reshape(1, 16, 6, 4).transpose(1, 2)
    torch.testing.assert_close(gate_query, expected, rtol=0, atol=0)
    native_q = torch.randn_like(gate_query) * 20
    reader = same_set_mean_reader(adapter, state, 3, gate_query=gate_query)
    means = tuple(value.mean(2, keepdim=True) for value in adapter.layers[1].query_gates(gate_query))
    actual = reader(1, model.blocks[1].attn, native_q, cache.k[1], cache.v[1])
    wanted = adapter.layers[1].read(model.blocks[1].attn, native_q, cache.k[1], cache.v[1],
        state.codes[1], code_count=3, gate_constants=means)
    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    stats = WithinHistoryGates()
    stats.add("v3", adapter, [native_q, native_q], 3, gate_query=gate_query)
    for row in stats.result()["rows"]:
        side = 0 if row["side"] == "key" else 1
        standard = adapter.layers[row["layer"]].query_gates(gate_query)[side][0].std(1, unbiased=False).mean()
        assert abs(row["mean_within_set_coordinate_std"] - float(standard)) < 1e-12
    # No fallback to this layer's contextual query is permitted.
    import pytest
    with pytest.raises(ValueError, match="Current item embedding"):
        adapter.make_history_read(state)
    with pytest.raises(ValueError, match="Current item embedding"):
        adapter.layers[1].read(model.blocks[1].attn, native_q, cache.k[1], cache.v[1], state.codes[1])
