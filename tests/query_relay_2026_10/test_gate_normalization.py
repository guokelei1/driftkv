"""RMS gate input retains raw native Q and a genuinely nonlinear reader."""
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from code_structure_2026_10.structure import StructuredKVAdapter
from hstu_kvcache.models import HSTUKVCache, PointwiseAttention, PointwiseAttentionConfig
from query_relay_2026_10.adapter import UnifiedRelayAdapter


def fixture():
    torch.manual_seed(20261014)
    source = StructuredKVAdapter(1, 6, 4, layout="shared12_private6").double().eval()
    with torch.no_grad():
        source.layers[0].input_weight.normal_(std=.05)
        source.layers[0].input_bias.normal_(std=.03)
        source.layers[0].decoder.normal_(std=.04)
        source.layers[0].key_decoder.normal_(std=.04)
    cache = HSTUKVCache(torch.randn(1, 1, 5, 24, dtype=torch.float64),
                       torch.randn(1, 1, 5, 24, dtype=torch.float64), 5)
    items = torch.randn(1, 5, 24, dtype=torch.float64)
    q = torch.randn(1, 6, 3, 4, dtype=torch.float64)
    attention = PointwiseAttention(PointwiseAttentionConfig(24, 6, head_dim=4)).double().eval()
    return source, cache, items, q, attention


@pytest.mark.parametrize("normalization", ["raw", "rms"])
def test_zero_initialized_gate_preserves_original_read(normalization):
    source, cache, items, q, attention = fixture()
    adapter = UnifiedRelayAdapter.from_structured(source, mode="legacy",
                                                   gate_normalization=normalization)
    state = adapter.encode_initial(cache, items)
    for gate in adapter.layers[0].query_gates(q * 1000):
        torch.testing.assert_close(gate, torch.ones_like(gate), rtol=0, atol=0)
    actual = adapter.make_history_read(state)(0, attention, q, cache.k[0], cache.v[0])
    wanted = source.layers[0].read(attention, q, cache.k[0], cache.v[0], state.codes[0])
    torch.testing.assert_close(actual, wanted, rtol=2e-12, atol=2e-12)
    assert adapter.get_config()["gate_normalization"] == normalization


def test_rms_gate_uses_normalized_q_but_native_projection_uses_raw_q():
    source, cache, items, q, attention = fixture()
    adapter = UnifiedRelayAdapter.from_structured(source, mode="legacy", gate_normalization="rms")
    layer = adapter.layers[0]
    with torch.no_grad():
        layer.key_gate_weight.normal_(std=.3)
        layer.value_gate_weight.normal_(std=.3)
    state = adapter.encode_initial(cache, items)
    normalized = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)
    key_gate = 1 + torch.tanh(normalized @ layer.key_gate_weight[None])
    value_gate = 1 + torch.tanh(normalized @ layer.value_gate_weight[None])
    gk, gv = layer.query_gates(q)
    torch.testing.assert_close(gk, key_gate, rtol=0, atol=0)
    torch.testing.assert_close(gv, value_gate, rtol=0, atol=0)
    z = layer.accessible_codes(state.codes[0])
    effective_key = torch.einsum("bhnr,bhqr,hrd->bhqnd", z, key_gate, layer.key_decoder)
    effective_value = torch.einsum("bhnr,bhqr,hrd->bhqnd", z, value_gate, layer.decoder)
    keys = cache.k[0].reshape(1, 5, 6, 4).transpose(1, 2)
    values = cache.v[0].reshape(1, 5, 6, 4).transpose(1, 2)
    weights = attention._activate(torch.einsum("bhqd,bhqnd->bhqn", q,
                                               keys[:, :, None] + effective_key) * attention.scale)
    expected = torch.einsum("bhqn,bhqnd->bhqd", weights, values[:, :, None] + effective_value)
    actual = layer.read(attention, q, cache.k[0], cache.v[0], state.codes[0])
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)


@torch.no_grad()
def test_rms_gate_retains_nonlinear_key_response_and_resists_magnitude_saturation():
    source, _, _, _, _ = fixture()
    adapter = UnifiedRelayAdapter.from_structured(source, mode="legacy", gate_normalization="rms")
    layer = adapter.layers[0]
    with torch.no_grad():
        layer.key_gate_weight.zero_()
        layer.key_gate_weight[:, 0, 0] = .25
        layer.key_decoder.zero_()
        layer.key_decoder[:, 0, 0] = 1
    q1 = torch.zeros(1, 6, 1, 4, dtype=torch.float64)
    q2 = torch.zeros_like(q1)
    q1[..., 0] = 2
    q2[..., 1] = 1

    def phi(q):
        return (q @ layer.key_decoder.transpose(-2, -1)[None]) * layer.query_gates(q)[0]

    assert float((phi(q1 + q2) - phi(q1) - phi(q2)).abs().max()) > .01
    # A directional change affects the gate, while simple norm inflation does
    # not force the gate toward 0/2. This is a numerical witness, not quality.
    base = layer.query_gates(q1)[0]
    large = layer.query_gates(q1 * 10000)[0]
    torch.testing.assert_close(base, large, rtol=0, atol=2e-7)
    assert float(large.max()) < 1.5


@pytest.mark.parametrize("shared,private", [(3, 3), (0, 3), (6, 3), (12, 3)])
def test_balanced_read_width_matches_masked_full_decoder_without_extra_state(shared, private):
    source, cache, items, q, attention = fixture()
    full = UnifiedRelayAdapter.from_structured(source, mode="legacy", gate_normalization="rms")
    with torch.no_grad():
        full.layers[0].key_gate_weight.normal_(std=.3)
        full.layers[0].value_gate_weight.normal_(std=.3)
    narrow, oracle = copy.deepcopy(full), copy.deepcopy(full)
    narrow.set_read_width(shared, private)
    kept = list(range(shared)) + list(range(12, 12 + private))
    omitted = [slot for slot in range(18) if slot not in kept]
    with torch.no_grad():
        oracle.layers[0].decoder[:, omitted] = 0
        oracle.layers[0].key_decoder[:, omitted] = 0
    state = full.encode_initial(cache, items)
    expected = oracle.make_history_read(state)(0, attention, q, cache.k[0], cache.v[0])
    actual = narrow.make_history_read(state)(0, attention, q, cache.k[0], cache.v[0])
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    assert state.codes.shape == (1, 1, 5, 6, 8)
    assert narrow.layers[0].read_slots.tolist() == kept
    assert narrow.accessible_rank == shared + private
    assert narrow.get_config()["read_private"] == private
    restored = UnifiedRelayAdapter(**narrow.get_config()).double()
    restored.load_state_dict(narrow.state_dict())
    assert restored.layers[0].read_slots.tolist() == kept
    restored.set_read_shared(3)
    assert restored.read_private == private


def test_old_configuration_defaults_to_raw_gate_and_six_private_slots():
    source, _, _, _, _ = fixture()
    adapter = UnifiedRelayAdapter.from_structured(source, mode="legacy")
    config = adapter.get_config()
    del config["gate_normalization"], config["read_private"]
    restored = UnifiedRelayAdapter(**config).double()
    restored.load_state_dict(adapter.state_dict())
    assert restored.gate_normalization == "raw"
    assert restored.read_private == 6
    assert restored.layers[0].read_slots.tolist() == list(range(18))
