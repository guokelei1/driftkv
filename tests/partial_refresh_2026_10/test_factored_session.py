"""Shared affine preparation and user lifecycle have distinct exact ledgers."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from hstu_kvcache.design_one.lightweight_fullitem_kv import LightweightFullItemKVAdapter
from hstu_kvcache.design_two.replay import NativeTrajectory
from hstu_kvcache.models import HSTU, HSTUConfig
from partial_refresh_2026_10.round2.factored_session import FactoredRefreshSession
from partial_refresh_2026_10.session import PartialRefreshSession


@pytest.fixture
def fixtures(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    FactoredRefreshSession.shared_encoders.clear()
    torch.manual_seed(20261014)
    cfg = HSTUConfig(num_items=96, num_behaviors=2, hidden_size=8, num_layers=3,
                     num_heads=2, max_seq_len=6, input_dropout=0., attn_dropout=0.)
    models = [HSTU(cfg).eval().requires_grad_(False) for _ in range(4)]
    adapters = [LightweightFullItemKVAdapter(3, 2, 4, rank=2, max_length=6)
                .eval().requires_grad_(False) for _ in range(3)]
    with torch.no_grad():
        for adapter in adapters:
            for layer in adapter.layers:
                layer.input_weight.normal_(std=.05)
                layer.input_bias.normal_(std=.04)
                layer.input_center.normal_(std=.1)
                layer.input_scale.uniform_(.6, 1.6)
                layer.decoder.normal_(std=.04)
                layer.key_decoder.normal_(std=.02)
    events = (np.arange(48), np.arange(1, 49), np.zeros(48, dtype=np.int64))
    static_file = tmp_path / "fit_only_static_ids.npz"
    np.savez(static_file, v3_2=np.array([1, 2]), v4_2=np.array([2, 3]),
             v5_2=np.array([3, 4]))

    def make():
        return NativeTrajectory(models[0], 2, events, 6, max_length=6, append_band_size=2)

    yield models, adapters, make, str(static_file)
    FactoredRefreshSession.shared_encoders.clear()


@pytest.mark.parametrize("mode,rho", [("masked", 1.), ("raw", .25)])
@torch.inference_mode()
def test_three_release_equivalence_native_state_and_row_cost_split(fixtures, mode, rho):
    models, adapters, make, static_file = fixtures
    factored = FactoredRefreshSession(make(), storage_mode=mode, exit_when_clear=False,
                                      cache_capacity=2, static_item_file=static_file)
    original = PartialRefreshSession(make(), storage_mode=mode, exit_when_clear=False)
    expected = Counter()
    for index, (release, first, later) in enumerate(((6, 8, 10), (11, 13, 15), (16, 17, 20))):
        for session in (factored, original):
            session.publish(models[index + 1], index + 3, release, adapters[index], rho=rho)
            session.advance(first)
        plan = factored.refresh_plan()
        expected["initial"] += plan["missing"]
        expected["refresh"] += plan["selected"] - plan["missing"]
        factored.prepare_read(first)
        original.prepare_read(first)
        for session in (factored, original):
            session.advance(later)
        expected["append"] += later - first
        ids = torch.tensor([[70, 71, 72]])
        torch.testing.assert_close(factored.state.codes, original.state.codes, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(factored.score(later, ids), original.score(later, ids),
                                   atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(factored.cache.k, original.cache.k, rtol=0, atol=0)
        torch.testing.assert_close(factored.cache.v, original.cache.v, rtol=0, atol=0)
        np.testing.assert_array_equal(factored.code_versions, original.code_versions)
        np.testing.assert_array_equal(factored.support.flags, original.support.flags)
        assert factored.trajectory.ledger == original.trajectory.ledger
        assert factored.state.storage_bytes() == original.state.storage_bytes()
        assert factored.memory == original.memory
    # Old-release native appends between later reads and the next publication.
    expected["append"] += 2
    assert factored.row_encoder_flops == 3 * 2 * (4 * 4 * 2 + 2 * 2)
    for role, rows in expected.items():
        assert factored.costs[f"{role}_encode_rows"] == rows
        assert factored.costs[f"{role}_encode_flops"] == rows * factored.row_encoder_flops
    assert factored.costs["initial_encode_flops"] >= 0
    assert factored.costs["refresh_encode_flops"] >= 0
    inventory = factored.encoder_inventory()
    assert len(inventory) == 3
    assert all(e["memory"]["shared_item_table_current_bytes"] == 2 * 3 * 2 * 2 * 4
               for e in inventory)
    assert all(e["memory"]["folded_parameter_bytes"] > 0 for e in inventory)
    assert sum(e["counters"]["row_kv_flops"] for e in inventory) == sum(
        factored.costs[f"{role}_encode_flops"] for role in expected)
    assert sum(e["counters"]["parameter_preparation_flops"] for e in inventory) == factored.costs["encoder_fold_preparation_flops"]
    assert sum(e["counters"]["item_projection_flops"] for e in inventory) == factored.costs["item_projection_flops"]
    before = Counter(factored.costs)
    factored.prepare_read(20)
    assert factored.costs == before


@torch.inference_mode()
def test_constructor_static_preparation_is_charged_once_across_users(fixtures):
    models, adapters, make, static_file = fixtures
    first = FactoredRefreshSession(make(), cache_capacity=2, static_item_file=static_file)
    second = FactoredRefreshSession(make(), cache_capacity=2, static_item_file=static_file)
    for session in (first, second):
        session.publish(models[1], 3, 6, adapters[0])
    encoder = first._encoder()
    width = adapters[0].num_layers * adapters[0].heads * adapters[0].rank
    item_cost = (2 * adapters[0].heads * adapters[0].head_dim + 1) * width
    fold_cost = 3 * sum(layer.input_weight.numel() for layer in adapters[0].layers)
    # Shared constructor work exists and is charged before any user encoding.
    assert first.costs["encoder_fold_preparation_flops"] == fold_cost
    assert first.costs["item_projection_flops"] == 2 * item_cost
    assert first.costs["static_precomputed_items"] == 2
    assert first.costs["initial_encode_flops"] == 0
    before = Counter(first.costs)
    assert first._encoder() is second._encoder() is encoder
    assert first.costs == before
    assert second.costs["encoder_fold_preparation_flops"] == 0
    assert second.costs["static_precomputed_items"] == 0
    first.prepare_read(6)
    assert first.costs["encoder_fold_preparation_flops"] == fold_cost
    assert first.costs["item_projection_flops"] == 6 * item_cost  # 2 static + 4 misses.
    second.prepare_read(6)
    # Nonadmitted items remain misses; static preparation does not recur.
    assert second.costs["item_projection_flops"] == 4 * item_cost
    assert second.costs["hits"] == first.costs["hits"] == 2
    assert second.costs["misses"] == first.costs["misses"] == 4
    assert len(FactoredRefreshSession.shared_encoders) == 1
    assert encoder.counters["parameter_preparation_flops"] == fold_cost
    assert encoder.counters["item_projection_flops"] == (
        first.costs["item_projection_flops"] + second.costs["item_projection_flops"])
    assert encoder.counters["static_precomputed_items"] == 2
    assert encoder.counters["misses"] == first.costs["misses"] + second.costs["misses"]


@torch.inference_mode()
def test_nocache_miss_deltas_do_not_double_count_and_inventory_is_separate(fixtures):
    models, adapters, make, _ = fixtures
    session = FactoredRefreshSession(make(), cache_capacity=0)
    baseline = PartialRefreshSession(make())
    for candidate in (session, baseline):
        candidate.publish(models[1], 3, 6, adapters[0])
        candidate.prepare_read(6)
    encoder = session._encoder()
    original_items = session.trajectory.retained_events[1]
    fold_before = session.costs["encoder_fold_preparation_flops"]
    before = Counter(session.costs)
    state = session._encode_cache(session.cache, original_items, session.support.flags,
                                  "append_encode_flops")
    item_cost = (2 * adapters[0].heads * adapters[0].head_dim + 1) * (
        adapters[0].num_layers * adapters[0].heads * adapters[0].rank)
    assert session.costs["item_projection_flops"] - before["item_projection_flops"] == 6 * item_cost
    assert session.costs["misses"] - before["misses"] == 6
    assert session.costs["hits"] - before["hits"] == 0
    assert session.costs["append_encode_flops"] - before["append_encode_flops"] == 6 * session.row_encoder_flops
    assert session.costs["encoder_fold_preparation_flops"] == fold_before
    torch.testing.assert_close(state.codes, baseline.state.codes, atol=2e-6, rtol=2e-5)
    assert session.memory == baseline.memory
    assert session.state.storage_bytes() == baseline.state.storage_bytes()
    inventory = session.encoder_inventory()
    assert len(inventory) == 1
    assert inventory[0]["memory"]["folded_parameter_bytes"] > 0
    assert inventory[0]["memory"]["shared_item_table_current_bytes"] == 0
    assert encoder.counters["row_kv_flops"] == (
        session.costs["initial_encode_flops"] + session.costs["append_encode_flops"])
    # Reads reuse the user codes and cannot incur another encoder/cache miss.
    before = Counter(session.costs)
    session.score(6, torch.tensor([[70, 71]]))
    for key in ("initial_encode_flops", "refresh_encode_flops", "append_encode_flops",
                "item_projection_flops", "encoder_fold_preparation_flops", "hits", "misses"):
        assert session.costs[key] == before[key]
