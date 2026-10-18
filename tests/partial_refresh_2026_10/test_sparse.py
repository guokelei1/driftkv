"""Sparse support encoding preserves actual masked-state and native semantics."""
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
from partial_refresh_2026_10.round2.sparse_session import SparseMaskedSession
from partial_refresh_2026_10.session import PartialRefreshSession


@pytest.fixture
def fixtures(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
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
                layer.decoder.normal_(std=.04)
                layer.key_decoder.normal_(std=.02)
    events = (np.arange(48), np.arange(1, 49), np.zeros(48, dtype=np.int64))

    def make():
        return NativeTrajectory(models[0], 2, events, 6, max_length=6, append_band_size=2)
    return models, adapters, make


@torch.inference_mode()
def test_mixed_support_encodes_only_true_layer_rows(fixtures, monkeypatch):
    models, adapters, make = fixtures
    sparse, full = SparseMaskedSession(make()), PartialRefreshSession(make())
    sparse.publish(models[1], 3, 6, adapters[0])
    full.publish(models[1], 3, 6, adapters[0])
    flags = np.array([[False] * 6, [True, False, True, False, False, True], [True] * 6])
    visited = Counter()
    for i, layer in enumerate(adapters[0].layers):
        original = layer.encode

        def record(k, v, item, i=i, original=original):
            visited[i] += k.shape[1]
            return original(k, v, item)
        monkeypatch.setattr(layer, "encode", record)
    items = sparse.trajectory.retained_events[1]
    actual = sparse._encode_cache(sparse.cache, items, flags, "append_encode_flops")
    assert visited == {1: 3, 2: 6}
    expected = full._encode_cache(full.cache, items, flags, None)
    torch.testing.assert_close(actual.codes, expected.codes, atol=2e-6, rtol=2e-5)
    assert not actual.codes[0].any()
    assert sparse.costs["actual_encoded_layer_rows"] == 9
    assert sparse.costs["eligible_layer_rows"] == 18
    assert sparse.costs["skipped_encoder_layer_rows"] == 9
    assert sparse.costs["append_encode_flops"] == 9 * sparse.layer_row_encoder_flops
    assert sparse.costs["support_mask_flops"] == sparse.costs["support_mask_transfer_bytes"] == 0
    before = Counter(sparse.costs)
    zeros = sparse._encode_cache(sparse.cache, items, np.zeros_like(flags), None)
    assert not zeros.codes.any() and sparse.costs == before


@torch.inference_mode()
@pytest.mark.parametrize("rho", [1., .25])
def test_sparse_lifecycle_matches_masked_scores_and_correct_cost_split(fixtures, rho):
    models, adapters, make = fixtures
    sparse = SparseMaskedSession(make(), exit_when_clear=False)
    full = PartialRefreshSession(make(), exit_when_clear=False)
    candidates = torch.tensor([[70, 71]])
    expected_layer_rows = Counter()
    eligible = 0
    for index, (release, first, later) in enumerate(((6, 8, 10), (11, 13, 15), (16, 17, 20))):
        for session in (sparse, full):
            session.publish(models[index + 1], index + 3, release, adapters[index], rho=rho)
            session.advance(first)
        plan = sparse.refresh_plan()
        missing = plan["indices"][sparse.code_versions[plan["indices"]] == -1]
        initial = int(sparse.support.flags[:, missing].sum())
        total = int(sparse.support.flags[:, plan["indices"]].sum())
        expected_layer_rows["initial"] += initial
        expected_layer_rows["refresh"] += total - initial
        eligible += sparse.adapter.num_layers * len(plan["indices"])
        sparse.prepare_read(first)
        full.prepare_read(first)
        for session in (sparse, full):
            session.advance(later)
        # Active appends in this small fixture are below the layer1 clearance
        # threshold: each arriving row's two upper layers have support.
        expected_layer_rows["append"] += 2 * (later - first)
        eligible += 3 * (later - first)
        torch.testing.assert_close(sparse.state.codes, full.state.codes, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(sparse.score(later, candidates), full.score(later, candidates),
                                   atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(sparse.cache.k, full.cache.k, rtol=0, atol=0)
        torch.testing.assert_close(sparse.cache.v, full.cache.v, rtol=0, atol=0)
        np.testing.assert_array_equal(sparse.code_versions, full.code_versions)
        np.testing.assert_array_equal(sparse.support.flags, full.support.flags)
        assert sparse.trajectory.ledger == full.trajectory.ledger
        assert sparse.state.storage_bytes() == full.state.storage_bytes()
        assert sparse.memory == full.memory
    # Old-release appends between 'later' and the next publish also use sparse
    # encoders. They comprise one row per transition, two upper layers each.
    expected_layer_rows["append"] += 4
    eligible += 6
    for key, count in expected_layer_rows.items():
        assert sparse.costs[f"{key}_encoded_layer_rows"] == count
        assert sparse.costs[f"{key}_encode_flops"] == count * sparse.layer_row_encoder_flops
    assert sparse.costs["actual_encoded_layer_rows"] == sum(expected_layer_rows.values())
    assert sparse.costs["eligible_layer_rows"] == eligible
    assert sparse.costs["support_mask_flops"] == 0
    assert sparse.costs["initial_encode_flops"] >= 0 and sparse.costs["refresh_encode_flops"] >= 0
    before = Counter(sparse.costs)
    sparse.prepare_read(20)
    assert sparse.costs == before


@torch.inference_mode()
def test_sparse_clearance_and_rebuild_match_masked(fixtures):
    models, adapters, make = fixtures
    sparse, full = SparseMaskedSession(make()), PartialRefreshSession(make())
    for session in (sparse, full):
        session.publish(models[1], 3, 6, adapters[0])
        session.prepare_read(6)
        session.advance(40)
    assert sparse.support.clear and sparse.native_mode and len(sparse.code_versions) == 0
    torch.testing.assert_close(sparse.score(40, torch.tensor([[70]])),
                               full.score(40, torch.tensor([[70]])))
    for session in (sparse, full):
        session.publish(models[2], 4, 40, adapters[1])
        session.rebuild(41)
    assert sparse.state is None and sparse.native_mode and not sparse.pending_encode
    torch.testing.assert_close(sparse.cache.k, full.cache.k, rtol=0, atol=0)
    assert sparse.trajectory.ledger == full.trajectory.ledger
