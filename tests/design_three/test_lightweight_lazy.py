"""Lazy code preparation preserves reads and native state across releases."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from design_system.evaluate_lightweight import LightweightSession
from design_system.lightweight_lazy import LazyLightweightSession
from hstu_kvcache.design_one.lightweight import LightweightValueAdapter
from hstu_kvcache.design_one.lightweight_fullitem import LightweightFullItemAdapter
from hstu_kvcache.design_two.replay import NativeTrajectory
from hstu_kvcache.design_three.lifecycle import InheritedSupport
from hstu_kvcache.models import HSTU, HSTUConfig


def fixtures(adapter_type, monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(20261004)
    cfg = HSTUConfig(num_items=64, num_behaviors=2, hidden_size=8, num_layers=3,
        num_heads=2, max_seq_len=4, input_dropout=0., attn_dropout=0.)
    models = [HSTU(cfg).eval().requires_grad_(False) for _ in range(3)]
    adapters = [adapter_type(3, 2, 4, rank=2, response_hidden_width=0,
        max_length=4).eval().requires_grad_(False) for _ in range(2)]
    for adapter in adapters:
        for layer in adapter.layers:
            layer.decoder.normal_(std=.03)
    events = (np.arange(32), np.arange(1, 33), np.zeros(32, dtype=np.int64))
    def trajectory():
        return NativeTrajectory(models[0], 2, events, 4, max_length=4, append_band_size=4)
    return models, adapters, trajectory


@pytest.mark.parametrize("adapter_type", [LightweightValueAdapter, LightweightFullItemAdapter])
@torch.inference_mode()
def test_first_read_and_later_appends_match_eager(adapter_type, monkeypatch):
    models, adapters, make = fixtures(adapter_type, monkeypatch)
    eager, lazy = LightweightSession(make(), kernel="torch"), LazyLightweightSession(make(), kernel="torch")
    candidates = torch.tensor([[40, 41]])
    for index, (release, first, later) in enumerate(((4, 7, 9), (10, 12, 14))):
        for session in (eager, lazy):
            session.publish(models[index+1], index+3, release, adapters[index])
            session.advance(first)
        assert lazy.pending_encode and lazy.state is None and not lazy.native_mode
        before = lazy.costs["initial_encode_flops"]
        actual, expected = lazy.score(first, candidates), eager.score(first, candidates)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(lazy.state.codes, eager.state.codes, atol=2e-6, rtol=2e-5)
        lazy.prepare_read(first)
        assert lazy.costs["initial_encode_flops"] - before == adapters[index].estimate_flops(tokens=4)["token_transform"]
        for session in (eager, lazy):
            session.advance(later)
        torch.testing.assert_close(lazy.score(later, candidates), eager.score(later, candidates), atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(lazy.cache.k, eager.cache.k, atol=0, rtol=0)
        torch.testing.assert_close(lazy.cache.v, eager.cache.v, atol=0, rtol=0)
        np.testing.assert_array_equal(lazy.support.flags, eager.support.flags)
        assert lazy.trajectory.ledger == eager.trajectory.ledger
    assert lazy.costs["first_read_code_initializations"] == 2
    assert lazy.costs["append_encode_flops"] < eager.costs["append_encode_flops"]


@torch.inference_mode()
def test_unread_release_rebinding_and_causal_clearance(monkeypatch):
    models, adapters, make = fixtures(LightweightValueAdapter, monkeypatch)
    eager, lazy = LightweightSession(make(), kernel="torch"), LazyLightweightSession(make(), kernel="torch")
    for session in (eager, lazy):
        session.publish(models[1], 3, 4, adapters[0])
        session.publish(models[2], 4, 7, adapters[1])
    assert lazy.costs["superseded_pending_releases"] == 1
    assert lazy.costs["initial_encode_flops"] == lazy.costs["append_encode_flops"] == 0
    assert lazy.adapter is adapters[1] and lazy.pending_encode
    for session in (eager, lazy):
        session.advance(20)
    assert eager.support.clear and lazy.support.clear
    assert not lazy.pending_encode and lazy.native_mode and lazy.state is None
    assert lazy.costs["unmaterialized_retirement_events"] == 1
    torch.testing.assert_close(lazy.score(20, torch.tensor([[40]])), eager.score(20, torch.tensor([[40]])), atol=2e-6, rtol=2e-5)
    assert lazy.costs["initial_encode_flops"] == lazy.costs["append_encode_flops"] == 0


@torch.inference_mode()
def test_explicit_rebuild_cancels_pending_code_generation(monkeypatch):
    models, adapters, make = fixtures(LightweightValueAdapter, monkeypatch)
    lazy = LazyLightweightSession(make(), kernel="torch")
    lazy.publish(models[1], 3, 4, adapters[0])
    lazy.rebuild(6)
    assert not lazy.pending_encode and lazy.support.clear and lazy.native_mode
    lazy.score(7, torch.tensor([[40]]))
    assert lazy.costs["first_read_code_initializations"] == 0


def test_shape_retirement_threshold_matches_dependency_tracker():
    # The CPU population ledger uses this closed form; the runtime tracker
    # independently propagates every layer/row's inherited dependency.
    for layers, cap in ((3, 4), (6, 1024)):
        for initial in (1, cap):
            support = InheritedSupport(layers, initial, cap, inherited=True)
            threshold = layers*(cap-1)+1
            for _ in range(threshold-1):
                support.append(1)
                assert not support.clear
            support.append(1)
            assert support.clear
