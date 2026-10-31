"""Lifecycle checks protect causal exit, native trajectory, and code alignment."""

import numpy as np
import torch
import sys
from pathlib import Path


from experiments.archive.compact_read.evaluate_lightweight import LightweightSession

from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.methods.cachetide.compact.lightweight_fullitem_kv import LightweightFullItemKVAdapter
from hstu_kvcache.methods.cachetide.lifecycle import InheritedSupport
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.models import HSTU, HSTUConfig


def fixtures(seed=1):
    torch.manual_seed(seed)
    cfg = HSTUConfig(num_items=64, num_behaviors=3, hidden_size=8, num_layers=3,
                     num_heads=2, max_seq_len=4, input_dropout=0., attn_dropout=0.)
    models = [HSTU(cfg).eval() for _ in range(3)]
    adapters = []
    for _ in range(2):
        adapter = LightweightFullItemKVAdapter(3, 2, 4, rank=2, max_length=4).eval()
        for layer in adapter.layers:
            layer.input_weight.data.normal_(std=.1)
            layer.decoder.data.normal_(std=.05)
            layer.key_decoder.data.normal_(std=.03)
        adapters.append(adapter)
    raw = (np.arange(24), np.arange(1, 25), np.ones(24, dtype=np.int64))
    return models, adapters, raw


def trajectory(model, raw):
    return NativeTrajectory(model, 2, raw, 4, max_length=4, append_band_size=4)


@torch.inference_mode()
def test_causal_clearance_matches_native_counterfactual_but_not_fresh():
    models, adapters, raw = fixtures()
    session = LightweightSession(trajectory(models[0], raw), kernel="torch")
    exact = trajectory(models[0], raw)
    session.publish(models[1], 3, 4, adapters[0])
    exact.switch(models[1], 3, 4, rebuild=True)
    # Four appends remove every old producer, but high-layer inherited state remains.
    session.advance(8)
    exact.advance(8)
    assert np.all(session.trajectory.producers == 3)
    assert not session.support.clear and session.read_mode == "corrected"
    assert (session.cache.k - exact.cache.k).abs().max() > 1e-5
    session.advance(14)
    exact.advance(14)
    assert session.support.clear and session.read_mode == "native"
    torch.testing.assert_close(session.cache.k, exact.cache.k, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(session.cache.v, exact.cache.v, rtol=1e-5, atol=2e-6)
    # Independence from inherited K/V is not equality to prefix reconstruction.
    assert (session.cache.k - session.trajectory.fresh_cache().k).abs().max() > 1e-5
    assert session.state is None and session.costs["retirement_events"] == 1
    assert session.events[-1]["reason"] == "inherited_support_cleared"
    actual = session.score(14, torch.tensor([[28, 29]]))
    expected = score(models[1], exact.cache, torch.tensor([[28, 29]]), torch.tensor([1.]))[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)


@torch.inference_mode()
def test_codes_append_release_invalidation_and_native_trajectory():
    models, adapters, raw = fixtures(2)
    plain = trajectory(models[0], raw)
    session = LightweightSession(trajectory(models[0], raw), kernel="torch", exit_when_clear=False)
    for index, (at, end) in enumerate(((4, 9), (9, 18))):
        version = index + 3
        model, adapter = models[index + 1], adapters[index]
        old_state = session.state
        session.publish(model, version, at, adapter)
        plain.switch(model, version, at)
        assert session.state is not old_state
        for timestamp in (at, at + 1, end):
            session.advance(timestamp)
            plain.advance(timestamp)
            torch.testing.assert_close(session.cache.k, plain.cache.k, rtol=0, atol=0)
            torch.testing.assert_close(session.cache.v, plain.cache.v, rtol=0, atol=0)
            ids = torch.as_tensor(session.trajectory.retained_events[1])[None]
            reference = adapter.encode_cache(session.cache, model.lookup_item_embeddings(ids))
            torch.testing.assert_close(session.state.codes, reference.codes, rtol=1e-5, atol=2e-6)
            candidates = torch.tensor([[25, 26]])
            delta = torch.tensor([timestamp - plain.last_timestamp], dtype=torch.float32)
            expected = score(model, plain.cache, candidates, delta,
                history_read_fn=adapter.make_history_read(reference))[0]
            torch.testing.assert_close(session.score(timestamp, candidates), expected, rtol=1e-5, atol=2e-6)
            assert session.state.seq_len == session.cache.seq_len <= 4
            assert session.support.flags.shape == (3, session.cache.seq_len)
    assert session.costs["initial_encode_flops"] == 2 * adapters[0].estimate_flops(tokens=4)["token_transform"]
    assert session.memory["compact_peak_bytes"] == 3 * 4 * 4 * 4
    assert session.memory["support_peak_bytes"] == 3 * 4


@torch.inference_mode()
def test_rebuild_policies_replace_live_state_then_continue_native_and_exclude_ties():
    models, adapters, raw = fixtures(3)
    for policy in ("rebuild_at_release", "rebuild_on_first_request"):
        session = LightweightSession(trajectory(models[0], raw), kernel="torch")
        reference = trajectory(models[0], raw)
        session.publish(models[1], 3, 4, adapters[0], policy=policy)
        reference.switch(models[1], 3, 4, rebuild=policy == "rebuild_at_release")
        # Idle writes occur before the request. Event t=7 remains excluded at query t=7.
        if policy == "rebuild_on_first_request":
            reference.switch(models[1], 3, 7, rebuild=True)
        reference.advance(7)
        session.score(7, torch.tensor([[25]]))
        assert session.trajectory.last_timestamp == 6
        assert not session.pending_rebuild and session.state is None and session.support.clear
        torch.testing.assert_close(session.cache.k, reference.cache.k, rtol=0, atol=0)
        session.advance(14)
        reference.advance(14)
        torch.testing.assert_close(session.cache.k, reference.cache.k, rtol=0, atol=0)
        assert session.costs["rebuild_events"] == 1
        assert session.costs["initial_encode_flops"] == 0
        assert sum(count for (action, *_), count in session.trajectory.ledger.items()
                   if action == "rebuild") == 1
        # A later release invalidates this clean state and creates new target codes.
        session.publish(models[2], 4, 14, adapters[1])
        assert not session.support.clear and session.state is not None


def test_support_matches_dependency_set_reference_with_partial_prefix_and_band_splits():
    for cap in (1, 4):
        for initial in (0, 2 if cap > 1 else 1, cap):
            tracker = InheritedSupport(3, initial, cap, inherited=True)
            # Distinct symbolic ancestors are an independent exact support reference.
            rows = [[{(layer, pos)} for pos in range(initial)] for layer in range(3)]
            for _ in range(15):
                rows = [layer[-(cap-1):] if cap > 1 else [] for layer in rows]
                new = [set()]
                for layer in range(1, 3):
                    new.append(set().union(new[-1], *rows[layer-1]))
                rows = [old + [value] for old, value in zip(rows, new)]
                tracker.append(1)
                assert tracker.flags.tolist() == [[bool(value) for value in layer] for layer in rows]
            banded = InheritedSupport(3, initial, cap, inherited=True)
            for width in (4, 3, 8):
                banded.append(width)
            np.testing.assert_array_equal(tracker.flags, banded.flags)
