"""Partial-refresh lifecycle invariants with the actual full-item KV adapter."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from design_system.evaluate_lightweight_support_mask import SupportMaskedSession
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.design_one.lightweight_fullitem_kv import LightweightFullItemKVAdapter
from hstu_kvcache.design_two.replay import NativeTrajectory
from hstu_kvcache.models import HSTU, HSTUConfig
from partial_refresh_2026_10.session import MISSING_VERSION, PartialRefreshSession, refresh_plan


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(20261013)
    config = HSTUConfig(num_items=96, num_behaviors=2, hidden_size=8, num_layers=3,
                        num_heads=2, max_seq_len=6, input_dropout=0., attn_dropout=0.)
    models = [HSTU(config).eval().requires_grad_(False) for _ in range(4)]
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
def test_all_refresh_matches_current_supportmasked_lazy(setup):
    models, adapters, make = setup
    partial = PartialRefreshSession(make(), kernel="torch", exit_when_clear=False)
    current = SupportMaskedSession(make(), kernel="torch", exit_when_clear=False)
    candidates = torch.tensor([[70, 71]])
    for index, (release, first, later) in enumerate(((6, 8, 10), (11, 13, 15), (16, 17, 20))):
        for session in (partial, current):
            session.publish(models[index + 1], index + 3, release, adapters[index])
            session.advance(first)
        assert partial.pending_encode
        torch.testing.assert_close(partial.score(first, candidates), current.score(first, candidates),
                                   atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(partial.state.codes, current.state.codes, atol=2e-6, rtol=2e-5)
        partial.prepare_read(later)
        current.prepare_read(later)
        torch.testing.assert_close(partial.state.codes, current.state.codes, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(partial.score(later, candidates), current.score(later, candidates),
                                   atol=2e-6, rtol=2e-5)
        np.testing.assert_array_equal(partial.support.flags, current.support.flags)
        np.testing.assert_array_equal(partial.code_versions, np.full(6, index + 3))
        torch.testing.assert_close(partial.cache.k, current.cache.k, atol=0, rtol=0)
        torch.testing.assert_close(partial.cache.v, current.cache.v, atol=0, rtol=0)
        assert partial.trajectory.ledger == current.trajectory.ledger
    assert partial.costs["initial_encode_flops"] + partial.costs["refresh_encode_flops"] == current.costs["initial_encode_flops"]
    assert partial.costs["append_encode_flops"] == current.costs["append_encode_flops"]
    assert partial.costs["support_mask_flops"] == current.costs["support_mask_flops"]
    assert partial.costs["support_mask_transfer_bytes"] == current.costs["support_mask_transfer_bytes"]
    assert partial.memory["code_version_peak_bytes"] == 6 * np.dtype(np.int32).itemsize


@torch.inference_mode()
def test_old_codes_preserved_then_mandatory_older_refresh(setup):
    models, adapters, make = setup
    session = PartialRefreshSession(make(), exit_when_clear=False)
    session.publish(models[1], 3, 6, adapters[0], rho=0)
    session.prepare_read(6)
    old_codes = session.state.codes.clone()
    session.publish(models[2], 4, 6, adapters[1], rho=.25)
    snapshot = session.snapshot_codes()
    reference = session.encode_reference()
    costs = Counter(session.costs)
    candidate = session.candidate_state(.25, reference)
    torch.testing.assert_close(session.state.codes, old_codes, rtol=0, atol=0)
    assert session.costs == costs and session.pending_encode
    assert session.refresh_plan()["indices"].tolist() == [4, 5]
    session.prepare_read(6)
    torch.testing.assert_close(session.state.codes, candidate.codes, rtol=0, atol=0)
    torch.testing.assert_close(session.state.codes[:, :, :4], snapshot.codes[:, :, :4], rtol=0, atol=0)
    np.testing.assert_array_equal(session.code_versions, [3, 3, 3, 3, 4, 4])
    assert not torch.allclose(session.state.codes[:, :, :4], reference.codes[:, :, :4])
    preserved_v4 = session.state.codes[:, :, 4:].clone()
    session.publish(models[3], 5, 6, adapters[2], rho=0)
    assert session.refresh_plan()["older"] == 4
    assert session.refresh_plan()["selected"] == 4
    session.prepare_read(6)
    np.testing.assert_array_equal(session.code_versions, [5, 5, 5, 5, 4, 4])
    torch.testing.assert_close(session.state.codes[:, :, 4:], preserved_v4, rtol=0, atol=0)
    assert session.costs["forced_older_rows"] == 4
    assert session.costs["initial_encode_rows"] == 6
    assert session.costs["refresh_encode_rows"] == 6


def test_selection_excludes_current_and_forces_skipped_versions():
    versions = np.array([MISSING_VERSION, 2, 3, 4, 4], dtype=np.int32)
    plan = refresh_plan(versions, 4, .2)
    assert plan.indices.tolist() == [0, 1]
    assert plan.missing_rows == plan.older_rows == 1
    assert plan.current_rows == 2 and plan.retained_previous_rows == 1
    assert refresh_plan(versions, 4, 1).indices.tolist() == [0, 1, 2]
    assert refresh_plan([3, 3, 3], 5, 0).indices.tolist() == [0, 1, 2]


@torch.inference_mode()
def test_pending_appends_are_missing_fifo_and_native_writes_equal(setup):
    models, adapters, make = setup
    session, plain = PartialRefreshSession(make(), exit_when_clear=False), make()
    session.publish(models[1], 3, 6, adapters[0], rho=0)
    plain.switch(models[1], 3, 6)
    session.prepare_read(6)
    previous = session.state.codes.clone()
    session.publish(models[2], 4, 6, adapters[1], rho=0)
    plain.switch(models[2], 4, 6)
    session.advance(8)
    plain.advance(8)
    assert session.pending_encode
    np.testing.assert_array_equal(session.code_versions, [3, 3, 3, 3, -1, -1])
    torch.testing.assert_close(session.state.codes[:, :, :4], previous[:, :, 2:], rtol=0, atol=0)
    assert not session.state.codes[:, :, -2:].any()
    assert session.refresh_plan()["indices"].tolist() == [4, 5]
    session.prepare_read(8)
    np.testing.assert_array_equal(session.code_versions, [3, 3, 3, 3, 4, 4])
    # New lower-layer rows are independent of inherited state and remain zero.
    assert not session.state.codes[0, :, -2:].any()
    session.advance(11)
    plain.advance(11)
    np.testing.assert_array_equal(session.code_versions, [3, 4, 4, 4, 4, 4])
    torch.testing.assert_close(session.cache.k, plain.cache.k, rtol=0, atol=0)
    torch.testing.assert_close(session.cache.v, plain.cache.v, rtol=0, atol=0)
    np.testing.assert_array_equal(session.trajectory.producers, plain.producers)
    assert session.trajectory.ledger == plain.ledger
    assert session.costs["append_encode_rows"] == 3
    candidates = torch.tensor([[70]])
    delta = torch.tensor([11 - plain.last_timestamp], dtype=torch.float32)
    expected = score(models[2], plain.cache, candidates, delta,
                     history_read_fn=adapters[1].make_history_read(session.state))[0]
    torch.testing.assert_close(session.score(11, candidates), expected)
    encode_costs = {key: session.costs[key] for key in
                   ("initial_encode_flops", "refresh_encode_flops", "append_encode_flops")}
    session.score(11, candidates)
    assert encode_costs == {key: session.costs[key] for key in encode_costs}


@torch.inference_mode()
def test_unread_releases_clearance_and_rebuild(setup):
    models, adapters, make = setup
    session = PartialRefreshSession(make())
    session.publish(models[1], 3, 6, adapters[0], rho=0)
    session.publish(models[2], 4, 8, adapters[1], rho=0)
    assert session.costs["superseded_pending_releases"] == 1
    assert session.state is None and session.refresh_plan()["missing"] == 6
    session.advance(40)
    assert session.support.clear and session.native_mode and not session.pending_encode
    assert len(session.code_versions) == 0
    assert session.costs["initial_encode_flops"] == session.costs["refresh_encode_flops"] == 0
    session.publish(models[3], 5, 40, adapters[2], rho=.5)
    session.rebuild(41)
    assert session.native_mode and session.state is None and not session.pending_encode
    assert session.support.clear and len(session.code_versions) == 0
    assert session.costs["rebuild_events"] == 1


@torch.inference_mode()
def test_raw_full_refresh_matches_existing_masked_scores_and_charges_per_batch(setup):
    models, adapters, make = setup
    raw = PartialRefreshSession(make(), storage_mode="raw", exit_when_clear=False)
    current = SupportMaskedSession(make(), kernel="torch", exit_when_clear=False)
    batches = 0
    for index, (release, first, later) in enumerate(((6, 8, 10), (11, 13, 15), (16, 17, 20))):
        for session in (raw, current):
            session.publish(models[index + 1], index + 3, release, adapters[index])
        for at, candidates in ((first, torch.tensor([[70, 71, 72]])),
                               (later, torch.tensor([[73]]))):
            torch.testing.assert_close(raw.score(at, candidates), current.score(at, candidates),
                                       atol=2e-6, rtol=2e-5)
            batches += 1
            gated = raw.state.codes * torch.as_tensor(raw.support.flags)[:, None, :, None, None]
            torch.testing.assert_close(gated, current.state.codes, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(raw.cache.k, current.cache.k, rtol=0, atol=0)
            torch.testing.assert_close(raw.cache.v, current.cache.v, rtol=0, atol=0)
            assert raw.trajectory.ledger == current.trajectory.ledger
            assert raw.state.storage_bytes() == current.state.storage_bytes()
    assert raw.costs["support_mask_flops"] == raw.costs["support_mask_transfer_bytes"] == 0
    assert raw.costs["read_support_mask_flops"] == batches * 3 * 6 * 2 * 2
    assert raw.costs["read_support_mask_transfer_bytes"] == batches * 3 * 6
    assert raw.costs["raw_masked_read_batches"] == batches
    assert raw.costs["extra_history_read_flops"] == current.costs["extra_history_read_flops"]
    assert raw.costs["initial_encode_flops"] + raw.costs["refresh_encode_flops"] == current.costs["initial_encode_flops"]
    assert raw.costs["append_encode_flops"] == current.costs["append_encode_flops"]


@torch.inference_mode()
def test_raw_dormant_codes_are_preserved_and_reactivated_on_release(setup):
    models, adapters, make = setup
    raw = PartialRefreshSession(make(), storage_mode="raw", exit_when_clear=False)
    masked = PartialRefreshSession(make(), exit_when_clear=False)
    for session in (raw, masked):
        session.publish(models[1], 3, 6, adapters[0], rho=0)
        session.prepare_read(6)
        session.advance(8)
    assert not raw.support.flags[0, -2:].any()
    assert raw.state.codes[0, :, -2:].abs().sum() > 0
    assert not masked.state.codes[0, :, -2:].any()
    # Reference helpers use the chosen storage semantics even before publication.
    torch.testing.assert_close(raw.encode_reference().codes, raw.state.codes,
                               atol=2e-6, rtol=2e-5)
    old_raw = raw.state.codes.clone()
    for session in (raw, masked):
        session.publish(models[2], 4, 8, adapters[1], rho=0)
    assert raw.support.flags.all()
    reference = raw.encode_reference()
    before = Counter(raw.costs)
    candidate = raw.candidate_state(0, reference)
    torch.testing.assert_close(candidate.codes, old_raw, rtol=0, atol=0)
    assert raw.costs == before
    raw.prepare_read(8)
    torch.testing.assert_close(raw.state.codes, old_raw, rtol=0, atol=0)
    assert raw.costs["refresh_encode_rows"] == 0
    assert raw.state.codes[0, :, -2:].abs().sum() > 0
    np.testing.assert_array_equal(raw.code_versions, np.full(6, 3))
    candidates = torch.tensor([[70, 71]])
    gap = torch.tensor([8 - raw.trajectory.last_timestamp], dtype=torch.float32)
    wanted = score(models[2], raw.cache, candidates, gap,
                   history_read_fn=adapters[1].make_history_read(candidate))[0]
    torch.testing.assert_close(raw.score(8, candidates), wanted, atol=2e-6, rtol=2e-5)
    assert raw.events[-1]["storage_mode"] == "raw"


@torch.inference_mode()
def test_raw_partial_candidate_matches_first_read_and_mask_does_not_touch_storage(setup):
    models, adapters, make = setup
    raw = PartialRefreshSession(make(), storage_mode="raw", exit_when_clear=False)
    raw.publish(models[1], 3, 6, adapters[0])
    raw.prepare_read(6)
    raw.publish(models[2], 4, 6, adapters[1], rho=.25)
    raw.advance(8)
    reference = raw.encode_reference()
    candidate = raw.candidate_state(.25, reference)
    # The two pending new rows are missing and forced; previous rows are intact.
    assert raw.refresh_plan()["indices"].tolist() == [4, 5]
    gap = torch.tensor([8 - raw.trajectory.last_timestamp], dtype=torch.float32)
    gated = candidate.codes * torch.as_tensor(raw.support.flags)[:, None, :, None, None]
    wanted_state = type(candidate)(gated, candidate.seq_len)
    ids = torch.tensor([[70, 71, 72, 73]])
    wanted = score(models[2], raw.cache, ids, gap,
                   history_read_fn=adapters[1].make_history_read(wanted_state))[0]
    torch.testing.assert_close(raw.score(8, ids), wanted, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(raw.state.codes, candidate.codes, atol=0, rtol=0)
    before_codes, before_versions = raw.state.codes.clone(), raw.code_versions.copy()
    encode_flops = sum(raw.costs[k] for k in ("initial_encode_flops", "refresh_encode_flops", "append_encode_flops"))
    before_mask_flops = raw.costs["read_support_mask_flops"]
    raw.score(8, torch.tensor([[70]]))
    torch.testing.assert_close(raw.state.codes, before_codes, rtol=0, atol=0)
    np.testing.assert_array_equal(raw.code_versions, before_versions)
    assert sum(raw.costs[k] for k in ("initial_encode_flops", "refresh_encode_flops", "append_encode_flops")) == encode_flops
    assert raw.costs["read_support_mask_flops"] - before_mask_flops == raw.state.codes.numel()
