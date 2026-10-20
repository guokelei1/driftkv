from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from system_study_2026_10.budgeted_rebuild import BudgetedRebuildSession, plan_release
from hstu_kvcache.design_two.replay import NativeTrajectory
from hstu_kvcache.models import HSTU, HSTUConfig


def history(times):
    return (np.asarray(times), np.ones(len(times), dtype=np.int64), np.zeros(len(times), dtype=np.int64))


def test_selection_ignores_tied_and_future_behaviors():
    before = {1: history([1, 8, 9]), 2: history([1, 2, 8, 9])}
    after = {1: history([1, 8, 9, 10, 10, 11]), 2: history([1, 2, 8, 9, 12])}
    a = plan_release(before, 10, .5, lambda n: n*n, lookback_seconds=3)
    b = plan_release(after, 10, .5, lambda n: n*n, lookback_seconds=3)
    assert a == b
    assert a["selected_uids"] == [1]
    assert a["unallocated_estimated_flops"] == 3


def test_whole_user_budget_and_deterministic_tie():
    users = {3: history([1, 2]), 1: history([1, 2]), 2: history([1, 2])}
    assert plan_release(users, 10, 0, lambda n: 10)["selected_uids"] == []
    assert plan_release(users, 10, 1, lambda n: 10)["selected_uids"] == [1, 2, 3]
    plan = plan_release(users, 10, .2, lambda n: 10)
    assert plan["selected_uids"] == []
    assert plan["unallocated_estimated_flops"] == 6
    assert plan_release(users, 10, .5, lambda n: 10)["selected_uids"] == [1]


@torch.inference_mode()
def test_rebuild_is_delayed_once_and_persists_across_unselected_release(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.set_num_threads(1)
    torch.manual_seed(17)
    cfg = HSTUConfig(num_items=64, num_behaviors=2, hidden_size=8, num_layers=3,
        num_heads=2, max_seq_len=4, input_dropout=0., attn_dropout=0.)
    models = [HSTU(cfg).eval().requires_grad_(False) for _ in range(3)]
    raw = (np.arange(20), np.arange(1, 21), np.zeros(20, dtype=np.int64))
    make = lambda: NativeTrajectory(models[0], 2, raw, 4, max_length=4, append_band_size=4)
    actual = BudgetedRebuildSession(make(), 1)
    reference = make()
    actual.publish(models[1], 3, 4, True)
    reference.switch(models[1], 3, 4)
    actual.advance(6)
    assert actual.pending_rebuild and not actual.costs["rebuild_events"]
    actual.score(7, torch.tensor([[30]]))
    reference.switch(models[1], 3, 7, rebuild=True)
    assert actual.trajectory.last_timestamp == 6
    assert actual.costs["rebuild_events"] == 1
    actual.score(8, torch.tensor([[31]]))
    reference.advance(8)
    assert actual.costs["rebuild_events"] == 1
    actual.publish(models[2], 4, 9, False)
    reference.switch(models[2], 4, 9)
    actual.score(13, torch.tensor([[32]]))
    reference.advance(13)
    assert actual.costs["rebuild_events"] == 1
    assert actual.trajectory.ledger == reference.ledger
    torch.testing.assert_close(actual.cache.k, reference.cache.k, rtol=0, atol=0)
    torch.testing.assert_close(actual.cache.v, reference.cache.v, rtol=0, atol=0)


@torch.inference_mode()
def test_selected_idle_user_does_not_rebuild_at_publication_or_drain(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    cfg = HSTUConfig(num_items=16, num_behaviors=2, hidden_size=8, num_layers=2,
        num_heads=2, max_seq_len=4, input_dropout=0., attn_dropout=0.)
    model = HSTU(cfg).eval().requires_grad_(False)
    session = BudgetedRebuildSession(NativeTrajectory(model, 2, history([1, 2, 6]),
        4, max_length=4, append_band_size=4), 1)
    session.publish(model, 3, 4, True)
    session.advance(9)
    session.publish(model, 4, 9, False)
    assert not session.events and not session.pending_rebuild
    assert not any(key[0] == "rebuild" for key in session.trajectory.ledger)
