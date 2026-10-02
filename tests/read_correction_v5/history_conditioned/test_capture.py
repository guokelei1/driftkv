"""Causal split, true rolling replay and exact terminal-teacher checks."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[3] / "src"),
               str(Path(__file__).resolve().parents[3] / "scripts")]

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.models.state_transition import append_with_rolling_cap
from read_correction_v5.history_conditioned.capture import (
    assign_append_targets, prepare_timeline, capture_mixed,
)
from selective_recompute_2026_09.cost import CostModel


def _model(seed):
    torch.manual_seed(seed)
    return HSTU(HSTUConfig(num_items=64, num_behaviors=3, hidden_size=32, num_heads=1,
        num_layers=2, max_seq_len=8, input_dropout=0., temporal_num_freqs=16)).eval().requires_grad_(False)


def _raw(times):
    return np.asarray(times), np.arange(1, len(times) + 1), np.ones(len(times), dtype=np.int64)


def _inputs(events):
    times = torch.as_tensor(events[0])[None]
    items = torch.as_tensor(events[1])[None]
    behaviors = torch.as_tensor(events[2])[None]
    deltas = torch.zeros_like(times).float()
    deltas[:, 1:] = times[:, 1:] - times[:, :-1]
    return items, behaviors, deltas


def test_balanced_causal_assignment_and_complete_timestamp_groups():
    assigned = assign_append_targets(list(range(128)), list(range(128, 144)))
    assert Counter(assigned[u] for u in range(128)) == dict.fromkeys((0, 256, 512, 896), 32)
    assert Counter(assigned[u] for u in range(128, 144)) == dict.fromkeys((0, 256, 512, 896), 4)
    assert assigned == assign_append_targets(list(reversed(range(128))), list(reversed(range(128, 144))))
    raw = _raw([1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 100, 101])
    timeline = prepare_timeline(raw, cutover=100, history_length=8, append_target=4)
    assert timeline["actual_append"] == 3
    assert timeline["parent_last_timestamp"] == 5
    assert timeline["pseudo_release_timestamp"] == 6
    assert timeline["terminal_last_timestamp"] == 8
    assert timeline["inherited_count"] == 5
    np.testing.assert_array_equal(timeline["terminal"][0], [3, 4, 5, 5, 5, 6, 7, 8])
    tied = prepare_timeline(_raw([1, 1, 1]), cutover=2, history_length=8, append_target=7)
    assert tied["actual_append"] == 0
    assert len(tied["prefix"][0]) == 3


def test_mixed_capture_matches_scalar_rolling_exact_teacher_and_executed_cost():
    torch.set_num_threads(1)
    parent, current = _model(17), _model(29)
    # Two users share a full prefix/band batch; another tests A=0, one is short.
    raw = {11: _raw(range(1, 14)), 12: _raw(range(21, 34)),
           13: _raw(range(41, 50)), 14: _raw([51, 52, 52, 53, 54])}
    targets = {11: 5, 12: 5, 13: 0, 14: 2}
    timelines = {u: prepare_timeline(v, cutover=100, history_length=8,
        append_target=targets[u]) for u, v in raw.items()}
    rows, ledger, states = capture_mixed(parent, current, timelines, list(raw),
        cutover=100, known=64, queries=3, device="cpu", batch_size=2,
        history_length=8, attention_backend="torch", append_band_size=4)
    for uid, timeline in timelines.items():
        initial = parent.compute_kv(*_inputs(timeline["prefix"]))
        suffix = timeline["suffix"]
        if len(suffix[0]):
            times = torch.as_tensor(suffix[0])[None]
            previous = torch.cat((times.new_tensor([[timeline["parent_last_timestamp"]]]), times[:, :-1]), 1)
            reference = append_with_rolling_cap(current, initial,
                torch.as_tensor(suffix[1])[None], torch.as_tensor(suffix[2])[None],
                (times - previous).clamp(0, 7 * 86400).float(), 8)
        else:
            reference = initial
        exact = current.compute_kv(*_inputs(timeline["terminal"]))
        for field in ("k", "v"):
            torch.testing.assert_close(getattr(rows[uid]["parent"], field), getattr(reference, field), atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(getattr(rows[uid]["teacher"], field), getattr(exact, field), atol=2e-6, rtol=2e-5)
        assert rows[uid]["query_delta"] == 100 - timeline["terminal_last_timestamp"]
        assert rows[uid]["parent"].seq_len == rows[uid]["teacher"].seq_len
        assert states[str(uid)]["actual_append"] == targets[uid]
    assert not torch.allclose(rows[11]["parent"].k, rows[11]["teacher"].k)
    cost = CostModel(32, 2, 1, "torch")
    expected_parent = cost.full_cache(8, batch=2) + cost.full_cache(8) + cost.full_cache(3)
    expected_teacher = cost.full_cache(8, batch=2) + cost.full_cache(8) + cost.full_cache(5)
    expected_append = (cost.band_append(8, 4, batch=2, window_size=8)
        + cost.band_append(8, 1, batch=2, window_size=8) + cost.band_append(3, 2, window_size=8))
    assert ledger["parent_full_flops"] == expected_parent
    assert ledger["teacher_full_flops"] == expected_teacher
    assert ledger["append_flops"] == expected_append
    assert ledger["total_flops"] == expected_parent + expected_teacher + expected_append
