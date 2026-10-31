import copy
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.ch3_motivation.code.run_query_gap_control import metrics, score_gap_grid
from experiments.ch3_motivation.code.analyze_query_gap_control import normalized_change
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTU, HSTUConfig


@torch.no_grad()
def test_distinct_gap_cache_batch_matches_independent_native_scores(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.set_num_threads(1)
    torch.manual_seed(61)
    model = HSTU(HSTUConfig(num_items=300, num_behaviors=2, hidden_size=8,
        num_heads=2, num_layers=2, max_seq_len=4,
        input_dropout=0., attn_dropout=0.)).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    cache = model.compute_kv(ids, torch.zeros_like(ids), torch.ones_like(ids).float())
    changed = copy.deepcopy(cache)
    changed.v.add_(.1)
    before = cache.k.clone(), cache.v.clone()
    panel = torch.tensor([[5, 6, 7]])
    gaps = [1., 86400.]
    actual = score_gap_grid(model, [cache, changed], panel, gaps)
    expected = torch.stack([torch.stack([score(model, c, panel,
        torch.tensor([g]))[0][0] for c in [cache, changed]]) for g in gaps])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(cache.k, before[0], atol=0, rtol=0)
    torch.testing.assert_close(cache.v, before[1], atol=0, rtol=0)


def test_common_shift_decomposition_does_not_claim_relative_error():
    teacher = np.array([0., 1., 3., 5.])
    m = metrics(teacher + 7., teacher)
    assert m["total_mse"] == 49.
    assert m["centered_mse"] == 0.
    assert m["pair_discord"] == 0.
    changed = metrics(teacher + np.array([1., -1., 1., -1.]), teacher)
    assert changed["common_offset_mse"] == 0.
    assert changed["centered_mse"] == 1.


def test_teacher_signal_rescaling_is_not_relative_error_growth():
    error, signal = np.array([1., 2.]), np.array([10., 20.])
    weights = np.array([[1, 1], [2, 0], [0, 2]])
    result = normalized_change(error*4, signal*4, error, signal, weights)
    assert result["mean"] == 0.
    assert result["ci95"] == [0., 0.]
