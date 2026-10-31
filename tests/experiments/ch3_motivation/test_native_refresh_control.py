"""Native-source control captures the intended old/new native prefix pair."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from experiments.ch3_motivation.code.fit_native_refresh_control import capture_native_sources
from experiments.ch3_motivation.code.common import raw_prefix
from hstu_kvcache.models import HSTU, HSTUConfig


@torch.no_grad()
def test_native_source_and_teacher_use_same_causal_history_but_distinct_models(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.set_num_threads(1)
    torch.manual_seed(71)
    old = HSTU(HSTUConfig(num_items=300, num_behaviors=2, hidden_size=8,
        num_heads=2, num_layers=2, max_seq_len=4,
        input_dropout=0., attn_dropout=0.)).eval().requires_grad_(False)
    new = copy.deepcopy(old)
    next(new.parameters()).add_(.2)
    events = (np.array([86300, 86320, 86390, 86400]), np.arange(1, 5),
              np.zeros(4, dtype=np.int64))
    histories = SimpleNamespace(rows={7: events})
    cfg = dict(release_day=1, history_length=4, calibration_queries_per_user=16, seed=17)
    samples = capture_native_sources(old, new, histories, [7], cfg, torch.device("cpu"))
    sample = samples[7]
    raw, _, end = raw_prefix(histories, 7, cfg, torch.device("cpu"))
    native_old, teacher_new = old.compute_kv(*raw), new.compute_kv(*raw)
    assert end == 3 and sample["history_count"] == 3
    assert sample["parent"] is sample["parent_cache"]
    assert sample["teacher"] is sample["teacher_cache"]
    torch.testing.assert_close(sample["parent_cache"].k, native_old.k, atol=0, rtol=0)
    torch.testing.assert_close(sample["parent_cache"].v, native_old.v, atol=0, rtol=0)
    torch.testing.assert_close(sample["teacher_cache"].k, teacher_new.k, atol=0, rtol=0)
    torch.testing.assert_close(sample["teacher_cache"].v, teacher_new.v, atol=0, rtol=0)
    assert (native_old.k - teacher_new.k).abs().max() > 1e-6
    assert sample["query_delta"].item() == 10
    assert sample["candidates"].shape == (1, 16)
