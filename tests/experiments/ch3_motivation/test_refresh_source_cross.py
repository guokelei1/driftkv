"""Crossed source controls use one new teacher and preserve both inputs."""
import copy
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.ch3_motivation.code.probe_variants import build_variant
from experiments.ch3_motivation.code.run_refresh_source_cross import METHODS, replay_user
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.models import HSTU, HSTUConfig


@torch.no_grad()
def test_crossed_maps_receive_native_and_actual_and_zero_control_preserves_them(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.set_num_threads(1)
    torch.manual_seed(83)
    source = HSTU(HSTUConfig(num_items=300, num_behaviors=2, hidden_size=8,
        num_heads=2, num_layers=2, max_seq_len=4,
        input_dropout=0., attn_dropout=0.)).eval().requires_grad_(False)
    old, new = copy.deepcopy(source), copy.deepcopy(source)
    next(old.parameters()).view(-1)[::2].add_(.15)
    next(new.parameters()).view(-1)[::2].add_(.35)
    previous = build_variant("full_item", old, stage="f", hidden_width=4, depth=1).eval()
    actual_map = build_variant("full_item", new, stage="f", hidden_width=4, depth=1).eval()
    native_map = build_variant("full_item", new, stage="f", hidden_width=4, depth=1).eval()
    for module in previous.f:
        module.weights[-1].normal_(std=.025)
    for module in actual_map.f:
        module.weights[-1].normal_(std=.02)
    events = (np.array([86340, 86360, 86390, 86401]), np.arange(1, 5),
              np.zeros(4, dtype=np.int64))
    cfg = dict(edge=2, release_day=1, history_length=4, append_band_size=2, seed=17)
    result = replay_user(source, old, new, previous, actual_map, native_map, events, 11, cfg)
    trajectory = NativeTrajectory(source, 1, events, 86400, max_length=4, append_band_size=2)
    ids = torch.as_tensor(trajectory.retained_events[1], dtype=torch.long)[None]
    native = trajectory.fresh_cache(old)
    actual = previous.prepare_cache(old, trajectory.cache, ids)
    teacher = trajectory.fresh_cache(new)
    caches = [native, actual, native, actual, actual_map.prepare_cache(new, native, ids),
              actual_map.prepare_cache(new, actual, ids), teacher]
    panel = torch.tensor([result["panel_candidates"]])
    delta = torch.tensor([result["query_delta"]], dtype=torch.float32)
    for method, cache in zip(METHODS, caches, strict=True):
        expected = score(new, cache, panel, delta)[0][0]
        torch.testing.assert_close(torch.tensor(result["scores"][method]), expected, atol=1e-6, rtol=1e-5)
        expected_mse = (expected-score(new, teacher, panel, delta)[0][0]).double().square().mean()
        assert np.isclose(result["score_mse"][method], expected_mse.item(), rtol=1e-4, atol=1e-10)
    assert result["scores"]["train_native_on_native"] == result["scores"]["native_unadapted"]
    assert result["scores"]["train_native_on_actual"] == result["scores"]["actual_unadapted"]
    assert result["score_mse"]["teacher"] == 0.
    assert result["history_length"] == 3 and result["query_delta"] == 10
