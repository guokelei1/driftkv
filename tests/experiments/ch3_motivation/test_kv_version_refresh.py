"""Version diagnostics preserve old-map inputs and compare each own teacher."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from experiments.ch3_motivation.code.probe_variants import build_variant
from experiments.ch3_motivation.code.run_kv_version_refresh import replay_user, validate_refresh_pair
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.models import HSTU, HSTUConfig


class ObserveMap:
    def __init__(self, probe):
        self.probe = probe
        self.calls = []

    def prepare_cache(self, model, cache, ids):
        self.calls.append((model, cache, ids))
        return self.probe.prepare_cache(model, cache, ids)


@torch.no_grad()
def test_version_user_keeps_old_embedding_and_zero_refresh_matches_unchanged(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.set_num_threads(1)
    torch.manual_seed(41)
    source_model = HSTU(HSTUConfig(num_items=300, num_behaviors=2, hidden_size=8,
        num_heads=2, num_layers=2, max_seq_len=4,
        input_dropout=0., attn_dropout=0.)).eval().requires_grad_(False)
    old, new = copy.deepcopy(source_model), copy.deepcopy(source_model)
    next(old.parameters()).view(-1)[::2].add_(.1)
    next(new.parameters()).view(-1)[::2].add_(.3)
    previous = build_variant("full_item", old, stage="f", hidden_width=4, depth=1,
                             query_chunk=16, row_chunk=4).eval()
    for module in previous.f:
        module.weights[-1].normal_(std=.03)
    refresh = build_variant("full_item", new, stage="f", hidden_width=4, depth=1,
                            query_chunk=16, row_chunk=4).eval()
    tracked_old, tracked_new = ObserveMap(previous), ObserveMap(refresh)
    times = np.array([86360, 86370, 86380, 86390, 86401])
    events = times, np.arange(1, 6), np.zeros(5, dtype=np.int64)
    cfg = dict(edge=2, release_day=1, history_length=4, append_band_size=2, seed=17)
    result = replay_user(source_model, old, new, tracked_old, tracked_new, events, 9, cfg)
    assert len(tracked_old.calls) == len(tracked_new.calls) == 1
    assert tracked_old.calls[0][0] is old and tracked_new.calls[0][0] is new
    source = NativeTrajectory(source_model, 1, events, 86400, max_length=4, append_band_size=2)
    ids = torch.as_tensor(source.retained_events[1], dtype=torch.long)[None]
    actual_previous = previous.prepare_cache(old, source.cache, ids)
    wrong_embedding = previous.prepare_cache(new, source.cache, ids)
    torch.testing.assert_close(tracked_new.calls[0][1].k, actual_previous.k, atol=0, rtol=0)
    torch.testing.assert_close(tracked_new.calls[0][1].v, actual_previous.v, atol=0, rtol=0)
    assert float((actual_previous.k - wrong_embedding.k).abs().max()) > 1e-6
    old_teacher, new_teacher = source.fresh_cache(old), source.fresh_cache(new)
    caches = dict(source_unmapped=source.cache, unchanged=actual_previous,
                  refresh=actual_previous, teacher=new_teacher)
    panel = torch.tensor([result["panel_candidates"]])
    delta = torch.tensor([result["query_delta"]], dtype=torch.float32)
    for target_name, model, teacher in (("old_target", old, old_teacher), ("new_target", new, new_teacher)):
        for method, values in result["endpoints"][target_name]["scores"].items():
            cache = teacher if method == "teacher" else caches[method]
            expected = score(model, cache, panel, delta)[0][0]
            torch.testing.assert_close(torch.tensor(values), expected, atol=1e-6, rtol=1e-5)
    endpoint = result["endpoints"]["new_target"]
    torch.testing.assert_close(torch.tensor(endpoint["scores"]["unchanged"]),
                               torch.tensor(endpoint["scores"]["refresh"]), atol=0, rtol=0)
    assert endpoint["score_mse"]["teacher"] == 0.
    assert result["history_length"] == 4 and result["query_delta"] == 10


def test_validation_rejects_ordinary_adjacent_map_and_changed_predecessor():
    from types import SimpleNamespace
    cfg = SimpleNamespace(num_layers=2, num_heads=2, head_dim=4, item_dim=8, include_item=True)
    probe = SimpleNamespace(stage="f", cfg=cfg)
    old_identity = dict(edge=2, models={"v1": "a", "v2": "b"},
        input_binding_sha256="input", population_sha256="population",
        fit_uids=[1], validation_users=[2])
    identity = dict(old_identity, protocol="frozen_mapped_source_version_refresh_v1",
        source_kind="actual_predecessor_mapped_cache_not_native_old_target",
        history_edge=2, source_native_version=1, predecessor_target_version=2,
        target_version=3, history_cutoff=86400, predecessor_checkpoint_sha256="previous",
        models={"v1": "a", "v2": "b", "v3": "c"})
    metadata = dict(method_label="refresh_map", identity=identity)
    kwargs = dict(history_edge=2, target_version=3, models=identity["models"], uids=[3],
                  predecessor_sha256="previous", history_cutoff=86400)
    validate_refresh_pair(probe, {"identity": old_identity}, probe, metadata, **kwargs)
    adjacent = dict(metadata, identity=dict(identity, source_kind="native_previous_target"))
    with pytest.raises(ValueError, match="actual-mapped-input"):
        validate_refresh_pair(probe, {"identity": old_identity}, probe, adjacent, **kwargs)
    with pytest.raises(ValueError, match="identity differs"):
        validate_refresh_pair(probe, {"identity": old_identity}, probe, metadata,
                              **dict(kwargs, predecessor_sha256="another"))
