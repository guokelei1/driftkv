"""Actual CPU optimizer, paired data reuse, controls and partial-stage resume."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import run_probe_query_centered as driver
from experiments.ch3_motivation.code.probe_variants import load_variant
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.common.artifacts import sha256


@pytest.fixture(autouse=True)
def backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_actual_pair_controls_and_deterministic_partial_resume(tmp_path, monkeypatch):
    root = tmp_path / "input"
    folder = root / "edge_1"
    folder.mkdir(parents=True)
    fit, val = list(range(1000, 1064)), list(range(2000, 2016))
    population = dict(fit=fit, validation=val)
    (root / "population.json").write_text(json.dumps(population))
    for name in ("history", "requests"):
        # In particular requests is deliberately not readable parquet: fitting
        # must not parse unused future request labels.
        (folder / f"{name}.pack").write_text(name)
    cutoff = 20 * 86400
    binding = dict(population_sha256=sha256(root / "population.json"),
        settings=dict(edge=1, release_day=20, history_length=8, attention_backend="torch"),
        **{name: dict(path=f"{name}.pack", sha256=sha256(folder / f"{name}.pack")) for name in ("history", "requests")})
    (folder / "inputs.json").write_text(json.dumps(binding))
    candidates = dict(population_sha256=sha256(root / "population.json"),
        fitting_pool_uids=fit, validation_uids=val,
        recent_items={str(uid): list(range(1, 17)) for uid in fit + val})
    (folder / "calibration_feedback_candidates.json").write_text(json.dumps(candidates))
    torch.manual_seed(819)
    cfg = HSTUConfig(num_items=320, num_behaviors=2, hidden_size=64, num_heads=2,
        num_layers=2, max_seq_len=8, input_dropout=0., attn_dropout=0.)
    old = HSTU(cfg).eval().requires_grad_(False)
    current = copy.deepcopy(old)
    with torch.no_grad():
        current.blocks[0].attn.k_proj.weight.add_(.03 * torch.randn_like(current.blocks[0].attn.k_proj.weight))
    histories = SimpleNamespace(rows={uid: (np.array([cutoff - 31, cutoff - 30, cutoff + 1]),
        np.array([1, 2, 63]), np.ones(3, dtype=np.int64)) for uid in fit + val})
    loads, captures, fitted = [], [], []
    monkeypatch.setattr(driver, "load_models", lambda *a, **k:
        (loads.append(1) or old, current, dict(old="old-model", current="current-model")))
    monkeypatch.setattr(driver, "restore_pack", lambda *a: histories)
    capture = driver.capture_samples

    def capture_past(*args, **kwargs):
        samples = capture(*args, **kwargs)
        assert all(63 not in value["item_ids"] for value in samples.values())
        captures.append(1)
        return samples

    monkeypatch.setattr(driver, "capture_samples", capture_past)
    original_fit = driver.fit_query_conditioned

    def actual_fit(*args, **kwargs):
        fitted.append(args[1].stage)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(driver, "fit_query_conditioned", actual_fit)
    args = SimpleNamespace(input_root=root, output=tmp_path / "rms", edge=1, gpu=0, device="cpu",
        seed=17, fit_users=64, queries=16, width=7, depth=1, steps=2, learning_rate=3e-4,
        read_weight=1., score_weight=1., patience=0, validation_every=1,
        stages="f,fg,fgh", controls=True, candidate_source="feedback", recent_fraction=.75,
        variant="full_item", query_conditioning="rms", factorized_h=True)
    cache = {}
    prepared = driver.run(args, preparation_cache=cache, preparation_only=True)
    assert prepared["status"] == "inputs_prepared_only"
    assert fitted == [] and not (args.output / "f.pt").exists()
    rms = driver.run(args, preparation_cache=cache)
    args.output, args.query_conditioning = tmp_path / "centered", "center_std"
    centered = driver.run(args, preparation_cache=cache)
    fresh_rng = torch.get_rng_state().clone()
    assert len(loads) == len(captures) == 1
    assert [row["method_label"] for row in centered["stages"]] == ["f", "f_extended", "fg", "fg_extended", "fgh"]
    rms_f, _ = load_variant(tmp_path / "rms/f.pt")
    centered_f, _ = load_variant(args.output / "f.pt")
    assert all(torch.equal(value, centered_f.state_dict()[key])
        for key, value in rms_f.state_dict().items() if key != "query_scale")
    for result in (rms, centered):
        for row in result["stages"]:
            assert row["total_calibration_flops_estimate"] == sum(row["arithmetic"].values())
            assert row["query_conditioning_statistics"]["users"] == 64
            assert row["execution_adapter"]["binding"]["format"] == "resident_probe_fitting_v1"
    expected = {stage: load_variant(args.output / f"{stage}.pt")[0].state_dict() for stage in ("f", "fg", "fgh")}
    (args.output / "fgh.pt").unlink()
    (args.output / "fgh.json").unlink()
    (args.output / "f.json").unlink()  # valid committed checkpoint recovers JSON
    fitted.clear()
    driver.run(args, preparation_cache=cache)
    assert fitted == ["fgh"]
    assert torch.equal(torch.get_rng_state(), fresh_rng)
    for stage, wanted in expected.items():
        actual, _ = load_variant(args.output / f"{stage}.pt")
        assert all(torch.equal(value, actual.state_dict()[key]) for key, value in wanted.items())
    assert (args.output / "f.json").exists()
