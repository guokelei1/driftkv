"""Common-validation fairness checks with real tiny backbones, targets and losses."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code import validate_probe_common as driver
from experiments.ch3_motivation.code.prepare_candidates import candidates_for
from experiments.ch3_motivation.code.probe_variants import build_variant, load_variant, save_variant
from experiments.ch3_motivation.code.probes import SCALE_VERSION, probe_loss, sample_to_device
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.common.artifacts import sha256


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "fit256"
    edge = root / "edge_1"
    edge.mkdir(parents=True)
    population = dict(fit=list(range(1000, 1300)), validation=list(range(2000, 2016)))
    (root / "population.json").write_text(json.dumps(population))
    for name in ("history", "requests"):
        (edge / f"{name}.pack").write_text(name)
    population_sha = sha256(root / "population.json")
    binding = dict(population_sha256=population_sha,
        settings=dict(edge=1, release_day=231, history_length=8, seed=17,
                      calibration_queries_per_user=16, attention_backend="torch"),
        **{name: dict(path=f"{name}.pack", sha256=sha256(edge / f"{name}.pack"))
           for name in ("history", "requests")})
    (edge / "inputs.json").write_text(json.dumps(binding))
    source = tmp_path / "original_candidates.json"
    source.write_text("original observed pre-release feedback IDs, no labels\n")
    uids = population["fit"][:64] + population["validation"]
    candidate_data = dict(edge=1, release_day=231, seed=17, known_items=64,
        population_sha256=population_sha, fitting_pool_uids=population["fit"][:256],
        validation_uids=population["validation"],
        source_candidate_path=str(source), source_candidate_sha256=sha256(source),
        recent_items={str(uid): [64, 60, 56, 52, 48, 44, 40, 36, 32, 28] for uid in uids})
    (edge / "calibration_feedback_candidates.json").write_text(json.dumps(candidate_data))

    torch.manual_seed(812)
    cfg = HSTUConfig(num_items=320, num_behaviors=2, hidden_size=8, num_heads=2,
        num_layers=2, max_seq_len=8, input_dropout=0., attn_dropout=0.)
    parent = HSTU(cfg).eval().requires_grad_(False)
    current = copy.deepcopy(parent)
    with torch.no_grad():
        current.blocks[0].attn.k_proj.weight.add_(
            torch.randn_like(current.blocks[0].attn.k_proj.weight) * .03)
        current.item_emb.weight[1:].add_(torch.randn_like(current.item_emb.weight[1:]) * .02)
    histories = SimpleNamespace(rows={})
    for uid in uids:
        n = 2 if uid % 2 else 5
        # The real capture helper must ignore this last, post-release item.
        histories.rows[uid] = (np.r_[231 * 86400 - np.arange(n, 0, -1), 231 * 86400 + 2],
            np.r_[np.arange(1, n + 1), 63], np.ones(n + 1, dtype=np.int64))
    models = {"old/model.pt": "old-fixture-hash", "current/model.pt": "current-fixture-hash"}
    captured = []
    real_capture = driver.capture_samples

    def capture(old, new, values, selected, settings, device):
        assert selected == uids
        assert settings["calibration_queries_per_user"] == 16 and settings["seed"] == 17
        samples = real_capture(old, new, values, selected, settings, device)
        for uid, sample in samples.items():
            assert 63 not in sample["item_ids"]
            # Simulate an already-prepared capture with invalid old candidates.
            # The driver must discard all candidate-dependent cached targets.
            sample.update(candidates=torch.full((1, 16), 300),
                teacher_scores=torch.full((1, 16), 1e20),
                target_moment_version=SCALE_VERSION, read_moment_sum=torch.zeros(2, 2),
                score_moment_sum=torch.tensor(0.), scale_users=1,
                read_scales=torch.full((2, 1, 2, 1, 1), 1e-4), score_scale=torch.tensor(1e-4),
                loss_scale_version=SCALE_VERSION)
        # One validation user has exactly zero teacher/old cache difference.
        sample = samples[population["validation"][0]]
        sample["teacher_cache"] = sample["parent_cache"]
        captured.append(samples)
        return samples

    real_bind = driver.bind_fit_scales

    def bind(model, train, validation):
        for sample in train + validation:
            assert "teacher_scores" not in sample and "target_moment_version" not in sample
            assert sample["candidates"].shape == (1, 16)
            assert sample["candidates"][0, :8].tolist() == candidate_data["recent_items"][str(sample["uid"])][:8]
        return real_bind(model, train, validation)

    monkeypatch.setattr(driver, "load_models", lambda edge, device, backend: (parent, current, models))
    monkeypatch.setattr(driver, "restore_pack", lambda path: histories)
    monkeypatch.setattr(driver, "capture_samples", capture)
    monkeypatch.setattr(driver, "bind_fit_scales", bind)
    return SimpleNamespace(root=root, edge=edge, population=population, binding=binding,
        candidate_data=candidate_data, model=current, parent=parent, histories=histories,
        models=models, captured=captured)


def make_weight(tmp_path, context, *, variant="full_item", fit_users=64, best_step=0,
                recipe="own", original_loss=999., method_label="fgh", mutate=None):
    stage = {"fg_extended": "fg", "f_extended": "f"}.get(method_label, method_label)
    probe = build_variant(variant, context.model, stage=stage, hidden_width=5, depth=1,
                          query_chunk=16, row_chunk=8)
    with torch.no_grad():
        # Distinct fixed trained statistics, not recalibrated by validation.
        probe.row_scale.fill_(2.)
        probe.query_scale.fill_(1.5)
        probe.statistics_initialized.fill_(True)
        for group in (probe.f, probe.g, probe.h):
            if group is None:
                continue
            for layer in group:
                if hasattr(layer, "weights"):
                    layer.weights[-1].normal_(0, .0002)
                else:
                    layer.network[-1].weight.normal_(0, .0002)
    metadata = dict(seed=17, best_step=best_step, completed_steps=32,
        best_validation_loss=original_loss, objective=dict(read_weight=.1, score_weight=1.),
        method_label=method_label, fit_pool_loss_scales=dict(score_scale=1e-4 if recipe == "own" else 100.),
        identity=dict(edge=1, models=context.identity["models"],
            numerical_sources=context.identity["numerical_sources"],
            population_sha256=context.identity["population_sha256"],
            input_binding_sha256=context.identity["input_binding_sha256"],
            fit_uids=context.population["fit"][:fit_users],
            validation_users=context.identity["validation_uids"],
            candidate_source=recipe, calibration_candidate_sha256="recipe-" + recipe, seed=17))
    if mutate:
        mutate(probe, metadata)
    path = tmp_path / f"{variant}_{fit_users}_{recipe}_{method_label}.pt"
    save_variant(path, probe, metadata)
    return path, probe, metadata


def test_fixed_candidates_before_targets_and_shared_fit_only_scales(tmp_path, monkeypatch):
    fx = fixture(tmp_path, monkeypatch)
    context = driver.prepare_common_validation(fx.root, 1, torch.device("cpu"))
    assert len(context.fit_samples) == 64 and len(context.validation_samples) == 16
    assert context.identity["scale_fit_uids"] == fx.population["fit"][:64]
    assert context.identity["fitting_candidate_pool_uids"] == fx.population["fit"][:256]
    first = context.fit_samples[0]
    read = sum(s["read_moment_sum"].double() for s in context.fit_samples) / 64
    score_var = sum(s["score_moment_sum"].double() for s in context.fit_samples) / 64
    expected_read = read.sqrt().clamp_min(1e-4).float()
    expected_score = score_var.sqrt().clamp_min(1e-4).float()
    for sample in context.fit_samples + context.validation_samples:
        torch.testing.assert_close(sample["read_scales"][:, 0, :, 0, 0], expected_read)
        torch.testing.assert_close(sample["score_scale"], expected_score)
        assert sample["read_scales"].data_ptr() == first["read_scales"].data_ptr()
        wanted, _ = score(context.model, sample["teacher_cache"], sample["candidates"], sample["query_delta"])
        torch.testing.assert_close(sample["teacher_scores"], wanted)
        assert sample["candidates"][0].tolist() == candidates_for(sample["uid"],
            fx.candidate_data["recent_items"][str(sample["uid"])], 64, count=16, recent_fraction=.5, seed=17)
    assert {s["parent_cache"].seq_len for s in context.fit_samples} == {2, 5}
    native = driver.native_baseline(context)
    zero_difference = native["metrics"]["per_user"][0]
    assert zero_difference["loss"] == 0.
    assert native["metrics"]["loss"] > 0.
    assert all(np.isfinite(row["loss"]) for row in native["metrics"]["per_user"])
    assert context.identity["candidate_file_sha256"] == sha256(fx.edge / "calibration_feedback_candidates.json")


@pytest.mark.parametrize("variant", ["head", "full_item", "layerwise_kv"])
def test_typed_weight_uses_common_loss_and_preserves_trained_statistics(tmp_path, monkeypatch, variant):
    fx = fixture(tmp_path, monkeypatch)
    context = driver.prepare_common_validation(fx.root, 1, torch.device("cpu"))
    path, original, metadata = make_weight(tmp_path, context, variant=variant)
    before_hash = sha256(path)
    row = driver.evaluate_checkpoint(context, path)
    assert row["status"] == "validated" and row["variant"] == variant
    assert row["step_zero"] and row["best_step"] == 0
    assert row["original_validation_loss"] == 999.
    assert row["original_objective"] == dict(read_weight=.1, score_weight=1.)
    assert row["common_validation_sha256"] == context.fingerprint
    expected = []
    for sample in context.validation_samples:
        loss, parts = probe_loss(context.model, original, sample_to_device(sample, "cpu"),
                                 read_weight=1., score_weight=1.)
        expected.append(float(loss.detach()))
    assert row["metrics"]["loss"] == pytest.approx(np.mean(expected), rel=1e-7)
    assert row["metrics"]["loss"] == pytest.approx(
        row["metrics"]["read_mse"] + row["metrics"]["score_mse"], rel=1e-7)
    reloaded, _ = load_variant(path)
    assert sha256(path) == before_hash
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(tensor, reloaded.state_dict()[name], rtol=0, atol=0)


def test_different_recipes_fitcounts_and_original_scales_have_identical_common_loss(tmp_path, monkeypatch):
    fx = fixture(tmp_path, monkeypatch)
    context = driver.prepare_common_validation(fx.root, 1, torch.device("cpu"))
    first, probe, metadata = make_weight(tmp_path, context, best_step=0)
    changed = copy.deepcopy(metadata)
    changed.update(best_step=32, best_validation_loss=.001, fit_pool_loss_scales=dict(score_scale=100.))
    changed["identity"].update(fit_uids=context.population["fit"][:256],
        candidate_source="fitting_pool", calibration_candidate_sha256="different-candidates")
    second = tmp_path / "same_weights_other_training_recipe.pt"
    save_variant(second, probe, changed)
    a, b = [driver.evaluate_checkpoint(context, path) for path in (first, second)]
    assert a["metrics"] == b["metrics"]
    assert (a["fitting_users"], b["fitting_users"]) == (64, 256)
    assert a["step_zero"] and not b["step_zero"]
    assert a["original_validation_loss"] != b["original_validation_loss"]


@pytest.mark.parametrize("mismatch", ["edge", "models", "validation_users", "fit_uids"])
def test_incompatible_weight_retained_as_failed(tmp_path, monkeypatch, mismatch):
    fx = fixture(tmp_path, monkeypatch)
    context = driver.prepare_common_validation(fx.root, 1, torch.device("cpu"))

    def mutate(probe, metadata):
        values = dict(edge=2, models={"different": "hash"}, validation_users=[991], fit_uids=[991])
        metadata["identity"][mismatch] = values[mismatch]

    path, _, _ = make_weight(tmp_path, context, mutate=mutate)
    row = driver.evaluate_checkpoint(context, path)
    assert row["status"] == "failed" and not row["selection_eligible"]
    assert row["error_type"] == "ValueError" and "metrics" not in row
    assert row["checkpoint_sha256"] == sha256(path) and row["best_step"] == 0


def test_nonfinite_and_diagnostic_weights_are_not_silently_eligible(tmp_path, monkeypatch):
    fx = fixture(tmp_path, monkeypatch)
    context = driver.prepare_common_validation(fx.root, 1, torch.device("cpu"))

    def nonfinite(probe, metadata):
        with torch.no_grad():
            probe.h[0].weights[-1].fill_(float("nan"))

    bad, _, _ = make_weight(tmp_path, context, mutate=nonfinite)
    row = driver.evaluate_checkpoint(context, bad)
    assert row["status"] == "failed" and "nonfinite" in row["error"]
    diagnostic, _, _ = make_weight(tmp_path, context, recipe="diagnostic",
        mutate=lambda p, m: m.update(diagnostic_only=True, quality_use_permitted=False))
    row = driver.evaluate_checkpoint(context, diagnostic)
    assert row["status"] == "validated" and not row["selection_eligible"]


def test_existing_untagged_head_format_and_nonfinite_original_loss_are_retained(tmp_path, monkeypatch):
    fx = fixture(tmp_path, monkeypatch)
    context = driver.prepare_common_validation(fx.root, 1, torch.device("cpu"))
    _, probe, metadata = make_weight(tmp_path, context, variant="head", original_loss=float("inf"))
    # This is the concrete older run_probes.py format, without a variant tag.
    from dataclasses import asdict
    path = tmp_path / "old_head.pt"
    torch.save(dict(config=asdict(probe.cfg), stage=probe.stage,
                    state_dict=probe.state_dict(), metadata=metadata), path)
    row = driver.evaluate_checkpoint(context, path)
    assert row["status"] == "validated" and row["variant"] == "head"
    assert row["original_validation_loss"] == "inf"
    json.dumps(row, allow_nan=False)


def test_corrupted_checkpoint_and_changed_numerical_definition_are_failed_rows(tmp_path, monkeypatch):
    fx = fixture(tmp_path, monkeypatch)
    context = driver.prepare_common_validation(fx.root, 1, torch.device("cpu"))
    broken = tmp_path / "truncated.pt"
    broken.write_bytes(b"not a torch checkpoint")
    assert driver.evaluate_checkpoint(context, broken)["status"] == "failed"

    def mutate(probe, metadata):
        metadata["identity"]["numerical_sources"] = dict(metadata["identity"]["numerical_sources"])
        metadata["identity"]["numerical_sources"]["experiments/ch3_motivation/code/probes.py"] = "different"

    bad_source, _, _ = make_weight(tmp_path, context, mutate=mutate)
    row = driver.evaluate_checkpoint(context, bad_source)
    assert row["status"] == "failed" and "numerical source" in row["error"]


def test_cpu_driver_writes_all_points_hashes_and_native_baseline(tmp_path, monkeypatch):
    fx = fixture(tmp_path, monkeypatch)
    context = driver.prepare_common_validation(fx.root, 1, torch.device("cpu"))
    good, _, _ = make_weight(tmp_path, context, method_label="fg_extended")
    absent = tmp_path / "absent.pt"
    # Preparation sharing is tested through the real context above. This also
    # ensures a many-weight run calls the preparation API exactly once.
    calls = []

    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return context

    monkeypatch.setattr(driver, "prepare_common_validation", prepare)
    args = SimpleNamespace(input_root=fx.root, edge=1, device="cpu", gpu=0,
        backend="torch", checkpoint=[good, absent], output=tmp_path / "output")
    summary = driver.run(args)
    assert len(calls) == 1
    assert summary["status"] == "complete_with_failures"
    assert (summary["validated"], summary["failed"]) == (1, 1)
    assert summary["points"][0]["method_label"] == "fg_extended"
    assert summary["points"][1]["status"] == "failed"
    assert summary["points"][1]["error_type"] == "FileNotFoundError"
    assert summary["quality_auc_computed"] is False
    assert summary["post_release_labels_loaded"] is False
    assert summary["fitting_performed"] is False
    saved = json.loads((args.output / "summary.json").read_text())
    assert saved == summary
    assert summary["native_reuse"]["common_validation_sha256"] == context.fingerprint
    assert all(len(digest) == 64 for digest in summary["common_validation"]["numerical_sources"].values())


@pytest.mark.parametrize("tamper", ["history", "candidate_roles", "source_candidates"])
def test_input_tamper_rejected_before_model_loading(tmp_path, monkeypatch, tamper):
    fx = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(driver, "load_models", lambda *a, **k: pytest.fail("must reject before model loading"))
    if tamper == "history":
        (fx.edge / "history.pack").write_text("changed")
    elif tamper == "candidate_roles":
        fx.candidate_data["fitting_pool_uids"] = fx.population["fit"][:64]
        (fx.edge / "calibration_feedback_candidates.json").write_text(json.dumps(fx.candidate_data))
    else:
        Path(fx.candidate_data["source_candidate_path"]).write_text("changed")
    with pytest.raises(ValueError):
        driver.prepare_common_validation(fx.root, 1, torch.device("cpu"))
