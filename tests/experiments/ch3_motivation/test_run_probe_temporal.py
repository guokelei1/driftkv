"""Temporal driver orchestration with real caches/targets and CPU tiny models."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import run_probe_temporal as driver
from experiments.ch3_motivation.code.probes import bind_fit_scales, native_query_flops
from experiments.ch3_motivation.code.probe_variants import build_variant, load_variant
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.common.artifacts import sha256

REAL_LOAD_GAP = driver.calibration_gaps.load_gap_manifest
REAL_APPLY_GAP = driver.calibration_gaps.apply_gap_view


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def setup(tmp_path, monkeypatch, rule="gap_mix50"):
    root = tmp_path / "fit64"
    directory = root / "edge_1"
    directory.mkdir(parents=True)
    population = dict(fit=list(range(1000, 1512)), validation=list(range(2000, 2016)))
    (root / "population.json").write_text(json.dumps(population))
    population_sha = sha256(root / "population.json")
    for name in ("history", "requests"):
        (directory / f"{name}.pack").write_text(name)
    binding = dict(population_sha256=population_sha,
        settings=dict(edge=1, release_day=231, history_length=8, seed=17,
            calibration_queries_per_user=16, attention_backend="torch"),
        **{name: dict(path=f"{name}.pack", sha256=sha256(directory / f"{name}.pack"))
           for name in ("history", "requests")})
    (directory / "inputs.json").write_text(json.dumps(binding))
    uids = population["fit"][:64] + population["validation"]
    candidates = dict(population_sha256=population_sha, known_items=64,
        fitting_pool_uids=population["fit"][:64], validation_uids=population["validation"],
        recent_items={str(uid): list(range(64, 48, -1)) for uid in uids})
    (directory / "calibration_feedback_candidates.json").write_text(json.dumps(candidates))
    manifest = tmp_path / "gap_manifest.json"
    manifest.write_text(json.dumps(dict(rule=rule, source_cutoff=231 * 86400)))
    torch.manual_seed(812)
    model_cfg = HSTUConfig(num_items=320, num_behaviors=2, hidden_size=64, num_heads=2,
        num_layers=2, max_seq_len=8, input_dropout=0., attn_dropout=0.)
    parent = HSTU(model_cfg).eval().requires_grad_(False)
    current = copy.deepcopy(parent)
    with torch.no_grad():
        current.blocks[0].attn.k_proj.weight.add_(torch.randn_like(current.blocks[0].attn.k_proj.weight) * .03)
    histories = SimpleNamespace(rows={})
    for uid in uids:
        n = 2 if uid % 2 else 5
        histories.rows[uid] = (231 * 86400 - 100000 - np.arange(n-1, -1, -1),
            np.arange(1, n+1), np.ones(n, dtype=np.int64))
    events, fits, prepared, gaps = [], [], [], {}
    for uid in uids:
        if rule == "gap_mix50":
            values = [100000.] * 4 + [1., 2., 3., 4.] + [100000.] * 4 + [5., 6., 7., 8.]
        else:
            values = list(range(1, 17))
        gaps[uid] = torch.tensor([values], dtype=torch.float32)
    provenance = dict(rule=rule, source_cutoff=231 * 86400,
        time_rule="strict previous behavior, fitting-only past feedback law",
        manifest_sha256=sha256(manifest), candidate_order=dict(feedback=list(range(8)), uniform=list(range(8,16))),
        cpu_law_construction_seconds=.02, cpu_sampling_seconds=.01)

    def load_gap(path, input_root, edge, requested_rule):
        assert Path(path) == manifest and Path(input_root) == root and edge == 1
        events.append("gap_manifest_loaded")
        return gaps, dict(provenance, rule=requested_rule)

    real_capture, real_prepare = driver.capture_samples, driver.prepare_training_samples

    def capture(*args, **kwargs):
        values = real_capture(*args, **kwargs)
        for sample in values.values():
            assert sample["query_delta"].shape == (1,)
            sample["teacher_scores"] = torch.full((1, 16), 1e20)
            sample["target_moment_version"] = "stale"
        events.append("capture")
        return values

    def apply(samples, views, gap_provenance):
        result = {}
        for uid, sample in samples.items():
            assert sample["candidates"][0, :8].tolist() == list(range(64, 56, -1))
            assert sample["query_delta"].shape == (1,)
            changed = dict(sample, query_delta=views[uid].clone())
            for key in ("teacher_scores", "target_moment_version"):
                changed.pop(key, None)
            result[uid] = changed
        events.append("gap_before_targets")
        return result

    def prepare(model, samples):
        assert "gap_before_targets" in events
        for sample in samples:
            assert sample["query_delta"].shape == (1, 16)
            assert "teacher_scores" not in sample
            torch.testing.assert_close(sample["query_delta"], gaps[sample["uid"]])
        result = real_prepare(model, samples)
        for sample in result:
            wanted, _ = score(model, sample["teacher_cache"], sample["candidates"], sample["query_delta"])
            torch.testing.assert_close(sample["teacher_scores"], wanted)
        prepared.extend(result)
        events.append(("targets", len(samples)))
        return result

    def fit(model, module, train, validation, *, steps, progress_callback, **kwargs):
        assert len(train) == 64 and len(validation) == 16
        assert kwargs["learning_rate"] == .00003 and kwargs["validation_every"] == 8
        assert kwargs["patience"] == 128 and kwargs["read_weight"] == kwargs["score_weight"] == 1.
        assert all(sample["query_delta"].shape == (1, 16) for sample in train + validation)
        _, _, scales = bind_fit_scales(model, train, validation)
        weight = module.f[0].weights[-1]
        before = float(weight.detach().flatten()[0])
        with torch.no_grad():
            weight.add_(steps * 1e-6)
        fits.append(dict(stage=module.stage, before=before, after=float(weight.detach().flatten()[0]), steps=steps))
        progress_callback(dict(step=steps, validation_loss=1.))
        return dict(stage=module.stage, config=asdict(module.cfg), seed=kwargs["seed"], best_step=steps,
            completed_steps=steps, training_users=len(train), validation_users=len(validation),
            objective=dict(read_weight=1., score_weight=1.), fit_pool_loss_scales=scales,
            total_calibration_flops_estimate=10 * steps, history=[])

    monkeypatch.setattr(driver, "load_models", lambda *a, **k: (parent, current, {"old": "sha-old", "new": "sha-new"}))
    monkeypatch.setattr(driver, "restore_pack", lambda _: histories)
    monkeypatch.setattr(driver, "capture_samples", capture)
    monkeypatch.setattr(driver, "prepare_training_samples", prepare)
    monkeypatch.setattr(driver.calibration_gaps, "load_gap_manifest", load_gap)
    monkeypatch.setattr(driver.calibration_gaps, "apply_gap_view", apply)
    monkeypatch.setattr(driver, "fit_probe", fit)
    args = SimpleNamespace(device="cpu", gpu=0, input_root=root, output=tmp_path / "output", edge=1,
        gap_rule=rule, gap_manifest=manifest, seed=17, controls=None)
    return SimpleNamespace(args=args, events=events, fits=fits, prepared=prepared, gaps=gaps,
        provenance=provenance, current=current, parent=parent, binding=binding, histories=histories)


@pytest.mark.parametrize("rule,labels", [
    ("gap_mix50", ["f", "f_extended", "fg", "fg_extended", "fgh"]),
    ("gap_empirical100", ["f", "fg", "fgh"])])
def test_gap_before_targets_identity_nested_controls_and_complete_cost(tmp_path, monkeypatch, rule, labels):
    fx = setup(tmp_path, monkeypatch, rule)
    result = driver.run(fx.args)
    assert [row["method_label"] for row in result["stages"]] == labels
    assert fx.events[:3] == ["gap_manifest_loaded", "capture", "gap_before_targets"]
    assert fx.events[3:] == [("targets", 64), ("targets", 16)]
    identity = result["identity"]
    assert identity["fitting_driver"] == "run_probe_temporal.py"
    assert identity["fitting_driver_sha256"] == sha256(Path(driver.__file__))
    assert identity["query_delta_rule"] == rule and identity["gap_manifest_sha256"] == sha256(fx.args.gap_manifest)
    assert identity["source_cutoff"] == 231 * 86400 and identity["candidate_order"] == fx.provenance["candidate_order"]
    prep = result["calibration_preparation"]
    assert prep["query_delta_matrix_bytes"] == 80 * 16 * 4
    assert prep["release_gap_scalar_bytes"] == 80 * 4
    assert prep["additional_delta_input_bytes"] == 80 * 15 * 4
    shared = prep["total_model_preparation_flops"]
    assert [row["cumulative_calibration_flops_estimate"] - shared for row in result["stages"]] == (
        [5120, 15360, 10240, 15360, 15360] if rule == "gap_mix50" else [5120, 10240, 15360])
    expected_before = [0., .000512, .000512, .001024, .001024] if rule == "gap_mix50" else [0., .000512, .001024]
    assert [row["before"] for row in fx.fits] == pytest.approx(expected_before, rel=1e-6)
    for label, summary in zip(labels, result["stages"], strict=True):
        probe, metadata = load_variant(fx.args.output / f"{label}.pt")
        assert metadata["identity"] == identity and metadata["quality_auc_computed"] is False
        assert metadata["method_label"] == label and metadata["gap_provenance"]["rule"] == rule
        assert summary["checkpoint_sha256"] == sha256(fx.args.output / f"{label}.pt")


def test_scalar_and_matrix_native_work_same_formula_and_per_candidate_path(tmp_path, monkeypatch):
    fx = setup(tmp_path, monkeypatch)
    driver.run(fx.args)
    sample = fx.prepared[0]
    fixed = dict(sample, query_delta=torch.full((1,), 100000.))
    constant_matrix = fixed["query_delta"][:, None].expand(1, 16).clone()
    torch.testing.assert_close(fx.current.embed_query_tokens(sample["candidates"], fixed["query_delta"]),
                               fx.current.embed_query_tokens(sample["candidates"], constant_matrix), rtol=0, atol=0)
    a = driver.preparation_cost(fx.current, [sample], fx.binding["settings"], manifest_bytes=17)
    b = driver.preparation_cost(fx.current, [fixed], fx.binding["settings"], manifest_bytes=17)
    assert a["total_model_preparation_flops"] == b["total_model_preparation_flops"]
    assert a["teacher_and_old_target_score_flops"] == 2 * native_query_flops(fx.current,
        history_length=sample["history_count"], queries=16)
    assert a["additional_delta_input_bytes"] - b["additional_delta_input_bytes"] == 60
    matrix_teacher, _ = score(fx.current, sample["teacher_cache"], sample["candidates"], sample["query_delta"])
    release_teacher, _ = score(fx.current, sample["teacher_cache"], sample["candidates"], fixed["query_delta"])
    assert float((matrix_teacher - release_teacher).abs().max()) > 1e-6


@pytest.mark.parametrize("stage", ["f", "fg", "fgh"])
def test_nonzero_probe_matrix_queries_equal_independent_single_queries(tmp_path, monkeypatch, stage):
    fx = setup(tmp_path, monkeypatch)
    driver.run(fx.args)
    sample = fx.prepared[0]
    torch.manual_seed(123)
    probe = build_variant("full_item", fx.current, stage=stage, hidden_width=7, depth=1,
                          query_chunk=16, row_chunk=8)
    probe.fit_input_statistics(fx.current, [sample])
    with torch.no_grad():
        for group in (probe.f, probe.g, probe.h):
            if group is not None:
                for layer in group:
                    layer.weights[-1].normal_(0, .002)
        batched, trace = probe.score(fx.current, sample["parent_cache"], sample["item_ids"],
                                    sample["candidates"], sample["query_delta"], trace=True)
        singles, reads = [], [[] for _ in fx.current.blocks]
        for index in range(16):
            value, details = probe.score(fx.current, sample["parent_cache"], sample["item_ids"],
                sample["candidates"][:, index:index+1], sample["query_delta"][:, index:index+1], trace=True)
            singles.append(value)
            for layer, read in enumerate(details.history_heads):
                reads[layer].append(read)
        torch.testing.assert_close(batched, torch.cat(singles, 1), atol=1e-6, rtol=1e-5)
        for layer, wanted in enumerate(reads):
            torch.testing.assert_close(trace.history_heads[layer], torch.cat(wanted, 2), atol=1e-6, rtol=1e-5)


def test_resume_uses_only_same_temporal_identity_and_retains_original_weights(tmp_path, monkeypatch):
    fx = setup(tmp_path, monkeypatch)
    first = driver.run(fx.args)
    hashes = {label: sha256(fx.args.output / f"{label}.pt") for label in ("f", "fg", "fgh", "f_extended", "fg_extended")}
    monkeypatch.setattr(driver, "fit_probe", lambda *a, **k: pytest.fail("finished temporal branches must not be refitted"))
    second = driver.run(fx.args)
    assert second["identity"] == first["identity"]
    assert all(sha256(fx.args.output / f"{label}.pt") == digest for label, digest in hashes.items())
    changed = copy.copy(fx.args)
    changed.gap_rule = "gap_empirical100"
    with pytest.raises(ValueError, match="configuration changed"):
        driver.run(changed)


def test_future_cutoff_rejected_before_model_loading(tmp_path, monkeypatch):
    fx = setup(tmp_path, monkeypatch)
    fx.provenance["source_cutoff"] += 1
    monkeypatch.setattr(driver, "load_models", lambda *a, **k: pytest.fail("must reject causal binding first"))
    with pytest.raises(ValueError, match="cutoff"):
        driver.run(fx.args)


def test_real_gap_manifest_loading_and_application_before_teacher_targets(tmp_path, monkeypatch):
    fx = setup(tmp_path, monkeypatch)
    helper = driver.calibration_gaps
    root = fx.args.input_root
    population = json.loads((root / "population.json").read_text())
    candidate_path = root / "edge_1/calibration_feedback_candidates.json"
    candidate_data = json.loads(candidate_path.read_text())
    fit, validation = population["fit"][:64], population["validation"]
    cutoff = 231 * 86400
    behavior = {uid: fx.histories.rows[uid][0].tolist() for uid in fit}
    feedback = {uid: [cutoff-99950, cutoff-99900, cutoff+1] for uid in fit}
    feedback[validation[0]] = [cutoff-2]  # Never contributes to the shared law.
    law = helper.fit_gap_law(feedback, behavior, fit, cutoff)
    assert law["coverage"]["observed_feedback_instances"] == 128
    assert set(law["users"]) == set(fit)
    views = helper.make_views(fit, validation, candidate_data,
        {uid: 100000. for uid in fit + validation}, law, edge=1, cutoff=cutoff)
    binding_path = root / "edge_1/inputs.json"
    history_path = root / "edge_1/history.pack"
    payload = dict(schema_version=1, status="prepared", edge=1, release_day=231,
        source_cutoff=cutoff, time_rule=helper.TIME_RULE, fit_uids=fit, validation_uids=validation,
        candidate_order="indices0..7 feedback,8..15 uniform", users=views, gap_law=law,
        input_binding=dict(input_root=str(root.resolve()), population_sha256=sha256(root / "population.json"),
            sha256=sha256(binding_path), candidate_sha256=sha256(candidate_path),
            history_path=str(history_path), history_sha256=sha256(history_path)),
        sources=dict(source_sha256={str(Path(helper.__file__).resolve().relative_to(ROOT.resolve())):
                                   sha256(Path(helper.__file__))}),
        cpu_seconds=dict(fit_law=.001, assignments=.001), scope="synthetic CPU integration fixture")
    fx.args.gap_manifest.write_text(json.dumps(payload))
    actual, _ = REAL_LOAD_GAP(fx.args.gap_manifest, root, 1, "gap_mix50")
    fx.gaps.clear()
    fx.gaps.update(actual)
    for values in actual.values():
        assert int((values[0, :8] == 100000).sum()) == 4
        assert int((values[0, 8:] == 100000).sum()) == 4

    def load(*args):
        fx.events.append("gap_manifest_loaded")
        return REAL_LOAD_GAP(*args)

    def apply(*args):
        result = REAL_APPLY_GAP(*args)
        fx.events.append("gap_before_targets")
        return result

    monkeypatch.setattr(helper, "load_gap_manifest", load)
    monkeypatch.setattr(helper, "apply_gap_view", apply)
    result = driver.run(fx.args)
    assert result["gap_provenance"]["gap_law_coverage"]["observed_feedback_instances"] == 128
    assert result["identity"]["gap_manifest_sha256"] == sha256(fx.args.gap_manifest)
    assert all(sample["query_delta"].shape == (1,16) for sample in fx.prepared)


def test_real_two_step_training_has_finite_matrix_gap_gradients(tmp_path, monkeypatch):
    fx = setup(tmp_path, monkeypatch)
    driver.run(fx.args)
    from experiments.ch3_motivation.code.probes import fit_probe
    probe = build_variant("full_item", fx.current, stage="fgh", hidden_width=7,
                          depth=1, query_chunk=16, row_chunk=8)
    result = fit_probe(fx.current, probe, fx.prepared[:64], fx.prepared[64:],
        steps=2, learning_rate=.00003, seed=17, validation_every=1, patience=0)
    assert result["completed_steps"] == 2
    trained = [row for row in result["history"] if row["step"] > 0]
    assert len(trained) == 2 and all(np.isfinite(row["training_loss"]) for row in trained)
    assert all(np.isfinite(row["gradient_norm"]) and row["gradient_norm"] > 0 for row in trained)
