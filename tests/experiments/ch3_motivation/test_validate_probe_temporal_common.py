"""Three-context common targets/scales, immutable weights and provenance gates."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import validate_probe_temporal_common as driver
from experiments.ch3_motivation.code.calibration_gaps import fit_gap_law, make_views, TIME_RULE
from experiments.ch3_motivation.code.prepare_temporal_common_plan import RULES
from experiments.ch3_motivation.code.probe_variants import build_variant, load_variant, save_variant
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.common.artifacts import sha256


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "fit64"
    folder = root / "edge_1"
    folder.mkdir(parents=True)
    fit, val = list(range(1000, 1064)), list(range(2000, 2016))
    population = dict(fit=fit + list(range(3000, 3448)), validation=val)
    (root / "population.json").write_text(json.dumps(population))
    for name in ("history", "requests"):
        (folder / f"{name}.pack").write_text(name)
    cutoff = 231 * 86400
    binding = dict(population_sha256=sha256(root / "population.json"),
        settings=dict(edge=1, release_day=231, history_length=8, seed=17,
                      calibration_queries_per_user=16, attention_backend="torch"),
        **{name: dict(path=f"{name}.pack", sha256=sha256(folder / f"{name}.pack"))
           for name in ("history", "requests")})
    (folder / "inputs.json").write_text(json.dumps(binding))
    source = tmp_path / "source_candidates.json"
    source.write_text("fitting-only observed item IDs, without labels")
    candidates = dict(known_items=64, fitting_pool_uids=fit, validation_uids=val,
        population_sha256=sha256(root / "population.json"),
        recent_items={str(uid): list(range(64, 48, -1)) for uid in fit + val},
        source_candidate_path=str(source), source_candidate_sha256=sha256(source))
    candidate_path = folder / "calibration_feedback_candidates.json"
    candidate_path.write_text(json.dumps(candidates))
    release = {uid: 100000. for uid in fit + val}
    law = fit_gap_law({uid: [cutoff - 500, cutoff - 200] for uid in fit},
        {uid: [cutoff - 1000, cutoff - 600] for uid in fit}, fit, cutoff)
    manifest = dict(schema_version=1, status="prepared", edge=1, source_cutoff=cutoff,
        release_day=231, time_rule=TIME_RULE, fit_uids=fit, validation_uids=val,
        candidate_order="first8feedback_then8uniform", gap_law=law,
        users=make_views(fit, val, candidates, release, law, edge=1, cutoff=cutoff),
        input_binding=dict(input_root=str(root), population_path=str(root / "population.json"),
            population_sha256=sha256(root / "population.json"), path=str(folder / "inputs.json"),
            sha256=sha256(folder / "inputs.json"), candidate_path=str(candidate_path),
            candidate_sha256=sha256(candidate_path), history_path=str(folder / "history.pack"),
            history_sha256=sha256(folder / "history.pack")),
        sources=dict(source_sha256={}), cpu_seconds=dict(fit_law=.01, assignments=.01), scope="CPU fixture")
    manifest_path = tmp_path / "gap_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    torch.manual_seed(812)
    cfg = HSTUConfig(num_items=320, num_behaviors=2, hidden_size=8, num_heads=2,
        num_layers=2, max_seq_len=8, input_dropout=0., attn_dropout=0.)
    parent = HSTU(cfg).eval().requires_grad_(False)
    model = copy.deepcopy(parent)
    with torch.no_grad():
        model.blocks[0].attn.k_proj.weight.add_(torch.randn_like(model.blocks[0].attn.k_proj.weight) * .03)
    histories = SimpleNamespace(rows={})
    for uid in fit + val:
        n = 2 if uid % 2 else 5
        histories.rows[uid] = (np.r_[cutoff - 100000 - np.arange(n-1, -1, -1), cutoff + 1],
            np.r_[np.arange(1, n+1), 63], np.ones(n+1, dtype=np.int64))
    real_capture = driver.capture_samples
    captures = []

    def capture(*args, **kwargs):
        samples = real_capture(*args, **kwargs)
        for sample in samples.values():
            assert 63 not in sample["item_ids"]
            sample.update(teacher_scores=torch.full((1,16), 1e20), target_moment_version="stale",
                          read_scales="stale", score_scale="stale")
        # A validation user with zero teacher difference must remain finite.
        samples[val[0]]["teacher_cache"] = samples[val[0]]["parent_cache"]
        captures.append(samples)
        return samples

    monkeypatch.setattr(driver, "load_models", lambda *a, **k: (parent, model, {"old": "old-sha", "new": "new-sha"}))
    monkeypatch.setattr(driver, "restore_pack", lambda _: histories)
    monkeypatch.setattr(driver, "capture_samples", capture)
    return SimpleNamespace(root=root, manifest=manifest_path, model=model, fit=fit, val=val,
                           captures=captures, candidates=candidates)


def contexts(fx):
    return driver.prepare_contexts(fx.root, 1, torch.device("cpu"), fx.manifest)


def make_weight(tmp_path, contexts, *, temporal=False, step=0, mutate=None):
    context = contexts["cutover"]
    probe = build_variant("full_item", context.model, stage="fgh", hidden_width=5, depth=1,
                          query_chunk=16, row_chunk=8)
    with torch.no_grad():
        probe.statistics_initialized.fill_(True)
        probe.row_scale.fill_(2.)
        probe.query_scale.fill_(1.5)
        for group in (probe.f, probe.g, probe.h):
            for layer in group:
                layer.weights[-1].normal_(0, .0002)
    identity = dict(edge=1, seed=17, models=context.identity["models"],
        population_sha256=context.identity["population_sha256"],
        input_binding_sha256=context.identity["input_binding_sha256"],
        fit_uids=context.population["fit"][:64], validation_users=context.identity["validation_uids"],
        numerical_sources=context.identity["numerical_sources"])
    spec = dict(origin="temporal_40" if temporal else "release_gap_240", recipe="fixture",
        edge=1, seed=17, method_label="fgh", fitting_driver="run_probe_temporal.py" if temporal else "run_probe_variants.py")
    if temporal:
        identity.update(fitting_driver=spec["fitting_driver"],
            fitting_driver_sha256=sha256(ROOT / "experiments/ch3_motivation/code/run_probe_temporal.py"),
            query_delta_rule="gap_mix50", calibration_candidate_sha256=context.identity["candidate_file_sha256"],
            **{key: context.identity[key] for key in ("gap_manifest_sha256", "time_rule", "source_cutoff", "candidate_order")})
        spec.update(query_delta_rule="gap_mix50", training_gap_manifest_sha256=context.identity["gap_manifest_sha256"])
    metadata = dict(identity=identity, method_label="fgh", best_step=step, completed_steps=32,
        best_validation_loss=float("nan"), cumulative_calibration_flops_estimate=12345,
        fit_pool_loss_scales=dict(score_scale=.0001))
    if mutate:
        mutate(probe, metadata)
    path = tmp_path / ("temporal.pt" if temporal else "old.pt")
    save_variant(path, probe, metadata)
    return dict(spec, checkpoint=str(path)), probe


def test_pooled_three_context_fit_only_scales_and_cache_sharing(tmp_path, monkeypatch):
    fx = fixture(tmp_path, monkeypatch)
    values = contexts(fx)
    assert list(values) == RULES and len(fx.captures) == 1
    all_fit = [s for context in values.values() for s in context.fit_samples]
    expected_read = (sum(s["read_moment_sum"].double() for s in all_fit) / 192).sqrt().clamp_min(1e-4).float()
    expected_score = (sum(s["score_moment_sum"].double() for s in all_fit) / 192).sqrt().clamp_min(1e-4).float()
    first = all_fit[0]
    for rule, context in values.items():
        assert len(context.fit_samples) == 64 and len(context.validation_samples) == 16
        assert context.identity["loss_scales"]["users"] == 192
        assert context.identity["loss_scales"]["unique_fitting_users"] == 64
        for sample in context.fit_samples + context.validation_samples:
            torch.testing.assert_close(sample["read_scales"][:,0,:,0,0], expected_read)
            torch.testing.assert_close(sample["score_scale"], expected_score)
            assert sample["read_scales"].data_ptr() == first["read_scales"].data_ptr()
            target, _ = score(fx.model, sample["teacher_cache"], sample["candidates"], sample["query_delta"])
            torch.testing.assert_close(sample["teacher_scores"], target)
        zero_user = driver.native_baseline(context)["metrics"]["per_user"][0]
        assert zero_user["loss"] == 0.
    uid = fx.fit[0]
    assert len({id(c.fit_samples[0]["parent_cache"]) for c in values.values()}) == 1
    assert len({c.fit_samples[0]["query_delta"].data_ptr() for c in values.values()}) == 3
    assert torch.unique(values["cutover"].fit_samples[0]["query_delta"]).tolist() == [100000.]
    assert (values["gap_mix50"].fit_samples[0]["query_delta"] == 100000).sum() == 8
    assert (values["gap_empirical100"].fit_samples[0]["query_delta"] < 100000).all()
    assert fx.captures[0][uid]["teacher_scores"].max() > 1e19


def test_old_and_temporal_same_weights_same_new_loss_preserve_step0_and_stats(tmp_path, monkeypatch):
    fx = fixture(tmp_path, monkeypatch)
    values = contexts(fx)
    torch.manual_seed(100)
    old_spec, probe = make_weight(tmp_path, values)
    torch.manual_seed(100)
    temporal_spec, _ = make_weight(tmp_path, values, temporal=True)
    old_hash = sha256(Path(old_spec["checkpoint"]))
    old = driver.evaluate_planned_weight(values, old_spec)
    new = driver.evaluate_planned_weight(values, temporal_spec)
    assert old["status"] == new["status"] == "validated"
    assert old["step_zero"] and new["step_zero"]
    assert old["original_validation_loss"] == "nan"
    assert old["selection_eligible"] and old["mean_context_loss"] == new["mean_context_loss"]
    assert list(old["contexts"]) == RULES
    for rule in RULES:
        assert old["contexts"][rule]["metrics"] == new["contexts"][rule]["metrics"]
    reloaded, _ = load_variant(Path(old_spec["checkpoint"]))
    for name, tensor in probe.state_dict().items():
        torch.testing.assert_close(tensor, reloaded.state_dict()[name], rtol=0, atol=0)
    assert sha256(Path(old_spec["checkpoint"])) == old_hash


@pytest.mark.parametrize("fault", ["candidate", "manifest", "driver", "seed", "time_rule"])
def test_temporal_identity_mismatch_retained_as_failed(tmp_path, monkeypatch, fault):
    fx = fixture(tmp_path, monkeypatch)
    values = contexts(fx)
    key = {"candidate": "calibration_candidate_sha256", "manifest": "gap_manifest_sha256",
           "driver": "fitting_driver", "seed": "seed", "time_rule": "time_rule"}[fault]
    spec, _ = make_weight(tmp_path, values, temporal=True,
                          mutate=lambda p, m: m["identity"].update({key: "changed"}))
    row = driver.evaluate_planned_weight(values, spec)
    assert row["status"] == "failed" and not row["selection_eligible"]
    assert row["step_zero"] and row["contexts"] == {} and "error" in row


def test_missing_checkpoint_is_pending_not_validated(tmp_path):
    row = driver.evaluate_planned_weight({}, dict(checkpoint=str(tmp_path / "missing.pt")))
    assert row["status"] == "pending" and not row["selection_eligible"]


@pytest.mark.parametrize("fault", ["future_cutoff", "fallback", "candidate_order", "wrong_fit_role"])
def test_common_context_rejects_changed_causality_or_candidate_binding(tmp_path, monkeypatch, fault):
    fx = fixture(tmp_path, monkeypatch)
    manifest = json.loads(fx.manifest.read_text())
    if fault == "future_cutoff":
        manifest["source_cutoff"] += 1
    elif fault == "fallback":
        manifest["gap_law"]["fallback_used"] = True
    elif fault == "candidate_order":
        # Preserve the loader's internal index/order check; the installed real
        # candidate IDs must still disagree and be rejected by apply_gap_view.
        user = manifest["users"][str(fx.fit[0])]
        user["candidate_ids"][0] = 1
        for rows in user["rules"].values():
            rows[0]["item_idx"] = 1
    else:
        manifest["fit_uids"] = list(reversed(manifest["fit_uids"]))
    fx.manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        contexts(fx)




@pytest.mark.parametrize("fault", [None, "time_rule", "numerical_source", "scale_users"])
def test_selector_rechecks_real_common_context_provenance(tmp_path, monkeypatch, fault):
    from experiments.ch3_motivation.code import select_probe_temporal_common as selector
    fx = fixture(tmp_path, monkeypatch)
    values = contexts(fx)
    first = values["cutover"].identity
    binding = dict(input_root=str(fx.root), input_binding_sha256=first["input_binding_sha256"],
        population_sha256=first["population_sha256"], candidate_file_sha256=first["candidate_file_sha256"],
        gap_manifest=str(fx.manifest), gap_manifest_sha256=first["gap_manifest_sha256"])
    summary = dict(format=driver.FORMAT, edge=1, quality_auc_computed=False, fitting_performed=False,
        post_release_labels_loaded=False, independent_fitting_users=64, scale_user_context_observations=192,
        contexts={rule:copy.deepcopy(context.identity) for rule,context in values.items()},
        native_reuse={rule:driver.native_baseline(context) for rule,context in values.items()})
    if fault == "time_rule":
        for identity in summary["contexts"].values():identity["time_rule"] = "wrong"
    elif fault == "numerical_source":
        for identity in summary["contexts"].values():identity["numerical_sources"] = {}
    elif fault == "scale_users":
        for identity in summary["contexts"].values():identity["loss_scales"]["users"] = 64
    if fault:
        with pytest.raises(ValueError):selector._check_contexts(summary, dict(edge_bindings={"1":binding}), 1)
    else:
        checked = selector._check_contexts(summary, dict(edge_bindings={"1":binding}), 1)
        assert list(checked) == RULES
