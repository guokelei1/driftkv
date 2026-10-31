"""Five-edge pairing, ratio-of-mean gains and the frozen 3/1-seed roles."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.analyze_probe_quality_multiedge import (
    aggregate_loaded, analyze, DEFAULT_SCOPE, _training_record, _check_planned_job, _recipe, JOB_SETTINGS,
)
from hstu_kvcache.common.artifacts import sha256


def scope():
    return json.loads(DEFAULT_SCOPE.read_text())


def checkpoint_payload(summary):
    evaluation = summary["identity"]
    stage = {"f_extended": "f", "fg_extended": "fg"}.get(summary["method_label"], summary["method_label"])
    identity = dict(edge=summary["edge"], seed=summary["seed"], models=evaluation["models"],
        input_binding_sha256=evaluation["input_binding_sha256"], population_sha256="reserved_roles",
        fit_uids=list(range(64)), validation_users=list(range(100, 116)),
        queries=16, candidate_seed=17, candidate_source="feedback", recent_fraction=.5,
        calibration_candidate_sha256=f"candidates{summary['edge']}",
        learning_rate=.0001, steps=256, patience=128, validation_every=32,
        read_weight=1., score_weight=1., weight_decay=.0001, gradient_clip=5.,
        stages="f,fg,fgh", controls=True, numerical_sources={"fit.py": "fixed"})
    config = dict(num_layers=2, num_heads=2, head_dim=4, item_dim=8,
        hidden_width=summary["width"], depth=summary["depth"], normalize=True, include_item=True,
        query_chunk=16, row_chunk=1024)
    metadata = dict(identity=identity, config=config, variant=summary["variant"], stage=stage,
        method_label=summary["method_label"], seed=summary["seed"], training_users=64,
        objective=dict(read_weight=1., score_weight=1.),
        fit_pool_loss_scales=dict(version="fit_pool_equal_user_length_normalized_v1", source="fitting_users_only"),
        best_step=0 if summary["method_label"] == "fgh" else 256, completed_steps=256)
    return dict(format="ch3_probe_variant_v1", variant=summary["variant"], config=config,
        stage=stage, metadata=metadata, state_dict={})


def loaded_fixture(*, correlation=False, gap="positive"):
    result = []
    for panel, start, seeds, methods in (("main", 10000, (17, 23, 29),
            ("f", "fg", "fgh", "f_extended", "fg_extended")),
            ("confirmation", 20000, (17,), ("f", "fg", "fgh"))):
        users = list(range(start, start+2000))
        active = users[:512]  # Eligible users without feedback stay sampled.
        labels = np.tile([0, 1], len(active))
        rng = np.random.default_rng(start)
        noise, drift = rng.normal(size=len(labels)), rng.normal(size=len(labels))
        for edge in range(1, 6):
            events = [dict(uid=uid, query_timestamp=edge*86400+j, raw_item_id=100+j,
                item_idx=20+j, label=j) for uid in active for j in (0, 1)]
            def rows(values, method):
                return [dict(row, method=method, score=float(score))
                    for row, score in zip(events, values, strict=True)]
            reuse = np.zeros(len(labels))
            if correlation:
                rebuild = labels.astype(float)
            else:
                reuse = noise
                rebuild = noise+np.array([.1, .4, .8, -.3, .2])[edge-1]*labels
            if gap == "zero":
                rebuild = reuse.copy()
            elif gap == "negative":
                rebuild = reuse-.4*labels
            elif gap == "crossing":
                reuse = np.zeros(len(labels))
                rebuild = np.repeat(np.r_[np.ones(257), -np.ones(255)], 2)*labels
            anchors = dict(reuse=rows(reuse, "reuse"), rebuild=rows(rebuild, "rebuild"))
            for seed in seeds:
                if correlation:
                    f = noise if edge in (1, 3) else -noise if edge in (2, 4) else np.zeros(len(labels))
                else:
                    f = noise+np.array([.2, .05, .7, -.1, .1])[edge-1]*labels+.002*(seed-17)*drift
                scores = dict(f=f, fg=f+.15*labels, fgh=f+.4*labels,
                    f_extended=f+.12*labels, fg_extended=f+.15*labels-7)
                identity = dict(edge=edge, panel=panel, users=users, models={f"model{edge}": "m"},
                    backend="projected", input_binding_sha256=f"input{edge}", scenario=None, sources={"source.py": "s"})
                for method in methods:
                    checkpoint_hash = hashlib.sha256(f"{edge}/{seed}/{method}".encode()).hexdigest()
                    method_identity = dict(identity, checkpoint=f"/weights/edge{edge}/seed{seed}/{method}.pt",
                        checkpoint_sha256=checkpoint_hash)
                    summary = dict(identity=method_identity, edge=edge, panel=panel, seed=seed, method_label=method,
                        variant="full_item", fit_users=64, width=128, depth=2,
                        auc=float(roc_auc_score(labels, scores[method])),
                        reuse_auc=float(roc_auc_score(labels, reuse)), rebuild_auc=float(roc_auc_score(labels, rebuild)),
                        evaluation_window=dict(name="E14_partial" if edge == 5 else "E14"))
                    result.append(dict(summary=summary, anchors=anchors, rows=rows(scores[method], method),
                        training=_training_record(summary, checkpoint_payload(summary)),
                        path=f"/{panel}/edge{edge}/seed{seed}/{method}/summary.json"))
    return result


def test_cross_edge_inverse_auc_correlation_cancels_only_with_shared_uid_draws():
    report = aggregate_loaded(loaded_fixture(correlation=True), scope(), replicates=1000)
    assert report["status"] == "complete"
    for panel in report["panels"].values():
        assert panel["quality_users"] == 2000
        assert panel["confidence"]["shared_uid_multiplicity_across_edges"]
        assert panel["confidence"]["all_eligible_uids_including_no_feedback"]
        f = panel["across_seed"]["methods"]["f"]
        assert f["mean_auc"] == .5
        assert f["conditional_uid_auc_ci95"] == pytest.approx([.5, .5], abs=1e-14)
        assert all(edge["users_with_feedback"] == 512 for edge in panel["edge_records"])
    assert report["panels"]["main"]["across_seed"]["seed_status"]["three_seed_complete"]
    confirmation = report["panels"]["confirmation"]
    assert confirmation["complete_grid"]
    status = confirmation["across_seed"]["seed_status"]
    assert status["observed_seeds"] == [17] and status["planned_seeds_complete"]
    assert not status["three_seed_complete"] and not status["training_seed_robustness_evaluated"]


def test_ratio_uses_five_edge_mean_gains_and_direct_extended_contrasts():
    inputs = loaded_fixture()
    report = aggregate_loaded(inputs, scope(), replicates=100)
    main = report["panels"]["main"]
    selected = [row["summary"] for row in inputs if row["summary"]["panel"] == "main"
        and row["summary"]["seed"] == 17 and row["summary"]["method_label"] == "f"]
    gains = np.array([row["auc"]-row["reuse_auc"] for row in selected])
    gaps = np.array([row["rebuild_auc"]-row["reuse_auc"] for row in selected])
    f = main["per_seed"][0]["methods"]["f"]
    assert f["recovery_percent"] == pytest.approx(100*gains.mean()/gaps.mean())
    assert abs(f["recovery_percent"]-np.mean(100*gains/gaps)) > 1
    assert selected[3]["rebuild_auc"] < selected[3]["reuse_auc"]  # Adverse edge retained.
    increments = main["per_seed"][0]["increments"]
    assert increments["fg_extended_minus_fg"]["mean_auc_difference"] == 0
    assert increments["fg_extended_minus_fg"]["conditional_uid_auc_difference_ci95"] == [0., 0.]
    assert increments["fgh_minus_fg_extended"]["mean_auc_difference"] == increments["h"]["mean_auc_difference"]
    assert increments["fg_minus_f_extended"]["mean_auc_difference"] > 0
    assert increments["fgh_minus_f_extended"]["mean_auc_difference"] > increments["h"]["mean_auc_difference"]
    across = main["across_seed"]["increments"]["h"]
    values = [point["increments"]["h"]["mean_auc_difference"] for point in main["per_seed"]]
    assert across["mean_auc_difference"] == pytest.approx(np.mean(values))
    assert across["training_seed_point_range"]["auc_difference"] == [min(values), max(values)]
    assert across["conditional_uid_auc_difference_ci95"][0] > 0
    assert main["edge_records"][-1]["evaluation_window"]["name"] == "E14_partial"


@pytest.mark.parametrize("missing", ["seed", "one_cell", "edge"])
def test_incomplete_main_keeps_not_ready_and_never_calls_confirmation_three_seed(missing):
    values = loaded_fixture()
    def keep(item):
        s = item["summary"]
        if s["panel"] != "main":
            return True
        return not ((missing == "seed" and s["seed"] == 29) or
            (missing == "one_cell" and (s["edge"], s["seed"], s["method_label"]) == (5, 29, "fgh")) or
            (missing == "edge" and s["edge"] == 5))
    report = aggregate_loaded([item for item in values if keep(item)], scope(), replicates=5)
    assert report["status"] == "not_ready"
    main = report["panels"]["main"]
    assert main["missing"] and main["status"] == "not_ready"
    assert not main["primary_aggregate_computed"] and "across_seed" not in main
    assert not main["observed_seed_status"]["three_seed_complete"]
    if missing == "seed":
        assert main["observed_seed_status"]["observed_seed_count"] == 2
        assert main["observed_seed_status"]["missing_expected_seeds"] == [29]
    confirmation = report["panels"]["confirmation"]
    assert confirmation["status"] == "complete"
    assert not confirmation["across_seed"]["seed_status"]["three_seed_complete"]


@pytest.mark.parametrize("gap", ["negative", "zero"])
def test_nonpositive_mean_gap_retains_absolute_increment_without_bounded_recovery_ci(gap):
    report = aggregate_loaded(loaded_fixture(gap=gap), scope(), replicates=30)
    for panel in report["panels"].values():
        assert panel["anchors"]["mean_reference_gap"] <= 0
        assert panel["anchors"]["nonpositive_or_invalid_gap_replicates"] == 30
        h = panel["across_seed"]["increments"]["h"]
        assert h["mean_auc_difference"] > 0
        assert h["conditional_uid_auc_difference_ci95"][0] > 0
        assert h["conditional_uid_recovery_difference_ci95"] is None
        if gap == "zero":
            assert h["recovery_percentage_point_difference"] is None
        else:
            assert h["recovery_percentage_point_difference"] < 0


def test_positive_gap_point_with_bootstrap_zero_crossing_does_not_get_bounded_ratio_interval():
    report = aggregate_loaded(loaded_fixture(gap="crossing"), scope(), replicates=100)
    for panel in report["panels"].values():
        assert panel["anchors"]["mean_reference_gap"] > 0
        assert panel["anchors"]["conditional_uid_reference_gap_ci95"][0] < 0
        assert panel["anchors"]["nonpositive_or_invalid_gap_replicates"] > 0
        assert panel["across_seed"]["methods"]["f"]["conditional_uid_recovery_ci95"] is None
        h = panel["across_seed"]["increments"]["h"]
        assert h["recovery_percentage_point_difference"] is not None
        assert h["conditional_uid_recovery_difference_ci95"] is None


@pytest.mark.parametrize("fault", ["duplicate", "uid_order", "context", "configuration", "labels", "summary_auc", "panel_overlap", "recipe_lr", "recipe_candidates", "confirmation_weights"])
def test_invalid_paired_inputs_are_rejected(fault):
    values = loaded_fixture()
    if fault == "duplicate":
        values.append(values[0])
    elif fault == "uid_order":
        values[0] = copy.deepcopy(values[0])
        values[0]["summary"]["identity"]["users"] = list(reversed(values[0]["summary"]["identity"]["users"]))
    elif fault == "context":
        values[0] = copy.deepcopy(values[0])
        values[0]["summary"]["identity"]["sources"] = {"changed.py": "s"}
    elif fault == "configuration":
        values[0] = copy.deepcopy(values[0])
        values[0]["summary"]["width"] = 256
    elif fault == "labels":
        values[0]["rows"][0]["label"] = 1-values[0]["rows"][0]["label"]
    elif fault == "summary_auc":
        values[0]["summary"]["auc"] += .1
    elif fault in ("recipe_lr", "recipe_candidates"):
        values[0] = copy.deepcopy(values[0])
        key, value = ("learning_rate", .00003) if fault == "recipe_lr" else ("candidate_source", "history")
        values[0]["training"]["identity"][key] = value
    elif fault == "confirmation_weights":
        point = next(item for item in values if item["summary"]["identity"]["panel"] == "confirmation")
        point["training"]["checkpoint_sha256"] = "different_fixed_weights"
    else:
        for item in values:
            if item["summary"]["panel"] == "confirmation":
                item["summary"]["identity"]["users"] = list(range(10000, 12000))
                for row in item["rows"]:
                    row["uid"] -= 10000
                # Shared anchor objects only need the remapping once.
        visited = set()
        for item in values:
            if item["summary"]["panel"] == "confirmation" and id(item["anchors"]) not in visited:
                visited.add(id(item["anchors"]))
                for rows in item["anchors"].values():
                    for row in rows:
                        row["uid"] -= 10000
    with pytest.raises(ValueError):
        aggregate_loaded(values, scope(), replicates=3)


def test_empty_inputs_and_missing_summary_paths_are_explicitly_not_ready(tmp_path):
    missing = tmp_path/"not_yet_written/summary.json"
    result = analyze([missing])
    assert result["status"] == "not_ready" and result["pending_summary_paths"] == [str(missing)]
    assert result["panels"]["main"]["expected_points"] == 75
    assert result["panels"]["confirmation"]["expected_points"] == 15
    assert len(result["panels"]["main"]["missing"]) == 75


def test_unprepared_formal_plan_blocks_primary_aggregation_even_when_all_points_exist():
    result = aggregate_loaded(loaded_fixture(), scope(), formal_plan_ready=False, replicates=3)
    assert result["status"] == "not_ready"
    for panel in result["panels"].values():
        assert panel["complete_grid"] and not panel["primary_aggregate_computed"]
        assert not panel["formal_plan_prepared"] and "across_seed" not in panel


@pytest.mark.parametrize("fault", ["seed", "label", "binding", "roles", "diagnostic", "old_loss_scale"])
def test_actual_checkpoint_metadata_must_match_quality_identity(fault):
    summary = loaded_fixture()[0]["summary"]
    payload = checkpoint_payload(summary)
    if fault == "seed":
        payload["metadata"]["identity"]["seed"] = 29
    elif fault == "label":
        payload["metadata"]["method_label"] = "f_extended"
    elif fault == "binding":
        payload["metadata"]["identity"]["input_binding_sha256"] = "different_inputs"
    elif fault == "roles":
        payload["metadata"]["identity"]["fit_uids"][0] = summary["identity"]["users"][0]
    elif fault == "diagnostic":
        payload["metadata"]["diagnostic_only"] = True
    else:
        payload["metadata"]["fit_pool_loss_scales"]["version"] = "old_per_user_scale"
    with pytest.raises(ValueError):
        _training_record(summary, payload)


@pytest.mark.parametrize("fault", [None, "learning_rate", "budget", "candidate_sha", "fit_roles"])
def test_checkpoint_recipe_is_bound_to_existing_formal_jobs_and_inputs(fault, tmp_path):
    item = loaded_fixture()[0]
    s, t = item["summary"], item["training"]
    identity = t["identity"]
    job = {**{key: s[key] for key in ("edge", "seed", "variant", "fit_users", "width", "depth")},
        **{key: identity[key] for key in JOB_SETTINGS}, "input_root": str(tmp_path)}
    binding = dict(input_root=str(tmp_path), input_binding_sha256=identity["input_binding_sha256"],
        population_sha256=identity["population_sha256"], candidate_file_sha256=identity["calibration_candidate_sha256"])
    population = dict(fit=identity["fit_uids"], validation=identity["validation_users"])
    if fault == "learning_rate":
        job["learning_rate"] = .00003
    elif fault == "budget":
        job["steps"] = 512
    elif fault == "candidate_sha":
        binding["candidate_file_sha256"] = "different_view"
    elif fault == "fit_roles":
        population["fit"] = list(reversed(population["fit"]))
    if fault is None:
        _check_planned_job(item, job, binding, population, {})
    else:
        with pytest.raises(ValueError):
            _check_planned_job(item, job, binding, population, {})


def _item_from_weight(path, payload):
    """Metadata identity only: this helper does not fabricate or evaluate scores."""
    metadata, config = payload["metadata"], payload["config"]
    identity = metadata["identity"]
    summary = dict(edge=identity["edge"], seed=identity["seed"], variant=payload["variant"],
        fit_users=metadata["training_users"], width=config["hidden_width"], depth=config["depth"],
        method_label=metadata["method_label"], identity=dict(models=identity["models"],
            input_binding_sha256=identity["input_binding_sha256"], users=[9000001],
            checkpoint=str(path), checkpoint_sha256=sha256(path)))
    return dict(summary=summary, training=_training_record(summary, payload))


@pytest.mark.parametrize("fault", [None, "rule", "initialization", "manifest", "driver", "teacher_n"])
def test_new_initialization_metadata_and_real_weight_are_bound_before_aggregation(tmp_path, fault):
    from tests.experiments.ch3_motivation.test_plan_probe_quality import fixture as typed_fixture
    from experiments.ch3_motivation.code.probe_variants import save_variant
    path, probe, metadata, job, binding = typed_fixture(tmp_path, "gap_mix50", "balanced")
    metadata["objective"] = dict(read_weight=1., score_weight=1.)
    save_variant(path, probe, metadata)
    # Keep planned objects independent: changing the observed identity must not
    # accidentally change the expected identity through a shared nested dict.
    job = copy.deepcopy(job)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    item = _item_from_weight(path, payload)
    population = json.loads((Path(binding["input_root"])/"population.json").read_text())
    if fault == "rule":
        item["training"]["identity"]["query_delta_rule"] = "gap_empirical100"
    elif fault == "initialization":
        item["training"]["identity"]["h_initialization"] = "standard"
    elif fault == "manifest":
        item["training"]["identity"]["sampling_manifest_sha256"] = "a"*64
    elif fault == "driver":
        item["training"]["identity"]["fitting_driver"] = "run_probe_calibration.py"
    elif fault == "teacher_n":
        item["training"]["identity"]["teacher_fitting_users"] = 256
    if fault is not None:
        with pytest.raises(ValueError):
            _check_planned_job(item, job, binding, population, {}, audit_checkpoint=True)
    else:
        audit = _check_planned_job(item, job, binding, population, {}, audit_checkpoint=True)
        assert audit["sampling_identity"]["h_initialization"] == "balanced"
        assert audit["checkpoint_sha256"] == item["training"]["checkpoint_sha256"]
        assert "formal_main_calibration_bound" not in item["training"]
        # A stale or swapped weight hash is rejected even if metadata agrees.
        item["summary"]["identity"]["checkpoint_sha256"] = "a"*64
        with pytest.raises(ValueError, match="score weight binding"):
            _check_planned_job(item, job, binding, population, {}, audit_checkpoint=True)


def test_legacy_temporal_metadata_uses_the_same_strict_planned_time_and_driver_fields(tmp_path):
    item = loaded_fixture()[0]
    summary, identity = item["summary"], item["training"]["identity"]
    identity.update(fitting_driver="run_probe_temporal.py", fitting_driver_sha256="driver-sha",
        query_delta_rule="gap_mix50", gap_manifest=str(tmp_path/"gap.json"), gap_manifest_sha256="gap-sha",
        time_rule="strictly before release", source_cutoff=86400, candidate_order="feedback then uniform",
        query_delta_layout="[B,16], same causal manifest for every nested stage and sibling control")
    job = copy.deepcopy(dict(identity, input_root=str(tmp_path), **{
        key: summary[key] for key in ("variant", "fit_users", "width", "depth")}))
    binding = dict(input_root=str(tmp_path), input_binding_sha256=identity["input_binding_sha256"],
        population_sha256=identity["population_sha256"], candidate_file_sha256=identity["calibration_candidate_sha256"])
    population = dict(fit=identity["fit_uids"], validation=identity["validation_users"])
    _check_planned_job(item, job, binding, population, {})
    for key, wrong in (("query_delta_rule", "gap_empirical100"), ("gap_manifest_sha256", "changed"),
            ("source_cutoff", 86401), ("query_delta_layout", "scalar"),
            ("fitting_driver_sha256", "changed-driver")):
        changed = copy.deepcopy(item)
        changed["training"]["identity"][key] = wrong
        with pytest.raises(ValueError):
            _check_planned_job(changed, job, binding, population, {})


@pytest.mark.parametrize("field,value", [("query_delta_rule", "gap_empirical100"),
    ("time_law_source_users", 64), ("candidate_catalog_source_users", 64),
    ("gradient_fitting_users", 128), ("teacher_fitting_users", 128),
    ("h_initialization", "balanced"), ("fitting_driver", "run_probe_calibration_init.py"),
    ("candidate_order", "changed order")])
def test_bound_cross_edge_normalization_keeps_algorithm_and_calibration_settings(field, value):
    values = loaded_fixture()
    left = copy.deepcopy(values[0]["training"])
    right = copy.deepcopy(next(item["training"] for item in values if item["summary"]["edge"] == 2))
    for edge, training in ((1, left), (2, right)):
        training["identity"].update(sampling_manifest=f"/edge{edge}/sampling.json",
            sampling_manifest_sha256=str(edge)*64, gap_manifest=f"/edge{edge}/sampling.json",
            gap_manifest_sha256=str(edge)*64, source_cutoff=edge*86400, query_delta_rule="gap_mix50",
            time_law_source_users=256, candidate_catalog_source_users=256,
            gradient_fitting_users=64, teacher_fitting_users=64)
    assert _recipe(left, across_edges=True) != _recipe(right, across_edges=True)
    left["formal_main_calibration_bound"] = right["formal_main_calibration_bound"] = True
    assert _recipe(left, across_edges=True) == _recipe(right, across_edges=True)
    assert _recipe(left, across_edges=False) != _recipe(right, across_edges=False)
    right["identity"][field] = value
    assert _recipe(left, across_edges=True) != _recipe(right, across_edges=True)


def test_actual_phase_a_two_edge_metadata_accepts_only_verified_natural_input_differences():
    # Existing development weights are read unchanged. This is CPU identity
    # evidence, not a formal plan, a population-quality run, or a three-seed claim.
    folder = ROOT/"experiments/ch3_motivation/runs/e5/calibration_sampling/attempt_1"/(
        "full_item128_fit256_fb50_cutover_lr3e5_cap2048_pat1024")
    paths = [folder/f"edge_{edge}/seed_17/fgh.pt" for edge in (1, 2)]
    if not all(path.exists() for path in paths):
        pytest.skip("retained PhaseA checkpoints are not available in this checkout")
    items = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        item = _item_from_weight(path, payload)
        identity = item["training"]["identity"]
        source = json.loads(Path(identity["sampling_manifest"]).read_text())["sourcebinding"]
        job = copy.deepcopy(dict(identity, input_root=source["input_root"]))
        binding = dict(input_root=source["input_root"], population_sha256=source["population_sha256"],
            input_binding_sha256=source["sha256"], candidate_file=source["candidate_path"],
            candidate_file_sha256=source["candidate_sha256"], sampling_manifest=identity["sampling_manifest"],
            sampling_manifest_sha256=identity["sampling_manifest_sha256"])
        population = json.loads((Path(source["input_root"])/"population.json").read_text())
        audit = _check_planned_job(item, job, binding, population, {}, audit_checkpoint=True)
        assert audit["sampling_identity"]["source_cutoff"] == identity["source_cutoff"]
        for key, wrong in (("query_delta_rule", "gap_empirical100"),
                ("h_initialization", "balanced"), ("sampling_manifest_sha256", "b"*64)):
            changed = copy.deepcopy(item)
            changed["training"]["identity"][key] = wrong
            with pytest.raises(ValueError):
                _check_planned_job(changed, job, binding, population, {}, audit_checkpoint=True)
        items.append(item)
    left, right = (item["training"] for item in items)
    assert _recipe(left, across_edges=True) != _recipe(right, across_edges=True)
    # Exercise only the post-audit normalization branch; no formal-ready result
    # is emitted from these development metadata records.
    left["formal_main_calibration_bound"] = right["formal_main_calibration_bound"] = True
    assert _recipe(left, across_edges=True) == _recipe(right, across_edges=True)


def test_formal_main_enables_edge_normalization_only_after_real_auditor_and_uid_binding(tmp_path, monkeypatch):
    from experiments.ch3_motivation.code import analyze_probe_quality_multiedge as module
    from experiments.ch3_motivation.code.analyze_probe_quality import ShardLoader
    item = loaded_fixture()[0]
    identity = item["training"]["identity"]
    population = dict(fit=identity["fit_uids"], validation=identity["validation_users"],
        main=item["summary"]["identity"]["users"], confirmation=list(range(20000, 22000)))
    (tmp_path/"population.json").write_text(json.dumps(population))
    bindings, jobs = {}, []
    for edge in range(1, 6):
        folder = tmp_path/f"edge_{edge}"
        folder.mkdir()
        (folder/"inputs.json").write_text(json.dumps(dict(edge=edge)))
        candidate = folder/"candidates.json"
        candidate.write_text(json.dumps(dict(edge=edge)))
        bindings[str(edge)] = dict(input_root=str(tmp_path),
            population_sha256=sha256(tmp_path/"population.json"),
            input_binding_sha256=sha256(folder/"inputs.json"), candidate_file=str(candidate),
            candidate_file_sha256=sha256(candidate))
        for seed in (17, 23, 29):
            jobs.append(dict(edge=edge, seed=seed, variant="full_item", fit_users=64, width=128, depth=2,
                input_root=str(tmp_path), **{key: identity[key] for key in JOB_SETTINGS}))
    for key in ("input_binding_sha256", "population_sha256"):
        identity[key] = bindings["1"][key]
    identity["calibration_candidate_sha256"] = bindings["1"]["candidate_file_sha256"]
    plan = dict(status="prepared_for_root_review", source_scope=str(DEFAULT_SCOPE),
        source_scope_sha256=sha256(DEFAULT_SCOPE), main_jobs=jobs, main_input_bindings=bindings)
    calls = []
    def audit(path, job, binding, method):
        calls.append((str(path), job["seed"], method))
        return dict(checkpoint_sha256=item["training"]["checkpoint_sha256"])
    monkeypatch.setattr(module.quality, "audit_weight", audit)
    no_controls = copy.deepcopy(plan)
    no_controls["main_jobs"][0]["controls"] = False
    with pytest.raises(ValueError, match="extended controls"):
        module._formal_plan_audit([item], no_controls, ShardLoader(), DEFAULT_SCOPE)
    failed = copy.deepcopy(item)
    failed["summary"]["identity"]["users"].reverse()
    with pytest.raises(ValueError, match="UID order"):
        module._formal_plan_audit([failed], plan, ShardLoader(), DEFAULT_SCOPE)
    assert not calls and "formal_main_calibration_bound" not in failed["training"]
    failed = copy.deepcopy(item)
    failed["training"]["checkpoint_sha256"] = "wrong_actual_weight"
    with pytest.raises(ValueError, match="score weight binding"):
        module._formal_plan_audit([failed], plan, ShardLoader(), DEFAULT_SCOPE)
    assert "formal_main_calibration_bound" not in failed["training"]
    assert module._formal_plan_audit([item], plan, ShardLoader(), DEFAULT_SCOPE)
    assert item["training"]["formal_main_calibration_bound"] and len(calls) == 2


def test_actual_committed_shard_loader_is_used_and_rejects_corrupted_parquet(tmp_path):
    item = loaded_fixture()[0]
    summary = item["summary"]
    folder = tmp_path/"point"
    folder.mkdir()
    model, checkpoint = tmp_path/"model.bin", tmp_path/"weights.bin"
    model.write_bytes(b"synthetic frozen model")
    identity = dict(summary["identity"], checkpoint=str(checkpoint),
        models={str(model): sha256(model)})
    summary["identity"] = identity
    torch.save(checkpoint_payload(summary), checkpoint)
    identity["checkpoint_sha256"] = sha256(checkpoint)
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    write(folder/"configuration.json", identity)
    anchor_folder = folder/"anchors"
    fingerprint = digest(identity)
    for offset in range(0, 2000, 256):
        users = identity["users"][offset:offset+256]
        allowed = set(users)
        probe = folder/"probe_blocks"/f"part_{offset:06d}.parquet"
        probe.parent.mkdir(exist_ok=True)
        rows = [row for row in item["rows"] if row["uid"] in allowed]
        pq.write_table(pa.Table.from_pylist(rows), probe)
        write(probe.with_suffix(".json"), dict(users=users, fingerprint=fingerprint, sha256=sha256(probe)))
        anchor = anchor_folder/"quality_shards"/f"part_{offset:06d}.parquet"
        anchor.parent.mkdir(parents=True, exist_ok=True)
        rows = [row for values in item["anchors"].values() for row in values if row["uid"] in allowed]
        pq.write_table(pa.Table.from_pylist(rows), anchor)
        write(anchor.with_suffix(".json"), dict(offset=offset, users=len(users), fingerprint=fingerprint,
            uids_sha256=hashlib.sha256(json.dumps(users).encode()).hexdigest(), scores=len(rows), scores_sha256=sha256(anchor)))
    summary.update(identity=identity, status="complete", experiment="E5", stage="f", quality_users=2000,
        feedback_events=len(item["rows"]), anchor_source=dict(path=str(anchor_folder),
            key=digest({name: identity[name] for name in ("edge", "models", "users", "input_binding_sha256", "sources")})))
    path = folder/"summary.json"
    write(path, summary)
    result = analyze([path])
    assert result["panels"]["main"]["supplied_points"] == 1 and result["status"] == "not_ready"
    assert str(checkpoint) in result["verification"]["artifact_sha256"]
    corrupted = folder/"probe_blocks/part_000000.parquet"
    with corrupted.open("ab") as stream:
        stream.write(b"changed after commit")
    with pytest.raises(ValueError, match="artifact hash changed"):
        analyze([path])
