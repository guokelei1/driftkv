"""Global three-context selection, complete coverage and strict recipe identity."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import select_probe_temporal_common as selector
from experiments.ch3_motivation.code.prepare_temporal_common_plan import RULES
from experiments.ch3_motivation.code.select_probe_common import write_json, sha256, fingerprint


def metrics(loss):
    return dict(users=16, loss=loss, read_mse=loss/2, score_mse=loss/2,
        raw_score_mse=loss/100, per_user=[dict(uid=uid, loss=loss, read_mse=loss/2,
        score_mse=loss/2, raw_score_mse=loss/100) for uid in range(16)])


def test_source_jobs_keep_actual_f_only_sweep_default_and_explicit_ladder(tmp_path):
    plans = {}
    for filename, job in (
            ("f_row_input_sweep.json", dict(name="row", edge=1)),
            ("probe_ladder_development.json", dict(name="ladder", edge=1, stages="f,fg,fgh", controls=True)),
            ("probe_temporal_calibration.json", dict(name="temporal", edge=1, stages="f,fg,fgh", controls=False))):
        path = tmp_path / filename
        write_json(path, dict(jobs=[job]))
        plans[str(path)] = sha256(path)
    jobs = selector._jobs(dict(source_fitting_plans_sha256=plans))
    assert jobs["f_row_input_sweep/row", 1, 17]["stages"] == "f"
    assert jobs["f_row_input_sweep/row", 1, 17]["controls"] is False
    assert jobs["ladder_development/ladder", 1, 17]["stages"] == "f,fg,fgh"
    assert jobs["ladder_development/ladder", 1, 17]["controls"] is True
    assert jobs["temporal_calibration/temporal", 1, 17]["stages"] == "f,fg,fgh"


def fixture(tmp_path, monkeypatch, losses=None):
    """Synthetic scalar losses, while the real aggregation/coverage checks run."""
    specs, summaries, paths = [], {}, []
    jobs = {}
    losses = losses or {}
    shared = dict(population_sha256="roles", scale_fit_uids=list(range(64)), validation_uids=list(range(16)),
        queries=16, recent_fraction=.5, candidate_seed=17, time_rule="past-only",
        candidate_order="fixed", objective=dict(read_weight=1., score_weight=1.), backend="torch", numerical_sources={})
    for edge in selector.EDGES:
        contexts = {rule: dict(shared, edge=edge, query_delta_rule=rule) for rule in RULES}
        summary = dict(format=selector.FORMAT, status="complete", edge=edge,
            contexts=contexts, points=[], validated=56, failed=0, pending=0)
        for index in range(56):
            recipe = f"source/recipe{index}"
            checkpoint = str(tmp_path / "weights" / f"edge{edge}_recipe{index}.pt")
            spec = dict(recipe=recipe, edge=edge, seed=17, method_label="fgh", checkpoint=checkpoint,
                        fitting_driver="run_probe_variants.py", origin="release_gap_240")
            specs.append(spec)
            jobs[(recipe, edge, 17)] = dict(index=index)
            value = losses.get(index, [2.] * 5)[edge-1]
            context_losses = value if isinstance(value, list) else [value] * 3
            point = dict(spec, status="validated", variant="full_item" if index < 4 else "head",
                stage="fgh", config=dict(hidden_width=128 if index != 2 else 64, depth=1),
                fitting_users=64 if index != 1 else 128, best_step=0 if edge == 1 else 16,
                completed_steps=32, step_zero=edge == 1, selection_eligible=True,
                cumulative_calibration_flops_estimate=100 if index != 3 else 90,
                contexts={rule: dict(status="validated", common_validation_sha256=fingerprint(contexts[rule]),
                    metrics=metrics(context_losses[i])) for i, rule in enumerate(RULES)},
                mean_context_loss=sum(context_losses)/3)
            summary["points"].append(point)
        summaries[edge] = summary
    plan = dict(format="ch3_temporal_common_plan_v1", rules=RULES, weights=specs,
                executor_sources_sha256={}, source_fitting_plans_sha256={},
                planner_sha256=sha256(ROOT / "experiments/ch3_motivation/code/prepare_temporal_common_plan.py"),
                selection=dict(policy="predeclared"))
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    for edge, summary in summaries.items():
        summary["plan_sha256"] = sha256(plan_path)
        path = tmp_path / f"edge_{edge}/summary.json"
        write_json(path, summary)
        paths.append(path)
    monkeypatch.setattr(selector, "_jobs", lambda _: jobs)
    monkeypatch.setattr(selector, "_check_contexts", lambda summary, plan, edge: summary["contexts"])

    def check(point, spec, job, common):
        settings = dict(variant=point["variant"], width=point["config"]["hidden_width"], depth=1,
            input_root="same/fitting/root", fit_uids=list(range(point["fitting_users"])),
            fitting_driver="run_probe_variants.py", fitting_driver_sha256="same-driver", query_delta_rule="cutover",
            edge=spec["edge"], models={str(spec["edge"]): "model"},
            input_binding_sha256=f"input{spec['edge']}", calibration_candidate_sha256=f"ids{spec['edge']}")
        return settings, dict(edge=spec["edge"], input_root="same/fitting/root")

    monkeypatch.setattr(selector, "_check_training", check)
    return plan_path, paths, summaries, check


def test_global_context_then_edge_equal_mean_fixed_one_percent_and_step0(tmp_path, monkeypatch):
    # r0 is the minimum; r1 is below1% but larger fit; r2 below1% has less width;
    # r3 less cost but wider. The fixed minimum prevents r2's1% expanding again.
    plan, paths, summaries, _ = fixture(tmp_path, monkeypatch, {
        0:[1.]*5, 1:[1.005]*5, 2:[1.009]*5, 3:[1.018]*5})
    result = selector.rollup(plan, paths)
    chosen = result["selection"]["recommendation"]
    assert result["selection"]["status"] == "recommendation_for_review"
    assert chosen["recipe"] == "source/recipe2"
    assert chosen["step_zero_edges"] == [1] and chosen["correction_seeds"] == [17,23,29]
    assert result["selection"]["fixed_one_percent_threshold"] == 1.01
    assert "source/recipe3" not in result["selection"]["within_one_percent"]
    assert len(result["points"]) == 280 and len(result["groups"]) == 56


def test_all_three_contexts_and_edges_share_one_global_recipe(tmp_path, monkeypatch):
    # Excellent cutover alone cannot beat a uniformly good recipe. Large edge1
    # loss cannot be dropped; the selected recipe is global across all edges.
    plan, paths, _, _ = fixture(tmp_path, monkeypatch, {
        0:[[0.,3.,3.]]*5, 1:[10.,0.,0.,0.,0.], 2:[1.5]*5, 3:[1.7]*5})
    chosen = selector.rollup(plan, paths)["selection"]["recommendation"]
    assert chosen["recipe"] == "source/recipe2" and chosen["mean_common_loss"] == 1.5
    assert chosen["mean_loss_by_context"] == {rule:1.5 for rule in RULES}


@pytest.mark.parametrize("fault", ["missing_edge", "pending", "missing_row"])
def test_incomplete_grid_never_recommends(tmp_path, monkeypatch, fault):
    plan, paths, summaries, _ = fixture(tmp_path, monkeypatch)
    if fault == "missing_edge":
        paths.pop()
    elif fault == "pending":
        summary = summaries[5]
        summary["status"] = "waiting_weights"
        summary["points"][0]["status"] = "pending"
        summary.update(validated=55, pending=1)
        write_json(paths[4], summary)
    else:
        summary = summaries[5]
        summary["status"] = "in_progress"
        summary["points"].pop()
        write_json(paths[4], summary)
    result = selector.rollup(plan, paths)
    assert result["selection"]["status"] == "not_ready"
    assert result["selection"]["recommendation"] is None


def test_failed_recipe_retained_other_complete_recipe_for_review(tmp_path, monkeypatch):
    plan, paths, summaries, _ = fixture(tmp_path, monkeypatch, {0:[.1]*5,2:[1.]*5})
    summary = summaries[1]
    summary["status"] = "complete_with_failures"
    summary["points"][0].update(status="failed", selection_eligible=False, error="numerical error")
    summary.update(validated=55, failed=1)
    write_json(paths[0], summary)
    result = selector.rollup(plan, paths)
    assert result["selection"]["needs_failure_review"]
    assert result["selection"]["recommendation"]["recipe"] != "source/recipe0"
    group = next(g for g in result["groups"] if g["recipe"] == "source/recipe0")
    assert group["failed_edges"] == [1] and group["mean_common_loss"] is None
    assert len(result["points"]) == 280


def test_non_edge_recipe_change_is_rejected(tmp_path, monkeypatch):
    plan, paths, _, check = fixture(tmp_path, monkeypatch)
    def changed(point, spec, job, common):
        settings, binding = check(point, spec, job, common)
        settings["query_delta_rule"] = "gap_mix50" if spec["edge"] == 2 else "cutover"
        return settings, binding
    monkeypatch.setattr(selector, "_check_training", changed)
    with pytest.raises(ValueError, match="non-edge fitting settings"):
        selector.rollup(plan, paths)


def test_normalization_retains_time_rule_driver_seed_optimizer_and_roles():
    identity = dict(edge=1, models={"one":"sha"}, calibration_candidate_sha256="candidate",
        input_binding_sha256="input", gap_manifest="edge1.json", gap_manifest_sha256="manifest",
        source_cutoff=100, query_delta_rule="gap_mix50", time_rule="strict-past", candidate_order="fixed",
        query_delta_layout="B,Q", fitting_driver_sha256="driver", seed=17, learning_rate=3e-5, fit_uids=[1,2])
    normalized = selector.normalized_recipe_identity(identity)
    assert not set(selector.EDGE_FIELDS) & set(normalized)
    assert normalized == {key:value for key,value in identity.items() if key not in selector.EDGE_FIELDS}
    assert normalized["query_delta_rule"] == "gap_mix50" and normalized["learning_rate"] == 3e-5


def test_changed_point_context_or_equal_user_metric_is_rejected(tmp_path, monkeypatch):
    plan, paths, summaries, _ = fixture(tmp_path, monkeypatch)
    summaries[1]["points"][0]["contexts"]["cutover"]["metrics"]["per_user"][0]["loss"] += .5
    write_json(paths[0], summaries[1])
    with pytest.raises(ValueError, match="equal-user mean"):
        selector.rollup(plan, paths)


@pytest.mark.parametrize("fault", [None, "candidate", "driver_sha", "learning_rate", "stage"])
def test_typed_metadata_checks_real_source_recipe_bindings(tmp_path, fault):
    population = dict(fit=list(range(512)), validation=list(range(1000,1016)))
    root = tmp_path / "inputs"
    write_json(root / "population.json", population)
    write_json(root / "edge_1/inputs.json", dict(population_sha256=sha256(root / "population.json")))
    write_json(root / "edge_1/calibration_feedback_candidates.json", dict(
        population_sha256=sha256(root / "population.json"), fitting_pool_uids=population["fit"][:64]))
    binding = selector._input_binding(root, 1)
    source_key = "experiments/ch3_motivation/code/run_probe_variants.py"
    identity = dict(selector.LEGACY_DEFAULTS, variant="full_item", width=128, edge=1, seed=17,
        models={"native":"frozen-model"}, population_sha256=binding["population_sha256"],
        input_binding_sha256=binding["input_binding_sha256"], fit_uids=population["fit"][:64],
        validation_users=population["validation"], calibration_candidate_sha256=binding["candidate_file_sha256"],
        numerical_sources={source_key:sha256(ROOT / source_key)})
    identity.pop("fit_users")
    config = dict(hidden_width=128, depth=1, include_item=True, normalize=True)
    metadata = dict(identity=identity, method_label="fgh", best_step=0, completed_steps=128,
        cumulative_calibration_flops_estimate=12345,
        fit_pool_loss_scales=dict(version="fit_pool_equal_user_length_normalized_v1"))
    if fault == "candidate":identity["calibration_candidate_sha256"] = "wrong"
    elif fault == "driver_sha":identity["numerical_sources"][source_key] = "wrong"
    elif fault == "learning_rate":identity["learning_rate"] = .001
    payload = dict(metadata=metadata, config=config, stage="fg" if fault == "stage" else "fgh", variant="full_item")
    path = tmp_path / "weight.pt"
    torch.save(payload, path)
    point = dict(checkpoint_sha256=sha256(path), training_identity=identity, config=config,
        stage=payload["stage"], variant="full_item", best_step=0, completed_steps=128,
        cumulative_calibration_flops_estimate=12345)
    spec = dict(checkpoint=str(path), edge=1, seed=17, method_label="fgh", fitting_driver="run_probe_variants.py")
    job = dict(variant="full_item", width=128, input_root=str(root))
    common = dict(models=identity["models"], validation_uids=population["validation"])
    if fault:
        with pytest.raises(ValueError):selector._check_training(point, spec, job, common)
    else:
        settings, actual_binding = selector._check_training(point, spec, job, common)
        assert actual_binding == binding and settings["query_delta_rule"] == "cutover"
