"""The plan expands a selected recipe without changing the GPU launch state."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import plan_probe_formal as planner


def setup(tmp_path, *, steps=512):
    population = dict(fit=list(range(512)), validation=list(range(1000, 1016)))
    roots = [tmp_path / "selected_inputs", tmp_path / "curve_fit256"]
    for root in roots:
        planner.write_json(root / "population.json", population)
        population_sha = planner.sha256(root / "population.json")
        for edge in planner.EDGES:
            directory = root / f"edge_{edge}"
            planner.write_json(directory / "inputs.json", dict(population_sha256=population_sha,
                settings=dict(edge=edge)))
            candidates = dict(population_sha256=population_sha)
            if root == roots[1]:
                candidates["fitting_pool_uids"] = population["fit"][:256]
            planner.write_json(directory / "calibration_feedback_candidates.json", candidates)
    settings = dict(input_root=str(roots[0]), learning_rate=.00003, steps=steps,
        read_weight=.1, score_weight=1., patience=128, validation_every=8,
        candidate_source="feedback", recent_fraction=.5, queries=16,
        row_chunk=1024, query_chunk=16, weight_decay=.0001, gradient_clip=5.)
    selection = dict(status="recommendation_for_review", all_five_edges_complete=True, selection_seed=17,
        recommendation=dict(recipe="sweep/attempt_1/selected", variant="full_item", method_label="fgh",
            edges=planner.EDGES, correction_seeds=planner.FORMAL_SEEDS, fit_users=64,
            width=128, depth=1, settings=settings, step_zero_edges=[1],
            checkpoints_by_edge={str(edge): dict(checkpoint=f"existing_edge{edge}_fgh.pt") for edge in planner.EDGES}))
    selection_path = tmp_path / "selection.json"
    planner.write_json(selection_path, selection)
    scope = json.loads((ROOT / "experiments/ch3_motivation/configs/formal_quality_scope.json").read_text())
    scope_path = tmp_path / "scope.json"
    planner.write_json(scope_path, scope)
    return SimpleNamespace(selection_path=selection_path, scope_path=scope_path, selection=selection,
                           scope=scope, main_root=roots[0], curve_root=roots[1])


@pytest.mark.parametrize("steps", [128, 512])
def test_main_recipe_and_curve_candidates_budgets_population_accounting(tmp_path, steps):
    fx = setup(tmp_path, steps=steps)
    result = planner.generate(fx.selection_path, fx.scope_path, curve_input_root=fx.curve_root)
    assert result["status"] == "prepared_for_root_review" and result["gpu_launch_performed"] is False
    assert result["correction_seeds"] == [17, 23, 29] and result["backbone_retraining"] is False
    assert len(result["main_jobs"]) == 15 and len(result["curve_jobs"]) == 45
    for job in result["main_jobs"]:
        assert job["steps"] == steps and job["patience"] == 128
        assert job["input_root"] == str(fx.main_root) and job["controls"] is True
        assert job["stages"] == "f,fg,fgh"
        assert (job["learning_rate"], job["read_weight"], job["score_weight"]) == (.00003, .1, 1.)
    for edge in planner.EDGES:
        for seed in planner.FORMAL_SEEDS:
            jobs = [job for job in result["curve_jobs"] if job["edge"] == edge and job["seed"] == seed]
            assert {job["fit_users"] for job in jobs} == {64, 128, 256}
            assert {job["steps"] for job in jobs} == {max(256, steps)}
            assert all(job["patience"] == 0 and job["stages"] == "f" and not job["controls"] for job in jobs)
            assert {job["input_root"] for job in jobs} == {str(fx.curve_root)}
            assert {(job["queries"], job["recent_fraction"]) for job in jobs} == {(16, .5)}
    for accounting in result["population_accounting"]:
        assert accounting["gradient_fit_users"] == accounting["teacher_fit_users"] == accounting["training_loss_RMS_source_users"]
        assert accounting["candidate_catalog_source_users"] == 256
        assert accounting["common_validation_loss_RMS_source_users"] == 64
    assert result["population_curve_policy"]["original_validation_losses_cross_N_comparable"] is False
    assert result["population_curve_policy"]["fixed_training_RMS"].startswith("optional")
    assert result["reuse_main_f_for_curve"] is False
    assert result["job_counts"]["confirmation_additional_fitting"] == 0
    assert result["selected_h_step_zero_edges"] == [1]


def test_missing_or_incomplete_selection_is_only_a_pending_plan(tmp_path):
    fx = setup(tmp_path)
    absent = tmp_path / "not_yet_selected.json"
    result = planner.generate(absent, fx.scope_path, curve_input_root=fx.curve_root)
    assert result["status"] == "waiting_selection" and not result["main_jobs"] and not result["curve_jobs"]
    incomplete = copy.deepcopy(fx.selection)
    incomplete["all_five_edges_complete"] = False
    planner.write_json(fx.selection_path, incomplete)
    result = planner.generate(fx.selection_path, fx.scope_path, curve_input_root=fx.curve_root)
    assert result["status"] == "waiting_selection" and result["gpu_launch_performed"] is False


@pytest.mark.parametrize("tamper", ["seed_scope", "input_population", "candidate_catalog"])
def test_changed_scope_or_unmatched_curve_roles_cannot_generate_jobs(tmp_path, tamper):
    fx = setup(tmp_path)
    if tamper == "seed_scope":
        fx.scope["training_seeds"] = [17, 29, 43]
        planner.write_json(fx.scope_path, fx.scope)
    elif tamper == "input_population":
        planner.write_json(fx.curve_root / "population.json", dict(fit=[99], validation=[]))
    else:
        path = fx.curve_root / "edge_1/calibration_feedback_candidates.json"
        candidates = json.loads(path.read_text())
        candidates["fitting_pool_uids"] = candidates["fitting_pool_uids"][:64]
        planner.write_json(path, candidates)
    with pytest.raises(ValueError):
        planner.generate(fx.selection_path, fx.scope_path, curve_input_root=fx.curve_root)


def test_cli_writes_runner_compatible_plans_and_scope_without_executing(tmp_path):
    fx = setup(tmp_path)
    args = SimpleNamespace(selection=fx.selection_path, scope=fx.scope_path,
        curve_input_root=fx.curve_root, output=tmp_path / "formal_plans")
    planner.main(args)
    result = json.loads((args.output / "plan.json").read_text())
    main = json.loads((args.output / "main_fitting.json").read_text())
    curve = json.loads((args.output / "f_population_curve.json").read_text())
    assert main["jobs"] == result["main_jobs"] and curve["jobs"] == result["curve_jobs"]
    assert main["jobs"][0]["seed"] == 17 and result["gpu_launch_performed"] is False
    assert result["source_scope_sha256"] == planner.sha256(fx.scope_path)
    assert result["source_selection_sha256"] == planner.sha256(fx.selection_path)
