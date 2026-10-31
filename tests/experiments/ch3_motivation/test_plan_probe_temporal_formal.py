"""Formal temporal main dispatch stays ready while the curve stays explicit."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import plan_probe_temporal_formal as planner
from experiments.ch3_motivation.code.select_probe_common import sha256, write_json


def fixture(tmp_path, monkeypatch, temporal=True):
    population = dict(fit=list(range(512)), validation=list(range(1000,1016)))
    roots = [tmp_path / "main_fit64", tmp_path / "curve_fit256"]
    for root in roots:
        write_json(root / "population.json", population)
        for edge in planner.EDGES:
            folder = root / f"edge_{edge}"
            write_json(folder / "inputs.json", dict(population_sha256=sha256(root / "population.json"),
                                                    settings=dict(edge=edge)))
            write_json(folder / "calibration_feedback_candidates.json", dict(
                population_sha256=sha256(root / "population.json"),
                fitting_pool_uids=population["fit"][:64 if root == roots[0] else 256]))
    driver = "run_probe_temporal.py" if temporal else "run_probe_variants.py"
    source_sha = sha256(ROOT / "experiments/ch3_motivation/code" / driver)
    rule = "gap_mix50" if temporal else "cutover"
    settings = dict(planner.FIXED, input_root=str(roots[0]), fitting_driver=driver,
                    fitting_driver_sha256=source_sha, query_delta_rule=rule)
    provenance = {}
    for edge in planner.EDGES:
        manifest = tmp_path / f"edge_{edge}_gap.json"
        write_json(manifest, dict(edge=edge, rule=rule))
        provenance[str(edge)] = dict(fitting_driver=driver, fitting_driver_sha256=source_sha,
            query_delta_rule=rule, gap_manifest=str(manifest), gap_manifest_sha256=sha256(manifest),
            source_cutoff=edge*86400, time_rule="strict-past", candidate_order="8feedback8uniform",
            query_delta_layout="[B,16]")
    recommendation = dict(recipe="temporal/selected", variant="full_item", width=128, depth=1,
        fit_users=64, method_label="fgh", edges=planner.EDGES, correction_seeds=planner.FORMAL_SEEDS,
        settings=settings, fitting_driver=driver, fitting_driver_sha256=source_sha,
        query_delta_rule=rule, gap_provenance_by_edge=provenance, step_zero_edges=[1],
        checkpoints_by_edge={str(e):dict(checkpoint=f"edge{e}_fgh.pt") for e in planner.EDGES})
    selection = dict(status="recommendation_for_review", all_five_edges_complete=True,
                     selection_seed=17, recommendation=recommendation)
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)
    scope_path = tmp_path / "scope.json"
    write_json(scope_path, json.loads((ROOT / "experiments/ch3_motivation/configs/formal_quality_scope.json").read_text()))

    def load(manifest, root, edge, requested_rule):
        prov = provenance[str(edge)]
        assert Path(manifest) == Path(prov["gap_manifest"]) and requested_rule == rule
        return {uid:None for uid in population["fit"][:64] + population["validation"]}, dict(
            prov, fallback_used=False, fit_uids=population["fit"][:64], validation_uids=population["validation"])
    monkeypatch.setattr(planner.calibration_gaps, "load_gap_manifest", load)
    return SimpleNamespace(selection=selection, selection_path=selection_path, scope_path=scope_path,
                           curve=roots[1], provenance=provenance)


def test_temporal_main_fifteen_ready_exact_per_edge_manifest_curve_pending(tmp_path, monkeypatch):
    fx = fixture(tmp_path, monkeypatch)
    result = planner.generate(fx.selection_path, fx.scope_path, curve_input_root=fx.curve)
    assert result["status"] == result["main_status"] == "prepared_for_root_review"
    assert result["curve_plan_status"] == "pending_temporal_curve_driver"
    assert result["curve_jobs"] == [] and result["curve_input_bindings"] == {}
    assert len(result["main_jobs"]) == 15 and result["gpu_launch_performed"] is False
    for job in result["main_jobs"]:
        assert job["seed"] in (17,23,29) and job["fitting_driver"] == "run_probe_temporal.py"
        assert job["gap_rule"] == job["query_delta_rule"] == "gap_mix50"
        assert job["gap_manifest_sha256"] == fx.provenance[str(job["edge"])]["gap_manifest_sha256"]
        assert job["controls"] and job["stages"] == "f,fg,fgh"
        assert (job["fit_users"],job["width"],job["steps"],job["patience"]) == (64,128,512,128)
    assert result["curve_pending"]["candidate_catalog_source_users"] == 256
    assert result["curve_pending"]["time_law_source_fit_users"] == 64
    assert result["curve_pending"]["executable_jobs_generated"] is False


def test_legacy_choice_reuses_frozen_planner_and_generates_real_legacy_jobs(tmp_path, monkeypatch):
    fx = fixture(tmp_path, monkeypatch, temporal=False)
    result = planner.generate(fx.selection_path, fx.scope_path, curve_input_root=fx.curve)
    assert result["status"] == "prepared_for_root_review"
    assert len(result["main_jobs"]) == 15 and len(result["curve_jobs"]) == 45
    assert all(j["fitting_driver"] == "run_probe_variants.py" and j["query_delta_rule"] == "cutover"
               for j in result["main_jobs"] + result["curve_jobs"])


@pytest.mark.parametrize("fault", ["driver_sha", "rule", "manifest_sha", "unsupported_lr"])
def test_no_silent_old_driver_or_changed_temporal_recipe(tmp_path, monkeypatch, fault):
    fx = fixture(tmp_path, monkeypatch)
    chosen = fx.selection["recommendation"]
    if fault == "driver_sha":
        chosen["fitting_driver_sha256"] = "wrong"
    elif fault == "rule":
        chosen["query_delta_rule"] = "cutover"
    elif fault == "manifest_sha":
        chosen["gap_provenance_by_edge"]["3"]["gap_manifest_sha256"] = "wrong"
    else:
        chosen["settings"]["learning_rate"] = .001
    write_json(fx.selection_path, fx.selection)
    with pytest.raises(ValueError):
        planner.generate(fx.selection_path, fx.scope_path, curve_input_root=fx.curve)


def test_waiting_selection_does_not_generate_jobs(tmp_path, monkeypatch):
    fx = fixture(tmp_path, monkeypatch)
    result = planner.generate(tmp_path / "not_ready.json", fx.scope_path, curve_input_root=fx.curve)
    assert result["status"] == "waiting_selection" and not result["main_jobs"] and not result["curve_jobs"]
