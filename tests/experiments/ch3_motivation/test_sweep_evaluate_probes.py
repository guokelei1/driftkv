"""A pending natural-cohort plan must never become a zero-job completion."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import sweep_evaluate_probes as sweep


@pytest.mark.parametrize("status,jobs",[
    ("waiting_selection",[]),
    ("waiting_weights",[{"name":"not_ready","checkpoint":"missing.pt"}]),
    ("prepared_for_root_review",[]),
])
def test_pending_and_empty_plans_fail_before_output_or_gpu(tmp_path,monkeypatch,status,jobs):
    plan=tmp_path/"plan.json";plan.write_text(json.dumps(dict(status=status,jobs=jobs)))
    output=tmp_path/"quality"
    monkeypatch.setattr(sweep,"run",lambda *a,**kw:pytest.fail("pending plan reached GPU execution"))
    with pytest.raises(ValueError,match="ready, nonempty plan"):
        sweep.main(SimpleNamespace(plan=plan,output=output,gpu=0))
    assert not output.exists()


@pytest.mark.parametrize("problem",["missing_weight","missing_population","duplicate_name"])
def test_missing_inputs_or_colliding_outputs_fail_before_gpu(tmp_path,monkeypatch,problem):
    checkpoint=tmp_path/"weights.pt";checkpoint.write_bytes(b"routing fixture, never loaded as weights")
    population=tmp_path/"scenario.json";population.write_text("{}")
    job=dict(name="natural_edge_1_seed_17_f",edge=1,panel="natural_active",checkpoint=str(checkpoint),population_file=str(population))
    jobs=[job]
    if problem=="missing_weight":checkpoint.unlink()
    elif problem=="missing_population":population.unlink()
    else:jobs.append(dict(job))
    plan=tmp_path/"plan.json";plan.write_text(json.dumps(dict(status="prepared_for_root_review",jobs=jobs)))
    output=tmp_path/"quality"
    monkeypatch.setattr(sweep,"run",lambda *a,**kw:pytest.fail("invalid plan reached GPU execution"))
    with pytest.raises(ValueError,match="missing before GPU|unique"):
        sweep.main(SimpleNamespace(plan=plan,output=output,gpu=0))
    assert not output.exists()


def test_natural_panel_routes_via_namespace_to_its_actual_scenario(tmp_path,monkeypatch):
    checkpoint=tmp_path/"weights.pt";checkpoint.write_bytes(b"routing fixture, never loaded as weights")
    population=tmp_path/"scenario.json";population.write_text("{}")
    plan=tmp_path/"plan.json"
    plan.write_text(json.dumps(dict(status="prepared_for_root_review",jobs=[dict(
        name="natural_edge_1_seed_17_f",edge=1,panel="natural_active",backend="projected",
        checkpoint=str(checkpoint),population_file=str(population))])))
    output=tmp_path/"quality";calls=[]
    def run(args,*,shared_context):
        calls.append(args)
        assert args.panel=="natural_active" and args.population_file==population
        assert args.checkpoint==checkpoint and args.backend=="projected"
        result=dict(edge=1,variant="full_item",method_label="f",stage="f",width=128,depth=1,
            panel=args.panel,seed=17,fit_users=64,quality_users=2000,feedback_events=2000,
            auc=.6,reuse_auc=.5,rebuild_auc=.7,recovery_percent=50.,
            costs=dict(complete_added_flops=100,candidate_reads_230=dict(percent_of_once_rebuild=20.)))
        sweep.write_json(args.output/"summary.json",result)
        return result
    monkeypatch.setattr(sweep,"run",run)
    sweep.main(SimpleNamespace(plan=plan,output=output,gpu=0))
    result=json.loads((output/"summary.json").read_text())
    assert len(calls)==result["jobs"]==1 and result["status"]=="complete"
    assert result["results"][0]["panel"]=="natural_active"


def prepared(tmp_path,*,family="main"):
    checkpoint=tmp_path/"weights.pt";checkpoint.write_bytes(b"CPU routing fixture; not a typed weight")
    population=tmp_path/"scenario.json";population.write_text('{"frozen_uids":[1,2]}')
    formal=tmp_path/"formal.json";formal.write_text('{"status":"prepared_for_root_review"}')
    source=tmp_path/"source.py";source.write_text("# bound source fixture\n")
    relative=lambda path:os.path.relpath(path,sweep.ROOT)
    panel={"confirmation":"confirmation","natural":"natural_active"}.get(family,"main")
    job=dict(name=f"{family}_edge_1_seed_17_f",edge=1,panel=panel,backend="projected",
        checkpoint=relative(checkpoint),population_file=relative(population),
        input_root=relative(tmp_path))
    plan=dict(status="prepared_for_root_review",family=family,jobs=[job],
        actual_weight_audit=[dict(checkpoint=str(checkpoint),checkpoint_sha256=sweep.sha256(checkpoint))],
        source_sha256={relative(source):sweep.sha256(source)},
        frozen_sha256={str(population):sweep.sha256(population),
                       str(checkpoint):sweep.sha256(checkpoint)},
        formal_plan=relative(formal),formal_plan_sha256=sweep.sha256(formal))
    path=tmp_path/"plan.json"
    path.write_text(json.dumps(plan))
    return SimpleNamespace(plan=path,output=tmp_path/"quality",gpu=0),plan,dict(
        checkpoint=checkpoint,population=population,formal=formal,source=source)


def fake_result(args):
    result=dict(edge=args.edge,variant="full_item",method_label="f",stage="f",width=128,depth=1,
        panel=args.panel,seed=17,fit_users=64,quality_users=2000,feedback_events=2000,
        auc=.6,reuse_auc=.5,rebuild_auc=.7,recovery_percent=50.,
        costs=dict(complete_added_flops=100,candidate_reads_230=dict(percent_of_once_rebuild=20.)))
    sweep.write_json(args.output/"summary.json",result)
    return result


@pytest.mark.parametrize("changed",["checkpoint","population","formal","source"])
def test_stale_existing_binding_rejected_before_output_and_gpu(tmp_path,monkeypatch,changed):
    args,_,paths=prepared(tmp_path)
    paths[changed].write_text("changed after planning")
    monkeypatch.setattr(sweep,"run",lambda *a,**kw:pytest.fail("stale bound plan reached GPU"))
    with pytest.raises(ValueError,match="frozen dependency changed"):
        sweep.main(args)
    assert not args.output.exists()


@pytest.mark.parametrize("problem",["missing_audit","conflicting_alias","missing_formal_sha"])
def test_incomplete_or_conflicting_existing_audit_is_not_silently_ignored(tmp_path,monkeypatch,problem):
    args,plan,paths=prepared(tmp_path)
    if problem=="missing_audit":plan["actual_weight_audit"]=[]
    elif problem=="conflicting_alias":plan["frozen_sha256"][plan["jobs"][0]["checkpoint"]]="0"*64
    else:del plan["formal_plan_sha256"]
    args.plan.write_text(json.dumps(plan))
    monkeypatch.setattr(sweep,"run",lambda *a,**kw:pytest.fail("invalid audit reached GPU"))
    with pytest.raises(ValueError,match="does not cover|conflicting|formal plan requires"):
        sweep.main(args)
    assert not args.output.exists()


@pytest.mark.parametrize("family",["main","confirmation","f_population_curve","natural"])
def test_bound_plans_root_relative_paths_route_without_new_development_schema(tmp_path,monkeypatch,family):
    args,plan,paths=prepared(tmp_path,family=family)
    calls=[]
    def run(options,*,shared_context):
        calls.append(options)
        assert options.checkpoint==paths["checkpoint"].resolve()
        assert options.population_file==paths["population"].resolve()
        assert options.input_root==tmp_path.resolve()
        assert options.panel==plan["jobs"][0]["panel"] and options.backend=="projected"
        return fake_result(options)
    monkeypatch.setattr(sweep,"run",run)
    sweep.main(args)
    result=json.loads((args.output/"summary.json").read_text())
    assert result["status"]=="complete" and len(calls)==1
    assert result["identity"]["plan_sha256"]==sweep.sha256(args.plan)


@pytest.mark.parametrize("changed",["checkpoint","population","formal","source","plan"])
def test_completion_rechecks_all_bindings_and_the_plan_itself(tmp_path,monkeypatch,changed):
    args,_,paths=prepared(tmp_path)
    def run(options,*,shared_context):
        result=fake_result(options)
        (args.plan if changed=="plan" else paths[changed]).write_text("changed during evaluation")
        return result
    monkeypatch.setattr(sweep,"run",run)
    with pytest.raises(ValueError,match="frozen dependency changed"):
        sweep.main(args)
    assert not (args.output/"summary.json").exists()
    assert (args.output/"points.csv").is_file()  # Valid partial evidence is retained.


def test_repeated_weight_and_aliases_are_hashed_once_per_boundary_not_per_job(tmp_path,monkeypatch):
    args,plan,paths=prepared(tmp_path)
    plan["jobs"].append(dict(plan["jobs"][0],name="same_weight_other_point"))
    plan["frozen_sha256"][plan["jobs"][0]["checkpoint"]]=plan["actual_weight_audit"][0]["checkpoint_sha256"]
    args.plan.write_text(json.dumps(plan))
    original=sweep.sha256
    hashes=[]
    def counted(path):
        hashes.append(Path(path).resolve())
        return original(path)
    monkeypatch.setattr(sweep,"sha256",counted)
    monkeypatch.setattr(sweep,"run",lambda options,**kwargs:fake_result(options))
    sweep.main(args)
    assert hashes.count(paths["checkpoint"].resolve())==2
    assert hashes.count(paths["formal"].resolve())==2
    assert hashes.count(paths["population"].resolve())==2
    assert hashes.count(paths["source"].resolve())==2
