"""Scenario evidence and calibration must describe the actual loaded population."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code import common
from experiments.ch3_motivation.code.scenario_inputs import (
    history_signature, load_scenario_population, load_scenario_data,
)
from hstu_kvcache.common.artifacts import sha256


@pytest.fixture
def scenario(tmp_path):
    population=tmp_path/"population.json"
    common.write_json(population,dict(fit=[1],validation=[2]))
    histories=SimpleNamespace(rows={uid:(np.array([10,20]),np.array([uid,uid+1]),np.array([0,1])) for uid in range(1,6)})
    common.save_histories(tmp_path/"histories.npz",histories,[1,2,3,4,5])
    requests=tmp_path/"requests.parquet"
    pq.write_table(pa.Table.from_pylist([dict(uid=uid,query_timestamp=25,raw_item_id=uid,item_idx=uid,label=uid%2) for uid in (3,4)]),requests)
    binding=tmp_path/"inputs.json"
    common.write_json(binding,dict(settings=dict(edge=1,release_day=1,end_day=15,history_length=4),
        population_sha256=sha256(population),history=dict(path="histories.npz",sha256=sha256(tmp_path/"histories.npz")),
        requests=dict(path="requests.parquet",sha256=sha256(requests))))
    users=[3,4]
    qualification=dict(new_rebuild_minus_old_full=.2,new_rebuild_minus_reuse=.1,point_qualifies=True,uid95_qualifies=True)
    summary=tmp_path/"summary.json"
    common.write_json(summary,dict(quality_users=2,quality_uids=users,quality_uids_sha256=hashlib.sha256(json.dumps(users).encode()).hexdigest(),
        auc=dict(old_full=.5,reuse=.6,rebuild=.7),qualification=qualification,**qualification,
        new_rebuild_minus_old_full_ci95=[.1,.3],new_rebuild_minus_reuse_ci95=[.01,.2],
        identity=dict(input_binding_sha256=sha256(binding),models=dict(parent="old",current="new"))))
    manifest=tmp_path/"scenario_manifest.json"
    common.write_json(manifest,dict(schema_version=1,edge=1,uids=users,roles=dict(quality=users,e6=users),
        input_binding=dict(path=str(binding),sha256=sha256(binding)),
        history=dict(path=str(tmp_path/"histories.npz"),sha256=sha256(tmp_path/"histories.npz")),
        requests=dict(path=str(requests),sha256=sha256(requests)),source_population=dict(path=str(population),sha256=sha256(population)),
        qualification=dict(summary_path=str(summary),summary_sha256=sha256(summary),scope="outcome-conditioned diagnostic",**qualification),
        selection=dict(panel_name="test",description="frozen before correction quality")))
    return manifest,histories


def update(path,fn):
    value=json.loads(path.read_text())
    fn(value)
    common.write_json(path,value)


def test_exact_population_and_raw_data_are_loaded(scenario):
    manifest,histories=scenario
    binding,users,population,provenance=load_scenario_population(manifest,1,role="e6",minimum_users=2)
    assert users==[3,4] and population["fit"]==[1]
    assert provenance["calibration_input_binding_sha256"]==provenance["input_binding_sha256"]
    restored,requests=load_scenario_data(provenance)
    assert history_signature(restored,[1,2])==history_signature(histories,[1,2])
    assert sorted(requests)==[3,4] and requests[4][25][0]["item_idx"]==4


def test_measured_uid_identity_cannot_be_rebound(scenario):
    manifest,_=scenario
    update(manifest,lambda v:v.update(uids=[4,3],roles=dict(quality=[4,3],e6=[4,3])))
    with pytest.raises(ValueError,match="another population"):
        load_scenario_population(manifest,1,minimum_users=2)


def test_calibration_members_cannot_enter_quality(scenario):
    manifest,_=scenario
    update(manifest,lambda v:v.update(uids=[1,4],roles=dict(quality=[1,4],e6=[1,4])))
    with pytest.raises(ValueError,match="overlaps reserved"):
        load_scenario_population(manifest,1,minimum_users=2)


def test_qualification_is_bound_to_actual_anchor_difference(scenario):
    manifest,_=scenario
    summary=manifest.parent/"summary.json"
    update(summary,lambda v:v["auc"].update(rebuild=.8))
    update(manifest,lambda v:v["qualification"].update(summary_sha256=sha256(summary)))
    with pytest.raises(ValueError,match="AUC differences"):
        load_scenario_population(manifest,1,minimum_users=2)


def test_e6_cannot_reuse_resident_qualification_or_idle_members(scenario):
    manifest,_=scenario
    update(manifest,lambda v:v["roles"].update(e6=[3]))
    with pytest.raises(ValueError,match="exactly its feedback population"):
        load_scenario_population(manifest,1,role="e6",minimum_users=1)
    update(manifest,lambda v:v["roles"].update(e6=[3,5]))
    with pytest.raises(ValueError,match="membership/count"):
        load_scenario_population(manifest,1,role="e6",minimum_users=2)


def test_unqualified_candidate_only_allowed_for_explicit_baseline_preparation(scenario):
    manifest,_=scenario
    update(manifest,lambda v:v.update(qualification=dict(status="not_yet_measured")))
    _,users,_,provenance=load_scenario_population(manifest,1,minimum_users=2,require_qualification=False)
    assert users==[3,4] and provenance["mode"]=="unqualified_candidate_inputs"
    with pytest.raises(KeyError):
        load_scenario_population(manifest,1,minimum_users=2)


def test_measured_negative_edge_remains_reportable_without_positive_screen(scenario):
    manifest,_=scenario
    summary=manifest.parent/"summary.json"
    negative=dict(new_rebuild_minus_old_full=-.1,new_rebuild_minus_reuse=-.2,point_qualifies=False,uid95_qualifies=False)
    update(summary,lambda v:v.update(auc=dict(old_full=.5,reuse=.6,rebuild=.4),qualification=negative,**negative,
        new_rebuild_minus_old_full_ci95=[-.2,0.],new_rebuild_minus_reuse_ci95=[-.3,-.1]))
    update(manifest,lambda v:v["qualification"].update(summary_sha256=sha256(summary),**negative))
    with pytest.raises(ValueError,match="lacks positive"):
        load_scenario_population(manifest,1,minimum_users=2)
    _,users,_,provenance=load_scenario_population(manifest,1,minimum_users=2,require_positive=False)
    assert users==[3,4] and not provenance["qualification"]["point_qualifies"]
    assert provenance["mode"]=="frozen_three_anchor_measured_scenario"


def test_new_evaluation_pack_preserves_actual_calibration_tuples(scenario):
    manifest,histories=scenario
    original=manifest.parent/"inputs.json"
    binding=json.loads(original.read_text())
    new_folder=manifest.parent/"new"
    new_folder.mkdir()
    common.save_histories(new_folder/"histories.npz",histories,[1,2,3,4])
    binding.update(history=dict(path="histories.npz",sha256=sha256(new_folder/"histories.npz")),
        calibration_source_binding=dict(path=str(original),sha256=sha256(original)),calibration_inputs_unchanged=True,
        calibration_history_sha256=history_signature(histories,[1,2]))
    new_binding=new_folder/"inputs.json"
    common.write_json(new_binding,binding)
    update(manifest,lambda v:v.update(input_binding=dict(path=str(new_binding),sha256=sha256(new_binding)),
        history=dict(path=str(new_folder/"histories.npz"),sha256=sha256(new_folder/"histories.npz"))))
    _,_,_,provenance=load_scenario_population(manifest,1,minimum_users=2,require_qualification=False)
    assert provenance["calibration_input_binding_sha256"]==sha256(original)
    histories.rows[1][1][0]+=1
    common.save_histories(new_folder/"histories.npz",histories,[1,2,3,4])
    binding["history"]["sha256"]=sha256(new_folder/"histories.npz")
    common.write_json(new_binding,binding)
    update(manifest,lambda v:v.update(input_binding=dict(path=str(new_binding),sha256=sha256(new_binding)),
        history=dict(path=str(new_folder/"histories.npz"),sha256=sha256(new_folder/"histories.npz"))))
    with pytest.raises(ValueError,match="calibration history tuples"):
        load_scenario_population(manifest,1,minimum_users=2,require_qualification=False)
