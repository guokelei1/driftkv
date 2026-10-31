"""The continuation audit includes idle UIDs and exact raw-score coverage."""
import hashlib
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code import common
from experiments.ch3_motivation.code.audit_scenario_qualification import aggregate_releases, audit_blocks
from experiments.ch3_motivation.code.continue_e3_queue import main as continue_queue
from hstu_kvcache.common.artifacts import sha256


def test_block_audit_binds_idle_members_and_actual_feedback(tmp_path):
    users=[3,4]
    identity=dict(test="new")
    event=dict(uid=3,query_timestamp=5,raw_item_id=7,item_idx=7,label=1)
    path=tmp_path/"part_000000.parquet"
    rows=[dict(event,method="old_full",score=.5)]
    pq.write_table(pa.Table.from_pylist(rows),path)
    marker=dict(fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),
        uids_sha256=hashlib.sha256(json.dumps(users).encode()).hexdigest(),offset=0,users=2,
        completed_users=2,scores=1,scores_sha256=sha256(path))
    common.write_json(tmp_path/"part_000000.json",marker)
    actual,audit=audit_blocks(tmp_path,users,identity,["old_full"],[event])
    assert actual==rows and audit["blocks"]==1
    with pytest.raises(ValueError,match="identity/hash"):
        audit_blocks(tmp_path,[3,5],identity,["old_full"],[event])
    rows[0]["method"]="reuse"
    pq.write_table(pa.Table.from_pylist(rows),path)
    marker["scores_sha256"]=sha256(path)
    common.write_json(tmp_path/"part_000000.json",marker)
    with pytest.raises(ValueError,match="feedback/method coverage"):
        audit_blocks(tmp_path,users,identity,["old_full"],[event])


def test_cross_release_cluster_draws_preserve_same_uid_correlation():
    users=list(range(512))
    edge=[]
    for uid in users:
        label=uid%2
        for method,score in (("old_full",.5),("reuse",.5),("rebuild",float(label))):
            edge.append(dict(uid=uid,query_timestamp=1,raw_item_id=uid,label=label,method=method,score=score))
    summary=aggregate_releases([edge]*5,users,replicates=32)
    assert summary["new_rebuild_minus_old_full"]==.5
    assert summary["new_rebuild_minus_old_full_ci95"]==[.5,.5]
    assert summary["joint_positive_edges"]==5 and summary["overall_scenario_passed"]
    assert summary["shared_uid_multiplicity_across_releases"]


def test_opposite_release_differences_cancel_in_every_shared_uid_draw():
    users=list(range(512))
    forward,reverse=[],[]
    for uid in users:
        a=((uid*17)%31)/31
        b=((uid*13)%29)/29
        for method,score,opposite in (("old_full",a,b),("reuse",a,b),("rebuild",b,a)):
            event=dict(uid=uid,query_timestamp=1,raw_item_id=uid,label=uid%2,method=method)
            forward.append(dict(event,score=score))
            reverse.append(dict(event,score=opposite))
    summary=aggregate_releases([forward,reverse],users,replicates=32)
    assert summary["new_rebuild_minus_old_full"]==0
    assert summary["new_rebuild_minus_old_full_ci95"]==[0.,0.]
    assert summary["new_rebuild_minus_reuse_ci95"]==[0.,0.]


def test_continuation_stops_before_population_when_wrapper_check_fails(tmp_path,monkeypatch):
    original=tmp_path/"original"
    original.mkdir()
    common.write_json(original/"queue_status.json",dict(state="complete",exit_code=0))
    check=tmp_path/"check.json"
    common.write_json(check,dict(passed=False,quality_metrics_computed=False,users=8))
    population_marker=tmp_path/"population_started"
    jobs=[dict(name="wrapper_check_8",command=[sys.executable,"-c","print('diagnostic')"],
               log=str(tmp_path/"diagnostic.log"),check_path=str(check)),
          dict(name="population",command=[sys.executable,"-c",f"open({str(population_marker)!r},'w').close()"],
               log=str(tmp_path/"population.log"))]
    common.write_json(tmp_path/"queue_config.json",dict(wait_for_python_pid=999999999,qualification_root=str(original),source_sha256={},frozen_inputs={},jobs=jobs))
    common.write_json(tmp_path/"launch.json",dict(config_sha256=sha256(tmp_path/"queue_config.json")))
    monkeypatch.setattr(sys,"argv",["continue_e3_queue.py","--queue-dir",str(tmp_path)])
    with pytest.raises(ValueError,match="wrapper numerical check failed"):
        continue_queue()
    assert not population_marker.exists()
    assert json.loads((tmp_path/"queue_status.json").read_text())["state"]=="failed"
