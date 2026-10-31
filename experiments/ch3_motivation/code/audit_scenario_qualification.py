#!/usr/bin/env python3
"""CPU audit of the fresh three-anchor screen before E3 continuation."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow.parquet as pq

from experiments.ch3_motivation.code import common
from experiments.ch3_motivation.code.qualify_scenario import bootstrap_anchors, frozen_panels, sources
from experiments.ch3_motivation.code.scenario_inputs import load_scenario_population
from hstu_kvcache.evaluation.statistics import weighted_auc
from hstu_kvcache.common.artifacts import sha256


def _hash(value,*,sort=False):
    return hashlib.sha256(json.dumps(value,sort_keys=sort).encode()).hexdigest()


def audit_blocks(folder,users,identity,methods,expected_events,*,block_size=256):
    fingerprint=_hash(identity,sort=True)
    rows=[]
    markers=sorted(folder.glob("part_*.json"))
    expected_names=[f"part_{offset:06d}" for offset in range(0,len(users),block_size)]
    if [path.stem for path in markers]!=expected_names or [path.stem for path in sorted(folder.glob("part_*.parquet"))]!=expected_names:
        raise ValueError("three-anchor block coverage differs")
    for offset,name in zip(range(0,len(users),block_size),expected_names,strict=True):
        marker=json.loads((folder/(name+".json")).read_text())
        block=users[offset:offset+block_size]
        path=folder/(name+".parquet")
        if (marker["fingerprint"]!=fingerprint or marker["uids_sha256"]!=_hash(block)
                or marker["offset"]!=offset or marker["users"]!=len(block)
                or marker["completed_users"]!=offset+len(block) or marker["scores_sha256"]!=sha256(path)):
            raise ValueError("three-anchor block identity/hash differs")
        actual=pq.read_table(path).to_pylist()
        if marker["scores"]!=len(actual) or any(row["uid"] not in block for row in actual):
            raise ValueError("three-anchor block row/UID coverage differs")
        rows.extend(actual)
    expected={(int(row["uid"]),int(row["query_timestamp"]),int(row["raw_item_id"]),int(row["item_idx"]),int(row["label"]),method)
              for row in expected_events for method in methods}
    keys=[(row["uid"],row["query_timestamp"],row["raw_item_id"],row["item_idx"],row["label"],row["method"]) for row in rows]
    if len(keys)!=len(expected) or set(keys)!=expected or any(not np.isfinite(row["score"]) for row in rows):
        raise ValueError("three-anchor actual feedback/method coverage differs")
    return rows,dict(blocks=len(markers),scores=len(rows),all_hashes_and_uids_passed=True)


def aggregate_releases(edge_rows,users,*,replicates=1000,seed=20261027):
    """One UID multiplicity draw is shared by every release and anchor."""
    population=np.asarray(sorted(users),dtype=np.int64)
    rng=np.random.default_rng(seed)
    weights=np.asarray([np.bincount(rng.integers(len(users),size=len(users)),minlength=len(users))
                        for _ in range(replicates)],dtype=np.uint16)
    draws=[]
    points=[]
    selected=set(users)
    for rows in edge_rows:
        groups=defaultdict(list)
        for row in rows:
            if row["uid"] in selected:
                groups[row["method"]].append(row)
        values={}
        for tag,group in groups.items():
            group.sort(key=lambda row:(row["uid"],row["query_timestamp"],row["raw_item_id"]))
            labels=np.asarray([row["label"] for row in group],dtype=np.float64)
            scores=np.asarray([row["score"] for row in group],dtype=np.float64)
            index=np.searchsorted(population,np.asarray([row["uid"] for row in group],dtype=np.int64))
            values[tag]=weighted_auc(scores,labels,index,weights)
        summary=bootstrap_anchors(rows,users,replicates=1,seed=seed)
        points.append(dict(new_rebuild_minus_old_full=summary["new_rebuild_minus_old_full"],new_rebuild_minus_reuse=summary["new_rebuild_minus_reuse"]))
        draws.append(dict(new_rebuild_minus_old_full=values["rebuild"]-values["old_full"],
                          new_rebuild_minus_reuse=values["rebuild"]-values["reuse"]))
    result={}
    for field in ("new_rebuild_minus_old_full","new_rebuild_minus_reuse"):
        means=np.mean([value[field] for value in draws],axis=0)
        valid=means[np.isfinite(means)]
        result[field]=float(np.mean([value[field] for value in points]))
        result[field+"_ci95"]=np.percentile(valid,[2.5,97.5]).tolist() if len(valid) else None
    result.update(joint_positive_edges=sum(value["new_rebuild_minus_old_full"]>0 and value["new_rebuild_minus_reuse"]>0 for value in points),
        edges=len(edge_rows),confidence_replicates=replicates,
        shared_uid_multiplicity_across_releases=True,aggregation="equal release mean of pooled raw-feedback AUC differences")
    result["overall_scenario_passed"]=bool(result["new_rebuild_minus_old_full"]>0 and result["new_rebuild_minus_reuse"]>0
                                             and result["joint_positive_edges"]>=3)
    return result


def audit(qualification_root,input_root,output):
    qualification_root,input_root,output=Path(qualification_root),Path(input_root),Path(output)
    status=json.loads((qualification_root/"queue_status.json").read_text())
    if status["state"]!="complete" or status["exit_code"]!=0:
        raise ValueError("original three-anchor queue did not complete successfully")
    population_path=input_root/"population.json"
    population=json.loads(population_path.read_text())
    panels=frozen_panels(population)
    union=sorted(set(uid for users in panels.values() for uid in users))
    union_set=set(union)
    code=sources()
    launch=json.loads((qualification_root/"launch.json").read_text())
    if launch["source_sha256"]!=code:
        raise ValueError("original frozen qualification sources changed")
    edges=[]
    all_rows=[]
    models={}
    for edge in range(1,6):
        folder=qualification_root/f"edge_{edge}"
        identity=json.loads((folder/"identity.json").read_text())
        binding=json.loads((input_root/f"edge_{edge}/inputs.json").read_text())
        if (identity["sources"]!=code or identity["edge"]!=edge or identity["panels"]!=panels
                or identity["population_sha256"]!=sha256(population_path)
                or identity["input_binding_sha256"]!=sha256(input_root/f"edge_{edge}/inputs.json")):
            raise ValueError("qualification frozen edge identity changed")
        models.update(identity["models"])
        requests=pq.read_table(input_root/f"edge_{edge}"/binding["requests"]["path"]).to_pylist()
        events=[row for row in requests if row["uid"] in union_set]
        old,old_audit=audit_blocks(folder/"union_old_full/quality_shards",union,
            dict(identity,union=union,branch="old_weights_for_initial_append_and_score"),["old_full"],events)
        current,current_audit=audit_blocks(folder/"union_current_anchors/quality_shards",union,
            dict(identity,union=union,branch="current_reuse_and_once_rebuild"),["reuse","rebuild"],events)
        rows=old+current
        all_rows.append(rows)
        results={}
        for name in [*panels,"feedback_union_2000"]:
            _,users,_,provenance=load_scenario_population(folder/name/"scenario_manifest.json",edge,require_positive=False)
            if name in panels and users!=panels[name]:
                raise ValueError("qualification changed a predeclared random panel")
            saved=json.loads((folder/name/"summary.json").read_text())
            actual=bootstrap_anchors(rows,users,replicates=1000,seed=28121+edge)
            if any(saved[key]!=value for key,value in actual.items()):
                raise ValueError("three-anchor summary differs from independently reread score blocks")
            results[name]=dict(users=len(users),feedback_users=actual["feedback_users"],feedback_events=actual["feedback_events"],
                qualification=actual["qualification"],summary_sha256=provenance["qualification_summary_sha256"])
        edges.append(dict(edge=edge,old_full=old_audit,current_anchors=current_audit,panels=results,
                          evaluation_window=binding["settings"].get("evaluation_window",dict(name="E14",complete_source_horizon=True))))
    for path,digest in models.items():
        if sha256(ROOT/path)!=digest:
            raise ValueError("qualification backbone checkpoint changed")
    aggregate={name:aggregate_releases(all_rows,users) for name,users in panels.items()}
    passed=all(aggregate[name]["overall_scenario_passed"] for name in ("main","confirmation"))
    result=dict(state="complete",all_sources_and_blocks_passed=True,source_files=len(code),
        blocks=sum(edge[branch]["blocks"] for edge in edges for branch in ("old_full","current_anchors")),
        original_panels=4,edges=edges,aggregate=aggregate,overall_scenario_passed=passed,
        criterion="main and confirmation each: both five-edge equal-release anchor mean differences >0 and >=3/5 edges jointly positive; retain every edge",
        claim_scope="qualification verifies the target setting; it makes no claim about any selective-method curve",
        population_selection="original predeclared random role panels unchanged; all screening results retained")
    common.write_json(output,result)
    print(json.dumps(dict(audit=str(output),blocks=result["blocks"],overall_scenario_passed=passed,aggregate=aggregate)),flush=True)
    if not passed:
        raise ValueError("predeclared overall scenario criterion failed; retain audit and diagnose before continuing")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-root",type=Path,required=True)
    parser.add_argument("--input-root",type=Path,default=ROOT/"experiments/ch3_motivation/inputs")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    audit(args.qualification_root,args.input_root,args.output)
