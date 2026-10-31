"""Synthetic >=512-UID committed artifacts test direct paired increments."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.analyze_probe_quality import analyze
from hstu_kvcache.common.artifacts import sha256


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path,seeds=(17,29)):
    population=list(range(1000,1512))
    active=population[:500]  # Twelve eligible idle UIDs must remain sampled.
    rng=np.random.default_rng(515)
    labels=np.tile([0,1,0,1],len(active))
    noise=rng.normal(size=len(labels))
    drift=rng.normal(size=len(labels))
    events=[dict(uid=uid,query_timestamp=231*86400+j,raw_item_id=100+j,item_idx=20+j,label=j%2)
            for uid in active for j in range(4)]
    anchors=dict(reuse=noise,rebuild=noise+.5*labels)
    model=tmp_path/"model.bin";model.write_bytes(b"synthetic frozen model identity")
    source=tmp_path/"numeric_source.py";source.write_text("# synthetic source binding\n")
    common=dict(edge=1,panel="synthetic_fixture",users=population,models={str(model):sha256(model)},
        backend="projected",input_binding_sha256="d"*64,scenario=None,sources={str(source):sha256(source)})
    paths={};identities={};values={}
    methods=("f","fg","fgh","f_extended","fg_extended")
    for seed in seeds:
        f=noise+.1*labels+.002*(seed-17)*drift
        scores=dict(f=f,fg=f+3.,fgh=f+.25*labels,
                    f_extended=f+.12*labels,fg_extended=f-7.)
        for method in methods:
            folder=tmp_path/f"seed_{seed}"/method
            checkpoint=tmp_path/"weights"/f"seed_{seed}_{method}.pt"
            checkpoint.parent.mkdir(exist_ok=True)
            checkpoint.write_bytes(f"synthetic checkpoint {seed} {method}".encode())
            identity=dict(common,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint))
            write(folder/"configuration.json",identity)
            rows=[dict(row,method=method,score=float(score)) for row,score in zip(events,scores[method],strict=True)]
            for offset in (0,256):
                block=population[offset:offset+256]
                selected=[row for row in rows if row["uid"] in set(block)]
                path=folder/"probe_blocks"/f"part_{offset:06d}.parquet"
                path.parent.mkdir(parents=True,exist_ok=True)
                pq.write_table(pa.Table.from_pylist(selected),path)
                write(path.with_suffix(".json"),dict(users=block,fingerprint=digest(identity),sha256=sha256(path),execution={}))
            paths[seed,method]=folder/"summary.json"
            identities[seed,method]=identity
            values[seed,method]=scores[method]
    first=paths[seeds[0],"f"].parent
    anchor_folder=first/"anchors"
    anchor_identity=identities[seeds[0],"f"]
    anchor_key=digest({name:anchor_identity[name] for name in
        ("edge","models","users","input_binding_sha256","sources")})
    for offset in (0,256):
        block=population[offset:offset+256]
        rows=[]
        for method,scores in anchors.items():
            rows.extend(dict(row,method=method,score=float(score)) for row,score in zip(events,scores,strict=True)
                        if row["uid"] in set(block))
        path=anchor_folder/"quality_shards"/f"part_{offset:06d}.parquet"
        path.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows),path)
        write(path.with_suffix(".json"),dict(fingerprint=digest(anchor_identity),
            uids_sha256=hashlib.sha256(json.dumps(block).encode()).hexdigest(),offset=offset,users=len(block),
            scores=len(rows),scores_sha256=sha256(path)))
    for (seed,method),path in paths.items():
        write(path,dict(status="complete",experiment="E5",identity=identities[seed,method],
            edge=1,variant="full_item",fit_users=64,width=128,depth=1,seed=seed,
            quality_users=512,feedback_events=len(events),users_with_feedback=500,
            method_label=method,stage={"f_extended":"f","fg_extended":"fg"}.get(method,method),
            auc=float(roc_auc_score(labels,values[seed,method])),
            reuse_auc=float(roc_auc_score(labels,anchors["reuse"])),rebuild_auc=float(roc_auc_score(labels,anchors["rebuild"])),
            anchor_source=dict(path=str(anchor_folder),key=anchor_key,recomputed_in_this_goal=True)))
    return paths,anchor_folder


def test_direct_paired_increment_zero_for_global_score_offset_and_missing_seed_is_explicit(tmp_path):
    paths,_=fixture(tmp_path)
    result=analyze(paths.values(),expected_seeds=[17,29,43])
    assert result["quality_users"]==512 and result["users_with_feedback"]==500
    assert result["confidence"]["replicates"]==1000
    group=result["groups"][0]
    assert not group["three_seed_complete"] and group["missing_expected_seeds"]==[43]
    for point in group["per_seed"]:
        g=point["increments"]["g"]
        assert g["auc_difference"]==0
        assert g["auc_difference_ci95"]==[0.,0.]
        assert g["recovery_difference_ci95"]==[0.,0.]
        # Separate method uncertainty is nonzero, while its paired invariant
        # contrast is exactly known. Independent CI subtraction would fail.
        f_ci=point["methods"]["f"]["auc_ci95"]
        assert f_ci[1]>f_ci[0]
        h=point["increments"]["h"]
        assert h["auc_difference_ci95"][0]>0
        assert h["recovery_percentage_point_difference"]==pytest.approx(
            100*h["auc_difference"]/result["anchors"]["reference_gap"])
        other=point["increments"]["fg_extended_minus_fg"]
        assert other["auc_difference_ci95"]==[0.,0.]
    across=group["across_seed"]["increments"]["h"]
    estimates=[point["increments"]["h"]["auc_difference"] for point in group["per_seed"]]
    assert across["auc_difference_mean"]==pytest.approx(np.mean(estimates))
    assert across["auc_difference_range"]==[min(estimates),max(estimates)]
    assert across["mean_paired_uid_recovery_ci95"][0]>0
    assert group["across_seed"]["increments"]["g"]["mean_paired_uid_recovery_ci95"]==[0.,0.]
    assert not across["three_seed_complete"] and across["seed_count"]==2


def test_three_seed_complete_requires_every_included_method_seed_and_does_not_invent_missing_pairs(tmp_path):
    paths,_=fixture(tmp_path,seeds=(17,29,43))
    result=analyze(paths.values(),expected_seeds=[17,29,43])
    assert result["groups"][0]["three_seed_complete"]
    selected=[path for key,path in paths.items() if key!=(43,"fgh")]
    result=analyze(selected,expected_seeds=[17,29,43])
    group=result["groups"][0]
    assert not group["three_seed_complete"]
    assert group["across_seed"]["increments"]["h"]["seed_count"]==2
    assert not group["across_seed"]["increments"]["h"]["three_seed_complete"]
    assert "h" not in group["per_seed"][-1]["increments"]


@pytest.mark.parametrize("tamper",["probe_hash","anchor_hash","checkpoint_hash","anchor_key_cached","label_with_valid_hash","incomplete_block"])
def test_committed_artifact_tampering_and_semantic_misalignment_are_rejected(tmp_path,tamper):
    paths,anchor_folder=fixture(tmp_path,seeds=(17,))
    path=paths[17,"fg"]
    block=path.parent/"probe_blocks/part_000000.parquet"
    if tamper=="probe_hash":
        with block.open("ab") as stream:stream.write(b"changed")
    elif tamper=="anchor_hash":
        with (anchor_folder/"quality_shards/part_000000.parquet").open("ab") as stream:stream.write(b"changed")
    elif tamper=="checkpoint_hash":
        checkpoint=Path(json.loads(path.read_text())["identity"]["checkpoint"])
        checkpoint.write_bytes(b"changed checkpoint")
    elif tamper=="anchor_key_cached":
        value=json.loads(path.read_text());value["anchor_source"]["key"]="0"*64;write(path,value)
    elif tamper=="label_with_valid_hash":
        rows=pq.read_table(block).to_pylist();rows[0]["label"]=1-rows[0]["label"]
        pq.write_table(pa.Table.from_pylist(rows),block)
        marker=block.with_suffix(".json");value=json.loads(marker.read_text());value["sha256"]=sha256(block);write(marker,value)
    else:
        (path.parent/"probe_blocks/part_000256.json").unlink()
    with pytest.raises(ValueError):
        analyze(paths.values())


def test_duplicate_method_and_different_stage_population_are_rejected(tmp_path):
    paths,_=fixture(tmp_path,seeds=(17,))
    with pytest.raises(ValueError,match="duplicate variant"):
        analyze([*paths.values(),paths[17,"f"]])
    path=paths[17,"fg"]
    summary=json.loads(path.read_text())
    summary["identity"]["users"]=summary["identity"]["users"][:-1]
    write(path,summary)
    with pytest.raises(ValueError):
        analyze(paths.values())


def test_zero_reference_gap_keeps_auc_increment_but_never_fabricates_recovery_interval(tmp_path):
    paths,anchor_folder=fixture(tmp_path,seeds=(17,))
    for path in (anchor_folder/"quality_shards").glob("part_*.parquet"):
        rows=pq.read_table(path).to_pylist()
        reuse={tuple(row[name] for name in ("uid","query_timestamp","raw_item_id")):row["score"]
               for row in rows if row["method"]=="reuse"}
        for row in rows:
            if row["method"]=="rebuild":
                row["score"]=reuse[tuple(row[name] for name in ("uid","query_timestamp","raw_item_id"))]
        pq.write_table(pa.Table.from_pylist(rows),path)
        marker=path.with_suffix(".json");value=json.loads(marker.read_text());value["scores_sha256"]=sha256(path);write(marker,value)
    for path in paths.values():
        summary=json.loads(path.read_text());summary["rebuild_auc"]=summary["reuse_auc"];write(path,summary)
    result=analyze(paths.values())
    assert result["anchors"]["reference_gap"]==0
    assert result["anchors"]["nonpositive_or_invalid_gap_replicates"]==1000
    h=result["groups"][0]["per_seed"][0]["increments"]["h"]
    assert h["auc_difference"]>0 and h["auc_difference_ci95"][0]>0
    assert h["recovery_percentage_point_difference"] is None
    assert h["recovery_difference_ci95"] is None
