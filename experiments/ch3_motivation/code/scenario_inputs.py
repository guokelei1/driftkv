"""Qualified scenario inputs and separate causal packs for new cohorts.

Original population files, original packs and running numerical sources are
never edited. Calibration histories are copied exactly from their original
pack, so a change of evaluation membership does not change fitting inputs.
"""
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
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.ch3_motivation.code import common
from hstu_kvcache.evaluation.history import load_histories_semijoin, restore_pack
from hstu_kvcache.common.artifacts import sha256


def checked_reference(value,name):
    path=Path(value["path"])
    if not path.is_absolute() or sha256(path)!=value["sha256"]:
        raise ValueError(f"scenario {name} path/hash differs")
    return path


def load_scenario_population(manifest_path,edge,*,role="quality",minimum_users=512,require_qualification=True,require_positive=True):
    """Load exact measured membership; positivity is optional for full-edge reports."""
    manifest_path=Path(manifest_path).resolve()
    manifest=json.loads(manifest_path.read_text())
    if manifest["schema_version"]!=1 or manifest["edge"]!=edge:
        raise ValueError("scenario schema/edge differs")
    binding_path=checked_reference(manifest["input_binding"],"input binding")
    binding=json.loads(binding_path.read_text())
    history=checked_reference(manifest["history"],"history")
    requests=checked_reference(manifest["requests"],"requests")
    population_path=checked_reference(manifest["source_population"],"role source")
    population=json.loads(population_path.read_text())
    if binding["settings"]["edge"]!=edge or binding["population_sha256"]!=sha256(population_path):
        raise ValueError("scenario inputs do not match edge/frozen roles")
    for name,path in (("history",history),("requests",requests)):
        if binding[name]["sha256"]!=sha256(path):
            raise ValueError(f"scenario {name} differs from input binding")
    quality=[int(uid) for uid in manifest["roles"]["quality"]]
    users=[int(uid) for uid in manifest["roles"][role]]
    if (quality!=manifest["uids"] or len(set(quality))!=len(quality)
            or len(set(users))!=len(users) or len(users)<minimum_users or not set(users)<=set(quality)):
        raise ValueError("scenario user membership/count is invalid")
    if set(population["fit"])&set(population["validation"]) or set(quality)&set(population["fit"]+population["validation"]):
        raise ValueError("scenario evaluation overlaps reserved fit/validation users")
    with np.load(history,allow_pickle=False) as arrays:
        if not set(quality+population["fit"]+population["validation"])<=set(int(uid) for uid in arrays["uids"]):
            raise ValueError("scenario histories do not cover evaluation and calibration roles")
    if role=="e6":
        feedback=set(int(uid) for uid in pq.read_table(requests,columns=["uid"])["uid"].to_numpy())
        if not set(users)<=feedback:
            raise ValueError("E6 scenario includes users without observed feedback")
    provenance=dict(manifest_path=str(manifest_path),manifest_sha256=sha256(manifest_path),
        input_binding_path=str(binding_path),input_binding_sha256=sha256(binding_path),
        history_path=str(history),history_sha256=sha256(history),requests_path=str(requests),requests_sha256=sha256(requests),
        source_population_path=str(population_path),source_population_sha256=sha256(population_path),selection=manifest["selection"])
    if require_qualification:
        qualification=manifest["qualification"]
        summary_path=Path(qualification["summary_path"])
        if not summary_path.is_absolute() or sha256(summary_path)!=qualification["summary_sha256"]:
            raise ValueError("scenario three-anchor summary changed")
        summary=json.loads(summary_path.read_text())
        if summary["quality_uids"]!=quality or summary["quality_users"]!=len(quality) or summary["quality_uids_sha256"]!=hashlib.sha256(json.dumps(quality).encode()).hexdigest():
            raise ValueError("scenario qualification measured another population")
        fields=("new_rebuild_minus_old_full","new_rebuild_minus_reuse","point_qualifies","uid95_qualifies")
        if any(qualification[key]!=summary["qualification"][key] or qualification[key]!=summary[key] for key in fields):
            raise ValueError("scenario qualification metadata differs from summary")
        old_gain=summary["auc"]["rebuild"]-summary["auc"]["old_full"]
        cache_gap=summary["auc"]["rebuild"]-summary["auc"]["reuse"]
        if abs(old_gain-qualification["new_rebuild_minus_old_full"])>1e-12 or abs(cache_gap-qualification["new_rebuild_minus_reuse"])>1e-12:
            raise ValueError("scenario measured AUC differences disagree")
        if bool(qualification["point_qualifies"])!=(old_gain>0 and cache_gap>0):
            raise ValueError("scenario point qualification contradicts measured anchors")
        old_ci=summary["new_rebuild_minus_old_full_ci95"]
        cache_ci=summary["new_rebuild_minus_reuse_ci95"]
        if bool(qualification["uid95_qualifies"])!=bool(old_ci and cache_ci and old_ci[0]>0 and cache_ci[0]>0):
            raise ValueError("scenario interval qualification contradicts UID intervals")
        if require_positive and not qualification["point_qualifies"]:
            raise ValueError("scenario lacks positive overall model gain and positive cache loss")
        if role=="e6" and users!=quality:
            raise ValueError("E6 qualification must be recomputed on exactly its feedback population")
        if summary["identity"]["input_binding_sha256"]!=sha256(binding_path):
            raise ValueError("scenario qualification used another input binding")
        provenance.update(mode="frozen_three_anchor_qualified_scenario" if qualification["point_qualifies"] else "frozen_three_anchor_measured_scenario",
            qualification={key:qualification[key] for key in fields},
            qualification_summary_path=str(summary_path),qualification_summary_sha256=sha256(summary_path),
            anchor_models=summary["identity"]["models"],scope=qualification["scope"])
    else:
        provenance.update(mode="unqualified_candidate_inputs",scope="baseline qualification only; not approved as a method-quality scenario")
    calibration=binding.get("calibration_source_binding",manifest["input_binding"])
    calibration_path=checked_reference(calibration,"calibration input binding")
    calibration_binding=json.loads(calibration_path.read_text())
    if calibration_binding["settings"]!=binding["settings"] or calibration_binding["population_sha256"]!=sha256(population_path):
        raise ValueError("scenario calibration source uses another edge/role population")
    if calibration_path!=binding_path and not binding.get("calibration_inputs_unchanged",False):
        raise ValueError("changed evaluation pack has no preserved calibration-input identity")
    if calibration_path!=binding_path:
        source_history=calibration_path.parent/calibration_binding["history"]["path"]
        if sha256(source_history)!=calibration_binding["history"]["sha256"]:
            raise ValueError("original calibration history pack changed")
        calibration_users=population["fit"]+population["validation"]
        expected=history_signature(restore_pack(source_history),calibration_users)
        if (history_signature(restore_pack(history),calibration_users)!=expected
                or binding.get("calibration_history_sha256")!=expected):
            raise ValueError("new scenario changed frozen calibration history tuples")
    provenance.update(calibration_input_binding_path=str(calibration_path),calibration_input_binding_sha256=sha256(calibration_path))
    return binding,users,population,provenance


def load_scenario_data(provenance):
    histories=restore_pack(Path(provenance["history_path"]))
    by_user=defaultdict(lambda:defaultdict(list))
    for row in pq.read_table(provenance["requests_path"]).to_pylist():
        by_user[int(row["uid"])][int(row["query_timestamp"])].append(row)
    return histories,by_user


def history_signature(histories,users):
    digest=hashlib.sha256()
    for uid in users:
        digest.update(str(uid).encode()+b"\n")
        for column in histories.rows[uid]:
            digest.update(np.asarray(column).tobytes())
    return digest.hexdigest()


def prepare_natural_packs(input_root,strata_path,output_root,*,edges=(1,2,3,4,5),threads=8):
    """One new union pack per edge, two fixed 2k publishing-time cohort views."""
    input_root,output_root=Path(input_root),Path(output_root)
    population_path=input_root/"population.json"
    population=json.loads(population_path.read_text())
    strata=json.loads(Path(strata_path).read_text())
    if strata["source_population"]["sha256"]!=sha256(population_path):
        raise ValueError("strata use different frozen roles")
    dataset=json.loads(common.DATASET.read_text())
    results=[]
    for edge in edges:
        started=time.perf_counter()
        original=input_root/f"edge_{edge}"
        binding=json.loads((original/"inputs.json").read_text())
        settings=binding["settings"]
        panels={name:strata["subgroups_by_edge"][str(edge)][name]["first_2000_uid_hashes"]
                for name in ("long_feedback_active","long_feedback_dense")}
        if any(len(users)!=2000 for users in panels.values()):
            raise ValueError("natural scenario needs 2000 predeclared users")
        users=sorted(set(population["fit"]+population["validation"]+[uid for group in panels.values() for uid in group]))
        old_history=restore_pack(original/binding["history"]["path"])
        missing=[uid for uid in users if uid not in old_history.rows]
        output=output_root/f"edge_{edge}"
        output.mkdir(parents=True,exist_ok=True)
        if (output/"inputs.json").exists():
            prior=json.loads((output/"inputs.json").read_text())
            if prior["scenario_strata_sha256"]!=sha256(Path(strata_path)):
                raise ValueError("existing natural input attempt uses other strata")
            results.append(prior)
            continue
        additional=load_histories_semijoin(missing,dataset_path=common.DATASET,
            known_vocab_size=dataset["foundation_items"],oov_buckets=256,
            start_timestamp=settings["release_day"]*common.DAY,end_timestamp=settings["end_day"]*common.DAY,
            max_history=settings["history_length"],threads=threads) if missing else SimpleNamespace(rows={})
        histories=SimpleNamespace(rows={uid:(old_history.rows[uid] if uid in old_history.rows else additional.rows[uid]) for uid in users})
        calibration_users=population["fit"]+population["validation"]
        signature=history_signature(histories,calibration_users)
        if signature!=history_signature(old_history,calibration_users):
            raise ValueError("new scenario changed frozen calibration histories")
        common.save_histories(output/"histories.npz",histories,users)
        connection=duckdb.connect()
        connection.execute(f"PRAGMA threads={int(threads)}")
        connection.register("selected",pa.table({"uid":pa.array(users,type=pa.uint64())}))
        feedback=common.quoted((common.DATASET.parent/dataset["shared_feedback_glob"]).resolve())
        mapping=common.quoted((common.DATASET.parent/dataset["item_mapping_path"]).resolve())
        requests=connection.execute(f"""
            WITH grouped AS (
              SELECT uid,timestamp,raw_item_id,min(label) AS label
              FROM read_parquet('{feedback}',hive_partitioning=true) SEMI JOIN selected USING(uid)
              WHERE timestamp>=? AND timestamp<? GROUP BY uid,timestamp,raw_item_id HAVING min(label)=max(label)
            ) SELECT g.uid,g.timestamp AS query_timestamp,g.raw_item_id,m.item_idx,g.label
              FROM grouped g JOIN read_parquet('{mapping}') m USING(raw_item_id)
              ORDER BY uid,query_timestamp,raw_item_id
        """,[settings["release_day"]*common.DAY,settings["end_day"]*common.DAY]).fetch_arrow_table()
        connection.close()
        pq.write_table(requests,output/"requests.parquet")
        active=set(int(uid) for uid in requests["uid"].to_numpy())
        value=dict(settings=settings,population_sha256=sha256(population_path),
            history=dict(path="histories.npz",sha256=sha256(output/"histories.npz")),
            requests=dict(path="requests.parquet",sha256=sha256(output/"requests.parquet")),
            calibration_source_binding=dict(path=str((original/"inputs.json").resolve()),sha256=sha256(original/"inputs.json")),
            calibration_inputs_unchanged=True,calibration_history_sha256=signature,
            scenario_strata_sha256=sha256(Path(strata_path)),candidate_panels=panels,
            packed_users=len(users),original_pack_covered_users=len(users)-len(missing),newly_loaded_users=len(missing),
            cohort_coverage={name:dict(selected=len(group),covered=sum(uid in old_history.rows for uid in group),missing=sum(uid not in old_history.rows for uid in group)) for name,group in panels.items()},
            role="causal CPU candidate pack; no model prediction or qualification yet",seconds=time.perf_counter()-started)
        common.write_json(output/"inputs.json",value)
        for name,group in panels.items():
            manifest=dict(schema_version=1,edge=edge,uids=group,roles=dict(quality=group,e6=[uid for uid in group if uid in active]),
                qualification=dict(status="not_yet_measured"),
                input_binding=dict(path=str((output/"inputs.json").resolve()),sha256=sha256(output/"inputs.json")),
                history=dict(path=str((output/"histories.npz").resolve()),sha256=value["history"]["sha256"]),
                requests=dict(path=str((output/"requests.parquet").resolve()),sha256=value["requests"]["sha256"]),
                source_population=dict(path=str(population_path.resolve()),sha256=sha256(population_path)),
                selection=dict(panel_name=name,description="fixed 2000 UID hashes from declared publishing-time history/activity/feedback thresholds; no model-gain filtering",
                    strata_manifest_path=str(Path(strata_path).resolve()),strata_manifest_sha256=sha256(Path(strata_path)),definition=strata["definitions"][name]))
            common.write_json(output/name/"candidate_manifest.json",manifest)
        results.append(value)
        print(json.dumps(dict(edge=edge,packed_users=len(users),newly_loaded_users=len(missing),cohort_coverage=value["cohort_coverage"],seconds=value["seconds"])),flush=True)
    common.write_json(output_root/"preparation_summary.json",dict(role="CPU only, all cohorts retained, no new prediction",edges=results))
    return results


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,default=ROOT/"experiments/ch3_motivation/inputs")
    parser.add_argument("--strata-manifest",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--threads",type=int,default=8)
    args=parser.parse_args()
    prepare_natural_packs(args.input_root,args.strata_manifest,args.output_root,threads=args.threads)
