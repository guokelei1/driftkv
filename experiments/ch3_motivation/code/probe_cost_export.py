"""Final E5 raw/corrected costs from already audited quality summaries.

No score shards or event ledgers are reread. Five-edge ratios use summed FLOPs;
populations, correction seeds, methods, fitting sizes and recipes stay separate.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from experiments.ch3_motivation.code.calibration_cost_correction import checkpoint_correction, apply_to_costs
from hstu_kvcache.common.artifacts import sha256

ROOT=Path(__file__).resolve().parents[3]
ACTUAL_COMPONENTS=("calibration_flops","release_preparation_flops","extra_read_flops",
                   "extra_native_writer_flops","shared_graph_setup_flops")
REFERENCE_COMPONENTS=("calibration_flops","release_preparation_flops","extra_candidate_read_flops",
    "actual_behavior_read_extra_flops","extra_native_writer_flops","shared_graph_setup_flops")
COMPONENTS=tuple(dict.fromkeys(ACTUAL_COMPONENTS+REFERENCE_COMPONENTS))
EDGE_FIELDS={"edge","models","input_binding_sha256","calibration_candidate_sha256",
    "sampling_manifest","sampling_manifest_sha256","gap_manifest","gap_manifest_sha256","source_cutoff","seed"}


def sources():
    return {str(path.relative_to(ROOT)):sha256(path) for path in (Path(__file__),
        Path(__file__).with_name("calibration_cost_correction.py"),ROOT/"src/hstu_kvcache/common/artifacts.py")}


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def _validate_account(account,components):
    if any(type(account.get(key)) is not int for key in (*components,"complete_added_flops","once_rebuild_flops")):
        raise ValueError("actual integer cost components, total and denominator are required")
    if sum(account[key] for key in components)!=account["complete_added_flops"]:
        raise ValueError("E5 complete cost does not equal its declared components")
    if account["once_rebuild_flops"]<=0 or not math.isclose(account["percent_of_once_rebuild"],
            100*account["complete_added_flops"]/account["once_rebuild_flops"],rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError("E5 cost denominator/percentage differs from its actual FLOPs")


def _account_rows(point):
    shared={key:point[key] for key in ("family","panel","edge","seed","method_label","fit_users",
        "variant","width","depth","configuration_sha256","quality_users","quality_uids_sha256")}
    for ledger in ("actual_event","analytic230"):
        raw=point["raw"] if ledger=="actual_event" else point["raw"]["candidate_reads_230"]
        corrected=point["corrected"] if ledger=="actual_event" else point["corrected"]["candidate_reads_230"]
        yield dict(shared,ledger=ledger,source_summary=point["source_summary"],
            checkpoint_sha256=point["calibration_correction"]["checkpoint_sha256"],
            calibration_accounting="shared_row_normalization_corrected_v1",
            raw_calibration_flops=raw["calibration_flops"],
            calibration_normalization_deduction_flops=point["calibration_correction"]["shared_row_normalization_deduction_flops"],
            raw_complete_added_flops=raw["complete_added_flops"],complete_added_flops=corrected["complete_added_flops"],
            once_rebuild_flops=raw["once_rebuild_flops"],raw_percent_of_once_rebuild=raw["percent_of_once_rebuild"],
            percent_of_once_rebuild=corrected["percent_of_once_rebuild"],
            **{key:corrected.get(key) for key in COMPONENTS})


def collect_costs(loaded,analysis,*,family):
    """Called after the existing formal grid/quality audits, never by the core.

    Only source paths retained in the completed quality aggregate are exported;
    its explicitly out-of-scope points remain outside these cost aggregates.
    """
    if family not in ("ladder","f_population_curve"):
        raise ValueError("only the two existing formal E5 families are supported")
    code=sources()
    result=dict(format="e5_probe_cost_export_v1",status="pending_complete_audited_grid",family=family,
        points=[],aggregates=[],source_sha256=code,gpu_used=False,score_shards_reread=False,
        scope="raw and calibration-corrected standalone costs; each five-edge ratio is sum(complete_added)/sum(once_Rebuild); no pooling of populations, seeds, methods, fitting sizes or recipes",
        deployment_work_changed=False,original_summaries_and_weights_modified=False)
    if analysis.get("status")!="complete":
        return result
    if family=="ladder":
        reports=list(analysis["panels"].values())
        if any(not row["primary_aggregate_computed"] or not row["formal_plan_prepared"] for row in reports):
            return result
        paths=[method["source_summary"] for row in reports for edge in row["edge_records"] for method in edge["methods"]]
        bound_flag="formal_main_calibration_bound"
    else:
        if not analysis["primary_aggregate_computed"] or not analysis["formal_curve_plan_bound"]:
            return result
        paths=[point["source_summary"] for edge in analysis["edge_records"] for point in edge["points"]]
        bound_flag="formal_curve_calibration_bound"
    wanted={str(Path(path).resolve()) for path in paths}
    if not wanted or len(wanted)!=len(paths):
        raise ValueError("completed E5 aggregate has empty or duplicate source-summary coverage")
    selected={str(Path(item["path"]).resolve()):item for item in loaded if str(Path(item["path"]).resolve()) in wanted}
    if set(selected)!=wanted:
        raise ValueError("cost export is missing an audited quality summary")
    groups=defaultdict(list);cache={}
    for path,item in sorted(selected.items()):
        summary,training=item["summary"],item["training"]
        if training.get(bound_flag) is not True:
            raise ValueError("E5 cost export requires the actual per-edge formal checkpoint/input audit")
        identity=summary["identity"]
        expected=identity["checkpoint_sha256"]
        if expected!=training["checkpoint_sha256"]:
            raise ValueError("cost and training checkpoint bindings differ")
        key=(str(Path(identity["checkpoint"]).resolve()),expected)
        if key not in cache:
            cache[key]=checkpoint_correction(key[0],expected_sha256=expected)
        correction=cache[key]
        if (correction["fitting_identity"]!=training["identity"] or correction["config"]!=training["config"] or
                correction["method_label"]!=summary["method_label"] or correction["variant"]!=summary["variant"]):
            raise ValueError("actual calibration metadata differs from audited E5 training record")
        for account,components in ((summary["costs"],ACTUAL_COMPONENTS),
                (summary["costs"]["candidate_reads_230"],REFERENCE_COMPONENTS)):
            _validate_account(account,components)
        exported=apply_to_costs(summary["costs"],correction)
        for account,components in ((exported["corrected"],ACTUAL_COMPONENTS),
                (exported["corrected"]["candidate_reads_230"],REFERENCE_COMPONENTS)):
            _validate_account(account,components)
        config=dict(variant=summary["variant"],fit_users=summary["fit_users"],probe_config=training["config"],
            backend=identity["backend"],fitting_recipe={k:v for k,v in training["identity"].items() if k not in EDGE_FIELDS})
        point=dict(family=family,panel=identity["panel"],edge=summary["edge"],seed=summary["seed"],
            method_label=summary["method_label"],fit_users=summary["fit_users"],variant=summary["variant"],
            width=summary["width"],depth=summary["depth"],configuration=config,configuration_sha256=_digest(config),
            quality_users=len(identity["users"]),quality_uids_sha256=_digest(identity["users"]),
            source_summary=path,source_summary_sha256=sha256(Path(path)),**exported)
        if point["quality_users"]!=summary["quality_users"]:
            raise ValueError("cost export population differs from quality summary")
        result["points"].append(point)
        for row in _account_rows(point):
            group=tuple(row[key] for key in ("family","panel","quality_uids_sha256","seed","method_label",
                "fit_users","variant","width","depth","configuration_sha256","ledger"))
            groups[group].append(row)
    for group,rows in sorted(groups.items()):
        if sorted(row["edge"] for row in rows)!=[1,2,3,4,5]:
            raise ValueError("each E5 cost aggregate requires all five distinct prescribed edges")
        aggregate={key:rows[0][key] for key in ("family","panel","seed","method_label","fit_users","variant",
            "width","depth","configuration_sha256","ledger","quality_users","quality_uids_sha256","calibration_accounting")}
        for key in (*COMPONENTS,"raw_calibration_flops","calibration_normalization_deduction_flops",
                "raw_complete_added_flops","complete_added_flops","once_rebuild_flops"):
            aggregate[key]=sum(row[key] for row in rows) if rows[0][key] is not None else None
        aggregate.update(edges=[1,2,3,4,5],edge_count=5,user_release_pairs=5*aggregate["quality_users"],
            raw_percent_of_once_rebuild=100*aggregate["raw_complete_added_flops"]/aggregate["once_rebuild_flops"],
            percent_of_once_rebuild=100*aggregate["complete_added_flops"]/aggregate["once_rebuild_flops"],
            aggregation="sum FLOPs across five edges before ratio; no mean of edge percentages")
        result["aggregates"].append(aggregate)
    if sources()!=code:
        raise ValueError("cost export sources changed while auditing checkpoints")
    result.update(status="complete",point_count=len(result["points"]),aggregate_count=len(result["aggregates"]))
    return result


def write_cost_csvs(costs,output):
    if costs["status"]!="complete":
        return []
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    written=[]
    for name,rows in (("cost_points.csv",[row for point in costs["points"] for row in _account_rows(point)]),
            ("cost_aggregates.csv",costs["aggregates"])):
        path=output/name
        with path.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
            writer.writerows({key:json.dumps(value) if isinstance(value,(list,dict)) else value for key,value in row.items()} for row in rows)
        written.append(str(path.resolve()))
    return written
