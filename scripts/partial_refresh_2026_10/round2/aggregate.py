#!/usr/bin/env python3
"""Merge completed round-2 runs of one frozen strategy and exact UID prefix.

Preserve each run's shared preparation charges and launch source hashes. GPU
replicas each pay their own static-table preparation; a separate fixed30k cost
report describes a declared deployment replica count. Memory sums of individual
peaks are not observed loaded-serving peaks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from design_two.prepare import checked,record
from partial_refresh_2026_10.aggregate import resolved,same_columns,totals_from_units,IDENTITY,RELEASES
from partial_refresh_2026_10.analyze import summarize,summarize_cohorts,report
from unified_reuse_2026_09.common import sha256,write_json


def merge(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen_path=resolved(args.frozen)
    frozen=json.loads(frozen_path.read_text())
    candidate=frozen["candidates"][args.candidate]
    strategy=candidate["strategy"]
    mode=candidate["storage_mode"]
    strategy_path=checked(candidate["strategy_file"]).resolve()
    if json.loads(strategy_path.read_text())!=strategy:
        raise RuntimeError("frozen strategy bytes and payload differ")
    panel_dir=resolved(args.panel)
    if checked(frozen["quality_panel"]).resolve()!=panel_dir/"binding.json":
        raise RuntimeError("aggregate panel differs from frozen candidate population")
    binding=json.loads((panel_dir/"binding.json").read_text())
    groups=json.loads(checked(binding["users_file"]).read_text())
    if not 0<args.users<=len(groups["evaluation"]):
        raise ValueError("users must select a nonempty frozen evaluation prefix")
    selected=groups["evaluation"][:args.users]
    selected_set=set(selected)
    expected=pq.read_table(checked(binding["files"]["requests"]),filters=[("uid","in",selected)])
    expected=expected.sort_by([("request_id","ascending")])
    control_path=checked(frozen["control_scores"]).resolve()
    controls=pq.read_table(control_path,filters=[("uid","in",selected)]).sort_by([("request_id","ascending")])
    same_columns(controls,expected,IDENTITY,"immutable control identities vs frozen panel")
    if len(set(expected["request_id"].to_pylist()))!=len(expected):
        raise RuntimeError("frozen panel has duplicate request identities")
    for weight in frozen["weights"].values():
        if sha256(resolved(weight["path"]))!=weight["sha256"]:
            raise RuntimeError("frozen adapter weights changed")
    static_source=None
    if strategy.get("static_item_file"):
        static_source=frozen["static_item_ids"]
        if checked(static_source).resolve()!=resolved(strategy["static_item_file"]):
            raise RuntimeError("static item table differs from frozen strategy")
    tables,units,memberships,sources,inventories=[],[],[],[],[]
    source_hashes={}
    shared_selection_costs=Counter()
    checks=Counter()
    for folder_arg in args.runs:
        folder=resolved(folder_arg)
        summary_path,config_path=folder/"summary.json",folder/"configuration.json"
        summary,config=(json.loads(path.read_text()) for path in (summary_path,config_path))
        if summary["status"]!="complete":
            raise RuntimeError(f"unfinished run: {folder}")
        if summary["storage_mode"]!=mode or config["storage_mode"]!=mode:
            raise RuntimeError("run storage mode differs from frozen candidate")
        if summary["weights"]!=frozen["weights"]:
            raise RuntimeError("run weights differ from frozen candidate")
        if summary["exploration_strategy"]!=strategy or config["exploration_strategy"]!=strategy:
            raise RuntimeError("run strategy differs from frozen candidate")
        if (resolved(config["exploration_strategy_source"])!=strategy_path
                or config["exploration_strategy_sha256"]!=candidate["strategy_file"]["sha256"]):
            raise RuntimeError("launch strategy source does not bind frozen bytes")
        if (resolved(config["quality_panel"])!=panel_dir
                or config["quality_panel_sha256"]!=frozen["quality_panel"]["sha256"]):
            raise RuntimeError("launch panel does not bind frozen inputs")
        if (resolved(config["frozen_reference"])!=control_path
                or config["frozen_reference_sha256"]!=frozen["control_scores"]["sha256"]):
            raise RuntimeError("launch controls do not bind immutable reference")
        if static_source is not None and config["static_item_table_sha256"]!=static_source["sha256"]:
            raise RuntimeError("launch static item IDs differ from frozen candidate")
        if not config["no_policy_selection_on_evaluation"]:
            raise RuntimeError("run selected policy on its evaluation requests")
        if summary["policy"]!=config["policy"]:
            raise RuntimeError("nominal launch/final policy records differ")
        # Actual rhos follow strategy.long_rho. The shared legacy runner's
        # nominal policy may remain all1 and is not the serving strategy.
        declared=json.loads((folder/"users.json").read_text())
        prefix=groups["evaluation"][:config["users"]]
        selector=prefix[config.get("start_user",0):][config["shard_index"]::config["shards"]]
        if (config["all_evaluation_uids"]!=prefix or config["evaluation_uids"]!=selector
                or declared!=selector or len(declared)!=summary["users"]):
            raise RuntimeError("run membership differs from exact frozen prefix/start/shard selector")
        if len(set(declared))!=len(declared) or not set(declared).issubset(selected_set):
            raise RuntimeError("duplicate UID or UID outside aggregate prefix")
        if [unit["uid"] for unit in summary["units"]]!=declared:
            raise RuntimeError("per-user ledger differs from declared run membership")
        score_path=folder/"scores.parquet"
        table=pq.read_table(score_path).sort_by([("request_id","ascending")])
        mask=pa.compute.is_in(expected["uid"],value_set=pa.array(declared))
        same_columns(table,expected.filter(mask),IDENTITY,"run requests vs frozen panel")
        cmask=pa.compute.is_in(controls["uid"],value_set=pa.array(declared))
        same_columns(table,controls.filter(cmask),
            (*IDENTITY,"reuse_logit","full_logit","first_request_rebuild_logit"),
            "run immutable control scores")
        if len(table)!=summary["requests"] or len(set(table["request_id"].to_pylist()))!=len(table):
            raise RuntimeError("run request count/uniqueness differs from evidence")
        for column in ("reuse_logit","full_logit","partial_logit","first_request_rebuild_logit"):
            if not np.isfinite(table[column].to_numpy()).all():
                raise RuntimeError(f"nonfinite {column}")
        for version in RELEASES:
            selected_rows=table["version"].to_numpy()==version
            if not np.all(table["rho"].to_numpy()[selected_rows]==strategy["long_rho"][version]):
                raise RuntimeError("actual served rho differs from frozen strategy")
        counters,_,_=totals_from_units(summary["units"])
        if Counter(counters)!=Counter(summary["actual_service_counters"]):
            raise RuntimeError("per-user counters differ from run totals")
        inventory=summary["shared_encoder_inventory"]
        for actual_key,inventory_key in (("encoder_fold_preparation_flops","parameter_preparation_flops"),
                ("item_projection_flops","item_projection_flops"),("item_lookup_bytes","item_lookup_bytes")):
            recorded=sum(encoder["counters"].get(inventory_key,0) for encoder in inventory)
            if recorded!=counters.get(actual_key,0):
                raise RuntimeError(f"shared inventory fee mismatch: {actual_key}")
        recorded_kv=sum(encoder["counters"].get("row_kv_flops",0) for encoder in inventory)
        ledger_kv=sum(counters.get(key,0) for key in ("initial_encode_flops","refresh_encode_flops","append_encode_flops"))
        if recorded_kv!=ledger_kv:
            raise RuntimeError("shared encoder inventory row KV arithmetic differs from per-user ledger")
        for name,value in summary["checks"].items():
            if name.endswith("checked_snapshots"):
                checks[name]+=value
            else:
                checks[name]=max(checks[name],value)
        if summary["source_code_sha256"]!=config["source_code_sha256"]:
            raise RuntimeError("launch/final source hash records differ")
        source_hashes[str(folder)]=summary["source_code_sha256"]
        sources.append(dict(summary=record(summary_path),configuration=record(config_path),
                            users=record(folder/"users.json"),scores=record(score_path)))
        inventories.append(dict(run=str(folder),encoders=inventory))
        shared_selection_costs.update(summary.get("shared_selection_costs",{}))
        units.extend(summary["units"]);memberships.extend(declared);tables.append(table)
    if len(memberships)!=len(set(memberships)) or set(memberships)!=selected_set:
        raise RuntimeError("merged UID population must exactly cover selected users including idle users")
    merged=pa.concat_tables(tables).sort_by([("request_id","ascending")])
    same_columns(merged,expected,IDENTITY,"merged complete request coverage")
    if len(set(merged["request_id"].to_pylist()))!=len(merged):
        raise RuntimeError("merged requests repeat")
    unit_by_uid={unit["uid"]:unit for unit in units}
    units=[unit_by_uid[uid] for uid in selected]
    counters,by_release,memory=totals_from_units(units)
    quality=summarize(merged,selected,repetitions=args.bootstrap,releases=RELEASES)
    # Keep the historical95%-preservation test explicitly. The current user's
    # target is90%, reported as a separate ratio-CI decision and paired margin.
    threshold=frozen["target"]["full_code_gain_retention_lower_95_percent_above"]
    q90=summarize(merged,selected,repetitions=args.bootstrap,releases=RELEASES,
                  preservation_threshold=threshold/100)
    retention=quality["overall"]["paired"]["full_gain_retained_percent"]
    bounds=retention["percentile_95_interval"]
    acceptance=dict(target_retention_percent=threshold,
        paired_uid_ratio_lower_95_above_target=bool(bounds is not None and bounds[0]>threshold),
        paired_margin_at_current_target=q90["overall"]["paired"]["preservation_margin_auc"],
        paired_margin_supported_at_current_target=q90["overall"]["paired"]["noninferiority_supported_at_threshold"],
        maintenance_target_percent=frozen["target"]["maintenance_saving_percent_at_least"],
        maintenance_scope="Exact fixed30k serving/maintenance FLOPs are evaluated separately; this quality merge does not establish the saving target.")
    cohort_results=None
    if args.cohorts:
        cohort_path=resolved(args.cohorts)
        cohort_results=summarize_cohorts(merged,selected,json.loads(cohort_path.read_text()),
                                        repetitions=args.bootstrap,releases=RELEASES)
    args.output.mkdir(parents=True)
    pq.write_table(merged,args.output/"scores.parquet",compression="zstd")
    write_json(args.output/"users.json",selected)
    source=dict(aggregate=record(Path(__file__)),analyze=record(ROOT/"scripts/partial_refresh_2026_10/analyze.py"))
    configuration=dict(stage="round2_complete_strategy_aggregate",candidate=args.candidate,
        frozen_candidates=record(frozen_path),exploration_strategy=strategy,storage_mode=mode,
        quality_panel=str(panel_dir),quality_panel_sha256=frozen["quality_panel"]["sha256"],
        evaluation_uids=selected,all_evaluation_uids=selected,users=args.users,start_user=0,shards=1,shard_index=0,
        actual_rho_by_release=strategy["long_rho"],no_policy_selection_on_evaluation=True,
        frozen_reference=frozen["control_scores"],static_item_table=static_source,
        source_code_sha256=source,source_code_sha256_by_run=source_hashes)
    write_json(args.output/"configuration.json",configuration)
    result=dict(status="complete",role="round2_frozen_strategy_full_population_development",
        candidate=args.candidate,users=len(selected),requests=len(merged),storage_mode=mode,
        exploration_strategy=strategy,actual_rho_by_release=strategy["long_rho"],weights=frozen["weights"],
        quality=quality,acceptance=acceptance,units=units,checks=dict(checks),actual_service_counters=counters,
        by_release_service=by_release,memory=memory,shared_encoder_inventories_by_run=inventories,
        shared_selection_costs=dict(shared_selection_costs),
        shared_charge_scope="Every independently launched run pays its own fold/static-item preparation. All fees remain in these executed counters; fixed30k cost with a declared replica count is a separate evidence artifact.",
        sources=dict(runs=sources,frozen_candidates=record(frozen_path),panel=frozen["quality_panel"],
            controls=frozen["control_scores"],strategy=candidate["strategy_file"],
            static_item_ids=static_source,cohorts=record(resolved(args.cohorts)) if args.cohorts else None),
        scores=record(args.output/"scores.parquet"),source_code_sha256=source,source_code_sha256_by_run=source_hashes,
        verification=dict(exact_uid_union_including_idle_users=True,exact_request_label_timestamp_coverage=True,
            immutable_control_scores_exact=True,same_frozen_strategy_storage_weights=True,
            actual_served_rhos_match_strategy=True,shared_inventory_fees_match_executed_ledgers=True),
        scope=frozen.get("selection_scope","Existing development population; no independent blind or seed replication claim."))
    if cohort_results is not None:
        result["cohorts"]=cohort_results
        write_json(args.output/"cohorts_summary.json",cohort_results)
    write_json(args.output/"summary.json",result)
    text=(f"# Round 2: {args.candidate}\n\nUser target: ratio UID95% lower bound above {threshold:g}%. "
          f"Passed: {acceptance['paired_uid_ratio_lower_95_above_target']}. "
          f"Paired margin at that target: {acceptance['paired_margin_at_current_target']}.\n\n"
          "The historical95% preservation test is retained below, separately from the current90% target.\n\n")
    text+=report(quality)
    text+="\nShared item-table preparation is charged for each run replica. Maintenance saving and deployment memory require the separate fixed30k ledger; no loaded-service timing is measured.\n"
    if cohort_results is not None:
        for name,cohort in cohort_results.items():
            text+=f"\n## {name}\n\n"
            text+=report(cohort) if "overall" in cohort else json.dumps(cohort)+"\n"
    (args.output/"report.md").write_text(text)
    print(json.dumps(dict(status="complete",candidate=args.candidate,output=str(args.output),
                         users=len(selected),requests=len(merged),acceptance=acceptance)),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,nargs="+",required=True)
    parser.add_argument("--panel",type=Path,required=True)
    parser.add_argument("--users",type=int,default=3360)
    parser.add_argument("--frozen",type=Path,required=True)
    parser.add_argument("--candidate",required=True)
    parser.add_argument("--cohorts",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--bootstrap",type=int,default=1000)
    merge(parser.parse_args())
