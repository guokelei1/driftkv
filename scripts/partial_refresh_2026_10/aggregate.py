#!/usr/bin/env python3
"""Merge completed disjoint runs of one frozen continuous refresh policy.

The selected population is exactly the requested evaluation prefix, including
idle users. Request identities/labels are verified against the frozen panel;
retained baseline scores are also checked exactly when used. Different launch
source hashes are retained explicitly, without rewriting historical provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from design_two.prepare import checked, record
from design_one.prepare_history import restore_pack
from partial_refresh_2026_10.analyze import report, summarize, summarize_cohorts
from unified_reuse_2026_09.common import sha256, write_json

RELEASES = ["v3", "v4", "v5"]
IDENTITY = ("request_id", "uid", "version", "label", "query_timestamp")


def resolved(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def same_columns(actual, expected, keys, reason):
    if actual.num_rows != expected.num_rows:
        raise RuntimeError(f"{reason}: request count differs")
    for key in keys:
        if not np.array_equal(actual[key].to_numpy(), expected[key].to_numpy()):
            raise RuntimeError(f"{reason}: {key} differs")


def totals_from_units(units):
    totals = Counter()
    releases = {version:dict(counters=Counter(), native_query_flops=0) for version in RELEASES}
    memory_sum, memory_max = Counter(), Counter()
    for unit in units:
        if set(unit["releases"]) != set(RELEASES):
            raise RuntimeError("unit lacks a release; do not drop idle user/releases")
        for version, values in unit["releases"].items():
            totals.update(values["counters"])
            releases[version]["counters"].update(values["counters"])
            releases[version]["native_query_flops"] += values["native_query_flops"]
        for name, value in unit["memory"].items():
            memory_sum[name] += value
            memory_max[name] = max(memory_max[name],value)
    return (dict(totals),
        {version:dict(counters=dict(row["counters"]),native_query_flops=row["native_query_flops"])
         for version,row in releases.items()},
        dict(sum_of_individual_user_peak_bytes=dict(memory_sum),
             maximum_individual_user_peak_bytes=dict(memory_max),
             scope="Each user's peak retained state during sequential replay. Summed individual peaks are not simultaneously observed loaded-serving memory; frozen shared model/adapter weights and temporary scratch are excluded."))


def verify_prefix_panel(previous_dir, expanded_dir, expanded, expanded_groups, expected):
    """Allow reuse of a completed prefix only with exact input equivalence."""
    evidence_path = expanded_dir/"extension_check.json"
    evidence = json.loads(evidence_path.read_text())
    if (checked(evidence["previous_panel"]).resolve() != previous_dir/"binding.json"
            or checked(evidence["expanded_panel"]).resolve() != expanded_dir/"binding.json"
            or not all(evidence["checks"].values())):
        raise RuntimeError("prefix extension evidence does not bind these exact panels")
    previous = json.loads((previous_dir/"binding.json").read_text())
    previous_groups = json.loads(checked(previous["users_file"]).read_text())
    previous_uids = previous_groups["evaluation"]
    if expanded_groups["evaluation"][:len(previous_uids)] != previous_uids:
        raise RuntimeError("completed prefix users differ from expanded panel order")
    for key in ("models","model_config","context","cutover","end_timestamp","request_rule","state_rule"):
        if previous[key] != expanded[key]:
            raise RuntimeError(f"prefix model/causal binding differs: {key}")
    previous_requests = pq.read_table(checked(previous["files"]["requests"])).sort_by([("request_id","ascending")])
    expanded_requests = expected.filter(pa.compute.is_in(expected["uid"],value_set=pa.array(previous_uids)))
    if not previous_requests.equals(expanded_requests):
        raise RuntimeError("completed prefix request inputs are not bit equivalent")
    previous_history = restore_pack(checked(previous["files"]["history"]))
    expanded_history = restore_pack(checked(expanded["files"]["history"]))
    for uid in previous_uids:
        if any(not np.array_equal(a,b) for a,b in zip(
                previous_history.rows[uid],expanded_history.rows[uid],strict=True)):
            raise RuntimeError(f"completed prefix history inputs changed: {uid}")
    return dict(evidence=record(evidence_path),previous_panel=record(previous_dir/"binding.json"),
        users=len(previous_uids),requests=len(previous_requests),
        exact_uid_request_history_model_and_causal_equivalence=True)


def merge(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    panel_dir = resolved(args.panel)
    binding = json.loads((panel_dir/"binding.json").read_text())
    groups = json.loads(checked(binding["users_file"]).read_text())
    if not 0 < args.users <= len(groups["evaluation"]):
        raise ValueError("users must select a nonempty evaluation prefix")
    selected = groups["evaluation"][:args.users]
    selected_set = set(selected)
    expected = pq.read_table(checked(binding["files"]["requests"]),
        filters=[("uid","in",selected)]).sort_by([("request_id","ascending")])
    if len(set(expected["request_id"].to_pylist())) != expected.num_rows:
        raise RuntimeError("frozen panel request identities repeat")
    tables, units, source_runs, launch_sources = [], [], [], {}
    combined_users = []
    combined_checks = Counter()
    first = None
    policy_path = None
    accepted_prefixes = {}
    for folder_arg in args.runs:
        folder = resolved(folder_arg)
        summary_path, config_path = folder/"summary.json", folder/"configuration.json"
        summary, config = (json.loads(path.read_text()) for path in (summary_path,config_path))
        if summary["status"] != "complete":
            raise RuntimeError(f"unfinished run: {folder}")
        run_panel_dir = resolved(config["quality_panel"])
        if run_panel_dir != panel_dir and str(run_panel_dir) not in accepted_prefixes:
            accepted_prefixes[str(run_panel_dir)] = verify_prefix_panel(
                run_panel_dir,panel_dir,binding,groups,expected)
        if not config["no_policy_selection_on_evaluation"]:
            raise RuntimeError("run did not freeze its policy before evaluation")
        if config["policy"] != summary["policy"]:
            raise RuntimeError("launch and final policy disagree")
        storage_mode = summary.get("storage_mode", "masked")
        if config.get("storage_mode", "masked") != storage_mode:
            raise RuntimeError("launch and final code storage mode disagree")
        source_policy_path = resolved(config["policy_source"]) if config["policy_source"] else None
        if source_policy_path is None:
            raise RuntimeError("aggregate requires a frozen publication policy source")
        publication = json.loads(source_policy_path.read_text())
        if publication["status"] != "complete" or publication["policy"] != summary["policy"]:
            raise RuntimeError("publication policy differs from serving run")
        if publication["weights"] != summary["weights"]:
            raise RuntimeError("publication and serving weight bindings differ")
        if publication.get("storage_mode", "masked") != storage_mode:
            raise RuntimeError("publication and serving code storage modes differ")
        if publication["shared_validation_flops"] != summary["shared_publication_validation_flops"]:
            raise RuntimeError("shared validation costs differ from frozen policy")
        if selected_set & set(publication["fit_uids"]+publication["validation_uids"]):
            raise RuntimeError("evaluation population overlaps publication calibration")
        if first is None:
            first, policy_path = summary, source_policy_path
            first_config = config
            for weight in summary["weights"].values():
                if sha256(resolved(weight["path"])) != weight["sha256"]:
                    raise RuntimeError("frozen adapter weights changed")
        else:
            if first.get("storage_mode", "masked") != storage_mode:
                raise RuntimeError("cannot merge different code storage modes")
            for key in ("policy","weights","controls","shared_publication_validation_flops"):
                if summary[key] != first[key]:
                    raise RuntimeError(f"cannot merge runs with different {key}")
            if source_policy_path != policy_path:
                raise RuntimeError("runs use different publication validation sources")
            for key in ("calibrations","calibration_panel","frozen_reference"):
                if config[key] != first_config[key]:
                    raise RuntimeError(f"launch bindings differ: {key}")
        run_selected = json.loads((folder/"users.json").read_text())
        run_prefix = groups["evaluation"][:config["users"]]
        selector = run_prefix[config.get("start_user",0):][config["shard_index"]::config["shards"]]
        if config["all_evaluation_uids"] != run_prefix or config["evaluation_uids"] != selector:
            raise RuntimeError("launch UID selector differs from frozen panel")
        if run_selected != selector or len(run_selected) != summary["users"]:
            raise RuntimeError("run UID membership/count differs from exact selector")
        if not set(run_selected).issubset(selected_set):
            raise RuntimeError("run contains users outside aggregate evaluation prefix")
        if len(set(run_selected)) != len(run_selected):
            raise RuntimeError("duplicate declared run UID")
        if [unit["uid"] for unit in summary["units"]] != run_selected:
            raise RuntimeError("unit ledger differs from selected run population")
        score_path = folder/"scores.parquet"
        table = pq.read_table(score_path).sort_by([("request_id","ascending")])
        expected_run = expected.filter(pa.compute.is_in(expected["uid"],value_set=pa.array(run_selected)))
        same_columns(table,expected_run,IDENTITY,"run score identities vs frozen panel")
        if table.num_rows != summary["requests"]:
            raise RuntimeError("summary request count differs from scores")
        for name in ("reuse_logit","full_logit","partial_logit","first_request_rebuild_logit"):
            if not np.isfinite(table[name].to_numpy()).all():
                raise RuntimeError(f"nonfinite matched {name}")
        for version in RELEASES:
            mask = table["version"].to_numpy() == version
            if not np.all(table["rho"].to_numpy()[mask] == summary["policy"][version]):
                raise RuntimeError("request rho differs from frozen release policy")
        reference_path = config["frozen_reference"]
        if reference_path:
            reference = pq.read_table(resolved(reference_path),filters=[("uid","in",run_selected)])
            reference = reference.sort_by([("request_id","ascending")])
            same_columns(table,reference,IDENTITY,"retained matched control identities")
            same_columns(table,reference,("reuse_logit","first_request_rebuild_logit"),
                         "retained matched control scores")
            if not np.array_equal(table["full_logit"].to_numpy(),reference["lightweight_logit"].to_numpy()):
                raise RuntimeError("full-code logits differ from retained reference")
        counters,_,_ = totals_from_units(summary["units"])
        if Counter(counters) != Counter(summary["actual_service_counters"]):
            raise RuntimeError("run service counters differ from per-user ledgers")
        for name,value in summary["checks"].items():
            combined_checks[name] = max(combined_checks[name],value)
        if summary["source_code_sha256"] != config["source_code_sha256"]:
            raise RuntimeError("launch and final source hash records differ")
        launch_sources[str(folder)] = summary["source_code_sha256"]
        source_runs.append(dict(summary=record(summary_path),configuration=record(config_path),
                                scores=record(score_path),users=record(folder/"users.json")))
        combined_users.extend(run_selected)
        units.extend(summary["units"])
        tables.append(table)
    if len(combined_users) != len(set(combined_users)) or set(combined_users) != selected_set:
        raise RuntimeError("run UID union must equal complete selected population, without duplicates")
    table = pa.concat_tables(tables).sort_by([("request_id","ascending")])
    same_columns(table,expected,IDENTITY,"aggregate score identities vs frozen panel")
    if len(set(table["request_id"].to_pylist())) != table.num_rows:
        raise RuntimeError("duplicate merged request")
    unit_map = {unit["uid"]:unit for unit in units}
    units = [unit_map[uid] for uid in selected]
    counters,by_release,memory = totals_from_units(units)
    quality = summarize(table,selected,repetitions=args.bootstrap,releases=RELEASES)
    cohorts = None
    if args.cohorts:
        cohort_path = resolved(args.cohorts)
        cohorts = summarize_cohorts(table,selected,json.loads(cohort_path.read_text()),
            repetitions=args.bootstrap,releases=RELEASES)
    args.output.mkdir(parents=True)
    pq.write_table(table,args.output/"scores.parquet",compression="zstd")
    write_json(args.output/"users.json",selected)
    aggregate_sources = {name:sha256(Path(__file__).parent/name) for name in ("aggregate.py","analyze.py")}
    config = dict(stage="aggregate_completed_continuous_runs",quality_panel=str(panel_dir),
        evaluation_uids=selected,all_evaluation_uids=selected,users=args.users,start_user=0,
        shard_index=0,shards=1,policy=first["policy"],policy_source=str(policy_path),
        storage_mode=first.get("storage_mode", "masked"),
        calibration_panel=first_config["calibration_panel"],calibrations=first_config["calibrations"],
        frozen_reference=first_config["frozen_reference"],no_policy_selection_on_evaluation=True,
        source_code_sha256=aggregate_sources,source_code_sha256_by_run=launch_sources,
        selector="Exact panel evaluation UID prefix; constituent selectors recomputed from prefix/start/shard settings.",
        cost_scope="analytical FLOPs for selected quality population; no latency or throughput claim")
    write_json(args.output/"configuration.json",config)
    result = dict(status="complete",role="chapter4_4_continuous_aggregate",users=len(selected),
        requests=table.num_rows,policy=first["policy"],storage_mode=first.get("storage_mode", "masked"),
        weights=first["weights"],quality=quality,
        units=units,checks=dict(combined_checks),actual_service_counters=counters,
        by_release_service=by_release,memory=memory,controls=first["controls"],
        shared_publication_validation_flops=first["shared_publication_validation_flops"],
        shared_cost_note="Charged once for the frozen policy, not once per replay shard. Complete fixed30k costs are reported separately.",
        cost_scope="executed quality-user shapes; fixed30k complete cost reported separately",
        sources=dict(runs=source_runs,panel=record(panel_dir/"binding.json"),
            publication_policy=record(policy_path),cohorts=record(resolved(args.cohorts)) if args.cohorts else None,
            accepted_bit_equivalent_prefix_panels=accepted_prefixes),
        scores=record(args.output/"scores.parquet"),source_code_sha256=aggregate_sources,
        source_code_sha256_by_run=launch_sources,
        verification=dict(exact_population_union_including_idle_users=True,
            exact_request_identity_label_timestamp_alignment=True,duplicate_or_missing_requests=0,
            same_frozen_policy_and_weight_bindings=True,per_user_counter_totals_match=True,
            launch_source_hashes_may_differ_and_are_retained=True))
    if cohorts is not None:
        result["cohorts"] = cohorts
        write_json(args.output/"cohorts_summary.json",cohorts)
    write_json(args.output/"summary.json",result)
    text = report(quality) + "\nTheoretical FLOPs use recorded per-user service counters. "
    text += "Fixed30k complete costs and shared preparation are reported separately. No loaded-serving timing or throughput was measured.\n"
    if cohorts is not None:
        text += "\n## Predefined user groups\n\n"
        for name,cohort in cohorts.items():
            text += f"### {name}\n\n"
            text += report(cohort).replace("# Partial-refresh quality evidence\n","") if "overall" in cohort else json.dumps(cohort)+"\n"
    (args.output/"report.md").write_text(text)
    print(json.dumps(dict(status="complete",output=str(args.output),users=len(selected),
        requests=table.num_rows,quality=quality["overall"]["paired"])),flush=True)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,nargs="+",required=True)
    parser.add_argument("--panel",type=Path,required=True)
    parser.add_argument("--users",type=int,required=True)
    parser.add_argument("--cohorts",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--bootstrap",type=int,default=1000)
    merge(parser.parse_args())
