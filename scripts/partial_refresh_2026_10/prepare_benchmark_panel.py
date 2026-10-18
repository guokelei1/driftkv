#!/usr/bin/env python3
"""Freeze a pre-existing alternative user group without selecting on outcomes.

Use retained benchmark requests/histories. Exclude current compact-code
development users and every calibration/reservation/canary/formal-confirmation
membership retained in the fresh-panel role inventory. This is an additional
population boundary check, not an all-method first-blind confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from design_one.prepare_history import restore_pack
from design_two.prepare import checked, record
from partial_refresh_2026_10.prepare_panel import write_cohorts
from unified_reuse_2026_09.common import write_json

BENCHMARK=ROOT/"results/design_two_2026_10/benchmark_panel"
CURRENT=ROOT/"results/design_system_2026_10/phase2/representative4096/panel"
ROLE_SOURCE=ROOT/"results/partial_refresh_2026_10/fresh1024_v2/panel/excluded_roles.json"
OUTPUT=ROOT/"results/partial_refresh_2026_10/benchmark1024/panel"
NAMESPACE="partial_refresh_benchmark_2026_10"


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    source=json.loads((args.source/"binding.json").read_text())
    original_groups=json.loads(checked(source["users_file"]).read_text())
    current_groups=json.loads((CURRENT/"users.json").read_text())
    current_binding=json.loads((CURRENT/"binding.json").read_text())
    if source["models"] != current_binding["models"] or source["model_config"] != current_binding["model_config"]:
        raise RuntimeError("alternative panel has different foundation model bindings")
    roles=json.loads(args.roles.read_text())
    included_roles={}
    for name,uids in roles.items():
        path,key=name.rsplit(":",1)
        if (any(token in key.lower() for token in ("fit","validation","reserve","calibration","canary"))
                or path.endswith("calibration_users.json") or "confirmation/" in path):
            included_roles[name]=uids
    reserved=set().union(*(set(uids) for uids in included_roles.values()))
    current=set(current_groups["evaluation"])
    pool=set(original_groups["evaluation"])
    eligible=pool-current-reserved
    if len(eligible)<args.users:
        raise ValueError(f"only {len(eligible)} eligible alternative benchmark users")
    key=lambda uid:(hashlib.sha256(f"{NAMESPACE}/{uid}".encode()).hexdigest(),uid)
    chosen=sorted(eligible,key=key)[:args.users]
    chosen_set=set(chosen)
    selection=dict(namespace=NAMESPACE,pool_users=len(pool),eligible_users=len(eligible),
        users=len(chosen),selected_uids=chosen,
        pool_reason="Pre-existing alternative population from retained benchmark; selected to evaluate population sensitivity after freezing the final compact-code refresh variant.",
        rule="First SHA256(namespace/uid), then UID, after current development and all recorded calibration/reservation/canary/confirmation exclusions. No label, score, activity or request availability filter.",
        source_population=source["users_file"],current_development_users=record(CURRENT/"users.json"),
        calibration_role_inventory=record(args.roles),
        calibration_and_confirmation_roles_used=list(included_roles),
        calibration_and_confirmation_overlap_before_exclusion=len(pool&reserved),
        current_development_overlap_before_exclusion=len(pool&current),
        selected_current_development_overlap=len(chosen_set&current),
        selected_overlap_by_excluded_role={name:len(chosen_set&set(uids))
                                          for name,uids in included_roles.items()},
        labels_or_quality_outcomes_used=False,retain_inactive_users=True,
        scope="New current compact-code/refresh evaluation users, but this older benchmark population participated in previous research and arose from a retrospective development pool. Not an all-method first-blind result and does not replace the negative pure-fresh result.")
    print(json.dumps(dict(status="alternative_users_frozen",users=len(chosen),eligible_users=len(eligible),
                         current_dev_overlap=len(pool&current),reserved_overlap=len(pool&reserved))),flush=True)
    requests=pq.read_table(checked(source["files"]["requests"]))
    requests=requests.filter(pc.is_in(requests["uid"],value_set=pa.array(chosen,type=pa.uint64())))
    requests=requests.sort_by([("uid","ascending"),("query_timestamp","ascending"),("request_id","ascending")])
    if len(set(requests["request_id"].to_pylist()))!=len(requests):
        raise RuntimeError("retained request identities repeat")
    history=restore_pack(checked(source["files"]["history"]))
    ordered=sorted(chosen)
    lengths=np.asarray([len(history.rows[uid][0]) for uid in ordered],dtype=np.int64)
    args.output.mkdir(parents=True)
    groups=dict(evaluation=chosen,pilot=chosen[:8],fit=original_groups["fit"],validation=original_groups["validation"])
    write_json(args.output/"users.json",groups)
    write_json(args.output/"selection.json",selection)
    pq.write_table(requests,args.output/"requests.parquet",compression="zstd")
    np.savez(args.output/"histories.npz",uids=np.asarray(ordered,dtype=np.int64),
        offsets=np.r_[np.int64(0),lengths.cumsum()],
        timestamps=np.concatenate([history.rows[uid][0] for uid in ordered]),
        item_ids=np.concatenate([history.rows[uid][1] for uid in ordered]),
        behaviors=np.concatenate([history.rows[uid][2] for uid in ordered]))
    restored=restore_pack(args.output/"histories.npz")
    for uid in ordered:
        if any(not np.array_equal(a,b) for a,b in zip(history.rows[uid],restored.rows[uid],strict=True)):
            raise RuntimeError("retained source history changed on subset/roundtrip")
    new_windows=[]
    for window in source["windows"]:
        subset=requests.filter(pc.equal(requests["version"],window["current"]))
        new_windows.append({**{k:v for k,v in window.items() if k!="groups"},
            "groups":dict(evaluation=dict(users=len(chosen),requests=len(subset),
                                          active_users=len(set(subset["uid"].to_pylist()))))})
    keys=("scale","model_chain","edges","initial_version","cutover","end_timestamp",
          "days","context","known_vocab_size","oov_buckets","model_config","versions","models",
          "state_rule","request_rule")
    binding={k:source[k] for k in keys}
    binding.update(status="complete",windows=new_windows,users_file=record(args.output/"users.json"),
        selection=selection,selection_file=record(args.output/"selection.json"),
        configuration=record(Path(__file__)),
        files={key:record(args.output/name) for key,name in (("requests","requests.parquet"),("history","histories.npz"))},
        sources=dict(retained_benchmark_binding=record(args.source/"binding.json"),
            retained_requests=source["files"]["requests"],retained_history=source["files"]["history"],
            dataset=source["sources"]["dataset"],labels=source["sources"]["labels"],
            calibration_role_inventory=record(args.roles)),
        checks=dict(all_selected_user_source_history_tuples_exact=True,
            fit_validation_evaluation_disjoint=not bool(chosen_set&set(groups["fit"]+groups["validation"])),
            current_development_disjoint=True,all_recorded_calibration_confirmation_roles_disjoint=True),
        history_events=int(lengths.sum()))
    write_json(args.output/"binding.json",binding)
    write_cohorts(args.output)
    (args.output/"README.md").write_text(
        f"# Alternative retained benchmark panel\n\n{len(chosen)} users, {len(requests)} requests, "
        "V2→V3→V4→V5. Users are hash-selected from the existing benchmark after excluding current "
        "compact-code development and all recorded calibration/reservation/canary/confirmation roles. "
        "No future labels, activity or quality outcomes select the subset.\n\n"
        "Requests and causal mapped history tuples are copied exactly from retained benchmark inputs. "
        "Shared weights, original inputs and previous scores are unchanged. The source population "
        "participated in older research; this is a population boundary check, not an all-method "
        "first-blind confirmation. The pure-fresh negative result remains separate.\n")
    print(json.dumps(dict(status="complete",output=str(args.output),users=len(chosen),requests=len(requests),
                         windows=new_windows,history_events=int(lengths.sum()))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=BENCHMARK)
    parser.add_argument("--roles",type=Path,default=ROLE_SOURCE)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--users",type=int,default=1024)
    main(parser.parse_args())
