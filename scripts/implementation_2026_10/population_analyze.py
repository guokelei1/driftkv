#!/usr/bin/env python3
"""Merge engineering replay shards with one complete-population UID bootstrap."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from design_two.prepare import load_panel
from implementation_2026_10.evaluate_population import (numeric_summary,paired_quality,
    reference_inputs,VERSIONS)
from query_relay_2026_10.evaluate import records
from unified_reuse_2026_09.common import sha256,write_json


def ledger_counter(rows,names):
    return Counter({tuple(row[name] for name in names):int(row["count"]) for row in rows})


def aggregate(inputs,*,panel,repetitions=1000,seed=17):
    if repetitions<1:
        raise ValueError("bootstrap repetitions must be positive")
    _,groups,_=load_panel(panel)
    population=list(groups["evaluation"])
    configurations,summaries,tables,bindings=[],[],[],[]
    seen=set()
    common=("panel_sha256","population","versions","workload","reference_sha256",
        "reference_summary_sha256","freeze_sha256","weights","generated_publications",
        "generated_fixed_phi","implementation","query_batch","source_sha256")
    for directory in inputs:
        settings=json.loads((directory/"configuration.json").read_text())
        summary=json.loads((directory/"summary.json").read_text())
        if summary["status"]!="complete" or settings["population"]!=population \
                or settings["panel_sha256"]!=sha256(panel/"binding.json"):
            raise ValueError("shard is incomplete or belongs to another frozen population")
        if configurations and any(settings[key]!=configurations[0][key] for key in common):
            raise ValueError("shards have different workloads, sources, fixed weights or generated-prefix bindings")
        selected=settings["selected_uids"]
        if len(set(selected))!=len(selected) or seen.intersection(selected):
            raise ValueError("UID execution shards overlap")
        if summary["selected_uids"]!=selected or len(summary["units"])!=len(selected) \
                or {unit["uid"] for unit in summary["units"]}!=set(selected):
            raise ValueError("shard per-user units do not retain every selected UID")
        if not all(summary["checks"].values()):
            raise ValueError("a shard did not pass its identity/service checks")
        table=pq.read_table(directory/"scores.parquet")
        if set(table["uid"].to_pylist())-set(selected):
            raise ValueError("scores contain an unselected UID")
        observed=Counter((row["uid"],row["version"]) for row in table.select(["uid","version"]).to_pylist())
        expected={(unit["uid"],version):unit["by_release"][version]["requests"]
                  for unit in summary["units"] for version in VERSIONS}
        if any(observed[key]!=count for key,count in expected.items()) or set(observed)-set(expected):
            raise ValueError("candidate counts disagree with retained UID/release units")
        numeric_summary(table)
        seen.update(selected);tables.append(table);configurations.append(settings);summaries.append(summary)
        bindings.append(dict(directory=str(directory.resolve()),
            configuration_sha256=sha256(directory/"configuration.json"),
            summary_sha256=sha256(directory/"summary.json"),
            scores_sha256=sha256(directory/"scores.parquet")))
    if seen!=set(population):
        raise ValueError("merged shards do not cover the complete frozen population, including idle users")
    table=pa.concat_tables(tables).sort_by([("request_id","ascending")])
    if pc.count_distinct(table["request_id"]).as_py()!=table.num_rows:
        raise ValueError("merged candidate request IDs overlap")
    first=configurations[0]
    reference=Path(first["reference"])
    if sha256(reference)!=first["reference_sha256"]:
        raise ValueError("retained reference changed after replay")
    _,old_summary,_=reference_inputs(reference,panel_hash=first["panel_sha256"],
        population=population,kind=first["workload"]["kind"])
    # These controls and identities were copied from retained rows by the runner;
    # verify the whole merged request set and every payload without rescoring.
    old=pq.read_table(reference).sort_by([("request_id","ascending")])
    if old.num_rows!=table.num_rows:
        raise ValueError("merged candidates do not cover retained reference requests")
    for name in ("request_id","uid","version","label","item_idx","query_timestamp",
                 "reuse_logit","full_logit","first_request_rebuild_logit","history_length","coded_rows"):
        if not old[name].equals(table[name]):
            raise ValueError(f"merged reference payload differs: {name}")
    if not old["partial_logit"].equals(table["old_relay_logit"]):
        raise ValueError("retained original relay scores were altered")
    counters,native,code=Counter(),Counter(),Counter()
    native_names=("action","version","before","width")
    code_names=("kind","version","old_length","queries","coded_rows","native_rows")
    for summary in summaries:
        counters.update(summary["actual_service_counters"])
        native.update(ledger_counter(summary["native_ledger"],native_names))
        code.update(ledger_counter(summary["code_read_ledger"],code_names))
    old_method_counters=Counter(old_summary["actual_service_counters"])
    reference_control_observers={name:old_method_counters.pop(name) for name in (
        "native_control_candidate_scores_computed","native_control_candidate_scores_reused")
        if name in old_method_counters}
    if counters!=old_method_counters \
            or native!=ledger_counter(old_summary["native_ledger"],native_names) \
            or code!=ledger_counter(old_summary["code_read_ledger"],code_names):
        raise ValueError("merged literal service operations differ from the original complete replay")
    print(json.dumps(dict(status="inputs_validated",users=len(population),requests=table.num_rows,
                          shards=len(inputs))),flush=True)
    quality,paired=paired_quality(table,population,repetitions=repetitions,seed=seed)
    inference_names={f"scripts/implementation_2026_10/{name}.py" for name in (
        "evaluate_population","backend","prepare","reader","reader_kernel","state")}
    inference_names.update(f"scripts/query_relay_2026_10/{name}.py" for name in (
        "adapter","session","workload","precision"))
    inference_sources={name:first["source_sha256"][name] for name in sorted(inference_names)}
    settings={key:first[key] for key in common}
    settings.update(panel=str(panel.resolve()),selected_uids=population,population=population,
        reference=str(reference),inputs=bindings,source_sha256_aggregate=sha256(Path(__file__)),
        source_sha256=inference_sources,source_inventory_at_replay=first["source_sha256"],
        source_inventory_scope="Actual captured inference sources are separated from auxiliary scripts inventoried before execution; the current analysis has its own SHA",
        bootstrap=dict(repetitions=repetitions,seed=seed),
        scope="Revealed frozen users re-evaluated after engineering changes; not new independent confirmation",
        aggregation="Recompute from complete matched scores, never average shard AUC or recovery")
    summary=dict(status="complete",selected_uids=population,quality=quality,
        optimized_minus_original_auc=paired,numeric=numeric_summary(table),weights=first["weights"],
        units=[unit for source in summaries for unit in source["units"]],
        actual_service_counters=dict(counters),native_ledger=records(native,native_names),
        code_read_ledger=records(code,code_names),generated_publications=first["generated_publications"],
        reference_control_scoring_observer_counters=reference_control_observers,
        native_control_reuse_scope="Only prior control-scoring observer counters are excluded from method-operation equality; retained control logits remain exact",
        inputs=bindings,source_sha256=inference_sources,
        source_sha256_aggregate=settings["source_sha256_aggregate"],scope=settings["scope"],
        checks=dict(disjoint_uid_shards=True,complete_frozen_population_including_idle=True,
            exact_reference_request_set_and_native_controls=True,unaltered_retained_relay_logits=True,
            matched_literal_service_counters_and_ledgers=True,corrected_writes=True,
            no_parameter_or_workload_selection=True))
    return table,settings,summary


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    began=time.perf_counter()
    table,settings,summary=aggregate(args.inputs,panel=args.panel,repetitions=args.bootstrap,seed=args.seed)
    args.output.mkdir(parents=True)
    pq.write_table(table,args.output/"scores.parquet",compression="zstd")
    write_json(args.output/"configuration.json",settings)
    write_json(args.output/"summary.json",summary)
    write_json(args.output/"units.json",summary["units"])
    print(json.dumps(dict(status="complete",users=len(summary["selected_uids"]),requests=table.num_rows,
        numeric=summary["numeric"],paired=summary["optimized_minus_original_auc"]["overall"],
        elapsed_seconds=time.perf_counter()-began)),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",nargs="+",type=Path,required=True)
    p.add_argument("--panel",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=1000)
    p.add_argument("--seed",type=int,default=17)
    return p


if __name__=="__main__":
    run(parser().parse_args())
