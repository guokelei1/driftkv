#!/usr/bin/env python3
"""Exact lazy-code arithmetic over the unchanged real30k event trace.

Read request identities/timestamps and native event histories, never labels or
scores. Check every eager native shape against the retained population ledger
and lazy code counts against the completed eight-user GPU probe.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from design_system.profile_costs import ShapeCosts, profile_user, START
from design_system.evaluate_lightweight_fullitem import candidate_release as eager_candidate_release
from design_one.prepare_history import load_histories_semijoin
from design_two.common import RELEASES, END
from hstu_kvcache.design_one.lightweight_fullitem import LightweightFullItemAdapter
from unified_reuse_2026_09.common import sha256, write_json

REFERENCE = ROOT / "results/design_system_2026_10/phase2/population_cost/summary_complete.json"
PROBE = ROOT / "results/design_system_2026_10/lightweight_read/lazy_codes/fullitem8_probe/summary.json"
OUTPUT = ROOT / "results/design_system_2026_10/lightweight_read/lazy_codes/population_cost"


def lazy_candidate_release(counts, *, layers, heads, head_dim, rank, response_width):
    """Keep native denominators; charge only actually materialized code rows."""
    row = eager_candidate_release(counts, layers=layers, heads=heads, head_dim=head_dim,
        rank=rank, response_width=response_width)
    per_token = row["encoder"]["flops_per_token"]
    queries = counts["lazy_corrected_requests"]
    pairs = counts["lazy_corrected_token_query_pairs"]
    if response_width:
        raise ValueError("the lazy full-item candidate has no Response")
    row["components_flops"] = dict(
        first_read_initialization=counts["lazy_initial_encode_tokens"]*per_token,
        post_initialization_append_encoding=counts["lazy_append_encode_tokens"]*per_token,
        query_added_weighted_value=layers*2*heads*rank*pairs,
        query_low_rank_decode=layers*queries*2*heads*rank*head_dim,
        query_value_residual_add=layers*queries*heads*head_dim,
        query_response=0)
    row["encoded_initial_tokens"] = counts["lazy_initial_encode_tokens"]
    row["encoded_append_tokens"] = counts["lazy_append_encode_tokens"]
    row["corrected_requests"] = queries
    row["corrected_token_query_pairs"] = pairs
    row["variable_flops"] = sum(row["components_flops"].values())
    row["extra_query_flops"] = sum(v for k,v in row["components_flops"].items() if k.startswith("query_"))
    row["variable_percent_of_once_rebuild"] = 100*row["variable_flops"]/row["first_request_rebuild_flops"]
    row["extra_percent_of_native_whole_query"] = 100*row["extra_query_flops"]/row["native_whole_query_flops"]
    row["extra_percent_of_native_history_attention"] = 100*row["extra_query_flops"]/row["native_history_attention_flops"]
    row["shared_work_budget_at_20_percent_flops"] = .2*row["first_request_rebuild_flops"]-row["variable_flops"]
    row["code_policy"] = "Initialize at first actual read; native writes/support always advance; append codes only after initialization and until causal retirement."
    return row


def lazy_shapes(timestamps, requests, *, layers=6, cap=1024):
    """Count rows at the same native append-band boundaries as LifecycleSession."""
    cursor = int(np.searchsorted(timestamps, START, side="left"))
    length = min(cursor, cap)
    by_version = defaultdict(Counter)
    for row in requests:
        by_version[row["version"]][int(row["query_timestamp"])] += 1
    releases = {}
    for index, (version, _) in enumerate(RELEASES):
        until = RELEASES[index+1][1] if index+1 < len(RELEASES) else END
        pending, active = length > 0, False
        remaining = layers*(cap-1)+1 if pending else 0
        counts = Counter()

        def advance(at):
            nonlocal cursor, length, pending, active, remaining
            stop = int(np.searchsorted(timestamps, at, side="left"))
            while cursor < stop:
                width = 1 << (min(32, stop-cursor).bit_length()-1)
                cursor += width
                length = min(length+width, cap)
                counts["lazy_support_token_layers"] += width*layers
                if pending or active:
                    remaining -= width
                    if remaining <= 0:
                        counts["lazy_causal_retirements"] += 1
                        if pending:
                            counts["lazy_unmaterialized_retirements"] += 1
                        pending = active = False
                    elif active:
                        counts["lazy_append_encode_tokens"] += width

        for timestamp, count in sorted(by_version[version].items()):
            advance(timestamp)
            if pending:
                counts["lazy_initial_encode_tokens"] += length
                counts["lazy_initializations"] += 1
                active, pending = True, False
            if active:
                counts["lazy_corrected_requests"] += count
                counts["lazy_corrected_token_query_pairs"] += count*length
            else:
                counts["lazy_native_requests"] += count
        advance(until)
        counts["lazy_pending_at_release_end"] = int(pending)
        releases[version] = dict(counts)
    return releases


def run(args):
    began = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    reference = json.loads(REFERENCE.read_text())
    config = json.loads(args.configuration.read_text())
    panel = json.loads((args.panel / "binding.json").read_text())
    probe = json.loads(PROBE.read_text())
    evidence = {str(REFERENCE):sha256(REFERENCE), str(PROBE):sha256(PROBE),
        str(args.configuration):sha256(args.configuration),
        str(args.panel / "binding.json"):sha256(args.panel / "binding.json")}

    def bound(source):
        path = Path(source["path"])
        path = path if path.is_absolute() else ROOT / path
        digest = sha256(path)
        if digest != source["sha256"] or reference["source_sha256"].get(str(path.resolve())) != digest:
            raise RuntimeError(f"population input no longer matches retained evidence: {path}")
        evidence[str(path.resolve())] = digest
        return path

    population = bound(config["data"]["population"])
    uids = sorted(pq.read_table(population, columns=["uid"])["uid"].to_pylist())
    if len(uids) != 30000 or len(set(uids)) != 30000:
        raise RuntimeError("fixed30k population changed")
    selected = set(uids)
    requests, seen = defaultdict(list), set()
    for window in config["profile"]["continuous_windows"]:
        if window["target"] not in dict(RELEASES):
            continue
        for source in window["request_sources"]:
            for row in pq.read_table(bound(source), columns=["request_id", "uid", "query_timestamp"]).to_pylist():
                if row["uid"] not in selected or not window["days_half_open"][0]*86400 <= row["query_timestamp"] < window["days_half_open"][1]*86400:
                    continue
                if row["request_id"] in seen:
                    raise RuntimeError("duplicated population request")
                seen.add(row["request_id"])
                requests[row["uid"]].append(dict(row, version=window["target"]))
    if len(seen) != reference["requests"]:
        raise RuntimeError("request population changed")
    dataset_path = bound(config["data"]["dataset"])
    dataset = json.loads(dataset_path.read_text())
    histories = load_histories_semijoin(uids, dataset_path=dataset_path,
        known_vocab_size=dataset["foundation_items"],
        oov_buckets=panel["model_config"]["num_items"]-dataset["foundation_items"],
        start_timestamp=START, end_timestamp=END, max_history=1024, threads=args.threads)
    print(json.dumps(dict(status="lazy_profile_metadata_loaded", users=len(uids), requests=len(seen),
        elapsed_seconds=time.perf_counter()-began)), flush=True)
    rows_path = Path(reference["rows"]["path"])
    if sha256(rows_path) != reference["rows"]["sha256"]:
        raise RuntimeError("original eager shape rows changed")
    evidence[str(rows_path)] = sha256(rows_path)
    original = {(r["uid"],r["version"]):r for r in pq.read_table(rows_path).to_pylist()}
    adapters = {version:LightweightFullItemAdapter(6,6,32,rank=8) for version,_ in RELEASES}
    costs = ShapeCosts(panel["model_config"], adapters)
    compare_keys = ("requests", "initial_history_length", "appends", "native_append_flops",
        "exact_rebuild_flops", "first_request_rebuild_flops", "fresh_per_request_history_flops",
        "native_query_flops", "support_token_layers", "final_history_length")
    probe_units = {unit["uid"]:unit for unit in probe["units"]}
    totals = {version:Counter() for version,_ in RELEASES}
    output_rows, matched_gpu_releases = [], 0
    for uid in uids:
        timestamps = histories.rows.get(uid, (np.array([], dtype=np.int64),))[0]
        eager = profile_user(uid, timestamps, requests[uid], costs)
        lazy = lazy_shapes(timestamps, requests[uid])
        for version, shaped in eager["releases"].items():
            old = original[uid,version]
            for key in compare_keys:
                if shaped.get(key,0) != old.get(key,0):
                    raise RuntimeError(f"eager shape mismatch {uid}/{version}/{key}")
            row = {**old, **lazy[version]}
            totals[version].update({key:value for key,value in row.items() if key not in ("uid","version")})
            output_rows.append(row)
            if uid in probe_units:
                actual = probe_units[uid]["releases"][version]["lazy_costs"]
                one = adapters[version].estimate_flops()["token_transform"]
                comparison = {
                    "initial_encode_flops":lazy[version].get("lazy_initial_encode_tokens",0)*one,
                    "append_encode_flops":lazy[version].get("lazy_append_encode_tokens",0)*one,
                    "support_token_layers":lazy[version].get("lazy_support_token_layers",0),
                    "corrected_requests":lazy[version].get("lazy_corrected_requests",0),
                    "native_requests":lazy[version].get("lazy_native_requests",0),
                    "first_read_code_initializations":lazy[version].get("lazy_initializations",0),
                    "extra_history_read_flops":6*(2*6*8*lazy[version].get("lazy_corrected_token_query_pairs",0)+
                        (2*6*8*32+192)*lazy[version].get("lazy_corrected_requests",0))}
                for key,value in comparison.items():
                    if value != actual.get(key,0):
                        raise RuntimeError(f"executed lazy probe cost mismatch {uid}/{version}/{key}")
                matched_gpu_releases += 1
    for version,total in totals.items():
        for key in compare_keys:
            if total[key] != reference["by_release"][version]["counts_and_flops"].get(key,0):
                raise RuntimeError(f"eager population total differs: {version}/{key}")
    if matched_gpu_releases != 24:
        raise RuntimeError("did not verify all eight-user three-release executed ledgers")
    args.output.mkdir(parents=True)
    columns = sorted({key for row in output_rows for key in row})
    table = pa.table({key:[row.get(key,0) for row in output_rows] for key in columns})
    output_rows_path = args.output / "user_release_costs.parquet"
    pq.write_table(table, output_rows_path)
    releases = {}
    for version,total in totals.items():
        settings = dict(layers=6,heads=6,head_dim=32,rank=8,response_width=0)
        releases[version] = dict(counts_and_flops=dict(total),
            eager_value_cost=eager_candidate_release(total,**settings),
            lazy_value_cost=lazy_candidate_release(total,**settings))
    source_names = ("scripts/design_system/profile_lightweight_lazy.py", "scripts/design_system/profile_costs.py",
        "scripts/design_system/lightweight_lazy.py", "scripts/design_system/evaluate_lightweight_fullitem.py",
        "scripts/design_system/lightweight_budget.py", "src/hstu_kvcache/design_three/lifecycle.py",
        "src/hstu_kvcache/design_two/replay.py", "scripts/read_correction_2026_09/cost.py")
    summary = dict(status="complete",role="Exact analytical lazy-code cost on retained real30k event shapes; not full-population GPU timing or quality",
        users=30000,requests=len(seen),active_users=len(requests),days=[259,301],model_chain=["v2","v3","v4","v5"],
        context=1024,query_batch=32,append_band_size=32,by_release=releases,
        population_dependent_flops=sum(x["lazy_value_cost"]["variable_flops"] for x in releases.values()),
        eager_population_dependent_flops=sum(x["eager_value_cost"]["variable_flops"] for x in releases.values()),
        first_request_rebuild_flops=sum(x["counts_and_flops"]["first_request_rebuild_flops"] for x in releases.values()),
        verification=dict(eager_native_shape_user_releases=90000,all_eager_shape_components_exact=True,
            executed_gpu_probe_user_releases=matched_gpu_releases,all_probe_lazy_cost_components_exact=True),
        rows=dict(path=str(output_rows_path.resolve()),sha256=sha256(output_rows_path)),source_sha256=evidence,
        source_code_sha256={name:sha256(ROOT/name) for name in source_names},
        elapsed_seconds=time.perf_counter()-began,
        policy="Pending code binding at release; materialize current retained native rows only at first actual request. "
            "Native appends/support always advance; encode appends only while materialized; retire at the first native band "
            "with no inherited support. No forecast or feedback labels. Shared calibration costs are supplied separately.")
    write_json(args.output / "summary.json", summary)
    print(json.dumps({key:summary[key] for key in ("status","users","requests","population_dependent_flops",
        "eager_population_dependent_flops","first_request_rebuild_flops","verification","elapsed_seconds")}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration",type=Path,default=ROOT / "configs/design/medium_v0_v5_development.json")
    parser.add_argument("--panel",type=Path,default=ROOT / "results/design_two_2026_10/benchmark_panel")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--threads",type=int,default=8)
    run(parser.parse_args())
