#!/usr/bin/env python3
"""Score only the inherited one-hop cache, one resumable user shard at a time."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "scripts"), str(ROOT / "src")]

from common import DEFAULT_OUTPUT, DAY, PLAN, binding, sha256, write_json
from evaluate_yambda500m_foundation_raw import evaluate_full_cache_cohort, load_histories, load_model
from evaluate_yambda500m_hstu_native_onehop_reuse_raw import evaluate_fallback_user
from hstu_kvcache.evaluation import append_timestamp_group, materialize_state, observe_rolling, timestamp_groups


KEEP = ("request_id", "uid", "query_timestamp", "hstu_logit", "append_count_since_cutover",
        "history_length", "cache_length", "rolling_evictions")


@torch.inference_mode()
def fallback_reuse(requests, history, parent, current, cutover, max_length):
    """Existing scalar one-hop semantics without computing either Full path."""
    uid = int(requests[0]["uid"])
    timestamps, items, behaviors = history.rows[uid]
    events = [(int(t), int(i), int(b)) for t, i, b in zip(timestamps, items, behaviors, strict=True)]
    prefix = [event for event in events if event[0] < cutover]
    if not prefix:
        raise RuntimeError("evaluation user has no strictly prior history")
    state = materialize_state(parent, prefix, producer_version="parent", max_length=max_length)
    post = list(timestamp_groups(event for event in events if event[0] >= cutover))
    by_time: dict[int, list[dict]] = defaultdict(list)
    for request in requests:
        by_time[int(request["query_timestamp"])].append(request)
    position = append_count = evictions = 0
    output = []
    for query_time, simultaneous in sorted(by_time.items()):
        while position < len(post) and post[position][0] < query_time:
            _, group = post[position]
            evictions += max(0, state.cache.seq_len + len(group) - max_length)
            state = append_timestamp_group(current, state, group, producer_version="current", max_length=max_length)
            append_count += len(group); position += 1
        stop = int(np.searchsorted(timestamps, query_time, side="left"))
        for request in simultaneous:
            score, _ = observe_rolling(current, state, candidate_id=int(request["item_idx"]),
                                       query_timestamp=query_time)
            output.append({"request_id": request["request_id"], "uid": uid,
                           "query_timestamp": query_time, "hstu_logit": float(score),
                           "append_count_since_cutover": append_count,
                           "history_length": min(stop, max_length),
                           "cache_length": state.cache.seq_len, "rolling_evictions": evictions})
        while position < len(post) and post[position][0] == query_time:
            _, group = post[position]
            evictions += max(0, state.cache.seq_len + len(group) - max_length)
            state = append_timestamp_group(current, state, group, producer_version="current", max_length=max_length)
            append_count += len(group); position += 1
    return output


def score_unit(uids, by_user, history, parent, current, b, max_length):
    full, fallback = [], []
    for uid in uids:
        timestamps = history.rows[uid][0]
        prefix = int(np.searchsorted(timestamps, b["cutover"], side="left"))
        (full if prefix >= max_length else fallback).append(uid)
    output = []
    for start in range(0, len(full), b["cohort_size"]):
        cohort = full[start:start + b["cohort_size"]]
        rows = evaluate_full_cache_cohort(
            uids=cohort, by_user=by_user, history=history, parent=parent, current=current,
            parent_name="parent", current_name="current", edge=b["edge"],
            checkpoint_hash="", parent_hash="", manifest_hash="", cutover=b["cutover"],
            lineage_models=[("parent", parent)], event_end_exclusive=b["days"][1] * DAY,
            include_request_local=False, include_parent_exact=False, reuse_only=True,
            query_chunk_size=b["query_chunk_size"], max_length=max_length,
        )
        output.extend({key: row[key] for key in KEEP} for row in rows)
    for uid in fallback:
        output.extend(fallback_reuse(by_user[uid], history, parent, current, b["cutover"], max_length))
    return output, len(full), len(fallback)


def check_reference(uid, by_user, history, parent, current, b, max_length, actual):
    """Check one user against the original scalar evaluator without a dual batched cache."""
    reference = evaluate_fallback_user(
        requests=by_user[uid], history=history, parent=parent, current=current,
        edge=b["edge"], parent_name="parent", current_name="current", cutover=b["cutover"],
        current_hash="", parent_hash="", manifest_hash="", include_parent_exact=False,
        max_length=max_length,
    )
    expected = {row["request_id"]: float(row["hstu_logit"]) for row in reference
                if row["path"] == "one_hop_reuse_rolling"}
    observed = {row["request_id"]: float(row["hstu_logit"]) for row in actual if row["uid"] == uid}
    if expected.keys() != observed.keys():
        raise RuntimeError("reuse-only reference request identities differ")
    largest = max((abs(observed[key] - value) for key, value in expected.items()), default=0.0)
    if largest > 1e-5:
        raise RuntimeError(f"reuse-only score differs from original paired evaluator: {largest}")
    return {"uid": uid, "requests": len(expected), "max_abs_logit_difference": largest}


def run(args):
    b = binding(args.scale, args.edge)
    if args.cohort_size:
        b["cohort_size"] = args.cohort_size
    prepared = args.prepared_root / args.scale / b["edge"]
    record = json.loads((prepared / "binding.json").read_text())
    if record["plan_sha256"] != sha256(PLAN):
        raise RuntimeError("prepared request panel does not match the current plan")
    rank_record = record["ranks"][args.rank]
    request_path = prepared / f"requests_rank{args.rank}.parquet"
    if sha256(request_path) != rank_record["sha256"]:
        raise RuntimeError("prepared request shard changed")
    source_files = [
        Path(__file__), Path(__file__).with_name("common.py"),
        ROOT / "scripts/evaluate_yambda500m_foundation_raw.py",
        ROOT / "scripts/evaluate_yambda500m_hstu_native_onehop_reuse_raw.py",
        ROOT / "src/hstu_kvcache/evaluation/cache_lineage.py",
        ROOT / "src/hstu_kvcache/models/state_transition.py",
        ROOT / "src/hstu_kvcache/models/hstu.py",
        ROOT / "src/hstu_kvcache/models/attention.py",
        ROOT / "src/hstu_kvcache/models/triton_attention.py",
    ]
    execution_source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_files}
    source_signature = hashlib.sha256(json.dumps({
        "plan_sha256": record["plan_sha256"], "edge": record["edge"], "days": record["days"],
        "rank_request_sha256": rank_record["sha256"],
        "execution_source_hashes": execution_source_hashes,
        "sources": {key: record["sources"][key]["sha256"] for key in
                    ("parent", "current", "dataset", "full_raw", "labels")},
    }, sort_keys=True).encode()).hexdigest()
    requests = pq.read_table(request_path).to_pylist()
    by_user: dict[int, list[dict]] = defaultdict(list)
    for request in requests:
        by_user[int(request["uid"])].append(request)
    all_uids = sorted(by_user, key=lambda uid: (-len(by_user[uid]), uid))
    eligible = all_uids[args.skip_top_users:]
    selected = eligible[:args.limit_users] if args.limit_users else eligible
    if not selected:
        raise RuntimeError("rank has no evaluation users")
    output = args.work_root / args.scale / b["edge"] / f"rank{args.rank}"
    finished = output / "complete.json"
    if finished.exists() and "execution_source_hashes" in json.loads(finished.read_text()):
        previous = json.loads(finished.read_text())
        if (previous["source_signature"] != source_signature
                or previous["users"] != len(selected)
                or previous["requests"] != sum(len(by_user[uid]) for uid in selected)):
            raise RuntimeError("completed rank belongs to a different input or user selection")
        print(json.dumps(previous, ensure_ascii=False), flush=True)
        return
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    torch.set_num_threads(args.torch_threads)
    pa.set_cpu_count(args.history_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    free, total = torch.cuda.mem_get_info(device)
    if free < 0.75 * total:
        raise RuntimeError(f"GPU {args.gpu} has less than 75% free memory before the job")
    torch.cuda.set_per_process_memory_fraction(0.70, device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    parent, parent_payload = load_model(b["parent"], device)
    current, current_payload = load_model(b["current"], device)
    model_load_seconds = time.perf_counter() - start
    if parent_payload["config"] != current_payload["config"]:
        raise RuntimeError("adjacent checkpoints use different HSTU configurations")
    max_length = int(current_payload["config"]["max_seq_len"])
    if max_length != 1024:
        raise RuntimeError("planned 1024-event cache differs from checkpoint")
    dataset = json.loads(b["dataset"].read_text())
    known = int(current_payload.get("known_vocab_size", dataset["foundation_items"]))
    oov = int(current_payload["config"]["num_items"]) - known
    history = load_histories(selected, oov_buckets=oov, dataset_path=b["dataset"],
                             known_vocab_size=known, start_timestamp=b["cutover"],
                             end_timestamp=b["days"][1] * DAY, max_history=max_length,
                             threads=args.history_threads)
    history_load_seconds = time.perf_counter() - start - model_load_seconds
    output.mkdir(parents=True, exist_ok=True)
    completed_requests = 0
    reference = None
    batch_reductions = []
    for unit, offset in enumerate(range(0, len(selected), args.unit_users)):
        uids = selected[offset:offset + args.unit_users]
        path = output / f"shard_{unit:05d}.parquet"
        seal = output / f"shard_{unit:05d}.seal.json"
        if seal.exists():
            previous = json.loads(seal.read_text())
            if (previous["sha256"] != sha256(path) or previous["uids"] != uids
                    or previous.get("source_signature") != source_signature):
                raise RuntimeError(f"completed shard changed: {path}")
            completed_requests += previous["requests"]
            continue
        unit_start = time.perf_counter()
        while True:
            try:
                rows, full_users, fallback_users = score_unit(uids, by_user, history, parent, current, b, max_length)
                break
            except torch.cuda.OutOfMemoryError:
                if b["cohort_size"] == 1:
                    raise
                previous = b["cohort_size"]
                b["cohort_size"] = max(1, previous // 2)
                batch_reductions.append({"unit": unit, "from": previous, "to": b["cohort_size"]})
                torch.cuda.empty_cache()
        expected = sum(len(by_user[uid]) for uid in uids)
        if len(rows) != expected or len({row["request_id"] for row in rows}) != expected:
            raise RuntimeError("reuse output lacks or duplicates requests")
        if args.verify and reference is None:
            torch.cuda.empty_cache()
            reference = check_reference(uids[0], by_user, history, parent, current, b, max_length, rows)
            write_json(output / "reference.json", reference)
        table = pa.Table.from_pylist(rows)
        partial = path.with_suffix(".parquet.partial")
        pq.write_table(table, partial, compression="zstd")
        os.replace(partial, path)
        write_json(seal, {"sha256": sha256(path), "source_signature": source_signature,
                          "uids": uids, "requests": len(rows),
                          "full_users": full_users, "fallback_users": fallback_users,
                          "cohort_size": b["cohort_size"], "unit_seconds": time.perf_counter()-unit_start})
        completed_requests += len(rows)
        write_json(output / "progress.json", {"rank": args.rank, "completed_users": offset + len(uids),
                   "total_users": len(selected), "completed_requests": completed_requests,
                   "total_requests": sum(len(by_user[uid]) for uid in selected),
                   "elapsed_seconds": time.perf_counter()-start,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated(device)/(1 << 30),
                   "peak_reserved_gib": torch.cuda.max_memory_reserved(device)/(1 << 30),
                   "cohort_size": b["cohort_size"], "batch_reductions": batch_reductions})
    result = {"status": "complete", "scale": args.scale, "edge": b["edge"], "rank": args.rank,
              "users": len(selected), "requests": completed_requests,
              "source_signature": source_signature,
              "execution_source_hashes": execution_source_hashes,
              "elapsed_seconds": time.perf_counter()-start,
              "peak_allocated_gib": torch.cuda.max_memory_allocated(device)/(1 << 30),
              "peak_reserved_gib": torch.cuda.max_memory_reserved(device)/(1 << 30),
              "model_load_seconds": model_load_seconds, "history_load_seconds": history_load_seconds,
              "cohort_size_final": b["cohort_size"], "batch_reductions": batch_reductions,
              "reference": reference}
    write_json(output / "complete.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", required=True, choices=["medium", "large", "max"])
    parser.add_argument("--edge", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--rank", type=int, required=True, choices=range(4))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-users", type=int, default=0)
    parser.add_argument("--skip-top-users", type=int, default=0, help="probe only; omit busiest assigned users")
    parser.add_argument("--cohort-size", type=int, default=0, help="measured batch override")
    parser.add_argument("--unit-users", type=int, default=128)
    parser.add_argument("--history-threads", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--verify", action="store_true")
    run(parser.parse_args())
