#!/usr/bin/env python3
"""Raw heldout quality logits for recursive lineage and frozen actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

import eval_p7_h_raw as p7eval
import eval_p8_release_raw as p8raw
import eval_p9_cutover_profiler_raw as profiler
import eval_p9_heldout_rolling_quality_raw as p9q
import eval_p9_materialized_lineage_canary as rolling
import eval_p11_recursive_population as p11
import train_p7_theta0 as p7
from hstu_kvcache.data.p7_training import load_p7_requests


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p11_4_recursive_policy_quality_v1.yaml"
MANIFEST = ROOT / "data/manifests/p8_release_v1"
POPULATION = ROOT / "data/manifests/p9_full_population_v1"
LISTENS = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
OUTPUT_ROOT = ROOT / "results/p11/p11_4_recursive_policy_quality_raw"
ACTIONS = p9q.ACTIONS
CUTOVER1, CUTOVER2 = 19_958_400, 21_168_000


def validate():
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p11_2_assignment_seal_sha256": ROOT / "results/p11/p11_2_recursive_scheduler_full_seal_v1.json",
        "p11_3_target_free_gate_sha256": ROOT / "results/p11/p11_3_recursive_scheduler_baseline_gate_v1.json",
        "p11_3_contract_sha256": ROOT / "configs/contracts/p11_3_recursive_scheduler_baseline_gate_v1.yaml",
        "p9_9_quality_contract_sha256": ROOT / "configs/contracts/p9_9_heldout_rolling_quality_contract_v1.yaml",
        "p8_edge2_raw_seal_sha256": ROOT / "results/p8/r1_edge2/raw_score_seal_v1.json",
        "p8_materialization_summary_sha256": MANIFEST / "materialization_summary.json",
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P11.4 input mismatch: {key}")
    return contract


@torch.no_grad()
def evaluate(model, seed, device, max_users, output):
    contract = validate()
    started = time.perf_counter()
    manifest_requests = load_p7_requests(MANIFEST, LISTENS, "edge2_evaluation", "F", manifest_kind="quality")
    sealed_source = ROOT / "results/p8/staleness_raw/r1_edge2" / f"{model}_seed{seed}/F_quality.parquet"
    sealed_ids = set(str(value) for value in pq.read_table(sealed_source, columns=["request_id"])["request_id"].to_pylist())
    requests = [request for request in manifest_requests if request.request_id in sealed_ids]
    grouped = {}
    for request in requests:
        grouped.setdefault(request.uid, []).append(request)
    for values in grouped.values():
        values.sort(key=lambda row: (row.query_timestamp, row.request_id))
    pointers = p9q.request_pointers("edge2_evaluation")
    edge1 = {int(row["uid"]): dict(row) for row in pq.read_table(POPULATION / "edge1/states.parquet").to_pylist()}
    edge2 = {int(row["uid"]): dict(row) for row in pq.read_table(POPULATION / "edge2/states.parquet").to_pylist()}
    eligible = set(grouped) & set(edge1) & set(edge2)
    cold_users = set(grouped) - eligible
    grouped = {uid: grouped[uid] for uid in eligible}
    if max_users is not None:
        chosen = sorted(grouped, key=lambda uid: hashlib.sha256(str(uid).encode()).digest())[:max_users]
        grouped = {uid: grouped[uid] for uid in chosen}
    reader = p9q.RangeReader(LISTENS)
    recursive_reader = profiler.RawStateReader()
    records = []
    for uid in sorted(grouped, key=lambda value: int(edge2[value]["raw_prefix_end_exclusive"])):
        row1, row2 = edge1[uid], edge2[uid]
        row1["cutover"], row2["cutover"] = CUTOVER1, CUTOVER2
        snapshot_end, snapshot_length = int(row2["raw_prefix_end_exclusive"]), int(row2["effective_prefix_length"])
        max_end = max(pointers[request.request_id] for request in grouped[uid])
        table = reader.rows(snapshot_end - snapshot_length, max_end)
        if not np.all(table["uid"].to_numpy(zero_copy_only=False) == uid):
            raise RuntimeError("P11.4 raw range crossed uid")
        timestamps, items, organic = p9q.make_event_arrays(table)
        last1_table = recursive_reader.rows(int(row1["raw_prefix_end_exclusive"]) - 1, int(row1["raw_prefix_end_exclusive"]))
        records.append({
            "uid": uid, "row1": row1, "row2": row2,
            "suffix": p11.raw_events(recursive_reader, row1, row2),
            "last1": int(last1_table["timestamp"][0].as_py()),
            "snapshot_end": snapshot_end, "snapshot_length": snapshot_length,
            "timestamps": timestamps, "items": items, "organic": organic, "requests": grouped[uid],
        })
    checkpoint2 = p8raw.TRAIN_ROOT / "r1_edge2" / f"{model}_seed{seed}/selected.pt"
    theta2, child2 = p8raw.load_model(checkpoint2, device)
    theta1, child1 = p8raw.load_model(ROOT / child2["parent_checkpoint"], device)
    theta0, _ = p8raw.load_model(ROOT / child1["parent_checkpoint"], device)
    base = p7eval.load_base("F", device)
    rows, max_exact = [], 0.0
    by_length = {}
    for record in records:
        by_length.setdefault(record["snapshot_length"], []).append(record)
    batch_size = 16
    for length in sorted(by_length):
        values = by_length[length]
        for begin in range(0, len(values), batch_size):
            batch_records = values[begin : begin + batch_size]
            recursive = p11.initial_recursive_states(theta0, recursive_reader, batch_records, device, batch_size)
            recursive = p11.append_recursive(theta1, recursive, batch_records, device, batch_size, 512)
            recursive_batch = p11.merge_caches(recursive)
            tensors = p9q.batch_snapshot_tensors(batch_records, device)
            migrated = [rolling.migrate(action, theta2, recursive_batch, tensors) for action in ACTIONS]
            current_snapshot = theta2.compute_kv(*tensors)
            max_exact = max(max_exact, float((migrated[-1].k - current_snapshot.k).abs().max()),
                            float((migrated[-1].v - current_snapshot.v).abs().max()))
            states = [p9q.stack_action_caches(migrated, index) for index in range(len(batch_records))]
            cursors, positions = [0] * len(batch_records), [0] * len(batch_records)
            previous = [int(record["timestamps"][record["snapshot_length"] - 1]) for record in batch_records]
            while any(position < len(record["requests"]) for position, record in zip(positions, batch_records, strict=True)):
                pending, payloads = [], []
                for index, record in enumerate(batch_records):
                    if positions[index] >= len(record["requests"]):
                        continue
                    request = record["requests"][positions[index]]
                    target_end = pointers[request.request_id] - record["snapshot_end"]
                    absolute_start = record["snapshot_length"] + cursors[index]
                    absolute_end = record["snapshot_length"] + target_end
                    event_timestamps = record["timestamps"][absolute_start:absolute_end]
                    if len(event_timestamps) and np.any(event_timestamps >= request.query_timestamp):
                        raise RuntimeError("P11.4 appended coincident/future event")
                    payloads.append({"index": index, "timestamps": event_timestamps,
                                     "items": record["items"][absolute_start:absolute_end],
                                     "organic": record["organic"][absolute_start:absolute_end]})
                    cursors[index] = target_end
                    pending.append((index, request))
                p9q.append_user_groups(theta2, states, previous, payloads, device)
                by_state = {}
                for item in pending:
                    by_state.setdefault(states[item[0]].seq_len, []).append(item)
                for selected in by_state.values():
                    indices = [item[0] for item in selected]
                    selected_requests = [item[1] for item in selected]
                    logits = p9q.score_action_groups(
                        theta2, base, [states[index] for index in indices], selected_requests,
                        [previous[index] for index in indices], device,
                    )
                    for request, request_logits in zip(selected_requests, logits, strict=True):
                        current = float(request_logits[-1])
                        for action, action_logit in zip(ACTIONS, request_logits, strict=True):
                            rows.append({"request_id": request.request_id, "uid": request.uid,
                                         "query_timestamp": request.query_timestamp, "action": action,
                                         "action_logit": float(action_logit), "current_exact_logit": current,
                                         "label": int(request.label), "request_weight": float(request.request_weight),
                                         "prior_30m_same_item": request.prior_30m_same_item,
                                         "latest_item": request.latest_item, "is_organic": request.is_organic})
                    for index in indices:
                        positions[index] += 1
            print(f"{model} seed{seed}: quality users {min(begin + batch_size, len(values))}/{len(values)} length={length}", flush=True)
    expected = sum(len(value) for value in grouped.values())
    if len(rows) != expected * len(ACTIONS) or max_exact > 1e-5:
        raise RuntimeError("P11.4 conservation or Exact gate failed")
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "quality_actions.parquet"
    pq.write_table(pa.Table.from_pylist(rows), raw_path, compression="zstd")
    payload = {"status": "passed_recursive_quality_raw_unadjudicated", "model": model, "seed": seed,
               "scope": "canary" if max_users is not None else "full", "users": len(grouped),
               "requests": expected, "actions": len(ACTIONS), "cold_users": len(cold_users),
               "max_recursive_exact_KV_difference": max_exact, "wall_seconds": time.perf_counter() - started,
               "raw_path": str(raw_path.relative_to(ROOT)), "raw_sha256": p7.sha256_file(raw_path),
               "contract_sha256": p7.sha256_file(CONTRACT), "metrics_computed": False}
    (output / "raw_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("m0_f", "m1"), required=True)
    parser.add_argument("--seed", choices=(17, 37, 71), type=int, required=True)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), required=True)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suffix = "full" if args.max_users is None else f"canary{args.max_users}"
    output = args.output or OUTPUT_ROOT / suffix / f"{args.model}_seed{args.seed}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    print(json.dumps(evaluate(args.model, args.seed, torch.device(args.device), args.max_users, output), indent=2))


if __name__ == "__main__":
    main()
