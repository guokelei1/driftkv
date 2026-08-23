#!/usr/bin/env python3
"""Verify that one migrated cutover state serves multiple later queries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

import eval_p8_release_raw as p8raw
import eval_p9_materialized_lineage_canary as rolling
import train_p7_theta0 as p7
from hstu_kvcache.data.p7_training import load_p7_requests
from hstu_kvcache.models import append_with_rolling_cap


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_7_uid_executor_canary_v1.yaml"
POPULATION = ROOT / "data/manifests/p9_full_population_v1/edge1/states.parquet"
P8_MANIFEST = ROOT / "data/manifests/p8_release_v1"
LISTENS = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
OUTPUT = ROOT / "results/p9/p9_7_uid_executor_canary_v1.json"
CUTOVER = 231 * 86_400


def validate() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "population_contract_sha256": ROOT / "configs/contracts/p9_7_full_population_contract_v1.yaml",
        "materialization_summary_sha256": ROOT / "data/manifests/p9_full_population_v1/materialization_summary.json",
        "population_audit_sha256": ROOT / "results/p9/p9_7_full_population_audit_v1.json",
        "p9_5_result_sha256": ROOT / "configs/contracts/p9_5_rolling_validation_result_v1.yaml",
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P9.7 uid-canary input hash mismatch: {key}")
    return contract


def append_events(current, state, events, previous_timestamp, device):
    if not events:
        return state
    tensors = rolling.event_tensors(events, device, previous_timestamp)
    return append_with_rolling_cap(current, state, *tensors, max_length=512)


@torch.no_grad()
def main() -> None:
    contract = validate()
    device = torch.device("cuda:0")
    requests = load_p7_requests(
        P8_MANIFEST, LISTENS, "edge1_evaluation", "F", manifest_kind="fidelity"
    )
    grouped = {}
    for request in requests:
        grouped.setdefault(request.uid, []).append(request)
    population_uids = set(
        int(value) for value in pq.read_table(POPULATION, columns=["uid"])["uid"].to_pylist()
    )
    eligible = [uid for uid, rows in grouped.items() if uid in population_uids and len(rows) >= 2]
    eligible.sort(key=lambda uid: hashlib.sha256(str(uid).encode()).digest())
    selected_uids = eligible[: int(contract["scope"]["users"])]
    selected_requests = {
        uid: sorted(grouped[uid], key=lambda row: (row.query_timestamp, row.request_id))[:2]
        for uid in selected_uids
    }
    events_by_uid = rolling.load_events(selected_uids, threads=24)
    checkpoint = p8raw.TRAIN_ROOT / "r2/m0_f_seed17/selected.pt"
    current, child = p8raw.load_model(checkpoint, device)
    parent, _ = p8raw.load_model(ROOT / child["parent_checkpoint"], device)
    actions = list(contract["actions"])
    rows = []
    max_shared_delta = max_exact_delta = max_query_mutation = 0.0
    migration_invocations = 0
    shared_append_events = 0
    independent_append_events = 0
    for uid in selected_uids:
        lifetime = events_by_uid[uid]
        snapshot = [row for row in lifetime if row[0] < CUTOVER][-512:]
        post = [row for row in lifetime if row[0] >= CUTOVER]
        snapshot_tensors = rolling.event_tensors(snapshot, device)
        parent_snapshot = parent.compute_kv(*snapshot_tensors)
        current_state = current.compute_kv(*snapshot_tensors)
        states = {}
        for action in actions:
            states[action] = rolling.migrate(action, current, parent_snapshot, snapshot_tensors)
            migration_invocations += 1
        cursor = 0
        previous_timestamp = snapshot[-1][0]
        for request in selected_requests[uid]:
            begin = cursor
            while cursor < len(post) and post[cursor][0] < request.query_timestamp:
                cursor += 1
            new_events = post[begin:cursor]
            for action in actions:
                states[action] = append_events(
                    current, states[action], new_events, previous_timestamp, device
                )
                shared_append_events += len(new_events)
            current_state = append_events(
                current, current_state, new_events, previous_timestamp, device
            )
            if new_events:
                previous_timestamp = new_events[-1][0]
            shared_scores = {}
            for action in actions:
                before_k = states[action].k.clone()
                before_v = states[action].v.clone()
                shared_scores[action] = rolling.score(
                    current, states[action], request, previous_timestamp, device
                )
                max_query_mutation = max(
                    max_query_mutation,
                    float((states[action].k - before_k).abs().max()),
                    float((states[action].v - before_v).abs().max()),
                )

            all_suffix = [row for row in post if row[0] < request.query_timestamp]
            independent_last = all_suffix[-1][0] if all_suffix else snapshot[-1][0]
            for action in actions:
                independent = rolling.migrate(action, current, parent_snapshot, snapshot_tensors)
                independent = append_events(
                    current, independent, all_suffix, snapshot[-1][0], device
                )
                independent_append_events += len(all_suffix)
                independent_score = rolling.score(
                    current, independent, request, independent_last, device
                )
                max_shared_delta = max(
                    max_shared_delta, abs(shared_scores[action] - independent_score)
                )
                rows.append({
                    "uid": uid, "request_id": request.request_id,
                    "query_timestamp": request.query_timestamp, "action": action,
                    "shared_logit": shared_scores[action],
                    "independent_logit": independent_score,
                })
            current_score = rolling.score(
                current, current_state, request, previous_timestamp, device
            )
            max_exact_delta = max(
                max_exact_delta, abs(shared_scores["exact_all"] - current_score)
            )
    expected_migrations = len(selected_uids) * len(actions)
    threshold = float(contract["gates"]["shared_vs_independent_max_abs_logit"])
    passed = (
        max_shared_delta <= threshold
        and max_exact_delta <= float(contract["gates"]["exact_action_vs_current_online_max_abs_logit"])
        and max_query_mutation == 0.0
        and migration_invocations == expected_migrations
    )
    payload = {
        "status": "passed" if passed else "failed",
        "contract_hash": p7.sha256_file(CONTRACT),
        "users": len(selected_uids),
        "queries": sum(len(values) for values in selected_requests.values()),
        "actions": actions,
        "migration_invocations": migration_invocations,
        "expected_migration_invocations": expected_migrations,
        "shared_append_action_events": shared_append_events,
        "independent_counterfactual_append_action_events": independent_append_events,
        "append_work_avoided_fraction_vs_per_request_replay": (
            1.0 - shared_append_events / independent_append_events
            if independent_append_events else 0.0
        ),
        "max_shared_vs_independent_abs_logit": max_shared_delta,
        "max_exact_action_vs_current_online_abs_logit": max_exact_delta,
        "max_query_state_mutation": max_query_mutation,
        "raw_rows": rows,
        "full_uid_keyed_executor_authorized": passed,
        "scheduler_authorized": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in payload if key != "raw_rows"}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
