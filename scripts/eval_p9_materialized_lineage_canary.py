#!/usr/bin/env python3
"""Evaluate a frozen true rolling-cache lineage canary for P9.4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

import eval_p7_h_raw as p7eval
import eval_p8_release_raw as p8raw
import train_p7_theta0 as p7
from hstu_kvcache.data.p7_training import P7Request, load_p7_requests
from hstu_kvcache.models import (
    HSTUKVCache,
    append_with_rolling_cap,
    hybrid_tail_refresh,
    project_exact_layer0_segment,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_4_materialized_lineage_canary_v1.yaml"
MANIFEST = ROOT / "data/manifests/p8_release_v1"
LISTENS = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
OUTPUT = ROOT / "results/p9/p9_4_materialized_lineage_canary_v1"
ACTIONS = (
    "noop", "layer0_recent128", "layer0_middle", "layer0_full",
    "hybrid_tail128", "exact_all",
)


def validate_contract() -> dict[str, Any]:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p8_release_contract_sha256": ROOT / "configs/contracts/f_release_chain_contract_v1.yaml",
        "p9_4_lineage_audit_sha256": ROOT / "results/p9/p9_4_release_lineage_audit_v1.json",
        "p9_4_request_local_raw_seal_sha256": ROOT / "results/p9/p9_4_executor_raw_seal_v1.json",
        "p8_materialization_summary_sha256": ROOT / "data/manifests/p8_release_v1/materialization_summary.json",
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"materialized-lineage canary input hash mismatch: {key}")
    if tuple(contract["actions"]) != ACTIONS:
        raise RuntimeError("materialized-lineage action set changed")
    return contract


def request_source(release: str, view: str) -> Path:
    return ROOT / "results/p8/staleness_raw" / release / "m0_f_seed17" / f"F_{view}.parquet"


def classify_requests(release: str, view: str, cutover: int, threads: int) -> list[dict[str, Any]]:
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={threads:d}")
    rows = connection.execute(
        """
        WITH requests AS (
          SELECT DISTINCT request_id, uid, query_timestamp,
                 prefix_tokens_at_cutover AS request_local_retained,
                 suffix_tokens_after_cutover AS request_local_suffix
          FROM read_parquet(?)
        ), counts AS (
          SELECT r.*,
                 least(512, count(*) FILTER (WHERE l.timestamp < ?))::INTEGER
                   AS snapshot_tokens,
                 count(*) FILTER (
                   WHERE l.timestamp >= ? AND l.timestamp < r.query_timestamp
                 )::INTEGER AS suffix_tokens
          FROM requests r
          JOIN read_parquet(?) l ON l.uid = r.uid AND l.timestamp < r.query_timestamp
          GROUP BY ALL
        )
        SELECT *, snapshot_tokens + suffix_tokens > 512 AS eviction_required
        FROM counts
        """,
        [str(request_source(release, view)), cutover, cutover, str(LISTENS)],
    ).fetchall()
    names = [column[0] for column in connection.description]
    return [dict(zip(names, row, strict=True)) for row in rows]


def select_requests(rows: list[dict[str, Any]], contract: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    definitions = {
        "no_eviction": lambda row: not row["eviction_required"],
        "eviction_short_suffix_1_to_32": lambda row: row["eviction_required"] and 1 <= row["suffix_tokens"] <= 32,
        "eviction_long_suffix_128_to_256": lambda row: row["eviction_required"] and 128 <= row["suffix_tokens"] <= 256,
    }
    selected: list[str] = []
    counts: dict[str, int] = {}
    for name, wanted in contract["scope"]["per_view_buckets"].items():
        candidates = [row for row in rows if definitions[name](row)]
        candidates.sort(key=lambda row: hashlib.sha256(str(row["request_id"]).encode()).digest())
        if len(candidates) < int(wanted):
            raise RuntimeError(f"insufficient requests for frozen bucket {name}")
        chosen = candidates[: int(wanted)]
        selected.extend(str(row["request_id"]) for row in chosen)
        counts[name] = len(chosen)
    if len(selected) != len(set(selected)):
        raise RuntimeError("canary selection buckets overlap")
    return selected, counts


def load_events(uids: list[int], threads: int) -> dict[int, list[tuple[int, int, int]]]:
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={threads:d}")
    connection.register("selected_uids", pa.table({"uid": pa.array(sorted(set(uids)), type=pa.int64())}))
    rows = connection.execute(
        """
        WITH raw AS (
          SELECT row_number() OVER () AS raw_order, uid, timestamp, item_id, is_organic
          FROM read_parquet(?)
        )
        SELECT raw.uid, raw.timestamp, raw.item_id, raw.is_organic
        FROM raw JOIN selected_uids USING(uid)
        ORDER BY raw.uid, raw.raw_order
        """,
        [str(LISTENS)],
    ).fetchall()
    output: dict[int, list[tuple[int, int, int]]] = {}
    for uid, timestamp, item_id, organic in rows:
        output.setdefault(int(uid), []).append((int(timestamp), int(item_id), int(organic)))
    return output


def event_tensors(
    events: list[tuple[int, int, int]], device: torch.device, previous_timestamp: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not events:
        raise ValueError("event tensors require at least one event")
    timestamps = np.asarray([row[0] for row in events], dtype=np.int64)
    deltas = np.zeros(len(events), dtype=np.float32)
    if previous_timestamp is not None:
        deltas[0] = np.clip(timestamps[0] - previous_timestamp, 0, 7 * 86_400)
    if len(events) > 1:
        deltas[1:] = np.diff(timestamps).clip(0, 7 * 86_400)
    items = torch.tensor([[row[1] for row in events]], dtype=torch.long, device=device)
    behaviors = torch.tensor([[1 + (1 - row[2]) for row in events]], dtype=torch.long, device=device)
    return items, behaviors, torch.tensor(deltas[None, :], dtype=torch.float32, device=device)


def migrate(
    action: str,
    current,
    parent_snapshot: HSTUKVCache,
    snapshot_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> HSTUKVCache:
    items, behaviors, deltas = snapshot_tensors
    if action == "noop":
        return parent_snapshot
    if action.startswith("layer0_"):
        segment = action.removeprefix("layer0_")
        segment = "recent_128" if segment == "recent128" else segment
        return project_exact_layer0_segment(
            current, parent_snapshot, items, behaviors, deltas, segment
        )
    if action == "hybrid_tail128":
        return hybrid_tail_refresh(current, parent_snapshot, items, behaviors, deltas, 128)
    if action == "exact_all":
        return current.compute_kv(items, behaviors, deltas)
    raise ValueError(f"unknown action {action}")


def score(current, state: HSTUKVCache, request: P7Request, last_timestamp: int, device: torch.device) -> float:
    candidate = torch.tensor(request.candidate_ids[None, :], dtype=torch.long, device=device)
    query_delta = torch.tensor(
        [np.clip(request.query_timestamp - last_timestamp, 0, 7 * 86_400)],
        dtype=torch.float32,
        device=device,
    )
    query_type = torch.full((1,), 2, dtype=torch.long, device=device)
    value = current.score_cc_reuse(
        state, candidate, query_delta,
        prefix_lengths=torch.tensor([state.seq_len], dtype=torch.long, device=device),
        query_type_ids=query_type,
    )
    return float(value[0, 0])


def bernoulli_js(left_logit: float, right_logit: float) -> float:
    p = 1.0 / (1.0 + math.exp(-left_logit))
    q = 1.0 / (1.0 + math.exp(-right_logit))
    midpoint = 0.5 * (p + q)
    epsilon = 1e-15
    p, q, midpoint = [min(1 - epsilon, max(epsilon, value)) for value in (p, q, midpoint)]
    def kl(a: float, b: float) -> float:
        return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))
    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def logloss(logit: float, label: int) -> float:
    return max(logit, 0.0) - label * logit + math.log1p(math.exp(-abs(logit)))


@torch.no_grad()
def evaluate_view(
    release: str,
    view: str,
    current,
    parent,
    device: torch.device,
    threads: int,
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split, cutover = p8raw.RELEASE_EDGE[release]
    classified = classify_requests(release, view, cutover, threads)
    selected_ids, bucket_counts = select_requests(classified, contract)
    classification = {str(row["request_id"]): row for row in classified if str(row["request_id"]) in selected_ids}
    all_requests = load_p7_requests(MANIFEST, LISTENS, split, "F", manifest_kind=view)
    by_id = {row.request_id: row for row in all_requests}
    requests = [by_id[request_id] for request_id in selected_ids]
    events_by_uid = load_events([row.uid for row in requests], threads)
    base = p7eval.load_base("F", device)
    old_table = pq.read_table(request_source(release, view)).to_pylist()
    old = {str(row["request_id"]): row for row in old_table}
    rows: list[dict[str, Any]] = []
    max_exact_delta = 0.0
    count_mismatches = 0
    for request in requests:
        lifetime = events_by_uid[request.uid]
        snapshot = [row for row in lifetime if row[0] < cutover][-512:]
        suffix = [row for row in lifetime if cutover <= row[0] < request.query_timestamp]
        expected = classification[request.request_id]
        if len(snapshot) != int(expected["snapshot_tokens"]) or len(suffix) != int(expected["suffix_tokens"]):
            count_mismatches += 1
        snapshot_tensors = event_tensors(snapshot, device)
        parent_snapshot = parent.compute_kv(*snapshot_tensors)
        current_snapshot = current.compute_kv(*snapshot_tensors)
        suffix_tensors = None if not suffix else event_tensors(suffix, device, snapshot[-1][0])
        current_online = current_snapshot
        if suffix_tensors is not None:
            current_online = append_with_rolling_cap(current, current_online, *suffix_tensors, max_length=512)
        last_timestamp = suffix[-1][0] if suffix else snapshot[-1][0]
        base_logit = float(base(torch.tensor(request.base_features[None, :, :], dtype=torch.float32, device=device))[0, 0])
        current_logit = base_logit + score(current, current_online, request, last_timestamp, device)
        for action in ACTIONS:
            state = migrate(action, current, parent_snapshot, snapshot_tensors)
            if suffix_tensors is not None:
                state = append_with_rolling_cap(current, state, *suffix_tensors, max_length=512)
            action_logit = base_logit + score(current, state, request, last_timestamp, device)
            if action == "exact_all":
                max_exact_delta = max(max_exact_delta, abs(action_logit - current_logit))
            old_row = old[request.request_id]
            row = {
                "request_id": request.request_id,
                "uid": request.uid,
                "query_timestamp": request.query_timestamp,
                "release": release,
                "view": view,
                "action": action,
                "snapshot_tokens": len(snapshot),
                "suffix_tokens": len(suffix),
                "eviction_required": len(snapshot) + len(suffix) > 512,
                "current_online_logit": current_logit,
                "action_online_logit": action_logit,
                "online_JS": bernoulli_js(current_logit, action_logit),
                "request_local_full_logit": float(old_row["current_full512_logit"]),
                "request_local_reuse_logit": float(old_row["reuse_parent_kv_logit"]),
                "request_local_JS": bernoulli_js(
                    float(old_row["current_full512_logit"]), float(old_row["reuse_parent_kv_logit"])
                ),
                "current_online_vs_request_local_full_JS": bernoulli_js(
                    current_logit, float(old_row["current_full512_logit"])
                ),
                "label": request.label if view == "quality" else None,
            }
            if view == "quality":
                assert request.label in (0, 1)
                row["action_logloss"] = logloss(action_logit, int(request.label))
                row["current_logloss"] = logloss(current_logit, int(request.label))
            rows.append(row)
    return rows, {
        "release": release,
        "view": view,
        "requests": len(requests),
        "bucket_counts": bucket_counts,
        "count_mismatches": count_mismatches,
        "max_exact_all_vs_current_online_abs_logit": max_exact_delta,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for release in ("r0", "r2"):
        for view in ("fidelity", "quality"):
            for action in ACTIONS:
                selected = [row for row in rows if row["release"] == release and row["view"] == view and row["action"] == action]
                js = np.asarray([row["online_JS"] for row in selected], dtype=np.float64)
                old_js = np.asarray([row["request_local_JS"] for row in selected], dtype=np.float64)
                entry = {
                    "release": release,
                    "view": view,
                    "action": action,
                    "requests": len(selected),
                    "online_JS_mean": float(js.mean()),
                    "online_JS_max": float(js.max()),
                    "request_local_JS_mean": float(old_js.mean()),
                    "online_vs_request_local_S_ratio": float(js.mean() / old_js.mean()) if old_js.mean() > 0 else None,
                    "current_online_vs_request_local_full_JS_mean": float(np.mean([
                        row["current_online_vs_request_local_full_JS"] for row in selected
                    ])),
                }
                if action != "noop":
                    noop = [row for row in rows if row["release"] == release and row["view"] == view and row["action"] == "noop"]
                    noop_js = float(np.mean([row["online_JS"] for row in noop]))
                    entry["signed_JS_recovery_fraction_vs_noop"] = (
                        (noop_js - float(js.mean())) / noop_js if noop_js > 1e-15 else None
                    )
                if view == "quality":
                    entry["action_minus_current_logloss"] = float(np.mean([
                        row["action_logloss"] - row["current_logloss"] for row in selected
                    ]))
                output.append(entry)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.device not in {"cuda:0", "cuda:1"}:
        raise ValueError("canary is restricted to cuda:0 or cuda:1")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    contract = validate_contract()
    device = torch.device(args.device)
    all_rows: list[dict[str, Any]] = []
    audits = []
    for release in contract["scope"]["releases"]:
        checkpoint = p8raw.TRAIN_ROOT / release / "m0_f_seed17" / "selected.pt"
        current, child = p8raw.load_model(checkpoint, device)
        parent, _ = p8raw.load_model(ROOT / child["parent_checkpoint"], device)
        for view in contract["scope"]["views"]:
            rows, audit = evaluate_view(release, view, current, parent, device, args.threads, contract)
            all_rows.extend(rows)
            audits.append(audit)
        del current, parent
        torch.cuda.empty_cache()
    summaries = summarize(all_rows)
    gates = contract["blocking_gates"]
    exact_max = max(row["max_exact_all_vs_current_online_abs_logit"] for row in audits)
    r0_noop_max = max(row["online_JS_max"] for row in summaries if row["release"] == "r0" and row["action"] == "noop")
    r0_all_max = max(row["online_JS_max"] for row in summaries if row["release"] == "r0")
    passed = (
        exact_max <= float(gates["exact_all_vs_current_online_max_abs_logit"])
        and r0_noop_max <= float(gates["r0_noop_JS_max"])
        and r0_all_max <= float(gates["r0_all_action_JS_max"])
        and all(row["count_mismatches"] == 0 for row in audits)
    )
    args.output.mkdir(parents=True)
    raw_path = args.output / "raw.parquet"
    pq.write_table(pa.Table.from_pylist(all_rows), raw_path, compression="zstd")
    payload = {
        "status": "passed" if passed else "failed",
        "contract": str(CONTRACT.relative_to(ROOT)),
        "contract_sha256": p7.sha256_file(CONTRACT),
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": p7.sha256_file(raw_path),
        "audits": audits,
        "blocking_gate_observations": {
            "exact_max_abs_logit": exact_max,
            "r0_noop_JS_max": r0_noop_max,
            "r0_all_action_JS_max": r0_all_max,
        },
        "summaries": summaries,
        "full_matrix_authorized": passed,
        "scheduler_authorized": False,
    }
    (args.output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
