#!/usr/bin/env python3
"""Evaluate sealed 8L policy assignments on rolling F quality requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import eval_p9_materialized_lineage_canary as transition
import eval_scale_8l_hs_raw as hs
import train_p7_theta0 as p7
from hstu_kvcache.data.p7_training import load_p7_requests
from hstu_kvcache.models import HSTUKVCache, append_with_rolling_cap

ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT_SEAL = ROOT / "results/scale_8l_v1/scheduler/assignment_seal.json"
SCHEDULER = ROOT / "results/scale_8l_v1/scheduler/scheduler_result.json"
MANIFEST = ROOT / "data/manifests/p8_release_v1"
LISTENS = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
OUTPUT = ROOT / "results/scale_8l_v1/policy_quality_raw"
BUDGETS = (0.05, 0.10, 0.25)


def combine(states: list[HSTUKVCache]) -> HSTUKVCache:
    if len({state.seq_len for state in states}) != 1: raise ValueError("state lengths differ")
    return HSTUKVCache(k=torch.cat([s.k for s in states], dim=1), v=torch.cat([s.v for s in states], dim=1), seq_len=states[0].seq_len)


def append(current, state, timestamps, items, organic, previous, device):
    if not len(timestamps): return state, previous
    deltas = np.zeros(len(timestamps), dtype=np.float32); deltas[0] = np.clip(int(timestamps[0])-previous, 0, 7*86400)
    if len(timestamps) > 1: deltas[1:] = np.diff(timestamps).clip(0, 7*86400)
    batch = state.k.shape[1]
    state = append_with_rolling_cap(current, state,
        torch.tensor(items[None], dtype=torch.long, device=device).repeat(batch, 1),
        torch.tensor((1+(1-organic))[None], dtype=torch.long, device=device).repeat(batch, 1),
        torch.tensor(deltas[None], dtype=torch.float32, device=device).repeat(batch, 1), 1024)
    return state, int(timestamps[-1])


def score(current, base, state, request, last_timestamp, device):
    batch = state.k.shape[1]
    candidate = torch.tensor(request.candidate_ids[None], dtype=torch.long, device=device).repeat(batch, 1)
    delta = torch.full((batch,), float(np.clip(request.query_timestamp-last_timestamp, 0, 7*86400)), dtype=torch.float32, device=device)
    residual = current.score_cc_reuse(state, candidate, delta,
        prefix_lengths=torch.full((batch,), state.seq_len, dtype=torch.long, device=device),
        query_type_ids=torch.full((batch,), 2, dtype=torch.long, device=device))[:, 0]
    base_logit = float(base(torch.tensor(request.base_features[None], dtype=torch.float32, device=device))[0, 0])
    return [base_logit + float(value) for value in residual]


def assignment_map(release: str):
    scheduler = json.loads(SCHEDULER.read_text()); cell = next(row for row in scheduler["cells"] if row["release"] == release)
    frame = pq.read_table(ROOT / cell["primary_assignments"]).to_pandas()
    return {budget: dict(zip(frame[np.isclose(frame.budget_fraction, budget)].uid.astype(int),
        frame[np.isclose(frame.budget_fraction, budget)].action.astype(str))) for budget in BUDGETS}


@torch.no_grad()
def evaluate(release: str, device: torch.device, output: Path, max_users: int | None = None):
    seal = json.loads(ASSIGNMENT_SEAL.read_text())
    if seal["status"] != "sealed_before_quality" or p7.sha256_file(SCHEDULER) != seal["scheduler_result_sha256"]:
        raise RuntimeError("policy assignments are not sealed")
    split, cutover = hs.RELEASES[release]
    quality = load_p7_requests(MANIFEST, LISTENS, split, "F", manifest_kind="quality", history_limit=1024)
    if max_users is not None:
        selected = set(sorted({row.uid for row in quality},
            key=lambda uid: hashlib.sha256(f"8l-policy-quality:{release}:{uid}".encode()).digest())[:max_users])
        quality = [row for row in quality if row.uid in selected]
    metadata = {row.request_id: row for row in quality}
    scoring = [replace(row, manifest_kind="fidelity_policy_quality_population", target_index=None, label=None,
        is_organic=None, prior_30m_same_item=None, latest_item=None, target_stratum=None) for row in quality]
    pointers = hs.request_pointers(split, "quality"); records, population = hs.build_records(scoring, pointers, cutover, None)
    current_path = hs.checkpoint_path(release); current, child = hs.load_model(current_path, device)
    parent, _ = hs.load_model(ROOT / child["parent_checkpoint"], device); bases, _ = p7.load_bases(("F",), device)
    assignments = assignment_map(release); rows = []; max_exact = max_noop = 0.0; started = time.perf_counter()
    by_length = {}; [by_length.setdefault(row["snapshot_length"], []).append(row) for row in records]
    hs_reference = {str(row["request_id"]): row for row in pq.read_table(ROOT / "results/scale_8l_v1/hs_raw" / release / "m0_f_seed17/F_quality.parquet").to_pylist()}
    for length in sorted(by_length):
        values = by_length[length]
        for begin in range(0, len(values), hs.SNAPSHOT_BATCH):
            batch = values[begin:begin+hs.SNAPSHOT_BATCH]; tensors = hs.snapshot_tensors(batch, device)
            current_batch, parent_batch = current.compute_kv(*tensors), parent.compute_kv(*tensors)
            for index, record in enumerate(batch):
                exact = hs.select_cache(current_batch, index); noop = hs.select_cache(parent_batch, index)
                parent_one = hs.select_cache(parent_batch, index)
                actions = [assignments[budget][int(record["uid"])] for budget in BUDGETS]
                policy_states = [transition.migrate(action, current, parent_one, tuple(t[index:index+1] for t in tensors)) for action in actions]
                state = combine([exact, noop, *policy_states]); cursor = 0; previous = int(record["snapshot_timestamps"][-1])
                for request in record["requests"]:
                    target = pointers[request.request_id][1] - (record["user_start"] + record["snapshot_end"])
                    state, previous = append(current, state, record["suffix_timestamps"][cursor:target],
                        record["suffix_items"][cursor:target], record["suffix_organic"][cursor:target], previous, device)
                    cursor = target; logits = score(current, bases["F"], state, request, previous, device)
                    reference = hs_reference[request.request_id]
                    max_exact = max(max_exact, abs(logits[0]-float(reference["current_exact_rolling_logit"])))
                    max_noop = max(max_noop, abs(logits[1]-float(reference["reuse_parent_rolling_logit"])))
                    label = metadata[request.request_id].label
                    rows.append({"request_id": request.request_id, "uid": int(request.uid), "release": release,
                        "label": int(label), "exact_logit": logits[0], "noop_logit": logits[1],
                        "policy_05_logit": logits[2], "policy_10_logit": logits[3], "policy_25_logit": logits[4],
                        "policy_05_action": actions[0], "policy_10_action": actions[1], "policy_25_action": actions[2]})
            del current_batch, parent_batch
    if max(max_exact, max_noop) > 1e-5: raise RuntimeError("policy replay disagrees with sealed rolling H/S reference")
    output.mkdir(parents=True, exist_ok=False); raw = output / "quality_scores.parquet"
    pq.write_table(pa.Table.from_pylist(rows), raw, compression="zstd")
    payload = {"status": "policy_quality_raw_written_unadjudicated", "release": release,
        "requests": len(rows), "users": len({r["uid"] for r in rows}), "population": population,
        "max_exact_reference_delta": max_exact, "max_noop_reference_delta": max_noop,
        "raw": str(raw.relative_to(ROOT)), "raw_sha256": p7.sha256_file(raw),
        "assignment_seal_sha256": p7.sha256_file(ASSIGNMENT_SEAL), "metrics_computed": False,
        "wall_seconds": time.perf_counter()-started, "qualification_or_theta3_read": False}
    (output / "raw_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("status", "release", "requests", "users", "max_exact_reference_delta", "max_noop_reference_delta", "wall_seconds")}, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--release", choices=("r1_edge1", "r1_edge2", "r2"), required=True)
    parser.add_argument("--device", choices=tuple(f"cuda:{i}" for i in range(4)), required=True); parser.add_argument("--output", type=Path)
    parser.add_argument("--max-users", type=int)
    args = parser.parse_args(); output = (args.output or OUTPUT / args.release / "m0_f_seed17").resolve()
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    evaluate(args.release, torch.device(args.device), output, args.max_users)


if __name__ == "__main__": main()
