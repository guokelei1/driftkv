#!/usr/bin/env python3
"""Evaluate true recursive version debt and frozen actions for one model/seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

import adjudicate_p9_heldout_rolling_quality as quality
import eval_p8_release_raw as p8raw
import eval_p9_cutover_profiler_raw as profiler
import eval_p9_materialized_lineage_canary as rolling
import train_p7_theta0 as p7
from hstu_kvcache.models import HSTUKVCache, append_with_rolling_cap


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p11_1_recursive_population_contract_v1.yaml"
POPULATION = ROOT / "data/manifests/p9_full_population_v1"
OUTPUT_ROOT = ROOT / "results/p11/p11_1_recursive_population_raw"
CUTOVER1 = 19_958_400
CUTOVER2 = 21_168_000
LINEAGES = ("one_hop", "direct_age2", "recursive_noop")


def validate() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "full_stack_freeze_sha256": ROOT / "configs/contracts/p10_6_full_stack_freeze_v1.yaml",
        "p11_0_contract_sha256": ROOT / "configs/contracts/p11_0_version_debt_contract_v1.yaml",
        "p11_0_result_sha256": ROOT / "results/p11/p11_0_version_debt_canary_v1/result.json",
        "p11_0_raw_sha256": ROOT / "results/p11/p11_0_version_debt_canary_v1/state_metrics.parquet",
        "edge1_manifest_sha256": POPULATION / "edge1/manifest.json",
        "edge2_manifest_sha256": POPULATION / "edge2/manifest.json",
        "edge2_probe_sha256": POPULATION / "edge2/cutover_probes.parquet",
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P11.1 input mismatch: {key}")
    return contract


def select_cache(cache: HSTUKVCache, index: int) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k[:, index : index + 1].clone(),
        v=cache.v[:, index : index + 1].clone(),
        seq_len=cache.seq_len,
    )


def merge_caches(caches: list[HSTUKVCache]) -> HSTUKVCache:
    if not caches or len({cache.seq_len for cache in caches}) != 1:
        raise ValueError("cache merge requires a nonempty equal-length group")
    return HSTUKVCache(
        k=torch.cat([cache.k for cache in caches], dim=1),
        v=torch.cat([cache.v for cache in caches], dim=1),
        seq_len=caches[0].seq_len,
    )


def raw_events(reader: profiler.RawStateReader, row1: dict, row2: dict) -> list[tuple[int, int, int]]:
    start, end, uid = (
        int(row1["raw_prefix_end_exclusive"]),
        int(row2["raw_prefix_end_exclusive"]),
        int(row1["uid"]),
    )
    if start == end:
        return []
    table = reader.rows(start, end)
    if not np.all(table["uid"].to_numpy(zero_copy_only=False) == uid):
        raise RuntimeError("recursive suffix crossed uid")
    timestamps = table["timestamp"].to_numpy(zero_copy_only=False).astype(np.int64)
    if np.any(np.diff(timestamps) < 0) or timestamps[-1] >= CUTOVER2:
        raise RuntimeError("recursive suffix is not chronological pre-cutover data")
    return list(zip(
        timestamps.tolist(), table["item_id"].to_pylist(), table["is_organic"].to_pylist(), strict=True
    ))


def initial_recursive_states(theta0, reader, records, device, batch_size):
    states: list[HSTUKVCache | None] = [None] * len(records)
    groups: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(int(record["row1"]["effective_prefix_length"]), []).append(index)
    for indices in groups.values():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            rows = [records[index]["row1"] for index in selected]
            items, behaviors, deltas, _ = profiler.state_tensors(reader, rows, device)
            merged = theta0.compute_kv(items, behaviors, deltas)
            for offset, index in enumerate(selected):
                states[index] = select_cache(merged, offset)
    if any(state is None for state in states):
        raise RuntimeError("failed to materialize every theta0 edge1 state")
    return states


def append_recursive(theta1, states, records, device, batch_size, cap):
    previous = [int(record["last1"]) for record in records]
    maximum = max((len(record["suffix"]) for record in records), default=0)
    for position in range(maximum):
        active = [index for index, record in enumerate(records) if position < len(record["suffix"])]
        groups: dict[int, list[int]] = {}
        for index in active:
            groups.setdefault(states[index].seq_len, []).append(index)
        for indices in groups.values():
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                merged = merge_caches([states[index] for index in selected])
                events = [records[index]["suffix"][position] for index in selected]
                timestamps = [int(event[0]) for event in events]
                if any(timestamp < previous[index] for timestamp, index in zip(timestamps, selected, strict=True)):
                    raise RuntimeError("recursive append timestamp regressed")
                items = torch.tensor([[int(event[1])] for event in events], dtype=torch.long, device=device)
                behaviors = torch.tensor(
                    [[1 + (1 - int(event[2]))] for event in events], dtype=torch.long, device=device
                )
                deltas = torch.tensor(
                    [[np.clip(timestamp - previous[index], 0, 7 * 86_400)]
                     for timestamp, index in zip(timestamps, selected, strict=True)],
                    dtype=torch.float32, device=device,
                )
                updated = append_with_rolling_cap(theta1, merged, items, behaviors, deltas, cap)
                for offset, (index, timestamp) in enumerate(zip(selected, timestamps, strict=True)):
                    states[index] = select_cache(updated, offset)
                    previous[index] = timestamp
    return states


def metric_row(uid, action, current, value, suffix_events):
    delta = value - current
    current_np, value_np = current.float().cpu().numpy(), value.float().cpu().numpy()
    mse = float(torch.mean(delta.float().square()))
    current_rms = float(torch.mean(current.float().square()).sqrt())
    current_prob = 1.0 / (1.0 + np.exp(-current_np))
    value_prob = 1.0 / (1.0 + np.exp(-value_np))
    return {
        "uid": int(uid), "action": action, "suffix_events": int(suffix_events),
        "mse": mse, "normalized_rms": float(np.sqrt(mse) / (current_rms + 1e-8)),
        "bernoulli_js": float(np.mean(quality.bernoulli_js(value_np, current_np))),
        "mean_abs_probability_shift": float(np.mean(np.abs(value_prob - current_prob))),
        "max_abs_logit_shift": float(torch.max(torch.abs(delta.float()))),
    }


@torch.no_grad()
def evaluate(model_name, seed, device, state_limit, output):
    contract = validate()
    edge1 = {int(row["uid"]): dict(row) for row in pq.read_table(POPULATION / "edge1/states.parquet").to_pylist()}
    edge2 = {int(row["uid"]): dict(row) for row in pq.read_table(POPULATION / "edge2/states.parquet").to_pylist()}
    probes = {int(row["uid"]): row for row in pq.read_table(POPULATION / "edge2/cutover_probes.parquet").to_pylist()}
    uids = list(set(edge1) & set(edge2))
    if state_limit is not None:
        uids.sort(key=lambda uid: hashlib.sha256(str(uid).encode()).digest())
        uids = uids[:state_limit]
    else:
        uids.sort(key=lambda uid: int(edge2[uid]["raw_prefix_end_exclusive"]))
        expected = int(contract["scope"]["expected_population"])
        if len(uids) != expected:
            raise RuntimeError(f"population changed: {len(uids)} != {expected}")
    checkpoint2 = p8raw.TRAIN_ROOT / "r1_edge2" / f"{model_name}_seed{seed}" / "selected.pt"
    theta2, child2 = p8raw.load_model(checkpoint2, device)
    checkpoint1 = ROOT / child2["parent_checkpoint"]
    theta1, child1 = p8raw.load_model(checkpoint1, device)
    checkpoint0 = ROOT / child1["parent_checkpoint"]
    theta0, _ = p8raw.load_model(checkpoint0, device)
    reader = profiler.RawStateReader()
    rows, total_suffix, max_exact = [], 0, 0.0
    chunk_size = int(contract["execution"]["population_chunk_size"])
    batch_size = int(contract["execution"]["model_batch_size"])
    actions = list(contract["scope"]["recursive_actions"])
    for chunk_start in range(0, len(uids), chunk_size):
        chunk_uids = uids[chunk_start : chunk_start + chunk_size]
        records = []
        for uid in chunk_uids:
            row1, row2 = edge1[uid], edge2[uid]
            row1["cutover"], row2["cutover"] = CUTOVER1, CUTOVER2
            suffix = raw_events(reader, row1, row2)
            raw1 = reader.rows(int(row1["raw_prefix_end_exclusive"]) - 1, int(row1["raw_prefix_end_exclusive"]))
            records.append({"uid": uid, "row1": row1, "row2": row2, "suffix": suffix,
                            "last1": int(raw1["timestamp"][0].as_py())})
            total_suffix += len(suffix)
        recursive = initial_recursive_states(theta0, reader, records, device, batch_size)
        recursive = append_recursive(
            theta1, recursive, records, device, batch_size, int(contract["execution"]["rolling_cap"])
        )
        groups: dict[int, list[int]] = {}
        for index, record in enumerate(records):
            expected = int(record["row2"]["effective_prefix_length"])
            if recursive[index].seq_len != expected:
                raise RuntimeError(f"recursive final length mismatch for uid {record['uid']}")
            groups.setdefault(expected, []).append(index)
        for indices in groups.values():
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                selected_records = [records[index] for index in selected]
                items, behaviors, deltas, last2 = profiler.state_tensors(
                    reader, [record["row2"] for record in selected_records], device
                )
                current_cache = theta2.compute_kv(items, behaviors, deltas)
                cache1 = theta1.compute_kv(items, behaviors, deltas)
                cache0 = theta0.compute_kv(items, behaviors, deltas)
                recursive_batch = merge_caches([recursive[index] for index in selected])
                candidates = torch.tensor(
                    [probes[record["uid"]]["candidate_ids"] for record in selected_records],
                    dtype=torch.long, device=device,
                )
                query_delta = torch.tensor(
                    [CUTOVER2 - timestamp for timestamp in last2], dtype=torch.float32, device=device
                ).clamp(0, 7 * 86_400)
                lengths = torch.full((len(selected),), current_cache.seq_len, dtype=torch.long, device=device)
                query_type = torch.full((len(selected),), 2, dtype=torch.long, device=device)
                def score(cache):
                    return theta2.score_cc_reuse(
                        cache, candidates, query_delta, prefix_lengths=lengths,
                        query_type_ids=query_type,
                    ).float()
                current = score(current_cache)
                values = {
                    "one_hop": score(cache1),
                    "direct_age2": score(cache0),
                    "recursive_noop": score(recursive_batch),
                }
                snapshot = (items, behaviors, deltas)
                for action in actions:
                    migrated = rolling.migrate(action, theta2, recursive_batch, snapshot)
                    values[f"recursive_{action}"] = score(migrated)
                max_exact = max(max_exact, float((values["recursive_exact_all"] - current).abs().max()))
                for offset, record in enumerate(selected_records):
                    for action, value in values.items():
                        rows.append(metric_row(
                            record["uid"], action, current[offset], value[offset], len(record["suffix"])
                        ))
        print(f"{model_name} seed{seed}: {min(chunk_start + chunk_size, len(uids))}/{len(uids)}", flush=True)
    if max_exact > float(contract["gates"]["exact_all_max_absolute_logit"]):
        raise RuntimeError(f"exact-all gate failed: {max_exact}")
    numeric = [value for row in rows for key, value in row.items() if key not in ("uid", "action")]
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("non-finite P11.1 metric")
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "state_metrics.parquet"
    pq.write_table(pa.Table.from_pylist(rows), raw_path, compression="zstd")
    payload = {
        "status": "passed_raw_target_free_unadjudicated", "model": model_name, "seed": seed,
        "states": len(uids), "actions": sorted(set(row["action"] for row in rows)),
        "rows": len(rows), "suffix_events": total_suffix, "max_exact_abs_logit": max_exact,
        "contract_sha256": p7.sha256_file(CONTRACT),
        "theta0_sha256": p7.sha256_file(checkpoint0), "theta1_sha256": p7.sha256_file(checkpoint1),
        "theta2_sha256": p7.sha256_file(checkpoint2),
        "raw_path": str(raw_path.relative_to(ROOT)), "raw_sha256": p7.sha256_file(raw_path),
        "state_limit": state_limit, "metrics_adjudicated": False, "future_labels_read": False,
    }
    (output / "raw_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("m0_f", "m1"), required=True)
    parser.add_argument("--seed", choices=(17, 37, 71), type=int, required=True)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), required=True)
    parser.add_argument("--state-limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suffix = "full" if args.state_limit is None else f"canary{args.state_limit}"
    output = args.output or OUTPUT_ROOT / suffix / f"{args.model}_seed{args.seed}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    result = evaluate(args.model, args.seed, torch.device(args.device), args.state_limit, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
