#!/usr/bin/env python3
"""P9.6 state-keyed logical migration costs and a batched GPU runtime canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pyarrow as pa
import torch
import yaml

import eval_p8_release_raw as p8raw
import eval_p9_materialized_lineage_canary as rolling
import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_6_transition_cost_contract_v1.yaml"
LISTENS = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
OUTPUT = ROOT / "results/p9/p9_6_transition_costs_v1.json"


def validate() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p9_5_result_contract_sha256": ROOT / "configs/contracts/p9_5_rolling_validation_result_v1.yaml",
        "p9_5_adjudication_sha256": ROOT / "results/p9/p9_5_rolling_validation_v1.json",
        "state_transition_source_sha256": ROOT / "src/hstu_kvcache/models/state_transition.py",
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P9.6 input hash mismatch: {key}")
    return contract


def population_lengths(cutover: int, source: Path, threads: int) -> list[tuple[int, int]]:
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={threads:d}")
    return [(int(uid), int(length)) for uid, length in connection.execute(
        """
        WITH users AS (SELECT DISTINCT uid FROM read_parquet(?))
        SELECT u.uid, least(512, count(*))::INTEGER AS snapshot_tokens
        FROM users u JOIN read_parquet(?) l ON l.uid=u.uid AND l.timestamp < ?
        GROUP BY u.uid ORDER BY u.uid
        """,
        [str(source), str(LISTENS), cutover],
    ).fetchall()]


def logical_cost(length: int, action: str) -> dict[str, int]:
    layers, width, element_bytes = 4, 128, 4
    kv_token_bytes = 2 * width * element_bytes
    raw_token_bytes = 8 + 8 + 4
    if action == "noop":
        tokens = token_layers = pairs = reads = writes = raw = 0
    elif action.startswith("layer0_"):
        segment = action.removeprefix("layer0_")
        tokens = min(128, length) if segment == "recent128" else (
            max(length // 4 + 1, (3 * length + 1) // 4) - length // 4
            if segment == "middle" else length
        )
        token_layers, pairs = tokens, 0
        reads = writes = tokens * kv_token_bytes
        raw = tokens * raw_token_bytes
    elif action == "hybrid_tail128":
        tokens = min(128, length)
        prefix = length - tokens
        token_layers = tokens * layers
        pairs = layers * (tokens * prefix + tokens * (tokens + 1) // 2)
        reads = prefix * layers * kv_token_bytes
        writes = tokens * layers * kv_token_bytes
        raw = tokens * raw_token_bytes
    elif action == "exact_all":
        tokens = length
        token_layers = length * layers
        pairs = layers * length * (length + 1) // 2
        reads = 0
        writes = length * layers * kv_token_bytes
        raw = length * raw_token_bytes
    else:
        raise ValueError(action)
    return {
        "projection_tokens": tokens,
        "recomputed_token_layers": token_layers,
        "attention_pair_work": pairs,
        "old_kv_read_bytes": reads,
        "new_kv_write_bytes": writes,
        "raw_history_read_bytes": raw,
    }


def aggregate_logical(lengths: list[tuple[int, int]], actions: list[str]) -> list[dict]:
    exact = [logical_cost(length, "exact_all") for _, length in lengths]
    output = []
    for action in actions:
        rows = [logical_cost(length, action) for _, length in lengths]
        totals = {key: int(sum(row[key] for row in rows)) for key in rows[0]}
        exact_totals = {key: int(sum(row[key] for row in exact)) for key in exact[0]}
        output.append({
            "action": action,
            "states": len(lengths),
            "totals": totals,
            "ratio_to_exact": {
                key: (totals[key] / exact_totals[key] if exact_totals[key] else None)
                for key in totals
            },
        })
    return output


def load_snapshots(uids: list[int], cutover: int, threads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={threads:d}")
    connection.register("selected", pa.table({"uid": pa.array(uids, type=pa.int64())}))
    rows = connection.execute(
        """
        WITH raw AS (
          SELECT row_number() OVER () AS raw_order, * FROM read_parquet(?)
        ), ranked AS (
          SELECT l.uid, l.timestamp, l.item_id, l.is_organic, l.raw_order,
                 row_number() OVER (
                   PARTITION BY l.uid ORDER BY l.timestamp DESC, l.raw_order DESC
                 ) AS reverse_rank
          FROM raw l JOIN selected s USING(uid) WHERE l.timestamp < ?
        )
        SELECT uid, timestamp, item_id, is_organic FROM ranked
        WHERE reverse_rank <= 512 ORDER BY uid, raw_order
        """,
        [str(LISTENS), cutover],
    ).fetchall()
    grouped = {uid: [] for uid in uids}
    for uid, timestamp, item, organic in rows:
        grouped[int(uid)].append((int(timestamp), int(item), int(organic)))
    if any(len(grouped[uid]) != 512 for uid in uids):
        raise RuntimeError("runtime population does not have 512-token snapshots")
    items, behaviors, deltas = [], [], []
    for uid in uids:
        events = grouped[uid]
        timestamps = np.asarray([row[0] for row in events], dtype=np.int64)
        delta = np.zeros(512, dtype=np.float32)
        delta[1:] = np.diff(timestamps).clip(0, 7 * 86_400)
        items.append([row[1] for row in events])
        behaviors.append([1 + (1 - row[2]) for row in events])
        deltas.append(delta)
    return (
        torch.tensor(items, dtype=torch.long),
        torch.tensor(behaviors, dtype=torch.long),
        torch.tensor(np.asarray(deltas), dtype=torch.float32),
    )


def benchmark(contract: dict, lengths: list[tuple[int, int]], device: torch.device, threads: int) -> list[dict]:
    spec = contract["runtime_canary"]
    candidates = [uid for uid, length in lengths if length == 512]
    candidates.sort(key=lambda uid: hashlib.sha256(str(uid).encode()).digest())
    uids = candidates[: int(spec["users"])]
    cpu_tensors = load_snapshots(uids, 231 * 86_400, threads)
    checkpoint = p8raw.TRAIN_ROOT / "r2" / "m0_f_seed17" / "selected.pt"
    current, child = p8raw.load_model(checkpoint, device)
    parent, _ = p8raw.load_model(ROOT / child["parent_checkpoint"], device)
    batch_size = int(spec["batch_size"])
    timings = {action: [] for action in contract["actions"]}
    for start in range(0, len(uids), batch_size):
        tensors = tuple(value[start : start + batch_size].to(device) for value in cpu_tensors)
        parent_cache = parent.compute_kv(*tensors)
        for action in contract["actions"]:
            for repeat in range(int(spec["warmup_batches"]) + int(spec["measured_repetitions"])):
                torch.cuda.synchronize(device)
                begin = time.perf_counter()
                migrated = rolling.migrate(action, current, parent_cache, tensors)
                torch.cuda.synchronize(device)
                elapsed = 1000.0 * (time.perf_counter() - begin)
                if repeat >= int(spec["warmup_batches"]):
                    timings[action].append(elapsed / len(tensors[0]))
                del migrated
    return [{
        "action": action,
        "per_state_ms_mean": float(np.mean(values)),
        "per_state_ms_p50": float(np.median(values)),
        "batch_measurements": len(values),
    } for action, values in timings.items()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), default="cuda:0")
    parser.add_argument("--threads", type=int, default=24)
    args = parser.parse_args()
    contract = validate()
    edge1 = population_lengths(
        231 * 86_400,
        ROOT / "results/p8/staleness_raw/r2/m0_f_seed17/F_fidelity.parquet",
        args.threads,
    )
    edge2 = population_lengths(
        245 * 86_400,
        ROOT / "results/p8/staleness_raw/r1_edge2/m0_f_seed17/F_fidelity.parquet",
        args.threads,
    )
    payload = {
        "status": "P9_6_state_keyed_transition_costs_measured",
        "contract_hash": p7.sha256_file(CONTRACT),
        "populations": {
            "edge1": {"states": len(edge1), "logical_costs": aggregate_logical(edge1, contract["actions"])},
            "edge2": {"states": len(edge2), "logical_costs": aggregate_logical(edge2, contract["actions"])},
        },
        "runtime_canary": benchmark(contract, edge1, torch.device(args.device), args.threads),
        "frontier_authorized": False,
        "scheduler_authorized": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
