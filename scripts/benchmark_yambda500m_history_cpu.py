#!/usr/bin/env python3
"""CPU-only, label-free history-loader parallelism canary for Medium D14."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from hstu_kvcache.data.yambda_history import load_yambda_histories


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/yambda500m_medium_hstu_native_d14_cpu_parallelism_canary_v1.yaml"
DAY = 86_400


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def balanced_users(rows: list[dict], world: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in rows:
        counts[int(row["uid"])] = counts.get(int(row["uid"]), 0) + 1
    loads = [0] * world
    assignment: dict[int, int] = {}
    for uid, count in sorted(counts.items(), key=lambda value: (-value[1], value[0])):
        rank = min(range(world), key=lambda value: (loads[value], value))
        assignment[uid] = rank
        loads[rank] += count
    return assignment


def history_digest(history) -> tuple[str, int]:
    digest = hashlib.sha256()
    events = 0
    for uid in sorted(history.rows):
        digest.update(int(uid).to_bytes(8, "little", signed=False))
        for values in history.rows[uid]:
            digest.update(values.tobytes())
        events += len(history.rows[uid][0])
    return digest.hexdigest(), events


def affinity(contract: dict, rank: int, threads: int) -> list[int]:
    values = list(contract["canary"][f"rank{rank}_numa1_primary_cpus"])
    if threads > len(values):
        values.extend(contract["canary"][f"rank{rank}_optional_sibling_cpus_for_20"])
    return list(map(int, values[:threads]))


def worker(rank: int, threads: int, uids: list[int], contract: dict, queue) -> None:
    cpus = affinity(contract, rank, threads)
    os.sched_setaffinity(0, cpus)
    started = time.perf_counter()
    history = load_yambda_histories(
        ROOT / "data/processed/yambda500m_unified_v1/scales/medium/dataset.json",
        uids, known_vocab_size=1_380_509, oov_buckets=256,
        start_timestamp=231 * DAY, end_timestamp=245 * DAY,
        max_pre_events=1024, threads=threads,
    )
    digest, events = history_digest(history)
    queue.put({
        "rank": rank, "threads": threads, "cpus": cpus,
        "seconds": time.perf_counter() - started, "users": len(uids),
        "events": events, "history_sha256": digest,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    parent = contract["frozen_parent"]
    for key in ("matrix_contract", "execution_contract", "manifest"):
        if sha256(ROOT / parent[key]) != parent[f"{key}_sha256"]:
            raise RuntimeError(f"CPU canary parent hash mismatch: {key}")

    request_path = ROOT / "data/manifests/yambda500m_medium_hstu_native_d7_d14_v1/requests_fidelity.parquet"
    table = pq.read_table(
        request_path,
        filters=[("time_block", "=", "matrix_horizon"), ("target_known", "=", True),
                 ("query_timestamp", ">=", 231 * DAY), ("query_timestamp", "<", 245 * DAY)],
        columns=["uid"],
    )
    rows = [{"uid": int(uid)} for uid in table["uid"].to_pylist()]
    assignment = balanced_users(rows, 2)
    uids_by_rank = [sorted(uid for uid, owner in assignment.items() if owner == rank) for rank in range(2)]

    context = mp.get_context("spawn")
    trials = []
    for index, threads in enumerate(map(int, contract["canary"]["trial_order"])):
        queue = context.Queue()
        processes = [context.Process(target=worker, args=(rank, threads, uids_by_rank[rank], contract, queue)) for rank in range(2)]
        started = time.perf_counter()
        for process in processes:
            process.start()
        values = [queue.get() for _ in processes]
        for process in processes:
            process.join()
            if process.exitcode:
                raise RuntimeError(f"CPU canary worker failed: {process.exitcode}")
        trials.append({
            "trial": index, "threads_per_rank": threads,
            "wall_seconds": time.perf_counter() - started,
            "rank_results": sorted(values, key=lambda value: value["rank"]),
        })
        print(json.dumps(trials[-1], indent=2), flush=True)

    reference = trials[-1]
    reference_by_rank = {value["rank"]: value for value in reference["rank_results"]}
    for trial in trials:
        for value in trial["rank_results"]:
            expected = reference_by_rank[value["rank"]]
            if (value["events"], value["history_sha256"]) != (expected["events"], expected["history_sha256"]):
                raise RuntimeError("history correctness differs across CPU candidates")
    candidates = {trial["threads_per_rank"]: trial for trial in trials if trial["threads_per_rank"] in (14, 20)}
    ratio = abs(candidates[14]["wall_seconds"] - candidates[20]["wall_seconds"]) / min(candidates[14]["wall_seconds"], candidates[20]["wall_seconds"])
    selected = 14 if ratio < 0.05 else min(candidates, key=lambda value: candidates[value]["wall_seconds"])
    selected_trial = candidates[selected]
    output = (ROOT / contract["outputs"]["selection"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "medium_D14_cpu_parallelism_selected",
        "contract": str(contract_path.relative_to(ROOT)), "contract_sha256": sha256(contract_path),
        "quality_metrics_read": False, "gpu_used": False,
        "selected_history_threads_per_rank": selected,
        "selected_arrow_cpu_threads_per_rank": selected,
        "selected_arrow_io_threads_per_rank": 4,
        "selected_omp_num_threads_per_rank": 4,
        "selected_affinity_by_rank": [affinity(contract, rank, selected) for rank in range(2)],
        "warm_reference_four_thread_seconds": reference["wall_seconds"],
        "selected_wall_seconds": selected_trial["wall_seconds"],
        "speedup_over_warm_four_threads": reference["wall_seconds"] / selected_trial["wall_seconds"],
        "trials": trials,
    }
    temporary = output.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
