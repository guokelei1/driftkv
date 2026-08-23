#!/usr/bin/env python3
"""Write raw all-state cutover-probe action scores for one P9.8 cell."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

import eval_p8_release_raw as p8raw
import eval_p9_materialized_lineage_canary as rolling
import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_8_cutover_profiler_contract_v1.yaml"
LISTENS = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
MANIFEST = ROOT / "data/manifests/p9_full_population_v1"
OUTPUT_ROOT = ROOT / "results/p9/cutover_profiler_raw"


class RawStateReader:
    def __init__(self) -> None:
        self.parquet = pq.ParquetFile(LISTENS)
        self.ends = np.cumsum([
            self.parquet.metadata.row_group(index).num_rows
            for index in range(self.parquet.num_row_groups)
        ]).tolist()
        self.group = -1
        self.table = None

    def rows(self, start: int, end: int):
        pieces = []
        cursor = start
        while cursor < end:
            group = bisect.bisect_right(self.ends, cursor)
            group_start = 0 if group == 0 else self.ends[group - 1]
            group_end = self.ends[group]
            if group != self.group:
                self.table = self.parquet.read_row_group(
                    group, columns=["uid", "timestamp", "item_id", "is_organic"]
                )
                self.group = group
            count = min(end, group_end) - cursor
            pieces.append(self.table.slice(cursor - group_start, count))
            cursor += count
        return pieces[0] if len(pieces) == 1 else pa.concat_tables(pieces)


def validate() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p9_7_population_contract_sha256": ROOT / "configs/contracts/p9_7_full_population_contract_v1.yaml",
        "p9_7_population_audit_sha256": ROOT / "results/p9/p9_7_full_population_audit_v1.json",
        "p9_7_uid_canary_sha256": ROOT / "results/p9/p9_7_uid_executor_canary_v1.json",
        "p9_7_population_cost_sha256": ROOT / "results/p9/p9_7_full_population_costs_v1.json",
        "p9_7_materialization_sha256": ROOT / "data/manifests/p9_full_population_v1/materialization_summary.json",
        "state_transition_source_sha256": ROOT / "src/hstu_kvcache/models/state_transition.py",
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P9.8 input hash mismatch: {key}")
    return contract


def state_tensors(reader: RawStateReader, rows: list[dict], device: torch.device):
    items, behaviors, deltas, last_timestamps = [], [], [], []
    for row in rows:
        length = int(row["effective_prefix_length"])
        end = int(row["raw_prefix_end_exclusive"])
        raw = reader.rows(end - length, end)
        uid = raw["uid"].to_numpy(zero_copy_only=False).astype(np.int64)
        timestamps = raw["timestamp"].to_numpy(zero_copy_only=False).astype(np.int64)
        if not np.all(uid == int(row["uid"])):
            raise RuntimeError("P9.8 raw pointer crossed uid")
        if not np.all(timestamps < int(row["cutover"])):
            raise RuntimeError("P9.8 state is not strictly pre-cutover")
        delta = np.zeros(length, dtype=np.float32)
        if length > 1:
            delta[1:] = np.diff(timestamps).clip(0, 7 * 86_400)
        organic = raw["is_organic"].to_numpy(zero_copy_only=False).astype(np.int64)
        items.append(raw["item_id"].to_numpy(zero_copy_only=False).astype(np.int64))
        behaviors.append(1 + (1 - organic))
        deltas.append(delta)
        last_timestamps.append(int(timestamps[-1]))
    return (
        torch.tensor(np.asarray(items), dtype=torch.long, device=device),
        torch.tensor(np.asarray(behaviors), dtype=torch.long, device=device),
        torch.tensor(np.asarray(deltas), dtype=torch.float32, device=device),
        last_timestamps,
    )


def schema() -> pa.Schema:
    return pa.schema([
        ("uid", pa.int64()), ("release", pa.string()), ("model", pa.string()),
        ("seed", pa.int32()), ("action", pa.string()),
        ("candidate_position", pa.int16()), ("candidate_id", pa.int64()),
        ("current_logit", pa.float32()), ("action_logit", pa.float32()),
    ])


@torch.no_grad()
def evaluate(release: str, model_name: str, seed: int, device: torch.device, limit: int | None, output: Path) -> dict:
    contract = validate()
    edge = contract["scope"]["edge_mapping"][release]
    states = pq.read_table(MANIFEST / edge / "states.parquet").to_pylist()
    probes = {
        int(row["uid"]): row
        for row in pq.read_table(MANIFEST / edge / "cutover_probes.parquet").to_pylist()
    }
    for row in states:
        row["cutover"] = 21168000 if edge == "edge2" else 19958400
    if limit is not None:
        states.sort(key=lambda row: hashlib.sha256(str(row["uid"]).encode()).digest())
        states = states[:limit]
    states.sort(key=lambda row: int(row["raw_prefix_end_exclusive"]))
    checkpoint = p8raw.TRAIN_ROOT / release / f"{model_name}_seed{seed}" / "selected.pt"
    current, child = p8raw.load_model(checkpoint, device)
    parent_path = ROOT / child["parent_checkpoint"]
    parent, _ = p8raw.load_model(parent_path, device)
    reader = RawStateReader()
    groups = {}
    for row in states:
        groups.setdefault(int(row["effective_prefix_length"]), []).append(row)
    raw_rows = []
    max_exact = max_r0 = 0.0
    actions = list(contract["scope"]["actions"])
    for length, group in groups.items():
        for start in range(0, len(group), int(contract["execution"]["batch_size"])):
            micro = group[start : start + int(contract["execution"]["batch_size"])]
            items, behaviors, deltas, last_timestamps = state_tensors(reader, micro, device)
            parent_cache = parent.compute_kv(items, behaviors, deltas)
            current_cache = current.compute_kv(items, behaviors, deltas)
            candidates = torch.tensor(
                [probes[int(row["uid"])]["candidate_ids"] for row in micro],
                dtype=torch.long, device=device,
            )
            query_delta = torch.tensor(
                [int(row["cutover"]) - timestamp for row, timestamp in zip(micro, last_timestamps, strict=True)],
                dtype=torch.float32, device=device,
            ).clamp(0, 7 * 86_400)
            query_types = torch.full((len(micro),), 2, dtype=torch.long, device=device)
            current_score = current.score_cc_reuse(
                current_cache, candidates, query_delta,
                prefix_lengths=torch.full((len(micro),), length, dtype=torch.long, device=device),
                query_type_ids=query_types,
            ).float()
            snapshot_tensors = (items, behaviors, deltas)
            for action in actions:
                cache = rolling.migrate(action, current, parent_cache, snapshot_tensors)
                action_score = current.score_cc_reuse(
                    cache, candidates, query_delta,
                    prefix_lengths=torch.full((len(micro),), length, dtype=torch.long, device=device),
                    query_type_ids=query_types,
                ).float()
                delta = float((action_score - current_score).abs().max())
                if action == "exact_all":
                    max_exact = max(max_exact, delta)
                if release == "r0":
                    max_r0 = max(max_r0, delta)
                for index, row in enumerate(micro):
                    for position, candidate in enumerate(probes[int(row["uid"])]["candidate_ids"]):
                        raw_rows.append({
                            "uid": int(row["uid"]), "release": release,
                            "model": model_name, "seed": seed, "action": action,
                            "candidate_position": position, "candidate_id": int(candidate),
                            "current_logit": float(current_score[index, position]),
                            "action_logit": float(action_score[index, position]),
                        })
    gates = contract["gates"]
    passed = (
        max_exact <= float(gates["exact_max_abs_logit"])
        and (release != "r0" or max_r0 <= float(gates["r0_all_action_max_abs_logit"]))
    )
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "cutover_action_scores.parquet"
    pq.write_table(pa.Table.from_pylist(raw_rows, schema=schema()), raw_path, compression="zstd")
    payload = {
        "status": "passed_raw_scores_unadjudicated" if passed else "failed",
        "release": release, "model": model_name, "seed": seed,
        "edge": edge, "states": len(states), "actions": actions,
        "candidate_rows": len(raw_rows), "contract_hash": p7.sha256_file(CONTRACT),
        "checkpoint_hash": p7.sha256_file(checkpoint),
        "parent_checkpoint_hash": p7.sha256_file(parent_path),
        "max_exact_abs_logit": max_exact, "max_r0_action_abs_logit": max_r0,
        "raw_path": str(raw_path.relative_to(ROOT)), "raw_sha256": p7.sha256_file(raw_path),
        "metrics_computed": False,
    }
    (output / "raw_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    if not passed:
        raise RuntimeError(f"P9.8 cell gate failed: {payload}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=("r0", "r1_edge1", "r1_edge2", "r2"), required=True)
    parser.add_argument("--model", choices=("m0_f", "m1"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 37, 71), required=True)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), required=True)
    parser.add_argument("--state-limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suffix = "full" if args.state_limit is None else f"canary{args.state_limit}"
    output = args.output or OUTPUT_ROOT / suffix / args.release / f"{args.model}_seed{args.seed}"
    result = evaluate(
        args.release, args.model, args.seed, torch.device(args.device), args.state_limit, output
    )
    print(json.dumps({key: result[key] for key in (
        "status", "release", "model", "seed", "states", "candidate_rows",
        "max_exact_abs_logit", "max_r0_action_abs_logit",
    )}, indent=2))


if __name__ == "__main__":
    main()
