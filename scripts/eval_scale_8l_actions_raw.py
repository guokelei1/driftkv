#!/usr/bin/env python3
"""Evaluate frozen legal actions on 8L all-state target-free cutover probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

import eval_p9_cutover_profiler_raw as p9prof
import eval_p9_materialized_lineage_canary as rolling
import eval_scale_8l_hs_raw as hs
import train_p7_theta0 as p7
from hstu_kvcache.models import transition_work

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/scale_8l_method_v1.yaml"
POPULATION = ROOT / "data/manifests/scale_8l_population_v1"
OUTPUT = ROOT / "results/scale_8l_v1/actions_raw"
RELEASE_EDGE = {"r0": "edge1", "r1_edge1": "edge1", "r1_edge2": "edge2", "r2": "edge1"}


def checkpoint(release: str) -> Path:
    return ROOT / "results/scale_8l_v1/releases" / release / "m0_f_seed17/selected.pt"


def validate() -> dict:
    value = yaml.safe_load(CONTRACT.read_text())
    summary = POPULATION / "materialization_summary.json"
    if not summary.exists():
        raise FileNotFoundError("run build_scale_8l_population.py first")
    if p7.sha256_file(ROOT / "src/hstu_kvcache/models/state_transition.py") != value["inputs"]["state_transition_source_sha256"]:
        raise RuntimeError("state transition implementation changed after method freeze")
    return value


def schema() -> pa.Schema:
    return pa.schema([
        ("uid", pa.int64()), ("release", pa.string()), ("action", pa.string()),
        ("candidate_position", pa.int16()), ("candidate_id", pa.int64()),
        ("current_logit", pa.float32()), ("action_logit", pa.float32()),
    ])


@torch.no_grad()
def evaluate(release: str, device: torch.device, limit: int | None, output: Path) -> dict:
    contract = validate(); edge = RELEASE_EDGE[release]
    states = pq.read_table(POPULATION / edge / "states.parquet").to_pylist()
    probes = {int(row["uid"]): row for row in pq.read_table(POPULATION / edge / "cutover_probes.parquet").to_pylist()}
    cutover = 21168000 if edge == "edge2" else 19958400
    for row in states: row["cutover"] = cutover
    if limit is not None:
        states.sort(key=lambda row: hashlib.sha256(f"8l-action:{row['uid']}".encode()).digest())
        states = states[:limit]
    states.sort(key=lambda row: (int(row["effective_prefix_length"]), int(row["raw_prefix_end_exclusive"])))
    current_path = checkpoint(release)
    if not current_path.exists(): raise FileNotFoundError(current_path)
    current, child = hs.load_model(current_path, device)
    if child.get("admitted") is not True: raise RuntimeError("action replay refuses non-admitted release")
    parent_path = ROOT / child["parent_checkpoint"]
    parent, _ = hs.load_model(parent_path, device)
    reader = p9prof.RawStateReader(); actions = tuple(contract["scope"]["actions"])
    groups = {}
    for row in states: groups.setdefault(int(row["effective_prefix_length"]), []).append(row)
    raw_rows, runtime, work = [], {a: 0.0 for a in actions}, {a: {} for a in actions}
    max_exact = max_r0 = 0.0; batch_size = int(contract["profiler"]["state_batch_size"])
    started = time.perf_counter()
    for length in sorted(groups):
        group = groups[length]
        for begin in range(0, len(group), batch_size):
            micro = group[begin:begin + batch_size]
            items, behaviors, deltas, last_timestamps = p9prof.state_tensors(reader, micro, device)
            parent_cache = parent.compute_kv(items, behaviors, deltas)
            current_cache = current.compute_kv(items, behaviors, deltas)
            candidates = torch.tensor([probes[int(row["uid"])]["candidate_ids"] for row in micro], dtype=torch.long, device=device)
            query_delta = torch.tensor([cutover - timestamp for timestamp in last_timestamps], dtype=torch.float32, device=device).clamp(0, 7 * 86400)
            query_type = torch.full((len(micro),), 2, dtype=torch.long, device=device)
            current_score = current.score_cc_reuse(current_cache, candidates, query_delta,
                prefix_lengths=torch.full((len(micro),), length, dtype=torch.long, device=device), query_type_ids=query_type).float()
            for action in actions:
                torch.cuda.synchronize(device); tick = time.perf_counter()
                cache = rolling.migrate(action, current, parent_cache, (items, behaviors, deltas))
                torch.cuda.synchronize(device); runtime[action] += time.perf_counter() - tick
                measured = transition_work(action, parent_cache, items, behaviors, deltas)
                for key, value in measured.__dict__.items(): work[action][key] = work[action].get(key, 0) + int(value)
                score = current.score_cc_reuse(cache, candidates, query_delta,
                    prefix_lengths=torch.full((len(micro),), length, dtype=torch.long, device=device), query_type_ids=query_type).float()
                delta = float((score - current_score).abs().max())
                if action == "exact_all": max_exact = max(max_exact, delta)
                if release == "r0": max_r0 = max(max_r0, delta)
                for index, row in enumerate(micro):
                    for position, candidate in enumerate(probes[int(row["uid"])]["candidate_ids"]):
                        raw_rows.append({"uid": int(row["uid"]), "release": release, "action": action,
                            "candidate_position": position, "candidate_id": int(candidate),
                            "current_logit": float(current_score[index, position]), "action_logit": float(score[index, position])})
                del cache, score
            del parent_cache, current_cache, current_score
    gates = contract["gates"]
    passed = max_exact <= float(gates["exact_max_abs_logit"]) and (release != "r0" or max_r0 <= float(gates["r0_all_action_max_abs_logit"]))
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "cutover_action_scores.parquet"
    pq.write_table(pa.Table.from_pylist(raw_rows, schema=schema()), raw, compression="zstd")
    payload = {"status": "passed_raw_scores_unadjudicated" if passed else "failed", "release": release,
        "model": "m0_f", "seed": 17, "edge": edge, "states": len(states), "actions": list(actions),
        "candidate_rows": len(raw_rows), "contract_sha256": p7.sha256_file(CONTRACT),
        "checkpoint": str(current_path.relative_to(ROOT)), "checkpoint_sha256": p7.sha256_file(current_path),
        "parent_checkpoint": str(parent_path.relative_to(ROOT)), "parent_checkpoint_sha256": p7.sha256_file(parent_path),
        "max_exact_abs_logit": max_exact, "max_r0_action_abs_logit": max_r0,
        "transition_runtime_seconds": runtime, "logical_work": work, "wall_seconds": time.perf_counter() - started,
        "raw": str(raw.relative_to(ROOT)), "raw_sha256": p7.sha256_file(raw), "metrics_computed": False,
        "qualification_or_theta3_read": False}
    (output / "raw_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("status", "release", "states", "max_exact_abs_logit", "max_r0_action_abs_logit", "wall_seconds")}, indent=2))
    if not passed: raise SystemExit(2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=tuple(RELEASE_EDGE), required=True)
    parser.add_argument("--device", choices=tuple(f"cuda:{i}" for i in range(4)), required=True)
    parser.add_argument("--state-limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); suffix = "full" if args.state_limit is None else f"canary{args.state_limit}"
    output = (args.output or OUTPUT / suffix / args.release / "m0_f_seed17").resolve()
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    evaluate(args.release, torch.device(args.device), args.state_limit, output)


if __name__ == "__main__": main()
