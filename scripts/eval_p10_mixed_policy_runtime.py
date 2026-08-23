#!/usr/bin/env python3
"""Measure GPU rollout time for one presealed P10 mixed state-action policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

import analyze_p9_transition_costs as p96
import eval_p8_release_raw as p8raw
import eval_p9_cutover_profiler_raw as profiler
import eval_p9_materialized_lineage_canary as rolling
import train_p7_theta0 as p7
from hstu_kvcache.models import HSTUKVCache


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p10_2_mixed_policy_runtime_contract_v1.yaml"
OPTIMIZATION_CONTRACT = ROOT / "configs/contracts/p10_5_executor_batching_optimization_contract_v1.yaml"
POLICY_SEAL = ROOT / "results/p10/p10_0_cheap_profiler_full_seal_v1.json"
POLICY_RESULT = ROOT / "results/p10/p10_0_cheap_profiler_full_v1.json"
P9_10 = ROOT / "results/p9/p9_10_full_population_runtime_v1.json"
POPULATION = ROOT / "data/manifests/p9_full_population_v1"
PROBE_ACTIONS = ("layer0_recent128", "layer0_middle", "layer0_full", "hybrid_tail128", "exact_all")


def validate() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p10_0_policy_seal_sha256": POLICY_SEAL,
        "p10_0_target_free_result_sha256": POLICY_RESULT,
        "p9_10_uniform_runtime_sha256": P9_10,
        "state_transition_source_sha256": ROOT / "src/hstu_kvcache/models/state_transition.py",
        "population_materialization_sha256": POPULATION / "materialization_summary.json",
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P10.2 input hash mismatch: {key}")
    return contract


def subset_cache(cache: HSTUKVCache, indices: torch.Tensor) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k.index_select(1, indices),
        v=cache.v.index_select(1, indices),
        seq_len=cache.seq_len,
    )


def spec_name(release: str, model: str, seed: int, sample: float, budget: float) -> str:
    return f"{release}_{model}_seed{seed}_sample{int(round(sample * 100)):02d}_budget{int(round(budget * 100)):02d}"


def assignment_artifact(release: str, model: str, seed: int) -> Path:
    seal = json.loads(POLICY_SEAL.read_text())
    match = [row for row in seal["artifacts"] if row["release"] == release and row["model"] == model and int(row["seed"]) == seed]
    if len(match) != 1:
        raise RuntimeError("sealed assignment artifact not unique")
    path = ROOT / match[0]["assignments"]
    if p7.sha256_file(path) != match[0]["assignments_sha256"]:
        raise RuntimeError("sealed assignment changed")
    return path


def checkpoint_path(release: str, model: str, seed: int) -> Path:
    return p8raw.TRAIN_ROOT / release / f"{model}_seed{seed}" / "selected.pt"


def subset_tensors(tensors, indices):
    return tuple(value.index_select(0, indices) for value in tensors)


def add_logical(totals: dict, length: int, action: str, count: int) -> None:
    values = p96.logical_cost(length, action)
    for key, value in values.items():
        totals[key] += int(value) * count


def build_groups(states: list[dict], assignment_by_uid: dict, mode: str) -> dict[tuple[int, str], list[dict]]:
    groups: dict[tuple[int, str], list[dict]] = {}
    for row in states:
        length = int(row["effective_prefix_length"])
        if mode == "reference":
            signature = "mixed"
        else:
            assignment = assignment_by_uid[int(row["uid"])]
            signature = "probe" if bool(assignment.calibration_sample) else str(assignment.action)
            if signature == "noop":
                continue
        groups.setdefault((length, signature), []).append(row)
    return groups


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True)
    parser.add_argument("--model", choices=("m0_f", "m1"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-fraction", type=float, required=True)
    parser.add_argument("--budget-fraction", type=float, required=True)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), required=True)
    parser.add_argument("--state-limit", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--batching-mode", choices=("reference", "grouped"), default="reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = validate()
    optimization_contract = yaml.safe_load(OPTIMIZATION_CONTRACT.read_text()) if args.batching_mode == "grouped" else None
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    edge = "edge2" if args.release == "r1_edge2" else "edge1"
    assignments = pq.read_table(assignment_artifact(args.release, args.model, args.seed)).to_pandas()
    assignments = assignments[
        np.isclose(assignments["sample_fraction"], args.sample_fraction)
        & np.isclose(assignments["budget_fraction"], args.budget_fraction)
    ]
    if assignments["uid"].duplicated().any():
        raise RuntimeError("sealed policy has multiple actions per uid")
    assignment_by_uid = {int(row.uid): row for row in assignments.itertuples(index=False)}
    states = pq.read_table(POPULATION / edge / "states.parquet").to_pylist()
    for row in states:
        row["cutover"] = 21168000 if edge == "edge2" else 19958400
    if args.state_limit is not None:
        states.sort(key=lambda row: hashlib.sha256(str(row["uid"]).encode()).digest())
        states = states[: args.state_limit]
    states.sort(key=lambda row: int(row["raw_prefix_end_exclusive"]))
    if any(int(row["uid"]) not in assignment_by_uid for row in states):
        raise RuntimeError("population uid absent from sealed assignment")
    device = torch.device(args.device)
    checkpoint = checkpoint_path(args.release, args.model, args.seed)
    current, child = p8raw.load_model(checkpoint, device)
    parent, _ = p8raw.load_model(ROOT / child["parent_checkpoint"], device)
    reader = profiler.RawStateReader()
    groups = build_groups(states, assignment_by_uid, args.batching_mode)
    repetitions = args.repetitions or int(contract["execution"]["repetitions"])
    points = [0.0] * repetitions
    operation_state_counts = {action: 0 for action in PROBE_ACTIONS}
    operation_batch_counts = {action: 0 for action in PROBE_ACTIONS}
    logical = {key: 0 for key in p96.logical_cost(1, "noop")}
    raw_seconds = h2d_seconds = parent_seconds = 0.0
    warmed = False
    started = time.perf_counter()
    for group_key in sorted(groups):
        length, _ = group_key
        group = groups[group_key]
        for begin_index in range(0, len(group), int(contract["execution"]["batch_size"])):
            micro = group[begin_index : begin_index + int(contract["execution"]["batch_size"])]
            begin = time.perf_counter()
            items, behaviors, deltas, _ = profiler.state_tensors(reader, micro, torch.device("cpu"))
            raw_seconds += time.perf_counter() - begin
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            tensors = tuple(value.to(device) for value in (items, behaviors, deltas))
            torch.cuda.synchronize(device)
            h2d_seconds += time.perf_counter() - begin
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            parent_cache = parent.compute_kv(*tensors)
            torch.cuda.synchronize(device)
            parent_seconds += time.perf_counter() - begin
            if not warmed:
                for action in PROBE_ACTIONS:
                    migrated = rolling.migrate(action, current, parent_cache, tensors)
                    del migrated
                torch.cuda.synchronize(device)
                warmed = True
            sampled_mask = torch.tensor(
                [bool(assignment_by_uid[int(row["uid"])].calibration_sample) for row in micro],
                dtype=torch.bool, device=device,
            )
            operation_indices = {}
            if sampled_mask.any():
                sampled_indices = torch.nonzero(sampled_mask, as_tuple=False).flatten()
                for action in PROBE_ACTIONS:
                    operation_indices.setdefault(action, []).append(sampled_indices)
            for action in PROBE_ACTIONS:
                chosen = torch.tensor([
                    (not bool(assignment_by_uid[int(row["uid"])].calibration_sample))
                    and str(assignment_by_uid[int(row["uid"])].action) == action
                    for row in micro
                ], dtype=torch.bool, device=device)
                if chosen.any():
                    operation_indices.setdefault(action, []).append(torch.nonzero(chosen, as_tuple=False).flatten())
            merged_indices = {
                action: torch.cat(values) for action, values in operation_indices.items()
            }
            for action, indices in merged_indices.items():
                count = int(indices.numel())
                operation_state_counts[action] += count
                operation_batch_counts[action] += 1
                add_logical(logical, length, action, count)
            for repetition in range(repetitions):
                torch.cuda.synchronize(device)
                begin = time.perf_counter()
                for action, indices in merged_indices.items():
                    cache_subset = subset_cache(parent_cache, indices)
                    tensor_subset = subset_tensors(tensors, indices)
                    migrated = rolling.migrate(action, current, cache_subset, tensor_subset)
                    del migrated, cache_subset, tensor_subset
                torch.cuda.synchronize(device)
                points[repetition] += time.perf_counter() - begin
            del parent_cache, tensors
    selected = assignments[assignments["uid"].isin([int(row["uid"]) for row in states])]
    expected_token_layers = float(selected["action_cost_token_layers"].sum())
    sampled = selected[selected["calibration_sample"]]
    for action in PROBE_ACTIONS:
        expected_token_layers += sum(
            p96.logical_cost(int(next(row["effective_prefix_length"] for row in states if int(row["uid"]) == int(uid))), action)["recomputed_token_layers"]
            for uid in sampled["uid"]
        )
    expected_token_layers -= float(sampled["action_cost_token_layers"].sum())
    if int(round(expected_token_layers)) != int(logical["recomputed_token_layers"]):
        raise RuntimeError(f"logical work mismatch: expected={expected_token_layers}, actual={logical['recomputed_token_layers']}")
    exact_total = sum(4 * int(row["effective_prefix_length"]) for row in states)
    payload = {
        "status": "P10_2_mixed_policy_runtime_measured",
        "condition": {
            "name": spec_name(args.release, args.model, args.seed, args.sample_fraction, args.budget_fraction),
            "release": args.release, "model": args.model, "seed": args.seed,
            "edge": edge, "sample_fraction": args.sample_fraction, "budget_fraction": args.budget_fraction,
        },
        "scope": "canary" if args.state_limit is not None else "full",
        "batching_mode": args.batching_mode,
        "states": len(states), "sampled_states": int(selected["calibration_sample"].sum()),
        "repetitions": repetitions,
        "mixed_rollout_seconds_points": points,
        "mixed_rollout_seconds_median": float(np.median(points)),
        "per_state_ms_median": float(1000 * np.median(points) / len(states)),
        "operation_state_counts": operation_state_counts,
        "operation_batch_counts": operation_batch_counts,
        "logical_totals": logical,
        "logical_token_layer_fraction_of_Exact": logical["recomputed_token_layers"] / exact_total,
        "raw_reconstruction_seconds": raw_seconds,
        "raw_history_H2D_seconds": h2d_seconds,
        "parent_cache_reference_build_seconds_excluded": parent_seconds,
        "total_benchmark_wall_seconds": time.perf_counter() - started,
        "contract_sha256": p7.sha256_file(CONTRACT),
        "optimization_contract_sha256": p7.sha256_file(OPTIMIZATION_CONTRACT) if optimization_contract else None,
        "evaluator_sha256": p7.sha256_file(Path(__file__)),
        "policy_assignment_sha256": p7.sha256_file(assignment_artifact(args.release, args.model, args.seed)),
        "checkpoint_sha256": p7.sha256_file(checkpoint),
        "quality_labels_read": False,
        "controller_authorized": False,
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"], "condition": payload["condition"]["name"],
        "scope": payload["scope"], "states": payload["states"],
        "logical_fraction": payload["logical_token_layer_fraction_of_Exact"],
        "rollout_seconds": payload["mixed_rollout_seconds_median"],
    }, indent=2))


if __name__ == "__main__":
    main()
