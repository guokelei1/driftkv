#!/usr/bin/env python3
"""Numerically compare per-state reference and operation-grouped final caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
import yaml

import eval_p8_release_raw as p8raw
import eval_p9_cutover_profiler_raw as profiler
import eval_p9_materialized_lineage_canary as rolling
import eval_p10_mixed_policy_runtime as runtime
import train_p7_theta0 as p7
from hstu_kvcache.models import HSTUKVCache


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p10_5_executor_batching_optimization_contract_v1.yaml"
OUTPUT = ROOT / "results/p10/p10_5_batching_equivalence_canary_v1.json"


def final_cache(action: str, sampled: bool, current, parent, tensors) -> HSTUKVCache:
    if sampled:
        output = None
        for probe_action in runtime.PROBE_ACTIONS:
            output = rolling.migrate(probe_action, current, parent, tensors)
        assert output is not None
        return output
    return rolling.migrate(action, current, parent, tensors)


@torch.no_grad()
def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    contract = yaml.safe_load(CONTRACT.read_text())
    spec = contract["numeric_canary"]["condition"]
    device = torch.device("cuda:0")
    assignment_path = runtime.assignment_artifact(spec["release"], spec["model"], int(spec["seed"]))
    assignments = pq.read_table(assignment_path).to_pandas()
    assignments = assignments[
        (assignments["sample_fraction"] == float(spec["sample_fraction"]))
        & (assignments["budget_fraction"] == float(spec["budget_fraction"]))
    ]
    assignment_by_uid = {int(row.uid): row for row in assignments.itertuples(index=False)}
    states = pq.read_table(runtime.POPULATION / "edge1/states.parquet").to_pylist()
    states.sort(key=lambda row: hashlib.sha256(str(row["uid"]).encode()).digest())
    states = states[: int(contract["numeric_canary"]["states"])]
    for row in states:
        row["cutover"] = 19958400
    checkpoint = runtime.checkpoint_path(spec["release"], spec["model"], int(spec["seed"]))
    current, child = p8raw.load_model(checkpoint, device)
    parent, _ = p8raw.load_model(ROOT / child["parent_checkpoint"], device)
    reader = profiler.RawStateReader()
    reference = {}
    for row in states:
        tensors = profiler.state_tensors(reader, [row], device)[:3]
        parent_cache = parent.compute_kv(*tensors)
        assignment = assignment_by_uid[int(row["uid"])]
        output = final_cache(str(assignment.action), bool(assignment.calibration_sample), current, parent_cache, tensors)
        reference[int(row["uid"])] = output.to("cpu")
    groups = {}
    for row in states:
        assignment = assignment_by_uid[int(row["uid"])]
        signature = "probe" if bool(assignment.calibration_sample) else str(assignment.action)
        groups.setdefault((int(row["effective_prefix_length"]), signature), []).append(row)
    max_k = max_v = 0.0
    compared = []
    for key in sorted(groups):
        group = groups[key]
        for begin in range(0, len(group), int(contract["optimization"]["batch_size"])):
            micro = group[begin : begin + int(contract["optimization"]["batch_size"])]
            tensors = profiler.state_tensors(reader, micro, device)[:3]
            parent_cache = parent.compute_kv(*tensors)
            assignment = assignment_by_uid[int(micro[0]["uid"])]
            output = final_cache(str(assignment.action), bool(assignment.calibration_sample), current, parent_cache, tensors).to("cpu")
            for index, row in enumerate(micro):
                uid = int(row["uid"])
                expected = reference[uid]
                max_k = max(max_k, float((output.k[:, index : index + 1] - expected.k).abs().max()))
                max_v = max(max_v, float((output.v[:, index : index + 1] - expected.v).abs().max()))
                compared.append(uid)
    tolerance = float(contract["numeric_canary"]["max_abs_tolerance"])
    if sorted(compared) != sorted(reference):
        raise RuntimeError("grouped batching lost or duplicated uid")
    if max(max_k, max_v) > tolerance:
        raise RuntimeError(f"batching numeric mismatch: K={max_k}, V={max_v}, tolerance={tolerance}")
    payload = {
        "status": "P10_5_grouped_batching_numeric_equivalence_passed",
        "states": len(states),
        "max_abs_K_difference": max_k,
        "max_abs_V_difference": max_v,
        "tolerance": tolerance,
        "contract_sha256": p7.sha256_file(CONTRACT),
        "assignment_sha256": p7.sha256_file(assignment_path),
        "checkpoint_sha256": p7.sha256_file(checkpoint),
        "policy_assignment_changed": False,
        "quality_labels_read": False,
        "full_runtime_replay_authorized": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
