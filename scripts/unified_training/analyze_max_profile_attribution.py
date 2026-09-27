#!/usr/bin/env python3
"""CPU-only attribution of retained Max traces; no weights or data payloads read.

Run from the repository root with: python <this script>
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "results/backend_acceleration/max_profile_2026_09_19/attribution"
MEASUREMENT = HERE.parent / "probe/measurement"
GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
EMBEDDING = "autograd::engine::evaluate_function: EmbeddingBackward0"
SCOPES = {
    "embedding_backward_and_gradient_accumulation": EMBEDDING,
    "prefix_forward": "model::prefix",
    "query_forward": "model::cc_queries",
    "optimizer": "Optimizer.step#AdamW.step",
}


def read_json(path, inputs):
    payload = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def cpu_ancestry(events):
    """Map GPU External id to its CPU launch operator and nested CPU scopes."""
    threads = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in {"cpu_op", "user_annotation"}:
            threads[event["tid"]].append(event)
    ancestry = {}
    for thread_events in threads.values():
        stack = []
        for event in sorted(thread_events, key=lambda row: (row["ts"], -row["dur"])):
            while stack and stack[-1]["ts"] + stack[-1]["dur"] <= event["ts"]:
                stack.pop()
            external_id = event.get("args", {}).get("External id")
            if external_id is not None:
                ancestry[external_id] = [row["name"] for row in stack] + [event["name"]]
            stack.append(event)
    return ancestry


def scope_work(events, ancestry, name, steps):
    selected = [event for event in events
                if event.get("cat") in GPU_CATEGORIES and event.get("ph") == "X"
                and name in ancestry.get(event.get("args", {}).get("External id"), [])]
    operators = defaultdict(lambda: {"gpu_activity_count": 0, "sum_gpu_activity_ms": 0.0})
    for event in selected:
        operator = ancestry[event["args"]["External id"]][-1]
        operators[operator]["gpu_activity_count"] += 1
        operators[operator]["sum_gpu_activity_ms"] += event["dur"] / 1000
    total = sum(event["dur"] for event in selected) / 1000
    kernel_total = sum(event["dur"] for event in selected if event["cat"] == "kernel") / 1000
    return {
        "cpu_scope_calls": sum(event.get("name") == name and event.get("ph") == "X"
                               and event.get("cat") in {"cpu_op", "user_annotation"} for event in events),
        "gpu_activity_counts": dict(Counter(event["cat"] for event in selected)),
        "sum_gpu_activity_ms_all_profile_steps": total,
        "mean_gpu_activity_ms_per_profile_step": total / steps,
        "sum_kernel_only_ms_all_profile_steps": kernel_total,
        "mean_kernel_only_ms_per_profile_step": kernel_total / steps,
        **({"leaf_cpu_operators": dict(sorted(operators.items()))} if name == EMBEDDING else {}),
    }


def main():
    inputs = {}
    fixture = read_json(HERE.parent / "fixture/configuration.json", inputs)
    fixture_ranks = {rank["rank"]: rank for rank in fixture["ranks"]}
    cells = []
    for variant in ("torch_default", "auto_default", "auto_fused"):
        source = read_json(MEASUREMENT / f"{variant}.summary.json", inputs)
        ranks = []
        for rank in source["ranks"]:
            rank_id = rank["rank"]
            profile = rank["profile"]
            steps = profile["profile_steps"]
            batches = [row["fixture_batch"] for row in profile["rows"]]
            path = MEASUREMENT / f"{variant}.rank{rank_id}.trace.json"
            events = read_json(path, inputs)["traceEvents"]
            ancestry = cpu_ancestry(events)
            scopes = {key: scope_work(events, ancestry, name, steps) for key, name in SCOPES.items()}
            # Cross-check the dense-gradient attribution against profiler aggregation.
            for key, name in SCOPES.items():
                expected = next(row for row in profile["selected_inclusive_operators"] if row["name"] == name)
                actual = scopes[key]["sum_gpu_activity_ms_all_profile_steps"]
                scopes[key]["profiler_inclusive_gpu_ms_all_profile_steps"] = expected["total_device_ms"]
                scopes[key]["difference_from_profiler_inclusive_ms"] = actual - expected["total_device_ms"]
                if key == "embedding_backward_and_gradient_accumulation" and abs(actual - expected["total_device_ms"]) > 0.01:
                    raise RuntimeError(f"Scope attribution differs: {variant} rank{rank_id} {key}")
            collectives = {}
            for kind, marker in (("all_gather", "AllGather"), ("reduce_scatter", "ReduceScatter")):
                found = sorted((event for event in events if event.get("cat") == "kernel"
                                and "nccl" in event["name"] and marker in event["name"]), key=lambda row: row["ts"])
                if len(found) != steps:
                    raise RuntimeError(f"Expected one {kind} per step")
                collectives[kind] = [
                    {"fixture_batch": batch, "start_trace_us": event["ts"],
                     "end_trace_us": event["ts"] + event["dur"], "duration_ms": event["dur"] / 1000,
                     "kernel_name": event["name"]}
                    for batch, event in zip(batches, found, strict=True)
                ]
            ranks.append({
                "rank": rank_id, "profile_steps": steps, "scopes": scopes,
                "matched_fixture_groups": [{"fixture_batch": batch,
                    "group_count": len(fixture_ranks[rank_id]["batches"][batch]["length_groups"]),
                    "length_groups": fixture_ranks[rank_id]["batches"][batch]["length_groups"]} for batch in batches],
                "collective_count_all_profile_steps": {kind: len(rows) for kind, rows in collectives.items()},
                "collectives": collectives, "timeline_ms_all_profile_steps": profile["timeline"],
            })
        cells.append({"variant": variant,
                      "clean_max_rank_wall_ms": source["max_rank_wall"],
                      "clean_max_rank_cuda_phase_ms": source["max_rank_cuda_phase"], "ranks": ranks})
    summary = {
        "status": "complete_cpu_only_trace_attribution",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": inputs,
        "scope": {
            "clean_timing": "Original phase intervals and wall medians; excludes profiler overhead and original CPU collate/startup/checkpoints.",
            "gpu_work": "Sum of linked GPU activity durations via CPU External id and nested CPU scope; includes kernels, memcpy and memset. Kernel-only totals are separate. These are work sums, not wall-clock intervals; not additive across nested scopes.",
            "attribution_check": "Each scope retains the original profiler inclusive total and extraction difference. Minor boundary/nesting discrepancies can occur; dense-gradient attribution must agree within0.01ms over all recorded steps.",
            "units": "Trace start/end timestamps are microseconds on the trace clock. Every field ending all_profile_steps is TOTAL over the recorded two steps. Per-profile-step fields divide that total by two.",
            "prefix": "GPU work within prefix forward only; excludes FSDP collectives outside that scope and all backward/query/optimizer work.",
            "embedding": "All EmbeddingBackward scopes, including dense zero/fill and gradient accumulation; includes small behavior/type/action tables as well as the large item table.",
            "collectives": "One all-gather and one reduce-scatter per recorded step. NCCL kernel durations include peer waiting and are not pure wire transmission. Cross-rank clocks are shared on this single host.",
            "overlap": "Retained timeline union metrics distinguish summed work from elapsed span; these traces report zero communication/noncommunication overlap.",
            "limits": "Two profiled batches (4 and 9) from the first twelve chronological startup batches, not a whole-epoch profile or a quality experiment. Profiler timing is not substituted for clean benchmark timing.",
        },
        "cells": cells,
    }
    output = HERE / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    main()
