#!/usr/bin/env python3
"""Summarize the bounded Max training profile without importing torch or reading quality results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    return {"median": statistics.median(values), "mean": statistics.mean(values),
            "min": min(values), "max": max(values), "count": len(values)}


def difference(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError("Cannot compare differently sized numerical samples")
    errors = [right - left for left, right in zip(reference, candidate, strict=True)]
    reference_squared = sum(value * value for value in reference)
    error_squared = sum(value * value for value in errors)
    return {"count": len(errors), "finite": all(math.isfinite(value) for value in reference + candidate),
            "max_absolute": max(map(abs, errors), default=0.0),
            "rms": math.sqrt(error_squared / max(len(errors), 1)),
            "relative_l2": math.sqrt(error_squared / max(reference_squared, 1e-30)),
            "exactly_equal_count": sum(left == right for left, right in zip(reference, candidate, strict=True))}


def summarize_rank(rank):
    clean = [row for row in rank["rows"] if not row["warmup"]]
    phases = clean[0]["cuda_interval_ms"]
    return {"rank": rank["rank"], "wall_ms": distribution([row["wall_ms"] for row in clean]),
            "cuda_interval_ms": {name: distribution([row["cuda_interval_ms"][name] for row in clean]) for name in phases},
            "cpu_enqueue_ms": {name: distribution([row["cpu_enqueue_ms"][name] for row in clean]) for name in phases},
            "peak_allocated_mib": rank["peak_allocated_mib"], "peak_reserved_mib": rank["peak_reserved_mib"]}


def summarize_trace(profile, top):
    trace_path = Path(profile["trace_path"])
    if not trace_path.is_absolute():
        trace_path = ROOT / trace_path
    kernels_path = trace_path.with_name(trace_path.stem + ".kernels.json")
    kernels = json.loads(kernels_path.read_text())
    trace = json.loads(trace_path.read_text())
    collective_scopes = defaultdict(int)
    for event in trace["traceEvents"]:
        name = event.get("name", "")
        if event.get("ph") == "X" and event.get("cat") in ("cpu_op", "user_annotation") and any(
            word in name.lower() for word in ("all_gather", "allgather", "reduce_scatter", "reducescatter")
        ):
            collective_scopes[name] += 1
    del trace
    groups = defaultdict(lambda: {"calls": 0, "sum_ms": 0.0, "longest_ms": 0.0})
    for kernel in kernels:
        group = groups[(kernel["category"], kernel["name"])]
        duration = kernel["duration_us"] / 1000
        group["calls"] += 1
        group["sum_ms"] += duration
        group["longest_ms"] = max(group["longest_ms"], duration)
    steps = profile["profile_steps"]
    top_kernels = [{"category": category, "name": name, **values,
                    "sum_ms_per_profile_step": values["sum_ms"] / steps}
                   for (category, name), values in groups.items()]
    top_kernels.sort(key=lambda row: row["sum_ms"], reverse=True)
    timeline = profile["timeline"]
    scoped = []
    for operator in profile["selected_inclusive_operators"]:
        if operator["name"].startswith(("phase::", "model::")) or any(
            text in operator["name"].lower() for text in ("adam", "embedding", "all_gather", "reduce_scatter")
        ):
            scoped.append({"name": operator["name"], "calls": operator["calls"],
                           "inclusive_cpu_ms_per_step": operator["total_cpu_ms"] / steps,
                           "inclusive_device_work_ms_per_step": operator["total_device_ms"] / steps,
                           "self_device_work_ms_per_step": operator["self_device_ms"] / steps})
    return {"profile_steps": steps, "fixture_batches": [row["fixture_batch"] for row in profile["rows"]],
            "kernel_file": str(kernels_path), "kernel_file_sha256": digest(kernels_path),
            "collective_cpu_scope_counts": {name: {"calls": count, "calls_per_step": count / steps}
                                             for name, count in collective_scopes.items()},
            "timeline_ms_per_profile_step": {name: value / steps for name, value in timeline.items()},
            "communication_without_noncommunication_ms_per_step": (
                timeline["communication_union_ms"] - timeline["communication_noncommunication_overlap_ms"]
            ) / steps,
            "kernel_categories_work_ms_per_step": {name: values["sum_kernel_ms"] / steps
                                                    for name, values in profile["categories"].items()},
            "top_kernels": top_kernels[:top], "selected_inclusive_operators": scoped,
            "top_cpu_operators_by_self_device_work": profile["top_operators_by_self_device_ms"][:top]}


def summarize_cell(cell, top):
    ranks = sorted(cell["ranks"], key=lambda row: row["rank"])
    clean_by_rank = [[row for row in rank["rows"] if not row["warmup"]] for rank in ranks]
    clean_steps = len(clean_by_rank[0])
    if any(len(rows) != clean_steps for rows in clean_by_rank):
        raise ValueError("Ranks have different clean timing step counts")
    max_rank_wall = []
    critical_ranks = []
    for step in range(clean_steps):
        row_identity = {(rows[step]["step"], rows[step]["fixture_batch"]) for rows in clean_by_rank}
        if len(row_identity) != 1:
            raise ValueError("Ranks have different fixture step identities")
        slowest = max(range(len(ranks)), key=lambda rank: clean_by_rank[rank][step]["wall_ms"])
        max_rank_wall.append(clean_by_rank[slowest][step]["wall_ms"])
        critical_ranks.append(ranks[slowest]["rank"])
    if max_rank_wall != cell["max_rank_wall_ms_by_step"]:
        raise ValueError("Recomputed per-step rank maxima differ from the retained source")
    global_losses = [statistics.mean([rank["rows"][step]["loss"] for rank in ranks])
                     for step in range(len(ranks[0]["rows"]))]
    result = {"variant": cell["variant"], "max_rank_wall_ms": distribution(max_rank_wall),
              "max_rank_wall_ms_by_step": max_rank_wall, "critical_rank_by_step": critical_ranks,
              "fixture_batches": [row["fixture_batch"] for row in clean_by_rank[0]],
              "peak_allocated_mib": max(rank["peak_allocated_mib"] for rank in ranks),
              "peak_reserved_mib": max(rank["peak_reserved_mib"] for rank in ranks),
              "per_rank": [summarize_rank(rank) for rank in ranks],
              "global_weighted_loss_by_step": global_losses,
              "loss_note": "Mean of rank-local losses; original loss scales each local weighted sum by world/global_batch.",
              "optimizer_defaults": ranks[0]["optimizer_defaults"],
              "all_execution_source_checks_passed": all(rank["source_hashes_unchanged"] for rank in ranks)}
    result["profiles"] = [{"rank": rank["rank"], **summarize_trace(rank["profile"], top)}
                          for rank in ranks if "profile" in rank]
    return result


def compare_cells(reference, candidate):
    if reference["fixture_batches"] != candidate["fixture_batches"]:
        raise ValueError("Compared cells have different timed fixture batches")
    result = {"reference": reference["variant"], "candidate": candidate["variant"]}
    for statistic in ("median", "mean"):
        old, new = reference["max_rank_wall_ms"][statistic], candidate["max_rank_wall_ms"][statistic]
        result[statistic] = {"speedup": old / new, "time_reduction_percent": 100 * (1 - new / old),
                             "saved_ms": old - new}
    result["paired_step_speedup"] = distribution([old / new for old, new in zip(
        reference["max_rank_wall_ms_by_step"], candidate["max_rank_wall_ms_by_step"], strict=True)])
    result["weighted_loss_trajectory_difference"] = difference(reference["global_weighted_loss_by_step"],
                                                              candidate["global_weighted_loss_by_step"])
    return result


def compare_numerics(reference, candidate):
    left_ranks = {rank["rank"]: rank for rank in reference["ranks"]}
    right_ranks = {rank["rank"]: rank for rank in candidate["ranks"]}
    if left_ranks.keys() != right_ranks.keys():
        raise ValueError("Compared cells have different ranks")
    logits_left, logits_right, samples_left, samples_right, per_parameter = [], [], [], [], []
    for rank in sorted(left_ranks):
        left, right = left_ranks[rank], right_ranks[rank]
        logits_left.extend(left["initial_logits"])
        logits_right.extend(right["initial_logits"])
        left_samples, right_samples = left["final_local_parameter_samples"], right["final_local_parameter_samples"]
        if left_samples.keys() != right_samples.keys():
            raise ValueError("Compared cells have different sampled local parameters")
        for name in left_samples:
            values_left, values_right = left_samples[name], right_samples[name]
            samples_left.extend(values_left)
            samples_right.extend(values_right)
            per_parameter.append({"rank": rank, "name": name, **difference(values_left, values_right)})
    per_parameter.sort(key=lambda row: row["max_absolute"], reverse=True)
    return {"initial_logits": difference(logits_left, logits_right),
            "final_parameter_samples": difference(samples_left, samples_right),
            "largest_sampled_parameter_differences": per_parameter[:8],
            "parameter_scope": "At most16 deterministic values per nonempty local FSDP parameter shard after clean steps, before profiler updates; not a full parameter equality check.",
            "claim": "Numerical differences are reported without an automatic quality or long-term training-equivalence claim."}


def markdown(report):
    lines = ["# Max training performance diagnostic", "", report["timing_scope"], "",
             "| Variant | Max-rank step median (ms) | Mean (ms) | Peak allocated (MiB) | Peak reserved (MiB) |",
             "|---|---:|---:|---:|---:|"]
    for cell in report["cells"]:
        wall = cell["max_rank_wall_ms"]
        lines.append(f"| {cell['variant']} | {wall['median']:.2f} | {wall['mean']:.2f} | {cell['peak_allocated_mib']:.0f} | {cell['peak_reserved_mib']:.0f} |")
    lines += ["", "| Comparison | Median speedup | Median time reduction | Mean speedup |",
              "|---|---:|---:|---:|"]
    for pair in report["comparisons"]:
        lines.append(f"| {pair['reference']} → {pair['candidate']} | {pair['median']['speedup']:.3f}× | {pair['median']['time_reduction_percent']:.2f}% | {pair['mean']['speedup']:.3f}× |")
    lines += ["", "Per-rank CUDA interval medians are shown below. Phases from different ranks, or separate phase medians, must not be added to reconstruct a whole-step wall time.", "",
              "| Variant | Rank | H2D | Forward | Loss | Zero grad | Backward | Optimizer | Wall median |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in report["cells"]:
        for rank in cell["per_rank"]:
            values = [rank["cuda_interval_ms"][phase]["median"] for phase in
                      ("data_H2D", "forward", "loss", "zero_grad", "backward", "optimizer")]
            lines.append(f"| {cell['variant']} | {rank['rank']} | " + " | ".join(f"{value:.2f}" for value in values) + f" | {rank['wall_ms']['median']:.2f} |")
    lines += ["", "Profiler values below are milliseconds per profiled step. They describe the separate instrumented trace, not the clean timing run; kernel work may overlap across streams.", "",
              "| Variant | Rank | Communication union | Noncommunication union | Overlap | Communication without concurrent noncommunication |",
              "|---|---:|---:|---:|---:|---:|"]
    for cell in report["cells"]:
        for profile in cell["profiles"]:
            timeline = profile["timeline_ms_per_profile_step"]
            lines.append(f"| {cell['variant']} | {profile['rank']} | {timeline['communication_union_ms']:.2f} | {timeline['noncommunication_union_ms']:.2f} | {timeline['communication_noncommunication_overlap_ms']:.2f} | {profile['communication_without_noncommunication_ms_per_step']:.2f} |")
    lines += ["", "NCCL durations include waiting for other ranks and are not pure network transfer time. Top kernels, collective CPU scope call counts, and inclusive prefix/query/embedding/operator work are retained per rank in summary.json. Inclusive operator totals and nested collective call counts cannot be added.", "",
              "| Numerical comparison | Initial logits max abs | Global loss trajectory max abs | Parameter sample max abs | Sample relative L2 |",
              "|---|---:|---:|---:|---:|"]
    for pair in report["comparisons"]:
        numeric = pair["numerics"]
        lines.append(f"| {pair['reference']} → {pair['candidate']} | {numeric['initial_logits']['max_absolute']:.6g} | {pair['weighted_loss_trajectory_difference']['max_absolute']:.6g} | {numeric['final_parameter_samples']['max_absolute']:.6g} | {numeric['final_parameter_samples']['relative_l2']:.6g} |")
    lines += ["", "Parameter samples cover only a small deterministic subset of local shards; they do not establish full-state or long-term training equivalence.", "",
              f"Model context: {report['parameter_layout_context']['total_parameters']:,} total parameters, of which the item embedding has {report['parameter_layout_context']['item_embedding_parameters']:,} ({report['parameter_layout_context']['item_embedding_percent']:.2f}%). Parameter fraction is not a runtime fraction.", "",
              f"Original CPU collate median: {report['cpu_collate_ms']['median']:.3f} ms per local batch (prepared separately; excluded from clean step times).", "",
              "No evaluation quality metrics were opened or used. These fixtures reproduce the first chronological batches of the earlier backend canary and do not establish later-window performance.", "",
              f"Input summary SHA256: `{report['input_summary_sha256']}`", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    source = json.loads(args.summary.read_text())
    cells = [summarize_cell(cell, args.top) for cell in source["cells"]]
    indexed = {cell["variant"]: cell for cell in cells}
    originals = {cell["variant"]: cell for cell in source["cells"]}
    pairs = [("torch_default", "auto_default"), ("auto_default", "auto_fused"), ("torch_default", "auto_fused")]
    comparisons = []
    for left, right in pairs:
        if left in indexed and right in indexed:
            pair = compare_cells(indexed[left], indexed[right])
            pair["numerics"] = compare_numerics(originals[left], originals[right])
            comparisons.append(pair)
    report = {"status": "complete_analysis" if len(cells) == len(source["variants"]) else "partial_cells_analysis",
              "input_summary": str(args.summary.resolve()), "input_summary_sha256": digest(args.summary),
              "analysis_script_sha256": digest(Path(__file__)), "timing_scope": source["timing_scope"],
              "cuda_phase_scope": source["cuda_phase_scope"], "warmup_steps": source["warmup_steps"],
              "timed_steps": source["timed_steps"], "profile_steps": source["profile_steps"],
              "quality_metrics_read": False,
              "parameter_layout_context": {
                  "total_parameters": 1138203521,
                  "total_parameters_source": "Frozen Max16L model contract; corroborated by CPU meta-model parameter count.",
                  "item_embedding_parameters": (source["config"]["num_items"] + 1) * source["config"]["hidden_size"],
                  "item_embedding_percent": 100 * (source["config"]["num_items"] + 1) * source["config"]["hidden_size"] / 1138203521,
                  "communication_note": "Root FULL_SHARD FSDP retains gathered parameters across forward-to-backward, so one parameter allgather and one gradient reduce-scatter per step are expected; verify observed scope counts. NCCL duration includes rank waiting, not just network transfer."},
              "cpu_collate_ms": distribution([batch["cpu_collate_seconds"] * 1000
                                               for rank in source["fixture"]["ranks"] for batch in rank["batches"]]),
              "cells": cells, "comparisons": comparisons,
              "caveats": ["Max-rank wall time is formed for each common step before its median/mean is calculated.",
                          "Each phase distribution belongs to one rank. Do not sum per-phase maxima from different ranks.",
                          "Default-stream intervals can include waits for side-stream work. They are not exclusive kernel execution times.",
                          "Profiler work/timeline figures are separate measurements with instrumentation overhead; no wall-time percentages are inferred from them.",
                          "NCCL kernel duration contains synchronization/straggler waiting; it must not be interpreted as measured pure network time or bandwidth.",
                          "Global loss and sparse parameter checks are development diagnostics, not evaluation quality or long-term equivalence evidence."]}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    text = markdown(report)
    (args.output_dir / "README.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
