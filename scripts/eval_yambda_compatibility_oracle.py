#!/usr/bin/env python3
"""Evaluate target-independent, first-post-release compatibility oracle records.

The HSTU next-item readout normally comes from a real appended event. At the
first request after a release there can be no current-version append. This
screen appends an explicit padding-item/readout token to both Full and Reuse so
the two paths have the same query semantics. It is a fidelity profiler, not a
recommendation-quality endpoint and never injects a future positive.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from train_yambda_theta0_medium import MAX_HISTORY, read_manifest
from train_yambda_two_edges import (
    THETA1_RELEASE,
    THETA2_RELEASE,
    compatibility_record,
    collect_chain_data,
    compact_history_tensors,
    load_checkpoint,
    split_lineage,
    summarize_compatibility,
)


def score_history(model, history, candidates, item_map, device):
    items, behaviors, deltas, lengths = compact_history_tensors(history, item_map, device)
    candidate_tensor = torch.tensor([[item_map[item] for item in candidates]], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden, _ = model(items, behaviors, deltas, lengths=lengths)
        return model.score_candidates(hidden, candidate_tensor, lengths).float().cpu()[0]


def timed_score_history(model, history, candidates, item_map, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    scores = score_history(model, history, candidates, item_map, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return scores, (time.perf_counter() - start) * 1000.0


def evaluate_edge(previous, current, rows, events_by_uid, item_map, device, release_timestamp):
    records = []
    skipped = {
        "empty_prefix_after_catalog": 0,
        "missing_candidate": 0,
        "other_invalid": 0,
    }
    drifts = []
    for row in rows:
        events = events_by_uid.get(int(row["uid"]), [])
        raw_prefix, raw_suffix, prefix, suffix = split_lineage(
            events, release_timestamp, int(row["request_timestamp"]), item_map, reserve_readout=1
        )
        if raw_suffix:
            raise ValueError("target-independent first-request manifest unexpectedly has an append suffix")
        if not prefix:
            skipped["empty_prefix_after_catalog"] += 1
            continue
        candidates = [int(item) for item in row["candidate_item_ids"]]
        if any(item not in item_map for item in candidates):
            skipped["missing_candidate"] += 1
            continue
        readout = (0, int(row["request_timestamp"]), 0)
        full_history = prefix + [readout]
        current_scores, full_latency_ms = timed_score_history(current, full_history, candidates, item_map, device)
        p_items, p_behaviors, p_deltas, p_lengths = compact_history_tensors(prefix, item_map, device)
        r_items, r_behaviors, r_deltas, _ = compact_history_tensors(
            [readout], item_map, device, previous_timestamp=prefix[-1][1]
        )
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            old_cache = previous.compute_kv(p_items, p_behaviors, p_deltas, p_lengths)
            new_cache = current.compute_kv(p_items, p_behaviors, p_deltas, p_lengths)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        reuse_start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden, _ = current.forward_with_cache(old_cache, r_items, r_behaviors, r_deltas)
            candidate_tensor = torch.tensor([[item_map[item] for item in candidates]], dtype=torch.long, device=device)
            reuse_scores = current.score_hidden(hidden[:, -1, :], candidate_tensor).float().cpu()[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        reuse_latency_ms = (time.perf_counter() - reuse_start) * 1000.0
        cache_bytes = (old_cache.k.numel() + old_cache.v.numel()) * old_cache.k.element_size()
        records.append(compatibility_record(
            int(row["uid"]), current_scores, reuse_scores,
            {
                "request_id": row["request_id"],
                "request_timestamp": int(row["request_timestamp"]),
                "raw_prefix_length": len(raw_prefix),
                "effective_prefix_length": len(prefix),
                "raw_post_release_append_count": len(raw_suffix),
                "post_release_append_count": len(suffix),
                "append_ratio": 0.0,
                "cache_age_seconds": 0,
                "readout_token": True,
            },
            {
                "full_recompute_latency_ms": full_latency_ms,
                "reuse_append_latency_ms": reuse_latency_ms,
                "recomputed_tokens": len(full_history),
                "reuse_append_tokens": 1,
                "prefix_kv_read_bytes": cache_bytes,
            },
        ))
        drifts.append(old_cache.difference_metrics(new_cache))
    result = {
        "evaluated_users": len(records),
        "skipped_reasons": skipped,
        "compatibility_summary": summarize_compatibility(records),
        "compatibility_records": records,
    }
    if drifts:
        result["mean_prefix_k_rms_drift"] = sum(item["k_rms"] for item in drifts) / len(drifts)
        result["mean_prefix_v_rms_drift"] = sum(item["v_rms"] for item in drifts) / len(drifts)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    theta0, _ = load_checkpoint(Path("checkpoints/yambda50m_v2_theta0_medium_batchfix_v3.pt"), device)
    theta1, _ = load_checkpoint(Path("checkpoints/yambda50m_v2_theta1_medium_batchfix_v3.pt"), device)
    theta2, _ = load_checkpoint(Path("checkpoints/yambda50m_v2_theta2_medium_batchfix_v3.pt"), device)
    edge1_rows = read_manifest(Path("data/manifests/yambda50m_v2_edge1_profiler_candidates.jsonl"))
    edge2_rows = read_manifest(Path("data/manifests/yambda50m_v2_edge2_profiler_candidates.jsonl"))
    if args.max_users is not None:
        edge1_rows, edge2_rows = edge1_rows[:args.max_users], edge2_rows[:args.max_users]
    uids = {int(row["uid"]) for row in edge1_rows + edge2_rows}
    _, _, events_by_uid = collect_chain_data(Path("data/raw/yambda/flat/50m/listens.parquet"), uids)
    _, _, item_map, _ = __import__("train_yambda_theta0_medium", fromlist=["build_foundation_data"]).build_foundation_data(
        Path("data/raw/yambda/flat/50m/listens.parquet"), set()
    )
    result = {
        "status": "target_independent_first_request_compatibility_oracle",
        "device": str(device),
        "candidate_protocol": "target_independent_compatibility_profiling",
        "target_injected": False,
        "readout_semantics": "explicit_padding_item_readout_token; fidelity_only_not_quality_endpoint",
        "max_history_including_readout": MAX_HISTORY,
        "edge_theta0_theta1": evaluate_edge(theta0, theta1, edge1_rows, events_by_uid, item_map, device, THETA1_RELEASE),
        "edge_theta1_theta2": evaluate_edge(theta1, theta2, edge2_rows, events_by_uid, item_map, device, THETA2_RELEASE),
    }
    output = Path("results/data_audit/yambda50m_v2/compatibility_profile_batchfix_v3_screen.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "edge1_evaluated": result["edge_theta0_theta1"]["evaluated_users"],
        "edge2_evaluated": result["edge_theta1_theta2"]["evaluated_users"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
