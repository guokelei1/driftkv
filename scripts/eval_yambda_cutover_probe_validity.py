#!/usr/bin/env python3
"""Test whether a release cutover probe predicts first observed-event risk.

Yambda records listening events rather than serving requests.  The first
post-release event is a trace-derived request proxy, not an online request
log.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.stats import spearmanr

from train_yambda_theta0_medium import DAY, MAX_HISTORY, build_foundation_data, read_manifest
from train_yambda_two_edges import (
    GAP,
    THETA1_RELEASE,
    THETA2_RELEASE,
    compatibility_record,
    compact_history_tensors,
    load_checkpoint,
)


EDGE = {
    "theta0_theta1": (
        THETA1_RELEASE,
        THETA2_RELEASE,
        "checkpoints/yambda50m_v2_theta0_medium_batchfix_v3.pt",
        "checkpoints/yambda50m_v2_theta1_medium_batchfix_v3.pt",
    ),
    # theta3 is deliberately not trained yet, but the daily release cadence
    # already fixes the latest timestamp eligible for an edge-2 proxy.
    "theta1_theta2": (
        THETA2_RELEASE,
        THETA2_RELEASE + DAY + GAP,
        "checkpoints/yambda50m_v2_theta1_medium_batchfix_v3.pt",
        "checkpoints/yambda50m_v2_theta2_medium_batchfix_v3.pt",
    ),
}


def state_hash(events):
    data = [[event[0], event[1], event[2]] for event in events]
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def score_pairs_batch(parent, current, prefixes, candidate_rows, request_timestamps, item_map, device):
    """Score same-length prefix states without padding their cached K/V.

    Pointwise-attention K/V does not carry a per-row valid-length mask after
    cache materialisation.  Consequently, padding variable-length prefix
    caches into a batch would change their semantics.  Callers group entries
    by exact effective prefix length before using this function.
    """
    if len({len(prefix) for prefix in prefixes}) != 1:
        raise ValueError("batching variable-length persistent caches is invalid")
    readouts = [(0, int(timestamp), 0) for timestamp in request_timestamps]
    full_parts = [compact_history_tensors(prefix + [readout], item_map, device) for prefix, readout in zip(prefixes, readouts)]
    prefix_parts = [compact_history_tensors(prefix, item_map, device) for prefix in prefixes]
    readout_parts = [
        compact_history_tensors(
            [readout], item_map, device, previous_timestamp=prefix[-1][1]
        )
        for prefix, readout in zip(prefixes, readouts)
    ]
    full_items, full_behaviors, full_deltas, full_lengths = [torch.cat([part[index] for part in full_parts], dim=0) for index in range(4)]
    prefix_items, prefix_behaviors, prefix_deltas, prefix_lengths = [torch.cat([part[index] for part in prefix_parts], dim=0) for index in range(4)]
    readout_items, readout_behaviors, readout_deltas, _ = [torch.cat([part[index] for part in readout_parts], dim=0) for index in range(4)]
    candidate_tensor = torch.tensor(
        [[item_map[item] for item in candidates] for candidates in candidate_rows], dtype=torch.long, device=device
    )
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        full_hidden, _ = current(full_items, full_behaviors, full_deltas, lengths=full_lengths)
        full_scores = current.score_candidates(full_hidden, candidate_tensor, full_lengths).float().cpu()
        old_cache = parent.compute_kv(prefix_items, prefix_behaviors, prefix_deltas, prefix_lengths)
        hidden, _ = current.forward_with_cache(old_cache, readout_items, readout_behaviors, readout_deltas)
        reuse_scores = current.score_hidden(hidden[:, -1, :], candidate_tensor).float().cpu()
    return full_scores, reuse_scores


def risk_columns(record: dict, prefix_length: int, delay: int):
    return {
        "top10_regret": record["current_model_top10_regret"],
        "top10_overlap_loss": record["top10_overlap_loss"],
        "margin_weighted_pairwise_disagreement": record["margin_weighted_pairwise_disagreement"],
        "js_divergence": record["js_divergence"],
        "normalized_score_rms": record["normalized_score_rms"],
        "prefix_length": prefix_length,
        "first_observed_event_delay_seconds": delay,
    }


def bucket_delay(delay):
    for upper, label in ((3600, "<=1h"), (6 * 3600, "1h-6h"), (DAY, "6h-1d"), (3 * DAY, "1d-3d"), (7 * DAY, "3d-7d")):
        if delay <= upper:
            return label
    return ">7d"


def bucket_prefix(length):
    if length < 64:
        return "<64"
    if length < 256:
        return "64-255"
    return "256-512"


def summarize(records, metric):
    cut = np.asarray([row[f"cutover_{metric}"] for row in records], dtype=float)
    req = np.asarray([row[f"request_{metric}"] for row in records], dtype=float)
    correlation = spearmanr(cut, req)
    n = len(records)
    deciles = []
    order = np.argsort(cut, kind="stable")
    for index, group in enumerate(np.array_split(order, 10), start=1):
        if not len(group):
            continue
        deciles.append({"cutover_risk_decile": index, "users": len(group), "request_risk_mean": float(req[group].mean())})
    recalls = {}
    for fraction in (0.1, 0.2):
        count = max(1, int(np.ceil(n * fraction)))
        high_cut = set(np.argsort(-cut, kind="stable")[:count].tolist())
        high_req = set(np.argsort(-req, kind="stable")[:count].tolist())
        recalls[f"top_{int(fraction * 100)}pct_high_risk_recall"] = len(high_cut & high_req) / len(high_req)
    return {
        "spearman_rho": float(correlation.statistic),
        "spearman_pvalue": float(correlation.pvalue),
        "decile_curve": deciles,
        **recalls,
    }


def cohort_summary(records, metric, key):
    output = []
    for value in sorted({row[key] for row in records}):
        subset = [row for row in records if row[key] == value]
        output.append({
            key: value,
            "users": len(subset),
            "cutover_mean": float(np.mean([row[f"cutover_{metric}"] for row in subset])),
            "request_mean": float(np.mean([row[f"request_{metric}"] for row in subset])),
        })
    return output


def evaluate_edge(edge, snapshot_path, cutover_probe_path, proxy_probe_path, raw_path, device, max_users, batch_size):
    release, next_release, parent_path, current_path = EDGE[edge]
    snapshot = pq.read_table(snapshot_path).to_pydict()
    snapshots = {int(uid): {key: values[index] for key, values in snapshot.items()} for index, uid in enumerate(snapshot["uid"])}
    cutover_probes = {int(row["uid"]): row for row in read_manifest(cutover_probe_path)}
    proxy_probes = {int(row["uid"]): row for row in read_manifest(proxy_probe_path)}
    if max_users is not None:
        selected = set(sorted(snapshots)[:max_users])
        snapshots = {uid: row for uid, row in snapshots.items() if uid in selected}
        cutover_probes = {uid: row for uid, row in cutover_probes.items() if uid in selected}
        proxy_probes = {uid: row for uid, row in proxy_probes.items() if uid in selected}
    _, _, item_map, _ = build_foundation_data(raw_path, set())
    parent, _ = load_checkpoint(Path(parent_path), device)
    current, _ = load_checkpoint(Path(current_path), device)
    records, skipped = [], {
        "no_observed_event": 0,
        "no_proxy_before_next_release": 0,
        "state_hash_mismatch": 0,
        "invalid_prefix": 0,
    }
    state_inputs = {}
    current_uid, prefix, first_request = None, [], None

    def consume(uid, prefix_events, first):
        if uid not in snapshots:
            return
        if first is None:
            skipped["no_observed_event"] += 1
            return
        if first >= next_release:
            skipped["no_proxy_before_next_release"] += 1
            return
        effective = [event for event in prefix_events if event[0] in item_map][-MAX_HISTORY:]
        if not effective:
            skipped["invalid_prefix"] += 1
            return
        if state_hash(effective) != snapshots[uid]["state_hash"]:
            skipped["state_hash_mismatch"] += 1
            return
        state_inputs[uid] = (effective, first)

    parquet = pq.ParquetFile(raw_path)
    for batch in parquet.iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id", "is_organic"]):
        for uid, timestamp, item, organic in zip(
            batch.column("uid").to_numpy(zero_copy_only=False),
            batch.column("timestamp").to_numpy(zero_copy_only=False),
            batch.column("item_id").to_numpy(zero_copy_only=False),
            batch.column("is_organic").to_numpy(zero_copy_only=False),
        ):
            uid, timestamp, item, organic = int(uid), int(timestamp), int(item), int(organic)
            if current_uid is not None and uid != current_uid:
                consume(current_uid, prefix, first_request)
                prefix, first_request = [], None
            current_uid = uid
            if timestamp < release:
                prefix.append((item, timestamp, 1 + (1 - organic)))
            # A timestamp exactly at the release boundary is not ordered after
            # the model cutover at Yambda's five-second precision.  The proxy
            # contract therefore requires a strictly later observed event.
            elif timestamp > release and first_request is None:
                first_request = timestamp
    if current_uid is not None:
        consume(current_uid, prefix, first_request)
    groups = {}
    for uid, (effective, first) in state_inputs.items():
        groups.setdefault(len(effective), []).append((uid, effective, first))
    for group in groups.values():
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            uids = [entry[0] for entry in chunk]
            prefixes = [entry[1] for entry in chunk]
            firsts = [entry[2] for entry in chunk]
            cut_candidates = [[int(item) for item in cutover_probes[uid]["candidate_item_ids"]] for uid in uids]
            proxy_candidates = [[int(item) for item in proxy_probes[uid]["candidate_item_ids"]] for uid in uids]
            cut_full, cut_reuse = score_pairs_batch(parent, current, prefixes, cut_candidates, [release] * len(chunk), item_map, device)
            req_full, req_reuse = score_pairs_batch(parent, current, prefixes, proxy_candidates, firsts, item_map, device)
            for index, (uid, prefix, first) in enumerate(chunk):
                lineage = {"request_timestamp": first, "effective_prefix_length": len(prefix)}
                cost = {"recomputed_tokens": len(prefix) + 1, "reuse_append_tokens": 1, "prefix_kv_read_bytes": 0, "full_recompute_latency_ms": 0.0, "reuse_append_latency_ms": 0.0}
                cut = compatibility_record(uid, cut_full[index], cut_reuse[index], lineage, cost)
                req = compatibility_record(uid, req_full[index], req_reuse[index], lineage, cost)
                record = {
                    "uid": uid,
                    "first_observed_event_delay_bucket": bucket_delay(first - release),
                    "prefix_length_bucket": bucket_prefix(len(prefix)),
                }
                for prefix_name, values in (("cutover", risk_columns(cut, len(prefix), 0)), ("request", risk_columns(req, len(prefix), first - release))):
                    record.update({f"{prefix_name}_{key}": value for key, value in values.items() if key not in {"prefix_length", "first_observed_event_delay_seconds"}})
                record["effective_prefix_length"] = len(prefix)
                record["first_observed_event_delay_seconds"] = first - release
                records.append(record)
    primary = "top10_regret"
    return {
        "snapshot_users": len(snapshots),
        "evaluated_users": len(records),
        "release_timestamp": release,
        "next_release_timestamp": next_release,
        "proxy_coverage_rate": len(records) / len(snapshots),
        "skipped_reasons": skipped,
        "primary_metric": primary,
        "primary_validity": summarize(records, primary),
        "all_metrics": {metric: summarize(records, metric) for metric in ("top10_overlap_loss", "margin_weighted_pairwise_disagreement", "js_divergence", "normalized_score_rms")},
        "delay_cohorts": cohort_summary(records, primary, "first_observed_event_delay_bucket"),
        "prefix_cohorts": cohort_summary(records, primary, "prefix_length_bucket"),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cutover-panel", choices=("a", "b"), default="a")
    parser.add_argument("--proxy-panel", choices=("a", "b"), default="a")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raw_path = Path("data/raw/yambda/flat/50m/listens.parquet")
    def panel_path(edge: str, panel: str) -> Path:
        suffix = "" if panel == "a" else "_panel_b"
        return Path(f"data/manifests/yambda50m_v2_cutover_probe{suffix}_{edge}.jsonl")
    result = {
        "status": "cutover_probe_first_observed_event_proxy_validity_development",
        "evaluation_semantics": "first post-release Yambda listening event; a request proxy, not an online request log",
        "device": str(device),
        "target_injected": False,
        "cutover_panel": args.cutover_panel,
        "proxy_panel": args.proxy_panel,
        "edges": {},
    }
    csv_rows = []
    for edge in EDGE:
        value = evaluate_edge(
            edge,
            Path(f"data/manifests/yambda50m_v2_release_snapshot_{edge}.parquet"),
            panel_path(edge, args.cutover_panel),
            panel_path(edge, args.proxy_panel),
            raw_path, device, args.max_users, args.batch_size,
        )
        result["edges"][edge] = value
        for row in value["records"]:
            csv_rows.append({"edge_id": edge, **row})
    output_dir = Path("results/data_audit/yambda50m_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "v2" if (args.cutover_panel, args.proxy_panel) == ("a", "a") else f"panel_{args.cutover_panel}_to_{args.proxy_panel}_v1"
    (output_dir / f"cutover_probe_validity_{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output_dir / f"cutover_probe_validity_{tag}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({edge: {key: value[key] for key in ("snapshot_users", "evaluated_users", "primary_validity")} for edge, value in result["edges"].items()}, indent=2))


if __name__ == "__main__":
    main()
