#!/usr/bin/env python3
"""Validate Q_main cutover risk against strict observed-event proxies.

This is a development-only external-validity test.  Yambda exposes listening
events, not serving requests: the first event strictly between release and the
next scheduled release is an observed-event proxy.  Cutover risk is averaged
over Q_main development panels; proxy risk is independently averaged over its
held-out panels.  Neither panel injects a future positive.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.stats import spearmanr

from eval_yambda_cutover_probe_validity import EDGE, state_hash
from eval_yambda_multi_panel_risk import DEV, HELD, PANELS, regret, top_recall
from train_yambda_theta0_medium import DAY, MAX_HISTORY, build_foundation_data
from train_yambda_two_edges import compact_history_tensors, load_checkpoint


ROOT = Path("results/data_audit/yambda50m_v2")
RAW = Path("data/raw/yambda/flat/50m/listens.parquet")


def risk_metrics(cut: np.ndarray, proxy: np.ndarray) -> dict:
    return {
        "spearman": float(spearmanr(cut, proxy).statistic),
        "top10_precision": top_recall(cut, proxy, 0.1),
        "top10_recall": top_recall(cut, proxy, 0.1),
        "top10_enrichment_over_random": top_recall(cut, proxy, 0.1) / 0.1,
        "top20_precision": top_recall(cut, proxy, 0.2),
        "top20_recall": top_recall(cut, proxy, 0.2),
        "top20_enrichment_over_random": top_recall(cut, proxy, 0.2) / 0.2,
    }


def bootstrap(cut: np.ndarray, proxy: np.ndarray, seed: int, rounds: int = 400) -> dict:
    rng = np.random.default_rng(seed)
    n = len(cut)
    metrics = {key: [] for key in ("spearman", "top10_recall", "top20_recall")}
    for _ in range(rounds):
        sample = rng.integers(0, n, size=n)
        value = risk_metrics(cut[sample], proxy[sample])
        for key in metrics:
            metrics[key].append(value[key])
    return {
        key: {"p2_5": float(np.quantile(values, 0.025)), "p97_5": float(np.quantile(values, 0.975))}
        for key, values in metrics.items()
    }


def quartile(values: np.ndarray) -> np.ndarray:
    # Stable rank bins avoid ambiguous qcut behavior with many tied zeroes.
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.int8)
    for index, group in enumerate(np.array_split(order, 4)):
        result[group] = index + 1
    return result


def score_group(parent, current, prefixes, timestamps, panel_rows, item_map, device):
    if len({len(prefix) for prefix in prefixes}) != 1:
        raise ValueError("variable-length materialized KV states cannot be padded into one batch")
    readouts = [(0, int(timestamp), 0) for timestamp in timestamps]
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
    candidates = torch.tensor(
        [[item_map[int(item)] for panel in panels for item in panel["candidate_item_ids"]] for panels in panel_rows],
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        full_hidden, _ = current(full_items, full_behaviors, full_deltas, lengths=full_lengths)
        full_scores = current.score_candidates(full_hidden, candidates, full_lengths).float().cpu().numpy()
        parent_cache = parent.compute_kv(prefix_items, prefix_behaviors, prefix_deltas, prefix_lengths)
        reuse_hidden, _ = current.forward_with_cache(parent_cache, readout_items, readout_behaviors, readout_deltas)
        reuse_scores = current.score_hidden(reuse_hidden[:, -1, :], candidates).float().cpu().numpy()
    return full_scores.reshape(len(prefixes), PANELS, 100), reuse_scores.reshape(len(prefixes), PANELS, 100)


def first_proxy_inputs(edge: str, states: dict[int, dict], item_map: dict[int, int]) -> tuple[dict[int, tuple[list, int]], dict[str, int]]:
    release, next_release, _, _ = EDGE[edge]
    inputs: dict[int, tuple[list, int]] = {}
    skipped = {"no_event_before_next_release": 0, "event_after_next_release": 0, "invalid_prefix": 0, "state_hash_mismatch": 0}
    current_uid, prefix, first_after = None, [], None

    def consume(uid: int, events: list, first: int | None) -> None:
        if uid not in states:
            return
        if first is None:
            skipped["no_event_before_next_release"] += 1
            return
        if first >= next_release:
            skipped["event_after_next_release"] += 1
            return
        effective = [event for event in events if event[0] in item_map][-MAX_HISTORY:]
        if not effective:
            skipped["invalid_prefix"] += 1
            return
        if state_hash(effective) != states[uid]["state_hash"]:
            skipped["state_hash_mismatch"] += 1
            return
        inputs[uid] = (effective, first)

    for batch in pq.ParquetFile(RAW).iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id", "is_organic"]):
        for uid, timestamp, item, organic in zip(
            batch.column("uid").to_numpy(zero_copy_only=False),
            batch.column("timestamp").to_numpy(zero_copy_only=False),
            batch.column("item_id").to_numpy(zero_copy_only=False),
            batch.column("is_organic").to_numpy(zero_copy_only=False),
        ):
            uid, timestamp, item, organic = int(uid), int(timestamp), int(item), int(organic)
            if current_uid is not None and uid != current_uid:
                consume(current_uid, prefix, first_after)
                prefix, first_after = [], None
            current_uid = uid
            if timestamp < release:
                prefix.append((item, timestamp, 1 + (1 - organic)))
            elif timestamp > release and timestamp < next_release and first_after is None:
                first_after = timestamp
            elif timestamp >= next_release and first_after is None:
                # A sentinel retains the first exclusion reason exactly once.
                first_after = next_release
    if current_uid is not None:
        consume(current_uid, prefix, first_after)
    return inputs, skipped


def evaluate_edge(edge: str, device: torch.device) -> tuple[dict, list[dict]]:
    release, next_release, parent_path, current_path = EDGE[edge]
    snapshot_data = pq.read_table(f"data/manifests/yambda50m_v2_release_snapshot_{edge}.parquet").to_pydict()
    states = {int(uid): {key: values[index] for key, values in snapshot_data.items()} for index, uid in enumerate(snapshot_data["uid"])}
    panel_rows = pq.read_table(f"data/manifests/yambda50m_v2_qmain32_v2_{edge}.parquet").to_pylist()
    panels: dict[int, list[dict]] = {}
    for row in panel_rows:
        panels.setdefault(int(row["uid"]), []).append(row)
    for value in panels.values():
        value.sort(key=lambda row: row["panel_id"])
    _, _, item_map, _ = build_foundation_data(RAW, set())
    inputs, skipped = first_proxy_inputs(edge, states, item_map)
    parent, _ = load_checkpoint(Path(parent_path), device)
    current, _ = load_checkpoint(Path(current_path), device)
    rows = []
    groups: dict[int, list] = {}
    for uid, (prefix, timestamp) in inputs.items():
        groups.setdefault(len(prefix), []).append((uid, prefix, timestamp))
    for group in groups.values():
        for start in range(0, len(group), 16):
            chunk = group[start : start + 16]
            uids, prefixes, timestamps = zip(*chunk)
            cut_full, cut_reuse = score_group(parent, current, list(prefixes), [release] * len(chunk), [panels[uid] for uid in uids], item_map, device)
            proxy_full, proxy_reuse = score_group(parent, current, list(prefixes), list(timestamps), [panels[uid] for uid in uids], item_map, device)
            cut_values, cut_floor = regret(cut_full, cut_reuse)
            proxy_values, proxy_floor = regret(proxy_full, proxy_reuse)
            for index, uid in enumerate(uids):
                row = {
                    "edge_id": edge,
                    "uid": int(uid),
                    "proxy_timestamp": int(timestamps[index]),
                    "proxy_delay_seconds": int(timestamps[index] - release),
                    "effective_prefix_length": int(len(prefixes[index])),
                    "raw_prefix_length": int(states[uid]["raw_prefix_length"]),
                    "last_activity_age_seconds": int(states[uid]["last_activity_age_seconds"]),
                    "events_last_7d": int(states[uid]["events_last_7d"]),
                    "cutover_development_mean_regret": float(cut_values[index, DEV].mean()),
                    "proxy_heldout_mean_regret": float(proxy_values[index, HELD].mean()),
                    "cutover_score_std_floor_panels": int(cut_floor[index].sum()),
                    "proxy_score_std_floor_panels": int(proxy_floor[index].sum()),
                }
                rows.append(row)
    cut = np.asarray([row["cutover_development_mean_regret"] for row in rows])
    proxy = np.asarray([row["proxy_heldout_mean_regret"] for row in rows])
    for feature in ("last_activity_age_seconds", "events_last_7d", "effective_prefix_length"):
        values = np.asarray([row[feature] for row in rows])
        for row, bucket in zip(rows, quartile(values)):
            row[f"{feature}_quartile"] = int(bucket)
    cohorts = {}
    for feature in ("last_activity_age_seconds", "events_last_7d", "effective_prefix_length"):
        cohorts[feature] = []
        for bucket in range(1, 5):
            mask = np.asarray([row[f"{feature}_quartile"] == bucket for row in rows])
            cohorts[feature].append({"quartile": bucket, "states": int(mask.sum()), **risk_metrics(cut[mask], proxy[mask]), "bootstrap_95ci": bootstrap(cut[mask], proxy[mask], 1000 + bucket)})
    summary = {
        "release_timestamp": release,
        "next_release_timestamp": next_release,
        "snapshot_states": len(states),
        "proxy_eligible_states": len(rows),
        "strict_proxy_coverage": len(rows) / len(states),
        "proxy_semantics": "first Yambda listening event strictly after release and strictly before next scheduled release; observed-event proxy, not serving request",
        "cutover_risk": "mean current-model Top-K regret across Q_main development panels 0..15",
        "proxy_risk": "mean current-model Top-K regret across disjoint Q_main held-out panels 16..31 at proxy timestamp",
        "target_injection": False,
        "skipped_reasons": skipped,
        "primary_validity": risk_metrics(cut, proxy),
        "bootstrap_95ci": bootstrap(cut, proxy, 100 + len(rows)),
        "stratified_by_pre_release_feature_quartile": cohorts,
        "score_std_floor_state_fraction": float(np.mean([(row["cutover_score_std_floor_panels"] + row["proxy_score_std_floor_panels"]) > 0 for row in rows])),
    }
    return summary, rows


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = {"status": "strict_q_main_multi_panel_external_validity_development", "distribution": "Q_main_rank_decay_v1", "edges": {}}
    csv_rows = []
    for edge in EDGE:
        summary, rows = evaluate_edge(edge, device)
        result["edges"][edge] = summary
        csv_rows.extend(rows)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "external_validity_v3.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ROOT / "external_validity_v3.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
