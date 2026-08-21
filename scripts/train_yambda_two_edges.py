#!/usr/bin/env python3
"""Train theta1/theta2 and evaluate a release-correct one-hop chain screen."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from hstu_kvcache.models import HSTU, HSTUConfig
from train_yambda_theta0_medium import (
    CANDIDATE_SIZE,
    DAY,
    MAX_HISTORY,
    collate_histories,
    history_to_arrays,
    make_candidates,
    metrics,
    read_manifest,
)


GAP = 1_800
BASE_END = 210 * DAY
W1_START = BASE_END + GAP
W1_END = W1_START + DAY
THETA1_RELEASE = W1_END + GAP
W2_START = THETA1_RELEASE
W2_END = W2_START + DAY
THETA2_RELEASE = W2_END + GAP


def load_checkpoint(path: Path, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = HSTU(HSTUConfig(**saved["config"])).to(device).eval()
    model.load_state_dict(saved["model"])
    return model, saved["config"]


def collect_chain_data(path: Path, eval_uids: set[int]):
    updates1, updates2 = [], []
    eval_events = {uid: [] for uid in eval_uids}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id", "is_organic"]):
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        ts = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        organic = batch.column("is_organic").to_numpy(zero_copy_only=False).astype(np.int64)
        mask = ts < W2_END + DAY + GAP
        uid, ts, item, organic = uid[mask], ts[mask], item[mask], organic[mask]
        boundaries = np.flatnonzero(uid[1:] != uid[:-1]) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, len(uid)]
        for start, end in zip(starts, ends):
            u = int(uid[start])
            group_ts, group_item, group_org = ts[start:end], item[start:end], organic[start:end]
            events = [(int(i), int(t), int(1 + (1 - int(o)))) for i, t, o in zip(group_item, group_ts, group_org)]
            w1_indices = [index for index, event in enumerate(events) if W1_START <= event[1] < W1_END]
            w2_indices = [index for index, event in enumerate(events) if W2_START <= event[1] < W2_END]
            if w1_indices:
                target_index = w1_indices[-1]
                updates1.append((events[:target_index], events[target_index][0]))
            if w2_indices:
                target_index = w2_indices[-1]
                updates2.append((events[:target_index], events[target_index][0]))
            if u in eval_events:
                eval_events[u] = events
    return updates1, updates2, eval_events


def train_update(parent, samples, item_map, popular_items, device):
    model = copy.deepcopy(parent).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    usable = [
        (history, target)
        for history, target in samples
        if target in item_map and any(event[0] in item_map for event in history)
    ]
    order = np.arange(len(usable))
    np.random.default_rng(1).shuffle(order)
    batch_size = 16
    losses = []
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        histories = [[event for event in usable[i][0] if event[0] in item_map] for i in indices]
        targets = [usable[i][1] for i in indices]
        items, behaviors, deltas, lengths = collate_histories(histories, item_map)
        candidates = torch.from_numpy(np.asarray([
            [item_map[x] for x in make_candidates(target, popular_items)] for target in targets
        ], dtype=np.int64)).to(device)
        items, behaviors, deltas, lengths = [x.to(device) for x in (items, behaviors, deltas, lengths)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden, _ = model(items, behaviors, deltas, lengths=lengths)
            scores = model.score_candidates(hidden, candidates, lengths)
            loss = F.cross_entropy(scores, torch.zeros(len(indices), dtype=torch.long, device=device))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return model.eval(), {
        "samples": len(usable),
        "dropped_unknown_targets": len(samples) - len(usable),
        "steps": len(losses),
        "mean_loss": float(np.mean(losses)),
    }


def to_model_history(events, item_map):
    filtered = [x for x in events if x[0] in item_map][-MAX_HISTORY:]
    return filtered


def compact_history_tensors(history, item_map, device, *, previous_timestamp=None):
    """Return an unpadded contiguous event slice.

    For an incremental append, ``previous_timestamp`` must be the timestamp
    of the last cached prefix event, so the first suffix token retains its
    actual temporal delta instead of being reset to zero.
    """
    item_ids, behaviors, deltas = history_to_arrays(
        history, item_map, previous_timestamp=previous_timestamp
    )
    if len(item_ids) == 0:
        raise ValueError("cache history must contain at least one token")
    return (
        torch.from_numpy(item_ids[None, :]).to(device),
        torch.from_numpy(behaviors[None, :]).to(device),
        torch.from_numpy(deltas[None, :]).to(device),
        torch.tensor([len(item_ids)], dtype=torch.long, device=device),
    )


def score_one(model, events, candidates, item_map, device):
    history = to_model_history(events, item_map)
    if not history:
        items = torch.zeros((1, 1), dtype=torch.long, device=device)
        behaviors = torch.zeros_like(items)
        deltas = torch.zeros((1, 1), dtype=torch.float32, device=device)
        lengths = torch.zeros(1, dtype=torch.long, device=device)
    else:
        items, behaviors, deltas, lengths = compact_history_tensors(history, item_map, device)
    candidate_tensor = torch.tensor([[item_map[x] for x in candidates]], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden, _ = model(items, behaviors, deltas, lengths=lengths)
        scores = model.score_candidates(hidden, candidate_tensor, lengths)
    return scores.float().cpu()[0]


def compatibility_record(uid, current_scores, reuse_scores, lineage, cost, target_index=None):
    """Return label-free compatibility features plus target-rank diagnostics."""
    current_prob = torch.softmax(current_scores, dim=0)
    reuse_prob = torch.softmax(reuse_scores, dim=0)
    midpoint = (current_prob + reuse_prob) / 2
    js = 0.5 * (
        torch.sum(current_prob * (torch.log(current_prob.clamp_min(1e-12)) - torch.log(midpoint.clamp_min(1e-12))))
        + torch.sum(reuse_prob * (torch.log(reuse_prob.clamp_min(1e-12)) - torch.log(midpoint.clamp_min(1e-12))))
    )
    current_order = torch.argsort(current_scores, descending=True)
    reuse_order = torch.argsort(reuse_scores, descending=True)
    current_top10 = set(current_order[:10].tolist())
    reuse_top10 = set(reuse_order[:10].tolist())
    score_std = float(current_scores.float().std(unbiased=False).clamp_min(1e-8))
    pairwise_current = current_scores[:, None] > current_scores[None, :]
    pairwise_reuse = reuse_scores[:, None] > reuse_scores[None, :]
    upper = torch.triu(torch.ones_like(pairwise_current, dtype=torch.bool), diagonal=1)
    full_top10_utility = current_scores[current_order[:10]].mean()
    reuse_top10_utility = current_scores[reuse_order[:10]].mean()
    score_gaps = torch.abs(current_scores[:, None] - current_scores[None, :])
    disagreement = pairwise_current[upper] != pairwise_reuse[upper]
    record = {
        "uid": int(uid),
        "score_rms": float(torch.sqrt(torch.mean((current_scores - reuse_scores) ** 2))),
        "normalized_score_rms": float(torch.sqrt(torch.mean((current_scores - reuse_scores) ** 2)) / score_std),
        "score_max_abs": float(torch.max(torch.abs(current_scores - reuse_scores))),
        "js_divergence": float(js),
        "top10_overlap": len(current_top10 & reuse_top10) / 10.0,
        "top10_overlap_loss": 1.0 - len(current_top10 & reuse_top10) / 10.0,
        "current_model_top10_regret": float((full_top10_utility - reuse_top10_utility) / score_std),
        "pairwise_inversion_rate": float((pairwise_current[upper] != pairwise_reuse[upper]).float().mean()),
        "margin_weighted_pairwise_disagreement": float(
            score_gaps[upper][disagreement].sum() / score_gaps[upper].sum().clamp_min(1e-8)
        ),
        "reuse_score_entropy": float(-(reuse_prob * torch.log(reuse_prob.clamp_min(1e-12))).sum()),
        "reuse_top10_boundary_margin": float(
            reuse_scores[reuse_order[9]] - reuse_scores[reuse_order[10]]
        ),
        "current_scores": [float(value) for value in current_scores],
        "reuse_scores": [float(value) for value in reuse_scores],
        **lineage,
        **cost,
    }
    if target_index is not None:
        current_rank = int((current_scores >= current_scores[target_index]).sum())
        reuse_rank = int((reuse_scores >= reuse_scores[target_index]).sum())
        record.update({
            "target_index": int(target_index),
            "target_rank_current": current_rank,
            "target_rank_reuse": reuse_rank,
            "target_rank_displacement": reuse_rank - current_rank,
        })
    return record


def summarize_compatibility(records):
    if not records:
        return {}
    summary = {}
    for key in (
        "score_rms", "normalized_score_rms", "score_max_abs", "js_divergence",
        "top10_overlap", "top10_overlap_loss", "pairwise_inversion_rate",
    ):
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        }
    if "target_rank_displacement" in records[0]:
        values = np.asarray([record["target_rank_displacement"] for record in records], dtype=np.float64)
        summary["target_rank_displacement"] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        }
    summary["label_free_nontrivial_js_fraction_gt_1e-6"] = float(
        np.mean(np.asarray([record["js_divergence"] for record in records]) > 1e-6)
    )
    summary["label_free_top10_changed_fraction"] = float(
        np.mean(np.asarray([record["top10_overlap"] for record in records]) < 1.0)
    )
    return summary


def split_lineage(events, release_timestamp, request_timestamp, item_map, reserve_readout=0):
    raw_prefix = [event for event in events if event[1] < release_timestamp]
    raw_suffix = [event for event in events if release_timestamp <= event[1] < request_timestamp]
    prefix = [event for event in raw_prefix if event[0] in item_map]
    suffix = [event for event in raw_suffix if event[0] in item_map]
    budget = MAX_HISTORY - reserve_readout
    suffix = suffix[-budget:]
    prefix = prefix[-max(0, budget - len(suffix)):]
    return raw_prefix, raw_suffix, prefix, suffix


def timed_score(model, events, candidates, item_map, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    scores = score_one(model, events, candidates, item_map, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return scores, (time.perf_counter() - start) * 1000.0


def evaluate_edge(previous, current, rows, eval_events, item_map, device, release_timestamp, max_users):
    paths = {"previous_full": [], "current_full": [], "reuse": [], "current_suffix_only": []}
    drifts = []
    compatibility = []
    skipped = {
        "zero_post_release_append": 0,
        "oov_only_append": 0,
        "empty_prefix_after_catalog": 0,
        "missing_candidate_or_target": 0,
        "other_invalid": 0,
    }
    for row in rows[:max_users]:
        events = eval_events.get(int(row["uid"]), [])
        raw_pre, raw_suffix, pre, suffix = split_lineage(
            events, release_timestamp, int(row["request_timestamp"]), item_map
        )
        if not raw_suffix:
            skipped["zero_post_release_append"] += 1
            continue
        candidates = [int(x) for x in row["candidate_item_ids"]]
        if any(candidate not in item_map for candidate in candidates):
            skipped["missing_candidate_or_target"] += 1
            continue
        if not suffix:
            skipped["oov_only_append"] += 1
            continue
        if not pre:
            skipped["empty_prefix_after_catalog"] += 1
            continue
        full_history = pre + suffix
        previous_scores, _ = timed_score(previous, full_history, candidates, item_map, device)
        current_scores, full_latency_ms = timed_score(current, full_history, candidates, item_map, device)
        suffix_scores, _ = timed_score(current, suffix, candidates, item_map, device)
        paths["previous_full"].append(previous_scores)
        paths["current_full"].append(current_scores)
        paths["current_suffix_only"].append(suffix_scores)
        p_items, p_behaviors, p_deltas, p_lengths = compact_history_tensors(pre, item_map, device)
        s_items, s_behaviors, s_deltas, _ = compact_history_tensors(
            suffix, item_map, device, previous_timestamp=pre[-1][1]
        )
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            old_cache = previous.compute_kv(p_items, p_behaviors, p_deltas, p_lengths)
            new_cache = current.compute_kv(p_items, p_behaviors, p_deltas, p_lengths)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        append_start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden, _ = current.forward_with_cache(old_cache, s_items, s_behaviors, s_deltas)
            last = hidden[:, -1, :]
            candidate_tensor = torch.tensor([[item_map[x] for x in candidates]], dtype=torch.long, device=device)
            reuse_scores = current.score_hidden(last, candidate_tensor).float().cpu()[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        reuse_latency_ms = (time.perf_counter() - append_start) * 1000.0
        paths["reuse"].append(reuse_scores)
        cache_bytes = (old_cache.k.numel() + old_cache.v.numel()) * old_cache.k.element_size()
        compatibility.append(compatibility_record(
            int(row["uid"]), current_scores, reuse_scores,
            {
                "request_timestamp": int(row["request_timestamp"]),
                "raw_prefix_length": len(raw_pre),
                "effective_prefix_length": len(pre),
                "raw_post_release_append_count": len(raw_suffix),
                "post_release_append_count": len(suffix),
                "append_ratio": len(suffix) / max(1, len(pre) + len(suffix)),
                "cache_age_seconds": int(row["request_timestamp"]) - release_timestamp,
            },
            {
                "full_recompute_latency_ms": full_latency_ms,
                "reuse_append_latency_ms": reuse_latency_ms,
                "recomputed_tokens": len(full_history),
                "reuse_append_tokens": len(suffix),
                "prefix_kv_read_bytes": cache_bytes,
            },
            target_index=0,
        ))
        drifts.append(old_cache.difference_metrics(new_cache))
    result = {name: metrics(torch.stack(values)) for name, values in paths.items() if values}
    result["evaluated_users"] = len(paths["current_full"])
    result["skipped_reasons"] = skipped
    if drifts:
        result["mean_prefix_k_rms_drift"] = float(np.mean([x["k_rms"] for x in drifts]))
        result["mean_prefix_v_rms_drift"] = float(np.mean([x["v_rms"] for x in drifts]))
    result["compatibility_summary"] = summarize_compatibility(compatibility)
    result["compatibility_records"] = compatibility
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-eval-users", type=int, default=256)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    theta0, config = load_checkpoint(Path("checkpoints/yambda50m_v2_theta0_medium_batchfix_v3.pt"), device)
    dev1 = read_manifest(Path("data/manifests/yambda50m_v2_theta0_dev_candidates.jsonl"))
    qual1 = read_manifest(Path("data/manifests/yambda50m_v2_theta0_qualification_candidates.jsonl"))
    dev2 = read_manifest(Path("data/manifests/yambda50m_v2_edge2_dev_candidates.jsonl"))
    qual2 = read_manifest(Path("data/manifests/yambda50m_v2_edge2_qualification_candidates.jsonl"))
    eval_uids = {int(r["uid"]) for r in dev1 + qual1 + dev2 + qual2}
    updates1, updates2, eval_events = collect_chain_data(Path("data/raw/yambda/flat/50m/listens.parquet"), eval_uids)
    _, _, item_map, popular_items = __import__("train_yambda_theta0_medium", fromlist=["build_foundation_data"]).build_foundation_data(
        Path("data/raw/yambda/flat/50m/listens.parquet"), set()
    )
    if args.eval_only:
        theta1, _ = load_checkpoint(Path("checkpoints/yambda50m_v2_theta1_medium_batchfix_v3.pt"), device)
        theta2, _ = load_checkpoint(Path("checkpoints/yambda50m_v2_theta2_medium_batchfix_v3.pt"), device)
        train1 = {"reused_checkpoint": True}
        train2 = {"reused_checkpoint": True}
    else:
        theta1, train1 = train_update(theta0, updates1, item_map, popular_items, device)
        theta2, train2 = train_update(theta1, updates2, item_map, popular_items, device)
        torch.save({"config": config, "model": theta1.state_dict(), "seed": 1, "contract": "yambda50m_v2", "release_timestamp": THETA1_RELEASE}, "checkpoints/yambda50m_v2_theta1_medium_batchfix_v3.pt")
        torch.save({"config": config, "model": theta2.state_dict(), "seed": 1, "contract": "yambda50m_v2", "release_timestamp": THETA2_RELEASE}, "checkpoints/yambda50m_v2_theta2_medium_batchfix_v3.pt")
    result = {
        "status": "one_hop_two_edge_chain_batchfix_v3_screen",
        "device": str(device),
        "windows": {
            "foundation_days": 210, "update_window": "1d", "release_gap_seconds": GAP,
            "theta1_release_timestamp": THETA1_RELEASE,
            "theta2_release_timestamp": THETA2_RELEASE,
        },
        "training": {"theta1": train1, "theta2": train2},
        "edge_theta0_theta1": evaluate_edge(theta0, theta1, qual1, eval_events, item_map, device, THETA1_RELEASE, args.max_eval_users),
        "edge_theta1_theta2": evaluate_edge(theta1, theta2, qual2, eval_events, item_map, device, THETA2_RELEASE, args.max_eval_users),
    }
    output = Path("results/data_audit/yambda50m_v2/two_edge_chain_batchfix_v3_screen.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
