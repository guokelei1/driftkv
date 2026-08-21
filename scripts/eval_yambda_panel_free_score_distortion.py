#!/usr/bin/env python3
"""Audit panel-free Q_main score distortion on its frozen 1,000-item support.

For the rank-decay proposal q, this computes
sqrt(E_q[(s_full-s_reuse)^2]) / (sqrt(E_q[s_full^2]) + eps).
The finite support is constructed before the release exactly as Q_main panels
are, but no 100-way panel sampling is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.stats import spearmanr

from build_yambda_release_snapshot import FOUNDATION_END, prepare_catalog_and_popularity
from eval_yambda_cutover_probe_validity import EDGE, state_hash
from eval_yambda_release_budget_oracle import BUDGETS, greedy_indices, summarize
from train_yambda_theta0_medium import MAX_HISTORY, build_foundation_data
from train_yambda_two_edges import compact_history_tensors, load_checkpoint


ROOT = Path("results/data_audit/yambda50m_v2")
RAW = Path("data/raw/yambda/flat/50m/listens.parquet")
POOL = 1000
POWER = .5


def score_batch(parent, current, prefixes, candidates, release, item_map, device):
    readouts = [(0, release, 0)] * len(prefixes)
    fp = [compact_history_tensors(h + [r], item_map, device) for h, r in zip(prefixes, readouts)]
    pp = [compact_history_tensors(h, item_map, device) for h in prefixes]
    rp = [
        compact_history_tensors([r], item_map, device, previous_timestamp=prefix[-1][1])
        for prefix, r in zip(prefixes, readouts)
    ]
    fi, fb, fd, fl = [torch.cat([x[i] for x in fp]) for i in range(4)]
    pi, pb, pd, pl = [torch.cat([x[i] for x in pp]) for i in range(4)]
    ri, rb, rd, _ = [torch.cat([x[i] for x in rp]) for i in range(4)]
    candidate = torch.tensor([[item_map[int(item)] for item in row] for row in candidates], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden, _ = current(fi, fb, fd, lengths=fl)
        full = current.score_candidates(hidden, candidate, fl).float().cpu().numpy()
        cache = parent.compute_kv(pi, pb, pd, pl)
        reuse_hidden, _ = current.forward_with_cache(cache, ri, rb, rd)
        reuse = current.score_hidden(reuse_hidden[:, -1, :], candidate).float().cpu().numpy()
    return full, reuse


def inputs_and_pools(edge, states, item_map, popular):
    release, _, _, _ = EDGE[edge]
    output = {}
    current_uid, prefix, seen = None, [], set()
    def consume(uid, events, base_seen):
        if uid not in states:
            return
        effective = [event for event in events if event[0] in item_map][-MAX_HISTORY:]
        if not effective or state_hash(effective) != states[uid]["state_hash"]:
            raise ValueError("release state does not match reconstructed pre-release prefix")
        pool = [int(item) for item in popular if int(item) not in base_seen][:POOL]
        if len(pool) != POOL:
            raise ValueError("insufficient Q_main proposal support")
        output[uid] = (effective, pool)
    for batch in pq.ParquetFile(RAW).iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id", "is_organic", "played_ratio_pct"]):
        for uid, timestamp, item, organic, played in zip(
            batch.column("uid").to_numpy(zero_copy_only=False), batch.column("timestamp").to_numpy(zero_copy_only=False),
            batch.column("item_id").to_numpy(zero_copy_only=False), batch.column("is_organic").to_numpy(zero_copy_only=False),
            batch.column("played_ratio_pct").to_numpy(zero_copy_only=False),
        ):
            uid, timestamp, item, organic, played = int(uid), int(timestamp), int(item), int(organic), int(played)
            if current_uid is not None and uid != current_uid:
                consume(current_uid, prefix, seen); prefix, seen = [], set()
            current_uid = uid
            if timestamp < release:
                prefix.append((item, timestamp, 1 + (1 - organic)))
            if timestamp < FOUNDATION_END and played > 50:
                seen.add(item)
    if current_uid is not None:
        consume(current_uid, prefix, seen)
    if len(output) != len(states):
        raise ValueError("proposal reconstruction lost release states")
    return output


def area(values):
    return float(np.trapezoid([row["residual_primary_fidelity_loss"]["mean"] for row in values], [row["exact_equivalent_work_ratio"] for row in values]))


def evaluate_edge(edge, device):
    release, _, parent_path, current_path = EDGE[edge]
    snapshot = pq.read_table(f"data/manifests/yambda50m_v2_release_snapshot_{edge}.parquet").to_pydict()
    states = {int(uid): {key: values[i] for key, values in snapshot.items()} for i, uid in enumerate(snapshot["uid"])}
    risk = pq.read_table(ROOT / f"multi_panel_risk_v1_{edge}.parquet").to_pydict()
    multi = {int(uid): (float(dev), float(held)) for uid, dev, held in zip(risk["uid"], risk["dev_mean"], risk["heldout_mean"])}
    _, popular = prepare_catalog_and_popularity(RAW)
    _, _, item_map, _ = build_foundation_data(RAW, set())
    inputs = inputs_and_pools(edge, states, item_map, popular)
    parent, _ = load_checkpoint(Path(parent_path), device); current, _ = load_checkpoint(Path(current_path), device)
    weights = np.arange(1, POOL + 1, dtype=float) ** (-POWER); weights /= weights.sum()
    rows, groups = [], {}
    for uid, entry in inputs.items(): groups.setdefault(len(entry[0]), []).append((uid, *entry))
    for group in groups.values():
        for start in range(0, len(group), 16):
            chunk = group[start:start + 16]
            full, reuse = score_batch(parent, current, [x[1] for x in chunk], [x[2] for x in chunk], release, item_map, device)
            diff = full - reuse
            distortion = np.sqrt((diff * diff * weights[None, :]).sum(1)) / np.maximum(np.sqrt((full * full * weights[None, :]).sum(1)), 1e-8)
            for index, (uid, history, _) in enumerate(chunk):
                dev, held = multi[uid]
                rows.append({"edge_id": edge, "uid": uid, "effective_prefix_length": len(history), "exact_token_layer_work": states[uid]["exact_token_layer_work"], "catalog_weighted_score_distortion": float(distortion[index]), "qmain_development_mean_topk_regret": dev, "qmain_heldout_mean_topk_regret": held})
    d = np.asarray([row["catalog_weighted_score_distortion"] for row in rows]); held = np.asarray([row["qmain_heldout_mean_topk_regret"] for row in rows]); costs = np.asarray([row["exact_token_layer_work"] for row in rows], dtype=float)
    total = float(costs.sum()); frontier = []
    for budget in BUDGETS:
        selected = greedy_indices(np.argsort(-(d / costs), kind="stable"), costs, total * budget)
        frontier.append(summarize("panel_free_distortion_priority", selected, held, costs, total))
    summary = {"states": len(rows), "metric": "sqrt(E_q[(full-reuse)^2])/sqrt(E_q[full^2]) over Q_main 1000-item rank-decay support", "spearman_vs_development_multi_panel_regret": float(spearmanr(d, [row["qmain_development_mean_topk_regret"] for row in rows]).statistic), "spearman_vs_heldout_multi_panel_regret": float(spearmanr(d, held).statistic), "heldout_panel_frontier_area_when_prioritized_by_panel_free_distortion": area(frontier), "heldout_panel_frontier": [{"budget_ratio_requested": budget, **value} for budget, value in zip(BUDGETS, frontier)]}
    pq.write_table(pa.Table.from_pylist(rows), ROOT / f"panel_free_score_distortion_v1_{edge}.parquet", compression="zstd")
    return summary


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = {"status": "panel_free_qmain_score_distortion_development", "distribution": "Q_main_rank_decay_v1", "target_injection": False, "edges": {}}
    for edge in EDGE:
        result["edges"][edge] = evaluate_edge(edge, device)
    (ROOT / "panel_free_score_distortion_v1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
