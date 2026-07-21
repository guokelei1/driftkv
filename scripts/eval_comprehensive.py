"""Comprehensive multi-metric evaluation of streaming KV-cache motivation.

Fixes the issues found in run_complete_motivation.py:
  1. Full-catalog ranking (score ALL items, not random-negative hit@10 which is
     popularity-dominated and climbs to 96% artificially).
  2. Per-step staleness (theta_{t-1} model + theta_{t-2} KV) in addition to
     cumulative staleness (theta_{t-1} model + theta_0 KV). Per-step is the
     realistic operating point; cumulative is the worst case.
  3. Per-user metrics for within-day signal-chain correlation (controls for
     day_idx, avoiding the monotonic-sequence tautology).
  4. Multiple metrics: full-catalog Recall@10/MRR/NDCG, popularity-stratified
     hit@10, Spearman, hidden cosine, KV drift.

Conditions per day t (model = theta_{t-1}, trained on days 1..t-1, predicts day t):
  fresh       : model_t forward with its own KV           (gold standard)
  frozen      : model_0 forward with its own KV           (no streaming update)
  stale_step  : model_t forward_stale_kv with theta_{t-2} KV  (one-step stale, realistic)
  stale_cum   : model_t forward_stale_kv with theta_0 KV      (cumulative stale, worst case)

No train/test leak: eval uses history BEFORE day t + day t's positives; model
theta_{t-1} was trained on days 1..t-1 only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hstu_kvcache.data import StreamingDataPlan
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.utils import save_json

MODEL_CFG = dict(hidden_size=256, num_layers=6, num_heads=8, head_dim=64, max_seq_len=512, activation="relu")
MAX_USERS = 300
FULL_CATALOG = True
K = 10


def make_model(device, num_items, num_behaviors):
    cfg = HSTUConfig(num_items=num_items, num_behaviors=num_behaviors, **MODEL_CFG)
    return HSTU(cfg).to(device)


def score_all_items(model, hidden_last, all_items):
    """Score all items. hidden_last: [B, H]. all_items: [num_items+1]. Returns [B, num_items+1]."""
    with torch.no_grad():
        return model.item_emb.score(hidden_last, all_items.unsqueeze(0).expand(hidden_last.shape[0], -1))


def full_catalog_metrics(scores, pos_items_list, all_item_ids, k=K):
    """Per-user full-catalog ranking metrics.

    scores: [B, num_items+1] scores for all items (index 0 = padding, ignore).
    pos_items_list: list of sets of positive item ids per user.
    Returns per-user arrays: recall@k, mrr, ndcg, best_rank.
    """
    B = scores.shape[0]
    recalls, mrrs, ndcgs, best_ranks = [], [], [], []
    for i in range(B):
        pos = pos_items_list[i]
        if not pos:
            recalls.append(0.0); mrrs.append(0.0); ndcgs.append(0.0); best_ranks.append(99999)
            continue
        sc = scores[i].cpu().numpy()
        order = np.argsort(-sc)
        ranks = []
        for p in pos:
            idx = int(np.where(order == p)[0][0])
            ranks.append(idx)
        best = min(ranks)
        best_ranks.append(best)
        mrrs.append(1.0 / (best + 1))
        in_topk = [r for r in ranks if r < k]
        recalls.append(len(in_topk) / len(pos))
        dcg = sum(1.0 / np.log2(r + 2) for r in in_topk)
        idcg = sum(1.0 / np.log2(j + 2) for j in range(min(len(pos), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return np.array(recalls), np.array(mrrs), np.array(ndcgs), np.array(best_ranks)


def popstrat_hit_at_k(scores, pos_items_list, item_pop_bins, num_neg=1000, k=K, rng=None):
    """Popularity-stratified hit@k: sample negatives proportional to sqrt(popularity).

    Harder than uniform-random (which over-samples unpopular items) and more
    realistic than interaction-weighted (which over-samples popular items).
    """
    if rng is None:
        rng = np.random.default_rng()
    B = scores.shape[0]
    hits = []
    for i in range(B):
        pos = list(pos_items_list[i])
        if not pos:
            hits.append(0.0); continue
        negs = rng.choice(item_pop_bins, size=num_neg, replace=True)
        cands = np.unique(np.concatenate([pos, negs]))
        cands = cands[cands > 0]
        cands_t = torch.tensor(cands, dtype=torch.long, device=scores.device)
        sc = scores[i, cands_t].cpu().numpy()
        order = np.argsort(-sc)
        pos_mask = np.array([c in set(pos) for c in cands])
        ranked_pos = np.where(pos_mask[order])[0]
        hits.append(1.0 if len(ranked_pos) > 0 and ranked_pos[0] < k else 0.0)
    return np.array(hits)


def per_user_spearman(scores_a, scores_b, valid_slice):
    """Per-user Spearman correlation of full-catalog scores between two conditions."""
    sa = scores_a[:, valid_slice].cpu().numpy()
    sb = scores_b[:, valid_slice].cpu().numpy()
    rhos = []
    for i in range(sa.shape[0]):
        if np.std(sa[i]) < 1e-8 or np.std(sb[i]) < 1e-8:
            rhos.append(1.0)
        else:
            rho, _ = spearmanr(sa[i], sb[i])
            rhos.append(rho if not np.isnan(rho) else 1.0)
    return np.array(rhos)


def main():
    device = torch.device("cuda")
    np.random.seed(42)
    torch.manual_seed(0)
    seq_len = MODEL_CFG["max_seq_len"]

    print("=== Comprehensive Multi-Metric Streaming KV Evaluation ===")
    plan = StreamingDataPlan.from_csvs(
        ["data/kuairand/log_standard_4_08_to_4_21_1k.csv",
         "data/kuairand/log_standard_4_22_to_5_08_1k.csv"],
        base_num_days=14, max_seq_len=seq_len, max_items=20000)
    plan.init_base()
    print(f"users={plan.num_users} items={plan.num_items} seq_len={seq_len}")

    # item popularity bins for stratified negative sampling
    all_interactions = plan.trace.interactions
    item_pop = all_interactions["item_idx"].value_counts().to_dict()
    item_pop_arr = np.array([item_pop.get(i, 1) for i in range(plan.num_items + 1)], dtype=np.float64)
    item_pop_sqrt = np.sqrt(item_pop_arr)
    item_pop_bins = np.where(item_pop_sqrt > 0)[0]
    item_pop_bins = item_pop_bins[item_pop_bins > 0]

    all_items = torch.arange(0, plan.num_items + 1, device=device)
    valid_slice = slice(1, plan.num_items + 1)

    # load frozen model (theta_0) once
    model_frozen = make_model(device, plan.num_items, plan.num_behaviors)
    model_frozen.load_state_dict(torch.load("checkpoints/streaming_relu/theta_0.pt", map_location=device))
    model_frozen.eval()
    for p in model_frozen.parameters():
        p.requires_grad_(False)

    # working models (swap state dicts)
    model_t = make_model(device, plan.num_items, plan.num_behaviors)
    model_prev = make_model(device, plan.num_items, plan.num_behaviors)

    results = {"model_cfg": MODEL_CFG, "num_items": plan.num_items, "days": [], "per_user_days": []}
    rng = np.random.default_rng(42)

    for di, date in enumerate(plan.stream_dates):
        day_idx = di + 1
        if day_idx > 17:
            break

        # theta_{t-1}: model trained on days 1..t-1, predicts day t
        sd_t = torch.load(f"checkpoints/streaming_relu/theta_{day_idx - 1}.pt", map_location=device)
        model_t.load_state_dict(sd_t)
        model_t.eval()

        # theta_{t-2}: for per-step stale KV (skip day 1, no previous)
        has_prev = day_idx >= 2
        if has_prev:
            sd_prev = torch.load(f"checkpoints/streaming_relu/theta_{day_idx - 2}.pt", map_location=device)
            model_prev.load_state_dict(sd_prev)
            model_prev.eval()

        day_df = plan.daily_segments.get(date)
        if day_df is None or len(day_df) == 0:
            plan.ingest_day(date)
            continue

        # build eval samples: history (before today) + today's pos items
        samples = []
        for u in day_df["user_idx"].unique()[:MAX_USERS]:
            u = int(u)
            if u not in plan.user_histories or len(plan.user_histories[u]["item_ids"]) < 2:
                continue
            grp = day_df[day_df["user_idx"] == u].sort_values("time_ms")
            pos_items = grp["item_idx"].unique()
            if len(pos_items) == 0:
                continue
            hist = plan.user_histories[u]
            samples.append({
                "item_ids": hist["item_ids"][-seq_len:],
                "behaviors": hist["behaviors"][-seq_len:],
                "time_deltas": hist["time_deltas"][-seq_len:],
                "pos_items": pos_items.tolist(),
                "user_id": u,
            })
        if not samples:
            plan.ingest_day(date)
            continue

        # per-user accumulators
        pu = {
            cond: {"recall": [], "mrr": [], "ndcg": [], "best_rank": [], "pop_hit": [], "h_cos": []}
            for cond in ["fresh", "frozen", "stale_step", "stale_cum"]
        }
        pu["kv_drift_step"] = []
        pu["kv_drift_cum"] = []
        pu["spearman_step"] = []
        pu["spearman_cum"] = []
        pu["spearman_frozen"] = []
        pu["seq_lens"] = []
        pu["user_ids"] = []

        # batch processing
        BS = 16
        for bi in range(0, len(samples), BS):
            batch = samples[bi:bi + BS]
            B = len(batch)
            max_l = max(len(s["item_ids"]) for s in batch)
            max_l = min(max_l, seq_len)
            item_ids = torch.zeros(B, max_l, dtype=torch.long, device=device)
            behs = torch.zeros(B, max_l, dtype=torch.long, device=device)
            tds = torch.zeros(B, max_l, dtype=torch.float, device=device)
            pos_list = []
            for i, s in enumerate(batch):
                arr = s["item_ids"][-max_l:]
                n = len(arr)
                item_ids[i, :n] = torch.tensor(arr, device=device)
                behs[i, :n] = torch.tensor(s["behaviors"][-max_l:], device=device)
                tds[i, :n] = torch.tensor(s["time_deltas"][-max_l:], device=device)
                pos_list.append(set(s["pos_items"]))

            with torch.no_grad():
                # 1. FRESH: model_t forward with own KV
                h_fresh, kv_curr = model_t(item_ids, behs, tds, return_kv=True, return_hidden=True)
                sc_fresh = score_all_items(model_t, h_fresh[:, -1, :], all_items)

                # 2. FROZEN: model_0 forward with own KV
                h_frozen, kv_0 = model_frozen(item_ids, behs, tds, return_kv=True, return_hidden=True)
                sc_frozen = score_all_items(model_frozen, h_frozen[:, -1, :], all_items)

                # 3. STALE per-step: model_t + theta_{t-2} KV
                if has_prev:
                    kv_prev = model_prev.compute_kv(item_ids, behs, tds)
                    h_stale_step = model_t.forward_stale_kv(item_ids, behs, tds, kv_prev)
                    sc_stale_step = score_all_items(model_t, h_stale_step[:, -1, :], all_items)
                    # per-user KV drift (per-step)
                    for i in range(B):
                        dk = (kv_curr.k[:, i:i+1] - kv_prev.k[:, i:i+1]).float().norm().item()
                        dv = (kv_curr.v[:, i:i+1] - kv_prev.v[:, i:i+1]).float().norm().item()
                        base = (kv_curr.k[:, i:i+1].float().norm() + kv_curr.v[:, i:i+1].float().norm()).item()
                        pu["kv_drift_step"].append((dk + dv) / (base + 1e-12))
                else:
                    h_stale_step = h_fresh
                    sc_stale_step = sc_fresh
                    for i in range(B):
                        pu["kv_drift_step"].append(0.0)

                # 4. STALE cumulative: model_t + theta_0 KV
                h_stale_cum = model_t.forward_stale_kv(item_ids, behs, tds, kv_0)
                sc_stale_cum = score_all_items(model_t, h_stale_cum[:, -1, :], all_items)
                # per-user KV drift (cumulative)
                for i in range(B):
                    dk = (kv_curr.k[:, i:i+1] - kv_0.k[:, i:i+1]).float().norm().item()
                    dv = (kv_curr.v[:, i:i+1] - kv_0.v[:, i:i+1]).float().norm().item()
                    base = (kv_curr.k[:, i:i+1].float().norm() + kv_curr.v[:, i:i+1].float().norm()).item()
                    pu["kv_drift_cum"].append((dk + dv) / (base + 1e-12))

                # per-user full-catalog metrics
                for cond, sc, h in [("fresh", sc_fresh, h_fresh),
                                     ("frozen", sc_frozen, h_frozen),
                                     ("stale_step", sc_stale_step, h_stale_step),
                                     ("stale_cum", sc_stale_cum, h_stale_cum)]:
                    rec, mrr, ndcg, br = full_catalog_metrics(sc, pos_list, all_items, k=K)
                    pu[cond]["recall"].extend(rec.tolist())
                    pu[cond]["mrr"].extend(mrr.tolist())
                    pu[cond]["ndcg"].extend(ndcg.tolist())
                    pu[cond]["best_rank"].extend(br.tolist())
                    ph = popstrat_hit_at_k(sc, pos_list, item_pop_bins, num_neg=1000, k=K, rng=rng)
                    pu[cond]["pop_hit"].extend(ph.tolist())
                    # hidden cosine vs fresh
                    if cond == "fresh":
                        pu[cond]["h_cos"].extend([1.0] * B)
                    else:
                        cos = torch.nn.functional.cosine_similarity(
                            h_fresh[:, -1, :], h[:, -1, :], dim=-1).cpu().numpy()
                        pu[cond]["h_cos"].extend(cos.tolist())

                # per-user Spearman (full catalog)
                pu["spearman_step"].extend(per_user_spearman(sc_fresh, sc_stale_step, valid_slice).tolist())
                pu["spearman_cum"].extend(per_user_spearman(sc_fresh, sc_stale_cum, valid_slice).tolist())
                pu["spearman_frozen"].extend(per_user_spearman(sc_fresh, sc_frozen, valid_slice).tolist())

                for s in batch:
                    pu["seq_lens"].append(len(s["item_ids"]))
                    pu["user_ids"].append(s["user_id"])

        # day-level aggregates
        n = len(pu["fresh"]["recall"])
        rec = {
            "date": date, "day_idx": day_idx, "n_eval": n, "has_per_step": has_prev,
        }
        for cond in ["fresh", "frozen", "stale_step", "stale_cum"]:
            rec[f"{cond}_recall@10"] = round(float(np.mean(pu[cond]["recall"])), 5)
            rec[f"{cond}_mrr"] = round(float(np.mean(pu[cond]["mrr"])), 5)
            rec[f"{cond}_ndcg@10"] = round(float(np.mean(pu[cond]["ndcg"])), 5)
            rec[f"{cond}_pop_hit@10"] = round(float(np.mean(pu[cond]["pop_hit"])), 5)
            rec[f"{cond}_best_rank_med"] = round(float(np.median(pu[cond]["best_rank"])), 1)
            rec[f"{cond}_h_cos"] = round(float(np.mean(pu[cond]["h_cos"])), 5)
        rec["kv_drift_step_med"] = round(float(np.median(pu["kv_drift_step"])), 5)
        rec["kv_drift_cum_med"] = round(float(np.median(pu["kv_drift_cum"])), 5)
        rec["spearman_step_med"] = round(float(np.median(pu["spearman_step"])), 5)
        rec["spearman_cum_med"] = round(float(np.median(pu["spearman_cum"])), 5)
        rec["spearman_frozen_med"] = round(float(np.median(pu["spearman_frozen"])), 5)

        results["days"].append(rec)
        results["per_user_days"].append(pu)

        print(
            f"Day {day_idx:2d}: fresh_recall={rec['fresh_recall@10']:.4f} mrr={rec['fresh_mrr']:.4f} | "
            f"frozen_recall={rec['frozen_recall@10']:.4f} mrr={rec['frozen_mrr']:.4f} | "
            f"stale_step_recall={rec['stale_step_recall@10']:.4f} | "
            f"stale_cum_recall={rec['stale_cum_recall@10']:.4f} | "
            f"1-sp(step)={1-rec['spearman_step_med']:.4f} 1-sp(cum)={1-rec['spearman_cum_med']:.4f}"
        )

        plan.ingest_day(date)

    save_json(results, "results/streaming/eval_comprehensive.json")
    print("\nSaved results/streaming/eval_comprehensive.json")

    # === Summary ===
    print("\n=== Summary ===")
    d0 = results["days"][1] if len(results["days"]) > 1 else results["days"][0]
    dlast = results["days"][-1]
    print("Full-catalog Recall@10 (harder, less popularity-dominated):")
    print(f"  Day {d0['day_idx']}: fresh={d0['fresh_recall@10']:.4f} frozen={d0['frozen_recall@10']:.4f} "
          f"stale_step={d0['stale_step_recall@10']:.4f} stale_cum={d0['stale_cum_recall@10']:.4f}")
    print(f"  Day {dlast['day_idx']}: fresh={dlast['fresh_recall@10']:.4f} frozen={dlast['frozen_recall@10']:.4f} "
          f"stale_step={dlast['stale_step_recall@10']:.4f} stale_cum={dlast['stale_cum_recall@10']:.4f}")
    print("Full-catalog MRR:")
    print(f"  Day {d0['day_idx']}: fresh={d0['fresh_mrr']:.4f} frozen={d0['frozen_mrr']:.4f}")
    print(f"  Day {dlast['day_idx']}: fresh={dlast['fresh_mrr']:.4f} frozen={dlast['frozen_mrr']:.4f}")

    # per-user signal chain (within-day correlation)
    print("\nPer-user signal chain (within-day Spearman: kv_drift vs 1-spearman):")
    for di_idx, pu in enumerate(results["per_user_days"]):
        if di_idx < 1:
            continue
        day_idx = results["days"][di_idx]["day_idx"]
        kv_d = np.array(pu["kv_drift_step"])
        rank_loss = 1 - np.array(pu["spearman_step"])
        if len(kv_d) > 5 and np.std(kv_d) > 1e-8 and np.std(rank_loss) > 1e-8:
            rho, pval = spearmanr(kv_d, rank_loss)
            print(f"  Day {day_idx}: rho={rho:.3f} p={pval:.4f} (n={len(kv_d)})")

    # cross-day per-user signal chain (pooled within-day residuals)
    all_resid_kv = []
    all_resid_loss = []
    for di_idx, pu in enumerate(results["per_user_days"]):
        if di_idx < 1:
            continue
        kv_d = np.array(pu["kv_drift_step"])
        rank_loss = 1 - np.array(pu["spearman_step"])
        all_resid_kv.extend(kv_d.tolist())
        all_resid_loss.extend(rank_loss.tolist())
    if len(all_resid_kv) > 10:
        rho, pval = spearmanr(all_resid_kv, all_resid_loss)
        print(f"  Pooled per-user: rho={rho:.3f} p={pval:.4e} (n={len(all_resid_kv)})")


if __name__ == "__main__":
    main()
