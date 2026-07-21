"""Enhanced eval: longer KV cache + additional sensitivity metrics.

Addresses two concerns:
1. KV cache too short (512) -> try 1024, more positions for drift to accumulate
2. Metrics not sensitive enough -> add top-K overlap (user-facing recommendation
   change), Recall@100 (more stable than @10), and per-user SIGNED quality delta
   (shows staleness helps some users, hurts others = heterogeneity).

Top-K overlap is the most intuitive staleness metric: "what fraction of your
top-10 recommendations change when using stale KV?" If 30% change, the user
sees a very different recommendation list.
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

MODEL_CFG = dict(hidden_size=256, num_layers=6, num_heads=8, head_dim=64, max_seq_len=1024, activation="relu")
MAX_USERS = 300
BS = 32
K_LIST = [10, 100]
EVAL_DAYS = [3, 5, 8, 10, 13, 15, 17]


def make_model(device, num_items, num_behaviors):
    cfg = HSTUConfig(num_items=num_items, num_behaviors=num_behaviors, **MODEL_CFG)
    return HSTU(cfg).to(device)


def topk_overlap(scores_a, scores_b, k):
    """Per-user top-K overlap fraction. scores: [B, num_items+1]. Returns [B]."""
    top_a = torch.topk(scores_a, k, dim=-1).indices  # [B, k]
    top_b = torch.topk(scores_b, k, dim=-1).indices
    overlaps = []
    for i in range(scores_a.shape[0]):
        inter = len(set(top_a[i].cpu().numpy().tolist()) & set(top_b[i].cpu().numpy().tolist()))
        overlaps.append(inter / k)
    return np.array(overlaps)


def recall_at_k(scores, pos_list, k):
    """Per-user Recall@K. scores: [B, num_items+1]. pos_list: list of sets."""
    topk_idx = torch.topk(scores, k, dim=-1).indices.cpu().numpy()
    recalls = []
    for i in range(scores.shape[0]):
        pos = pos_list[i]
        if not pos:
            recalls.append(0.0); continue
        top_set = set(topk_idx[i].tolist())
        recalls.append(len(pos & top_set) / len(pos))
    return np.array(recalls)


def mrr_per_user(scores, pos_list):
    """Per-user MRR (best positive item)."""
    sc = scores.cpu().numpy()
    mrrs = []
    for i in range(sc.shape[0]):
        pos = pos_list[i]
        if not pos:
            mrrs.append(0.0); continue
        order = np.argsort(-sc[i])
        best_rank = min(int(np.where(order == p)[0][0]) for p in pos)
        mrrs.append(1.0 / (best_rank + 1))
    return np.array(mrrs)


def main():
    device = torch.device("cuda")
    np.random.seed(42)
    torch.manual_seed(0)

    for seq_len in [512, 1024]:
        print(f"\n{'='*60}")
        print(f"=== seq_len = {seq_len} ===")
        print(f"{'='*60}")

        plan = StreamingDataPlan.from_csvs(
            ["data/kuairand/log_standard_4_08_to_4_21_1k.csv",
             "data/kuairand/log_standard_4_22_to_5_08_1k.csv"],
            base_num_days=14, max_seq_len=seq_len, max_items=20000)
        plan.init_base()

        all_items = torch.arange(0, plan.num_items + 1, device=device)
        model_frozen = make_model(device, plan.num_items, plan.num_behaviors)
        model_frozen.load_state_dict(torch.load("checkpoints/streaming_relu/theta_0.pt", map_location=device))
        model_frozen.eval()
        for p in model_frozen.parameters():
            p.requires_grad_(False)
        model_t = make_model(device, plan.num_items, plan.num_behaviors)
        model_prev = make_model(device, plan.num_items, plan.num_behaviors)

        day_results = []

        for di, date in enumerate(plan.stream_dates):
            day_idx = di + 1
            if day_idx not in EVAL_DAYS:
                if day_idx <= 17:
                    plan.ingest_day(date)
                continue

            sd_t = torch.load(f"checkpoints/streaming_relu/theta_{day_idx - 1}.pt", map_location=device)
            model_t.load_state_dict(sd_t)
            model_t.eval()
            has_prev = day_idx >= 2
            if has_prev:
                sd_prev = torch.load(f"checkpoints/streaming_relu/theta_{day_idx - 2}.pt", map_location=device)
                model_prev.load_state_dict(sd_prev)
                model_prev.eval()

            day_df = plan.daily_segments.get(date)
            if day_df is None or len(day_df) == 0:
                plan.ingest_day(date); continue

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
                })
            if not samples:
                plan.ingest_day(date); continue

            pu = {f"top{k}_ov_step": [] for k in K_LIST}
            pu.update({f"top{k}_ov_cum": [] for k in K_LIST})
            pu.update({f"top{k}_ov_frozen": [] for k in K_LIST})
            pu.update({f"recall{k}_fresh": [] for k in K_LIST})
            pu.update({f"recall{k}_frozen": [] for k in K_LIST})
            pu.update({f"recall{k}_stale_cum": [] for k in K_LIST})
            pu["mrr_fresh"] = []; pu["mrr_frozen"] = []
            pu["mrr_stale_step"] = []; pu["mrr_stale_cum"] = []
            pu["sp_step"] = []; pu["sp_cum"] = []; pu["sp_frozen"] = []

            for bi in range(0, len(samples), BS):
                batch = samples[bi:bi + BS]
                B = len(batch)
                max_l = min(max(len(s["item_ids"]) for s in batch), seq_len)
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
                    h_fresh, kv_curr = model_t(item_ids, behs, tds, return_kv=True, return_hidden=True)
                    sc_fresh = model_t.item_emb.score(h_fresh[:, -1, :], all_items.unsqueeze(0).expand(B, -1))

                    h_frozen, kv_0 = model_frozen(item_ids, behs, tds, return_kv=True, return_hidden=True)
                    sc_frozen = model_frozen.item_emb.score(h_frozen[:, -1, :], all_items.unsqueeze(0).expand(B, -1))

                    if has_prev:
                        kv_prev = model_prev.compute_kv(item_ids, behs, tds)
                        h_stale_step = model_t.forward_stale_kv(item_ids, behs, tds, kv_prev)
                        sc_stale_step = model_t.item_emb.score(h_stale_step[:, -1, :], all_items.unsqueeze(0).expand(B, -1))
                    else:
                        sc_stale_step = sc_fresh

                    h_stale_cum = model_t.forward_stale_kv(item_ids, behs, tds, kv_0)
                    sc_stale_cum = model_t.item_emb.score(h_stale_cum[:, -1, :], all_items.unsqueeze(0).expand(B, -1))

                # top-K overlap (lower = more change = more staleness impact)
                for k in K_LIST:
                    pu[f"top{k}_ov_step"].extend(topk_overlap(sc_fresh, sc_stale_step, k).tolist())
                    pu[f"top{k}_ov_cum"].extend(topk_overlap(sc_fresh, sc_stale_cum, k).tolist())
                    pu[f"top{k}_ov_frozen"].extend(topk_overlap(sc_fresh, sc_frozen, k).tolist())
                    pu[f"recall{k}_fresh"].extend(recall_at_k(sc_fresh, pos_list, k).tolist())
                    pu[f"recall{k}_frozen"].extend(recall_at_k(sc_frozen, pos_list, k).tolist())
                    pu[f"recall{k}_stale_cum"].extend(recall_at_k(sc_stale_cum, pos_list, k).tolist())

                # MRR
                pu["mrr_fresh"].extend(mrr_per_user(sc_fresh, pos_list).tolist())
                pu["mrr_frozen"].extend(mrr_per_user(sc_frozen, pos_list).tolist())
                pu["mrr_stale_step"].extend(mrr_per_user(sc_stale_step, pos_list).tolist())
                pu["mrr_stale_cum"].extend(mrr_per_user(sc_stale_cum, pos_list).tolist())

                # Spearman (per user, on valid items)
                for i in range(B):
                    sf = sc_fresh[i, 1:].cpu().numpy()
                    for label, sc_other in [("step", sc_stale_step), ("cum", sc_stale_cum), ("frozen", sc_frozen)]:
                        so = sc_other[i, 1:].cpu().numpy()
                        if np.std(sf) > 1e-8 and np.std(so) > 1e-8:
                            rho, _ = spearmanr(sf, so)
                            pu[f"sp_{label}"].append(1 - (rho if not np.isnan(rho) else 1.0))
                        else:
                            pu[f"sp_{label}"].append(0.0)

            # summarize
            n = len(pu["mrr_fresh"])
            rec = {"day_idx": day_idx, "seq_len": seq_len, "n_eval": n}
            for k in K_LIST:
                ov_step = np.array(pu[f"top{k}_ov_step"])
                ov_cum = np.array(pu[f"top{k}_ov_cum"])
                ov_frz = np.array(pu[f"top{k}_ov_frozen"])
                rec[f"top{k}_change_step%"] = round(float(np.mean(1 - ov_step) * 100), 2)
                rec[f"top{k}_change_cum%"] = round(float(np.mean(1 - ov_cum) * 100), 2)
                rec[f"top{k}_change_frozen%"] = round(float(np.mean(1 - ov_frz) * 100), 2)
                rec[f"top{k}_change_step_p90%"] = round(float(np.percentile(1 - ov_step, 90) * 100), 2)
                rec[f"top{k}_change_cum_p90%"] = round(float(np.percentile(1 - ov_cum, 90) * 100), 2)
                rec[f"recall{k}_fresh"] = round(float(np.mean(pu[f"recall{k}_fresh"])), 5)
                rec[f"recall{k}_frozen"] = round(float(np.mean(pu[f"recall{k}_frozen"])), 5)
                rec[f"recall{k}_stale_cum"] = round(float(np.mean(pu[f"recall{k}_stale_cum"])), 5)

            mrr_f = np.array(pu["mrr_fresh"])
            mrr_z = np.array(pu["mrr_frozen"])
            mrr_sc = np.array(pu["mrr_stale_cum"])
            mrr_ss = np.array(pu["mrr_stale_step"])
            rec["mrr_fresh"] = round(float(np.mean(mrr_f)), 5)
            rec["mrr_frozen"] = round(float(np.mean(mrr_z)), 5)
            rec["mrr_stale_cum"] = round(float(np.mean(mrr_sc)), 5)
            rec["mrr_stale_step"] = round(float(np.mean(mrr_ss)), 5)
            rec["mrr_fresh_win_vs_frozen%"] = round(float(np.mean(mrr_f > mrr_z) * 100), 1)
            rec["mrr_stale_worse_than_fresh%"] = round(float(np.mean(mrr_sc < mrr_f) * 100), 1)
            rec["mrr_stale_better_than_fresh%"] = round(float(np.mean(mrr_sc > mrr_f) * 100), 1)
            # signed quality delta: positive = staleness hurts
            delta_cum = mrr_f - mrr_sc
            rec["mrr_delta_cum_mean"] = round(float(np.mean(delta_cum)), 6)
            rec["mrr_delta_cum_p10"] = round(float(np.percentile(delta_cum, 10)), 6)
            rec["mrr_delta_cum_p90"] = round(float(np.percentile(delta_cum, 90)), 6)
            rec["sp_step_med"] = round(float(np.median(pu["sp_step"])), 5)
            rec["sp_cum_med"] = round(float(np.median(pu["sp_cum"])), 5)
            rec["sp_frozen_med"] = round(float(np.median(pu["sp_frozen"])), 5)

            day_results.append(rec)
            print(f"Day {day_idx:2d} (L={seq_len}): "
                  f"top10 change: step={rec['top10_change_step%']:.1f}% cum={rec['top10_change_cum%']:.1f}% frozen={rec['top10_change_frozen%']:.1f}% | "
                  f"top10 change p90: cum={rec['top10_change_cum_p90%']:.1f}% | "
                  f"R@100: fresh={rec['recall100_fresh']:.4f} frozen={rec['recall100_frozen']:.4f} | "
                  f"MRR fresh_win={rec['mrr_fresh_win_vs_frozen%']:.0f}%")

            plan.ingest_day(date)

        save_json(day_results, f"results/streaming/eval_enhanced_L{seq_len}.json")

    # compare 512 vs 1024
    print(f"\n{'='*60}")
    print("=== 512 vs 1024 comparison (staleness impact) ===")
    import json as _json
    r512 = _json.load(open("results/streaming/eval_enhanced_L512.json")) if Path("results/streaming/eval_enhanced_L512.json").exists() else None
    r1024 = _json.load(open("results/streaming/eval_enhanced_L1024.json")) if Path("results/streaming/eval_enhanced_L1024.json").exists() else None
    if r512 and r1024:
        import json
        print(f'{"day":>3} | {"top10_chg_cum 512":>16} {"top10_chg_cum 1024":>18} | {"R@100 fresh 512":>15} {"R@100 fresh 1024":>16}')
        for d512 in r512:
            d1024 = next((d for d in r1024 if d["day_idx"] == d512["day_idx"]), None)
            if d1024:
                print(f'{d512["day_idx"]:>3} | {d512["top10_change_cum%"]:>16.1f} {d1024["top10_change_cum%"]:>18.1f} | '
                      f'{d512["recall100_fresh"]:>15.4f} {d1024["recall100_fresh"]:>16.4f}')


if __name__ == "__main__":
    import json
    main()
