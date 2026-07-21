"""Test different streaming update window sizes.

Instead of 1-day streaming updates (100 steps/day, small dtheta), use larger
windows: 3, 5, 10 days per update. Each update trains on W days of data with
100*W steps, producing a LARGER dtheta per update.

Hypothesis: larger dtheta per update -> larger KV drift -> visible quality
degradation from stale KV (which was invisible at W=1).

For each window size W:
  1. Reset to theta_0 (base model, 12 epochs on 14 days)
  2. Stream train: for each W-day window, ingest W days, train 100*W steps
  3. At each checkpoint, eval: fresh (theta_t KV) vs stale (theta_0 KV)
     - top-10 overlap, MRR, R@100, 1-Spearman

Also tests per-window staleness: theta_t model + theta_{t-1} KV (one window old).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hstu_kvcache.data import StreamingDataPlan
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.utils import save_json

MODEL_CFG = dict(hidden_size=256, num_layers=6, num_heads=8, head_dim=64, max_seq_len=512, activation="relu")
BASE_EPOCHS = 12
STEPS_PER_DAY = 100
MAX_USERS = 300
BS = 32
WINDOW_SIZES = [1, 3, 5, 10]
EVAL_USERS = 300


def make_model(device, num_items, num_behaviors):
    return HSTU(HSTUConfig(num_items=num_items, num_behaviors=num_behaviors, **MODEL_CFG)).to(device)


def score_all(model, h_last, all_items, B):
    return model.item_emb.score(h_last, all_items.unsqueeze(0).expand(B, -1))


def topk_overlap(scores_a, scores_b, k=10):
    top_a = torch.topk(scores_a, k, dim=-1).indices.cpu().numpy()
    top_b = torch.topk(scores_b, k, dim=-1).indices.cpu().numpy()
    return np.array([len(set(top_a[i]) & set(top_b[i])) / k for i in range(scores_a.shape[0])])


def recall_at_k(scores, pos_list, k=100):
    topk = torch.topk(scores, k, dim=-1).indices.cpu().numpy()
    return np.array([
        len(set(pos_list[i]) & set(topk[i])) / max(len(pos_list[i]), 1)
        for i in range(scores.shape[0])
    ])


def mrr_per_user(scores, pos_list):
    sc = scores.cpu().numpy()
    return np.array([
        1.0 / (min(int(np.where(np.argsort(-sc[i]) == p)[0][0]) for p in pos_list[i]) + 1)
        if pos_list[i] else 0.0
        for i in range(sc.shape[0])
    ])


def train_base(model, plan, optimizer, device):
    print("Base training (12 epochs)...", flush=True)
    for epoch in range(BASE_EPOCHS):
        for batch in plan.iter_base_train_batches(batch_size=BS):
            from hstu_kvcache.streaming.trainer import train_step
            train_step(model, batch, optimizer, device)
    print(f"  done. ||theta_0||={model_params_vec(model).norm().item():.2f}", flush=True)


def stream_train_windowed(model, plan, optimizer, device, window_size, replay_buffer):
    """Stream train with W-day windows. Returns list of (step_idx, checkpoint, eval_date, eval_samples)."""
    checkpoints = []  # [(window_idx, state_dict, theta_vec, eval_date, eval_samples)]
    theta0_vec = model_params_vec(model).detach().clone()

    window_idx = 0
    stream_dates = plan.stream_dates
    di = 0
    while di < len(stream_dates):
        # eval on the FIRST day of the next window (before ingest)
        eval_date = stream_dates[di]
        eval_samples = plan.get_eval_set(eval_date, max_users=EVAL_USERS)

        # ingest W days
        for w in range(window_size):
            if di + w < len(stream_dates):
                plan.ingest_day(stream_dates[di + w])

        # collect training batches for this window
        window_batches = []
        for w in range(window_size):
            if di + w < len(stream_dates):
                day_batches = list(plan.iter_train_batches(stream_dates[di + w], batch_size=BS))
                window_batches.extend(day_batches)

        # train 100*W steps with replay
        n_new = int(len(window_batches) * (1 - 0.5))
        np.random.shuffle(window_batches)
        mixed = window_batches[:n_new]
        for _ in range(len(window_batches) - n_new):
            if replay_buffer:
                mixed.append(copy.deepcopy(replay_buffer[np.random.randint(len(replay_buffer))]))
        np.random.shuffle(mixed)

        steps_trained = 0
        from hstu_kvcache.streaming.trainer import train_step
        for batch in mixed:
            train_step(model, batch, optimizer, device)
            steps_trained += 1
            if steps_trained >= STEPS_PER_DAY * window_size:
                break
        # update replay buffer
        for batch in window_batches[:10]:
            replay_buffer.append({k: v.clone() for k, v in batch.items()})
            if len(replay_buffer) > 300:
                replay_buffer.pop(np.random.randint(len(replay_buffer)))

        theta_vec = model_params_vec(model).detach().clone()
        dtheta_rel = (theta_vec - theta0_vec).norm().item() / (theta0_vec.norm().item() + 1e-12)
        checkpoints.append({
            "window_idx": window_idx,
            "dtheta_rel": dtheta_rel,
            "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "eval_date": eval_date,
            "eval_samples": eval_samples,
            "steps_trained": steps_trained,
        })
        print(f"  W={window_size} window {window_idx}: dtheta_rel={dtheta_rel:.4f} steps={steps_trained} eval_date={eval_date}", flush=True)

        window_idx += 1
        di += window_size

    return checkpoints, theta0_vec


def eval_staleness(model, model_frozen, checkpoints, theta0_state, plan, device, all_items):
    """For each checkpoint, eval fresh vs stale (theta_0 KV) vs frozen."""
    results = []
    seq_len = MODEL_CFG["max_seq_len"]

    for ckpt in checkpoints:
        eval_samples = ckpt["eval_samples"]
        if not eval_samples:
            continue

        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        model_frozen.load_state_dict(theta0_state)
        model_frozen.eval()

        # build user batches
        samples = []
        for s in eval_samples[:MAX_USERS]:
            seq = s["history"]
            if len(seq["item_ids"]) < 2:
                continue
            samples.append({
                "item_ids": seq["item_ids"][-seq_len:],
                "behaviors": seq["behaviors"][-seq_len:],
                "time_deltas": seq["time_deltas"][-seq_len:],
                "pos_items": s["pos_items"],
            })
        if not samples:
            continue

        all_top10_chg = []
        all_mrr_fresh = []
        all_mrr_stale = []
        all_r100_fresh = []
        all_r100_stale = []
        all_sp = []

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
                # fresh: theta_t model + theta_t KV
                h_fresh, kv_fresh = model(item_ids, behs, tds, return_kv=True, return_hidden=True)
                sc_fresh = score_all(model, h_fresh[:, -1, :], all_items, B)

                # stale: theta_t model + theta_0 KV
                kv_0 = model_frozen.compute_kv(item_ids, behs, tds)
                h_stale = model.forward_stale_kv(item_ids, behs, tds, kv_0)
                sc_stale = score_all(model, h_stale[:, -1, :], all_items, B)

                # frozen: theta_0 model + theta_0 KV
                h_frozen, _ = model_frozen(item_ids, behs, tds, return_kv=False, return_hidden=True)
                sc_frozen = score_all(model_frozen, h_frozen[:, -1, :], all_items, B)

            ov = topk_overlap(sc_fresh, sc_stale, k=10)
            all_top10_chg.extend((1 - ov).tolist())
            all_mrr_fresh.extend(mrr_per_user(sc_fresh, pos_list).tolist())
            all_mrr_stale.extend(mrr_per_user(sc_stale, pos_list).tolist())
            all_r100_fresh.extend(recall_at_k(sc_fresh, pos_list, k=100).tolist())
            all_r100_stale.extend(recall_at_k(sc_stale, pos_list, k=100).tolist())
            for i in range(B):
                sf = sc_fresh[i, 1:].cpu().numpy()
                ss = sc_stale[i, 1:].cpu().numpy()
                if np.std(sf) > 1e-8 and np.std(ss) > 1e-8:
                    rho, _ = spearmanr(sf, ss)
                    all_sp.append(1 - (rho if not np.isnan(rho) else 1.0))

        mrr_f = np.array(all_mrr_fresh)
        mrr_s = np.array(all_mrr_stale)
        r100_f = np.array(all_r100_fresh)
        r100_s = np.array(all_r100_stale)
        rec = {
            "window_idx": ckpt["window_idx"],
            "dtheta_rel": round(ckpt["dtheta_rel"], 5),
            "n_eval": len(mrr_f),
            "top10_change%": round(float(np.mean(all_top10_chg) * 100), 2),
            "top10_change_p90%": round(float(np.percentile(all_top10_chg, 90) * 100), 2),
            "mrr_fresh": round(float(np.mean(mrr_f)), 5),
            "mrr_stale": round(float(np.mean(mrr_s)), 5),
            "mrr_diff": round(float(np.mean(mrr_f) - np.mean(mrr_s)), 6),
            "mrr_stale_worse%": round(float(np.mean(mrr_s < mrr_f) * 100), 1),
            "r100_fresh": round(float(np.mean(r100_f)), 5),
            "r100_stale": round(float(np.mean(r100_s)), 5),
            "r100_diff": round(float(np.mean(r100_f) - np.mean(r100_s)), 6),
            "1-spearman": round(float(np.median(all_sp)), 5),
        }
        results.append(rec)
        print(f"    window {ckpt['window_idx']}: dtheta={rec['dtheta_rel']:.4f} top10_chg={rec['top10_change%']:.1f}% "
              f"MRR fresh={rec['mrr_fresh']:.5f} stale={rec['mrr_stale']:.5f} diff={rec['mrr_diff']:.6f} "
              f"stale_worse={rec['mrr_stale_worse%']:.0f}% R@100 diff={rec['r100_diff']:.6f}", flush=True)

    return results


def main():
    device = torch.device("cuda")
    np.random.seed(42)
    torch.manual_seed(0)
    seq_len = MODEL_CFG["max_seq_len"]

    print("=== Streaming Window Size Comparison ===", flush=True)
    plan = StreamingDataPlan.from_csvs(
        ["data/kuairand/log_standard_4_08_to_4_21_1k.csv",
         "data/kuairand/log_standard_4_22_to_5_08_1k.csv"],
        base_num_days=14, max_seq_len=seq_len, max_items=20000)
    plan.init_base()
    print(f"users={plan.num_users} items={plan.num_items}", flush=True)

    all_items = torch.arange(0, plan.num_items + 1, device=device)

    # train base model once
    model = make_model(device, plan.num_items, plan.num_behaviors)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_base(model, plan, optimizer, device)
    theta0_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model_frozen = make_model(device, plan.num_items, plan.num_behaviors)
    model_frozen.load_state_dict(theta0_state)
    model_frozen.eval()
    for p in model_frozen.parameters():
        p.requires_grad_(False)

    all_results = {}

    for W in WINDOW_SIZES:
        print(f"\n{'='*50}", flush=True)
        print(f"=== Window size W={W} (train {STEPS_PER_DAY*W} steps per {W}-day window) ===", flush=True)

        # reset to theta_0
        model.load_state_dict(theta0_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        replay_buffer = []

        # need fresh plan (ingest_day mutates state)
        plan_w = StreamingDataPlan.from_csvs(
            ["data/kuairand/log_standard_4_08_to_4_21_1k.csv",
             "data/kuairand/log_standard_4_22_to_5_08_1k.csv"],
            base_num_days=14, max_seq_len=seq_len, max_items=20000)
        plan_w.init_base()

        checkpoints, theta0_vec = stream_train_windowed(
            model, plan_w, optimizer, device, W, replay_buffer)

        # eval
        results = eval_staleness(model, model_frozen, checkpoints, theta0_state, plan_w, device, all_items)
        all_results[f"W{W}"] = results

    save_json(all_results, "results/streaming/window_comparison.json")
    print(f"\n{'='*50}")
    print("=== SUMMARY: stale KV quality degradation by window size ===")
    print(f'{"W":>3} {"window":>3} {"dtheta":>7} {"top10chg":>8} {"MRR fresh":>9} {"MRR stale":>9} {"MRR diff":>8} {"stale<":>6} {"R100diff":>8}')
    for W in WINDOW_SIZES:
        for r in all_results.get(f"W{W}", []):
            print(f'{W:>3} {r["window_idx"]:>3} {r["dtheta_rel"]:>7.4f} {r["top10_change%"]:>7.1f}% '
                  f'{r["mrr_fresh"]:>9.5f} {r["mrr_stale"]:>9.5f} {r["mrr_diff"]:>8.6f} {r["mrr_stale_worse%"]:>5.0f}% '
                  f'{r["r100_diff"]:>8.6f}')

    print("\nSaved results/streaming/window_comparison.json")


if __name__ == "__main__":
    main()
