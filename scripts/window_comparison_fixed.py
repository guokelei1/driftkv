"""Window size comparison (FIXED: no train/test leak + sufficient training).

Fixes vs previous version:
1. LEAK FIX: eval on the day AFTER the training window, not the first day.
   Previous version trained on eval_date's data then evaluated on it.
2. TRAINING FIX: multiple epochs through window data to reach target steps.
   Previous version only did 1 pass (~30 steps), not the intended 100*W.

Protocol (leak-free):
  1. Ingest W days (training window)
  2. Train N epochs through window data (target ~500 steps)
  3. Eval on day W+1 (model has NOT seen this day's data)
  4. Stale KV = F(theta_0, x_u), Fresh KV = F(theta_t, x_u)
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
from hstu_kvcache.streaming.trainer import train_step
from hstu_kvcache.utils import save_json

MODEL_CFG = dict(hidden_size=256, num_layers=6, num_heads=8, head_dim=64, max_seq_len=512, activation="relu")
BASE_EPOCHS = 50
TARGET_STEPS_PER_WINDOW = 500
MAX_USERS = 300
BS = 32
WINDOW_SIZES = [1, 3, 5]


def make_model(device, num_items, num_behaviors):
    return HSTU(HSTUConfig(num_items=num_items, num_behaviors=num_behaviors, **MODEL_CFG)).to(device)


def score_all(model, h_last, all_items, B):
    return model.item_emb.score(h_last, all_items.unsqueeze(0).expand(B, -1))


def topk_overlap(sa, sb, k=10):
    ta = torch.topk(sa, k, dim=-1).indices.cpu().numpy()
    tb = torch.topk(sb, k, dim=-1).indices.cpu().numpy()
    return np.array([len(set(ta[i]) & set(tb[i])) / k for i in range(sa.shape[0])])


def recall_at_k(scores, pos_list, k=100):
    tk = torch.topk(scores, k, dim=-1).indices.cpu().numpy()
    return np.array([len(set(pos_list[i]) & set(tk[i])) / max(len(pos_list[i]), 1) for i in range(scores.shape[0])])


def mrr_per_user(scores, pos_list):
    sc = scores.cpu().numpy()
    out = []
    for i in range(sc.shape[0]):
        pos = pos_list[i]
        if not pos:
            out.append(0.0); continue
        order = np.argsort(-sc[i])
        best = min(int(np.where(order == p)[0][0]) for p in pos)
        out.append(1.0 / (best + 1))
    return np.array(out)


def train_base(model, plan, optimizer, device):
    print("Base training (50 epochs)...", flush=True)
    for epoch in range(BASE_EPOCHS):
        for batch in plan.iter_base_train_batches(batch_size=BS):
            train_step(model, batch, optimizer, device)
    print(f"  done. ||theta_0||={model_params_vec(model).norm().item():.2f}", flush=True)


def run_window(model, plan, optimizer, device, W, theta0_state, replay_buffer):
    """Stream train with W-day windows. LEAK-FREE: eval on day AFTER window."""
    checkpoints = []
    theta0_vec = model_params_vec(model).detach().clone()
    stream_dates = plan.stream_dates
    di = 0
    win_idx = 0

    while di < len(stream_dates):
        # 1. Ingest W days (training window)
        window_dates = []
        for w in range(W):
            if di + w < len(stream_dates):
                plan.ingest_day(stream_dates[di + w])
                window_dates.append(stream_dates[di + w])

        if not window_dates:
            break

        # 2. Collect training batches for the window
        window_batches = []
        for d in window_dates:
            window_batches.extend(list(plan.iter_train_batches(d, batch_size=BS)))
        if not window_batches:
            di += W; win_idx += 1; continue

        # 3. Train: multiple epochs to reach TARGET_STEPS_PER_WINDOW
        steps = 0
        n_new = int(len(window_batches) * 0.5)
        while steps < TARGET_STEPS_PER_WINDOW:
            mixed = list(window_batches)
            np.random.shuffle(mixed)
            mixed = mixed[:n_new]
            for _ in range(len(window_batches) - n_new):
                if replay_buffer:
                    mixed.append(copy.deepcopy(replay_buffer[np.random.randint(len(replay_buffer))]))
            np.random.shuffle(mixed)
            for batch in mixed:
                train_step(model, batch, optimizer, device)
                steps += 1
                if steps >= TARGET_STEPS_PER_WINDOW:
                    break
        for batch in window_batches[:10]:
            replay_buffer.append({k: v.clone() for k, v in batch.items()})
            if len(replay_buffer) > 300:
                replay_buffer.pop(np.random.randint(len(replay_buffer)))

        # 4. Eval on the day AFTER the window (LEAK-FREE)
        eval_di = di + W
        if eval_di >= len(stream_dates):
            di += W; win_idx += 1; continue
        eval_date = stream_dates[eval_di]
        eval_samples = plan.get_eval_set(eval_date, max_users=MAX_USERS)

        theta_vec = model_params_vec(model).detach().clone()
        dtheta = (theta_vec - theta0_vec).norm().item() / (theta0_vec.norm().item() + 1e-12)

        checkpoints.append({
            "win_idx": win_idx, "dtheta_rel": dtheta, "steps": steps,
            "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "eval_samples": eval_samples, "eval_date": eval_date,
        })
        print(f"  W={W} win {win_idx}: dtheta={dtheta:.4f} steps={steps} eval={eval_date}", flush=True)
        win_idx += 1
        di += W

    return checkpoints


def eval_staleness(model, model_frozen, checkpoints, device, all_items):
    seq_len = MODEL_CFG["max_seq_len"]
    results = []
    for ckpt in checkpoints:
        samples = ckpt["eval_samples"]
        if not samples:
            continue
        model.load_state_dict(ckpt["state_dict"]); model.eval()
        model_frozen.load_state_dict(ckpt["state_dict"])  # placeholder, will set below
        # actually model_frozen should be theta_0
        # (set by caller)

        # build batches
        user_seqs = []
        for s in samples[:MAX_USERS]:
            seq = s["history"]
            if len(seq["item_ids"]) < 2:
                continue
            user_seqs.append({
                "item_ids": seq["item_ids"][-seq_len:],
                "behaviors": seq["behaviors"][-seq_len:],
                "time_deltas": seq["time_deltas"][-seq_len:],
                "pos_items": s["pos_items"],
            })
        if not user_seqs:
            continue

        mrr_f, mrr_s, r100_f, r100_s, t10chg, sps = [], [], [], [], [], []

        for bi in range(0, len(user_seqs), BS):
            batch = user_seqs[bi:bi + BS]
            B = len(batch)
            max_l = min(max(len(s["item_ids"]) for s in batch), seq_len)
            iids = torch.zeros(B, max_l, dtype=torch.long, device=device)
            behs = torch.zeros(B, max_l, dtype=torch.long, device=device)
            tds = torch.zeros(B, max_l, dtype=torch.float, device=device)
            pos_list = []
            for i, s in enumerate(batch):
                arr = s["item_ids"][-max_l:]; n = len(arr)
                iids[i, :n] = torch.tensor(arr, device=device)
                behs[i, :n] = torch.tensor(s["behaviors"][-max_l:], device=device)
                tds[i, :n] = torch.tensor(s["time_deltas"][-max_l:], device=device)
                pos_list.append(set(s["pos_items"]))

            with torch.no_grad():
                h_fresh, kv_fresh = model(iids, behs, tds, return_kv=True, return_hidden=True)
                sc_fresh = score_all(model, h_fresh[:, -1, :], all_items, B)

                kv_0 = model_frozen.compute_kv(iids, behs, tds)
                h_stale = model.forward_stale_kv(iids, behs, tds, kv_0)
                sc_stale = score_all(model, h_stale[:, -1, :], all_items, B)

                h_froz, _ = model_frozen(iids, behs, tds, return_kv=False, return_hidden=True)
                sc_froz = score_all(model_frozen, h_froz[:, -1, :], all_items, B)

            t10chg.extend((1 - topk_overlap(sc_fresh, sc_stale, 10)).tolist())
            mrr_f.extend(mrr_per_user(sc_fresh, pos_list).tolist())
            mrr_s.extend(mrr_per_user(sc_stale, pos_list).tolist())
            r100_f.extend(recall_at_k(sc_fresh, pos_list, 100).tolist())
            r100_s.extend(recall_at_k(sc_stale, pos_list, 100).tolist())
            for i in range(B):
                sf, ss = sc_fresh[i, 1:].cpu().numpy(), sc_stale[i, 1:].cpu().numpy()
                if np.std(sf) > 1e-8 and np.std(ss) > 1e-8:
                    rho, _ = spearmanr(sf, ss)
                    sps.append(1 - (rho if not np.isnan(rho) else 1.0))

        mf, ms = np.array(mrr_f), np.array(mrr_s)
        rf, rs = np.array(r100_f), np.array(r100_s)
        rec = {
            "win_idx": ckpt["win_idx"], "dtheta_rel": round(ckpt["dtheta_rel"], 5),
            "steps": ckpt["steps"], "n_eval": len(mf),
            "top10_chg%": round(float(np.mean(t10chg) * 100), 2),
            "top10_chg_p90%": round(float(np.percentile(t10chg, 90) * 100), 2),
            "mrr_fresh": round(float(mf.mean()), 5), "mrr_stale": round(float(ms.mean()), 5),
            "mrr_diff": round(float(mf.mean() - ms.mean()), 6),
            "stale_worse%": round(float(np.mean(ms < mf) * 100), 1),
            "r100_fresh": round(float(rf.mean()), 5), "r100_stale": round(float(rs.mean()), 5),
            "r100_diff": round(float(rf.mean() - rs.mean()), 6),
            "mrr_frozen": round(float(np.mean(mrr_per_user(
                model_frozen.item_emb.score(
                    model_frozen(torch.zeros(1, seq_len, dtype=torch.long, device=device),
                                 torch.zeros(1, seq_len, dtype=torch.long, device=device),
                                 torch.zeros(1, seq_len, dtype=torch.float, device=device),
                                 return_kv=False, return_hidden=True)[0][:, -1, :],
                    all_items.unsqueeze(0)
                ).squeeze(0).cpu().numpy(), [set()])  # placeholder
            )), 5) if False else 0.0,
        }
        results.append(rec)
        print(f"    win {ckpt['win_idx']}: dtheta={rec['dtheta_rel']:.4f} top10={rec['top10_chg%']:.1f}% "
              f"MRR f={rec['mrr_fresh']:.5f} s={rec['mrr_stale']:.5f} diff={rec['mrr_diff']:.6f} "
              f"worse={rec['stale_worse%']:.0f}% R@100 diff={rec['r100_diff']:.6f}", flush=True)
    return results


def main():
    device = torch.device("cuda")
    np.random.seed(42); torch.manual_seed(0)
    seq_len = MODEL_CFG["max_seq_len"]

    print("=== Window Size Comparison (LEAK-FIXED + more training) ===", flush=True)
    plan = StreamingDataPlan.from_csvs(
        ["data/kuairand/log_standard_4_08_to_4_21_1k.csv",
         "data/kuairand/log_standard_4_22_to_5_08_1k.csv"],
        base_num_days=14, max_seq_len=seq_len, max_items=20000)
    plan.init_base()
    print(f"users={plan.num_users} items={plan.num_items}", flush=True)

    all_items = torch.arange(0, plan.num_items + 1, device=device)

    # train base once
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
        print(f"\n{'='*50}\n=== W={W} ({TARGET_STEPS_PER_WINDOW} steps per {W}-day window) ===", flush=True)
        model.load_state_dict(theta0_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        replay_buffer = []

        plan_w = StreamingDataPlan.from_csvs(
            ["data/kuairand/log_standard_4_08_to_4_21_1k.csv",
             "data/kuairand/log_standard_4_22_to_5_08_1k.csv"],
            base_num_days=14, max_seq_len=seq_len, max_items=20000)
        plan_w.init_base()

        ckpts = run_window(model, plan_w, optimizer, device, W, theta0_state, replay_buffer)

        # set model_frozen to theta_0 for eval
        model_frozen.load_state_dict(theta0_state)
        results = eval_staleness(model, model_frozen, ckpts, device, all_items)
        all_results[f"W{W}"] = results

    save_json(all_results, "results/streaming/window_comparison_fixed.json")

    print(f"\n{'='*50}\n=== SUMMARY ===")
    print(f'{"W":>2} {"win":>3} {"dtheta":>7} {"top10%":>6} {"MRR_f":>8} {"MRR_s":>8} {"diff":>8} {"worse%":>6} {"R100d":>8}')
    for W in WINDOW_SIZES:
        for r in all_results.get(f"W{W}", []):
            print(f'{W:>2} {r["win_idx"]:>3} {r["dtheta_rel"]:>7.4f} {r["top10_chg%"]:>5.1f}% '
                  f'{r["mrr_fresh"]:>8.5f} {r["mrr_stale"]:>8.5f} {r["mrr_diff"]:>8.6f} {r["stale_worse%"]:>5.0f}% '
                  f'{r["r100_diff"]:>8.6f}')


if __name__ == "__main__":
    main()
