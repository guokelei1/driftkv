"""Triangular evaluation matrix: models x eval windows.

Protocol (leak-free):
  W0 = base period (14 days, long), train theta_0 (50 epochs)
  W1..W6 = streaming windows (3 days each), train theta_1..5 (500 steps each)
    theta_t = theta_{t-1} + 500 steps on Wt data (continual training)

  Evaluation matrix (Model t, Eval Ws) where s > t:
    - Model t was trained on W0..Wt, has NOT seen Ws data
    - Fresh: theta_t model + F(theta_t, x_u) KV
    - Stale: theta_t model + F(theta_0, x_u) KV  (base model's KV)
    - dtheta = ||theta_t - theta_0|| / ||theta_0||  (grows with t)

  As t increases, the model has changed more from theta_0, so stale KV
  (from theta_0) should hurt more. This is the key motivation signal.
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
STREAM_EPOCHS = 3
WINDOW_SIZE = 3
MAX_USERS = 300
BS = 32


def make_model(device, ni, nb):
    return HSTU(HSTUConfig(num_items=ni, num_behaviors=nb, **MODEL_CFG)).to(device)


def score_all(model, h, all_items, B):
    return model.item_emb.score(h, all_items.unsqueeze(0).expand(B, -1))


def topk_overlap(sa, sb, k=10):
    ta = torch.topk(sa, k, -1).indices.cpu().numpy()
    tb = torch.topk(sb, k, -1).indices.cpu().numpy()
    return np.array([len(set(ta[i]) & set(tb[i])) / k for i in range(sa.shape[0])])


def recall_at_k(sc, pos, k=100):
    tk = torch.topk(sc, k, -1).indices.cpu().numpy()
    return np.array([len(set(pos[i]) & set(tk[i])) / max(len(pos[i]), 1) for i in range(sc.shape[0])])


def mrr_per(sc, pos):
    s = sc.cpu().numpy()
    return np.array([
        1.0 / (min(int(np.where(np.argsort(-s[i]) == p)[0][0]) for p in pos[i]) + 1) if pos[i] else 0.0
        for i in range(s.shape[0])
    ])


def build_batches(samples, seq_len, device, bs=BS):
    """Yield (item_ids, behs, tds, pos_list) batches from eval samples."""
    for bi in range(0, len(samples), bs):
        batch = samples[bi:bi + bs]
        B = len(batch)
        max_l = min(max(len(s["history"]["item_ids"]) for s in batch), seq_len)
        iids = torch.zeros(B, max_l, dtype=torch.long, device=device)
        behs = torch.zeros(B, max_l, dtype=torch.long, device=device)
        tds = torch.zeros(B, max_l, dtype=torch.float, device=device)
        pos = []
        for i, s in enumerate(batch):
            arr = s["history"]["item_ids"][-max_l:]; n = len(arr)
            iids[i, :n] = torch.tensor(arr, device=device)
            behs[i, :n] = torch.tensor(s["history"]["behaviors"][-max_l:], device=device)
            tds[i, :n] = torch.tensor(s["history"]["time_deltas"][-max_l:], device=device)
            pos.append(set(s["pos_items"]))
        yield iids, behs, tds, pos, B


def compute_all_metrics(sc_fresh, sc_stale, h_fresh, h_stale, pos_list):
    """Comprehensive per-user metrics from fresh vs stale scores."""
    B = sc_fresh.shape[0]
    sf = sc_fresh.cpu().numpy()
    ss = sc_stale.cpu().numpy()
    cos = torch.nn.functional.cosine_similarity(h_fresh[:, -1, :], h_stale[:, -1, :], dim=-1).cpu().numpy()
    out = {k: [] for k in [
        "mrr", "ndcg10", "hit1", "hit10", "best_rank", "mean_rank",
        "top10_pos", "rank_disp_best", "rank_disp_mean", "t10_ov", "sp", "h_cos", "r10"]}
    for i in range(B):
        pos = pos_list[i]
        if not pos:
            for k in out:
                out[k].append((0, 0) if k != "h_cos" else cos[i])
            continue
        of = np.argsort(-sf[i])
        os_ = np.argsort(-ss[i])
        rf = {p: int(np.where(of == p)[0][0]) for p in pos}
        rs = {p: int(np.where(os_ == p)[0][0]) for p in pos}
        bf, bs = min(rf.values()), min(rs.values())

        def ndcg(order, k=10):
            dcg = sum(1.0 / np.log2(j + 2) for j in range(min(k, len(order))) if order[j] in pos)
            idcg = sum(1.0 / np.log2(j + 2) for j in range(min(len(pos), k)))
            return dcg / idcg if idcg > 0 else 0

        out["mrr"].append((1.0 / (bf + 1), 1.0 / (bs + 1)))
        out["ndcg10"].append((ndcg(of), ndcg(os_)))
        out["hit1"].append((1 if of[0] in pos else 0, 1 if os_[0] in pos else 0))
        out["hit10"].append((1 if any(p in of[:10] for p in pos) else 0,
                              1 if any(p in os_[:10] for p in pos) else 0))
        out["best_rank"].append((bf, bs))
        out["mean_rank"].append((np.mean(list(rf.values())), np.mean(list(rs.values()))))
        out["top10_pos"].append((sum(1 for x in of[:10] if x in pos), sum(1 for x in os_[:10] if x in pos)))
        bp = min(pos, key=lambda p: rf[p])
        out["rank_disp_best"].append((0, rs[bp] - rf[bp]))
        out["rank_disp_mean"].append((0, np.mean([rs[p] - rf[p] for p in pos])))
        out["t10_ov"].append((0, len(set(of[:10]) & set(os_[:10])) / 10))
        if np.std(sf[i, 1:]) > 1e-8 and np.std(ss[i, 1:]) > 1e-8:
            rho, _ = spearmanr(sf[i, 1:], ss[i, 1:])
            out["sp"].append((0, 1 - (rho if not np.isnan(rho) else 1.0)))
        else:
            out["sp"].append((0, 0))
        out["h_cos"].append(cos[i])
        ps = set(pos)
        out["r10"].append((len(ps & set(of[:10])) / max(len(pos), 1),
                           len(ps & set(os_[:10])) / max(len(pos), 1)))
    return out


def main():
    device = torch.device("cuda")
    np.random.seed(42); torch.manual_seed(0)
    seq_len = MODEL_CFG["max_seq_len"]

    print("=== Triangular Evaluation Matrix (Full Metrics) ===", flush=True)
    print(f"Window size: {WINDOW_SIZE} days, Base epochs: {BASE_EPOCHS}, Stream epochs: {STREAM_EPOCHS}", flush=True)

    plan = StreamingDataPlan.from_csvs(
        ["data/kuairand/log_standard_4_08_to_4_21_1k.csv",
         "data/kuairand/log_standard_4_22_to_5_08_1k.csv"],
        base_num_days=14, max_seq_len=seq_len, max_items=20000)
    plan.init_base()
    print(f"users={plan.num_users} items={plan.num_items} stream_days={len(plan.stream_dates)}", flush=True)

    model = make_model(device, plan.num_items, plan.num_behaviors)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    print("Training base (50 epochs)...", flush=True)
    for ep in range(BASE_EPOCHS):
        for batch in plan.iter_base_train_batches(batch_size=BS):
            train_step(model, batch, opt, device)
    theta_states = [{k: v.cpu().clone() for k, v in model.state_dict().items()}]
    theta0_vec = model_params_vec(model).detach().cpu().clone()
    print(f"  ||theta_0||={theta0_vec.norm().item():.2f}", flush=True)

    stream_dates = plan.stream_dates
    n_windows = (len(stream_dates) + WINDOW_SIZE - 1) // WINDOW_SIZE
    print(f"\nStreaming windows: {n_windows}", flush=True)
    replay_buffer = []
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for wi in range(n_windows):
        start = wi * WINDOW_SIZE
        end = min(start + WINDOW_SIZE, len(stream_dates))
        window_dates = stream_dates[start:end]
        for d in window_dates:
            plan.ingest_day(d)
        w_batches = []
        for d in window_dates:
            w_batches.extend(list(plan.iter_train_batches(d, batch_size=BS)))
        if not w_batches:
            theta_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
            continue
        n_new = int(len(w_batches) * 0.5)
        steps = 0
        for _ in range(STREAM_EPOCHS):
            mixed = w_batches[:]; np.random.shuffle(mixed); mixed = mixed[:n_new]
            for _ in range(len(w_batches) - n_new):
                if replay_buffer:
                    mixed.append(copy.deepcopy(replay_buffer[np.random.randint(len(replay_buffer))]))
            np.random.shuffle(mixed)
            for batch in mixed:
                train_step(model, batch, opt, device)
                steps += 1
        for b in w_batches[:10]:
            replay_buffer.append({k: v.clone() for k, v in b.items()})
            if len(replay_buffer) > 300:
                replay_buffer.pop(np.random.randint(len(replay_buffer)))
        vec = model_params_vec(model).detach().cpu().clone()
        dtheta = (vec - theta0_vec).norm().item() / (theta0_vec.norm().item() + 1e-12)
        theta_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        print(f"  theta_{wi+1}: dtheta_rel={dtheta:.4f} steps={steps}", flush=True)

    n_models = len(theta_states)
    print(f"\nTotal models: {n_models}", flush=True)

    # precompute eval samples
    plan_eval = StreamingDataPlan.from_csvs(
        ["data/kuairand/log_standard_4_08_to_4_21_1k.csv",
         "data/kuairand/log_standard_4_22_to_5_08_1k.csv"],
        base_num_days=14, max_seq_len=seq_len, max_items=20000)
    plan_eval.init_base()
    eval_samples_by_window = {}
    for s in range(1, n_models + 1):
        start = (s - 1) * WINDOW_SIZE
        if start >= len(stream_dates):
            break
        eval_date = stream_dates[start]
        eval_samples_by_window[s] = plan_eval.get_eval_set(eval_date, max_users=MAX_USERS)
        for w in range(WINDOW_SIZE):
            if start + w < len(stream_dates):
                plan_eval.ingest_day(stream_dates[start + w])
        print(f"  W{s} eval: {len(eval_samples_by_window[s])} samples on {eval_date}", flush=True)
    n_eval_windows = len(eval_samples_by_window)

    # eval matrix
    all_items = torch.arange(0, plan.num_items + 1, device=device)
    model_frozen = make_model(device, plan.num_items, plan.num_behaviors)
    model_frozen.load_state_dict(theta_states[0])
    model_frozen.eval()
    for p in model_frozen.parameters():
        p.requires_grad_(False)

    results = []
    for s in range(1, n_eval_windows + 1):
        samples = eval_samples_by_window[s]
        if not samples:
            continue
        print(f"\n--- Eval W{s} ---", flush=True)
        stale_kvs = []
        for iids, behs, tds, pos, B in build_batches(samples, seq_len, device):
            with torch.no_grad():
                stale_kvs.append(model_frozen.compute_kv(iids, behs, tds))

        for t in range(min(s, n_models)):
            model.load_state_dict(theta_states[t])
            model.eval()
            vec = model_params_vec(model).detach().cpu()
            dtheta = (vec - theta0_vec).norm().item() / (theta0_vec.norm().item() + 1e-12)

            all_pairs = {k: [] for k in [
                "mrr", "ndcg10", "hit1", "hit10", "best_rank", "mean_rank",
                "top10_pos", "rank_disp_best", "rank_disp_mean", "t10_ov", "sp", "r10"]}
            all_hcos = []

            for bi, (iids, behs, tds, pos, B) in enumerate(build_batches(samples, seq_len, device)):
                kv_0 = stale_kvs[bi]
                with torch.no_grad():
                    h_fresh, kv_fresh = model(iids, behs, tds, return_kv=True, return_hidden=True)
                    sc_fresh = score_all(model, h_fresh[:, -1, :], all_items, B)
                    h_stale = model.forward_stale_kv(iids, behs, tds, kv_0)
                    sc_stale = score_all(model, h_stale[:, -1, :], all_items, B)
                m = compute_all_metrics(sc_fresh, sc_stale, h_fresh, h_stale, pos)
                for k in all_pairs:
                    all_pairs[k].extend(m[k])
                all_hcos.extend(m["h_cos"])

            rec = {"model_t": t, "eval_s": s, "dtheta_rel": round(dtheta, 5),
                   "n": len(all_pairs["mrr"]), "h_cos": round(float(np.mean(all_hcos)), 5)}
            for k in ["mrr", "ndcg10", "hit1", "hit10", "best_rank", "mean_rank", "top10_pos", "r10"]:
                arr = np.array([(f, s2) for f, s2 in all_pairs[k]])
                mf, ms = arr[:, 0], arr[:, 1]
                rec[f"{k}_f"] = round(float(mf.mean()), 6)
                rec[f"{k}_s"] = round(float(ms.mean()), 6)
                rec[f"{k}_diff"] = round(float(mf.mean() - ms.mean()), 6)
                rec[f"{k}_worse%"] = round(float(np.mean(ms < mf) * 100), 1)
            rd_best = np.array([x[1] for x in all_pairs["rank_disp_best"]])
            rec["rank_disp_best"] = round(float(np.mean(rd_best)), 2)
            rec["rank_disp_pos%"] = round(float(np.mean(rd_best > 0) * 100), 1)
            rec["t10_chg%"] = round(float(np.mean([1 - x[1] for x in all_pairs["t10_ov"]]) * 100), 2)
            rec["1-sp_mean"] = round(float(np.mean([x[1] for x in all_pairs["sp"]])), 5)
            results.append(rec)
            print(f"  t{t}->W{s}: dt={dtheta:.3f} t10={rec['t10_chg%']:.0f}% "
                  f"MRR d={rec['mrr_diff']:.5f} w={rec['mrr_worse%']:.0f}% "
                  f"NDCG d={rec['ndcg10_diff']:.5f} w={rec['ndcg10_worse%']:.0f}% "
                  f"Hit1 d={rec['hit1_diff']:.4f} w={rec['hit1_worse%']:.0f}% "
                  f"bestR d={rec['best_rank_diff']:.1f} "
                  f"disp={rec['rank_disp_best']:.1f} pos%={rec['rank_disp_pos%']:.0f}% "
                  f"hcos={rec['h_cos']:.4f}", flush=True)

    save_json(results, "results/streaming/triangular_eval_full.json")

    # summary
    print(f"\n{'='*70}")
    print("=== FULL METRIC SUMMARY (dtheta>0 only) ===")
    nz = [r for r in results if r["dtheta_rel"] > 0]
    metrics = ["mrr", "ndcg10", "hit1", "hit10", "best_rank", "mean_rank", "top10_pos", "r10"]
    print(f'{"metric":>12} {"fresh":>9} {"stale":>9} {"diff":>9} {"worse%":>6} {"cells+":>6}')
    for m in metrics:
        fs = [r[f"{m}_f"] for r in nz]; ss_ = [r[f"{m}_s"] for r in nz]
        ds = [r[f"{m}_diff"] for r in nz]; ws = [r[f"{m}_worse%"] for r in nz]
        pos_ct = sum(1 for d in ds if d > 0)
        print(f'{m:>12} {np.mean(fs):>9.5f} {np.mean(ss_):>9.5f} {np.mean(ds):>9.6f} '
              f'{np.mean(ws):>5.0f}% {pos_ct:>2}/{len(nz)}')
    print(f'\n{"rank_disp":>12} {"":>9} {"":>9} {np.mean([r["rank_disp_best"] for r in nz]):>9.2f} '
          f'{"":>6} {np.mean([r["rank_disp_pos%"] for r in nz]):>5.0f}%')
    print(f'{"t10_chg%":>12} {np.mean([r["t10_chg%"] for r in nz]):>9.1f}')
    print(f'{"1-sp_mean":>12} {np.mean([r["1-sp_mean"] for r in nz]):>9.5f}')
    print(f'{"h_cos":>12} {np.mean([r["h_cos"] for r in nz]):>9.5f}')

    print(f"\n=== Spearman(dtheta, metric) excluding dtheta=0 ===")
    dts = [r["dtheta_rel"] for r in nz]
    for m in metrics + ["rank_disp_best", "t10_chg%", "1-sp_mean"]:
        vals = [r[m if m in ("rank_disp_best", "t10_chg%", "1-sp_mean") else f"{m}_diff"] for r in nz]
        rho, p = spearmanr(dts, vals)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  {m:>15}: rho={rho:.3f} p={p:.4f} {sig}")
    print("\nSaved results/streaming/triangular_eval_full.json")


if __name__ == "__main__":
    main()
