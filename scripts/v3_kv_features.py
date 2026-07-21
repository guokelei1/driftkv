"""V3 enhanced (vectorized): predict drift norm from cached KV statistics.

Fully batched: all KV computations, feature extraction, and drift norms are
tensor operations. No per-user Python loops, no .item() in the hot path.

Tests whether the ALREADY-CACHED old KV (free in the real system) can predict
how much it will drift under a parameter update. This is the simplified form
of path 2 (Fisher spectrum): cached-representation sensitivity prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hstu_kvcache.data import StreamingDataPlan
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.utils import save_json

MODEL_CFG = dict(hidden_size=256, num_layers=6, num_heads=8, head_dim=64, max_seq_len=512, activation="relu")
MAX_USERS = 300
BS = 64


def compute_seq_features_batch(seqs, item_pop_dict):
    """Compute sequence features for a batch of sequences (numpy, CPU)."""
    out = []
    for seq in seqs:
        items = seq["item_ids"]
        behs = seq["behaviors"]
        tds = seq["time_deltas"]
        n = len(items)
        if n == 0:
            out.append([0.0] * 10)
            continue
        unique_items = len(set(items.tolist()))
        pops = np.array([item_pop_dict.get(int(i), 1) for i in items], dtype=np.float64)
        beh_counts = np.bincount(behs.astype(int), minlength=10)
        beh_probs = beh_counts[beh_counts > 0] / n
        beh_entropy = -np.sum(beh_probs * np.log(beh_probs + 1e-12))
        tds_nz = tds[tds > 0]
        td_mean = float(np.mean(tds_nz)) if len(tds_nz) > 0 else 0.0
        td_std = float(np.std(tds_nz)) if len(tds_nz) > 0 else 0.0
        out.append([
            float(n), float(unique_items), float(unique_items) / n,
            float(beh_entropy), td_mean, td_std,
            float(np.mean(pops)), float(np.std(pops)),
            float(np.max(pops)), float(np.sum(np.log1p(pops))),
        ])
    return np.array(out)


def compute_kv_features_batch(kv, num_layers):
    """Compute KV features as batch tensor ops. kv.k: [L_layers, B, L, inner] -> [B, F]."""
    k = kv.k.float()  # [L, B, L_seq, inner]
    v = kv.v.float()
    feats = []
    for layer in range(num_layers):
        kl = k[layer]  # [B, L, inner]
        vl = v[layer]
        k_norms = kl.norm(dim=-1)  # [B, L]
        v_norms = vl.norm(dim=-1)
        feats.append(k_norms.mean(dim=-1))  # [B]
        feats.append(k_norms.std(dim=-1))
        feats.append(v_norms.mean(dim=-1))
        feats.append(v_norms.std(dim=-1))
        feats.append(kl.reshape(kl.shape[0], -1).norm(dim=-1))  # fro [B]
        feats.append(vl.reshape(vl.shape[0], -1).norm(dim=-1))
        feats.append(kl.abs().amax(dim=(1, 2)))  # max [B]
        feats.append(vl.abs().amax(dim=(1, 2)))
    # cross-layer
    k_layer_norms = k.reshape(num_layers, k.shape[1], -1).norm(dim=-1)  # [L, B]
    v_layer_norms = v.reshape(num_layers, v.shape[1], -1).norm(dim=-1)
    feats.append(k_layer_norms.max(dim=0)[0] / (k_layer_norms.min(dim=0)[0] + 1e-12))  # [B]
    feats.append(v_layer_norms.max(dim=0)[0] / (v_layer_norms.min(dim=0)[0] + 1e-12))
    feats.append(k_layer_norms.std(dim=0))
    feats.append(v_layer_norms.std(dim=0))
    return torch.stack(feats, dim=-1)  # [B, F]


def compute_drift_batch(kv_curr, kv_prev):
    """Per-user relative drift norm as batch op. Returns [B]."""
    dk = (kv_curr.k - kv_prev.k).float()  # [L, B, L_seq, inner]
    dv = (kv_curr.v - kv_prev.v).float()
    B = dk.shape[1]
    drift_k = dk.reshape(-1, B).norm(dim=0)  # [B]
    drift_v = dv.reshape(-1, B).norm(dim=0)
    base_k = kv_curr.k.float().reshape(-1, B).norm(dim=0)
    base_v = kv_curr.v.float().reshape(-1, B).norm(dim=0)
    return (drift_k + drift_v) / (base_k + base_v + 1e-12)


def main():
    device = torch.device("cuda")
    np.random.seed(42)
    torch.manual_seed(0)
    seq_len = MODEL_CFG["max_seq_len"]

    print("=== V3 Enhanced (vectorized): KV-cache features for drift prediction ===")
    plan = StreamingDataPlan.from_csvs(
        ["data/kuairand/log_standard_4_08_to_4_21_1k.csv",
         "data/kuairand/log_standard_4_22_to_5_08_1k.csv"],
        base_num_days=14, max_seq_len=seq_len, max_items=20000)
    plan.init_base()
    item_pop = plan.trace.interactions["item_idx"].value_counts().to_dict()

    num_layers = MODEL_CFG["num_layers"]
    model = HSTU(HSTUConfig(num_items=plan.num_items, num_behaviors=plan.num_behaviors, **MODEL_CFG)).to(device)

    all_seq_feats = []
    all_kv_feats = []
    all_targets_step = []
    all_day_idx = []

    for di, date in enumerate(plan.stream_dates):
        day_idx = di + 1
        if day_idx > 17 or day_idx < 2:
            if day_idx <= 17:
                plan.ingest_day(date)
            continue

        sd_prev = torch.load(f"checkpoints/streaming_relu/theta_{day_idx - 2}.pt", map_location=device)
        sd_curr = torch.load(f"checkpoints/streaming_relu/theta_{day_idx - 1}.pt", map_location=device)

        day_df = plan.daily_segments.get(date)
        if day_df is None or len(day_df) == 0:
            plan.ingest_day(date)
            continue

        samples = []
        for u in day_df["user_idx"].unique()[:MAX_USERS]:
            u = int(u)
            if u not in plan.user_histories or len(plan.user_histories[u]["item_ids"]) < 2:
                continue
            hist = plan.user_histories[u]
            samples.append({
                "item_ids": hist["item_ids"][-seq_len:],
                "behaviors": hist["behaviors"][-seq_len:],
                "time_deltas": hist["time_deltas"][-seq_len:],
            })
        if not samples:
            plan.ingest_day(date)
            continue

        # process in large batches
        for bi in range(0, len(samples), BS):
            batch = samples[bi:bi + BS]
            B = len(batch)
            max_l = min(max(len(s["item_ids"]) for s in batch), seq_len)
            item_ids = torch.zeros(B, max_l, dtype=torch.long, device=device)
            behs = torch.zeros(B, max_l, dtype=torch.long, device=device)
            tds = torch.zeros(B, max_l, dtype=torch.float, device=device)
            for i, s in enumerate(batch):
                arr = s["item_ids"][-max_l:]
                n = len(arr)
                item_ids[i, :n] = torch.tensor(arr, device=device)
                behs[i, :n] = torch.tensor(s["behaviors"][-max_l:], device=device)
                tds[i, :n] = torch.tensor(s["time_deltas"][-max_l:], device=device)

            with torch.no_grad():
                model.load_state_dict(sd_prev)
                kv_prev = model.compute_kv(item_ids, behs, tds)
                model.load_state_dict(sd_curr)
                kv_curr = model.compute_kv(item_ids, behs, tds)

                # batch drift norms
                drift_step = compute_drift_batch(kv_curr, kv_prev)  # [B]
                # batch KV features from kv_prev (the cached KV)
                kv_feats = compute_kv_features_batch(kv_prev, num_layers)  # [B, F]

            # sequence features (CPU)
            seq_feats = compute_seq_features_batch(batch, item_pop)  # [B, 10]

            # move to CPU once
            all_seq_feats.append(seq_feats)
            all_kv_feats.append(kv_feats.cpu().numpy())
            all_targets_step.append(drift_step.cpu().numpy())
            all_day_idx.extend([day_idx] * B)

        print(f"Day {day_idx:2d}: {len(samples)} users", flush=True)
        plan.ingest_day(date)

    X_seq = np.vstack(all_seq_feats)
    X_kv = np.vstack(all_kv_feats)
    X_all = np.hstack([X_seq, X_kv])
    y_step = np.concatenate(all_targets_step)
    days = np.array(all_day_idx)

    print(f"\nCollected {len(X_all)} samples, {X_all.shape[1]} features")

    def eval_predictor(X, y, days, model_fn):
        unique_days = sorted(set(days.tolist()))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        preds = np.zeros(len(y))
        importances = np.zeros(X.shape[1])
        for train_idx, test_idx in kf.split(unique_days):
            tr_d = [unique_days[i] for i in train_idx]
            te_d = [unique_days[i] for i in test_idx]
            tr_m = np.isin(days, tr_d)
            te_m = np.isin(days, te_d)
            sc = StandardScaler()
            X_tr = sc.fit_transform(X[tr_m])
            X_te = sc.transform(X[te_m])
            m = model_fn()
            m.fit(X_tr, y[tr_m])
            preds[te_m] = m.predict(X_te)
            if hasattr(m, "feature_importances_"):
                importances += m.feature_importances_ / 5
        mae = np.mean(np.abs(preds - y))
        rel_mae = mae / (np.mean(np.abs(y)) + 1e-12)
        rho, pval = spearmanr(preds, y)
        q75 = np.percentile(y, 75)
        true_hi = y >= q75
        pred_hi = preds >= np.percentile(preds, 75)
        tp = np.sum(true_hi & pred_hi)
        return {
            "rel_mae": float(rel_mae), "spearman": float(rho), "pval": float(pval),
            "triage_prec": float(tp / (pred_hi.sum() + 1e-12)),
            "triage_recall": float(tp / (true_hi.sum() + 1e-12)),
            "importances": importances.tolist(),
        }

    models = [
        ("ridge", lambda: Ridge(alpha=1.0)),
        ("rf", lambda: RandomForestRegressor(n_estimators=80, max_depth=6, random_state=42, n_jobs=-1)),
        ("gbm", lambda: GradientBoostingRegressor(n_estimators=80, max_depth=4, random_state=42)),
    ]

    results = {"n_samples": len(X_all)}
    print("\n=== Per-step drift prediction (target: ||F(theta_t) - F(theta_{t-1})||) ===", flush=True)
    for label, X in [("seq_only", X_seq), ("kv_only", X_kv), ("seq+kv", X_all)]:
        print(f"\n  {label} ({X.shape[1]} feats):", flush=True)
        for mname, fn in models:
            r = eval_predictor(X, y_step, days, fn)
            print(f"    {mname:6s}: rel_mae={r['rel_mae']:.4f} spearman={r['spearman']:.3f} "
                  f"triage_prec={r['triage_prec']:.3f}", flush=True)
            results[f"step_{label}_{mname}"] = r

    best = max([(k, v["spearman"]) for k, v in results.items() if k.startswith("step_")], key=lambda x: x[1])
    verdict = "PASS" if best[1] >= 0.5 else ("MARGINAL" if best[1] >= 0.3 else "FAIL")
    print(f"\nBest: {best[0]} spearman={best[1]:.3f} -> {verdict}")

    # feature importance
    bk = best[0]
    imps = results[bk]["importances"]
    kv_names = [f"kv_{i}" for i in range(X_kv.shape[1])]
    seq_names = [f"seq_{i}" for i in range(X_seq.shape[1])]
    all_names = seq_names + kv_names
    order = np.argsort(-np.array(imps))[:12]
    print(f"\nTop-12 features ({bk}):")
    for i in order:
        print(f"  {all_names[i]:10s}: {imps[i]:.4f}")

    results["verdict"] = verdict
    save_json(results, "results/phase0/V3_kv_features.json")
    print("\nSaved results/phase0/V3_kv_features.json")


if __name__ == "__main__":
    main()
