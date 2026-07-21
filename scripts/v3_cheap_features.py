"""Fixed V3: predict per-user drift NORM from genuinely cheap sequence features.

Original V3 bug: used J.dtheta itself (the expensive per-user JVP output) as the
"feature" to predict ||J.dtheta|| - circular. This version uses only sequence
statistics (length, diversity, popularity, behavior entropy, time deltas) that
are free to compute (no forward, no JVP).

If drift norm is predictable from cheap features, we can triage users (reuse vs
recompute) without any expensive per-user computation - the core of path 1
(cross-user sharing).

Tests both per-step drift (theta_t vs theta_{t-1}) and cumulative drift
(theta_t vs theta_0). Uses ground-truth ||F(theta_new) - F(theta_old)|| as the
target (computed via two forwards, which we already have from eval_comprehensive).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hstu_kvcache.data import StreamingDataPlan
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.utils import save_json

MODEL_CFG = dict(hidden_size=256, num_layers=6, num_heads=8, head_dim=64, max_seq_len=512, activation="relu")
MAX_USERS = 300


def compute_cheap_features(seq, item_pop_dict):
    """Compute cheap sequence features (no model forward needed).

    Args:
        seq: dict with item_ids, behaviors, time_deltas (numpy arrays).
        item_pop_dict: {item_idx: interaction_count} for popularity features.
    Returns: list of float features.
    """
    items = seq["item_ids"]
    behs = seq["behaviors"]
    tds = seq["time_deltas"]

    n = len(items)
    if n == 0:
        return [0.0] * 12

    unique_items = len(set(items.tolist()))
    pops = np.array([item_pop_dict.get(int(i), 1) for i in items], dtype=np.float64)

    # behavior entropy
    beh_counts = np.bincount(behs.astype(int), minlength=10)
    beh_probs = beh_counts[beh_counts > 0] / n
    beh_entropy = -np.sum(beh_probs * np.log(beh_probs + 1e-12))

    # time delta stats (exclude first event's 0)
    tds_nonzero = tds[tds > 0]
    td_mean = float(np.mean(tds_nonzero)) if len(tds_nonzero) > 0 else 0.0
    td_std = float(np.std(tds_nonzero)) if len(tds_nonzero) > 0 else 0.0

    return [
        float(n),                                          # seq_len
        float(unique_items),                               # num_unique
        float(unique_items) / n,                           # diversity
        float(beh_entropy),                                # behavior_entropy
        td_mean,                                           # mean_time_delta
        td_std,                                            # std_time_delta
        float(np.mean(pops)),                              # item_pop_mean
        float(np.std(pops)),                               # item_pop_std
        float(np.max(pops)),                               # item_pop_max
        float(pops[-1]),                                   # last_item_pop
        float(np.sum(np.log1p(pops))),                     # log_pop_sum (attention capacity proxy)
        float(np.mean(pops > np.median(pops))),            # frac_popular (above median)
    ]


def main():
    device = torch.device("cuda")
    np.random.seed(42)
    torch.manual_seed(0)
    seq_len = MODEL_CFG["max_seq_len"]

    print("=== Fixed V3: cheap-feature drift norm prediction ===")
    plan = StreamingDataPlan.from_csvs(
        ["data/kuairand/log_standard_4_08_to_4_21_1k.csv",
         "data/kuairand/log_standard_4_22_to_5_08_1k.csv"],
        base_num_days=14, max_seq_len=seq_len, max_items=20000)
    plan.init_base()

    item_pop = plan.trace.interactions["item_idx"].value_counts().to_dict()

    # load comprehensive eval per-user data (has kv_drift_step, kv_drift_cum per user per day)
    comp = None
    comp_path = Path("results/streaming/eval_comprehensive.json")
    if comp_path.exists():
        import json
        comp = json.load(open(comp_path))
        print(f"Loaded comprehensive eval: {len(comp.get('per_user_days', []))} days of per-user data")
    else:
        print("WARNING: eval_comprehensive.json not found. Run eval_comprehensive.py first.")
        return

    # collect (features, target_step, target_cum, day_idx) for all users across days
    feature_names = [
        "seq_len", "num_unique", "diversity", "beh_entropy",
        "td_mean", "td_std", "pop_mean", "pop_std", "pop_max",
        "last_pop", "log_pop_sum", "frac_popular",
    ]
    all_features = []
    all_target_step = []
    all_target_cum = []
    all_day_idx = []
    all_ranking_loss_step = []

    for di, pu in enumerate(comp["per_user_days"]):
        day_idx = comp["days"][di]["day_idx"]
        if day_idx < 2:
            continue
        kv_drift_step = np.array(pu["kv_drift_step"])
        kv_drift_cum = np.array(pu["kv_drift_cum"])
        spearman_step = np.array(pu["spearman_step"])
        user_ids = pu["user_ids"]
        seq_lens = pu["seq_lens"]

        for i, uid in enumerate(user_ids):
            uid = int(uid)
            hist = plan.user_histories.get(uid)
            if hist is None or len(hist["item_ids"]) == 0:
                continue
            seq = {
                "item_ids": hist["item_ids"][-seq_len:],
                "behaviors": hist["behaviors"][-seq_len:],
                "time_deltas": hist["time_deltas"][-seq_len:],
            }
            feats = compute_cheap_features(seq, item_pop)
            all_features.append(feats)
            all_target_step.append(float(kv_drift_step[i]))
            all_target_cum.append(float(kv_drift_cum[i]))
            all_day_idx.append(day_idx)
            all_ranking_loss_step.append(float(1 - spearman_step[i]))

    X = np.array(all_features)
    y_step = np.array(all_target_step)
    y_cum = np.array(all_target_cum)
    days = np.array(all_day_idx)
    rank_loss = np.array(all_ranking_loss_step)
    print(f"Collected {len(X)} user-day samples across {len(set(days))} days")
    print(f"Features: {feature_names}")

    # target statistics
    print(f"\nTarget: per-step KV drift norm (rel Frobenius)")
    print(f"  mean={y_step.mean():.4f} std={y_step.std():.4f} "
          f"min={y_step.min():.4f} max={y_step.max():.4f}")
    print(f"Target: cumulative KV drift norm")
    print(f"  mean={y_cum.mean():.4f} std={y_cum.std():.4f} "
          f"min={y_cum.min():.4f} max={y_cum.max():.4f}")

    results = {"feature_names": feature_names, "n_samples": len(X), "n_days": len(set(days))}

    # === Cross-validated prediction of drift NORM ===
    def eval_predictor(name, X, y, days, model_fn):
        """Cross-validate: train on 4/5 days, test on 1/5 days (day-level CV)."""
        unique_days = sorted(set(days.tolist()))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        preds = np.zeros(len(y))
        importances = np.zeros(X.shape[1])
        for train_idx, test_idx in kf.split(unique_days):
            train_days = [unique_days[i] for i in train_idx]
            test_days = [unique_days[i] for i in test_idx]
            tr_mask = np.isin(days, train_days)
            te_mask = np.isin(days, test_days)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_mask])
            X_te = scaler.transform(X[te_mask])
            model = model_fn()
            model.fit(X_tr, y[tr_mask])
            preds[te_mask] = model.predict(X_te)
            if hasattr(model, "feature_importances_"):
                importances += model.feature_importances_ / 5
        mae = np.mean(np.abs(preds - y))
        rel_mae = mae / (np.mean(np.abs(y)) + 1e-12)
        rho, pval = spearmanr(preds, y)
        # quartile-based triage accuracy: can we identify top-25% drift users?
        q75 = np.percentile(y, 75)
        true_high = y >= q75
        pred_high = preds >= np.percentile(preds, 75)
        tp = np.sum(true_high & pred_high)
        precision = tp / (pred_high.sum() + 1e-12)
        recall = tp / (true_high.sum() + 1e-12)
        return {
            "mae": float(mae), "rel_mae": float(rel_mae),
            "spearman": float(rho), "pval": float(pval),
            "triage_precision": float(precision), "triage_recall": float(recall),
            "importances": importances.tolist(),
        }

    # per-step drift norm prediction
    print("\n=== Per-step drift norm prediction (realistic operating point) ===")
    for name, fn in [
        ("ridge", lambda: Ridge(alpha=1.0)),
        ("random_forest", lambda: RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)),
        ("gbm", lambda: GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)),
    ]:
        r = eval_predictor(name, X, y_step, days, fn)
        print(f"  {name:15s}: rel_mae={r['rel_mae']:.4f} spearman={r['spearman']:.3f} "
              f"triage_prec={r['triage_precision']:.3f} triage_recall={r['triage_recall']:.3f}")
        results[f"step_{name}"] = r

    # cumulative drift norm prediction
    print("\n=== Cumulative drift norm prediction (worst case) ===")
    for name, fn in [
        ("ridge", lambda: Ridge(alpha=1.0)),
        ("random_forest", lambda: RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)),
        ("gbm", lambda: GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)),
    ]:
        r = eval_predictor(name, X, y_cum, days, fn)
        print(f"  {name:15s}: rel_mae={r['rel_mae']:.4f} spearman={r['spearman']:.3f} "
              f"triage_prec={r['triage_precision']:.3f} triage_recall={r['triage_recall']:.3f}")
        results[f"cum_{name}"] = r

    # feature importance (from GBM on per-step)
    gbm_imp = results["step_gbm"]["importances"]
    imp_order = np.argsort(-np.array(gbm_imp))
    print("\nFeature importance (GBM, per-step drift):")
    for i in imp_order:
        print(f"  {feature_names[i]:15s}: {gbm_imp[i]:.4f}")

    # === Direct prediction of ranking loss (skip drift norm as intermediate) ===
    print("\n=== Direct prediction of ranking loss from cheap features ===")
    for name, fn in [
        ("ridge", lambda: Ridge(alpha=1.0)),
        ("gbm", lambda: GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)),
    ]:
        r = eval_predictor(name, X, rank_loss, days, fn)
        print(f"  {name:15s}: rel_mae={r['rel_mae']:.4f} spearman={r['spearman']:.3f} "
              f"triage_prec={r['triage_precision']:.3f}")
        results[f"rankloss_{name}"] = r

    # === Gating verdict ===
    best_step = max(["step_ridge", "step_random_forest", "step_gbm"],
                    key=lambda k: results[k]["spearman"])
    best_cum = max(["cum_ridge", "cum_random_forest", "cum_gbm"],
                   key=lambda k: results[k]["spearman"])
    print(f"\nBest per-step predictor: {best_step} (spearman={results[best_step]['spearman']:.3f})")
    print(f"Best cumulative predictor: {best_cum} (spearman={results[best_cum]['spearman']:.3f})")

    step_rho = results[best_step]["spearman"]
    cum_rho = results[best_cum]["spearman"]
    if step_rho >= 0.5:
        verdict = "PASS"
    elif step_rho >= 0.3:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"
    print(f"V3 verdict: {verdict} (per-step drift predictable from cheap features, spearman={step_rho:.3f})")

    results["verdict"] = verdict
    results["best_step_spearman"] = step_rho
    results["best_cum_spearman"] = cum_rho
    save_json(results, "results/phase0/V3_fixed_cheap_features.json")
    print(f"\nSaved results/phase0/V3_fixed_cheap_features.json")


if __name__ == "__main__":
    main()
