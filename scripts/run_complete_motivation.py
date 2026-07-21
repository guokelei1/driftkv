"""Complete motivation experiment: 3 findings in one run.

Finding 1 - Streaming necessity: frozen theta_0 decays vs daily-updated theta_t.
Finding 2 - KV reuse loss: stale KV (theta_0's K,V + theta_t's Q) causes ranking drift.
Finding 3 - Signal chain: dtheta -> KV drift -> ranking change are correlated,
           meaning dtheta (trivial to get) can predict KV staleness impact.

Uses relu 6L/256D/512S model checkpoints. Three conditions per day t:
  FROZEN:       forward(history, theta_0)              -- no model update at all
  FULL RECOMPUTE: forward(history, theta_{t-1})        -- gold standard
  STALE KV:     forward_stale_kv(history, theta_{t-1}, KV(theta_0))  -- reuse old KV

Metrics: hit@10, MRR (coarse) + Spearman, hidden cosine (sensitive).
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
from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.utils import save_json

MODEL_CFG = dict(hidden_size=256, num_layers=6, num_heads=8, head_dim=64, max_seq_len=512, activation="relu")
MAX_USERS = 300
NUM_NEG = 1000
K = 10


def compute_hit_mrr(scores, cands, pos_set, k=K):
    order = np.argsort(-scores)
    pos_mask = np.array([c in pos_set for c in cands])
    pp = np.where(pos_mask[order])[0]
    if len(pp) == 0:
        return 0.0, 0.0
    mrr = 1.0 / (pp[0] + 1)
    hit = 1.0 if pp[0] < k else 0.0
    return hit, mrr


def main():
    device = torch.device("cuda")
    np.random.seed(42); torch.manual_seed(0)
    seq_len = MODEL_CFG["max_seq_len"]

    plan = StreamingDataPlan.from_csvs(
        ["data/kuairand/log_standard_4_08_to_4_21_1k.csv",
         "data/kuairand/log_standard_4_22_to_5_08_1k.csv"],
        base_num_days=14, max_seq_len=seq_len, max_items=20000)
    plan.init_base()

    cfg = HSTUConfig(num_items=plan.num_items, num_behaviors=plan.num_behaviors, **MODEL_CFG)
    model = HSTU(cfg).to(device)
    ckpt_dir = Path("checkpoints/streaming_relu")
    theta0_sd = torch.load(ckpt_dir / "theta_0.pt", map_location=device)
    theta0_vec = None

    all_items = torch.arange(1, plan.num_items + 1, device=device)  # for Spearman

    results = {"days": []}

    for di, date in enumerate(plan.stream_dates):
        day_idx = di + 1
        if day_idx > 17:
            break

        theta_new_sd = torch.load(ckpt_dir / f"theta_{day_idx - 1}.pt", map_location=device)
        if theta0_vec is None:
            model.load_state_dict(theta0_sd)
            theta0_vec = model_params_vec(model).detach().clone()
        model.load_state_dict(theta_new_sd)
        theta_new_vec = model_params_vec(model).detach().clone()
        dtheta_rel = (theta_new_vec - theta0_vec).norm().item() / (theta0_vec.norm().item() + 1e-12)

        day_df = plan.daily_segments.get(date)
        if day_df is None:
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

        # per-user metrics for 3 conditions
        m = {c: {"hit": [], "mrr": [], "spearman": [], "h_cos": []}
             for c in ["frozen", "full", "stale_kv"]}
        kv_drifts = []

        for s in samples:
            item_ids = torch.tensor(s["item_ids"], dtype=torch.long, device=device).unsqueeze(0)
            behs = torch.tensor(s["behaviors"], dtype=torch.long, device=device).unsqueeze(0)
            tds = torch.tensor(s["time_deltas"], dtype=torch.float, device=device).unsqueeze(0)
            pos_set = set(s["pos_items"])
            negs = np.random.randint(1, plan.num_items + 1, size=NUM_NEG)
            cands = np.unique(np.concatenate([list(pos_set), negs]))
            cand_t = torch.tensor(cands, device=device).unsqueeze(0)

            with torch.no_grad():
                # FROZEN: theta_0 for everything
                model.load_state_dict(theta0_sd); model.eval()
                h_frozen, _ = model(item_ids, behs, tds, return_kv=False)
                sc_frozen = model.item_emb.score(h_frozen[:, -1, :], cand_t).squeeze(0).cpu().numpy()
                sc_frozen_all = model.item_emb.score(h_frozen[:, -1, :], all_items.unsqueeze(0)).squeeze(0).cpu().numpy()

                # FULL RECOMPUTE: theta_{t-1} for everything
                model.load_state_dict(theta_new_sd); model.eval()
                h_full, _ = model(item_ids, behs, tds, return_kv=False)
                sc_full = model.item_emb.score(h_full[:, -1, :], cand_t).squeeze(0).cpu().numpy()
                sc_full_all = model.item_emb.score(h_full[:, -1, :], all_items.unsqueeze(0)).squeeze(0).cpu().numpy()

                # STALE KV: theta_0's KV + theta_{t-1}'s Q
                model.load_state_dict(theta0_sd)
                kv0 = model.compute_kv(item_ids, behs, tds)
                kv_new = model.compute_kv(item_ids, behs, tds)  # actually need theta_new for this
                model.load_state_dict(theta_new_sd)
                kv_new = model.compute_kv(item_ids, behs, tds)
                kv_drifts.append(kv0.drift_norm(kv_new))
                h_stale = model.forward_stale_kv(item_ids, behs, tds, kv0)
                sc_stale = model.item_emb.score(h_stale[:, -1, :], cand_t).squeeze(0).cpu().numpy()
                sc_stale_all = model.item_emb.score(h_stale[:, -1, :], all_items.unsqueeze(0)).squeeze(0).cpu().numpy()

                # hit@10, MRR (coarse metrics)
                for cond, sc in [("frozen", sc_frozen), ("full", sc_full), ("stale_kv", sc_stale)]:
                    h, r = compute_hit_mrr(sc, cands, pos_set)
                    m[cond]["hit"].append(h); m[cond]["mrr"].append(r)

                # Spearman (sensitive metric) - on full item catalog
                rho_full_stale, _ = spearmanr(sc_full_all, sc_stale_all)
                rho_full_frozen, _ = spearmanr(sc_full_all, sc_frozen_all)
                m["stale_kv"]["spearman"].append(rho_full_stale)
                m["frozen"]["spearman"].append(rho_full_frozen)
                m["full"]["spearman"].append(1.0)

                # hidden cosine
                m["stale_kv"]["h_cos"].append(torch.nn.functional.cosine_similarity(
                    h_full[:, -1, :], h_stale[:, -1, :]).item())
                m["frozen"]["h_cos"].append(torch.nn.functional.cosine_similarity(
                    h_full[:, -1, :], h_frozen[:, -1, :]).item())
                m["full"]["h_cos"].append(1.0)

        avg_drift = float(np.mean([d["k_fro_rel"] + d["v_fro_rel"] for d in kv_drifts]))
        rec = {
            "date": date, "day_idx": day_idx, "n_eval": len(samples),
            "dtheta_rel": round(dtheta_rel, 5),
            "kv_drift_rel": round(avg_drift, 4),
            "frozen_hit@10": round(float(np.mean(m["frozen"]["hit"])), 4),
            "full_hit@10": round(float(np.mean(m["full"]["hit"])), 4),
            "stale_kv_hit@10": round(float(np.mean(m["stale_kv"]["hit"])), 4),
            "frozen_mrr": round(float(np.mean(m["frozen"]["mrr"])), 4),
            "full_mrr": round(float(np.mean(m["full"]["mrr"])), 4),
            "stale_kv_mrr": round(float(np.mean(m["stale_kv"]["mrr"])), 4),
            "frozen_spearman": round(float(np.mean(m["frozen"]["spearman"])), 4),
            "stale_kv_spearman": round(float(np.mean(m["stale_kv"]["spearman"])), 4),
            "frozen_h_cos": round(float(np.mean(m["frozen"]["h_cos"])), 4),
            "stale_kv_h_cos": round(float(np.mean(m["stale_kv"]["h_cos"])), 4),
        }
        results["days"].append(rec)
        print(f"Day {day_idx:2d}: dtheta={dtheta_rel:.4f} kv_drift={avg_drift:.4f} | "
              f"hit@10: frozen={rec['frozen_hit@10']:.3f} full={rec['full_hit@10']:.3f} stale={rec['stale_kv_hit@10']:.3f} | "
              f"spearman: frozen={rec['frozen_spearman']:.3f} stale={rec['stale_kv_spearman']:.3f}")

        plan.ingest_day(date)

    save_json(results, "results/streaming/complete_motivation.json")
    print("\nSaved results/streaming/complete_motivation.json")

    # === PLOTS ===
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = [d["day_idx"] for d in results["days"]]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: hit@10 over days (necessity + KV loss, coarse)
    ax = axes[0, 0]
    ax.plot(days, [d["frozen_hit@10"]*100 for d in results["days"]], "o--", color="#d62728", label="Frozen theta_0 (no update)", linewidth=2)
    ax.plot(days, [d["full_hit@10"]*100 for d in results["days"]], "s-", color="#2ca02c", label="Full recompute (daily update)", linewidth=2)
    ax.plot(days, [d["stale_kv_hit@10"]*100 for d in results["days"]], "^:", color="#ff7f0e", label="Stale KV (reuse theta_0 KV)", linewidth=2)
    ax.set_xlabel("Streaming day"); ax.set_ylabel("Hit@10 (%)")
    ax.set_title("Finding 1+2: Hit@10 (coarse - top-10 dominated by popularity)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xticks(days[::2])

    # Plot 2: Spearman over days (KV loss, sensitive)
    ax = axes[0, 1]
    ax.plot(days, [d["frozen_spearman"] for d in results["days"]], "o--", color="#d62728", label="Frozen vs Full", linewidth=2)
    ax.plot(days, [d["stale_kv_spearman"] for d in results["days"]], "^-", color="#ff7f0e", label="Stale KV vs Full", linewidth=2)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Streaming day"); ax.set_ylabel("Spearman rank correlation")
    ax.set_title("Finding 2: KV reuse causes ranking drift (sensitive metric)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xticks(days[::2])
    ax.set_ylim(0.7, 1.02)

    # Plot 3: Signal chain - dtheta & kv_drift & (1-spearman) over days
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.plot(days, [d["dtheta_rel"] for d in results["days"]], "o-", color="#1f77b4", label="||dtheta||/||theta||", linewidth=2)
    ax.plot(days, [d["kv_drift_rel"] for d in results["days"]], "s-", color="#ff7f0e", label="KV drift (Frobenius rel)", linewidth=2)
    ax2.plot(days, [1-d["stale_kv_spearman"] for d in results["days"]], "^--", color="#d62728", label="1-Spearman (ranking loss)", linewidth=2)
    ax.set_xlabel("Streaming day"); ax.set_ylabel("Drift magnitude")
    ax2.set_ylabel("1-Spearman (ranking loss)")
    ax.set_title("Finding 3: dtheta -> KV drift -> ranking loss (signal chain)", fontsize=11)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3); ax.set_xticks(days[::2])

    # Plot 4: Scatter - dtheta vs (1-spearman) and kv_drift vs (1-spearman)
    ax = axes[1, 1]
    dthetas = [d["dtheta_rel"] for d in results["days"]]
    kv_drifts_s = [d["kv_drift_rel"] for d in results["days"]]
    losses = [1-d["stale_kv_spearman"] for d in results["days"]]
    ax.scatter(dthetas, losses, color="#1f77b4", s=80, label="dtheta vs ranking loss", zorder=3)
    ax.scatter(kv_drifts_s, losses, color="#ff7f0e", s=80, marker="s", label="KV drift vs ranking loss", zorder=3)
    # fit lines
    for xs, color, label in [(dthetas, "#1f77b4", "dtheta"), (kv_drifts_s, "#ff7f0e", "kv_drift")]:
        xs_arr = np.array(xs); ys_arr = np.array(losses)
        if len(xs_arr) > 2:
            coef = np.polyfit(xs_arr, ys_arr, 1)
            x_fit = np.linspace(xs_arr.min(), xs_arr.max(), 50)
            ax.plot(x_fit, np.polyval(coef, x_fit), "--", color=color, alpha=0.5)
            rho, _ = spearmanr(xs_arr, ys_arr)
            ax.annotate(f"{label} Spearman={rho:.3f}", xy=(0.05, 0.95-0.05*[dthetas, kv_drifts_s].index(xs)),
                        xycoords="axes fraction", fontsize=9, color=color)
    ax.set_xlabel("Drift magnitude"); ax.set_ylabel("1-Spearman (ranking loss)")
    ax.set_title("Finding 3: Drift metrics predict ranking loss", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle("Complete Motivation: Streaming Necessity + KV Reuse Loss + Signal Chain", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/streaming/complete_motivation_plot.png", dpi=150, bbox_inches="tight")
    print("Saved results/streaming/complete_motivation_plot.png")

    # Summary
    print("\n=== Complete Motivation Summary ===")
    print("Finding 1 (streaming necessity):")
    print(f"  Frozen hit@10: {results['days'][0]['frozen_hit@10']*100:.1f}% -> {results['days'][-1]['frozen_hit@10']*100:.1f}%")
    print(f"  Full hit@10:   {results['days'][0]['full_hit@10']*100:.1f}% -> {results['days'][-1]['full_hit@10']*100:.1f}%")
    print("Finding 2 (KV reuse loss):")
    print(f"  Spearman (stale KV): {results['days'][0]['stale_kv_spearman']:.4f} -> {results['days'][-1]['stale_kv_spearman']:.4f}")
    print(f"  1-Spearman (ranking loss): {(1-results['days'][0]['stale_kv_spearman'])*100:.2f}% -> {(1-results['days'][-1]['stale_kv_spearman'])*100:.2f}%")
    print("Finding 3 (signal chain):")
    dthetas = [d["dtheta_rel"] for d in results["days"]]
    kv_d = [d["kv_drift_rel"] for d in results["days"]]
    losses = [1-d["stale_kv_spearman"] for d in results["days"]]
    rho1, _ = spearmanr(dthetas, losses)
    rho2, _ = spearmanr(kv_d, losses)
    rho3, _ = spearmanr(dthetas, kv_d)
    print(f"  dtheta vs ranking loss: Spearman={rho1:.3f}")
    print(f"  KV drift vs ranking loss: Spearman={rho2:.3f}")
    print(f"  dtheta vs KV drift: Spearman={rho3:.3f}")


if __name__ == "__main__":
    main()
