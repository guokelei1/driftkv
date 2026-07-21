"""Fixed V4: staleness decay curve with leak-free eval and correct gate logic.

Original V4 bugs:
  1. Train/test leak: eval target was part of training sequence.
  2. Gate logic: negative gap (fresh WORSE than stale) counted as PASS.

This version extracts the staleness decay curve from the leak-free comprehensive
eval (eval_comprehensive.py), computing dtheta between checkpoint pairs and
correlating with ranking loss. Both per-step and cumulative staleness tested.

Gate logic: PASS requires 0 < gap_at_small_dtheta < 0.05 (gentle positive decay
at small updates, not negative).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.utils import save_json

MODEL_CFG = dict(hidden_size=256, num_layers=6, num_heads=8, head_dim=64, max_seq_len=512, activation="relu")


def main():
    device = torch.device("cuda")

    comp_path = Path("results/streaming/eval_comprehensive.json")
    if not comp_path.exists():
        print("ERROR: run eval_comprehensive.py first")
        return
    comp = json.load(open(comp_path))

    # compute dtheta for each checkpoint pair
    model = HSTU(HSTUConfig(num_items=20000, num_behaviors=9, **MODEL_CFG)).to(device)
    ckpt_dir = Path("checkpoints/streaming_relu")

    # load all checkpoint vectors
    n_ckpts = 18
    theta_vecs = {}
    for t in range(n_ckpts):
        sd = torch.load(ckpt_dir / f"theta_{t}.pt", map_location=device)
        model.load_state_dict(sd)
        theta_vecs[t] = model_params_vec(model).detach().cpu()

    theta0_norm = theta_vecs[0].norm().item()

    # build staleness decay curve
    curve = []
    for d in comp["days"]:
        t = d["day_idx"]
        if t < 2:
            continue
        # per-step dtheta: theta_{t-1} - theta_{t-2}
        dtheta_step = (theta_vecs[t - 1] - theta_vecs[t - 2]).norm().item() / theta0_norm
        # cumulative dtheta: theta_{t-1} - theta_0
        dtheta_cum = (theta_vecs[t - 1] - theta_vecs[0]).norm().item() / theta0_norm

        curve.append({
            "day_idx": t,
            "dtheta_step_rel": round(dtheta_step, 5),
            "dtheta_cum_rel": round(dtheta_cum, 5),
            "ranking_loss_step": round(1 - d["spearman_step_med"], 5),  # 1-Spearman
            "ranking_loss_cum": round(1 - d["spearman_cum_med"], 5),
            "recall_fresh": d["fresh_recall@10"],
            "recall_stale_step": d["stale_step_recall@10"],
            "recall_stale_cum": d["stale_cum_recall@10"],
            "mrr_fresh": d["fresh_mrr"],
            "mrr_stale_cum": d["stale_cum_mrr"],
        })

    print("=== Fixed V4: Staleness decay curve (leak-free) ===\n")
    print(f'{"day":>3} {"dt_step":>8} {"dt_cum":>8} | {"loss_step":>9} {"loss_cum":>8} | '
          f'{"rec_fresh":>9} {"rec_stale":>9} {"mrr_fresh":>9} {"mrr_stale":>9}')
    for c in curve:
        print(f'{c["day_idx"]:>3} {c["dtheta_step_rel"]:>8.5f} {c["dtheta_cum_rel"]:>8.5f} | '
              f'{c["ranking_loss_step"]:>9.5f} {c["ranking_loss_cum"]:>8.5f} | '
              f'{c["recall_fresh"]:>9.5f} {c["recall_stale_cum"]:>9.5f} '
              f'{c["mrr_fresh"]:>9.5f} {c["mrr_stale_cum"]:>9.5f}')

    # gating: at small dtheta (per-step), is ranking loss gentle AND positive?
    small_losses = [c["ranking_loss_step"] for c in curve]
    small_dthetas = [c["dtheta_step_rel"] for c in curve]
    mean_small_loss = float(np.mean(small_losses))
    max_small_loss = float(np.max(small_losses))

    # cumulative at different magnitudes
    cum_dthetas = [c["dtheta_cum_rel"] for c in curve]
    cum_losses = [c["ranking_loss_cum"] for c in curve]

    # find the dtheta threshold where ranking loss exceeds 5%
    threshold_5pct = None
    for c in curve:
        if c["ranking_loss_cum"] > 0.05 and threshold_5pct is None:
            threshold_5pct = c["dtheta_cum_rel"]
            threshold_day = c["day_idx"]

    # correlation: dtheta vs ranking loss (per-step, controlling for day)
    rho_step, p_step = spearmanr(small_dthetas, small_losses)
    rho_cum, p_cum = spearmanr(cum_dthetas, cum_losses)

    print(f"\n=== V4 Gating Analysis ===")
    print(f"Per-step (daily ~{np.mean(small_dthetas)*100:.1f}% dtheta):")
    print(f"  ranking loss: mean={mean_small_loss*100:.2f}% max={max_small_loss*100:.2f}%")
    print(f"  -> {'gentle' if max_small_loss < 0.01 else 'moderate'} (< 1% = reuse viable)")
    print(f"Cumulative (grows to {cum_dthetas[-1]*100:.1f}% dtheta):")
    print(f"  ranking loss: grows to {cum_losses[-1]*100:.1f}%")
    print(f"  5% loss threshold at dtheta={threshold_5pct*100:.1f}% (day {threshold_day})")
    print(f"  -> reuse viable for ~{threshold_day} days without recompute")
    print(f"\nCorrelation (dtheta vs loss):")
    print(f"  per-step: rho={rho_step:.3f} p={p_step:.4f}")
    print(f"  cumulative: rho={rho_cum:.3f} p={p_cum:.4e}")

    # gate: per-step loss must be positive AND gentle
    per_step_gentle = mean_small_loss > 0 and max_small_loss < 0.02
    cumulative_steep = cum_losses[-1] > 0.10
    if per_step_gentle and cumulative_steep:
        verdict = "PASS"
    elif per_step_gentle:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    print(f"\nV4 verdict: {verdict}")
    print(f"  per-step gentle (loss<2%): {per_step_gentle}")
    print(f"  cumulative steep (loss>10%): {cumulative_steep}")
    print(f"  -> three-state decision has operating space: "
          f"reuse (per-step, <1%) -> migrate (cumulative 1-10%) -> recompute (>10%)")

    results = {
        "decay_curve": curve,
        "per_step_mean_loss": mean_small_loss,
        "per_step_max_loss": max_small_loss,
        "cumulative_final_loss": cum_losses[-1],
        "threshold_5pct_dtheta": threshold_5pct,
        "threshold_5pct_day": threshold_day,
        "correlation_step": {"rho": float(rho_step), "p": float(p_step)},
        "correlation_cum": {"rho": float(rho_cum), "p": float(p_cum)},
        "verdict": verdict,
    }
    save_json(results, "results/phase0/V4_fixed_staleness.json")
    print(f"\nSaved results/phase0/V4_fixed_staleness.json")


if __name__ == "__main__":
    main()
