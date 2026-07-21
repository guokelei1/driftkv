"""Phase 0 - V3: cross-user J.dtheta sharing / low-rank fit.

Roadmap Insight 5.1 (path 1): if per-user drift vectors J_u.dtheta live in a
low-rank subspace, a small probe set of JVPs can be extrapolated to all users,
dropping the online cost from |users| JVPs to |probe| JVPs.

This script:
  1. Loads KuaiRand, trains a small HSTU for a few streaming chunks to get a
     realistic dtheta (not random noise).
  2. Computes J.dtheta for a probe set of users (forward-mode JVP per user).
  3. SVDs the probe drift matrix and reports the explained-variance curve.
  4. Cross-validates: fit a low-rank predictor on half the probe set, evaluate
     drift-norm prediction error on the other half.

Gating: a small rank (e.g. 16-64) capturing >=90% variance => path 1 viable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hstu_kvcache.data import build_user_sequences, collate_batch, load_kuairand
from hstu_kvcache.drift import collect_probe_drifts, fit_low_rank
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming import StreamingTrainer
from hstu_kvcache.utils import save_json


class SeqDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = list(seqs.values())

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]


def main(
    out_path: str = "results/phase0/V3_cross_user_sharing.json",
    steps_per_chunk: int = 200,
    num_chunks: int = 3,
    probe_users: int = 64,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading KuaiRand (window 1, top-20k items)...")
    trace = load_kuairand(
        ["data/kuairand/log_standard_4_08_to_4_21_1k.csv"],
        min_interactions_per_user=5,
        max_seq_len=128,
        max_items=20000,
    )
    print(f"  users={trace.num_users} items={trace.num_items} interactions={len(trace.interactions)}")
    seqs = build_user_sequences(trace, max_seq_len=128)
    ds = SeqDataset(seqs)
    dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=lambda b: collate_batch(b, max_seq_len=128))

    cfg = HSTUConfig(
        num_items=trace.num_items,
        num_behaviors=trace.num_behaviors,
        hidden_size=128,
        num_layers=3,
        num_heads=4,
        head_dim=32,
        max_seq_len=128,
    )
    torch.manual_seed(0)
    model = HSTU(cfg).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    trainer = StreamingTrainer(model, lr=3e-4, device=device)

    print(f"Streaming training: {num_chunks} chunks x {steps_per_chunk} steps...")
    for c in range(num_chunks):
        losses = trainer.stream_chunk(dl, steps=steps_per_chunk, step_log=100)
        print(f"  chunk {c}: mean_loss={np.mean(losses):.4f}")

    dthetas = trainer.dtheta_sequence()
    dtheta_vec = dthetas[-1].to(device)
    dtheta_norm = dtheta_vec.norm().item()
    theta_norm = trainer.checkpoints[-1].state_vec.norm().item()
    print(f"dtheta: ||dtheta||={dtheta_norm:.4f}  ||theta||={theta_norm:.4f}  rel={dtheta_norm/theta_norm:.4f}")

    # probe users: compute J.dtheta per user (pad to fixed length for stacking)
    print(f"Computing J.dtheta for {probe_users} probe users...")
    probe_keys = list(seqs.keys())[:probe_users]
    fixed_len = 128
    probe_batches = [collate_batch([seqs[k]], max_seq_len=fixed_len, pad_to=fixed_len) for k in probe_keys]
    probe_matrix = collect_probe_drifts(model, probe_batches, dtheta_vec, device)
    print(f"  probe drift matrix: {probe_matrix.shape}")

    # per-user drift norms
    norms = np.linalg.norm(probe_matrix, axis=1)
    print(f"  ||J.dtheta|| per user: min={norms.min():.4f} median={np.median(norms):.4f} max={norms.max():.4f}")

    # SVD explained variance
    ranks = [1, 2, 4, 8, 16, 32, 64]
    fits = fit_low_rank(probe_matrix, ranks)
    print("\nExplained variance by rank:")
    ev_curve = []
    for r, fit in fits.items():
        print(f"  rank={r:3d}: explained_var={fit.explained_var:.4f}  recon_err={fit.fit_recon_err:.4f}")
        ev_curve.append({"rank": r, "explained_var": fit.explained_var, "recon_err": fit.fit_recon_err})

    # cross-validation: predict drift norm from low-rank features
    half = probe_users // 2
    train_m, test_m = probe_matrix[:half], probe_matrix[half:]
    train_norms = np.linalg.norm(train_m, axis=1)
    test_norms = np.linalg.norm(test_m, axis=1)
    U, S, Vt = np.linalg.svd(train_m, full_matrices=False)
    cv_results = []
    for r in [2, 4, 8, 16]:
        if r > len(S):
            break
        # regress drift-norm on top-r left singular value magnitudes
        feats_tr = np.abs(train_m @ Vt[:r].T)
        feats_te = np.abs(test_m @ Vt[:r].T)
        # least squares
        A = np.hstack([feats_tr, np.ones((half, 1))])
        coef, *_ = np.linalg.lstsq(A, train_norms, rcond=None)
        pred_te = feats_te @ coef[:r] + coef[r]
        mae = float(np.mean(np.abs(pred_te - test_norms)))
        rel_mae = float(mae / (np.mean(test_norms) + 1e-12))
        cv_results.append({"rank": r, "mae": mae, "rel_mae": rel_mae})
        print(f"  CV rank={r}: norm-pred rel_mae={rel_mae:.4f}")

    # gating verdict
    ev_at_16 = next((e["explained_var"] for e in ev_curve if e["rank"] == 16), 0.0)
    verdict = "PASS" if ev_at_16 >= 0.9 else ("MARGINAL" if ev_at_16 >= 0.7 else "FAIL")
    print(f"\nV3 verdict: {verdict} (explained_var@16={ev_at_16:.4f}, gate>=0.9)")

    save_json(
        {
            "dtheta_rel_norm": dtheta_norm / theta_norm,
            "probe_shape": list(probe_matrix.shape),
            "drift_norm_stats": {
                "min": float(norms.min()),
                "median": float(np.median(norms)),
                "max": float(norms.max()),
            },
            "explained_variance_curve": ev_curve,
            "cross_val": cv_results,
            "verdict": verdict,
        },
        out_path,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
