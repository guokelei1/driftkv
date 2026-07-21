"""Phase 0 - V4: stale-KV accuracy decay vs streaming-update magnitude.

Roadmap V4: "参数小步长更新后，直接用旧 KV 的推荐精度衰减". Gating:
gentle decay => reuse/migrate has space; steep decay => only recompute.

We train theta_0 on KuaiRand window 1, then apply streaming updates of
*increasing* size on window 2 (10, 25, 50, 100, 200 steps). At each update
magnitude we measure:
  - stale  : hit@10 of theta_0 (serving with KV_0 = F(theta_0, x_u))
  - fresh  : hit@10 of theta_1 (serving with KV_1 = F(theta_1, x_u))
  - gap    : fresh - stale  = staleness decay at that update size
  - ||dtheta|| / ||theta|| : the relative parameter drift

The decay *curve* (gap vs dtheta size) is the gating artefact: a gentle slope
near small dtheta means reuse/migration is viable for hourly-scale updates.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hstu_kvcache.data import build_user_sequences, collate_batch, load_kuairand
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming import StreamingTrainer, model_params_vec
from hstu_kvcache.utils import save_json


class SeqDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = list(seqs.values())

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]


def evaluate_hit_at_k(model, seqs, num_items, device, k=10, num_neg=50, max_users=300):
    model.eval()
    keys = list(seqs.keys())[:max_users]
    hits, ndcgs = [], []
    with torch.no_grad():
        for key in keys:
            s = seqs[key]
            if len(s["item_ids"]) < 2:
                continue
            batch = collate_batch([s], max_seq_len=128, pad_to=128)
            hidden, _ = model(
                batch["item_ids"].to(device),
                batch["behaviors"].to(device),
                batch["time_deltas"].to(device),
                return_kv=False,
            )
            last_h = hidden[:, -1, :]
            pos_item = int(s["item_ids"][-1])
            negs = np.random.randint(1, num_items + 1, size=num_neg)
            cands = np.unique(np.concatenate([[pos_item], negs]))
            if pos_item not in cands:
                cands = np.concatenate([[pos_item], cands])
            scores = model.item_emb.score(last_h, torch.tensor(cands, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
            pos_idx = int(np.where(cands == pos_item)[0][0])
            topk = np.argsort(-scores)[:k]
            hit = int(pos_idx in topk)
            hits.append(hit)
            ndcgs.append(1.0 / np.log2(list(topk).index(pos_idx) + 2) if hit else 0.0)
    return {"hit@10": float(np.mean(hits)), "ndcg@10": float(np.mean(ndcgs)), "n": len(hits)}


def main(out_path: str = "results/phase0/V4_staleness_decay.json") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(0)
    torch.manual_seed(0)

    print("Loading KuaiRand window 1 (Apr 8-21)...")
    trace1 = load_kuairand(["data/kuairand/log_standard_4_08_to_4_21_1k.csv"],
                           min_interactions_per_user=5, max_seq_len=128, max_items=20000)
    seqs1 = build_user_sequences(trace1, max_seq_len=128)
    dl1 = DataLoader(SeqDataset(seqs1), batch_size=32, shuffle=True,
                     collate_fn=lambda b: collate_batch(b, max_seq_len=128))

    cfg = HSTUConfig(num_items=trace1.num_items, num_behaviors=trace1.num_behaviors,
                     hidden_size=128, num_layers=3, num_heads=4, head_dim=32, max_seq_len=128)
    model = HSTU(cfg).to(device)
    trainer = StreamingTrainer(model, lr=3e-4, device=device)

    print("Training theta_0 on window 1 (4 chunks x 200 steps)...")
    for c in range(4):
        losses = trainer.stream_chunk(dl1, steps=200)
        print(f"  chunk {c}: loss={np.mean(losses):.4f}")
    theta0_state = copy.deepcopy(trainer.checkpoints[-1].model_state)
    theta0_vec = trainer.checkpoints[-1].state_vec

    print("Loading KuaiRand window 2 (Apr 22-May 8)...")
    trace2 = load_kuairand(["data/kuairand/log_standard_4_22_to_5_08_1k.csv"],
                           min_interactions_per_user=5, max_seq_len=128, max_items=20000)
    trace2.interactions = trace2.interactions[trace2.interactions["video_id"].isin(trace1.item_map)]
    trace2.interactions = trace2.interactions[trace2.interactions["user_id"].isin(trace1.user_map)]
    trace2.interactions["user_idx"] = trace2.interactions["user_id"].map(trace1.user_map)
    trace2.interactions["item_idx"] = trace2.interactions["video_id"].map(trace1.item_map)
    seqs2 = build_user_sequences(trace2, max_seq_len=128)
    dl2 = DataLoader(SeqDataset(seqs2), batch_size=32, shuffle=True,
                     collate_fn=lambda b: collate_batch(b, max_seq_len=128))

    # stale baseline (theta_0 on window 2)
    trainer.load_checkpoint(len(trainer.checkpoints) - 1)
    stale_base = evaluate_hit_at_k(model, seqs2, trace1.num_items, device)
    print(f"  STALE baseline (theta_0): {stale_base}")

    # sweep streaming update magnitudes on window 2
    update_sizes = [10, 25, 50, 100, 200]
    curve = []
    it2 = iter(dl2)
    for nsteps in update_sizes:
        # reset to theta_0 each time to isolate update magnitude
        model.load_state_dict(theta0_state)
        model.to(device)
        trainer.model = model
        trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        for _ in range(nsteps):
            try:
                batch = next(it2)
            except StopIteration:
                it2 = iter(dl2)
                batch = next(it2)
            from hstu_kvcache.streaming.trainer import train_step
            train_step(model, batch, trainer.optimizer, device)
        theta1_vec = model_params_vec(model)
        dtheta_rel = (theta1_vec - theta0_vec.to(device)).norm().item() / (theta0_vec.to(device).norm().item() + 1e-12)
        fresh = evaluate_hit_at_k(model, seqs2, trace1.num_items, device)
        gap = fresh["hit@10"] - stale_base["hit@10"]
        rec = {"update_steps": nsteps, "dtheta_rel": round(dtheta_rel, 5),
               "stale_hit@10": round(stale_base["hit@10"], 4),
               "fresh_hit@10": round(fresh["hit@10"], 4),
               "gap": round(gap, 4), "fresh_ndcg@10": round(fresh["ndcg@10"], 4)}
        print(f"  update={nsteps:3d} steps  dtheta_rel={dtheta_rel:.4f}  stale={stale_base['hit@10']:.4f}  fresh={fresh['hit@10']:.4f}  gap={gap:.4f}")
        curve.append(rec)

    # gating: at the smallest update (10 steps, ~hourly scale), is the gap gentle?
    small = curve[0]
    gentle = small["gap"] < 0.05  # <5% absolute hit@10 drop tolerated for reuse
    verdict = "PASS" if gentle else ("MARGINAL" if small["gap"] < 0.15 else "FAIL")
    print(f"\nV4 verdict: {verdict} (smallest-update gap={small['gap']:.4f}, gate<0.05)")

    save_json({"stale_baseline": stale_base, "decay_curve": curve, "verdict": verdict}, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
