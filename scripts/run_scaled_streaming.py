"""Scaled streaming training: 6-layer/256-dim/512-seq model on KuaiRand-1K.

Produces theta_0..theta_17 with the larger model, and runs the necessity
experiment (stale vs fresh) at each day. Saves checkpoints for KV reuse tests.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hstu_kvcache.data import StreamingDataPlan
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.utils import save_json

# Scaled config: 6 layers, 256 hidden, 8 heads, 64 head_dim, 512 seq
MODEL_CFG = dict(hidden_size=256, num_layers=6, num_heads=8, head_dim=64, max_seq_len=512, activation="relu")
BASE_EPOCHS = 12
STREAM_STEPS = 100
STREAM_LR = 1e-4
REPLAY_RATIO = 0.5
REPLAY_SIZE = 300
MAX_EVAL_USERS = 500
NUM_NEG = 1000


def eval_multi(model, eval_samples, num_items, device, num_neg=NUM_NEG, k=10):
    model.eval()
    hits, ndcgs, mrrs, recalls = [], [], [], []
    with torch.no_grad():
        for sample in eval_samples:
            seq = sample["history"]
            pos_set = set(sample["pos_items"])
            n_pos = len(pos_set)
            item_ids = torch.tensor(seq["item_ids"], device=device).unsqueeze(0)
            behs = torch.tensor(seq["behaviors"], device=device).unsqueeze(0)
            tds = torch.tensor(seq["time_deltas"], device=device).unsqueeze(0)
            hidden, _ = model(item_ids, behs, tds, return_kv=False)
            last_h = hidden[:, -1, :]
            negs = np.random.randint(1, num_items + 1, size=num_neg)
            cands = np.unique(np.concatenate([list(pos_set), negs]))
            cand_t = torch.tensor(cands, device=device).unsqueeze(0)
            scores = model.item_emb.score(last_h, cand_t).squeeze(0).cpu().numpy()
            order = np.argsort(-scores)
            pos_mask = np.array([c in pos_set for c in cands])
            ranked_pos = pos_mask[order]
            pos_pos = np.where(ranked_pos)[0]
            if len(pos_pos) == 0:
                hits.append(0.0); ndcgs.append(0.0); mrrs.append(0.0); recalls.append(0.0)
                continue
            mrrs.append(1.0 / (pos_pos[0] + 1))
            in_topk = pos_pos[pos_pos < k]
            hits.append(1.0 if len(in_topk) > 0 else 0.0)
            recalls.append(len(in_topk) / n_pos)
            dcg = sum(1.0 / np.log2(r + 2) for r in in_topk)
            idcg = sum(1.0 / np.log2(j + 2) for j in range(min(n_pos, k)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return {"hit@10": float(np.mean(hits)), "ndcg@10": float(np.mean(ndcgs)),
            "mrr": float(np.mean(mrrs)), "recall@10": float(np.mean(recalls)), "n": len(hits)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(0); torch.manual_seed(0)
    seq_len = MODEL_CFG["max_seq_len"]

    print("=== Scaled Streaming Training (6L/256D/512S, relu activation) ===")
    plan = StreamingDataPlan.from_csvs(
        ["data/kuairand/log_standard_4_08_to_4_21_1k.csv",
         "data/kuairand/log_standard_4_22_to_5_08_1k.csv"],
        base_num_days=14, max_seq_len=seq_len, max_items=20000)
    plan.init_base()
    print(f"users={plan.num_users} items={plan.num_items} seq_len={seq_len}")

    cfg = HSTUConfig(num_items=plan.num_items, num_behaviors=plan.num_behaviors, **MODEL_CFG)
    model = HSTU(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} ({n_params/1e6:.1f}M)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # base training with replay buffer
    from hstu_kvcache.streaming.trainer import train_step
    print(f"Base training ({BASE_EPOCHS} epochs)...")
    replay_buffer = []
    for epoch in range(BASE_EPOCHS):
        step = 0
        for batch in plan.iter_base_train_batches(batch_size=32):
            loss = train_step(model, batch, optimizer, device)
            if epoch == BASE_EPOCHS - 1 and step < REPLAY_SIZE:
                replay_buffer.append({k: v.clone() for k, v in batch.items()})
            step += 1
            if step % 500 == 0:
                print(f"  epoch {epoch} step {step}: loss={loss:.4f}")
    print(f"Base done. ||theta_0||={model_params_vec(model).norm().item():.2f}")

    # stale model (frozen theta_0)
    model_stale = HSTU(cfg).to(device)
    model_stale.load_state_dict(copy.deepcopy(model.state_dict()))
    model_stale.eval()
    for p in model_stale.parameters():
        p.requires_grad_(False)

    # save theta_0
    ckpt_dir = Path("checkpoints/streaming_relu")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "theta_0.pt")

    # switch to streaming lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=STREAM_LR, weight_decay=1e-4)

    results = {"model_cfg": MODEL_CFG, "n_params": n_params, "days": []}

    for di, date in enumerate(plan.stream_dates):
        day_idx = di + 1
        eval_samples = plan.get_eval_set(date, max_users=MAX_EVAL_USERS)
        if not eval_samples:
            plan.ingest_day(date); continue

        stale_m = eval_multi(model_stale, eval_samples, plan.num_items, device)
        fresh_m = eval_multi(model, eval_samples, plan.num_items, device)

        prev_vec = model_params_vec(model).detach().clone()
        plan.ingest_day(date)

        # stream train with replay
        new_batches = list(plan.iter_train_batches(date, batch_size=32))
        np.random.shuffle(new_batches)
        n_new = int(len(new_batches) * (1 - REPLAY_RATIO))
        mixed = new_batches[:n_new]
        for _ in range(len(new_batches) - n_new):
            rb = replay_buffer[np.random.randint(len(replay_buffer))]
            mixed.append({k: v.clone() for k, v in rb.items()})
        np.random.shuffle(mixed)
        losses = []
        for batch in mixed:
            losses.append(train_step(model, batch, optimizer, device))
            if len(losses) >= STREAM_STEPS:
                break
        for batch in new_batches[:10]:
            replay_buffer.append({k: v.clone() for k, v in batch.items()})
            if len(replay_buffer) > REPLAY_SIZE:
                replay_buffer.pop(np.random.randint(len(replay_buffer)))

        curr_vec = model_params_vec(model).detach().clone()
        dtheta_rel = (curr_vec - prev_vec).norm().item() / (prev_vec.norm().item() + 1e-12)
        torch.save(model.state_dict(), ckpt_dir / f"theta_{day_idx}.pt")

        print(f"Day {day_idx:2d} ({date}): stale hit@10={stale_m['hit@10']:.4f} mrr={stale_m['mrr']:.4f} | "
              f"fresh hit@10={fresh_m['hit@10']:.4f} mrr={fresh_m['mrr']:.4f} | dtheta={dtheta_rel:.4f}")
        results["days"].append({
            "date": date, "day_idx": day_idx,
            "stale": stale_m, "fresh": fresh_m, "dtheta_rel": round(dtheta_rel, 5),
            "train_loss": float(np.mean(losses)),
        })

    save_json(results, "results/streaming/relu_necessity.json")
    print(f"\nSaved results + {len(results['days'])+1} checkpoints to {ckpt_dir}")
    # summary
    sh = [d["stale"]["hit@10"] for d in results["days"]]
    fh = [d["fresh"]["hit@10"] for d in results["days"]]
    sm = [d["stale"]["mrr"] for d in results["days"]]
    fm = [d["fresh"]["mrr"] for d in results["days"]]
    print(f"hit@10: stale {sh[0]*100:.1f}%->{sh[-1]*100:.1f}% | fresh {fh[0]*100:.1f}%->{fh[-1]*100:.1f}%")
    print(f"mrr:    stale {sm[0]*100:.1f}%->{sm[-1]*100:.1f}% | fresh {fm[0]*100:.1f}%->{fm[-1]*100:.1f}%")


if __name__ == "__main__":
    main()
