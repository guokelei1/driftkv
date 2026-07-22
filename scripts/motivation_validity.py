"""Small, validity-first motivation experiment for version-stale HSTU KV caches."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from hstu_kvcache.data import StreamingDataPlan, collate_batch
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.streaming.trainer import build_next_item_targets, train_step
from hstu_kvcache.utils import save_json

STANDARD_LOGS = [
    "data/kuairand/log_standard_4_08_to_4_21_1k.csv",
    "data/kuairand/log_standard_4_22_to_5_08_1k.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="validity_v1_incremental_prefix_cache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-items", type=int, default=10000)
    parser.add_argument("--base-days", type=int, default=14)
    parser.add_argument("--base-epochs", type=int, default=8)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--stream-lr", type=float, default=1e-4)
    parser.add_argument("--stream-window-days", type=int, default=3)
    parser.add_argument("--stream-epochs", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=5)
    parser.add_argument("--max-eval-users", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--training-sequences",
        choices=["latest", "all_chunks"],
        default="latest",
    )
    parser.add_argument("--output", default="results/validity/motivation_seed0.json")
    parser.add_argument("--checkpoint-dir")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(args: argparse.Namespace, num_items: int, num_behaviors: int) -> HSTU:
    cfg = HSTUConfig(
        num_items=num_items,
        num_behaviors=num_behaviors,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        max_seq_len=args.seq_len,
        activation="relu",
    )
    return HSTU(cfg).to(args.device)


def clone_state(model: HSTU) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def prepare_sequence(history: dict, seq_len: int) -> dict:
    length = min(len(history["item_ids"]), seq_len)
    return {
        "item_ids": history["item_ids"][-length:],
        "behaviors": history["behaviors"][-length:],
        "time_deltas": history["time_deltas"][-length:],
        "labels": history["labels"][-length:],
    }


def eval_batches(samples: list[dict], seq_len: int, batch_size: int):
    usable = [sample for sample in samples if len(sample["history"]["item_ids"]) >= 2]
    for start in range(0, len(usable), batch_size):
        selected = usable[start : start + batch_size]
        full_sequences = [prepare_sequence(sample["history"], seq_len) for sample in selected]
        prefix_sequences = [
            {name: values[:-1] for name, values in sequence.items()}
            for sequence in full_sequences
        ]
        full = collate_batch(full_sequences, max_seq_len=seq_len)
        prefix = collate_batch(prefix_sequences, max_seq_len=seq_len - 1)
        suffix = {
            "item_ids": torch.tensor(
                [[sequence["item_ids"][-1]] for sequence in full_sequences],
                dtype=torch.long,
            ),
            "behaviors": torch.tensor(
                [[sequence["behaviors"][-1]] for sequence in full_sequences],
                dtype=torch.long,
            ),
            "time_deltas": torch.tensor(
                [[sequence["time_deltas"][-1]] for sequence in full_sequences],
                dtype=torch.float32,
            ),
        }
        yield selected, full, prefix, suffix


def move_batch(batch: dict, device: torch.device) -> dict:
    return {name: value.to(device) for name, value in batch.items()}


def ranking_metrics(scores: torch.Tensor, positives: list[int]) -> dict[str, float]:
    num_items = scores.numel()
    pos = torch.tensor(
        sorted({item - 1 for item in positives if 1 <= item <= num_items}),
        device=scores.device,
    )
    pos_scores = scores[pos]
    ranks = torch.stack([(scores > score).sum() + 1 for score in pos_scores]).float()
    best_rank = float(ranks.min().item())
    mean_rank = float(ranks.mean().item())
    metrics = {
        "mrr": 1.0 / best_rank,
        "best_rank": best_rank,
        "mean_rank": mean_rank,
        "rank_utility": -math.log1p(best_rank),
    }
    pos_set = set(pos.tolist())
    for k in (10, 100):
        actual_k = min(k, num_items)
        top = torch.topk(scores, actual_k).indices.tolist()
        relevance = [1.0 if item in pos_set else 0.0 for item in top]
        dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevance))
        idcg = sum(
            1.0 / math.log2(rank + 2)
            for rank in range(min(len(pos_set), actual_k))
        )
        metrics[f"hit@{k}"] = float(any(relevance))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return metrics


def bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def safe_spearman(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"rho": float("nan"), "p": float("nan")}
    rho, p = spearmanr(x, y)
    return {"rho": float(rho), "p": float(p)}


def selection_curves(
    records: list[dict],
    rng: np.random.Generator,
) -> list[dict]:
    drift = np.array([record["kv_drift_rel"] for record in records])
    gain = np.array([record["rank_utility_gain"] for record in records])
    stale = np.array([record["stale_rank_utility"] for record in records])
    base = float(stale.mean())
    output = []
    for budget in (0.1, 0.2, 0.5):
        count = max(1, int(round(len(records) * budget)))
        drift_idx = np.argsort(-drift)[:count]
        oracle_idx = np.argsort(-gain)[:count]
        random_values = []
        for _ in range(200):
            random_idx = rng.choice(len(records), count, replace=False)
            random_values.append(base + float(gain[random_idx].sum() / len(records)))
        output.append(
            {
                "budget": budget,
                "count": count,
                "all_reuse": base,
                "all_recompute": base + float(gain.mean()),
                "drift_select": base + float(gain[drift_idx].sum() / len(records)),
                "oracle_select": base + float(gain[oracle_idx].sum() / len(records)),
                "random_select_mean": float(np.mean(random_values)),
                "random_select_ci95": [
                    float(np.quantile(random_values, 0.025)),
                    float(np.quantile(random_values, 0.975)),
                ],
            }
        )
    return output


def summarize_records(
    records: list[dict],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict:
    summary: dict[str, object] = {"n": len(records)}
    for metric in ("mrr", "ndcg@10", "ndcg@100", "hit@10", "hit@100"):
        fresh = np.array([record[f"fresh_{metric}"] for record in records])
        stale = np.array([record[f"stale_{metric}"] for record in records])
        gain = fresh - stale
        summary[metric] = {
            "fresh": float(fresh.mean()),
            "stale": float(stale.mean()),
            "gain": float(gain.mean()),
            "gain_ci95": bootstrap_interval(gain, rng, bootstrap_samples),
            "fresh_better_fraction": float(np.mean(gain > 0)),
        }
    for metric in ("best_rank", "mean_rank"):
        fresh = np.array([record[f"fresh_{metric}"] for record in records])
        stale = np.array([record[f"stale_{metric}"] for record in records])
        gain = stale - fresh
        summary[metric] = {
            "fresh": float(fresh.mean()),
            "stale": float(stale.mean()),
            "gain": float(gain.mean()),
            "gain_ci95": bootstrap_interval(gain, rng, bootstrap_samples),
            "fresh_better_fraction": float(np.mean(gain > 0)),
        }
    kv_drift = np.array([record["kv_drift_rel"] for record in records])
    rank_gain = np.array([record["rank_utility_gain"] for record in records])
    mrr_gain = np.array([record["mrr_gain"] for record in records])
    summary["fidelity"] = {
        "kv_drift_rel": float(kv_drift.mean()),
        "hidden_cosine": float(np.mean([record["hidden_cosine"] for record in records])),
        "score_cosine": float(np.mean([record["score_cosine"] for record in records])),
        "top10_changed_fraction": float(
            np.mean([record["top10_changed_fraction"] for record in records])
        ),
        "incremental_parity_max_abs": float(
            max(record["incremental_parity_max_abs"] for record in records)
        ),
    }
    summary["drift_quality_correlation"] = {
        "rank_utility_gain": safe_spearman(kv_drift, rank_gain),
        "mrr_gain": safe_spearman(kv_drift, mrr_gain),
    }
    summary["selection"] = selection_curves(records, rng)
    return summary


@torch.inference_mode()
def evaluate_version_pair(
    current_model: HSTU,
    old_model: HSTU,
    samples: list[dict],
    args: argparse.Namespace,
    window: int,
) -> tuple[list[dict], dict]:
    device = torch.device(args.device)
    current_model.eval()
    old_model.eval()
    all_items = torch.arange(1, current_model.cfg.num_items + 1, device=device)
    records = []
    for selected, full_cpu, prefix_cpu, suffix_cpu in eval_batches(
        samples,
        args.seq_len,
        args.batch_size,
    ):
        full = move_batch(full_cpu, device)
        prefix = move_batch(prefix_cpu, device)
        suffix = move_batch(suffix_cpu, device)
        full_hidden, _ = current_model(
            full["item_ids"],
            full["behaviors"],
            full["time_deltas"],
            lengths=full["lengths"],
        )
        fresh_hidden = current_model.last_hidden(full_hidden, full["lengths"])
        fresh_kv = current_model.compute_kv(
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            lengths=prefix["lengths"],
        )
        old_kv = old_model.compute_kv(
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            lengths=prefix["lengths"],
        )
        fresh_incremental, _ = current_model.forward_with_cache(
            fresh_kv,
            suffix["item_ids"],
            suffix["behaviors"],
            suffix["time_deltas"],
        )
        stale_incremental, _ = current_model.forward_with_cache(
            old_kv,
            suffix["item_ids"],
            suffix["behaviors"],
            suffix["time_deltas"],
        )
        fresh_incremental = fresh_incremental[:, 0]
        stale_incremental = stale_incremental[:, 0]
        parity = (fresh_incremental - fresh_hidden).abs().amax(dim=1)
        fresh_scores = current_model.item_emb.score(
            fresh_hidden,
            all_items.unsqueeze(0).expand(len(selected), -1),
        )
        stale_scores = current_model.item_emb.score(
            stale_incremental,
            all_items.unsqueeze(0).expand(len(selected), -1),
        )
        drift_sq = (fresh_kv.k.float() - old_kv.k.float()).square().sum(dim=(0, 2, 3))
        drift_sq += (fresh_kv.v.float() - old_kv.v.float()).square().sum(dim=(0, 2, 3))
        fresh_sq = fresh_kv.k.float().square().sum(dim=(0, 2, 3))
        fresh_sq += fresh_kv.v.float().square().sum(dim=(0, 2, 3))
        drift_rel = drift_sq.sqrt() / fresh_sq.sqrt().clamp_min(1e-12)
        hidden_cosine = F.cosine_similarity(fresh_incremental, stale_incremental, dim=-1)
        score_cosine = F.cosine_similarity(fresh_scores, stale_scores, dim=-1)
        fresh_top10 = torch.topk(fresh_scores, min(10, all_items.numel()), dim=-1).indices
        stale_top10 = torch.topk(stale_scores, min(10, all_items.numel()), dim=-1).indices
        for row, sample in enumerate(selected):
            fresh_metrics = ranking_metrics(fresh_scores[row], sample["pos_items"])
            stale_metrics = ranking_metrics(stale_scores[row], sample["pos_items"])
            overlap = len(set(fresh_top10[row].tolist()) & set(stale_top10[row].tolist()))
            record = {
                "window": window,
                "user_id": int(sample["history"]["user_id"]),
                "history_length": int(full["lengths"][row].item()),
                "num_positives": len(sample["pos_items"]),
                "kv_drift_rel": float(drift_rel[row].item()),
                "hidden_cosine": float(hidden_cosine[row].item()),
                "score_cosine": float(score_cosine[row].item()),
                "top10_changed_fraction": 1.0 - overlap / min(10, all_items.numel()),
                "incremental_parity_max_abs": float(parity[row].item()),
            }
            for name, value in fresh_metrics.items():
                record[f"fresh_{name}"] = value
            for name, value in stale_metrics.items():
                record[f"stale_{name}"] = value
            record["mrr_gain"] = record["fresh_mrr"] - record["stale_mrr"]
            record["rank_utility_gain"] = (
                record["fresh_rank_utility"] - record["stale_rank_utility"]
            )
            records.append(record)
    rng = np.random.default_rng(args.seed * 1000 + window)
    return records, summarize_records(records, rng, args.bootstrap_samples)


def save_checkpoint(model: HSTU, directory: str | None, name: str) -> None:
    if directory is None:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / f"{name}.pt")


def batch_coverage(batches: list[dict]) -> dict[str, int]:
    sequences = 0
    tokens = 0
    targets = 0
    for batch in batches:
        lengths = batch["lengths"]
        sequences += len(lengths)
        tokens += int(lengths.sum())
        _, valid = build_next_item_targets(
            batch["item_ids"],
            lengths,
            batch.get("labels"),
            batch.get("train_mask"),
        )
        targets += int(valid.sum())
    return {"sequences": sequences, "tokens": tokens, "eligible_targets": targets}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    print(f"device={args.device} seed={args.seed}", flush=True)
    plan = StreamingDataPlan.from_csvs(
        STANDARD_LOGS,
        base_num_days=args.base_days,
        max_seq_len=args.seq_len,
        max_items=args.max_items,
        fit_vocabulary_on_base=True,
    )
    plan.init_base()
    print(
        f"users={plan.num_users} items={plan.num_items} stream_days={len(plan.stream_dates)}",
        flush=True,
    )
    model = make_model(args, plan.num_items, plan.num_behaviors)
    old_model = make_model(args, plan.num_items, plan.num_behaviors)
    base_model = make_model(args, plan.num_items, plan.num_behaviors)
    for frozen_model in (old_model, base_model):
        for parameter in frozen_model.parameters():
            parameter.requires_grad_(False)
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}", flush=True)

    base_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=1e-4,
    )
    base_losses = []
    base_coverage = []
    all_chunks = args.training_sequences == "all_chunks"
    for epoch in range(args.base_epochs):
        batches = list(
            plan.iter_base_train_batches(
                args.batch_size,
                all_chunks=all_chunks,
            )
        )
        base_coverage.append(batch_coverage(batches))
        losses = [
            train_step(model, batch, base_optimizer, torch.device(args.device))
            for batch in batches
        ]
        mean_loss = float(np.mean([loss for loss in losses if loss > 0]))
        base_losses.append(mean_loss)
        print(f"base_epoch={epoch + 1} loss={mean_loss:.5f}", flush=True)
    theta0 = model_params_vec(model).detach().cpu().clone()
    base_model.load_state_dict(model.state_dict())
    save_checkpoint(model, args.checkpoint_dir, "theta_0")

    stream_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.stream_lr,
        weight_decay=1e-4,
    )
    result = {
        "protocol": args.protocol,
        "args": vars(args),
        "data": {
            "num_users": plan.num_users,
            "num_items": plan.num_items,
            "base_dates": plan.base_dates,
            "stream_dates": plan.stream_dates,
        },
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "base_losses": base_losses,
        "base_training_coverage": base_coverage,
        "windows": [],
    }
    window_days = args.stream_window_days
    possible_windows = max(0, (len(plan.stream_dates) - 1) // window_days)
    for window in range(1, min(args.max_windows, possible_windows) + 1):
        start = (window - 1) * window_days
        end = start + window_days
        train_dates = plan.stream_dates[start:end]
        eval_date = plan.stream_dates[end]
        previous_state = clone_state(model)
        previous_vec = model_params_vec(model).detach().cpu().clone()
        window_losses = []
        window_coverage = []
        for date in train_dates:
            plan.ingest_day(date)
            batches = list(
                plan.iter_train_batches(
                    date,
                    args.batch_size,
                    all_chunks=all_chunks,
                )
            )
            window_coverage.append({"date": date, **batch_coverage(batches)})
            for _ in range(args.stream_epochs):
                random.shuffle(batches)
                for batch in batches:
                    loss = train_step(
                        model,
                        batch,
                        stream_optimizer,
                        torch.device(args.device),
                    )
                    if loss > 0:
                        window_losses.append(loss)
        current_vec = model_params_vec(model).detach().cpu().clone()
        step_dtheta = float((current_vec - previous_vec).norm() / previous_vec.norm())
        cumulative_dtheta = float((current_vec - theta0).norm() / theta0.norm())
        old_model.load_state_dict(previous_state)
        eval_samples = plan.get_eval_set(eval_date, args.max_eval_users)
        records, summary = evaluate_version_pair(
            model,
            old_model,
            eval_samples,
            args,
            window,
        )
        if window == 1:
            cumulative_records = records
            cumulative_summary = summary
        else:
            cumulative_records, cumulative_summary = evaluate_version_pair(
                model,
                base_model,
                eval_samples,
                args,
                window,
            )
        window_result = {
            "window": window,
            "train_dates": train_dates,
            "eval_date": eval_date,
            "step_dtheta_rel": step_dtheta,
            "cumulative_dtheta_rel": cumulative_dtheta,
            "train_loss": float(np.mean(window_losses)),
            "training_coverage": window_coverage,
            "summary": summary,
            "per_user": records,
            "cumulative_theta0": {
                "summary": cumulative_summary,
                "per_user": cumulative_records,
            },
        }
        result["windows"].append(window_result)
        save_checkpoint(model, args.checkpoint_dir, f"theta_{window}")
        save_json(result, args.output)
        rank = summary["best_rank"]
        cumulative_rank = cumulative_summary["best_rank"]
        corr = summary["drift_quality_correlation"]["rank_utility_gain"]
        print(
            f"window={window} eval={eval_date} n={summary['n']} "
            f"dtheta={step_dtheta:.5f} kv={summary['fidelity']['kv_drift_rel']:.5f} "
            f"rank_fresh={rank['fresh']:.1f} rank_stale={rank['stale']:.1f} "
            f"gain={rank['gain']:.1f} ci={rank['gain_ci95']} "
            f"cumulative_gain={cumulative_rank['gain']:.1f} "
            f"drift_gain_rho={corr['rho']:.3f}",
            flush=True,
        )

    pooled = [
        record
        for window_result in result["windows"]
        for record in window_result["per_user"]
    ]
    if pooled:
        result["pooled_descriptive"] = summarize_records(
            pooled,
            np.random.default_rng(args.seed + 9999),
            args.bootstrap_samples,
        )
    pooled_cumulative = [
        record
        for window_result in result["windows"]
        for record in window_result["cumulative_theta0"]["per_user"]
    ]
    if pooled_cumulative:
        result["pooled_cumulative_theta0_descriptive"] = summarize_records(
            pooled_cumulative,
            np.random.default_rng(args.seed + 19999),
            args.bootstrap_samples,
        )
    result["runtime_seconds"] = time.perf_counter() - started
    save_json(result, args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "windows": len(result["windows"]),
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
