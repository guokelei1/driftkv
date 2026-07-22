from __future__ import annotations

import argparse
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from layerwise_validity import summarize, timed_call
from motivation_validity import move_batch, ranking_metrics, seed_everything

from hstu_kvcache.data import collate_batch, load_movielens_hard
from hstu_kvcache.migration import (
    capture_layerwise_state,
    extra_state_numel,
    migrate_suffix_cache,
    sample_relative_cache_error,
)
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.streaming.trainer import train_step
from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="data/movielens/pilot20")
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=24)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-epochs", type=int, default=4)
    parser.add_argument("--stream-epochs", type=int, default=2)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--stream-lr", type=float, default=1e-4)
    parser.add_argument("--max-train-users", type=int)
    parser.add_argument("--max-eval-users", type=int, default=1000)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--output")
    parser.add_argument("--checkpoint-dir")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    if args.output is None:
        args.output = f"results/scaling/movielens_seed{args.seed}.json"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints/scaling/movielens_seed{args.seed}"


def make_model(args: argparse.Namespace, num_items: int) -> HSTU:
    cfg = HSTUConfig(
        num_items=num_items,
        num_behaviors=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        max_seq_len=args.seq_len,
        activation="relu",
    )
    return HSTU(cfg).to(args.device)


def sequence(record: dict, include_target_mask: bool) -> dict:
    length = len(record["history"])
    values = {
        "item_ids": record["history"],
        "behaviors": np.ones(length, dtype=np.int64),
        "time_deltas": np.zeros(length, dtype=np.float32),
        "labels": np.ones(length, dtype=np.int64),
        "train_mask": np.ones(length, dtype=np.bool_),
    }
    if include_target_mask:
        values["train_mask"][:-1] = False
    return values


def train_epoch(
    model: HSTU,
    records: list[dict],
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    rng: random.Random,
    include_target_mask: bool,
) -> float:
    order = list(range(len(records)))
    rng.shuffle(order)
    losses = []
    for start in range(0, len(order), args.batch_size):
        selected = [sequence(records[index], include_target_mask) for index in order[start : start + args.batch_size]]
        batch = collate_batch(selected, max_seq_len=args.seq_len, pad_to=args.seq_len)
        loss = train_step(model, batch, optimizer, torch.device(args.device))
        if loss > 0:
            losses.append(loss)
    return float(np.mean(losses))


def config_name(depth: int, num_layers: int) -> str:
    if depth == 0:
        return "cheap_all"
    if depth == num_layers:
        return "recompute"
    return f"suffix_{depth}"


def selected_suffix_depths(num_layers: int) -> list[int]:
    return sorted(
        {
            0,
            max(1, round(num_layers / 3)),
            max(1, round(2 * num_layers / 3)),
            max(1, num_layers - 1),
            num_layers,
        }
    )


def candidate_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    positive = torch.where(labels > 0)[0]
    ranks = torch.stack([(scores > scores[index]).sum() + 1 for index in positive]).float()
    rank = float(ranks.min().item())
    return {
        "mrr": 1.0 / rank,
        "ndcg@10": float(1.0 / np.log2(rank + 1.0)) if rank <= 10 else 0.0,
        "hit@10": float(rank <= 10),
        "best_rank": rank,
    }


def candidate_summary(per_user: list[dict]) -> dict:
    names = list(per_user[0]["candidate_configs"])
    metrics = ("mrr", "ndcg@10", "hit@10", "best_rank")
    output = {}
    for name in names:
        values = {
            metric: float(np.mean([row["candidate_configs"][name][metric] for row in per_user]))
            for metric in metrics
        }
        gains = {}
        for metric in metrics:
            current = np.array([row["candidate_configs"][name][metric] for row in per_user])
            reuse = np.array([row["candidate_configs"]["reuse"][metric] for row in per_user])
            gains[metric] = float(np.mean(reuse - current) if metric == "best_rank" else np.mean(current - reuse))
        output[name] = {"metrics": values, "gain_over_reuse": gains}
    fresh = {
        metric: float(np.mean([row["candidate_fresh"][metric] for row in per_user]))
        for metric in metrics
    }
    return {"fresh_full": fresh, "configs": output}


def evaluation_batches(records: list[dict], seq_len: int, batch_size: int):
    usable = sorted(
        (record for record in records if len(record["history"]) >= 2),
        key=lambda record: min(len(record["history"]), seq_len),
    )
    group_start = 0
    while group_start < len(usable):
        effective_length = min(len(usable[group_start]["history"]), seq_len)
        group_end = group_start + 1
        while (
            group_end < len(usable)
            and min(len(usable[group_end]["history"]), seq_len) == effective_length
        ):
            group_end += 1
        group = usable[group_start:group_end]
        for start in range(0, len(group), batch_size):
            selected = group[start : start + batch_size]
            full_sequences = [sequence(record, False) for record in selected]
            full_sequences = [
                {name: values[-seq_len:] for name, values in value.items()}
                for value in full_sequences
            ]
            prefix_sequences = [
                {name: values[:-1] for name, values in value.items()}
                for value in full_sequences
            ]
            full = collate_batch(full_sequences, max_seq_len=seq_len)
            prefix = collate_batch(prefix_sequences, max_seq_len=seq_len - 1)
            suffix = {
                "item_ids": torch.tensor([[value["item_ids"][-1]] for value in full_sequences]),
                "behaviors": torch.ones(len(selected), 1, dtype=torch.long),
                "time_deltas": torch.zeros(len(selected), 1),
            }
            candidates = torch.from_numpy(np.stack([record["candidates"] for record in selected]))
            labels = torch.from_numpy(np.stack([record["labels"] for record in selected]))
            yield selected, full, prefix, suffix, candidates, labels
        group_start = group_end


@torch.inference_mode()
def evaluate(
    current: HSTU,
    old: HSTU,
    records: list[dict],
    args: argparse.Namespace,
) -> dict:
    device = torch.device(args.device)
    current.eval()
    old.eval()
    all_items = torch.arange(1, current.cfg.num_items + 1, device=device)
    suffix_depths = selected_suffix_depths(len(current.blocks))
    per_user = []
    timing = {"reuse": 0.0}
    cache_error = {"reuse": []}
    extra_numel = {"reuse": 0}
    cache_numel = 0
    exact_max_abs = 0.0
    parity_max_abs = 0.0
    for selected, full_cpu, prefix_cpu, suffix_cpu, candidates_cpu, labels_cpu in evaluation_batches(
        records,
        args.seq_len,
        args.batch_size,
    ):
        full = move_batch(full_cpu, device)
        prefix = move_batch(prefix_cpu, device)
        suffix = move_batch(suffix_cpu, device)
        candidates = candidates_cpu.to(device)
        labels = labels_cpu.to(device)
        full_hidden, _ = current(
            full["item_ids"],
            full["behaviors"],
            full["time_deltas"],
            lengths=full["lengths"],
        )
        fresh_hidden = current.last_hidden(full_hidden, full["lengths"])
        old_state = capture_layerwise_state(
            old,
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            prefix["lengths"],
        )
        fresh_cache = current.compute_kv(
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            lengths=prefix["lengths"],
        )
        fresh_incremental, _ = current.forward_with_cache(
            fresh_cache,
            suffix["item_ids"],
            suffix["behaviors"],
            suffix["time_deltas"],
        )
        parity_max_abs = max(
            parity_max_abs,
            float((fresh_incremental[:, 0] - fresh_hidden).abs().max().item()),
        )
        cache_numel += fresh_cache.k.numel() + fresh_cache.v.numel()
        caches = {"reuse": old_state.kv}
        for depth in suffix_depths:
            name = config_name(depth, len(current.blocks))
            fn = partial(
                migrate_suffix_cache,
                current,
                old_state,
                prefix["item_ids"],
                prefix["behaviors"],
                prefix["time_deltas"],
                depth,
            )
            migrated, elapsed = timed_call(fn, device, args.timing_repeats)
            caches[name] = migrated
            timing[name] = timing.get(name, 0.0) + elapsed
            extra_numel[name] = extra_numel.get(name, 0) + extra_state_numel(old_state, depth)
        recompute = caches["recompute"]
        exact_max_abs = max(
            exact_max_abs,
            float((recompute.k - fresh_cache.k).abs().max().item()),
            float((recompute.v - fresh_cache.v).abs().max().item()),
        )
        fresh_scores = current.item_emb.score(
            fresh_hidden,
            all_items.unsqueeze(0).expand(len(selected), -1),
        )
        fresh_candidate_scores = current.item_emb.score(fresh_hidden, candidates)
        config_metrics = {}
        candidate_config_metrics = {}
        for name, cache in caches.items():
            hidden, _ = current.forward_with_cache(
                cache,
                suffix["item_ids"],
                suffix["behaviors"],
                suffix["time_deltas"],
            )
            hidden = hidden[:, 0]
            scores = current.item_emb.score(
                hidden,
                all_items.unsqueeze(0).expand(len(selected), -1),
            )
            candidate_scores = current.item_emb.score(hidden, candidates)
            config_metrics[name] = [
                ranking_metrics(scores[row], selected[row]["positive_items"].tolist())
                for row in range(len(selected))
            ]
            candidate_config_metrics[name] = [
                candidate_metrics(candidate_scores[row], labels[row])
                for row in range(len(selected))
            ]
            cache_error.setdefault(name, []).extend(
                sample_relative_cache_error(cache, fresh_cache).cpu().tolist()
            )
        for row, record in enumerate(selected):
            per_user.append(
                {
                    "user_id": record["user_id"],
                    "history_length": int(full["lengths"][row].item()),
                    "fresh": ranking_metrics(
                        fresh_scores[row],
                        record["positive_items"].tolist(),
                    ),
                    "configs": {
                        name: values[row] for name, values in config_metrics.items()
                    },
                    "candidate_fresh": candidate_metrics(
                        fresh_candidate_scores[row],
                        labels[row],
                    ),
                    "candidate_configs": {
                        name: values[row]
                        for name, values in candidate_config_metrics.items()
                    },
                    "fresh_incremental_parity_max_abs": parity_max_abs,
                }
            )
    summary = summarize(per_user, timing, cache_error, extra_numel, cache_numel)
    summary["candidate_20"] = candidate_summary(per_user)
    summary["optimized_full_kv_max_abs"] = exact_max_abs
    summary["fresh_incremental_parity_max_abs"] = parity_max_abs
    return {"summary": summary, "per_user": per_user}


def save_checkpoint(model: HSTU, directory: str, version: int) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / f"theta_{version}.pt")


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    started = time.perf_counter()
    splits = {
        name: load_movielens_hard(args.data_dir, name)
        for name in ("train", "dev", "test")
    }
    maps = {
        name: {record["user_id"]: record for record in records}
        for name, records in splits.items()
    }
    common_users = sorted(set(maps["train"]) & set(maps["dev"]) & set(maps["test"]))
    rng = random.Random(args.seed)
    rng.shuffle(common_users)
    train_users = common_users[: args.max_train_users] if args.max_train_users else common_users
    eval_users = common_users[: args.max_eval_users]
    train_records = [maps["train"][user] for user in train_users]
    dev_update_records = [maps["dev"][user] for user in train_users]
    test_update_records = [maps["test"][user] for user in train_users]
    with Path(args.data_dir, "item_catalog.jsonl").open() as catalog_file:
        num_items = sum(1 for _ in catalog_file)
    model = make_model(args, num_items)
    old = make_model(args, num_items)
    for parameter in old.parameters():
        parameter.requires_grad_(False)
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-4)
    train_rng = random.Random(args.seed + 1000)
    base_losses = [
        train_epoch(model, train_records, base_optimizer, args, train_rng, False)
        for _ in range(args.base_epochs)
    ]
    old.load_state_dict(model.state_dict())
    theta0 = model_params_vec(model).detach().clone()
    save_checkpoint(model, args.checkpoint_dir, 0)
    result = {
        "protocol": "movielens_chronological_holdout_scaling_v1",
        "args": vars(args),
        "data": {
            "dataset": "movielens_1m_hard_v5_pilot20",
            "num_users": len(common_users),
            "train_users": len(train_users),
            "eval_users": len(eval_users),
            "num_items": num_items,
            "split_semantics": "same-user chronological train, dev, and test targets",
        },
        "operator": "fixed optimized deepest suffix with projection-only terminal layer",
        "base_losses": base_losses,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "versions": [],
    }
    stream_optimizer = torch.optim.AdamW(model.parameters(), lr=args.stream_lr, weight_decay=1e-4)
    for version, (update_records, eval_split) in enumerate(
        ((dev_update_records, "dev"), (test_update_records, "test")),
        start=1,
    ):
        losses = [
            train_epoch(model, update_records, stream_optimizer, args, train_rng, True)
            for _ in range(args.stream_epochs)
        ]
        save_checkpoint(model, args.checkpoint_dir, version)
        torch.set_float32_matmul_precision("highest")
        evaluation = evaluate(
            model,
            old,
            [maps[eval_split][user] for user in eval_users],
            args,
        )
        torch.set_float32_matmul_precision("high")
        current = model_params_vec(model).detach()
        evaluation.update(
            {
                "version": version,
                "stale_version": 0,
                "eval_split": eval_split,
                "stream_losses": losses,
                "dtheta_rel": float((current - theta0).norm() / theta0.norm()),
            }
        )
        result["versions"].append(evaluation)
        save_json(result, args.output)
        configs = evaluation["summary"]["configs"]
        print(
            f"version={version} split={eval_split} dtheta={evaluation['dtheta_rel']:.6f}",
            flush=True,
        )
        for name, value in configs.items():
            print(
                f"  {name:>12} cost={value['migration_ratio_to_recompute']:.3f} "
                f"rank_gain={value['gain_over_reuse']['best_rank']:.2f} "
                f"ndcg100_gain={value['gain_over_reuse']['ndcg@100']:.5f}",
                flush=True,
            )
    result["runtime_seconds"] = time.perf_counter() - started
    save_json(result, args.output)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
