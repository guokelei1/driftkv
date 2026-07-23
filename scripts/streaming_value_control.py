"""Validity control for frozen, full-reuse, and full-compute serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from layerwise_validity import load_model, reconstruct_eval_samples
from motivation_validity import (
    build_streaming_plan,
    eval_batches,
    move_batch,
    ranking_metrics,
    seed_everything,
)

from hstu_kvcache.models import HSTU
from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.utils import save_json

METRICS = (
    "mrr",
    "ndcg@10",
    "ndcg@100",
    "hit@10",
    "hit@100",
    "best_rank",
    "mean_rank",
    "rank_utility",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="streaming_value_control_v1_incremental_prefix_cache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-result")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--model-ts", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--max-eval-users", type=int)
    parser.add_argument("--output")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    if args.run_result is None:
        args.run_result = f"results/validity/core6l_seed{args.seed}.json"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints/validity/core6l_seed{args.seed}"
    if args.output is None:
        args.output = f"results/validity/streaming_control6l_seed{args.seed}.json"


def gain(first: float, second: float, metric: str) -> float:
    if metric in ("best_rank", "mean_rank"):
        return second - first
    return first - second


def summarize(records: list[dict]) -> dict:
    conditions = ("frozen", "full_reuse", "full_compute")
    condition_means = {
        condition: {
            metric: float(
                np.mean([record["conditions"][condition][metric] for record in records])
            )
            for metric in METRICS
        }
        for condition in conditions
    }
    comparisons = {
        "streaming_value_full_compute": ("full_compute", "frozen"),
        "streaming_value_full_reuse": ("full_reuse", "frozen"),
        "cache_maintenance_value": ("full_compute", "full_reuse"),
    }
    contrasts = {}
    for name, (first, second) in comparisons.items():
        contrasts[name] = {
            metric: float(
                np.mean(
                    [
                        gain(
                            record["conditions"][first][metric],
                            record["conditions"][second][metric],
                            metric,
                        )
                        for record in records
                    ]
                )
            )
            for metric in METRICS
        }
    return {
        "conditions": condition_means,
        "contrasts": contrasts,
        "current_incremental_parity_max_abs": max(
            record["current_incremental_parity_max_abs"] for record in records
        ),
        "frozen_incremental_parity_max_abs": max(
            record["frozen_incremental_parity_max_abs"] for record in records
        ),
    }


@torch.inference_mode()
def evaluate_pair(
    current: HSTU,
    frozen: HSTU,
    samples: list[dict],
    metadata: dict,
    device: torch.device,
) -> dict:
    all_items = torch.arange(1, current.cfg.num_items + 1, device=device)
    records = []
    for selected, full_cpu, prefix_cpu, suffix_cpu in eval_batches(
        samples,
        metadata["seq_len"],
        metadata["batch_size"],
    ):
        full = move_batch(full_cpu, device)
        prefix = move_batch(prefix_cpu, device)
        suffix = move_batch(suffix_cpu, device)

        current_hidden, _ = current(
            full["item_ids"],
            full["behaviors"],
            full["time_deltas"],
            lengths=full["lengths"],
        )
        current_last = current.last_hidden(current_hidden, full["lengths"])
        current_cache = current.compute_kv(
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            lengths=prefix["lengths"],
        )
        current_incremental, _ = current.forward_with_cache(
            current_cache,
            suffix["item_ids"],
            suffix["behaviors"],
            suffix["time_deltas"],
        )

        frozen_hidden, _ = frozen(
            full["item_ids"],
            full["behaviors"],
            full["time_deltas"],
            lengths=full["lengths"],
        )
        frozen_last = frozen.last_hidden(frozen_hidden, full["lengths"])
        frozen_cache = frozen.compute_kv(
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            lengths=prefix["lengths"],
        )
        frozen_incremental, _ = frozen.forward_with_cache(
            frozen_cache,
            suffix["item_ids"],
            suffix["behaviors"],
            suffix["time_deltas"],
        )
        reused_hidden, _ = current.forward_with_cache(
            frozen_cache,
            suffix["item_ids"],
            suffix["behaviors"],
            suffix["time_deltas"],
        )

        candidate_ids = all_items.unsqueeze(0).expand(len(selected), -1)
        scores = {
            "frozen": frozen.item_emb.score(frozen_last, candidate_ids),
            "full_reuse": current.item_emb.score(reused_hidden[:, 0], candidate_ids),
            "full_compute": current.item_emb.score(current_last, candidate_ids),
        }
        current_parity = (current_incremental[:, 0] - current_last).abs().amax(dim=1)
        frozen_parity = (frozen_incremental[:, 0] - frozen_last).abs().amax(dim=1)

        for row, sample in enumerate(selected):
            records.append(
                {
                    "user_id": int(sample["history"]["user_id"]),
                    "history_length": int(full["lengths"][row].item()),
                    "current_incremental_parity_max_abs": float(
                        current_parity[row].item()
                    ),
                    "frozen_incremental_parity_max_abs": float(
                        frozen_parity[row].item()
                    ),
                    "conditions": {
                        condition: ranking_metrics(
                            condition_scores[row],
                            sample["pos_items"],
                        )
                        for condition, condition_scores in scores.items()
                    },
                }
            )
    return {"summary": summarize(records), "per_user": records}


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(args.seed)
    metadata_result = json.loads(Path(args.run_result).read_text())
    metadata = metadata_result["args"]
    max_eval_users = args.max_eval_users or metadata["max_eval_users"]
    plan, _ = build_streaming_plan(metadata)
    plan.init_base()
    samples = reconstruct_eval_samples(
        plan,
        args.model_ts,
        metadata["stream_window_days"],
        max_eval_users,
    )
    result = {
        "protocol": args.protocol,
        "conditions": {
            "frozen": "theta-0 model with a consistent theta-0 cache and theta-0 scoring head",
            "full_reuse": "current model with a theta-0 prefix cache and current scoring head",
            "full_compute": "current model with a current prefix cache and current scoring head",
        },
        "seed": args.seed,
        "source_run": args.run_result,
        "checkpoint_dir": args.checkpoint_dir,
        "model_ts": args.model_ts,
        "max_eval_users": max_eval_users,
        "pairs": [],
    }

    frozen = load_model(
        metadata,
        plan.num_items,
        plan.num_behaviors,
        args.device,
        args.checkpoint_dir,
        0,
    )
    frozen_vec = model_params_vec(frozen).detach()
    for model_t in args.model_ts:
        current = load_model(
            metadata,
            plan.num_items,
            plan.num_behaviors,
            args.device,
            args.checkpoint_dir,
            model_t,
        )
        pair = evaluate_pair(
            current,
            frozen,
            samples[model_t],
            metadata,
            device,
        )
        current_vec = model_params_vec(current).detach()
        pair.update(
            {
                "model_t": model_t,
                "frozen_t": 0,
                "dtheta_rel": float(
                    (current_vec - frozen_vec).norm() / frozen_vec.norm()
                ),
                "eval_date": plan.stream_dates[
                    model_t * metadata["stream_window_days"]
                ],
                "n_users": len(pair["per_user"]),
            }
        )
        result["pairs"].append(pair)
        save_json(result, args.output)
        contrasts = pair["summary"]["contrasts"]
        print(
            f"theta=0->{model_t} users={pair['n_users']} "
            f"stream_full_rank={contrasts['streaming_value_full_compute']['best_rank']:.2f} "
            f"stream_reuse_rank={contrasts['streaming_value_full_reuse']['best_rank']:.2f} "
            f"cache_rank={contrasts['cache_maintenance_value']['best_rank']:.2f}",
            flush=True,
        )
        del current

    save_json(result, args.output)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
