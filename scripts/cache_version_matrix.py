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
from streaming_value_control import METRICS, gain

from hstu_kvcache.models import HSTU
from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="cache_version_matrix_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-result", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--current-t", type=int, required=True)
    parser.add_argument("--stale-ts", type=int, nargs="+")
    parser.add_argument("--max-eval-users", type=int)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def summarize(records: list[dict]) -> dict:
    conditions = ("frozen", "reuse", "full_compute")
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
        "streaming_value_reuse": ("reuse", "frozen"),
        "cache_maintenance_value": ("full_compute", "reuse"),
    }
    contrasts = {
        name: {
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
        for name, (first, second) in comparisons.items()
    }
    return {"conditions": condition_means, "contrasts": contrasts}


@torch.inference_mode()
def evaluate(
    current: HSTU,
    frozen: HSTU,
    stale: HSTU,
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
        frozen_hidden, _ = frozen(
            full["item_ids"],
            full["behaviors"],
            full["time_deltas"],
            lengths=full["lengths"],
        )
        frozen_last = frozen.last_hidden(frozen_hidden, full["lengths"])
        stale_cache = stale.compute_kv(
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            lengths=prefix["lengths"],
        )
        reused_hidden, _ = current.forward_with_cache(
            stale_cache,
            suffix["item_ids"],
            suffix["behaviors"],
            suffix["time_deltas"],
        )
        candidate_ids = all_items.unsqueeze(0).expand(len(selected), -1)
        scores = {
            "frozen": frozen.item_emb.score(frozen_last, candidate_ids),
            "reuse": current.item_emb.score(reused_hidden[:, 0], candidate_ids),
            "full_compute": current.item_emb.score(current_last, candidate_ids),
        }
        for row, sample in enumerate(selected):
            records.append(
                {
                    "user_id": int(sample["history"]["user_id"]),
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
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(args.seed)
    source = json.loads(Path(args.run_result).read_text())
    metadata = source["args"]
    max_eval_users = args.max_eval_users or metadata["max_eval_users"]
    stale_ts = args.stale_ts or list(range(args.current_t - 1, -1, -1))
    if any(stale_t < 0 or stale_t >= args.current_t for stale_t in stale_ts):
        raise ValueError("stale versions must satisfy 0 <= stale_t < current_t")
    stale_ts = sorted(set(stale_ts), reverse=True)
    plan, _ = build_streaming_plan(metadata)
    plan.init_base()
    samples = reconstruct_eval_samples(
        plan,
        [args.current_t],
        metadata["stream_window_days"],
        max_eval_users,
    )[args.current_t]
    current = load_model(
        metadata,
        plan.num_items,
        plan.num_behaviors,
        args.device,
        args.checkpoint_dir,
        args.current_t,
    )
    frozen = load_model(
        metadata,
        plan.num_items,
        plan.num_behaviors,
        args.device,
        args.checkpoint_dir,
        0,
    )
    current_vec = model_params_vec(current).detach()
    result = {
        "protocol": args.protocol,
        "seed": args.seed,
        "source_run": args.run_result,
        "checkpoint_dir": args.checkpoint_dir,
        "current_t": args.current_t,
        "eval_date": plan.stream_dates[
            args.current_t * metadata["stream_window_days"]
        ],
        "n_users": len(samples),
        "points": [],
    }
    for stale_t in stale_ts:
        stale = load_model(
            metadata,
            plan.num_items,
            plan.num_behaviors,
            args.device,
            args.checkpoint_dir,
            stale_t,
        )
        point = evaluate(current, frozen, stale, samples, metadata, device)
        stale_vec = model_params_vec(stale).detach()
        point.update(
            {
                "stale_t": stale_t,
                "cache_age": args.current_t - stale_t,
                "dtheta_rel": float(
                    (current_vec - stale_vec).norm() / stale_vec.norm()
                ),
                "n_users": len(point["per_user"]),
            }
        )
        result["points"].append(point)
        save_json(result, args.output)
        contrasts = point["summary"]["contrasts"]
        stream = contrasts["streaming_value_full_compute"]["best_rank"]
        cache = contrasts["cache_maintenance_value"]["best_rank"]
        tax = cache / stream if stream > 0 else float("nan")
        print(
            f"current={args.current_t} stale={stale_t} age={args.current_t - stale_t} "
            f"dtheta={point['dtheta_rel']:.5f} cache_rank={cache:.2f} tax={tax:.3f}",
            flush=True,
        )
        del stale
    save_json(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
