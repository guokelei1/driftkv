from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from interval_oracle import pareto_configs
from layerwise_validity import (
    load_model,
    reconstruct_eval_samples,
    summarize,
    timed_call,
)
from motivation_validity import (
    build_streaming_plan,
    eval_batches,
    move_batch,
    ranking_metrics,
    seed_everything,
)

from hstu_kvcache.migration import (
    capture_layerwise_state,
    interval_extra_state_numel,
    migrate_contiguous_cache,
    migrate_current_norm_cache,
    migrate_embedding_delta_cache,
    sample_relative_cache_error,
)
from hstu_kvcache.models import HSTU
from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-result", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--model-t", type=int, default=5)
    parser.add_argument("--max-users", type=int, default=600)
    parser.add_argument("--probe-users", type=int, default=120)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--embedding-delta-scales", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def config_functions(
    current: HSTU,
    old_state,
    prefix: dict[str, torch.Tensor],
    scales: list[float],
) -> tuple[dict[str, object], dict[str, int]]:
    num_layers = len(current.blocks)
    values: dict[str, object] = {
        "cheap_oldnorm": partial(
            migrate_contiguous_cache,
            current,
            old_state,
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            None,
            None,
        ),
        "current_renorm": partial(migrate_current_norm_cache, current, old_state),
        "suffix_2": partial(
            migrate_contiguous_cache,
            current,
            old_state,
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            num_layers - 2,
            num_layers - 1,
        ),
        "suffix_4": partial(
            migrate_contiguous_cache,
            current,
            old_state,
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            num_layers - 4,
            num_layers - 1,
        ),
        "recompute": partial(
            migrate_contiguous_cache,
            current,
            old_state,
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            0,
            num_layers - 1,
        ),
    }
    for scale in scales:
        values[f"embedding_delta_{scale:g}"] = partial(
            migrate_embedding_delta_cache,
            current,
            old_state,
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            torch.full(
                (num_layers,),
                scale,
                device=prefix["item_ids"].device,
                dtype=old_state.hidden_states[0].dtype,
            ),
        )
    hidden_numel = sum(value.numel() for value in old_state.hidden_states)
    normed_numel = sum(value.numel() for value in old_state.normed_states)
    extra = {
        "cheap_oldnorm": normed_numel,
        "current_renorm": hidden_numel,
        "suffix_2": interval_extra_state_numel(
            old_state,
            num_layers - 2,
            num_layers - 1,
        ),
        "suffix_4": interval_extra_state_numel(
            old_state,
            num_layers - 4,
            num_layers - 1,
        ),
        "recompute": 0,
    }
    for scale in scales:
        extra[f"embedding_delta_{scale:g}"] = hidden_numel
    return values, extra


def add_frontiers(summary: dict) -> None:
    denominator = summary["configs"]["recompute"]["gain_over_reuse"]
    for value in summary["configs"].values():
        value["quality_recovery"] = {
            metric: gain / denominator[metric]
            if abs(denominator[metric]) > 1e-12
            else float("nan")
            for metric, gain in value["gain_over_reuse"].items()
        }
    summary["pareto_best_rank"] = pareto_configs(
        summary["configs"],
        ("best_rank",),
    )
    summary["pareto_joint"] = pareto_configs(
        summary["configs"],
        ("best_rank", "ndcg@100"),
    )


@torch.inference_mode()
def evaluate(
    current: HSTU,
    old: HSTU,
    samples: list[dict],
    metadata: dict,
    device: torch.device,
    timing_repeats: int,
    scales: list[float],
) -> dict:
    all_items = torch.arange(1, current.cfg.num_items + 1, device=device)
    records = []
    timing = {"reuse": 0.0}
    cache_error = {"reuse": []}
    extra_numel = {"reuse": 0}
    cache_numel = 0
    for selected, full_cpu, prefix_cpu, suffix_cpu in eval_batches(
        samples,
        metadata["seq_len"],
        metadata["batch_size"],
    ):
        full = move_batch(full_cpu, device)
        prefix = move_batch(prefix_cpu, device)
        suffix = move_batch(suffix_cpu, device)
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
        parity = (fresh_incremental[:, 0] - fresh_hidden).abs().amax(dim=1)
        candidate_ids = all_items.unsqueeze(0).expand(len(selected), -1)
        fresh_scores = current.item_emb.score(fresh_hidden, candidate_ids)
        cache_numel += fresh_cache.k.numel() + fresh_cache.v.numel()

        caches = {"reuse": old_state.kv}
        functions, batch_extra = config_functions(
            current,
            old_state,
            prefix,
            scales,
        )
        for name, fn in functions.items():
            cache, elapsed = timed_call(fn, device, timing_repeats)
            caches[name] = cache
            timing[name] = timing.get(name, 0.0) + elapsed
            extra_numel[name] = extra_numel.get(name, 0) + batch_extra[name]

        batch_metrics = {}
        for name, cache in caches.items():
            migrated_hidden, _ = current.forward_with_cache(
                cache,
                suffix["item_ids"],
                suffix["behaviors"],
                suffix["time_deltas"],
            )
            scores = current.item_emb.score(migrated_hidden[:, 0], candidate_ids)
            batch_metrics[name] = [
                ranking_metrics(scores[row], selected[row]["pos_items"])
                for row in range(len(selected))
            ]
            cache_error.setdefault(name, []).extend(
                sample_relative_cache_error(cache, fresh_cache).cpu().tolist()
            )

        for row, sample in enumerate(selected):
            records.append(
                {
                    "user_id": int(sample["history"]["user_id"]),
                    "history_length": int(full["lengths"][row].item()),
                    "fresh": ranking_metrics(
                        fresh_scores[row],
                        sample["pos_items"],
                    ),
                    "fresh_incremental_parity_max_abs": float(parity[row].item()),
                    "configs": {
                        name: batch_metrics[name][row]
                        for name in batch_metrics
                    },
                }
            )

    summary = summarize(records, timing, cache_error, extra_numel, cache_numel)
    add_frontiers(summary)
    return summary


def select_actions(probe: dict, test: dict) -> dict:
    configs = probe["configs"]
    full_gain = configs["recompute"]["gain_over_reuse"]["best_rank"]
    recovery_targets = (0.5, 0.75, 0.9)
    budget_targets = (0.1, 0.2, 0.4, 0.6, 0.8)
    output = {"recovery_targets": {}, "budget_targets": {}}
    for target in recovery_targets:
        eligible = [
            name
            for name, value in configs.items()
            if full_gain > 0
            and value["gain_over_reuse"]["best_rank"] >= target * full_gain
        ]
        selected = min(
            eligible,
            key=lambda name: configs[name]["migration_ratio_to_recompute"],
            default=None,
        )
        output["recovery_targets"][str(target)] = {
            "selected": selected,
            "probe": None if selected is None else configs[selected],
            "test": None if selected is None else test["configs"][selected],
        }
    for budget in budget_targets:
        eligible = [
            name
            for name, value in configs.items()
            if value["migration_ratio_to_recompute"] <= budget
        ]
        selected = max(
            eligible,
            key=lambda name: configs[name]["gain_over_reuse"]["best_rank"],
            default=None,
        )
        output["budget_targets"][str(budget)] = {
            "selected": selected,
            "probe": None if selected is None else configs[selected],
            "test": None if selected is None else test["configs"][selected],
        }
    return output


def main() -> None:
    args = parse_args()
    if args.probe_users <= 0 or args.probe_users >= args.max_users:
        raise ValueError("probe-users must be between zero and max-users")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(args.seed)
    source = json.loads(Path(args.run_result).read_text())
    metadata = source["args"]
    plan, _ = build_streaming_plan(metadata)
    plan.init_base()
    samples = reconstruct_eval_samples(
        plan,
        [args.model_t],
        metadata["stream_window_days"],
        args.max_users,
    )[args.model_t]
    rng = np.random.default_rng(7301 + args.seed)
    order = rng.permutation(len(samples))
    probe_indices = set(order[: args.probe_users].tolist())
    probe_samples = [
        sample for index, sample in enumerate(samples) if index in probe_indices
    ]
    test_samples = [
        sample for index, sample in enumerate(samples) if index not in probe_indices
    ]
    current = load_model(
        metadata,
        plan.num_items,
        plan.num_behaviors,
        args.device,
        args.checkpoint_dir,
        args.model_t,
    )
    old = load_model(
        metadata,
        plan.num_items,
        plan.num_behaviors,
        args.device,
        args.checkpoint_dir,
        0,
    )
    current_vec = model_params_vec(current).detach()
    old_vec = model_params_vec(old).detach()
    probe = evaluate(
        current,
        old,
        probe_samples,
        metadata,
        device,
        args.timing_repeats,
        args.embedding_delta_scales,
    )
    test = evaluate(
        current,
        old,
        test_samples,
        metadata,
        device,
        args.timing_repeats,
        args.embedding_delta_scales,
    )
    result = {
        "protocol": "migration_design_search_v1",
        "study_stage": "seed0_discovery",
        "seed": args.seed,
        "source_run": args.run_result,
        "checkpoint_dir": args.checkpoint_dir,
        "model_t": args.model_t,
        "stale_t": 0,
        "dtheta_rel": float((current_vec - old_vec).norm() / old_vec.norm()),
        "split": {
            "rule": "seeded user permutation independent of labels and outcomes",
            "probe_users": len(probe_samples),
            "test_users": len(test_samples),
            "split_seed": 7301 + args.seed,
        },
        "embedding_delta_scales": args.embedding_delta_scales,
        "probe": probe,
        "test": test,
        "selection": select_actions(probe, test),
    }
    save_json(result, args.output)
    print(f"probe_pareto={probe['pareto_joint']}")
    print(f"test_pareto={test['pareto_joint']}")
    for name in test["pareto_joint"]:
        value = test["configs"][name]
        print(
            f"{name:>22} cost={value['migration_ratio_to_recompute']:.3f} "
            f"rank={value['gain_over_reuse']['best_rank']:.2f} "
            f"ndcg100={value['gain_over_reuse']['ndcg@100']:.5f}"
        )
    print(args.output)


if __name__ == "__main__":
    main()
