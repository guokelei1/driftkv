"""One-seed oracle over optimized contiguous layer migration intervals."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from layerwise_validity import (
    layer_relative_errors,
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
    contiguous_intervals,
    interval_extra_state_numel,
    migrate_contiguous_cache,
    migrate_legacy_suffix_cache,
    sample_relative_cache_error,
)
from hstu_kvcache.models import HSTU
from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-result")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--model-ts", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--max-eval-users", type=int)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--config-names", nargs="+")
    parser.add_argument("--skip-legacy-comparison", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    if args.run_result is None:
        args.run_result = f"results/validity/core6l_seed{args.seed}.json"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints/validity/core6l_seed{args.seed}"
    if args.output is None:
        args.output = f"results/validity/interval_oracle_seed{args.seed}.json"


def interval_name(start: int, end: int, num_layers: int) -> str:
    if start == 0 and end == num_layers - 1:
        return "recompute"
    return f"interval_l{start + 1}_l{end + 1}"


def interval_configs(num_layers: int) -> dict[str, tuple[int | None, int | None]]:
    configs: dict[str, tuple[int | None, int | None]] = {
        "cheap_all": (None, None)
    }
    for start, end in contiguous_intervals(num_layers):
        configs[interval_name(start, end, num_layers)] = (start, end)
    return configs


def suffix_config_name(top_n: int, num_layers: int) -> str:
    if top_n == 0:
        return "cheap_all"
    return interval_name(num_layers - top_n, num_layers - 1, num_layers)


def pareto_configs(
    configs: dict,
    quality_metrics: tuple[str, ...],
) -> list[str]:
    names = list(configs)
    frontier = []
    for name in names:
        cost = configs[name]["migration_ratio_to_recompute"]
        quality = [configs[name]["gain_over_reuse"][metric] for metric in quality_metrics]
        dominated = False
        for other in names:
            if other == name:
                continue
            other_cost = configs[other]["migration_ratio_to_recompute"]
            other_quality = [
                configs[other]["gain_over_reuse"][metric]
                for metric in quality_metrics
            ]
            weakly_better = other_cost <= cost and all(
                candidate >= current
                for candidate, current in zip(other_quality, quality, strict=True)
            )
            strictly_better = other_cost < cost or any(
                candidate > current
                for candidate, current in zip(other_quality, quality, strict=True)
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(name)
    return sorted(
        frontier,
        key=lambda name: configs[name]["migration_ratio_to_recompute"],
    )


def annotate_summary(
    summary: dict,
    configs: dict[str, tuple[int | None, int | None]],
) -> None:
    full = summary["configs"]["recompute"]["gain_over_reuse"]
    for name, value in summary["configs"].items():
        interval = configs.get(name)
        value["interval_zero_based_inclusive"] = (
            None if interval is None or interval[0] is None else list(interval)
        )
        value["quality_recovery"] = {
            metric: value["gain_over_reuse"][metric] / denominator
            if abs(denominator) > 1e-12
            else float("nan")
            for metric, denominator in full.items()
        }
    summary["pareto_best_rank"] = pareto_configs(summary["configs"], ("best_rank",))
    summary["pareto_ndcg100"] = pareto_configs(summary["configs"], ("ndcg@100",))
    summary["pareto_joint"] = pareto_configs(
        summary["configs"],
        ("best_rank", "ndcg@100"),
    )


@torch.inference_mode()
def evaluate_pair(
    current: HSTU,
    old: HSTU,
    samples: list[dict],
    metadata: dict,
    device: torch.device,
    timing_repeats: int,
    config_names: list[str] | None,
    compare_legacy: bool,
) -> dict:
    num_layers = len(current.blocks)
    all_configs = interval_configs(num_layers)
    if config_names is None:
        configs = all_configs
    else:
        wanted = set(config_names) | {"cheap_all", "recompute"}
        unknown = wanted - set(all_configs)
        if unknown:
            raise ValueError(f"unknown configs: {sorted(unknown)}")
        configs = {
            name: interval
            for name, interval in all_configs.items()
            if name in wanted
        }
    all_items = torch.arange(1, current.cfg.num_items + 1, device=device)
    records = []
    timing = {"reuse": 0.0}
    legacy_timing = {top_n: 0.0 for top_n in range(num_layers + 1)}
    equivalence = {top_n: 0.0 for top_n in range(num_layers + 1)}
    cache_error = {"reuse": []}
    extra_numel = {"reuse": 0}
    cache_numel = 0
    layer_errors = []

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
        fresh_scores = current.item_emb.score(
            fresh_hidden,
            all_items.unsqueeze(0).expand(len(selected), -1),
        )
        layer_errors.append(layer_relative_errors(old_state.kv, fresh_cache).cpu().numpy())
        cache_numel += fresh_cache.k.numel() + fresh_cache.v.numel()

        caches = {"reuse": old_state.kv}
        for name, (start, end) in configs.items():
            fn = partial(
                migrate_contiguous_cache,
                current,
                old_state,
                prefix["item_ids"],
                prefix["behaviors"],
                prefix["time_deltas"],
                start,
                end,
            )
            cache, elapsed = timed_call(fn, device, timing_repeats)
            caches[name] = cache
            timing[name] = timing.get(name, 0.0) + elapsed
            extra_numel[name] = extra_numel.get(name, 0) + interval_extra_state_numel(
                old_state,
                start,
                end,
            )

        if compare_legacy:
            for top_n in range(num_layers + 1):
                legacy_fn = partial(
                    migrate_legacy_suffix_cache,
                    current,
                    old_state,
                    prefix["item_ids"],
                    prefix["behaviors"],
                    prefix["time_deltas"],
                    top_n,
                )
                legacy, elapsed = timed_call(legacy_fn, device, timing_repeats)
                legacy_timing[top_n] += elapsed
                optimized_name = suffix_config_name(top_n, num_layers)
                if optimized_name not in caches:
                    continue
                optimized = caches[optimized_name]
                max_abs = max(
                    float((legacy.k - optimized.k).abs().max().item()),
                    float((legacy.v - optimized.v).abs().max().item()),
                )
                equivalence[top_n] = max(equivalence[top_n], max_abs)

        batch_metrics = {}
        for name, cache in caches.items():
            migrated_hidden, _ = current.forward_with_cache(
                cache,
                suffix["item_ids"],
                suffix["behaviors"],
                suffix["time_deltas"],
            )
            scores = current.item_emb.score(
                migrated_hidden[:, 0],
                all_items.unsqueeze(0).expand(len(selected), -1),
            )
            batch_metrics[name] = [
                ranking_metrics(scores[row], selected[row]["pos_items"])
                for row in range(len(selected))
            ]
            errors = sample_relative_cache_error(cache, fresh_cache).cpu().tolist()
            cache_error.setdefault(name, []).extend(errors)

        for row, sample in enumerate(selected):
            records.append(
                {
                    "user_id": int(sample["history"]["user_id"]),
                    "history_length": int(full["lengths"][row].item()),
                    "fresh": ranking_metrics(fresh_scores[row], sample["pos_items"]),
                    "fresh_incremental_parity_max_abs": float(parity[row].item()),
                    "configs": {
                        name: batch_metrics[name][row]
                        for name in batch_metrics
                    },
                }
            )

    layer_error_array = np.concatenate(layer_errors, axis=1)
    summary = summarize(records, timing, cache_error, extra_numel, cache_numel)
    summary["per_layer_stale_cache_error_rel"] = layer_error_array.mean(axis=1).tolist()
    summary["per_layer_error_order"] = np.argsort(
        -layer_error_array.mean(axis=1)
    ).tolist()
    annotate_summary(summary, configs)

    summary["legacy_suffix_comparison"] = {}
    if compare_legacy:
        optimized_full = timing["recompute"]
        legacy_full = legacy_timing[num_layers]
        summary["legacy_suffix_comparison"] = {
            f"suffix_{top_n}": {
                "optimized_config": suffix_config_name(top_n, num_layers),
                "legacy_ms_per_user": legacy_timing[top_n] / len(records),
                "optimized_ms_per_user": timing[optimized_name] / len(records),
                "optimized_over_legacy": timing[optimized_name]
                / max(legacy_timing[top_n], 1e-12),
                "legacy_ratio_to_legacy_recompute": legacy_timing[top_n]
                / max(legacy_full, 1e-12),
                "optimized_ratio_to_optimized_recompute": timing[optimized_name]
                / max(optimized_full, 1e-12),
                "optimized_ratio_to_legacy_recompute": timing[optimized_name]
                / max(legacy_full, 1e-12),
                "kv_max_abs": equivalence[top_n],
            }
            for top_n in range(num_layers + 1)
            if (optimized_name := suffix_config_name(top_n, num_layers)) in timing
        }
    return {"summary": summary, "per_user": records}


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
        "protocol": "interval_oracle_v1_terminal_projection",
        "operator": {
            "cheap": "cached old Norm(x) projected by current Wk/Wv",
            "interval": "current full blocks from start through end-1, then current Norm plus Wk/Wv at the terminal layer",
            "outside_interval": "cheap projection refresh",
            "selection_scope": "one global interval per model-version pair",
            "layer_indexing": "result names are one-based and inclusive",
        },
        "seed": args.seed,
        "source_run": args.run_result,
        "checkpoint_dir": args.checkpoint_dir,
        "model_ts": args.model_ts,
        "stale_mode": "cumulative_theta0",
        "max_eval_users": max_eval_users,
        "timing_repeats": args.timing_repeats,
        "study_stage": "discovery" if args.config_names is None else "heldout_validation",
        "requested_config_names": args.config_names,
        "legacy_comparison": not args.skip_legacy_comparison,
        "pairs": [],
    }

    for model_t in args.model_ts:
        current = load_model(
            metadata,
            plan.num_items,
            plan.num_behaviors,
            args.device,
            args.checkpoint_dir,
            model_t,
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
        pair = evaluate_pair(
            current,
            old,
            samples[model_t],
            metadata,
            device,
            args.timing_repeats,
            args.config_names,
            not args.skip_legacy_comparison,
        )
        pair.update(
            {
                "model_t": model_t,
                "stale_t": 0,
                "dtheta_rel": float((current_vec - old_vec).norm() / old_vec.norm()),
                "eval_date": plan.stream_dates[
                    model_t * metadata["stream_window_days"]
                ],
                "n_users": len(pair["per_user"]),
            }
        )
        result["pairs"].append(pair)
        save_json(result, args.output)
        summary = pair["summary"]
        print(
            f"theta=0->{model_t} users={pair['n_users']} "
            f"joint_pareto={summary['pareto_joint']}",
            flush=True,
        )
        for name in summary["pareto_joint"]:
            value = summary["configs"][name]
            print(
                f"  {name:>18} time={value['migration_ratio_to_recompute']:.3f} "
                f"rank_gain={value['gain_over_reuse']['best_rank']:.2f} "
                f"ndcg100_gain={value['gain_over_reuse']['ndcg@100']:.5f}",
                flush=True,
            )
        del old
        del current

    save_json(result, args.output)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
