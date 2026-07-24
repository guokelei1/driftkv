from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
    move_batch,
    prepare_sequence,
    ranking_metrics,
    seed_everything,
)

from hstu_kvcache.data import collate_batch
from hstu_kvcache.migration import (
    cache_fidelity_recovery,
    capture_layerwise_state,
    contiguous_intervals,
    interval_extra_state_numel,
    migrate_contiguous_cache,
    migrate_prefix_residual_cache,
    migrate_recent_suffix_cache,
    prefix_residual_extra_state_numel,
    recent_suffix_extra_state_numel,
    sample_relative_cache_error,
    select_maximum_fidelity_actions,
    select_minimum_cost_actions,
)
from hstu_kvcache.models import HSTU
from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-result", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--model-t", type=int, default=11)
    parser.add_argument("--max-users", type=int, default=300)
    parser.add_argument("--probe-users", type=int, default=60)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument(
        "--depth-fractions",
        type=float,
        nargs="+",
        default=[1 / 3, 2 / 3, 1.0],
    )
    parser.add_argument(
        "--recent-fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
    )
    parser.add_argument(
        "--fidelity-targets",
        type=float,
        nargs="+",
        default=[0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 0.9],
    )
    parser.add_argument(
        "--budget-targets",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8],
    )
    parser.add_argument("--all-intervals", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def split_samples(
    samples: list[dict],
    max_users: int,
    probe_users: int,
    seed: int,
) -> tuple[list[dict], list[dict], int]:
    usable = [
        sample
        for sample in samples[:max_users]
        if len(sample["history"]["item_ids"]) >= 2
    ]
    if not 0 < probe_users < len(usable):
        raise ValueError(
            f"probe-users must be between zero and {len(usable)}"
        )
    split_seed = 28411 + seed
    order = np.random.default_rng(split_seed).permutation(len(usable))
    probe_indices = set(order[:probe_users].tolist())
    probe = [
        sample for index, sample in enumerate(usable) if index in probe_indices
    ]
    test = [
        sample for index, sample in enumerate(usable) if index not in probe_indices
    ]
    return probe, test, split_seed


def length_bucketed_batches(
    samples: list[dict],
    seq_len: int,
    batch_size: int,
):
    buckets = defaultdict(list)
    for sample in samples:
        full_length = min(len(sample["history"]["item_ids"]), seq_len)
        buckets[full_length].append(sample)
    for full_length in sorted(buckets):
        bucket = buckets[full_length]
        for start in range(0, len(bucket), batch_size):
            selected = bucket[start : start + batch_size]
            full_sequences = [
                prepare_sequence(sample["history"], seq_len)
                for sample in selected
            ]
            prefix_sequences = [
                {name: values[:-1] for name, values in sequence.items()}
                for sequence in full_sequences
            ]
            full = collate_batch(
                full_sequences,
                max_seq_len=full_length,
                pad_to=full_length,
            )
            prefix = collate_batch(
                prefix_sequences,
                max_seq_len=full_length - 1,
                pad_to=full_length - 1,
            )
            suffix = {
                "item_ids": torch.tensor(
                    [
                        [sequence["item_ids"][-1]]
                        for sequence in full_sequences
                    ],
                    dtype=torch.long,
                ),
                "behaviors": torch.tensor(
                    [
                        [sequence["behaviors"][-1]]
                        for sequence in full_sequences
                    ],
                    dtype=torch.long,
                ),
                "time_deltas": torch.tensor(
                    [
                        [sequence["time_deltas"][-1]]
                        for sequence in full_sequences
                    ],
                    dtype=torch.float32,
                ),
            }
            yield selected, full, prefix, suffix


def fraction_tag(value: float) -> int:
    return int(round(100 * value))


def action_space(
    num_layers: int,
    depth_fractions: list[float],
    recent_fractions: list[float],
    include_all_intervals: bool,
) -> dict[str, dict]:
    if any(not 0 < value <= 1 for value in depth_fractions):
        raise ValueError("depth fractions must be in (0, 1]")
    if any(not 0 < value <= 1 for value in recent_fractions):
        raise ValueError("recent fractions must be in (0, 1]")
    depths = sorted(
        {
            max(1, min(num_layers, int(round(num_layers * fraction))))
            for fraction in depth_fractions
        }
    )
    recencies = sorted(set(recent_fractions))
    actions = {
        "cheap_all": {
            "kind": "recent_suffix",
            "top_n_full": 0,
            "depth_fraction": 0.0,
            "recent_fraction": 0.0,
        }
    }
    for depth in depths:
        for recent in recencies:
            if depth == num_layers and recent == 1.0:
                name = "recompute"
            else:
                name = f"replay_k{depth}_r{fraction_tag(recent)}"
            actions[name] = {
                "kind": "recent_suffix",
                "top_n_full": depth,
                "depth_fraction": depth / num_layers,
                "recent_fraction": recent,
            }
    if "recompute" not in actions:
        actions["recompute"] = {
            "kind": "recent_suffix",
            "top_n_full": num_layers,
            "depth_fraction": 1.0,
            "recent_fraction": 1.0,
        }
    if include_all_intervals:
        existing_suffix_depths = {
            value["top_n_full"]
            for value in actions.values()
            if value["kind"] == "recent_suffix"
            and value["recent_fraction"] == 1.0
        }
        for start, end in contiguous_intervals(num_layers):
            if end == num_layers - 1 and num_layers - start in existing_suffix_depths:
                continue
            name = f"interval_l{start + 1}_l{end + 1}"
            actions[name] = {
                "kind": "interval",
                "start_layer": start,
                "end_layer": end,
                "depth_fraction": (end - start + 1) / num_layers,
                "recent_fraction": 1.0,
            }
    return actions


def annotate_summary(summary: dict, actions: dict[str, dict]) -> None:
    configs = summary["configs"]
    reuse_error = configs["reuse"]["cache_error_rel"]
    full_error = configs["recompute"]["cache_error_rel"]
    full_gain = configs["recompute"]["gain_over_reuse"]
    configs["reuse"]["action"] = {
        "top_n_full": 0,
        "depth_fraction": 0.0,
        "recent_fraction": 0.0,
    }
    for name, value in configs.items():
        if name in actions:
            value["action"] = actions[name]
        value["cache_fidelity_recovery"] = cache_fidelity_recovery(
            value["cache_error_rel"],
            reuse_error,
            full_error,
        )
        value["quality_recovery"] = {
            metric: gain / full_gain[metric]
            if abs(full_gain[metric]) > 1e-12
            else float("nan")
            for metric, gain in value["gain_over_reuse"].items()
        }
    summary["pareto_best_rank"] = pareto_configs(
        configs,
        ("best_rank",),
    )
    summary["pareto_rank_utility"] = pareto_configs(
        configs,
        ("rank_utility",),
    )
    summary["pareto_ndcg100"] = pareto_configs(
        configs,
        ("ndcg@100",),
    )
    summary["pareto_joint"] = pareto_configs(
        configs,
        ("best_rank", "rank_utility", "ndcg@100"),
    )


@torch.inference_mode()
def evaluate(
    current: HSTU,
    old: HSTU,
    samples: list[dict],
    metadata: dict,
    device: torch.device,
    timing_repeats: int,
    actions: dict[str, dict],
) -> dict:
    all_items = torch.arange(1, current.cfg.num_items + 1, device=device)
    records = []
    timing = {"reuse": 0.0}
    cache_error = {"reuse": []}
    extra_numel = {"reuse": 0}
    cache_numel = 0
    replay_tokens = defaultdict(list)
    for selected, full_cpu, prefix_cpu, suffix_cpu in length_bucketed_batches(
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
        prefix_len = prefix["item_ids"].shape[1]
        for name, action in actions.items():
            if action["kind"] == "interval":
                tokens = prefix_len
                fn = partial(
                    migrate_contiguous_cache,
                    current,
                    old_state,
                    prefix["item_ids"],
                    prefix["behaviors"],
                    prefix["time_deltas"],
                    action["start_layer"],
                    action["end_layer"],
                )
                state_numel = interval_extra_state_numel(
                    old_state,
                    action["start_layer"],
                    action["end_layer"],
                )
            elif action["kind"] == "prefix_residual":
                tokens = prefix_len
                fn = partial(
                    migrate_prefix_residual_cache,
                    current,
                    old_state,
                    prefix["item_ids"],
                    prefix["behaviors"],
                    prefix["time_deltas"],
                    action["prefix_depth"],
                )
                state_numel = prefix_residual_extra_state_numel(
                    old_state,
                    action["prefix_depth"],
                )
            else:
                tokens = min(
                    prefix_len,
                    int(round(prefix_len * action["recent_fraction"])),
                )
                fn = partial(
                    migrate_recent_suffix_cache,
                    current,
                    old_state,
                    prefix["item_ids"],
                    prefix["behaviors"],
                    prefix["time_deltas"],
                    action["top_n_full"],
                    tokens,
                )
                state_numel = recent_suffix_extra_state_numel(
                    old_state,
                    action["top_n_full"],
                    tokens,
                )
            cache, elapsed = timed_call(fn, device, timing_repeats)
            caches[name] = cache
            timing[name] = timing.get(name, 0.0) + elapsed
            extra_numel[name] = extra_numel.get(name, 0) + state_numel
            replay_tokens[name].extend([tokens] * len(selected))

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
                candidate_ids,
            )
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
                    "fresh_incremental_parity_max_abs": float(
                        parity[row].item()
                    ),
                    "configs": {
                        name: batch_metrics[name][row]
                        for name in batch_metrics
                    },
                }
            )

    summary = summarize(
        records,
        timing,
        cache_error,
        extra_numel,
        cache_numel,
    )
    annotate_summary(summary, actions)
    for name, values in replay_tokens.items():
        summary["configs"][name]["replay_tokens"] = {
            "min": min(values),
            "median": float(np.median(values)),
            "max": max(values),
        }
    return {
        "summary": summary,
        "per_user": records,
    }


def select_actions(
    probe: dict,
    test: dict,
    fidelity_targets: tuple[float, ...],
    budget_targets: tuple[float, ...],
) -> dict:
    probe_configs = probe["summary"]["configs"]
    test_configs = test["summary"]["configs"]
    fidelity = select_minimum_cost_actions(
        probe_configs,
        fidelity_targets,
    )
    budget = select_maximum_fidelity_actions(
        probe_configs,
        budget_targets,
    )
    for selection in (fidelity, budget):
        for value in selection.values():
            name = value["selected"]
            value["probe"] = probe_configs[name]
            value["test"] = test_configs[name]
    return {
        "fidelity_targets": fidelity,
        "budget_targets": budget,
    }


def main() -> None:
    args = parse_args()
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
    probe_samples, test_samples, split_seed = split_samples(
        samples,
        args.max_users,
        args.probe_users,
        args.seed,
    )
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
    actions = action_space(
        len(current.blocks),
        args.depth_fractions,
        args.recent_fractions,
        args.all_intervals,
    )
    probe = evaluate(
        current,
        old,
        probe_samples,
        metadata,
        device,
        args.timing_repeats,
        actions,
    )
    test = evaluate(
        current,
        old,
        test_samples,
        metadata,
        device,
        args.timing_repeats,
        actions,
    )
    result = {
        "protocol": "structural_replay_discovery_v1",
        "study_stage": "motivation_selected_seed_discovery",
        "seed": args.seed,
        "source_run": args.run_result,
        "checkpoint_dir": args.checkpoint_dir,
        "model_t": args.model_t,
        "stale_t": 0,
        "dtheta_rel": float((current_vec - old_vec).norm() / old_vec.norm()),
        "operator": {
            "name": "two_dimensional_recent_suffix_replay",
            "base": "current Wk/Wv over cached old Norm(x)",
            "replay": (
                "current native HSTU blocks over a recent-token by "
                "deep-suffix rectangle"
            ),
            "terminal": "projection-only final layer",
            "batching": "prefixes bucketed by effective length",
            "selection_scope": "one action per old/current model-version cohort",
            "learned_parameters": 0,
            "all_intervals": args.all_intervals,
        },
        "action_space": actions,
        "split": {
            "rule": "seeded user permutation independent of labels and outcomes",
            "probe_users": len(probe_samples),
            "test_users": len(test_samples),
            "split_seed": split_seed,
        },
        "selection_signal": (
            "relative K/V cache fidelity on probe users; task labels are "
            "used only for offline evaluation"
        ),
        "probe": probe,
        "test": test,
        "selection": select_actions(
            probe,
            test,
            tuple(args.fidelity_targets),
            tuple(args.budget_targets),
        ),
    }
    save_json(result, args.output)
    print(f"probe_pareto={probe['summary']['pareto_joint']}")
    print(f"test_pareto={test['summary']['pareto_joint']}")
    for target, value in result["selection"]["fidelity_targets"].items():
        selected = value["selected"]
        test_value = value["test"]
        print(
            f"fidelity={target} selected={selected} "
            f"cost={test_value['migration_ratio_to_recompute']:.3f} "
            f"cache={test_value['cache_fidelity_recovery']:.3f} "
            f"rank={test_value['gain_over_reuse']['best_rank']:.2f} "
            f"utility={test_value['gain_over_reuse']['rank_utility']:.5f} "
            f"ndcg100={test_value['gain_over_reuse']['ndcg@100']:.5f}"
        )
    print(args.output)


if __name__ == "__main__":
    main()
