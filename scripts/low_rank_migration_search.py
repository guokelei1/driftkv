from __future__ import annotations

import argparse
import json
import time
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
    compile_low_rank_cache_adapter,
    compile_projection_cache_adapter,
    fit_low_rank_cache_adapter,
    interval_extra_state_numel,
    migrate_compiled_low_rank_cache,
    migrate_contiguous_cache,
    migrate_fused_projection_cache,
    migrate_prefix_residual_cache,
    prefix_residual_extra_state_numel,
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
    parser.add_argument("--fit-users", type=int, default=80)
    parser.add_argument("--probe-users", type=int, default=120)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64, 96],
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--protocol",
        default="compiled_low_rank_migration_v1",
    )
    parser.add_argument("--study-stage")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def split_samples(
    samples: list[dict],
    fit_users: int,
    probe_users: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = np.random.default_rng(9151 + seed)
    order = rng.permutation(len(samples))
    fit_indices = set(order[:fit_users].tolist())
    probe_indices = set(order[fit_users : fit_users + probe_users].tolist())
    fit = [
        sample
        for index, sample in enumerate(samples)
        if index in fit_indices
    ]
    probe = [
        sample
        for index, sample in enumerate(samples)
        if index in probe_indices
    ]
    test = [
        sample
        for index, sample in enumerate(samples)
        if index not in fit_indices and index not in probe_indices
    ]
    return fit, probe, test


@torch.inference_mode()
def fit_adapter(
    current: HSTU,
    old: HSTU,
    samples: list[dict],
    metadata: dict,
    device: torch.device,
    max_rank: int,
    ridge: float,
):
    feature_chunks = [[] for _ in current.blocks]
    residual_chunks = [[] for _ in current.blocks]
    valid_tokens = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _, _, prefix_cpu, _ in eval_batches(
        samples,
        metadata["seq_len"],
        metadata["batch_size"],
    ):
        prefix = move_batch(prefix_cpu, device)
        old_state = capture_layerwise_state(
            old,
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            prefix["lengths"],
        )
        fresh = current.compute_kv(
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            lengths=prefix["lengths"],
        )
        cheap = migrate_fused_projection_cache(current, old_state)
        positions = torch.arange(prefix["item_ids"].shape[1], device=device)
        valid = positions.unsqueeze(0) < prefix["lengths"].unsqueeze(1)
        valid_tokens += int(valid.sum().item())
        for layer, features in enumerate(old_state.normed_states):
            residual = torch.cat(
                (
                    fresh.k[layer] - cheap.k[layer],
                    fresh.v[layer] - cheap.v[layer],
                ),
                dim=-1,
            )
            feature_chunks[layer].append(features[valid])
            residual_chunks[layer].append(residual[valid])
    feature_layers = [torch.cat(chunks, dim=0) for chunks in feature_chunks]
    residual_layers = [torch.cat(chunks, dim=0) for chunks in residual_chunks]
    adapter = fit_low_rank_cache_adapter(
        feature_layers,
        residual_layers,
        rank=max_rank,
        ridge=ridge,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return adapter, {
        "fit_users": len(samples),
        "valid_prefix_tokens": valid_tokens,
        "elapsed_ms": elapsed_ms,
        "max_rank": max_rank,
        "shared_parameter_numel": adapter.numel,
        "shared_fp16_bytes": 2 * adapter.numel,
        "ridge": ridge,
    }


def config_functions(
    current: HSTU,
    old_state,
    prefix: dict[str, torch.Tensor],
    compiled_cheap: object,
    adapters: dict[int, tuple[object, object]],
    include_prefix_ladder: bool = False,
    include_residual_ladder: bool = False,
) -> tuple[dict[str, object], dict[str, int], dict[str, int]]:
    num_layers = len(current.blocks)
    suffix_2_depth = min(2, num_layers)
    suffix_4_depth = min(4, num_layers)
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
        "cheap_fused": partial(
            migrate_fused_projection_cache,
            current,
            old_state,
        ),
        "cheap_prepacked": partial(
            migrate_compiled_low_rank_cache,
            old_state,
            compiled_cheap,
        ),
        "suffix_2": partial(
            migrate_contiguous_cache,
            current,
            old_state,
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            num_layers - suffix_2_depth,
            num_layers - 1,
        ),
        "suffix_4": partial(
            migrate_contiguous_cache,
            current,
            old_state,
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            num_layers - suffix_4_depth,
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
    normed_numel = sum(value.numel() for value in old_state.normed_states)
    extra = {
        "cheap_oldnorm": normed_numel,
        "cheap_fused": normed_numel,
        "cheap_prepacked": normed_numel,
        "suffix_2": interval_extra_state_numel(
            old_state,
            num_layers - suffix_2_depth,
            num_layers - 1,
        ),
        "suffix_4": interval_extra_state_numel(
            old_state,
            num_layers - suffix_4_depth,
            num_layers - 1,
        ),
        "recompute": 0,
    }
    shared = {
        "cheap_oldnorm": 0,
        "cheap_fused": 0,
        "cheap_prepacked": 0,
        "suffix_2": 0,
        "suffix_4": 0,
        "recompute": 0,
    }
    if include_prefix_ladder:
        for depth in range(1, num_layers):
            name = f"prefix_p{depth}"
            values[name] = partial(
                migrate_contiguous_cache,
                current,
                old_state,
                prefix["item_ids"],
                prefix["behaviors"],
                prefix["time_deltas"],
                0,
                depth - 1,
            )
            extra[name] = interval_extra_state_numel(
                old_state,
                0,
                depth - 1,
            )
            shared[name] = 0
    if include_residual_ladder:
        for depth in range(1, num_layers):
            name = f"residual_p{depth}"
            values[name] = partial(
                migrate_prefix_residual_cache,
                current,
                old_state,
                prefix["item_ids"],
                prefix["behaviors"],
                prefix["time_deltas"],
                depth,
            )
            extra[name] = prefix_residual_extra_state_numel(
                old_state,
                depth,
            )
            shared[name] = 0
    for rank, (adapter, compiled) in adapters.items():
        name = f"adapter_rank_{rank}"
        values[name] = partial(
            migrate_compiled_low_rank_cache,
            old_state,
            compiled,
        )
        extra[name] = normed_numel
        shared[name] = adapter.numel
    return values, extra, shared


def add_frontiers(summary: dict, shared_numel: dict[str, int]) -> None:
    reuse_error = summary["configs"]["reuse"]["cache_error_rel"]
    for name, value in summary["configs"].items():
        value["cache_fidelity_recovery"] = (
            (reuse_error - value["cache_error_rel"]) / max(reuse_error, 1e-12)
        )
        value["shared_adapter_numel"] = shared_numel.get(name, 0)
        value["shared_adapter_fp16_bytes"] = 2 * shared_numel.get(name, 0)
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
    compiled_cheap: object,
    adapters: dict[int, tuple[object, object]],
    include_prefix_ladder: bool = False,
    include_residual_ladder: bool = False,
) -> dict:
    all_items = torch.arange(1, current.cfg.num_items + 1, device=device)
    records = []
    timing = {"reuse": 0.0}
    cache_error = {"reuse": []}
    extra_numel = {"reuse": 0}
    shared_numel = {"reuse": 0}
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
        functions, batch_extra, batch_shared = config_functions(
            current,
            old_state,
            prefix,
            compiled_cheap,
            adapters,
            include_prefix_ladder,
            include_residual_ladder,
        )
        for name, fn in functions.items():
            cache, elapsed = timed_call(fn, device, timing_repeats)
            caches[name] = cache
            timing[name] = timing.get(name, 0.0) + elapsed
            extra_numel[name] = extra_numel.get(name, 0) + batch_extra[name]
            shared_numel[name] = batch_shared[name]

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

    summary = summarize(
        records,
        timing,
        cache_error,
        extra_numel,
        cache_numel,
    )
    add_frontiers(summary, shared_numel)
    return summary


def select_actions(probe: dict, test: dict) -> dict:
    configs = probe["configs"]
    recovery_targets = (0.1, 0.25, 0.5, 0.75, 0.9)
    budget_targets = (0.2, 0.3, 0.4, 0.6, 0.8)
    adapter_names = sorted(
        (
            name
            for name in configs
            if name.startswith("adapter_rank_")
        ),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    fidelity_ladder = ["cheap_prepacked", *adapter_names, "recompute"]
    output = {
        "signal": "probe cache fidelity with a frozen target or measured GPU budget",
        "fidelity_ladder": fidelity_ladder,
        "recovery_targets": {},
        "budget_targets": {},
    }
    for target in recovery_targets:
        eligible = [
            name
            for name in fidelity_ladder
            if configs[name]["cache_fidelity_recovery"] >= target
        ]
        selected = eligible[0] if eligible else None
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
        selected = min(
            eligible,
            key=lambda name: configs[name]["cache_error_rel"],
            default=None,
        )
        output["budget_targets"][str(budget)] = {
            "selected": selected,
            "probe": None if selected is None else configs[selected],
            "test": None if selected is None else test["configs"][selected],
        }
    output["recommended"] = {
        "rule": "smallest calibrated rank closing at least half of the probe cache gap",
        **output["recovery_targets"]["0.5"],
    }
    return output


def main() -> None:
    args = parse_args()
    if args.fit_users <= 0 or args.probe_users <= 0:
        raise ValueError("fit-users and probe-users must be positive")
    if args.fit_users + args.probe_users >= args.max_users:
        raise ValueError("fit-users plus probe-users must be below max-users")
    if any(rank <= 0 for rank in args.ranks):
        raise ValueError("ranks must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(args.seed)
    source = json.loads(Path(args.run_result).read_text())
    metadata = source["args"]
    max_supported_rank = min(
        metadata["hidden_size"],
        2 * metadata["num_heads"] * metadata["head_dim"],
    )
    ranks = sorted(set(rank for rank in args.ranks if rank <= max_supported_rank))
    if not ranks:
        raise ValueError(f"no rank is at most {max_supported_rank}")
    plan, _ = build_streaming_plan(metadata)
    plan.init_base()
    samples = reconstruct_eval_samples(
        plan,
        [args.model_t],
        metadata["stream_window_days"],
        args.max_users,
    )[args.model_t]
    fit_samples, probe_samples, test_samples = split_samples(
        samples,
        args.fit_users,
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
    basis, fit_summary = fit_adapter(
        current,
        old,
        fit_samples,
        metadata,
        device,
        max(ranks),
        args.ridge,
    )
    compiled_cheap = compile_projection_cache_adapter(current)
    adapters = {
        rank: (
            basis.truncate(rank),
            compile_low_rank_cache_adapter(current, basis.truncate(rank)),
        )
        for rank in ranks
    }
    current_vec = model_params_vec(current).detach()
    old_vec = model_params_vec(old).detach()
    probe = evaluate(
        current,
        old,
        probe_samples,
        metadata,
        device,
        args.timing_repeats,
        compiled_cheap,
        adapters,
    )
    test = evaluate(
        current,
        old,
        test_samples,
        metadata,
        device,
        args.timing_repeats,
        compiled_cheap,
        adapters,
    )
    result = {
        "protocol": args.protocol,
        "study_stage": args.study_stage
        or (
            "seed0_discovery"
            if args.seed == 0
            else "frozen_rule_replication"
        ),
        "seed": args.seed,
        "source_run": args.run_result,
        "checkpoint_dir": args.checkpoint_dir,
        "model_t": args.model_t,
        "stale_t": 0,
        "dtheta_rel": float((current_vec - old_vec).norm() / old_vec.norm()),
        "split": {
            "rule": "seeded user permutation independent of labels and outcomes",
            "fit_users": len(fit_samples),
            "probe_users": len(probe_samples),
            "test_users": len(test_samples),
            "split_seed": 9151 + args.seed,
        },
        "ranks": ranks,
        "fit": fit_summary,
        "execution": {
            "rule": "fold the calibrated low-rank residual into a fused K/V projection",
            "cheap_prepacked_parameter_numel": compiled_cheap.numel,
            "cheap_prepacked_parameter_bytes": compiled_cheap.nbytes,
            "compiled_parameter_numel": {
                str(rank): compiled.numel
                for rank, (_, compiled) in adapters.items()
            },
            "compiled_parameter_bytes": {
                str(rank): compiled.nbytes
                for rank, (_, compiled) in adapters.items()
            },
        },
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
            f"cache={value['cache_fidelity_recovery']:.3f} "
            f"rank={value['gain_over_reuse']['best_rank']:.2f} "
            f"ndcg100={value['gain_over_reuse']['ndcg@100']:.5f}"
        )
    print(args.output)


if __name__ == "__main__":
    main()
