from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from evaluate_kuairand_long_context_sync_design import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_PREPARED,
    DEFAULT_TRAINING,
    cache_output_metrics,
    reduce_timing,
    sha256,
    split_samples,
    summarize,
)
from layerwise_validity import timed_call
from motivation_validity import eval_batches, move_batch, ranking_metrics, seed_everything

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.migration import (
    CompiledCacheAdapter,
    LowRankCacheAdapter,
    capture_layerwise_state,
    compile_low_rank_cache_adapter,
    fit_low_rank_layer_adapter,
    migrate_compiled_low_rank_cache,
    migrate_fused_projection_cache,
    sample_relative_cache_error,
)
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming import (
    DistributedRuntime,
    close_distributed_runtime,
    gather_records,
    init_distributed_runtime,
    load_checkpoint_model,
    reconstruct_online_eval_samples,
    training_protocol_for_base_days,
    validate_long_context_plan,
)
from hstu_kvcache.utils import save_json

PROTOCOL = "kuairand_long_context_4plus12_compiled_rank_search_v1"
DEFAULT_OUTPUT = (
    "results/motivation_scale/"
    "long_context_4plus12_compiled_rank_search_seed0.json"
)
DEFAULT_PROGRAM_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/"
    "seed0/compiled_rank_search"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--program-dir", default=DEFAULT_PROGRAM_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributed-backend", default="nccl")
    parser.add_argument("--base-days", type=int, default=4)
    parser.add_argument("--current-version", type=int, default=11)
    parser.add_argument("--cache-versions", type=int, nargs="+", default=[0, 4, 10])
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512],
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-eval-users", type=int, default=1000)
    parser.add_argument("--fit-users", type=int, default=40)
    parser.add_argument("--probe-users", type=int, default=60)
    parser.add_argument("--max-fit-tokens", type=int, default=8192)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


@torch.inference_mode()
def fit_rank_family(
    current: HSTU,
    old: HSTU,
    samples: list[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[int, CompiledCacheAdapter], dict]:
    feature_chunks: list[list[torch.Tensor]] = [
        [] for _ in current.blocks
    ]
    residual_chunks: list[list[torch.Tensor]] = [
        [] for _ in current.blocks
    ]
    valid_tokens = 0
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _, _, prefix_cpu, _ in eval_batches(
        samples,
        args.seq_len,
        args.batch_size,
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
            feature_chunks[layer].append(features[valid].cpu())
            residual_chunks[layer].append(residual[valid].cpu())
    max_rank = max(args.ranks)
    layers = []
    sampled_tokens = []
    for layer, (features, residuals) in enumerate(
        zip(feature_chunks, residual_chunks, strict=True)
    ):
        feature = torch.cat(features)
        residual = torch.cat(residuals)
        if len(feature) > args.max_fit_tokens:
            rng = np.random.default_rng(args.seed * 1000 + layer)
            indices = torch.from_numpy(
                np.sort(
                    rng.choice(
                        len(feature),
                        args.max_fit_tokens,
                        replace=False,
                    )
                )
            )
            feature = feature[indices]
            residual = residual[indices]
        sampled_tokens.append(len(feature))
        layers.append(
            fit_low_rank_layer_adapter(
                feature.to(device),
                residual.to(device),
                rank=max_rank,
                ridge=args.ridge,
            )
        )
    maximum = LowRankCacheAdapter(
        layers=tuple(layers),
        ridge=args.ridge,
    )
    family = {
        rank: compile_low_rank_cache_adapter(
            current,
            maximum.truncate(rank),
        )
        for rank in args.ranks
    }
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return family, {
        "fit_users": len(samples),
        "valid_prefix_tokens_before_sampling": valid_tokens,
        "sampled_tokens_per_layer": sampled_tokens,
        "ranks": args.ranks,
        "maximum_rank_fit_once": max_rank,
        "ridge": args.ridge,
        "elapsed_ms": elapsed_ms,
        "compiled_parameter_numel_per_rank": next(iter(family.values())).numel,
        "compiled_fp32_bytes_per_rank": next(iter(family.values())).nbytes,
        "labels_used": False,
    }


def broadcast_family(
    family: dict[int, CompiledCacheAdapter] | None,
    fit: dict | None,
    cfg: HSTUConfig,
    args: argparse.Namespace,
    runtime: DistributedRuntime,
) -> tuple[dict[int, CompiledCacheAdapter], dict]:
    output_width = 2 * cfg.num_heads * cfg.head_dim
    if runtime.is_primary:
        assert family is not None
        weights = torch.stack([family[rank].weights for rank in args.ranks])
        biases = torch.stack([family[rank].biases for rank in args.ranks])
    else:
        weights = torch.empty(
            len(args.ranks),
            cfg.num_layers,
            cfg.hidden_size,
            output_width,
            device=runtime.device,
        )
        biases = torch.empty(
            len(args.ranks),
            cfg.num_layers,
            output_width,
            device=runtime.device,
        )
    dist.broadcast(weights, src=0)
    dist.broadcast(biases, src=0)
    objects = [fit]
    dist.broadcast_object_list(objects, src=0, device=runtime.device)
    fit = objects[0]
    assert isinstance(fit, dict)
    return {
        rank: CompiledCacheAdapter(
            weights=weights[index],
            biases=biases[index],
            source_rank=rank,
            ridge=args.ridge,
        )
        for index, rank in enumerate(args.ranks)
    }, fit


@torch.inference_mode()
def evaluate_local(
    current: HSTU,
    old: HSTU,
    family: dict[str, CompiledCacheAdapter],
    samples: list[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict]:
    names = [
        "reuse",
        "projection_only",
        *family,
        "recompute",
    ]
    timing = {name: 0.0 for name in names}
    records = []
    all_items = torch.arange(
        1,
        current.cfg.num_prediction_items + 1,
        device=device,
    )
    batches = 0
    for selected, full_cpu, prefix_cpu, suffix_cpu in eval_batches(
        samples,
        args.seq_len,
        args.batch_size,
    ):
        full = move_batch(full_cpu, device)
        prefix = move_batch(prefix_cpu, device)
        suffix = move_batch(suffix_cpu, device)
        full_output, _ = current(
            full["item_ids"],
            full["behaviors"],
            full["time_deltas"],
            lengths=full["lengths"],
        )
        fresh_hidden = current.last_hidden(full_output, full["lengths"])
        candidate_ids = all_items.unsqueeze(0).expand(len(selected), -1)
        fresh_scores = current.item_emb.score(fresh_hidden, candidate_ids)
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
        builders = {
            "reuse": lambda old_state=old_state: old_state.kv,
            "projection_only": lambda old_state=old_state: migrate_fused_projection_cache(
                current,
                old_state,
            ),
            **{
                name: (
                    lambda name=name, old_state=old_state: (
                        migrate_compiled_low_rank_cache(
                            old_state,
                            family[name],
                        )
                    )
                )
                for name in family
            },
            "recompute": lambda prefix=prefix: current.compute_kv(
                prefix["item_ids"],
                prefix["behaviors"],
                prefix["time_deltas"],
                lengths=prefix["lengths"],
            ),
        }
        values = {
            "fresh": [
                ranking_metrics(fresh_scores[row], selected[row]["pos_items"])
                for row in range(len(selected))
            ],
            "configs": {},
        }
        for name in names:
            if name == "reuse":
                cache = old_state.kv
                elapsed_ms = 0.0
            else:
                cache, elapsed_ms = timed_call(
                    builders[name],
                    device,
                    args.timing_repeats,
                )
            timing[name] += elapsed_ms
            metrics, hidden_cosine, score_cosine, top100_overlap = (
                cache_output_metrics(
                    current,
                    cache,
                    suffix,
                    candidate_ids,
                    fresh_hidden,
                    fresh_scores,
                    selected,
                )
            )
            errors = sample_relative_cache_error(cache, fresh_cache).cpu().tolist()
            values["configs"][name] = {
                "metrics": metrics,
                "cache_error_rel": errors,
                "hidden_cosine": hidden_cosine,
                "score_cosine": score_cosine,
                "top100_overlap": top100_overlap,
            }
            if name not in {"reuse", "recompute"}:
                del cache
        for row, sample in enumerate(selected):
            records.append(
                {
                    "user_id": int(sample["history"]["user_id"]),
                    "history_length": int(full["lengths"][row].item()),
                    "fresh": values["fresh"][row],
                    "configs": {
                        name: {
                            "metrics": values["configs"][name]["metrics"][row],
                            "cache_error_rel": float(
                                values["configs"][name]["cache_error_rel"][row]
                            ),
                            "hidden_cosine": float(
                                values["configs"][name]["hidden_cosine"][row]
                            ),
                            "score_cosine": float(
                                values["configs"][name]["score_cosine"][row]
                            ),
                            "top100_overlap": float(
                                values["configs"][name]["top100_overlap"][row]
                            ),
                        }
                        for name in names
                    },
                    "recompute_fresh_metric_max_abs": max(
                        abs(
                            values["configs"]["recompute"]["metrics"][row][metric]
                            - values["fresh"][row][metric]
                        )
                        for metric in values["fresh"][row]
                    ),
                }
            )
        batches += 1
    return records, {
        "milliseconds": timing,
        "users": len(records),
        "batches": batches,
    }


def select_global_rank(pairs: list[dict], ranks: list[int]) -> dict:
    candidates = []
    for rank in ranks:
        name = f"compiled_rank_{rank}"
        score_cosines = [
            pair["probe"]["configs"][name]["score_cosine"]
            for pair in pairs
        ]
        overlaps = [
            pair["probe"]["configs"][name]["top100_overlap"]
            for pair in pairs
        ]
        candidates.append(
            {
                "rank": rank,
                "mean_probe_score_cosine": float(np.mean(score_cosines)),
                "mean_probe_top100_overlap": float(np.mean(overlaps)),
            }
        )
    selected = max(
        candidates,
        key=lambda value: (
            value["mean_probe_score_cosine"],
            value["mean_probe_top100_overlap"],
            -value["rank"],
        ),
    )
    return {
        "rule": (
            "maximize mean label-free fresh-score cosine across the three "
            "source-version probe cohorts; break ties by top-100 overlap then lower rank"
        ),
        "candidates": candidates,
        "selected_rank": selected["rank"],
    }


def validate_args(args: argparse.Namespace) -> None:
    if len(set(args.ranks)) != len(args.ranks):
        raise ValueError("ranks must be unique")
    if args.ranks != sorted(args.ranks):
        raise ValueError("ranks must be sorted")
    if min(args.ranks) < 1 or max(args.ranks) > 512:
        raise ValueError("ranks must be in [1, 512]")
    if args.cache_versions != [0, 4, 10]:
        raise ValueError("rank search freezes cache versions at 0, 4, and 10")
    if args.current_version != 11 or args.base_days != 4:
        raise ValueError("rank search freezes theta11 under the 4+12 split")
    if args.fit_users != 40 or args.probe_users != 60:
        raise ValueError("rank search freezes the 40/60 fit/probe split")
    if args.batch_size != 4 or args.max_eval_users != 1000:
        raise ValueError("rank search freezes batch size and full user coverage")
    if args.max_fit_tokens != 8192 or args.ridge != 1e-3:
        raise ValueError("rank search freezes sampling and ridge")
    if args.timing_repeats != 3 or args.bootstrap_samples != 1000:
        raise ValueError("rank search freezes timing and bootstrap settings")


def main() -> None:
    args = parse_args()
    validate_args(args)
    runtime = init_distributed_runtime(args.device, args.distributed_backend)
    args.device = str(runtime.device)
    try:
        if runtime.device.type != "cuda":
            raise ValueError("rank search requires CUDA")
        if runtime.world_size != 2:
            raise ValueError("rank search freezes evaluation at two GPUs")
        seed_everything(0)
        training = json.loads(Path(args.training_result).read_text())
        if training.get("protocol") != training_protocol_for_base_days(args.base_days):
            raise ValueError("training protocol mismatch")
        if training.get("status") != "complete":
            raise ValueError("training result is incomplete")
        prepared_hash = sha256(args.prepared_data)
        if prepared_hash != training["prepared_data"]["sha256"]:
            raise ValueError("prepared data differs from training")
        plan, metadata = load_prepared_kuairand_plan(args.prepared_data)
        validate_long_context_plan(plan, metadata, args.base_days)
        cfg = HSTUConfig(**training["model"])
        args.seq_len = cfg.max_seq_len
        args.seed = int(training["args"]["seed"])
        date, samples = reconstruct_online_eval_samples(
            plan,
            (args.current_version,),
            args.max_eval_users,
        )[args.current_version]
        fit_samples, probe_samples, test_samples = split_samples(
            samples,
            args.fit_users,
            args.probe_users,
            args.seed,
        )
        current = load_checkpoint_model(
            cfg,
            args.checkpoint_dir,
            args.current_version,
            runtime.device,
        )
        program_dir = Path(args.program_dir)
        if runtime.is_primary:
            program_dir.mkdir(parents=True, exist_ok=True)
        if runtime.initialized:
            dist.barrier()
        pairs = []
        for cache_version in args.cache_versions:
            old = load_checkpoint_model(
                cfg,
                args.checkpoint_dir,
                cache_version,
                runtime.device,
            )
            if runtime.is_primary:
                family, fit = fit_rank_family(
                    current,
                    old,
                    fit_samples,
                    args,
                    runtime.device,
                )
            else:
                family = None
                fit = None
            family, fit = broadcast_family(
                family,
                fit,
                cfg,
                args,
                runtime,
            )
            named_family = {
                f"compiled_rank_{rank}": adapter
                for rank, adapter in family.items()
            }
            if runtime.is_primary:
                for rank, adapter in family.items():
                    torch.save(
                        {
                            "protocol": PROTOCOL,
                            "source_version": f"theta{cache_version}",
                            "target_version": f"theta{args.current_version}",
                            "rank": rank,
                            "ridge": args.ridge,
                            "weights": adapter.weights.cpu(),
                            "biases": adapter.biases.cpu(),
                            "fit": fit,
                        },
                        program_dir
                        / (
                            f"theta{cache_version}_to_theta"
                            f"{args.current_version}_rank{rank}.pt"
                        ),
                    )
            split_results = {}
            for split_name, split_values in (
                ("probe", probe_samples),
                ("test", test_samples),
            ):
                local = split_values[runtime.rank :: runtime.world_size]
                local.sort(key=lambda sample: len(sample["history"]["item_ids"]))
                local_records, local_timing = evaluate_local(
                    current,
                    old,
                    named_family,
                    local,
                    args,
                    runtime.device,
                )
                timing = reduce_timing(local_timing, runtime)
                records = gather_records(local_records, runtime)
                if runtime.is_primary:
                    assert records is not None
                    split_results[split_name] = summarize(
                        records,
                        timing,
                        args,
                    )
                    if split_name == "test":
                        split_results["per_user_test"] = records
            if runtime.is_primary:
                pairs.append(
                    {
                        "cache_version": cache_version,
                        "current_version": args.current_version,
                        "cache_age_updates": args.current_version - cache_version,
                        "fit": fit,
                        **split_results,
                    }
                )
            del old, family
            torch.cuda.empty_cache()
        if runtime.is_primary:
            selection = select_global_rank(pairs, args.ranks)
            selected_name = f"compiled_rank_{selection['selected_rank']}"
            result = {
                "protocol": PROTOCOL,
                "status": "design_search_complete",
                "source_training_result": args.training_result,
                "prepared_data": {
                    "path": args.prepared_data,
                    "sha256": prepared_hash,
                },
                "checkpoint_dir": args.checkpoint_dir,
                "program_dir": args.program_dir,
                "seed": args.seed,
                "world_size": runtime.world_size,
                "eval_date": date,
                "split": {
                    "fit_users": len(fit_samples),
                    "probe_users": len(probe_samples),
                    "test_users": len(test_samples),
                    "split_seed": 9151 + args.seed,
                },
                "design": {
                    "family": "compiled affine residual rank sweep",
                    "online_operator_invariant": (
                        "every rank compiles to one affine projection with "
                        "identical weight shape and program bytes"
                    ),
                    "ranks": args.ranks,
                    "selection": selection,
                },
                "selected_test": [
                    {
                        "cache_version": pair["cache_version"],
                        "cache_age_updates": pair["cache_age_updates"],
                        "summary": pair["test"]["configs"][selected_name],
                    }
                    for pair in pairs
                ],
                "pairs": pairs,
            }
            save_json(result, args.output)
            print(
                json.dumps(
                    {
                        "output": args.output,
                        "selected_rank": selection["selected_rank"],
                        "world_size": runtime.world_size,
                    },
                    indent=2,
                ),
                flush=True,
            )
    finally:
        close_distributed_runtime(runtime)


if __name__ == "__main__":
    main()
