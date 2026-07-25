"""Evaluate the compiled low-rank cache migration fast tier on the long-context task."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from layerwise_validity import timed_call
from motivation_validity import (
    bootstrap_interval,
    eval_batches,
    move_batch,
    ranking_metrics,
    seed_everything,
)

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
    METHOD_PROTOCOL,
    TRAINING_PROTOCOL,
    DistributedRuntime,
    close_distributed_runtime,
    gather_records,
    init_distributed_runtime,
    load_checkpoint_model,
    prefix_state_footprint,
    primary_log,
    reconstruct_online_eval_samples,
    validate_long_context_plan,
)
from hstu_kvcache.utils import save_json

LOWER_IS_BETTER = {"best_rank", "mean_rank", "median_rank"}
DEFAULT_TRAINING_RESULT = (
    "results/motivation_scale/long_context_8plus8_training_seed0.json"
)
DEFAULT_CHECKPOINT_DIR = "checkpoints/kuairand_long_context_8plus8/seed0"
DEFAULT_OUTPUT = "results/motivation_scale/long_context_8plus8_method_seed0.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared-data",
        default="data/processed/kuairand_long_context_8plus8_v2.npz",
    )
    parser.add_argument(
        "--training-result",
        default=DEFAULT_TRAINING_RESULT,
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributed-backend", default="nccl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-eval-users", type=int, default=1000)
    parser.add_argument("--fit-users", type=int, default=40)
    parser.add_argument("--max-fit-tokens", type=int, default=8192)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--distributed-smoke-test", action="store_true")
    return parser.parse_args()


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_fit_test(
    samples: list[dict],
    fit_users: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if not 0 < fit_users < len(samples):
        raise ValueError("fit_users must leave at least one held-out user")
    rng = np.random.default_rng(9151 + seed)
    fit_indices = set(rng.permutation(len(samples))[:fit_users].tolist())
    fit = [
        sample
        for index, sample in enumerate(samples)
        if index in fit_indices
    ]
    test = [
        sample
        for index, sample in enumerate(samples)
        if index not in fit_indices
    ]
    return fit, test


@torch.inference_mode()
def fit_compiled_adapter(
    current: HSTU,
    old: HSTU,
    samples: list[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[CompiledCacheAdapter, dict]:
    feature_chunks: list[list[torch.Tensor]] = [
        [] for _ in current.blocks
    ]
    residual_chunks: list[list[torch.Tensor]] = [
        [] for _ in current.blocks
    ]
    valid_tokens = 0
    if device.type == "cuda":
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
                rank=args.rank,
                ridge=args.ridge,
            )
        )
    adapter = LowRankCacheAdapter(
        layers=tuple(layers),
        ridge=args.ridge,
    )
    compiled = compile_low_rank_cache_adapter(current, adapter)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return compiled, {
        "fit_users": len(samples),
        "valid_prefix_tokens_before_sampling": valid_tokens,
        "sampled_tokens_per_layer": sampled_tokens,
        "rank": args.rank,
        "ridge": args.ridge,
        "elapsed_ms": elapsed_ms,
        "low_rank_parameter_numel": adapter.numel,
        "compiled_parameter_numel": compiled.numel,
        "compiled_fp32_bytes": compiled.nbytes,
        "labels_used": False,
    }


def broadcast_adapter(
    compiled: CompiledCacheAdapter | None,
    fit_summary: dict | None,
    cfg: HSTUConfig,
    args: argparse.Namespace,
    runtime: DistributedRuntime,
) -> tuple[CompiledCacheAdapter, dict]:
    if not runtime.initialized:
        assert compiled is not None and fit_summary is not None
        return compiled, fit_summary
    output_width = 2 * cfg.num_heads * cfg.head_dim
    if runtime.is_primary:
        assert compiled is not None
        weights = compiled.weights
        biases = compiled.biases
    else:
        weights = torch.empty(
            cfg.num_layers,
            cfg.hidden_size,
            output_width,
            device=runtime.device,
            dtype=torch.float32,
        )
        biases = torch.empty(
            cfg.num_layers,
            output_width,
            device=runtime.device,
            dtype=torch.float32,
        )
    dist.broadcast(weights, src=0)
    dist.broadcast(biases, src=0)
    objects = [fit_summary]
    dist.broadcast_object_list(objects, src=0, device=runtime.device)
    summary = objects[0]
    assert isinstance(summary, dict)
    return (
        CompiledCacheAdapter(
            weights=weights,
            biases=biases,
            source_rank=args.rank,
            ridge=args.ridge,
        ),
        summary,
    )


@torch.inference_mode()
def evaluate_local(
    current: HSTU,
    old: HSTU,
    compiled: CompiledCacheAdapter,
    samples: list[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict[str, float]]:
    all_items = torch.arange(
        1,
        current.cfg.num_prediction_items + 1,
        device=device,
    )
    records = []
    method_timing = 0.0
    recompute_timing = 0.0
    batch_count = 0
    for selected, full_cpu, prefix_cpu, suffix_cpu in eval_batches(
        samples,
        args.seq_len,
        args.batch_size,
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
        method_cache, method_ms = timed_call(
            lambda old_state=old_state, compiled=compiled: (
                migrate_compiled_low_rank_cache(old_state, compiled)
            ),
            device,
            args.timing_repeats,
        )
        recompute_cache, recompute_ms = timed_call(
            lambda prefix=prefix, current=current: current.compute_kv(
                prefix["item_ids"],
                prefix["behaviors"],
                prefix["time_deltas"],
                lengths=prefix["lengths"],
            ),
            device,
            args.timing_repeats,
        )
        caches = {
            "reuse": old_state.kv,
            f"compiled_rank_{args.rank}": method_cache,
            "recompute": recompute_cache,
        }
        candidate_ids = all_items.unsqueeze(0).expand(len(selected), -1)
        fresh_scores = current.item_emb.score(fresh_hidden, candidate_ids)
        fresh_incremental, _ = current.forward_with_cache(
            fresh_cache,
            suffix["item_ids"],
            suffix["behaviors"],
            suffix["time_deltas"],
        )
        parity = (fresh_incremental[:, 0] - fresh_hidden).abs().amax(dim=1)
        metrics = {}
        errors = {}
        for name, cache in caches.items():
            hidden, _ = current.forward_with_cache(
                cache,
                suffix["item_ids"],
                suffix["behaviors"],
                suffix["time_deltas"],
            )
            scores = current.item_emb.score(hidden[:, 0], candidate_ids)
            metrics[name] = [
                ranking_metrics(scores[row], selected[row]["pos_items"])
                for row in range(len(selected))
            ]
            errors[name] = sample_relative_cache_error(
                cache,
                fresh_cache,
            ).cpu().tolist()
        for row, sample in enumerate(selected):
            records.append(
                {
                    "user_id": int(sample["history"]["user_id"]),
                    "history_length": int(full["lengths"][row].item()),
                    "fresh": ranking_metrics(
                        fresh_scores[row],
                        sample["pos_items"],
                    ),
                    "configs": {
                        name: metrics[name][row]
                        for name in metrics
                    },
                    "cache_error_rel": {
                        name: float(errors[name][row])
                        for name in errors
                    },
                    "fresh_incremental_parity_max_abs": float(parity[row].item()),
                }
            )
        method_timing += method_ms
        recompute_timing += recompute_ms
        batch_count += 1
    return records, {
        "method_ms": method_timing,
        "recompute_ms": recompute_timing,
        "users": len(records),
        "batches": batch_count,
    }


def reduce_timing(
    timing: dict[str, float],
    runtime: DistributedRuntime,
) -> dict[str, float]:
    values = torch.tensor(
        [
            timing["method_ms"],
            timing["recompute_ms"],
            timing["users"],
            timing["batches"],
        ],
        dtype=torch.float64,
        device=runtime.device,
    )
    if runtime.initialized:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {
        "method_ms": float(values[0].item()),
        "recompute_ms": float(values[1].item()),
        "users": int(values[2].item()),
        "batches": int(values[3].item()),
    }


def summarize_method(
    records: list[dict],
    timing: dict[str, float],
    args: argparse.Namespace,
) -> dict:
    config_names = list(records[0]["configs"])
    metric_names = list(records[0]["fresh"])
    rng = np.random.default_rng(args.seed + 7717)
    fresh_metrics = {
        metric: float(np.mean([record["fresh"][metric] for record in records]))
        for metric in metric_names
    }
    configs = {}
    reuse_errors = np.asarray(
        [record["cache_error_rel"]["reuse"] for record in records]
    )
    for name in config_names:
        metrics = {}
        loss_from_fresh = {}
        gain_over_reuse = {}
        for metric in metric_names:
            fresh = np.asarray([record["fresh"][metric] for record in records])
            value = np.asarray(
                [record["configs"][name][metric] for record in records]
            )
            reuse = np.asarray(
                [record["configs"]["reuse"][metric] for record in records]
            )
            if metric in LOWER_IS_BETTER:
                loss = value - fresh
                gain = reuse - value
            else:
                loss = fresh - value
                gain = value - reuse
            metrics[metric] = float(value.mean())
            loss_from_fresh[metric] = {
                "mean": float(loss.mean()),
                "ci95": bootstrap_interval(
                    loss,
                    rng,
                    args.bootstrap_samples,
                ),
                "worse_fraction": float(np.mean(loss > 0)),
            }
            gain_over_reuse[metric] = {
                "mean": float(gain.mean()),
                "ci95": bootstrap_interval(
                    gain,
                    rng,
                    args.bootstrap_samples,
                ),
                "better_fraction": float(np.mean(gain > 0)),
            }
        cache_errors = np.asarray(
            [record["cache_error_rel"][name] for record in records]
        )
        configs[name] = {
            "metrics": metrics,
            "quality_loss_from_fresh": loss_from_fresh,
            "quality_gain_over_reuse": gain_over_reuse,
            "cache_error_rel": float(cache_errors.mean()),
            "cache_fidelity_recovery": float(
                (reuse_errors.mean() - cache_errors.mean())
                / max(reuse_errors.mean(), 1e-12)
            ),
        }
    users = max(int(timing["users"]), 1)
    method_name = f"compiled_rank_{args.rank}"
    method_ms_per_user = timing["method_ms"] / users
    recompute_ms_per_user = timing["recompute_ms"] / users
    configs["reuse"]["migration_ms_per_user"] = 0.0
    configs["reuse"]["migration_ratio_to_recompute"] = 0.0
    configs[method_name]["migration_ms_per_user"] = method_ms_per_user
    configs[method_name]["migration_ratio_to_recompute"] = (
        method_ms_per_user / max(recompute_ms_per_user, 1e-12)
    )
    configs["recompute"]["migration_ms_per_user"] = recompute_ms_per_user
    configs["recompute"]["migration_ratio_to_recompute"] = 1.0
    return {
        "n": len(records),
        "fresh": fresh_metrics,
        "configs": configs,
        "timing": timing,
        "fresh_incremental_parity_max_abs": max(
            record["fresh_incremental_parity_max_abs"]
            for record in records
        ),
    }


def smoke_samples(count: int) -> list[dict]:
    return [
        {
            "history": {
                "item_ids": np.asarray([1, 2, 3, 4], dtype=np.int64),
                "behaviors": np.asarray([1, 2, 3, 2], dtype=np.int64),
                "time_deltas": np.asarray(
                    [0.0, 1.0, 2.0, 3.0],
                    dtype=np.float32,
                ),
                "labels": np.asarray([0, 1, 1, 1], dtype=np.int64),
                "user_id": index + 1,
            },
            "pos_items": [5, 6],
        }
        for index in range(count)
    ]


def run_distributed_smoke_test(
    args: argparse.Namespace,
    runtime: DistributedRuntime,
) -> None:
    seed_everything(0)
    cfg = HSTUConfig(
        num_items=128,
        num_prediction_items=96,
        num_behaviors=9,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        head_dim=8,
        max_seq_len=8,
        activation="relu",
        input_dropout=0.0,
    )
    old = HSTU(cfg).to(runtime.device)
    current = HSTU(cfg).to(runtime.device)
    current.load_state_dict(old.state_dict())
    with torch.no_grad():
        for parameter in current.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.001)
    old.eval()
    current.eval()
    args.seq_len = 8
    args.seed = 0
    args.rank = 2
    args.fit_users = 4
    args.max_fit_tokens = 32
    args.timing_repeats = 1
    samples = smoke_samples(max(8, runtime.world_size * 3))
    fit, test = split_fit_test(samples, args.fit_users, args.seed)
    if runtime.is_primary:
        compiled, fit_summary = fit_compiled_adapter(
            current,
            old,
            fit,
            args,
            runtime.device,
        )
    else:
        compiled = None
        fit_summary = None
    compiled, _ = broadcast_adapter(
        compiled,
        fit_summary,
        cfg,
        args,
        runtime,
    )
    local_test = test[runtime.rank::runtime.world_size]
    records, timing = evaluate_local(
        current,
        old,
        compiled,
        local_test,
        args,
        runtime.device,
    )
    timing = reduce_timing(timing, runtime)
    gathered = gather_records(records, runtime)
    if runtime.is_primary:
        assert gathered is not None
        summary = summarize_method(gathered, timing, args)
        if summary["fresh_incremental_parity_max_abs"] > 1e-4:
            raise RuntimeError("method smoke test failed fresh parity")
        if f"compiled_rank_{args.rank}" not in summary["configs"]:
            raise RuntimeError("method smoke test omitted the compiled method")
        print(
            json.dumps(
                {
                    "world_size": runtime.world_size,
                    "test_users": len(gathered),
                    "method_ratio": summary["configs"][
                        f"compiled_rank_{args.rank}"
                    ]["migration_ratio_to_recompute"],
                    "status": "ok",
                },
                indent=2,
            ),
            flush=True,
        )


def main() -> None:
    args = parse_args()
    positive = {
        "batch_size": args.batch_size,
        "max_eval_users": args.max_eval_users,
        "fit_users": args.fit_users,
        "max_fit_tokens": args.max_fit_tokens,
        "rank": args.rank,
        "timing_repeats": args.timing_repeats,
        "bootstrap_samples": args.bootstrap_samples,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"positive values required for: {', '.join(invalid)}")
    if args.ridge < 0:
        raise ValueError("ridge must be nonnegative")
    frozen = {
        "batch_size": (args.batch_size, 4),
        "max_eval_users": (args.max_eval_users, 1000),
        "fit_users": (args.fit_users, 40),
        "max_fit_tokens": (args.max_fit_tokens, 8192),
        "rank": (args.rank, 16),
        "ridge": (args.ridge, 1e-3),
        "timing_repeats": (args.timing_repeats, 3),
        "bootstrap_samples": (args.bootstrap_samples, 1000),
    }
    changed = {
        name: {"expected": expected, "actual": actual}
        for name, (actual, expected) in frozen.items()
        if actual != expected
    }
    if changed and not args.distributed_smoke_test:
        raise ValueError(f"frozen method protocol arguments changed: {changed}")
    runtime = init_distributed_runtime(
        args.device,
        args.distributed_backend,
    )
    args.device = str(runtime.device)
    try:
        if args.distributed_smoke_test:
            run_distributed_smoke_test(args, runtime)
            return
        if runtime.world_size != 4:
            raise ValueError("formal long-context method evaluation requires exactly four workers")
        if runtime.device.type != "cuda":
            raise ValueError("method timing requires CUDA")
        source = json.loads(Path(args.training_result).read_text())
        if source.get("protocol") != TRAINING_PROTOCOL:
            raise ValueError("training result protocol does not match")
        if source.get("status") != "complete":
            raise ValueError("training result is incomplete")
        expected_hash = source["prepared_data"]["sha256"]
        actual_hash = artifact_sha256(args.prepared_data)
        if actual_hash != expected_hash:
            raise ValueError("prepared data differs from the training artifact")
        plan, prepared_metadata = load_prepared_kuairand_plan(args.prepared_data)
        validate_long_context_plan(plan, prepared_metadata)
        cfg = HSTUConfig(**source["model"])
        if args.rank > min(cfg.hidden_size, 2 * cfg.num_heads * cfg.head_dim):
            raise ValueError("adapter rank exceeds the supported matrix rank")
        args.seq_len = cfg.max_seq_len
        args.seed = int(source["args"]["seed"])
        if args.seed != 0 and (
            args.checkpoint_dir == DEFAULT_CHECKPOINT_DIR
            or args.output == DEFAULT_OUTPUT
        ):
            raise ValueError("a nonzero training seed requires seed-specific evaluation paths")
        date, samples = reconstruct_online_eval_samples(
            plan,
            (7,),
            args.max_eval_users,
        )[7]
        fit_samples, test_samples = split_fit_test(
            samples,
            args.fit_users,
            args.seed,
        )
        old = load_checkpoint_model(
            cfg,
            args.checkpoint_dir,
            0,
            runtime.device,
        )
        current = load_checkpoint_model(
            cfg,
            args.checkpoint_dir,
            7,
            runtime.device,
        )
        if runtime.is_primary:
            compiled, fit_summary = fit_compiled_adapter(
                current,
                old,
                fit_samples,
                args,
                runtime.device,
            )
        else:
            compiled = None
            fit_summary = None
        compiled, fit_summary = broadcast_adapter(
            compiled,
            fit_summary,
            cfg,
            args,
            runtime,
        )
        local_test = test_samples[runtime.rank::runtime.world_size]
        local_test.sort(key=lambda sample: len(sample["history"]["item_ids"]))
        local_records, local_timing = evaluate_local(
            current,
            old,
            compiled,
            local_test,
            args,
            runtime.device,
        )
        timing = reduce_timing(local_timing, runtime)
        records = gather_records(local_records, runtime)
        if runtime.is_primary:
            assert records is not None
            summary = summarize_method(records, timing, args)
            method_name = f"compiled_rank_{args.rank}"
            result = {
                "protocol": METHOD_PROTOCOL,
                "source_training_result": args.training_result,
                "prepared_data": {
                    "path": args.prepared_data,
                    "sha256": actual_hash,
                    "metadata": prepared_metadata,
                },
                "checkpoint_dir": args.checkpoint_dir,
                "seed": args.seed,
                "world_size": runtime.world_size,
                "eval_date": date,
                "current_version": 7,
                "cache_version": 0,
                "cache_age_updates": 7,
                "cache_age_semantics": (
                    "checkpoint-update distance used to encode the same resident "
                    "D16 prefix; it is not physical snapshot residence time"
                ),
                "physical_cache_lifecycle_scope": (
                    "literal snapshot survival, rolling eviction, and organically "
                    "mixed per-token versions require a separate lifecycle experiment"
                ),
                "split": {
                    "rule": (
                        "seeded fit/test user split independent of labels and "
                        "model outcomes"
                    ),
                    "fit_users": len(fit_samples),
                    "test_users": len(test_samples),
                    "split_seed": 9151 + args.seed,
                },
                "operator": {
                    "name": method_name,
                    "description": (
                        "version-level fresh-minus-current-projection residual "
                        "compiled into one affine K/V projection over cached "
                        "theta0 Norm(x)"
                    ),
                    "stored_per_user_state": (
                        "theta0-encoded resident-prefix K/V plus layerwise theta0 Norm(x)"
                    ),
                    "labels_used_for_fit_or_selection": False,
                    "rank": args.rank,
                    "ridge": args.ridge,
                },
                "state_footprint": {
                    "fit": prefix_state_footprint(fit_samples, cfg),
                    "test": prefix_state_footprint(test_samples, cfg),
                },
                "fit": fit_summary,
                "summary": summary,
                "per_user": records,
            }
            save_json(result, args.output)
            method = summary["configs"][method_name]
            primary_log(
                runtime,
                f"method={method_name} "
                f"cost={method['migration_ratio_to_recompute']:.3f}x "
                f"cache_recovery={method['cache_fidelity_recovery']:.3f} "
                f"best_rank_gain={method['quality_gain_over_reuse']['best_rank']['mean']:.3f} "
                f"ndcg100_gain={method['quality_gain_over_reuse']['ndcg@100']['mean']:.6f}",
            )
            print(args.output, flush=True)
    finally:
        close_distributed_runtime(runtime)


if __name__ == "__main__":
    main()
