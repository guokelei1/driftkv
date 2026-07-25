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
    reduce_timing,
    sha256,
    split_samples,
    summarize,
)
from motivation_validity import eval_batches, move_batch, seed_everything
from search_kuairand_long_context_compiled_rank import evaluate_local

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.migration import (
    CompiledCacheAdapter,
    capture_layerwise_state,
    compile_projection_cache_adapter,
    migrate_fused_projection_cache,
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

PROTOCOL = "kuairand_long_context_4plus12_compiled_ridge_search_v1"
DEFAULT_OUTPUT = (
    "results/motivation_scale/"
    "long_context_4plus12_compiled_ridge_search_seed0.json"
)
DEFAULT_PROGRAM_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/"
    "seed0/compiled_ridge_search"
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
        "--ridges",
        type=float,
        nargs="+",
        default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-eval-users", type=int, default=1000)
    parser.add_argument("--fit-users", type=int, default=40)
    parser.add_argument("--probe-users", type=int, default=60)
    parser.add_argument("--max-fit-tokens", type=int, default=8192)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def ridge_name(ridge: float) -> str:
    return f"compiled_ridge_{ridge:.0e}"


@torch.inference_mode()
def fit_ridge_family(
    current: HSTU,
    old: HSTU,
    samples: list[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, CompiledCacheAdapter], dict]:
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
    base = compile_projection_cache_adapter(current)
    family_weights = {ridge: [] for ridge in args.ridges}
    family_biases = {ridge: [] for ridge in args.ridges}
    sampled_tokens = []
    for layer, (features, residuals) in enumerate(
        zip(feature_chunks, residual_chunks, strict=True)
    ):
        x = torch.cat(features)
        y = torch.cat(residuals)
        if len(x) > args.max_fit_tokens:
            rng = np.random.default_rng(args.seed * 1000 + layer)
            indices = torch.from_numpy(
                np.sort(
                    rng.choice(
                        len(x),
                        args.max_fit_tokens,
                        replace=False,
                    )
                )
            )
            x = x[indices]
            y = y[indices]
        sampled_tokens.append(len(x))
        x = x.to(device).float()
        y = y.to(device).float()
        feature_mean = x.mean(dim=0)
        residual_mean = y.mean(dim=0)
        x = x - feature_mean
        y = y - residual_mean
        gram = x.transpose(0, 1) @ x / x.shape[0]
        cross = x.transpose(0, 1) @ y / x.shape[0]
        scale = gram.diagonal().mean().clamp_min(torch.finfo(gram.dtype).eps)
        identity = torch.eye(
            gram.shape[0],
            device=device,
            dtype=gram.dtype,
        )
        for ridge in args.ridges:
            correction = torch.linalg.solve(
                gram + ridge * scale * identity,
                cross,
            )
            correction_bias = (
                residual_mean - feature_mean @ correction
            )
            family_weights[ridge].append(
                base.weights[layer] + correction
            )
            family_biases[ridge].append(
                base.biases[layer] + correction_bias
            )
    family = {
        ridge_name(ridge): CompiledCacheAdapter(
            weights=torch.stack(family_weights[ridge]),
            biases=torch.stack(family_biases[ridge]),
            source_rank=current.cfg.hidden_size,
            ridge=ridge,
        )
        for ridge in args.ridges
    }
    torch.cuda.synchronize(device)
    return family, {
        "fit_users": len(samples),
        "valid_prefix_tokens_before_sampling": valid_tokens,
        "sampled_tokens_per_layer": sampled_tokens,
        "full_affine_rank": current.cfg.hidden_size,
        "ridges": args.ridges,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "compiled_parameter_numel_per_candidate": next(
            iter(family.values())
        ).numel,
        "compiled_fp32_bytes_per_candidate": next(
            iter(family.values())
        ).nbytes,
        "labels_used": False,
    }


def broadcast_family(
    family: dict[str, CompiledCacheAdapter] | None,
    fit: dict | None,
    cfg: HSTUConfig,
    args: argparse.Namespace,
    runtime: DistributedRuntime,
) -> tuple[dict[str, CompiledCacheAdapter], dict]:
    names = [ridge_name(ridge) for ridge in args.ridges]
    width = 2 * cfg.num_heads * cfg.head_dim
    if runtime.is_primary:
        assert family is not None
        weights = torch.stack([family[name].weights for name in names])
        biases = torch.stack([family[name].biases for name in names])
    else:
        weights = torch.empty(
            len(names),
            cfg.num_layers,
            cfg.hidden_size,
            width,
            device=runtime.device,
        )
        biases = torch.empty(
            len(names),
            cfg.num_layers,
            width,
            device=runtime.device,
        )
    dist.broadcast(weights, src=0)
    dist.broadcast(biases, src=0)
    objects = [fit]
    dist.broadcast_object_list(objects, src=0, device=runtime.device)
    fit = objects[0]
    assert isinstance(fit, dict)
    return {
        name: CompiledCacheAdapter(
            weights=weights[index],
            biases=biases[index],
            source_rank=cfg.hidden_size,
            ridge=args.ridges[index],
        )
        for index, name in enumerate(names)
    }, fit


def select_ridge(pairs: list[dict], args: argparse.Namespace) -> dict:
    candidates = []
    for ridge in args.ridges:
        name = ridge_name(ridge)
        candidates.append(
            {
                "ridge": ridge,
                "mean_probe_score_cosine": float(
                    np.mean(
                        [
                            pair["probe"]["configs"][name]["score_cosine"]
                            for pair in pairs
                        ]
                    )
                ),
                "mean_probe_top100_overlap": float(
                    np.mean(
                        [
                            pair["probe"]["configs"][name]["top100_overlap"]
                            for pair in pairs
                        ]
                    )
                ),
            }
        )
    selected = max(
        candidates,
        key=lambda value: (
            value["mean_probe_score_cosine"],
            value["mean_probe_top100_overlap"],
            -value["ridge"],
        ),
    )
    return {
        "rule": (
            "maximize mean label-free fresh-score cosine across all source "
            "version probe cohorts; break ties by top-100 overlap then stronger ridge"
        ),
        "candidates": candidates,
        "selected_ridge": selected["ridge"],
    }


def validate_args(args: argparse.Namespace) -> None:
    if len(set(args.ridges)) != len(args.ridges):
        raise ValueError("ridges must be unique")
    if args.ridges != sorted(args.ridges):
        raise ValueError("ridges must be sorted")
    if min(args.ridges) <= 0:
        raise ValueError("ridges must be positive")
    if args.cache_versions != [0, 4, 10]:
        raise ValueError("ridge search freezes cache versions")
    if args.current_version != 11 or args.base_days != 4:
        raise ValueError("ridge search freezes theta11 under 4+12")
    frozen = {
        "batch_size": (args.batch_size, 4),
        "max_eval_users": (args.max_eval_users, 1000),
        "fit_users": (args.fit_users, 40),
        "probe_users": (args.probe_users, 60),
        "max_fit_tokens": (args.max_fit_tokens, 8192),
        "timing_repeats": (args.timing_repeats, 3),
        "bootstrap_samples": (args.bootstrap_samples, 1000),
    }
    changed = {
        name: {"expected": expected, "actual": actual}
        for name, (actual, expected) in frozen.items()
        if actual != expected
    }
    if changed:
        raise ValueError(f"frozen ridge search settings changed: {changed}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    runtime = init_distributed_runtime(args.device, args.distributed_backend)
    args.device = str(runtime.device)
    try:
        if runtime.device.type != "cuda":
            raise ValueError("ridge search requires CUDA")
        if runtime.world_size != 2:
            raise ValueError("ridge search freezes evaluation at two GPUs")
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
                family, fit = fit_ridge_family(
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
            if runtime.is_primary:
                for name, adapter in family.items():
                    torch.save(
                        {
                            "protocol": PROTOCOL,
                            "source_version": f"theta{cache_version}",
                            "target_version": f"theta{args.current_version}",
                            "rank": cfg.hidden_size,
                            "ridge": adapter.ridge,
                            "weights": adapter.weights.cpu(),
                            "biases": adapter.biases.cpu(),
                            "fit": fit,
                        },
                        program_dir
                        / (
                            f"theta{cache_version}_to_theta"
                            f"{args.current_version}_{name}.pt"
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
                    family,
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
            selection = select_ridge(pairs, args)
            selected_name = ridge_name(selection["selected_ridge"])
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
                    "family": "compiled full-affine ridge sweep",
                    "online_operator_invariant": (
                        "every ridge uses the same one-pass affine projection"
                    ),
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
                        "selected_ridge": selection["selected_ridge"],
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
