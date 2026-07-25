from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from evaluate_kuairand_long_context_method import (
    broadcast_adapter,
    fit_compiled_adapter,
)
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
    MigrationProgram,
    capture_layerwise_state,
    migrate_compiled_low_rank_cache,
    migrate_fused_projection_cache,
    migrate_prefix_residual_cache,
    sample_relative_cache_error,
)
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from hstu_kvcache.streaming import (
    SYNC_DESIGN_PROTOCOL,
    DistributedRuntime,
    close_distributed_runtime,
    gather_records,
    init_distributed_runtime,
    load_checkpoint_model,
    prefix_state_footprint,
    primary_log,
    reconstruct_online_eval_samples,
    training_protocol_for_base_days,
    validate_long_context_plan,
)
from hstu_kvcache.utils import save_json

LOWER_IS_BETTER = {"best_rank", "mean_rank", "median_rank"}
DEFAULT_PREPARED = (
    "data/processed/kuairand_long_context_4plus12_exploration_v1.npz"
)
DEFAULT_TRAINING = (
    "results/motivation_scale/"
    "long_context_4plus12_training_exploration_seed0.json"
)
DEFAULT_CHECKPOINTS = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0"
)
DEFAULT_OUTPUT = (
    "results/motivation_scale/"
    "long_context_4plus12_progressive_sync_design_seed0.json"
)
DEFAULT_PROGRAM_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/"
    "seed0/sync_programs"
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-eval-users", type=int, default=1000)
    parser.add_argument("--fit-users", type=int, default=40)
    parser.add_argument("--probe-users", type=int, default=60)
    parser.add_argument("--max-fit-tokens", type=int, default=8192)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--residual-depths", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--allow-diagnostic-protocol",
        "--allow-diagnostic-world-size",
        dest="allow_diagnostic_protocol",
        action="store_true",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_samples(
    samples: list[dict],
    fit_users: int,
    probe_users: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    if fit_users < 1 or probe_users < 1:
        raise ValueError("fit_users and probe_users must be positive")
    if fit_users + probe_users >= len(samples):
        raise ValueError("fit and probe splits must leave held-out test users")
    order = np.random.default_rng(9151 + seed).permutation(len(samples))
    fit = [samples[index] for index in order[:fit_users]]
    probe = [
        samples[index]
        for index in order[fit_users : fit_users + probe_users]
    ]
    test = [samples[index] for index in order[fit_users + probe_users :]]
    return fit, probe, test


def program_payload(
    program: MigrationProgram,
    fit: dict,
    protocol: dict,
) -> dict:
    return {
        "protocol": SYNC_DESIGN_PROTOCOL,
        "source_version": program.source_version,
        "target_version": program.target_version,
        "weights": program.adapter.weights.detach().cpu(),
        "biases": program.adapter.biases.detach().cpu(),
        "source_rank": program.adapter.source_rank,
        "ridge": program.adapter.ridge,
        "fit": fit,
        "evaluation_protocol": protocol,
    }


def cache_output_metrics(
    model: HSTU,
    cache: HSTUKVCache,
    suffix: dict,
    candidate_ids: torch.Tensor,
    fresh_hidden: torch.Tensor,
    fresh_scores: torch.Tensor,
    selected: list[dict],
) -> tuple[list[dict], list[float], list[float], list[float]]:
    hidden, _ = model.forward_with_cache(
        cache,
        suffix["item_ids"],
        suffix["behaviors"],
        suffix["time_deltas"],
    )
    hidden = hidden[:, 0]
    scores = model.item_emb.score(hidden, candidate_ids)
    metrics = [
        ranking_metrics(scores[row], selected[row]["pos_items"])
        for row in range(len(selected))
    ]
    hidden_cosine = torch.nn.functional.cosine_similarity(
        hidden.float(),
        fresh_hidden.float(),
        dim=-1,
    )
    score_cosine = torch.nn.functional.cosine_similarity(
        scores.float(),
        fresh_scores.float(),
        dim=-1,
    )
    topk = min(100, scores.shape[1])
    fresh_top = torch.topk(fresh_scores, topk, dim=1).indices
    action_top = torch.topk(scores, topk, dim=1).indices
    overlap = (
        (action_top.unsqueeze(2) == fresh_top.unsqueeze(1))
        .any(dim=2)
        .float()
        .mean(dim=1)
    )
    return (
        metrics,
        hidden_cosine.cpu().tolist(),
        score_cosine.cpu().tolist(),
        overlap.cpu().tolist(),
    )


@torch.inference_mode()
def evaluate_local(
    current: HSTU,
    old: HSTU,
    compiled,
    samples: list[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict]:
    names = [
        "reuse",
        "projection_only",
        f"compiled_rank_{args.rank}",
        *(f"residual_p{depth}" for depth in args.residual_depths),
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
        action_builders = {
            "reuse": lambda old_state=old_state: old_state.kv,
            "projection_only": lambda old_state=old_state: migrate_fused_projection_cache(
                current,
                old_state,
            ),
            f"compiled_rank_{args.rank}": lambda old_state=old_state: (
                migrate_compiled_low_rank_cache(old_state, compiled)
            ),
            **{
                f"residual_p{depth}": (
                    lambda depth=depth, old_state=old_state, prefix=prefix: migrate_prefix_residual_cache(
                        current,
                        old_state,
                        prefix["item_ids"],
                        prefix["behaviors"],
                        prefix["time_deltas"],
                        depth,
                    )
                )
                for depth in args.residual_depths
            },
            "recompute": lambda prefix=prefix: current.compute_kv(
                prefix["item_ids"],
                prefix["behaviors"],
                prefix["time_deltas"],
                lengths=prefix["lengths"],
            ),
        }
        batch_values = {
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
                    action_builders[name],
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
            batch_values["configs"][name] = {
                "metrics": metrics,
                "cache_error_rel": errors,
                "hidden_cosine": hidden_cosine,
                "score_cosine": score_cosine,
                "top100_overlap": top100_overlap,
            }
            if name not in {"reuse", "recompute"}:
                del cache
        recompute_values = batch_values["configs"]["recompute"]
        for row, sample in enumerate(selected):
            configs = {}
            for name in names:
                values = batch_values["configs"][name]
                configs[name] = {
                    "metrics": values["metrics"][row],
                    "cache_error_rel": float(values["cache_error_rel"][row]),
                    "hidden_cosine": float(values["hidden_cosine"][row]),
                    "score_cosine": float(values["score_cosine"][row]),
                    "top100_overlap": float(values["top100_overlap"][row]),
                }
            records.append(
                {
                    "user_id": int(sample["history"]["user_id"]),
                    "history_length": int(full["lengths"][row].item()),
                    "fresh": batch_values["fresh"][row],
                    "configs": configs,
                    "recompute_fresh_metric_max_abs": max(
                        abs(
                            recompute_values["metrics"][row][metric]
                            - batch_values["fresh"][row][metric]
                        )
                        for metric in batch_values["fresh"][row]
                    ),
                }
            )
        batches += 1
    return records, {
        "milliseconds": timing,
        "users": len(records),
        "batches": batches,
    }


def reduce_timing(
    timing: dict,
    runtime: DistributedRuntime,
) -> dict:
    names = list(timing["milliseconds"])
    values = torch.tensor(
        [
            *(timing["milliseconds"][name] for name in names),
            timing["users"],
            timing["batches"],
        ],
        dtype=torch.float64,
        device=runtime.device,
    )
    if runtime.initialized:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {
        "milliseconds": {
            name: float(values[index].item())
            for index, name in enumerate(names)
        },
        "users": int(values[-2].item()),
        "batches": int(values[-1].item()),
    }


def summarize(
    records: list[dict],
    timing: dict,
    args: argparse.Namespace,
) -> dict:
    names = list(records[0]["configs"])
    metrics = list(records[0]["fresh"])
    rng = np.random.default_rng(args.seed + 7717)
    reuse_error = np.asarray(
        [
            record["configs"]["reuse"]["cache_error_rel"]
            for record in records
        ]
    )
    configs = {}
    for name in names:
        cache_error = np.asarray(
            [record["configs"][name]["cache_error_rel"] for record in records]
        )
        quality = {}
        for metric in metrics:
            fresh = np.asarray([record["fresh"][metric] for record in records])
            value = np.asarray(
                [record["configs"][name]["metrics"][metric] for record in records]
            )
            reuse = np.asarray(
                [
                    record["configs"]["reuse"]["metrics"][metric]
                    for record in records
                ]
            )
            loss = value - fresh if metric in LOWER_IS_BETTER else fresh - value
            gain = reuse - value if metric in LOWER_IS_BETTER else value - reuse
            quality[metric] = {
                "mean": float(value.mean()),
                "signed_loss_from_fresh": {
                    "mean": float(loss.mean()),
                    "ci95": bootstrap_interval(
                        loss,
                        rng,
                        args.bootstrap_samples,
                    ),
                },
                "absolute_deviation_from_fresh": float(
                    np.abs(value - fresh).mean()
                ),
                "signed_gain_over_reuse": float(gain.mean()),
            }
        users = max(timing["users"], 1)
        ms_per_user = timing["milliseconds"][name] / users
        configs[name] = {
            "quality": quality,
            "cache_error_rel": float(cache_error.mean()),
            "cache_fidelity_recovery": float(
                (reuse_error.mean() - cache_error.mean())
                / max(reuse_error.mean(), 1e-12)
            ),
            "hidden_cosine": float(
                np.mean(
                    [record["configs"][name]["hidden_cosine"] for record in records]
                )
            ),
            "score_cosine": float(
                np.mean(
                    [record["configs"][name]["score_cosine"] for record in records]
                )
            ),
            "top100_overlap": float(
                np.mean(
                    [record["configs"][name]["top100_overlap"] for record in records]
                )
            ),
            "migration_ms_per_user": ms_per_user,
        }
    recompute_ms = configs["recompute"]["migration_ms_per_user"]
    for values in configs.values():
        values["migration_ratio_to_recompute"] = (
            values["migration_ms_per_user"] / max(recompute_ms, 1e-12)
        )
    return {
        "n": len(records),
        "configs": configs,
        "timing": timing,
        "recompute_fresh_metric_max_abs": max(
            record["recompute_fresh_metric_max_abs"] for record in records
        ),
    }


def smoke_samples(count: int) -> list[dict]:
    return [
        {
            "history": {
                "item_ids": np.asarray([1, 2, 3, 4], dtype=np.int64),
                "behaviors": np.asarray([1, 2, 3, 2], dtype=np.int64),
                "time_deltas": np.asarray([0, 1, 2, 3], dtype=np.float32),
                "labels": np.asarray([0, 1, 1, 1], dtype=np.int64),
                "user_id": index + 1,
            },
            "pos_items": [5, 6],
        }
        for index in range(count)
    ]


def run_smoke(args: argparse.Namespace, runtime: DistributedRuntime) -> None:
    seed_everything(0)
    cfg = HSTUConfig(
        num_items=128,
        num_prediction_items=96,
        num_behaviors=9,
        hidden_size=32,
        num_layers=4,
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
    args.ridge = 1e-3
    args.max_fit_tokens = 32
    args.timing_repeats = 1
    args.bootstrap_samples = 10
    args.residual_depths = [1, 2, 3]
    samples = smoke_samples(max(12, runtime.world_size * 4))
    fit, _, test = split_samples(samples, 4, 2, 0)
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
    local = test[runtime.rank :: runtime.world_size]
    records, timing = evaluate_local(
        current,
        old,
        compiled,
        local,
        args,
        runtime.device,
    )
    timing = reduce_timing(timing, runtime)
    records = gather_records(records, runtime)
    if runtime.is_primary:
        assert records is not None
        result = summarize(records, timing, args)
        if result["recompute_fresh_metric_max_abs"] > 1e-4:
            raise RuntimeError("fresh and exact incremental paths differ")
        expected = {
            "reuse",
            "projection_only",
            "compiled_rank_2",
            "residual_p1",
            "residual_p2",
            "residual_p3",
            "recompute",
        }
        if set(result["configs"]) != expected:
            raise RuntimeError("smoke action library is incomplete")
        print(json.dumps({"status": "ok", "world_size": runtime.world_size}, indent=2))


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "batch_size",
        "max_eval_users",
        "fit_users",
        "probe_users",
        "max_fit_tokens",
        "rank",
        "timing_repeats",
        "bootstrap_samples",
    )
    invalid = [name for name in positive if getattr(args, name) < 1]
    if invalid:
        raise ValueError(f"positive values required for: {invalid}")
    if args.ridge < 0:
        raise ValueError("ridge must be nonnegative")
    if not args.cache_versions:
        raise ValueError("at least one cache version is required")
    if len(set(args.cache_versions)) != len(args.cache_versions):
        raise ValueError("cache versions must be unique")
    if any(version < 0 or version >= args.current_version for version in args.cache_versions):
        raise ValueError("cache versions must precede the current version")


def formal_configuration(args: argparse.Namespace) -> bool:
    return (
        args.base_days == 4
        and args.current_version == 11
        and args.cache_versions == [0, 4, 10]
        and args.batch_size == 4
        and args.max_eval_users == 1000
        and args.fit_users == 40
        and args.probe_users == 60
        and args.max_fit_tokens == 8192
        and args.rank == 32
        and args.ridge == 1e-3
        and args.residual_depths == [4, 8, 12]
        and args.timing_repeats == 3
        and args.bootstrap_samples == 1000
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    runtime = init_distributed_runtime(args.device, args.distributed_backend)
    args.device = str(runtime.device)
    try:
        if args.smoke_test:
            run_smoke(args, runtime)
            return
        if runtime.device.type != "cuda":
            raise ValueError("formal design timing requires CUDA")
        formal = formal_configuration(args)
        if not formal and not args.allow_diagnostic_protocol:
            raise ValueError(
                "formal design evaluation requires every frozen default; use "
                "--allow-diagnostic-protocol only for altered configurations"
            )
        source = json.loads(Path(args.training_result).read_text())
        expected_protocol = training_protocol_for_base_days(args.base_days)
        if source.get("protocol") != expected_protocol:
            raise ValueError("training result protocol does not match the requested split")
        if source.get("status") != "complete":
            raise ValueError("training result is incomplete")
        actual_hash = sha256(args.prepared_data)
        if actual_hash != source["prepared_data"]["sha256"]:
            raise ValueError("prepared data differs from the training artifact")
        plan, metadata = load_prepared_kuairand_plan(args.prepared_data)
        validate_long_context_plan(plan, metadata, args.base_days)
        cfg = HSTUConfig(**source["model"])
        args.seq_len = cfg.max_seq_len
        args.seed = int(source["args"]["seed"])
        if max(args.residual_depths) >= cfg.num_layers:
            raise ValueError("residual depths must be smaller than model depth")
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
        pair_results = []
        protocol_record = {
            "base_days": args.base_days,
            "current_version": args.current_version,
            "cache_versions": args.cache_versions,
            "rank": args.rank,
            "residual_depths": args.residual_depths,
            "split_seed": 9151 + args.seed,
            "labels_used_for_fit_or_action_selection": False,
            "action_policy": (
                "unconditional rank-32 synchronization, fixed p8 background "
                "refinement, exact endpoint"
            ),
        }
        program_dir = Path(args.program_dir)
        if runtime.is_primary:
            program_dir.mkdir(parents=True, exist_ok=True)
        if runtime.initialized:
            dist.barrier()
        for cache_version in args.cache_versions:
            old = load_checkpoint_model(
                cfg,
                args.checkpoint_dir,
                cache_version,
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
            if runtime.is_primary:
                program = MigrationProgram(
                    source_version=f"theta{cache_version}",
                    target_version=f"theta{args.current_version}",
                    adapter=compiled,
                )
                torch.save(
                    program_payload(program, fit_summary, protocol_record),
                    program_dir
                    / f"theta{cache_version}_to_theta{args.current_version}_rank{args.rank}.pt",
                )
            local_test = test_samples[runtime.rank :: runtime.world_size]
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
                pair_results.append(
                    {
                        "cache_version": cache_version,
                        "current_version": args.current_version,
                        "cache_age_updates": args.current_version - cache_version,
                        "fit": fit_summary,
                        "summary": summarize(records, timing, args),
                        "per_user": records,
                    }
                )
                selected = pair_results[-1]["summary"]["configs"][
                    f"compiled_rank_{args.rank}"
                ]
                primary_log(
                    runtime,
                    f"theta{cache_version}->theta{args.current_version} "
                    f"compiled={selected['migration_ratio_to_recompute']:.3f}x "
                    f"recovery={selected['cache_fidelity_recovery']:.3f}",
                )
            del old
            torch.cuda.empty_cache()
        if runtime.is_primary:
            result = {
                "protocol": SYNC_DESIGN_PROTOCOL,
                "status": "complete" if formal else "diagnostic_complete",
                "formal_protocol": formal,
                "source_training_result": args.training_result,
                "prepared_data": {
                    "path": args.prepared_data,
                    "sha256": actual_hash,
                    "metadata": metadata,
                },
                "checkpoint_dir": args.checkpoint_dir,
                "program_dir": args.program_dir,
                "seed": args.seed,
                "world_size": runtime.world_size,
                "worker_count_semantics": (
                    "evaluation-only data parallelism; worker count changes wall "
                    "time but not the frozen users, actions, or statistical unit"
                ),
                "eval_date": date,
                "model": source["model"],
                "protocol_record": protocol_record,
                "split": {
                    "fit_users": len(fit_samples),
                    "probe_users": len(probe_samples),
                    "test_users": len(test_samples),
                    "probe_role": (
                        "reserved for fixed-tier calibration; no per-version "
                        "reuse admission or task-label selection"
                    ),
                },
                "state_footprint": {
                    "all": prefix_state_footprint(samples, cfg),
                    "test": prefix_state_footprint(test_samples, cfg),
                },
                "pairs": pair_results,
            }
            save_json(result, args.output)
            print(args.output, flush=True)
    finally:
        close_distributed_runtime(runtime)


if __name__ == "__main__":
    main()
