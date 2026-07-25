"""Evaluate every older-cache/current-model pair in a long-context split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from motivation_validity import (
    bootstrap_interval,
    evaluate_version_pair,
    seed_everything,
    summarize_records,
)

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming import (
    SUPPORTED_LONG_CONTEXT_BASE_DAYS,
    DistributedRuntime,
    close_distributed_runtime,
    gather_records,
    init_distributed_runtime,
    load_checkpoint_model,
    long_context_split_name,
    motivation_protocol_for_base_days,
    prefix_state_footprint,
    primary_log,
    reconstruct_online_eval_samples,
    training_protocol_for_base_days,
    validate_long_context_plan,
)
from hstu_kvcache.utils import save_json

DEFAULT_TRAINING_RESULT = (
    "results/motivation_scale/long_context_8plus8_training_seed0.json"
)
DEFAULT_CHECKPOINT_DIR = "checkpoints/kuairand_long_context_8plus8/seed0"
DEFAULT_OUTPUT = (
    "results/motivation_scale/long_context_8plus8_motivation_all_pairs_seed0.json"
)
DEFAULT_PREPARED_DATA = "data/processed/kuairand_long_context_8plus8_v2.npz"
LOWER_IS_BETTER = {"best_rank", "mean_rank", "median_rank"}
AGE_STEP_METRICS = ("mean_rank", "auc", "best_rank", "ndcg@100", "hit@100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-days",
        type=int,
        choices=SUPPORTED_LONG_CONTEXT_BASE_DAYS,
        default=8,
    )
    parser.add_argument("--prepared-data")
    parser.add_argument("--training-result")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributed-backend", default="nccl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-eval-users", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--versions",
        type=int,
        nargs="+",
    )
    parser.add_argument("--distributed-smoke-test", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    split = long_context_split_name(args.base_days)
    exploration = "_exploration" if args.base_days != 8 else ""
    if args.prepared_data is None:
        args.prepared_data = (
            DEFAULT_PREPARED_DATA
            if args.base_days == 8
            else (
                f"data/processed/kuairand_long_context_{split}"
                "_exploration_v1.npz"
            )
        )
    if args.training_result is None:
        args.training_result = (
            DEFAULT_TRAINING_RESULT
            if args.base_days == 8 and args.seed == 0
            else (
                f"results/motivation_scale/long_context_{split}_training"
                f"{exploration}_seed{args.seed}.json"
            )
        )
    if args.checkpoint_dir is None:
        args.checkpoint_dir = (
            DEFAULT_CHECKPOINT_DIR
            if args.base_days == 8 and args.seed == 0
            else (
                f"checkpoints/kuairand_long_context_{split}"
                f"{exploration}/seed{args.seed}"
            )
        )
    if args.output is None:
        args.output = (
            DEFAULT_OUTPUT
            if args.base_days == 8 and args.seed == 0
            else (
                f"results/motivation_scale/long_context_{split}"
                f"_motivation_all_pairs{exploration}_seed{args.seed}.json"
            )
        )


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_comparison(
    current: HSTU,
    old: HSTU,
    samples: list[dict],
    current_version: int,
    cache_version: int,
    date: str,
    role: str,
    args: argparse.Namespace,
    runtime: DistributedRuntime,
) -> dict | None:
    local_samples = samples[runtime.rank::runtime.world_size]
    local_samples.sort(key=lambda sample: len(sample["history"]["item_ids"]))
    local_records, _ = evaluate_version_pair(
        current,
        old,
        local_samples,
        args,
        current_version,
        summarize=False,
    )
    for record in local_records:
        record["eval_date"] = date
        record["current_version"] = current_version
        record["cache_version"] = cache_version
        record["cache_age_updates"] = current_version - cache_version
        record["role"] = role
    records = gather_records(local_records, runtime)
    if not runtime.is_primary:
        return None
    assert records is not None
    summary = summarize_records(
        records,
        np.random.default_rng(args.seed * 1000 + current_version),
        args.bootstrap_samples,
    )
    return {
        "eval_date": date,
        "current_version": current_version,
        "cache_version": cache_version,
        "cache_age_updates": current_version - cache_version,
        "role": role,
        "fresh_definition": "complete current-model forward on the available history",
        "reuse_definition": (
            f"theta{cache_version} prefix K/V on the same input, plus the latest "
            f"token under theta{current_version}"
        ),
        "eligible_users": len(samples),
        "resident_context": prefix_state_footprint(samples, current.cfg),
        "summary": summary,
        "per_user": records,
    }


def metric_loss_values(comparison: dict, metric: str) -> np.ndarray:
    records = comparison["per_user"]
    fresh = np.asarray([record[f"fresh_{metric}"] for record in records])
    stale = np.asarray([record[f"stale_{metric}"] for record in records])
    return stale - fresh if metric in LOWER_IS_BETTER else fresh - stale


def triangular_version_pairs(
    versions: tuple[int, ...],
) -> list[tuple[int, int]]:
    return [
        (current_version, cache_version)
        for current_version in versions
        for cache_version in range(current_version)
    ]


def summarize_triangular_age_steps(
    comparisons: list[dict],
    seed: int,
    bootstrap_samples: int,
) -> dict:
    rows = []
    all_steps = []
    current_versions = sorted(
        {comparison["current_version"] for comparison in comparisons}
    )
    for current_version in current_versions:
        ordered = sorted(
            [
                comparison
                for comparison in comparisons
                if comparison["current_version"] == current_version
            ],
            key=lambda value: value["cache_age_updates"],
        )
        user_ids = [
            [record["user_id"] for record in comparison["per_user"]]
            for comparison in ordered
        ]
        if any(values != user_ids[0] for values in user_ids[1:]):
            raise RuntimeError(
                f"theta{current_version} comparisons do not contain paired users"
            )
        rng = np.random.default_rng(seed + 77291 + current_version * 1009)
        previous_losses = {
            metric: np.zeros(len(user_ids[0]), dtype=np.float64)
            for metric in AGE_STEP_METRICS
        }
        previous_cache_version = current_version
        previous_age = 0
        steps = []
        for older in ordered:
            metrics = {}
            for metric in AGE_STEP_METRICS:
                loss = metric_loss_values(older, metric)
                increment = loss - previous_losses[metric]
                metrics[metric] = {
                    "mean_additional_loss": float(increment.mean()),
                    "ci95": bootstrap_interval(
                        increment,
                        rng,
                        bootstrap_samples,
                    ),
                    "worsening_fraction": float(np.mean(increment > 0)),
                }
                previous_losses[metric] = loss
            step = {
                "current_version": current_version,
                "eval_date": older["eval_date"],
                "newer_cache_version": previous_cache_version,
                "older_cache_version": older["cache_version"],
                "cache_age_from": previous_age,
                "cache_age_to": older["cache_age_updates"],
                "metrics": metrics,
            }
            steps.append(step)
            all_steps.append(step)
            previous_cache_version = older["cache_version"]
            previous_age = older["cache_age_updates"]
        largest_index = int(
            np.argmax(
                [
                    step["metrics"]["mean_rank"]["mean_additional_loss"]
                    for step in steps
                ]
            )
        )
        rows.append(
            {
                "current_version": current_version,
                "eval_date": ordered[0]["eval_date"],
                "steps": steps,
                "largest_mean_rank_step": steps[largest_index],
            }
        )
    largest_global_index = int(
        np.argmax(
            [
                step["metrics"]["mean_rank"]["mean_additional_loss"]
                for step in all_steps
            ]
        )
    )
    return {
        "definition": (
            "within each fixed current-model/date row, paired additional quality "
            "loss when the cache moves one checkpoint farther back"
        ),
        "statistical_role": (
            "within-seed paired user bootstrap is diagnostic; the training seed "
            "remains the replication unit"
        ),
        "rows": rows,
        "largest_mean_rank_step_across_matrix": all_steps[largest_global_index],
    }


def summarize_fixed_cache_trajectories(comparisons: list[dict]) -> dict:
    trajectories = []
    cache_versions = sorted(
        {comparison["cache_version"] for comparison in comparisons}
    )
    for cache_version in cache_versions:
        ordered = sorted(
            [
                comparison
                for comparison in comparisons
                if comparison["cache_version"] == cache_version
            ],
            key=lambda value: value["current_version"],
        )
        points = []
        for comparison in ordered:
            quality_losses = {
                metric: float(summary["gain"])
                for metric, summary in comparison["summary"].items()
                if isinstance(summary, dict) and "gain" in summary
            }
            fidelity = comparison["summary"]["fidelity"]
            fidelity_losses = {
                "kv_drift_rel": float(fidelity["kv_drift_rel"]),
                "hidden_cosine_loss": 1.0 - float(fidelity["hidden_cosine"]),
                "score_cosine_loss": 1.0 - float(fidelity["score_cosine"]),
                "top10_changed_fraction": float(
                    fidelity["top10_changed_fraction"]
                ),
            }
            points.append(
                {
                    "current_version": comparison["current_version"],
                    "eval_date": comparison["eval_date"],
                    "cache_age_updates": comparison["cache_age_updates"],
                    "quality_losses": quality_losses,
                    "fidelity_losses": fidelity_losses,
                }
            )
        steps = []
        for previous, current in zip(points, points[1:], strict=False):
            steps.append(
                {
                    "current_version_from": previous["current_version"],
                    "current_version_to": current["current_version"],
                    "eval_date_from": previous["eval_date"],
                    "eval_date_to": current["eval_date"],
                    "cache_age_from": previous["cache_age_updates"],
                    "cache_age_to": current["cache_age_updates"],
                    "additional_quality_loss": {
                        metric: current["quality_losses"][metric]
                        - previous["quality_losses"][metric]
                        for metric in current["quality_losses"]
                    },
                    "additional_fidelity_loss": {
                        metric: current["fidelity_losses"][metric]
                        - previous["fidelity_losses"][metric]
                        for metric in current["fidelity_losses"]
                    },
                }
            )
        candidate_late_jumps = {}
        if len(steps) >= 3:
            for metric in points[0]["quality_losses"]:
                eligible = []
                for index, step in enumerate(steps[2:], start=2):
                    previous_increments = np.asarray(
                        [
                            prior["additional_quality_loss"][metric]
                            for prior in steps[:index]
                        ],
                        dtype=np.float64,
                    )
                    increment = step["additional_quality_loss"][metric]
                    baseline = float(np.median(np.abs(previous_increments)))
                    eligible.append(
                        {
                            "current_version_from": step["current_version_from"],
                            "current_version_to": step["current_version_to"],
                            "cache_age_from": step["cache_age_from"],
                            "cache_age_to": step["cache_age_to"],
                            "additional_quality_loss": increment,
                            "previous_step_median_abs": baseline,
                            "jump_ratio": (
                                increment / max(baseline, 1e-12)
                            ),
                        }
                    )
                candidate = max(
                    eligible,
                    key=lambda value: value["additional_quality_loss"],
                )
                candidate_late_jumps[metric] = (
                    candidate
                    if candidate["additional_quality_loss"] > 0
                    else None
                )
        trajectories.append(
            {
                "cache_version": cache_version,
                "pair_keys": [
                    {
                        "current_version": point["current_version"],
                        "cache_version": cache_version,
                    }
                    for point in points
                ],
                "points": points,
                "successive_model_update_steps": steps,
                "candidate_late_jumps": candidate_late_jumps,
            }
        )
    return {
        "definition": (
            "hold the cache-encoding checkpoint theta_j fixed and advance the "
            "deployed current model theta_(j+1), theta_(j+2), and so on"
        ),
        "cliff_screen": (
            "for every reported quality metric, select the largest positive "
            "successive loss increment after at least two earlier transitions"
        ),
        "statistical_caveat": (
            "successive points use different leak-free next-day endpoints, so "
            "jump ratios are exploratory diagnostics rather than paired tests; "
            "training seeds remain the replication unit"
        ),
        "trajectories": trajectories,
    }


def smoke_sample(user_id: int) -> dict:
    return {
        "history": {
            "item_ids": np.asarray([1, 2, 3, 4], dtype=np.int64),
            "behaviors": np.asarray([1, 2, 3, 2], dtype=np.int64),
            "time_deltas": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
            "labels": np.asarray([0, 1, 1, 1], dtype=np.int64),
            "user_id": user_id,
        },
        "pos_items": [5, 6],
    }


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
    args.device = str(runtime.device)
    args.seq_len = 8
    args.seed = 0
    local_records, _ = evaluate_version_pair(
        current,
        old,
        [smoke_sample(runtime.rank + 1)],
        args,
        1,
        summarize=False,
    )
    records = gather_records(local_records, runtime)
    if runtime.is_primary:
        assert records is not None
        last_evaluable_version = 15 - args.base_days
        pairs = triangular_version_pairs(
            tuple(range(1, last_evaluable_version + 1))
        )
        expected_pair_count = (
            last_evaluable_version * (last_evaluable_version + 1) // 2
        )
        if (
            len(pairs) != expected_pair_count
            or pairs[0] != (1, 0)
            or pairs[-1]
            != (last_evaluable_version, last_evaluable_version - 1)
        ):
            raise RuntimeError("triangular version-pair construction is invalid")
        if len(records) != runtime.world_size:
            raise RuntimeError("distributed evaluation did not gather every rank")
        if max(record["incremental_parity_max_abs"] for record in records) > 1e-4:
            raise RuntimeError("fresh full and incremental paths failed parity")
        required = {
            "fresh_mrr",
            "stale_mrr",
            "fresh_ndcg@100",
            "stale_ndcg@100",
            "fresh_auc",
            "stale_auc",
            "fresh_recall@20",
            "stale_recall@20",
        }
        if not required.issubset(records[0]):
            raise RuntimeError("distributed evaluation omitted ranking metrics")
        print(
            json.dumps(
                {
                    "world_size": runtime.world_size,
                    "records": len(records),
                    "formal_pair_count": len(pairs),
                    "metrics_checked": sorted(required),
                    "status": "ok",
                },
                indent=2,
            ),
            flush=True,
        )


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    if args.batch_size != 4 or args.max_eval_users != 1000:
        raise ValueError("the frozen motivation protocol requires batch 4 and user cap 1000")
    if args.bootstrap_samples != 1000:
        raise ValueError("the frozen motivation protocol requires 1000 bootstrap samples")
    training_protocol = training_protocol_for_base_days(args.base_days)
    motivation_protocol = motivation_protocol_for_base_days(args.base_days)
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
            raise ValueError("formal long-context evaluation requires exactly four workers")
        source = json.loads(Path(args.training_result).read_text())
        if source.get("protocol") != training_protocol:
            raise ValueError("training result protocol does not match")
        if source.get("status") != "complete":
            raise ValueError("training result is incomplete")
        expected_hash = source["prepared_data"]["sha256"]
        actual_hash = artifact_sha256(args.prepared_data)
        if actual_hash != expected_hash:
            raise ValueError("prepared data differs from the training artifact")
        plan, prepared_metadata = load_prepared_kuairand_plan(args.prepared_data)
        validate_long_context_plan(plan, prepared_metadata, args.base_days)
        last_evaluable_version = len(plan.stream_dates) - 1
        expected_versions = tuple(range(1, last_evaluable_version + 1))
        versions = (
            expected_versions
            if args.versions is None
            else tuple(sorted(set(args.versions)))
        )
        if versions != expected_versions:
            raise ValueError(
                "the selected split requires every evaluable version "
                f"{list(expected_versions)}"
            )
        cfg = HSTUConfig(**source["model"])
        if cfg.num_items != plan.num_items:
            raise ValueError("model context vocabulary and prepared data differ")
        args.seq_len = cfg.max_seq_len
        source_seed = int(source["args"]["seed"])
        if args.seed != source_seed:
            raise ValueError(
                f"evaluation seed {args.seed} differs from training seed {source_seed}"
            )
        samples = reconstruct_online_eval_samples(
            plan,
            versions,
            args.max_eval_users,
        )
        expected_pairs = triangular_version_pairs(versions)
        comparisons = []
        for current_version in versions:
            date, version_samples = samples[current_version]
            current = load_checkpoint_model(
                cfg,
                args.checkpoint_dir,
                current_version,
                runtime.device,
            )
            for cache_version in range(current_version):
                old = load_checkpoint_model(
                    cfg,
                    args.checkpoint_dir,
                    cache_version,
                    runtime.device,
                )
                role = (
                    "primary_oldest_cache"
                    if (
                        current_version == last_evaluable_version
                        and cache_version == 0
                    )
                    else "all_pairs_cache_matrix"
                )
                comparison = evaluate_comparison(
                    current,
                    old,
                    version_samples,
                    current_version,
                    cache_version,
                    date,
                    role,
                    args,
                    runtime,
                )
                if runtime.is_primary:
                    assert comparison is not None
                    comparisons.append(comparison)
                    primary_log(
                        runtime,
                        f"theta={current_version} cache=theta{cache_version} "
                        f"date={date} age={current_version - cache_version} "
                        f"users={comparison['eligible_users']} "
                        f"mean_rank_loss="
                        f"{comparison['summary']['mean_rank']['gain']:.3f} "
                        f"auc_loss={comparison['summary']['auc']['gain']:.6f}",
                    )
                del old
            del current
        if runtime.is_primary:
            actual_pairs = [
                (
                    comparison["current_version"],
                    comparison["cache_version"],
                )
                for comparison in comparisons
            ]
            if actual_pairs != expected_pairs:
                raise RuntimeError(
                    f"all-pairs matrix mismatch: expected {expected_pairs}, "
                    f"received {actual_pairs}"
                )
            primary = next(
                comparison
                for comparison in comparisons
                if comparison["current_version"] == last_evaluable_version
                and comparison["cache_version"] == 0
            )
            age_steps = summarize_triangular_age_steps(
                comparisons,
                args.seed,
                args.bootstrap_samples,
            )
            fixed_cache_trajectories = summarize_fixed_cache_trajectories(
                comparisons
            )
            result = {
                "protocol": motivation_protocol,
                "source_training_result": args.training_result,
                "prepared_data": {
                    "path": args.prepared_data,
                    "sha256": actual_hash,
                    "metadata": prepared_metadata,
                },
                "checkpoint_dir": args.checkpoint_dir,
                "seed": args.seed,
                "world_size": runtime.world_size,
                "versions": list(versions),
                "comparison_count": len(comparisons),
                "expected_comparison_count": sum(versions),
                "metric_direction": (
                    "positive gain is the quality lost by stale cache reuse "
                    "relative to fresh current-model inference"
                ),
                "metric_selection": {
                    "primary": "mean_rank",
                    "robust_secondary": "auc",
                    "standard_secondaries": ["ndcg@100", "hit@100"],
                    "selection_status": (
                        "frozen from the superseded 12+4 seed-0 exploratory run "
                        "before observing the current split"
                    ),
                },
                "cache_age_semantics": (
                    "checkpoint-update distance used to encode the same resident "
                    "prefix; it is not physical snapshot residence time"
                ),
                "physical_cache_lifecycle_scope": (
                    "literal snapshot survival, rolling eviction, and organically "
                    "mixed per-token versions are not claimed by this controlled "
                    "motivation protocol"
                ),
                "matrix_definition": (
                    "for every evaluable current theta_i on its next unseen date, "
                    "evaluate every older cache theta_j where j < i"
                ),
                "matrix_rows": [
                    {
                        "current_version": current_version,
                        "eval_date": samples[current_version][0],
                        "cache_versions": list(range(current_version)),
                        "pair_count": current_version,
                    }
                    for current_version in versions
                ],
                "theta0_moving_curve_pair_keys": [
                    {
                        "current_version": current_version,
                        "cache_version": 0,
                    }
                    for current_version in versions
                ],
                "last_current_fixed_endpoint_pair_keys": [
                    {
                        "current_version": last_evaluable_version,
                        "cache_version": cache_version,
                    }
                    for cache_version in range(last_evaluable_version)
                ],
                "fresh_reference_definition": (
                    "every comparison already contains a complete current-model "
                    "fresh path on the same users and history; diagonal self-pairs "
                    "are therefore omitted"
                ),
                "comparisons": comparisons,
                "adjacent_age_steps_by_current": age_steps,
                "fixed_cache_trajectories": fixed_cache_trajectories,
                "primary_contrast": {
                    "current_version": last_evaluable_version,
                    "cache_version": 0,
                    "eval_date": primary["eval_date"],
                    "summary": primary["summary"],
                    "resident_context": primary["resident_context"],
                },
            }
            save_json(result, args.output)
            print(args.output, flush=True)
    finally:
        close_distributed_runtime(runtime)


if __name__ == "__main__":
    main()
