from __future__ import annotations

import argparse
import json
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
from layerwise_validity import timed_call
from motivation_validity import eval_batches, move_batch, ranking_metrics, seed_everything
from search_kuairand_long_context_attention_weighted import (
    PROTOCOL as ATTENTION_PROGRAM_PROTOCOL,
)

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.migration import (
    CompiledCacheAdapter,
    FidelityContract,
    MigrationActionSpec,
    capture_layerwise_state,
    compile_verified_plan,
    migrate_compiled_low_rank_cache,
    migrate_fused_projection_cache,
    migrate_prefix_residual_cache,
    sample_relative_cache_error,
)
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from hstu_kvcache.streaming import (
    close_distributed_runtime,
    gather_records,
    init_distributed_runtime,
    load_checkpoint_model,
    reconstruct_online_eval_samples,
    training_protocol_for_base_days,
    validate_long_context_plan,
)
from hstu_kvcache.utils import save_json

PROTOCOL = "kuairand_long_context_4plus12_verified_compiler_v1"
DEFAULT_PROGRAM_RESULT = (
    "results/motivation_scale/"
    "long_context_4plus12_attention_weighted_search_seed0.json"
)
DEFAULT_PROGRAM_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/"
    "seed0/attention_weighted_search"
)
DEFAULT_MANIFEST_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/"
    "seed0/verified_plans"
)
DEFAULT_OUTPUT = (
    "results/motivation_scale/"
    "long_context_4plus12_verified_compiler_seed0.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--program-result", default=DEFAULT_PROGRAM_RESULT)
    parser.add_argument("--program-dir", default=DEFAULT_PROGRAM_DIR)
    parser.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributed-backend", default="nccl")
    parser.add_argument("--base-days", type=int, default=4)
    parser.add_argument("--current-version", type=int, default=11)
    parser.add_argument("--cache-versions", type=int, nargs="+", default=[0, 4, 10])
    parser.add_argument("--structural-depths", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-eval-users", type=int, default=1000)
    parser.add_argument("--fit-users", type=int, default=40)
    parser.add_argument("--selection-users", type=int, default=60)
    parser.add_argument("--certificate-users", type=int, default=60)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--recovery-target", type=float, default=0.7)
    parser.add_argument("--minimum-coverage", type=float, default=0.8)
    parser.add_argument("--confidence-level", type=float, default=0.9)
    parser.add_argument("--max-cost-ratio", type=float, default=0.3)
    parser.add_argument("--minimum-valid-users", type=int, default=50)
    return parser.parse_args()


def split_certificate_test(
    samples: list[dict],
    certificate_users: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if not 0 < certificate_users < len(samples):
        raise ValueError("certificate split must leave final test users")
    order = np.random.default_rng(27183 + seed).permutation(len(samples))
    certificate = [samples[index] for index in order[:certificate_users]]
    test = [samples[index] for index in order[certificate_users:]]
    return certificate, test


def program_path(args: argparse.Namespace, cache_version: int) -> Path:
    return Path(args.program_dir) / (
        f"theta{cache_version}_to_theta{args.current_version}_"
        "compiled_attention_mix_1.00.pt"
    )


def load_compiled_program(
    path: Path,
    cache_version: int,
    args: argparse.Namespace,
    cfg: HSTUConfig,
    device: torch.device,
) -> tuple[CompiledCacheAdapter, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != ATTENTION_PROGRAM_PROTOCOL:
        raise ValueError("compiled program protocol mismatch")
    if payload.get("source_version") != f"theta{cache_version}":
        raise ValueError("compiled program source version mismatch")
    if payload.get("target_version") != f"theta{args.current_version}":
        raise ValueError("compiled program target version mismatch")
    weights = payload["weights"]
    biases = payload["biases"]
    expected_width = 2 * cfg.num_heads * cfg.head_dim
    if weights.shape != (
        cfg.num_layers,
        cfg.hidden_size,
        expected_width,
    ):
        raise ValueError("compiled program weight shape mismatch")
    if biases.shape != (cfg.num_layers, expected_width):
        raise ValueError("compiled program bias shape mismatch")
    fit = payload.get("fit", {})
    if (
        fit.get("fit_users") != args.fit_users
        or fit.get("labels_used") is not False
    ):
        raise ValueError("compiled program fit provenance mismatch")
    return (
        CompiledCacheAdapter(
            weights=weights.to(device),
            biases=biases.to(device),
            source_rank=cfg.hidden_size,
            ridge=float(payload["ridge"]),
        ),
        payload,
    )


def action_specs(
    path: Path,
    depths: list[int],
) -> tuple[MigrationActionSpec, ...]:
    return (
        MigrationActionSpec(
            name="projection_only",
            kind="projection",
            required_state="normalized_capsule",
        ),
        MigrationActionSpec(
            name="compiled_full_affine",
            kind="compiled",
            required_state="normalized_capsule",
            program_path=str(path),
        ),
        *(
            MigrationActionSpec(
                name=f"structural_p{depth}",
                kind="structural_replay",
                required_state="history_and_layerwise_capsule",
                replay_depth=depth,
            )
            for depth in depths
        ),
        MigrationActionSpec(
            name="recompute",
            kind="exact",
            required_state="raw_history",
        ),
    )


@torch.inference_mode()
def semantic_output(
    model: HSTU,
    cache: HSTUKVCache,
    suffix: dict,
    candidate_ids: torch.Tensor,
    fresh_hidden: torch.Tensor,
    fresh_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden, _ = model.forward_with_cache(
        cache,
        suffix["item_ids"],
        suffix["behaviors"],
        suffix["time_deltas"],
    )
    hidden = hidden[:, 0]
    scores = model.item_emb.score(hidden, candidate_ids)
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
    return hidden_cosine, score_cosine, overlap, scores


def build_actions(
    current: HSTU,
    old_state,
    prefix: dict,
    compiled: CompiledCacheAdapter,
    names: list[str],
) -> dict[str, callable]:
    builders = {
        "reuse": lambda old_state=old_state: old_state.kv,
        "projection_only": lambda old_state=old_state: migrate_fused_projection_cache(
            current,
            old_state,
        ),
        "compiled_full_affine": lambda old_state=old_state: (
            migrate_compiled_low_rank_cache(old_state, compiled)
        ),
        "recompute": lambda prefix=prefix: current.compute_kv(
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            lengths=prefix["lengths"],
        ),
    }
    for name in names:
        if name.startswith("structural_p"):
            depth = int(name.removeprefix("structural_p"))
            builders[name] = (
                lambda depth=depth, old_state=old_state, prefix=prefix: (
                    migrate_prefix_residual_cache(
                        current,
                        old_state,
                        prefix["item_ids"],
                        prefix["behaviors"],
                        prefix["time_deltas"],
                        depth,
                    )
                )
            )
    return builders


@torch.inference_mode()
def evaluate_actions_local(
    current: HSTU,
    old: HSTU,
    compiled: CompiledCacheAdapter,
    samples: list[dict],
    names: list[str],
    args: argparse.Namespace,
    device: torch.device,
    include_quality: bool,
) -> tuple[list[dict], dict]:
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
        builders = build_actions(
            current,
            old_state,
            prefix,
            compiled,
            names,
        )
        fresh_metrics = None
        if include_quality:
            fresh_metrics = [
                ranking_metrics(
                    fresh_scores[row],
                    selected[row]["pos_items"],
                )
                for row in range(len(selected))
            ]
        values = {}
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
            hidden_cosine, score_cosine, overlap, scores = semantic_output(
                current,
                cache,
                suffix,
                candidate_ids,
                fresh_hidden,
                fresh_scores,
            )
            metrics = None
            if include_quality:
                metrics = [
                    ranking_metrics(
                        scores[row],
                        selected[row]["pos_items"],
                    )
                    for row in range(len(selected))
                ]
            values[name] = {
                "metrics": metrics,
                "cache_error_rel": sample_relative_cache_error(
                    cache,
                    fresh_cache,
                )
                .cpu()
                .tolist(),
                "hidden_cosine": hidden_cosine.cpu().tolist(),
                "score_cosine": score_cosine.cpu().tolist(),
                "top100_overlap": overlap.cpu().tolist(),
            }
            if name not in {"reuse", "recompute"}:
                del cache
        for row, sample in enumerate(selected):
            record = {
                "user_id": int(sample["history"]["user_id"]),
                "history_length": int(full["lengths"][row].item()),
                "configs": {
                    name: {
                        "cache_error_rel": float(
                            values[name]["cache_error_rel"][row]
                        ),
                        "hidden_cosine": float(
                            values[name]["hidden_cosine"][row]
                        ),
                        "score_cosine": float(
                            values[name]["score_cosine"][row]
                        ),
                        "top100_overlap": float(
                            values[name]["top100_overlap"][row]
                        ),
                        **(
                            {
                                "metrics": values[name]["metrics"][row],
                            }
                            if include_quality
                            else {}
                        ),
                    }
                    for name in names
                },
            }
            if include_quality:
                assert fresh_metrics is not None
                record["fresh"] = fresh_metrics[row]
                record["recompute_fresh_metric_max_abs"] = max(
                    abs(
                        values["recompute"]["metrics"][row][metric]
                        - fresh_metrics[row][metric]
                    )
                    for metric in fresh_metrics[row]
                )
            records.append(record)
        batches += 1
    return records, {
        "milliseconds": timing,
        "users": len(records),
        "batches": batches,
    }


def semantic_summary(
    records: list[dict],
    timing: dict,
) -> dict:
    names = list(records[0]["configs"])
    users = max(timing["users"], 1)
    exact_ms = timing["milliseconds"]["recompute"] / users
    return {
        "n": len(records),
        "labels_used": False,
        "configs": {
            name: {
                "cache_error_rel": float(
                    np.mean(
                        [
                            record["configs"][name]["cache_error_rel"]
                            for record in records
                        ]
                    )
                ),
                "hidden_cosine": float(
                    np.mean(
                        [
                            record["configs"][name]["hidden_cosine"]
                            for record in records
                        ]
                    )
                ),
                "score_cosine": float(
                    np.mean(
                        [
                            record["configs"][name]["score_cosine"]
                            for record in records
                        ]
                    )
                ),
                "top100_overlap": float(
                    np.mean(
                        [
                            record["configs"][name]["top100_overlap"]
                            for record in records
                        ]
                    )
                ),
                "migration_ms_per_user": timing["milliseconds"][name] / users,
                "migration_ratio_to_recompute": (
                    timing["milliseconds"][name]
                    / users
                    / max(exact_ms, 1e-12)
                ),
            }
            for name in names
        },
        "timing": timing,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.base_days != 4 or args.current_version != 11:
        raise ValueError("verified compiler freezes theta11 under 4+12")
    if args.cache_versions != [0, 4, 10]:
        raise ValueError("verified compiler freezes cache versions")
    if args.structural_depths != [4, 8]:
        raise ValueError("verified compiler freezes structural depths")
    frozen = {
        "batch_size": (args.batch_size, 4),
        "max_eval_users": (args.max_eval_users, 1000),
        "fit_users": (args.fit_users, 40),
        "selection_users": (args.selection_users, 60),
        "certificate_users": (args.certificate_users, 60),
        "timing_repeats": (args.timing_repeats, 3),
        "bootstrap_samples": (args.bootstrap_samples, 1000),
        "recovery_target": (args.recovery_target, 0.7),
        "minimum_coverage": (args.minimum_coverage, 0.8),
        "confidence_level": (args.confidence_level, 0.9),
        "max_cost_ratio": (args.max_cost_ratio, 0.3),
        "minimum_valid_users": (args.minimum_valid_users, 50),
    }
    changed = {
        name: {"expected": expected, "actual": actual}
        for name, (actual, expected) in frozen.items()
        if actual != expected
    }
    if changed:
        raise ValueError(f"frozen verified compiler settings changed: {changed}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    runtime = init_distributed_runtime(args.device, args.distributed_backend)
    args.device = str(runtime.device)
    try:
        if runtime.device.type != "cuda":
            raise ValueError("verified compiler evaluation requires CUDA")
        if runtime.world_size != 2:
            raise ValueError("verified compiler evaluation freezes two GPUs")
        seed_everything(0)
        training = json.loads(Path(args.training_result).read_text())
        if training.get("protocol") != training_protocol_for_base_days(args.base_days):
            raise ValueError("training protocol mismatch")
        if training.get("status") != "complete":
            raise ValueError("training result is incomplete")
        prepared_hash = sha256(args.prepared_data)
        if prepared_hash != training["prepared_data"]["sha256"]:
            raise ValueError("prepared data differs from training")
        program_result = json.loads(Path(args.program_result).read_text())
        if (
            program_result.get("protocol") != ATTENTION_PROGRAM_PROTOCOL
            or program_result.get("status") != "design_search_complete"
            or program_result["design"]["selection"][
                "selected_attention_mix"
            ]
            != 1.0
        ):
            raise ValueError("compiled program selection result is invalid")
        if program_result["split"]["fit_users"] != args.fit_users:
            raise ValueError("compiled program fit split mismatch")
        if program_result["split"]["probe_users"] != args.selection_users:
            raise ValueError("compiled program selection split mismatch")
        plan_data, metadata = load_prepared_kuairand_plan(args.prepared_data)
        validate_long_context_plan(plan_data, metadata, args.base_days)
        cfg = HSTUConfig(**training["model"])
        args.seq_len = cfg.max_seq_len
        args.seed = int(training["args"]["seed"])
        date, samples = reconstruct_online_eval_samples(
            plan_data,
            (args.current_version,),
            args.max_eval_users,
        )[args.current_version]
        fit_samples, selection_samples, remaining = split_samples(
            samples,
            args.fit_users,
            args.selection_users,
            args.seed,
        )
        certificate_samples, test_samples = split_certificate_test(
            remaining,
            args.certificate_users,
            args.seed,
        )
        contract = FidelityContract(
            recovery_target=args.recovery_target,
            minimum_coverage=args.minimum_coverage,
            confidence_level=args.confidence_level,
            max_cost_ratio=args.max_cost_ratio,
            bootstrap_samples=args.bootstrap_samples,
            minimum_probe_users=args.minimum_valid_users,
        )
        current = load_checkpoint_model(
            cfg,
            args.checkpoint_dir,
            args.current_version,
            runtime.device,
        )
        manifest_dir = Path(args.manifest_dir)
        if runtime.is_primary:
            manifest_dir.mkdir(parents=True, exist_ok=True)
        if runtime.initialized:
            dist.barrier()
        pair_results = []
        for cache_version in args.cache_versions:
            path = program_path(args, cache_version)
            compiled, payload = load_compiled_program(
                path,
                cache_version,
                args,
                cfg,
                runtime.device,
            )
            old = load_checkpoint_model(
                cfg,
                args.checkpoint_dir,
                cache_version,
                runtime.device,
            )
            actions = action_specs(path, args.structural_depths)
            certificate_names = [
                "reuse",
                *(action.name for action in actions),
            ]
            local_certificate = certificate_samples[
                runtime.rank :: runtime.world_size
            ]
            local_certificate.sort(
                key=lambda sample: len(sample["history"]["item_ids"])
            )
            local_records, local_timing = evaluate_actions_local(
                current,
                old,
                compiled,
                local_certificate,
                certificate_names,
                args,
                runtime.device,
                include_quality=False,
            )
            certificate_timing = reduce_timing(local_timing, runtime)
            certificate_records = gather_records(local_records, runtime)
            if runtime.is_primary:
                assert certificate_records is not None
                probe_summary = semantic_summary(
                    certificate_records,
                    certificate_timing,
                )
                costs = {
                    action.name: probe_summary["configs"][action.name][
                        "migration_ratio_to_recompute"
                    ]
                    for action in actions
                }
                verified = compile_verified_plan(
                    protocol=PROTOCOL,
                    source_version=f"theta{cache_version}",
                    target_version=f"theta{args.current_version}",
                    actions=actions,
                    records=certificate_records,
                    cost_ratios=costs,
                    contract=contract,
                    seed=args.seed + cache_version * 10007,
                )
                manifest_path = manifest_dir / (
                    f"theta{cache_version}_to_theta"
                    f"{args.current_version}_verified.json"
                )
                save_json(verified.to_dict(), manifest_path)
                selected_name = verified.selected_action
                plan_payload = verified.to_dict()
            else:
                probe_summary = None
                certificate_records = None
                manifest_path = None
                selected_name = None
                plan_payload = None
            objects = [selected_name, plan_payload]
            if runtime.initialized:
                dist.broadcast_object_list(
                    objects,
                    src=0,
                    device=runtime.device,
                )
            selected_name = objects[0]
            plan_payload = objects[1]
            assert isinstance(selected_name, str)
            assert isinstance(plan_payload, dict)
            test_names = ["reuse", selected_name]
            if "recompute" not in test_names:
                test_names.append("recompute")
            local_test = test_samples[runtime.rank :: runtime.world_size]
            local_test.sort(
                key=lambda sample: len(sample["history"]["item_ids"])
            )
            local_test_records, local_test_timing = evaluate_actions_local(
                current,
                old,
                compiled,
                local_test,
                test_names,
                args,
                runtime.device,
                include_quality=True,
            )
            test_timing = reduce_timing(local_test_timing, runtime)
            test_records = gather_records(local_test_records, runtime)
            if runtime.is_primary:
                assert test_records is not None
                test_summary = summarize(
                    test_records,
                    test_timing,
                    args,
                )
                pair_results.append(
                    {
                        "cache_version": cache_version,
                        "current_version": args.current_version,
                        "cache_age_updates": (
                            args.current_version - cache_version
                        ),
                        "program": {
                            "path": str(path),
                            "sha256": sha256(path),
                            "protocol": payload["protocol"],
                            "fit": payload["fit"],
                        },
                        "manifest_path": str(manifest_path),
                        "verified_plan": plan_payload,
                        "certificate": probe_summary,
                        "per_user_certificate": certificate_records,
                        "test": test_summary,
                        "per_user_test": test_records,
                    }
                )
            del old, compiled
            torch.cuda.empty_cache()
        if runtime.is_primary:
            result = {
                "protocol": PROTOCOL,
                "status": "verified_design_complete",
                "study_stage": "adaptive_seed0_exploration",
                "source_training_result": args.training_result,
                "source_program_result": args.program_result,
                "prepared_data": {
                    "path": args.prepared_data,
                    "sha256": prepared_hash,
                },
                "checkpoint_dir": args.checkpoint_dir,
                "manifest_dir": args.manifest_dir,
                "world_size": runtime.world_size,
                "seed": args.seed,
                "eval_date": date,
                "model": training["model"],
                "split": {
                    "all_users": len(samples),
                    "fit_users": len(fit_samples),
                    "program_selection_users": len(selection_samples),
                    "certificate_users": len(certificate_samples),
                    "final_test_users": len(test_samples),
                    "fit_and_selection_provenance": args.program_result,
                    "certificate_split_seed": 27183 + args.seed,
                },
                "contract": contract.to_dict(),
                "selection_semantics": (
                    "recommendation labels are unavailable to certification; "
                    "the compiler selects the minimum measured-cost action whose "
                    "cache, score, and top-100 gap-recovery lower bounds and "
                    "user-coverage lower bounds satisfy the frozen contract"
                ),
                "pairs": pair_results,
            }
            save_json(result, args.output)
            print(
                json.dumps(
                    {
                        "output": args.output,
                        "selected_actions": [
                            {
                                "cache_version": pair["cache_version"],
                                "cache_age_updates": pair[
                                    "cache_age_updates"
                                ],
                                "selected": pair["verified_plan"][
                                    "selected_action"
                                ],
                                "fallback": pair["verified_plan"][
                                    "fallback_actions"
                                ],
                            }
                            for pair in pair_results
                        ],
                        "final_test_users": len(test_samples),
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
