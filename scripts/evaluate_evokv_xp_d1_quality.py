from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from hstu_kvcache.migration.program import compile_migration_program
from hstu_kvcache.migration.stage45_oldkv import (
    compile_direct_oldkv_program,
    write_direct_oldkv_program,
)
from hstu_kvcache.migration.xp_d1_quality import (
    ACTION_PLAN_PROTOCOL,
    PROTOCOL,
    REUSE_EXACT_METHODS,
    SUFFIX_DIAGNOSTIC_PROTOCOL,
    build_action_plan_v2,
    direct_program_sha256,
    evaluate_quality_batch,
    merge_batch_reports,
    select_token_budget,
    split_batches,
    split_identity,
    timed_call,
)
from hstu_kvcache.migration.xp_exact_baseline import (
    canonical_sha256,
    file_sha256,
    load_fixed_inputs,
    load_inference_checkpoint,
)
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.streaming.xp_multiversion import (
    XPUpdateWindow,
    build_window_batches,
)
from hstu_kvcache.streaming.xp_version_training import (
    load_xp_fixed_edge_corpus,
    prepare_fixed_qualification,
)

DEFAULT_CONFIG = "configs/evokv_baselines/x_qk_xp_two_gpu_baseline_v0.json"
DEFAULT_CHECKPOINT_ROOT = "checkpoints/evokv_xp_qk_e4096_h1536/seed0"
DEFAULT_OUTPUT = "results/system/evokv_xp_d1_quality/theta0_to_theta1.json"
DEFAULT_PROGRAM = (
    "results/system/evokv_xp_d1_quality/"
    "theta0_to_theta1_direct_oldkv_fp16.pt"
)
DEFAULT_ACTION_PLAN = (
    "results/system/evokv_xp_d1_quality/"
    "theta0_to_theta1_action_plan_v2.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--source-checkpoint-root")
    parser.add_argument("--target-checkpoint-root")
    parser.add_argument("--source-version", type=int, default=0)
    parser.add_argument("--target-version", type=int, default=1)
    parser.add_argument("--history-end", type=int, default=64)
    parser.add_argument("--update-end", type=int)
    parser.add_argument("--training-history-end", type=int)
    parser.add_argument("--training-update-end", type=int)
    parser.add_argument("--probe-role", choices=("theta01", "theta12"))
    parser.add_argument("--qualification-role", default="qualification")
    parser.add_argument("--capacity", default="288")
    parser.add_argument("--batch-size-per-rank", type=int, default=1)
    parser.add_argument("--fit-batches", type=int, default=4)
    parser.add_argument("--probe-batches", type=int, default=4)
    parser.add_argument("--qualification-batches", type=int, default=0)
    parser.add_argument("--negative-count", type=int, default=99)
    parser.add_argument("--reuse-exact-suffix-offsets", action="store_true")
    parser.add_argument("--include-frozen-control", action="store_true")
    parser.add_argument("--diagnostic-negative-counts", default="99,999")
    parser.add_argument(
        "--diagnostic-evaluation-kind",
        choices=("prequential", "long_context_characterization"),
        default="prequential",
    )
    parser.add_argument("--candidate-seed", type=int, default=20260801)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--program-output", default=DEFAULT_PROGRAM)
    parser.add_argument("--action-plan-output", default=DEFAULT_ACTION_PLAN)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def initialize(args: argparse.Namespace) -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not dist.is_initialized():
        dist.init_process_group(
            backend=args.backend,
            rank=rank,
            world_size=world_size,
        )
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    return rank, world_size, device


def validate_args(args: argparse.Namespace, world_size: int) -> str:
    if (
        args.source_version < 0
        or args.target_version != args.source_version + 1
        or args.history_end < 2
        or (
            args.update_end is not None
            and args.update_end <= args.history_end
        )
        or (
            (args.training_history_end is None)
            != (args.training_update_end is None)
        )
        or (
            args.training_history_end is not None
            and (
                args.training_history_end < 1
                or args.training_update_end <= args.training_history_end
                or args.training_update_end != args.history_end
            )
        )
        or args.batch_size_per_rank < 1
        or args.fit_batches < 1
        or args.probe_batches < 1
        or args.qualification_batches < 0
        or (
            not args.reuse_exact_suffix_offsets
            and args.negative_count not in {99, 999}
        )
        or (
            not args.reuse_exact_suffix_offsets
            and args.diagnostic_evaluation_kind != "prequential"
        )
        or (
            args.include_frozen_control
            and not args.reuse_exact_suffix_offsets
        )
        or args.timing_repeats < 1
        or world_size not in {1, 2, 4}
    ):
        raise ValueError("XP D1 quality arguments differ")
    if args.probe_role is not None:
        return args.probe_role
    return "theta12" if args.source_version == 0 else "theta01"


def diagnostic_negative_counts(args: argparse.Namespace) -> tuple[int, ...]:
    if not args.reuse_exact_suffix_offsets:
        return ()
    try:
        values = tuple(
            int(value.strip())
            for value in args.diagnostic_negative_counts.split(",")
            if value.strip()
        )
    except ValueError as error:
        raise ValueError("XP diagnostic negative counts differ") from error
    if (
        not values
        or len(values) != len(set(values))
        or any(value not in {99, 999} for value in values)
    ):
        raise ValueError("XP diagnostic negative counts differ")
    return values


def ensure_fresh_outputs(
    rank: int,
    paths: tuple[Path, ...],
) -> None:
    local_error = None
    if rank == 0:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            local_error = f"refusing to overwrite existing artifacts: {existing}"
    values: list[object] = [local_error]
    dist.broadcast_object_list(values, src=0)
    if values[0] is not None:
        raise FileExistsError(str(values[0]))


def limit_batches(
    batches: list[dict[str, torch.Tensor]],
    limit: int,
) -> tuple[dict[str, torch.Tensor], ...]:
    if limit == 0:
        return tuple(batches)
    if limit > len(batches):
        raise ValueError("requested qualification batches exceed frozen role")
    return tuple(batches[:limit])


def prefix_cache(
    dense,
    embedding,
    batch: dict[str, torch.Tensor],
    history_end: int,
    device: torch.device,
    repeats: int,
) -> tuple[HSTUKVCache, dict[str, object]]:
    prefix_width = history_end - 1
    records = batch["record_indices"].to(device)
    lengths = torch.where(
        records >= 0,
        torch.full_like(records, prefix_width),
        torch.zeros_like(records),
    )
    item_ids = batch["item_ids"][:, :prefix_width].to(device)
    behaviors = batch["behaviors"][:, :prefix_width].to(device)
    deltas = batch["time_deltas"][:, :prefix_width].to(device)

    def compute() -> HSTUKVCache:
        vectors = embedding(item_ids, lengths)
        return dense.core.compute_kv_from_item_embeddings(
            vectors,
            behaviors,
            deltas,
            lengths,
        )

    timed = timed_call(compute, device, repeats)
    cache = HSTUKVCache(
        k=timed.value.k.detach().to(device="cpu", dtype=torch.float16),
        v=timed.value.v.detach().to(device="cpu", dtype=torch.float16),
        seq_len=timed.value.seq_len,
    )
    return cache, {
        "median_milliseconds": timed.median_milliseconds,
        "samples_milliseconds": list(timed.samples_milliseconds),
    }


def materialize_source(
    dense,
    embedding,
    heldout,
    history_end: int,
    device: torch.device,
    repeats: int,
) -> tuple[list[HSTUKVCache], list[dict[str, object]]]:
    caches = []
    timings = []
    for index, value in enumerate(heldout):
        cache, timing = prefix_cache(
            dense,
            embedding,
            value.batch,
            history_end,
            device,
            repeats,
        )
        caches.append(cache)
        timings.append(timing)
        print(
            json.dumps(
                {
                    "phase": "source_old_cache",
                    "batch": index + 1,
                    "batches": len(heldout),
                    "history_end": history_end,
                }
            ),
            flush=True,
        )
    return caches, timings


def evaluate_role(
    dense,
    embedding,
    heldout,
    caches,
    program,
    history_end: int,
    device: torch.device,
    repeats: int,
    role: str,
    mixed_exact_record_ids: frozenset[int],
    methods=tuple(),
    suffix_offset_breakdown: bool = False,
    common_cache_storage_dtype: torch.dtype | None = None,
) -> list[dict[str, object]]:
    if len(heldout) != len(caches):
        raise ValueError("XP heldout/source cache counts differ")
    reports = []
    for index, (value, cache) in enumerate(zip(heldout, caches, strict=True)):
        arguments = {}
        if methods:
            arguments["methods"] = methods
        if suffix_offset_breakdown:
            arguments["suffix_offset_breakdown"] = True
        if common_cache_storage_dtype is not None:
            arguments["common_cache_storage_dtype"] = (
                common_cache_storage_dtype
            )
        reports.append(
            evaluate_quality_batch(
                dense,
                embedding,
                value.batch,
                value.candidates,
                cache,
                program,
                history_end,
                device,
                repeats,
                mixed_exact_record_ids,
                **arguments,
            )
        )
        print(
            json.dumps(
                {
                    "phase": "quality",
                    "role": role,
                    "batch": index + 1,
                    "batches": len(heldout),
                }
            ),
            flush=True,
        )
    return reports


def gather_objects(value: object, world_size: int) -> list[object]:
    values: list[object] = [None] * world_size
    dist.all_gather_object(values, value)
    return values


def role_mixed_selection(
    heldout,
    history_end: int,
    world_size: int,
    role: str,
) -> tuple[frozenset[int], dict[str, object]]:
    prefix_tokens = history_end - 1
    local = [
        (int(record_id), prefix_tokens, f"{role}:prefix{prefix_tokens}")
        for value in heldout
        for record_id in value.batch["record_indices"].tolist()
        if int(record_id) >= 0
    ]
    gathered = gather_objects(local, world_size)
    records = [record for rank_records in gathered for record in rank_records]
    exact_ids, ledger = select_token_budget(
        records,
        selection_salt=f"evokv-xp-d1-quality-mixed-fixed20:{role}",
    )
    return frozenset(exact_ids), ledger


def global_split_identity(local: dict[str, object], world_size: int) -> dict[str, object]:
    gathered = gather_objects(local, world_size)
    roles = {}
    for name in ("fit", "probe", "qualification_test"):
        rank_values = [value["roles"][name] for value in gathered]
        roles[name] = {
            "records": sum(int(value["records"]) for value in rank_values),
            "rank_record_ids_sha256": [
                value["record_ids_sha256"] for value in rank_values
            ],
        }
    return {"roles": roles, "sha256": canonical_sha256(roles)}


def combine_reports(
    local_reports: list[dict[str, object]],
    world_size: int,
) -> dict[str, object]:
    gathered = gather_objects(local_reports, world_size)
    flattened = [report for rank_reports in gathered for report in rank_reports]
    combined = merge_batch_reports(flattened)
    for method, values in combined["methods"].items():
        values["gpu_cost"]["max_rank_maintenance_milliseconds"] = max(
            sum(
                float(report["methods"][method]["maintenance_milliseconds"])
                for report in rank_reports
            )
            for rank_reports in gathered
        )
        values["gpu_cost"]["max_rank_online_milliseconds"] = max(
            sum(
                float(
                    report["methods"][method][
                        "online_suffix_and_score_milliseconds"
                    ]
                )
                for report in rank_reports
            )
            for rank_reports in gathered
        )
    return combined


def main() -> None:
    args = parse_args()
    rank, world_size, device = initialize(args)
    probe_role = validate_args(args, world_size)
    diagnostic_counts = diagnostic_negative_counts(args)
    diagnostic_mode = bool(diagnostic_counts)
    output = Path(args.output)
    program_output = Path(args.program_output)
    action_plan_output = Path(args.action_plan_output)
    ensure_fresh_outputs(
        rank,
        (output,)
        if diagnostic_mode
        else (output, program_output, action_plan_output),
    )
    inputs = load_fixed_inputs(
        args.config,
        args.capacity,
        world_size=world_size,
    )
    edge_descriptor = inputs.benchmark["data"]["fixed_edge_inputs"]
    repository_root = Path(args.config).resolve().parents[2]
    edge_path = repository_root / str(
        edge_descriptor["path"]
    )
    edge_summary_path = repository_root / str(
        edge_descriptor["summary_path"]
    )
    if file_sha256(edge_summary_path) != str(
        edge_descriptor["summary_sha256"]
    ):
        raise ValueError("XP fixed edge summary hash differs")
    corpus = load_xp_fixed_edge_corpus(
        edge_path,
        edge_summary_path,
        num_embeddings=inputs.spec.num_embeddings,
        num_prediction_items=inputs.spec.num_prediction_items,
        num_behaviors=inputs.spec.num_behaviors,
    )
    update_end = (
        args.update_end
        if args.update_end is not None
        else args.history_end + 8
    )
    if diagnostic_mode and update_end - args.history_end != 8:
        raise ValueError("XP suffix diagnostic requires eight offsets")
    update = XPUpdateWindow(
        source_version=args.source_version,
        target_version=args.target_version,
        history_end=args.history_end,
        update_end=update_end,
    )
    if diagnostic_mode:
        fit_batches = ()
        probe_batches = ()
        probe_audit = None
    else:
        probe_all, probe_audit = build_window_batches(
            corpus,
            probe_role,
            update,
            max_seq_len=inputs.spec.max_seq_len,
            batch_size_per_rank=args.batch_size_per_rank,
            rank=rank,
            world_size=world_size,
        )
        fit_batches, probe_batches = split_batches(
            probe_all,
            args.fit_batches,
            args.probe_batches,
        )
    qualification_all, qualification_audit = build_window_batches(
        corpus,
        args.qualification_role,
        update,
        max_seq_len=inputs.spec.max_seq_len,
        batch_size_per_rank=args.batch_size_per_rank,
        rank=rank,
        world_size=world_size,
    )
    qualification_batches = limit_batches(
        qualification_all,
        args.qualification_batches,
    )
    if any(
        args.history_end >= int(batch["item_ids"].shape[1])
        for batch in (*probe_batches, *qualification_batches)
    ):
        raise ValueError("XP explicit history boundary exceeds role width")
    local_split = split_identity(
        fit_batches,
        probe_batches,
        qualification_batches,
    )
    split = global_split_identity(local_split, world_size)
    if diagnostic_mode:
        diagnostic_heldout = {}
        diagnostic_candidate_sha = {}
        for negative_count in diagnostic_counts:
            heldout, candidate_sha = prepare_fixed_qualification(
                list(qualification_batches),
                num_prediction_items=inputs.spec.num_prediction_items,
                negative_count=negative_count,
                seed=args.candidate_seed,
                rank=rank,
                world_size=world_size,
            )
            diagnostic_heldout[negative_count] = heldout
            diagnostic_candidate_sha[negative_count] = candidate_sha
        probe_heldout = []
        qualification_heldout = diagnostic_heldout[diagnostic_counts[0]]
        probe_candidate_sha = None
        qualification_candidate_sha = None
    else:
        probe_heldout, probe_candidate_sha = prepare_fixed_qualification(
            list(probe_batches),
            num_prediction_items=inputs.spec.num_prediction_items,
            negative_count=args.negative_count,
            seed=args.candidate_seed + 1_000_003,
            rank=rank,
            world_size=world_size,
        )
        qualification_heldout, qualification_candidate_sha = (
            prepare_fixed_qualification(
                list(qualification_batches),
                num_prediction_items=inputs.spec.num_prediction_items,
                negative_count=args.negative_count,
                seed=args.candidate_seed,
                rank=rank,
                world_size=world_size,
            )
        )
    probe_history_end = args.history_end
    qualification_history_end = args.history_end
    if diagnostic_mode:
        probe_mixed_exact_ids = frozenset()
        qualification_mixed_exact_ids = frozenset()
        probe_mixed_selection = None
        qualification_mixed_selection = None
    else:
        probe_mixed_exact_ids, probe_mixed_selection = role_mixed_selection(
            probe_heldout,
            probe_history_end,
            world_size,
            "probe",
        )
        qualification_mixed_exact_ids, qualification_mixed_selection = (
            role_mixed_selection(
                qualification_heldout,
                qualification_history_end,
                world_size,
                "qualification_test",
            )
        )
    source_checkpoint_root = (
        args.source_checkpoint_root or args.checkpoint_root
    )
    target_checkpoint_root = (
        args.target_checkpoint_root or args.checkpoint_root
    )
    source_dense, source_embedding, source_checkpoint = (
        load_inference_checkpoint(
            source_checkpoint_root,
            args.source_version,
            inputs.spec,
            rank=rank,
            world_size=world_size,
            device=device,
        )
    )
    if diagnostic_mode:
        probe_caches = []
        probe_source_timings = []
    else:
        probe_caches, probe_source_timings = materialize_source(
            source_dense,
            source_embedding,
            probe_heldout,
            probe_history_end,
            device,
            args.timing_repeats,
        )
    qualification_caches, qualification_source_timings = materialize_source(
        source_dense,
        source_embedding,
        qualification_heldout,
        qualification_history_end,
        device,
        args.timing_repeats,
    )
    if diagnostic_mode and args.include_frozen_control:
        frozen_diagnostic_quality = {}
        for negative_count in diagnostic_counts:
            frozen_reports = evaluate_role(
                source_dense,
                source_embedding,
                diagnostic_heldout[negative_count],
                qualification_caches,
                None,
                qualification_history_end,
                device,
                args.timing_repeats,
                "frozen_control",
                frozenset(),
                methods=REUSE_EXACT_METHODS,
                suffix_offset_breakdown=True,
                common_cache_storage_dtype=torch.float16,
            )
            frozen_diagnostic_quality[str(negative_count)] = combine_reports(
                frozen_reports,
                world_size,
            )
    else:
        frozen_diagnostic_quality = None
    source_dense.to("cpu")
    del source_embedding
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    target_dense, target_embedding, target_checkpoint = (
        load_inference_checkpoint(
            target_checkpoint_root,
            args.target_version,
            inputs.spec,
            rank=rank,
            world_size=world_size,
            device=device,
        )
    )
    if diagnostic_mode:
        direct = None
        compile_metrics = None
        program_identity_sha = None
        program_device = None
        del source_dense
    else:
        source_dense.to(device)
        compiled = compile_migration_program(
            target_dense.core,
            f"theta{args.source_version}",
            f"theta{args.target_version}",
        )
        direct, compile_metrics = compile_direct_oldkv_program(
            source_dense.core,
            compiled,
        )
        program_identity_sha = direct_program_sha256(direct)
        gathered_program_hashes = gather_objects(
            program_identity_sha,
            world_size,
        )
        if len(set(gathered_program_hashes)) != 1:
            raise RuntimeError("XP direct program differs across ranks")
        source_dense.to("cpu")
        del source_dense, compiled
        program_device = direct.to(device, dtype=torch.float16)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if diagnostic_mode:
        diagnostic_quality = {}
        for negative_count in diagnostic_counts:
            qualification_reports = evaluate_role(
                target_dense,
                target_embedding,
                diagnostic_heldout[negative_count],
                qualification_caches,
                None,
                qualification_history_end,
                device,
                args.timing_repeats,
                "qualification_test",
                frozenset(),
                methods=REUSE_EXACT_METHODS,
                suffix_offset_breakdown=True,
                common_cache_storage_dtype=torch.float16,
            )
            diagnostic_quality[str(negative_count)] = combine_reports(
                qualification_reports,
                world_size,
            )
        probe_result = None
        qualification_result = None
    else:
        probe_reports = evaluate_role(
            target_dense,
            target_embedding,
            probe_heldout,
            probe_caches,
            program_device,
            probe_history_end,
            device,
            args.timing_repeats,
            "probe",
            probe_mixed_exact_ids,
            common_cache_storage_dtype=torch.float16,
        )
        qualification_reports = evaluate_role(
            target_dense,
            target_embedding,
            qualification_heldout,
            qualification_caches,
            program_device,
            qualification_history_end,
            device,
            args.timing_repeats,
            "qualification_test",
            qualification_mixed_exact_ids,
            common_cache_storage_dtype=torch.float16,
        )
        probe_result = combine_reports(probe_reports, world_size)
        qualification_result = combine_reports(
            qualification_reports,
            world_size,
        )
    source_timing_local = {
        "qualification_test": sum(
            float(value["median_milliseconds"])
            for value in qualification_source_timings
        ),
    }
    if not diagnostic_mode:
        source_timing_local["probe"] = sum(
            float(value["median_milliseconds"])
            for value in probe_source_timings
        )
    source_timing_values = gather_objects(
        source_timing_local,
        world_size,
    )
    if diagnostic_mode:
        diagnostic_candidate_hashes = {
            str(negative_count): gather_objects(
                diagnostic_candidate_sha[negative_count],
                world_size,
            )
            for negative_count in diagnostic_counts
        }
        probe_candidate_hashes = None
        qualification_candidate_hashes = None
    else:
        probe_candidate_hashes = gather_objects(
            probe_candidate_sha,
            world_size,
        )
        qualification_candidate_hashes = gather_objects(
            qualification_candidate_sha,
            world_size,
        )
    if rank == 0 and diagnostic_mode:
        result = {
            "protocol": SUFFIX_DIAGNOSTIC_PROTOCOL,
            "scientific_result": False,
            "formal_result": False,
            "status": "complete",
            "diagnostic": "adjacent_reuse_exact_suffix_offsets",
            "evaluation_kind": args.diagnostic_evaluation_kind,
            "benchmark_id": inputs.benchmark["benchmark_id"],
            "edge": {
                "source_version": args.source_version,
                "target_version": args.target_version,
                "history_end": args.history_end,
                "update_end": update_end,
                "training_window": (
                    None
                    if args.training_history_end is None
                    else {
                        "history_end": args.training_history_end,
                        "update_end": args.training_update_end,
                    }
                ),
                "evaluation_window": {
                    "history_end": args.history_end,
                    "evaluation_end": update_end,
                    "semantics": (
                        "next_unseen_window"
                        if args.diagnostic_evaluation_kind == "prequential"
                        else "nonprequential_long_context_characterization"
                    ),
                    "suffix_offsets": list(range(1, 9)),
                },
            },
            "world_size": world_size,
            "bindings": {
                "source_checkpoint": source_checkpoint,
                "target_checkpoint": target_checkpoint,
                "fixed_edge_corpus": {
                    "path": str(corpus.path),
                    "sha256": corpus.file_sha256,
                    "content_sha256": corpus.content_sha256,
                    "summary_path": str(corpus.summary_path),
                    "summary_sha256": corpus.summary_sha256,
                },
                "evaluation_split": split,
            },
            "role": {
                "name": "qualification_test",
                "source_role": args.qualification_role,
                "selection_use": "held-out recommendation diagnostic",
                "audit": qualification_audit,
                "candidate_sha256_per_rank_by_negative_count": (
                    diagnostic_candidate_hashes
                ),
            },
            "quality_by_negative_count": diagnostic_quality,
            "frozen_quality_by_negative_count": frozen_diagnostic_quality,
            "gpu_cost": {
                "source_old_cache_max_rank_milliseconds": max(
                    float(value["qualification_test"])
                    for value in source_timing_values
                ),
                "checkpoint_and_source_cache_reused_across_candidate_sets": True,
                "cost_is_diagnostic": True,
            },
            "recommendation_contract": {
                "positive_candidate_column": 0,
                "negative_candidates": list(diagnostic_counts),
                "negative_sampling": (
                    "uniform_with_replacement_positive_excluded"
                ),
                "metrics": [
                    "sampled_cross_entropy",
                    "hit_rate_at_10",
                    "ndcg_at_10",
                    "mean_reciprocal_rank",
                ],
                "paired_target_contributions": [
                    "record_id",
                    "suffix_offset",
                    "all_reuse_rank",
                    "all_reuse_sampled_cross_entropy",
                    "all_exact_rank",
                    "all_exact_sampled_cross_entropy",
                ],
                "common_cache_endpoint": {
                    "storage_dtype": "torch.float16",
                    "consumption_dtype": "torch.float32",
                    "exact_path": "fp32_compute_to_fp16_storage_to_fp32_consume",
                    "reuse_path": "fp16_storage_to_fp32_consume",
                },
                "prediction": "item t+1 from current-model hidden state t",
            },
            "method_contract": {
                **(
                    {
                        "all_frozen": (
                            "theta-source model, embedding, scorer, and "
                            "FP16-stored prefix K/V on the same future "
                            "targets"
                        )
                    }
                    if args.include_frozen_control
                    else {}
                ),
                "all_reuse": (
                    "theta-source FP16-stored prefix K/V consumed as FP32 by "
                    "theta-target"
                ),
                "all_exact": (
                    "theta-target prefix K/V recomputed in FP32, stored as "
                    "FP16, and consumed as FP32"
                ),
            },
            "limitations": [
                "adjacent one-version reuse only",
                "uniform sampled negatives are a diagnostic candidate set",
                "all artifacts remain development-only",
            ],
            "args": vars(args),
        }
        atomic_json(output, result)
        print(json.dumps({"output": str(output), "status": "complete"}))
    if diagnostic_mode:
        dist.barrier()
        dist.destroy_process_group()
        return
    if rank == 0:
        program_descriptor = write_direct_oldkv_program(
            direct,
            program_output,
            provenance={
                "protocol": PROTOCOL,
                "source_checkpoint_sha256": source_checkpoint["sha256"],
                "target_checkpoint_sha256": target_checkpoint["sha256"],
                "fixed_edge_input_sha256": corpus.file_sha256,
                "split_sha256": split["sha256"],
                "program_identity_sha256": program_identity_sha,
                "fit_kind": "analytic_model_projection_no_quality_label_fit",
            },
            compile_metrics=compile_metrics,
        )
        action_plan = build_action_plan_v2(
            inputs.records,
            benchmark_id=str(inputs.benchmark["benchmark_id"]),
            source_version=args.source_version,
            target_version=args.target_version,
            program_sha256=program_descriptor["sha256"],
            source_checkpoint_sha256=str(source_checkpoint["sha256"]),
            target_checkpoint_sha256=str(target_checkpoint["sha256"]),
            workload_sha256=str(
                inputs.bindings["het_workload"]["sha256"]
            ),
            split_sha256=str(split["sha256"]),
            selection_salt="evokv-xp-d1-action-plan-v2-development",
        )
        atomic_json(action_plan_output, action_plan)
        result = {
            "protocol": PROTOCOL,
            "scientific_result": False,
            "formal_result": False,
            "status": "complete",
            "benchmark_id": inputs.benchmark["benchmark_id"],
            "edge": {
                "source_version": args.source_version,
                "target_version": args.target_version,
                "history_end": args.history_end,
                "update_end": update_end,
                "training_window": (
                    None
                    if args.training_history_end is None
                    else {
                        "history_end": args.training_history_end,
                        "update_end": args.training_update_end,
                    }
                ),
                "evaluation_window": {
                    "history_end": args.history_end,
                    "evaluation_end": update_end,
                    "semantics": "next_unseen_window",
                },
            },
            "world_size": world_size,
            "bindings": {
                **inputs.bindings,
                "source_checkpoint": source_checkpoint,
                "target_checkpoint": target_checkpoint,
                "fixed_edge_corpus": {
                    "path": str(corpus.path),
                    "sha256": corpus.file_sha256,
                    "content_sha256": corpus.content_sha256,
                    "summary_path": str(corpus.summary_path),
                    "summary_sha256": corpus.summary_sha256,
                },
                "evaluation_split": split,
                "program": {
                    **program_descriptor,
                    "identity_sha256": program_identity_sha,
                },
                "action_plan": {
                    "path": str(action_plan_output),
                    "sha256": file_sha256(action_plan_output),
                    "protocol": ACTION_PLAN_PROTOCOL,
                    "records_sha256": action_plan["records_sha256"],
                },
            },
            "roles": {
                "fit": {
                    "source_role": probe_role,
                    "consumed_by_compiler": False,
                    "reason": "direct-old-K/V compiler is analytic and label-free",
                },
                "probe": {
                    "source_role": probe_role,
                    "selection_use": "D1 fidelity diagnostic only",
                    "audit": probe_audit,
                    "candidate_sha256_per_rank": probe_candidate_hashes,
                    "mixed_fixed20_selection": probe_mixed_selection,
                },
                "qualification_test": {
                    "source_role": args.qualification_role,
                    "selection_use": "held-out recommendation accuracy",
                    "audit": qualification_audit,
                    "candidate_sha256_per_rank": (
                        qualification_candidate_hashes
                    ),
                    "mixed_fixed20_selection": (
                        qualification_mixed_selection
                    ),
                },
            },
            "quality": {
                "probe": probe_result,
                "qualification_test": qualification_result,
            },
            "gpu_cost": {
                "source_old_cache_max_rank_milliseconds": {
                    name: max(float(value[name]) for value in source_timing_values)
                    for name in ("probe", "qualification_test")
                },
                "direct_program_compile_seconds_per_rank": (
                    compile_metrics.elapsed_seconds
                ),
                "cost_is_measured": True,
                "mixed_cost_is_end_to_end": False,
            },
            "recommendation_contract": {
                "positive_candidate_column": 0,
                "negative_candidates": args.negative_count,
                "metrics": [
                    "sampled_cross_entropy",
                    "hit_rate_at_10",
                    "ndcg_at_10",
                    "mean_reciprocal_rank",
                ],
                "prediction": "item t+1 from current-model hidden state t",
                "common_cache_endpoint": {
                    "storage_dtype": "torch.float16",
                    "consumption_dtype": "torch.float32",
                    "methods": [
                        "all_reuse",
                        "compiled_direct_oldkv",
                        "mixed_fixed20",
                        "all_exact",
                    ],
                },
                "prefix": (
                    "source/current/migrated cache through the token before the "
                    "last history token, followed by current-model suffix replay"
                ),
            },
            "method_contract": {
                "all_reuse": (
                    "theta-source FP16-stored prefix K/V consumed by theta-target"
                ),
                "compiled_direct_oldkv": (
                    "one shared analytic affine over existing FP16 source K/V, "
                    "with FP16 destination storage"
                ),
                "all_exact": (
                    "theta-target prefix K/V recomputed in FP32 with FP16 "
                    "destination storage"
                ),
                "mixed_fixed20": (
                    "the frozen ActionPlan token-budget policy applied to the "
                    "user-disjoint quality role; exact rows plus compiled rows "
                    "are evaluated together"
                ),
            },
            "compile_metrics": compile_metrics.to_dict(),
            "limitations": [
                (
                    "the official fit/profile HET records are not materialized in "
                    "the fixed-edge artifact; fit/probe are disjoint slices of one "
                    "non-qualification fixed-edge role"
                ),
                (
                    "the analytic direct-old-K/V compiler consumes model weights, "
                    "not fit labels or per-record errors"
                ),
                (
                    "quality-role record IDs are intentionally user-disjoint from "
                    "HET ActionPlan IDs, so mixed quality applies the same frozen "
                    "label-free retained-token policy rather than reusing HET "
                    "record assignments"
                ),
                (
                    "the projection-only direct-old-K/V operator is an XP bridge "
                    "development diagnostic, not the cross-dataset active low-rank "
                    "fit headline; the historical fixed-task 3x3 evidence remains "
                    "the primary D1 motivation"
                ),
                "all artifacts remain development-only",
            ],
            "args": vars(args),
        }
        atomic_json(output, result)
        print(json.dumps({"output": str(output), "status": "complete"}))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
