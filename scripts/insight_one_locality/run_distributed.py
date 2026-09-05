#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from insight_one_locality.common import (  # noqa: E402
    CANDIDATES,
    CONTRACT,
    CUTOVER_DAYS,
    DATASET,
    DAY,
    EDGES,
    HISTORY,
    INPUT_MANIFEST,
    KNOWN_ITEMS,
    LOCALITY_CONFIGS,
    OOV_BUCKETS,
    PATH_IDS,
    POPULATION,
    RESULT_ROOT,
    checkpoint,
    config_mask,
    config_records,
    histories_at_cutover,
    hybrid_cache,
    load_input_manifest,
    score_cache_chunked,
    sha256_file,
    token_importance_scores,
    token_masks,
    verify_contract,
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_edges(value: str) -> tuple[int, ...]:
    output = tuple(int(part) for part in value.split(",") if part != "")
    if not output or any(index < 0 or index >= len(EDGES) for index in output):
        raise argparse.ArgumentTypeError("edge indices must be a comma-separated subset of 0..4")
    if len(set(output)) != len(output):
        raise argparse.ArgumentTypeError("edge indices contain duplicates")
    return output


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        raise RuntimeError("Insight 1 execution requires torchrun with four ranks")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise RuntimeError(f"frozen Insight 1 runtime requires four ranks, got {world}")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def verify_model_payload(payload: dict[str, Any]) -> None:
    config = payload["config"]
    expected = {
        "hidden_size": 192,
        "num_layers": 6,
        "num_heads": 6,
        "max_seq_len": 1024,
    }
    for name, value in expected.items():
        if int(config[name]) != value:
            raise RuntimeError(f"checkpoint {name} differs from contract: {config[name]} != {value}")


def max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(left.float() - right.float())))


def validate_hybrid(
    hybrid,
    parent,
    current,
    mask: torch.Tensor,
) -> tuple[float, float]:
    selected_error = max(
        max_difference(hybrid.k[mask], current.k[mask]),
        max_difference(hybrid.v[mask], current.v[mask]),
    )
    inverse = ~mask
    unselected_error = max(
        max_difference(hybrid.k[inverse], parent.k[inverse]),
        max_difference(hybrid.v[inverse], parent.v[inverse]),
    )
    return selected_error, unselected_error


def require_formal_gate(batch_size: int, candidate_chunk: int) -> None:
    canary_path = RESULT_ROOT / "canary" / "summary.json"
    estimate_path = RESULT_ROOT / "resource_estimate.json"
    if not canary_path.is_file() or not estimate_path.is_file():
        raise RuntimeError("formal run requires completed canary and resource estimate")
    canary = json.loads(canary_path.read_text(encoding="utf-8"))
    estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
    if not canary.get("passed") or canary.get("contract_sha256") != sha256_file(CONTRACT):
        raise RuntimeError("focused canary did not pass under the current contract")
    recommended = estimate["recommended"]
    if int(recommended["batch_size_per_rank"]) != batch_size:
        raise RuntimeError("formal batch size differs from the frozen resource estimate")
    if int(recommended["candidate_chunk"]) != candidate_chunk:
        raise RuntimeError("formal candidate chunk differs from the frozen resource estimate")


def output_path_for_scope(scope: str) -> Path:
    if scope == "canary":
        return RESULT_ROOT / "canary"
    if scope == "formal":
        return RESULT_ROOT / "formal_raw"
    raise ValueError("benchmark output must be specified explicitly")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("canary", "benchmark", "formal"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--edge-indices", type=parse_edges, default=tuple(range(5)))
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--candidate-chunk", type=int, required=True)
    parser.add_argument("--history-threads", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.candidate_chunk < 1 or CANDIDATES % args.candidate_chunk:
        raise ValueError("batch size must be positive and candidate chunk must divide 64")
    if args.output is None:
        args.output = output_path_for_scope(args.scope)
    expected_users = 32 if args.scope == "canary" else POPULATION if args.scope == "formal" else None
    if args.max_users is None:
        if expected_users is None:
            raise ValueError("benchmark scope requires --max-users")
        args.max_users = expected_users
    if not 4 <= args.max_users <= POPULATION:
        raise ValueError("max users must be in 4..3000")
    if args.scope in {"canary", "formal"}:
        if args.max_users != expected_users or args.edge_indices != tuple(range(5)):
            raise ValueError(f"{args.scope} must use its frozen population and all five edges")

    rank, local_rank, world = distributed_context()
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(args.torch_threads)
    started = time.perf_counter()

    verification = [None]
    if rank == 0:
        try:
            verify_contract()
            verification[0] = {"ok": True}
        except BaseException as error:
            verification[0] = {"ok": False, "error": repr(error)}
    dist.broadcast_object_list(verification, src=0)
    if not verification[0]["ok"]:
        raise RuntimeError(f"contract verification failed: {verification[0]['error']}")
    contract, all_uids, all_candidates, _ = load_input_manifest(verify_frozen=False)
    if args.scope == "formal":
        require_formal_gate(args.batch_size, args.candidate_chunk)

    partial = args.output.with_name(args.output.name + ".partial")
    if rank == 0:
        if args.output.exists() or partial.exists():
            raise FileExistsError(f"refusing to overwrite output: {args.output}")
        partial.mkdir(parents=True)
        atomic_json(
            partial / "configuration.json",
            {
                "contract_sha256": sha256_file(CONTRACT),
                "scope": args.scope,
                "max_users": args.max_users,
                "edge_indices": list(args.edge_indices),
                "batch_size_per_rank": args.batch_size,
                "candidate_chunk": args.candidate_chunk,
                "world_size": world,
                "path_ids": list(PATH_IDS),
                "locality_configs": config_records(),
            },
        )
    dist.barrier()
    rank_output = partial / f"rank{rank}"
    rank_output.mkdir()

    selected_indices = np.arange(args.max_users, dtype=np.int64)
    local_indices = selected_indices[rank::world]
    local_uids = all_uids[local_indices]
    print(
        json.dumps(
            {
                "phase": "load_histories",
                "rank": rank,
                "users": len(local_uids),
                "scope": args.scope,
            }
        ),
        flush=True,
    )
    history_started = time.perf_counter()
    history = load_histories(
        local_uids.tolist(),
        oov_buckets=OOV_BUCKETS,
        dataset_path=DATASET,
        known_vocab_size=KNOWN_ITEMS,
        end_timestamp=(CUTOVER_DAYS[-1] + 1) * DAY,
        threads=args.history_threads,
    )
    history_seconds = time.perf_counter() - history_started
    rank_edges: list[dict[str, Any]] = []
    global_correctness = {
        "empty_tensor_max_abs_error": 0.0,
        "full_tensor_max_abs_error": 0.0,
        "empty_score_max_abs_error": 0.0,
        "full_score_max_abs_error": 0.0,
        "selected_tensor_max_abs_error": 0.0,
        "unselected_tensor_max_abs_error": 0.0,
        "nonfinite_scores": 0,
    }

    for edge_index in args.edge_indices:
        edge_name = EDGES[edge_index]
        print(json.dumps({"phase": "edge_start", "rank": rank, "edge": edge_name}), flush=True)
        load_started = time.perf_counter()
        parent, parent_payload = load_model(checkpoint(edge_index), device)
        current, current_payload = load_model(checkpoint(edge_index + 1), device)
        verify_model_payload(parent_payload)
        verify_model_payload(current_payload)
        model_load_seconds = time.perf_counter() - load_started
        _, item_np, behavior_np, delta_np, query_delta_np = histories_at_cutover(
            history, local_uids, CUTOVER_DAYS[edge_index] * DAY
        )
        candidate_np = all_candidates[edge_index, local_indices]
        scores = np.empty((len(local_uids), len(PATH_IDS), CANDIDATES), dtype=np.float32)
        batches: list[dict[str, Any]] = []
        torch.cuda.reset_peak_memory_stats(device)

        for start in range(0, len(local_uids), args.batch_size):
            stop = min(start + args.batch_size, len(local_uids))
            batch_started = time.perf_counter()
            items = torch.as_tensor(item_np[start:stop], dtype=torch.long, device=device)
            behaviors = torch.as_tensor(behavior_np[start:stop], dtype=torch.long, device=device)
            deltas = torch.as_tensor(delta_np[start:stop], dtype=torch.float32, device=device)
            query_deltas = torch.as_tensor(
                query_delta_np[start:stop], dtype=torch.float32, device=device
            )
            candidates = torch.as_tensor(
                candidate_np[start:stop], dtype=torch.long, device=device
            )
            torch.cuda.synchronize(device)
            cache_started = time.perf_counter()
            parent_cache = parent.compute_kv(items, behaviors, deltas)
            current_cache = current.compute_kv(items, behaviors, deltas)
            torch.cuda.synchronize(device)
            cache_seconds = time.perf_counter() - cache_started

            anchor_started = time.perf_counter()
            reuse_scores = score_cache_chunked(
                current, parent_cache, candidates, query_deltas, args.candidate_chunk
            )
            exact_scores = score_cache_chunked(
                current, current_cache, candidates, query_deltas, args.candidate_chunk
            )
            scores[start:stop, 0] = reuse_scores.float().cpu().numpy()
            scores[start:stop, 1] = exact_scores.float().cpu().numpy()
            torch.cuda.synchronize(device)
            anchor_seconds = time.perf_counter() - anchor_started

            selector_started = time.perf_counter()
            selector_scores = token_importance_scores(
                current,
                parent_cache,
                current_cache,
                candidates,
                query_deltas,
                args.candidate_chunk,
            )
            selected_tokens = token_masks(selector_scores)
            torch.cuda.synchronize(device)
            selector_seconds = time.perf_counter() - selector_started

            locality_started = time.perf_counter()
            batch_size = stop - start
            for path_index, config in enumerate(LOCALITY_CONFIGS, start=2):
                mask = config_mask(config, batch_size, device, selected_tokens)
                hybrid = hybrid_cache(parent_cache, current_cache, mask)
                if args.scope == "canary":
                    selected_error, unselected_error = validate_hybrid(
                        hybrid, parent_cache, current_cache, mask
                    )
                    global_correctness["selected_tensor_max_abs_error"] = max(
                        global_correctness["selected_tensor_max_abs_error"], selected_error
                    )
                    global_correctness["unselected_tensor_max_abs_error"] = max(
                        global_correctness["unselected_tensor_max_abs_error"], unselected_error
                    )
                path_scores = score_cache_chunked(
                    current, hybrid, candidates, query_deltas, args.candidate_chunk
                )
                scores[start:stop, path_index] = path_scores.float().cpu().numpy()
                del hybrid, mask, path_scores
            torch.cuda.synchronize(device)
            locality_seconds = time.perf_counter() - locality_started

            if start == 0:
                empty = torch.zeros_like(parent_cache.k[..., 0], dtype=torch.bool)
                full = torch.ones_like(empty)
                empty_cache = hybrid_cache(parent_cache, current_cache, empty)
                full_cache = hybrid_cache(parent_cache, current_cache, full)
                global_correctness["empty_tensor_max_abs_error"] = max(
                    global_correctness["empty_tensor_max_abs_error"],
                    max(max_difference(empty_cache.k, parent_cache.k), max_difference(empty_cache.v, parent_cache.v)),
                )
                global_correctness["full_tensor_max_abs_error"] = max(
                    global_correctness["full_tensor_max_abs_error"],
                    max(max_difference(full_cache.k, current_cache.k), max_difference(full_cache.v, current_cache.v)),
                )
                empty_scores = score_cache_chunked(
                    current, empty_cache, candidates, query_deltas, args.candidate_chunk
                )
                full_scores = score_cache_chunked(
                    current, full_cache, candidates, query_deltas, args.candidate_chunk
                )
                global_correctness["empty_score_max_abs_error"] = max(
                    global_correctness["empty_score_max_abs_error"],
                    max_difference(empty_scores, reuse_scores),
                )
                global_correctness["full_score_max_abs_error"] = max(
                    global_correctness["full_score_max_abs_error"],
                    max_difference(full_scores, exact_scores),
                )
                del empty_cache, full_cache, empty_scores, full_scores, empty, full

            nonfinite = int(
                np.size(scores[start:stop]) - np.isfinite(scores[start:stop]).sum()
            )
            global_correctness["nonfinite_scores"] += nonfinite
            torch.cuda.synchronize(device)
            batch_seconds = time.perf_counter() - batch_started
            batches.append(
                {
                    "start": start,
                    "stop": stop,
                    "users": batch_size,
                    "cache_seconds": cache_seconds,
                    "anchor_seconds": anchor_seconds,
                    "selector_seconds": selector_seconds,
                    "locality_seconds": locality_seconds,
                    "total_seconds": batch_seconds,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1 << 20),
                    "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1 << 20),
                }
            )
            del (
                items,
                behaviors,
                deltas,
                query_deltas,
                candidates,
                parent_cache,
                current_cache,
                reuse_scores,
                exact_scores,
                selector_scores,
                selected_tokens,
            )

        edge_file = rank_output / f"{edge_name}.npz"
        with edge_file.open("wb") as stream:
            np.savez_compressed(
                stream,
                uids=local_uids,
                path_ids=np.asarray(PATH_IDS),
                scores=scores,
            )
        processing_seconds = float(sum(row["total_seconds"] for row in batches))
        edge_record = {
            "edge": edge_name,
            "users": len(local_uids),
            "raw_file": edge_file.name,
            "raw_sha256": sha256_file(edge_file),
            "raw_bytes": edge_file.stat().st_size,
            "model_load_seconds": model_load_seconds,
            "processing_seconds": processing_seconds,
            "batches": batches,
        }
        rank_edges.append(edge_record)
        print(
            json.dumps(
                {
                    "phase": "edge_complete",
                    "rank": rank,
                    "edge": edge_name,
                    "users": len(local_uids),
                    "processing_seconds": processing_seconds,
                }
            ),
            flush=True,
        )
        del parent, current, parent_payload, current_payload, scores
        torch.cuda.empty_cache()

    rank_summary = {
        "status": f"{args.scope}_rank_complete",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world,
        "uids": local_uids.tolist(),
        "history_load_seconds": history_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "correctness": global_correctness,
        "edges": rank_edges,
    }
    atomic_json(rank_output / "summary.json", rank_summary)
    dist.barrier()

    if rank == 0:
        rank_summaries = [
            json.loads((partial / f"rank{value}/summary.json").read_text(encoding="utf-8"))
            for value in range(world)
        ]
        observed_uids = [uid for summary in rank_summaries for uid in summary["uids"]]
        if len(observed_uids) != args.max_users or len(set(observed_uids)) != args.max_users:
            raise RuntimeError("rank UID shards are incomplete or overlapping")
        expected_edges = {EDGES[index] for index in args.edge_indices}
        for summary in rank_summaries:
            if {record["edge"] for record in summary["edges"]} != expected_edges:
                raise RuntimeError("rank edge set differs from requested edges")
        correctness = {
            key: (
                sum(int(summary["correctness"][key]) for summary in rank_summaries)
                if key == "nonfinite_scores"
                else max(float(summary["correctness"][key]) for summary in rank_summaries)
            )
            for key in global_correctness
        }
        processing_wall = max(
            sum(record["processing_seconds"] for record in summary["edges"])
            for summary in rank_summaries
        )
        model_load_wall = max(
            sum(record["model_load_seconds"] for record in summary["edges"])
            for summary in rank_summaries
        )
        user_edges = args.max_users * len(args.edge_indices)
        throughput = user_edges / processing_wall
        peak_allocated = max(
            batch["peak_allocated_mib"]
            for summary in rank_summaries
            for edge in summary["edges"]
            for batch in edge["batches"]
        )
        peak_reserved = max(
            batch["peak_reserved_mib"]
            for summary in rank_summaries
            for edge in summary["edges"]
            for batch in edge["batches"]
        )
        tolerance = contract["focused_canary"]
        passed = (
            correctness["nonfinite_scores"] == 0
            and correctness["empty_tensor_max_abs_error"] == 0.0
            and correctness["full_tensor_max_abs_error"] == 0.0
            and correctness["selected_tensor_max_abs_error"] == 0.0
            and correctness["unselected_tensor_max_abs_error"] == 0.0
            and correctness["empty_score_max_abs_error"]
            <= float(tolerance["empty_mask_reproduces_Reuse_max_abs_logit_error"])
            and correctness["full_score_max_abs_error"]
            <= float(tolerance["full_mask_reproduces_Current_Exact_max_abs_logit_error"])
        )
        summary = {
            "status": f"{args.scope}_raw_complete",
            "passed": passed,
            "contract_sha256": sha256_file(CONTRACT),
            "input_manifest_sha256": sha256_file(INPUT_MANIFEST / "manifest.json"),
            "scope": args.scope,
            "users": args.max_users,
            "edges": [EDGES[index] for index in args.edge_indices],
            "locality_configs": len(LOCALITY_CONFIGS),
            "paths_including_anchors": len(PATH_IDS),
            "batch_size_per_rank": args.batch_size,
            "candidate_chunk": args.candidate_chunk,
            "world_size": world,
            "correctness": correctness,
            "processing_wall_seconds": processing_wall,
            "model_load_wall_seconds": model_load_wall,
            "global_user_edge_throughput_per_second": throughput,
            "rough_formal_seconds_same_settings": (
                model_load_wall * len(EDGES) / len(args.edge_indices)
                + POPULATION * len(EDGES) / throughput
            ),
            "peak_allocated_mib": peak_allocated,
            "peak_reserved_mib": peak_reserved,
            "elapsed_seconds": max(summary["elapsed_seconds"] for summary in rank_summaries),
            "rank_summaries": [f"rank{value}/summary.json" for value in range(world)],
        }
        atomic_json(partial / "summary.json", summary)
        partial.rename(args.output)
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
