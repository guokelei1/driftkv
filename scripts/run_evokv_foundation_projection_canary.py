from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

from hstu_kvcache.migration.design2_distributed import (
    close_d2_distributed_runtime,
    init_d2_distributed_runtime,
)
from hstu_kvcache.migration.design2_embedding import (
    modulo_embedding_local_id,
    modulo_embedding_local_rows,
    modulo_embedding_owner,
)
from hstu_kvcache.migration.foundation_projection import (
    FoundationProjectedModuloEmbedding,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "evokv_foundation_projection_canary_v1"
PROTOCOL = "evokv_foundation_projection_canary_development_v1"
XP_NUM_EMBEDDINGS = 2_859_836
XP_EMBEDDING_WIDTH = 4_096
XP_OUTPUT_WIDTH = 1_536


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--logical-num-embeddings", type=int, default=128)
    parser.add_argument("--physical-num-embeddings", type=int)
    parser.add_argument("--embedding-width", type=int, default=4096)
    parser.add_argument("--output-width", type=int, default=1536)
    parser.add_argument(
        "--response-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--request-tokens-per-rank", type=int, default=16)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    return parser.parse_args(argv)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _visible_devices() -> tuple[str, ...]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be explicit")
    devices = tuple(part.strip() for part in value.split(","))
    if len(devices) != 2 or any(not part for part in devices):
        raise RuntimeError("exactly two visible CUDA devices are required")
    return devices


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _request_ids(
    *,
    rank: int,
    tokens: int,
    num_embeddings: int,
    device: torch.device,
) -> torch.Tensor:
    values = torch.arange(tokens, dtype=torch.long, device=device)
    if num_embeddings == 1:
        return torch.zeros_like(values)
    return torch.remainder(
        values + rank * tokens + 1,
        num_embeddings - 1,
    ) + 1


def _embedding_rows(
    global_ids: torch.Tensor,
    embedding_width: int,
) -> torch.Tensor:
    dimensions = torch.arange(
        embedding_width,
        dtype=torch.float32,
        device=global_ids.device,
    )
    row_phase = global_ids.float().unsqueeze(1) * 0.0017
    column_phase = dimensions.unsqueeze(0) * 0.0031
    return (
        torch.sin(row_phase + column_phase)
        + 0.5 * torch.cos(row_phase * 0.7 - column_phase * 1.3)
    )


def _projection_weight(
    *,
    embedding_width: int,
    output_width: int,
    device: torch.device,
) -> torch.Tensor:
    inputs = torch.arange(
        embedding_width,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    outputs = torch.arange(
        output_width,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    scale = 1.0 / math.sqrt(embedding_width)
    return torch.sin(
        (inputs + 1.0) * 0.0019 + (outputs + 1.0) * 0.0023
    ).mul_(scale)


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(
        contiguous.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def _validate_args(args: argparse.Namespace) -> None:
    num_embeddings = (
        args.physical_num_embeddings
        if args.physical_num_embeddings is not None
        else args.logical_num_embeddings
    )
    if (
        num_embeddings < 1
        or args.embedding_width < 1
        or args.output_width < 1
        or args.request_tokens_per_rank < 1
        or args.warmup_repeats < 0
        or args.repeats < 1
        or not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
    ):
        raise ValueError("foundation projection canary arguments are invalid")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    visible_devices = _visible_devices()
    runtime = init_d2_distributed_runtime(
        backend="nccl",
        timeout_seconds=args.timeout_seconds,
    )
    try:
        if runtime.world_size != 2 or runtime.device.type != "cuda":
            raise RuntimeError(
                "foundation projection canary requires two CUDA ranks"
            )
        rank = runtime.rank
        world_size = runtime.world_size
        device = runtime.device
        response_dtype = _dtype(args.response_dtype)
        num_embeddings = (
            args.physical_num_embeddings
            if args.physical_num_embeddings is not None
            else args.logical_num_embeddings
        )
        mode = (
            "physical_capacity_canary"
            if args.physical_num_embeddings is not None
            else "logical_smoke"
        )
        local_rows = modulo_embedding_local_rows(
            num_embeddings,
            rank,
            world_size,
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = _memory_snapshot(device)
        torch.cuda.synchronize(device)
        materialization_started = time.perf_counter()
        with torch.no_grad():
            local_weight = torch.empty(
                (local_rows, args.embedding_width),
                dtype=torch.float32,
                device=device,
            )
            local_weight.zero_()
            projection_weight = _projection_weight(
                embedding_width=args.embedding_width,
                output_width=args.output_width,
                device=device,
            )
        torch.cuda.synchronize(device)
        materialization_seconds = (
            time.perf_counter() - materialization_started
        )
        all_request_ids = torch.cat(
            [
                _request_ids(
                    rank=requester,
                    tokens=args.request_tokens_per_rank,
                    num_embeddings=num_embeddings,
                    device=device,
                )
                for requester in range(world_size)
            ]
        )
        owned_ids = torch.unique(
            all_request_ids[
                modulo_embedding_owner(
                    all_request_ids,
                    world_size,
                )
                == rank
            ],
            sorted=True,
        )
        with torch.no_grad():
            local_weight.index_copy_(
                0,
                modulo_embedding_local_id(
                    owned_ids,
                    world_size,
                ),
                _embedding_rows(
                    owned_ids,
                    args.embedding_width,
                ),
            )
        embedding = FoundationProjectedModuloEmbedding(
            local_weight=local_weight,
            projection_weight=projection_weight,
            num_embeddings=num_embeddings,
            rank=rank,
            world_size=world_size,
            response_dtype=response_dtype,
        )
        requested_ids = _request_ids(
            rank=rank,
            tokens=args.request_tokens_per_rank,
            num_embeddings=num_embeddings,
            device=device,
        )
        item_ids = torch.full(
            (1, args.request_tokens_per_rank + 2),
            -1,
            dtype=torch.long,
            device=device,
        )
        item_ids[0, : args.request_tokens_per_rank] = requested_ids
        lengths = torch.tensor(
            [args.request_tokens_per_rank],
            dtype=torch.long,
            device=device,
        )
        for _ in range(args.warmup_repeats):
            embedding.lookup(item_ids, lengths)
        dist.barrier()
        repeat_metrics = []
        last_result = None
        for _ in range(args.repeats):
            last_result = embedding.lookup(item_ids, lengths)
            repeat_metrics.append(last_result.metrics)
        if last_result is None:
            raise RuntimeError("foundation projection lookup did not run")
        expected = torch.zeros_like(last_result.item_vectors)
        expected[0, : args.request_tokens_per_rank] = F.linear(
            _embedding_rows(
                requested_ids,
                args.embedding_width,
            ),
            projection_weight,
        ).to(dtype=response_dtype)
        maximum_error = float(
            torch.max(
                torch.abs(
                    last_result.item_vectors.float() - expected.float()
                )
            ).item()
        )
        tolerance = {
            torch.float16: 0.003,
            torch.bfloat16: 0.02,
            torch.float32: 0.00002,
        }[response_dtype]
        local_correct = maximum_error <= tolerance
        correctness_tensor = torch.tensor(
            1 if local_correct else 0,
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(
            correctness_tensor,
            op=dist.ReduceOp.MIN,
        )
        memory_after = _memory_snapshot(device)
        rank_report = {
            "rank": rank,
            "device": {
                "logical_index": device.index,
                "visible_source": visible_devices[rank],
                "name": torch.cuda.get_device_name(device),
                "uuid": str(
                    torch.cuda.get_device_properties(device).uuid
                ),
            },
            "capacity": embedding.capacity.to_dict(),
            "materialization": {
                "storage_materialized": True,
                "seconds": materialization_seconds,
                "memory_before": memory_before,
                "memory_after": memory_after,
            },
            "request": {
                "tokens": args.request_tokens_per_rank,
                "item_ids_sha256": _tensor_sha256(requested_ids),
                "owned_initialized_rows": owned_ids.numel(),
            },
            "correctness": {
                "passed": local_correct,
                "maximum_absolute_error": maximum_error,
                "absolute_tolerance": tolerance,
                "output_sha256": _tensor_sha256(
                    last_result.item_vectors
                ),
            },
            "timing": {
                "lookup_seconds": _summary(
                    [value.lookup_seconds for value in repeat_metrics]
                ),
                "collective_seconds": _summary(
                    [
                        value.collective_seconds
                        for value in repeat_metrics
                    ]
                ),
                "projection_seconds": _summary(
                    [
                        value.projection_seconds
                        for value in repeat_metrics
                    ]
                ),
            },
            "last_lookup_metrics": last_result.metrics.to_dict(),
        }
        rank_reports: list[dict[str, object] | None] = [
            None for _ in range(world_size)
        ]
        dist.all_gather_object(rank_reports, rank_report)
        if runtime.is_primary:
            complete_reports = [
                value for value in rank_reports if value is not None
            ]
            global_table_bytes = (
                num_embeddings * args.embedding_width * 4
            )
            local_table_sum = sum(
                int(
                    value["capacity"][
                        "local_embedding_parameter_bytes"
                    ]
                )
                for value in complete_reports
            )
            report = {
                "schema": SCHEMA,
                "protocol": PROTOCOL,
                "scientific_result": False,
                "formal_design3": False,
                "mode": mode,
                "purpose": (
                    "Foundation-only owner-projected row-sharded "
                    "embedding capacity and communication canary"
                ),
                "claims": {
                    "owner_projects_before_vector_return": True,
                    "raw_embedding_width_returned": False,
                    "active_or_optimizer_updated_gate": {
                        "evaluated": False,
                        "claimed": False,
                        "status": "not_evaluated",
                    },
                    "trained_checkpoint": {
                        "evaluated": False,
                        "claimed": False,
                        "status": "not_evaluated",
                    },
                },
                "configuration": {
                    "num_embeddings": num_embeddings,
                    "embedding_width": args.embedding_width,
                    "output_width": args.output_width,
                    "response_dtype": args.response_dtype,
                    "world_size": world_size,
                    "request_tokens_per_rank": (
                        args.request_tokens_per_rank
                    ),
                    "warmup_repeats": args.warmup_repeats,
                    "repeats": args.repeats,
                    "xp_geometry": (
                        num_embeddings == XP_NUM_EMBEDDINGS
                        and args.embedding_width == XP_EMBEDDING_WIDTH
                        and args.output_width == XP_OUTPUT_WIDTH
                    ),
                },
                "capacity_ledger": {
                    "global_embedding_parameter_bytes": global_table_bytes,
                    "summed_local_embedding_parameter_bytes": (
                        local_table_sum
                    ),
                    "modulo_shards_cover_global_table_exactly": (
                        local_table_sum == global_table_bytes
                    ),
                    "replicated_projection_parameter_bytes_per_rank": (
                        args.embedding_width * args.output_width * 4
                    ),
                },
                "correctness": {
                    "all_ranks_passed": bool(
                        correctness_tensor.item()
                    ),
                },
                "software": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "script_sha256": _source_sha256(Path(__file__)),
                },
                "ranks": complete_reports,
            }
            output_path = _path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(report, indent=2, sort_keys=True))
        dist.barrier()
    finally:
        close_d2_distributed_runtime(runtime)


if __name__ == "__main__":
    main()
