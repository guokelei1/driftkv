from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from hstu_kvcache.migration.xp_exact_baseline import load_fixed_inputs
from hstu_kvcache.migration.xp_m2_lookup_baseline import (
    load_lookup_checkpoint,
    run_lookup_communication_baseline,
)

ROOT = Path(__file__).resolve().parents[1]


def _fractions(value: str) -> tuple[float, ...]:
    fractions = tuple(float(part) for part in value.split(","))
    if not fractions or any(not 0.0 <= part <= 1.0 for part in fractions):
        raise argparse.ArgumentTypeError("fractions must lie in [0,1]")
    if len(fractions) != len(set(fractions)):
        raise argparse.ArgumentTypeError("fractions must be unique")
    return fractions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/evokv_baselines/x_qk_xp_two_gpu_baseline_v0.json",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "checkpoints/evokv_xp_qk_e4096_h1536/seed0",
    )
    parser.add_argument("--checkpoint-version", type=int, default=2)
    parser.add_argument(
        "--fractions",
        type=_fractions,
        default=(0.0, 0.2, 0.5, 1.0),
    )
    parser.add_argument("--micro-batch-records", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--record-limit", type=int)
    parser.add_argument("--copy-chunk-rows", type=int, default=8192)
    parser.add_argument("--skip-checkpoint-artifact-hash", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("nccl",), default="nccl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size != 2:
        raise ValueError("XP M2 development baseline requires two ranks")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=args.backend, device_id=device)
    output_exists = [args.output.exists() if rank == 0 else None]
    dist.broadcast_object_list(output_exists, src=0)
    if bool(output_exists[0]):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    inputs = load_fixed_inputs(
        args.config,
        "resident_m2",
        world_size=world_size,
        record_limit=args.record_limit,
    )
    embedding, checkpoint = load_lookup_checkpoint(
        inputs,
        args.checkpoint_root,
        args.checkpoint_version,
        rank=rank,
        world_size=world_size,
        device=device,
        verify_hash=not args.skip_checkpoint_artifact_hash,
        copy_chunk_rows=args.copy_chunk_rows,
    )
    result = run_lookup_communication_baseline(
        inputs,
        embedding,
        fractions=args.fractions,
        rank=rank,
        world_size=world_size,
        micro_batch_records=args.micro_batch_records,
        warmup=args.warmup,
        repeats=args.repeats,
        checkpoint_binding=checkpoint,
    )
    if rank == 0:
        if result is None:
            raise RuntimeError("rank-zero XP M2 result is absent")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "records": result["records"],
                    "fractions": result["fractions"],
                    "cells": [
                        {
                            "fraction": cell["retained_budget"][
                                "fraction_requested"
                            ],
                            "realized": cell["retained_budget"][
                                "fraction_realized"
                            ],
                            "complete_tokens": cell["complete_wave"][
                                "requested_tokens"
                            ],
                            "complete_seconds": cell["complete_wave"][
                                "timing"
                            ]["median_max_rank_seconds"],
                        }
                        for cell in result["cells"]
                    ],
                },
                sort_keys=True,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
