from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from hstu_kvcache.migration.xp_exact_baseline import (
    load_fixed_inputs,
    run_exact_baseline,
    run_partial_exact_baseline,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/evokv_baselines/x_qk_xp_two_gpu_baseline_v0.json",
    )
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-version", type=int, required=True)
    parser.add_argument(
        "--capacity",
        choices=("resident_m2", "144", "288"),
        required=True,
    )
    parser.add_argument("--method", choices=("s0", "s1"), required=True)
    parser.add_argument(
        "--exact-fraction",
        type=float,
        choices=(0.0, 0.2, 0.5, 1.0),
    )
    parser.add_argument(
        "--endpoint",
        choices=("old", "target"),
        default="target",
    )
    parser.add_argument(
        "--group-target-gib",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--micro-batch-records",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--hash-mode",
        choices=("sampled", "full"),
        default="sampled",
    )
    parser.add_argument("--record-limit", type=int)
    parser.add_argument("--store-path", type=Path)
    parser.add_argument(
        "--store-mode",
        choices=("none", "create", "open"),
        default="none",
    )
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("nccl", "gloo"),
        default="nccl",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=args.backend)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    inputs = load_fixed_inputs(
        args.config,
        args.capacity,
        world_size=world_size,
        record_limit=args.record_limit,
    )
    if args.exact_fraction is not None:
        if args.capacity != "resident_m2":
            raise ValueError("partial exact requires resident_m2")
        result = run_partial_exact_baseline(
            inputs,
            checkpoint_root=args.checkpoint_root,
            checkpoint_version=args.checkpoint_version,
            fraction=args.exact_fraction,
            rank=rank,
            world_size=world_size,
            device=device,
            group_target_bytes=round(
                args.group_target_gib * (1 << 30)
            ),
            micro_batch_records=args.micro_batch_records,
            hash_mode=args.hash_mode,
        )
    else:
        source_manifest = (
            json.loads(args.source_manifest.read_text())
            if args.source_manifest is not None
            else None
        )
        result = run_exact_baseline(
            inputs,
            checkpoint_root=args.checkpoint_root,
            checkpoint_version=args.checkpoint_version,
            rank=rank,
            world_size=world_size,
            device=device,
            method=args.method,
            endpoint=args.endpoint,
            group_target_bytes=round(
                args.group_target_gib * (1 << 30)
            ),
            micro_batch_records=args.micro_batch_records,
            hash_mode=args.hash_mode,
            store_path=args.store_path,
            store_mode=args.store_mode,
            source_manifest=source_manifest,
        )
    if rank == 0:
        if result is None:
            raise RuntimeError("rank zero result is absent")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(
            args.output.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, args.output)
        if args.quiet:
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "capacity": result["capacity_name"],
                        "method": result["method"],
                        "endpoint": result["endpoint"],
                        "records": result["records"],
                        "max_rank_wall_seconds": result[
                            "max_rank_wall_seconds"
                        ],
                        "output_hash": result["output_hash"]["sha256"],
                    },
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
