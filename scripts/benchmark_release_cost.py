#!/usr/bin/env python3
"""Measure GPU-only HSTU prefill versus one-token KV-cache append cost.

The script is inert unless ``--run`` is supplied. It uses a deterministic,
untrained random-weight model and synthetic token IDs; no quality, data I/O,
host/device transfer, or checkpoint loading belongs to its timing boundary.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from hstu_kvcache.benchmark import (
    RELEASE_COST_CONFIGURATIONS,
    ReleaseCostConfiguration,
    estimate_release_card_hours,
    make_random_hstu,
)


def _configuration(name: str) -> ReleaseCostConfiguration:
    for configuration in RELEASE_COST_CONFIGURATIONS:
        if configuration.name == name:
            return configuration
    raise ValueError(f"unknown configuration {name!r}")


def prepare_model_checkpoint(
    output_dir: Path, configuration: ReleaseCostConfiguration, *, seed: int
) -> Path:
    """Persist the deterministic random model on CPU; this never touches CUDA."""
    path = output_dir / "models" / configuration.name / f"random_seed{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        model = make_random_hstu(configuration, seed=seed)
        torch.save(
            {
                "format": "evokv_random_release_cost_model_v1",
                "configuration": asdict(configuration),
                "seed": seed,
                "state_dict": model.state_dict(),
            },
            path,
        )
    return path


def _load_model(path: Path, configuration: ReleaseCostConfiguration, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "evokv_random_release_cost_model_v1":
        raise ValueError(f"unexpected checkpoint format in {path}")
    if payload.get("configuration") != asdict(configuration):
        raise ValueError(f"checkpoint configuration differs from requested configuration: {path}")
    model = make_random_hstu(configuration, seed=int(payload["seed"]))
    model.load_state_dict(payload["state_dict"])
    # Keep parameters in FP32 and execute under BF16 autocast. This matches the
    # foundation path and lets TemporalEncoder's FP32 frequency construction
    # coexist with the mixed-precision linear layers.
    return model.to(device=device).eval()


def _synthetic_batch(*, batch_size: int, sequence_length: int, device: torch.device):
    """Make one fixed B x L release batch before GPU timing begins."""
    generator = torch.Generator(device="cpu").manual_seed(9_100)
    return (
        torch.randint(1, 50_000, (batch_size, sequence_length), generator=generator).to(device),
        torch.randint(1, 8, (batch_size, sequence_length), generator=generator).to(device),
        torch.rand((batch_size, sequence_length), generator=generator).to(device),
    )


def _gpu_seconds(operation, device: torch.device, repetitions: int) -> float:
    """Mean GPU execution time measured with CUDA events only."""
    elapsed_ms = 0.0
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            operation()
        end.record()
        end.synchronize()
        elapsed_ms += start.elapsed_time(end)
    return elapsed_ms / repetitions / 1_000.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=[c.name for c in RELEASE_COST_CONFIGURATIONS],
                        default=RELEASE_COST_CONFIGURATIONS[0].name)
    parser.add_argument("--sequence-length", type=int,
                        help="L tokens per user; default is the selected configuration context")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--target-users", type=int, default=10_000_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=Path("results/release_cost_random_weight_v1"))
    parser.add_argument("--prepare-model", action="store_true",
                        help="write the random checkpoint on CPU only")
    parser.add_argument("--run", action="store_true", help="perform the explicit GPU-only timing")
    return parser


def main() -> None:
    args = _parser().parse_args()
    configuration = _configuration(args.configuration)
    sequence_length = args.sequence_length or configuration.context_length
    if not 2 <= sequence_length <= configuration.context_length:
        raise SystemExit("--sequence-length must be in [2, selected configuration context]")
    if args.batch_size < 1 or args.repetitions < 1:
        raise SystemExit("--batch-size and --repetitions must be positive")

    checkpoint = prepare_model_checkpoint(args.output_dir, configuration, seed=args.seed) \
        if (args.prepare_model or args.run) else args.output_dir / "models" / configuration.name / f"random_seed{args.seed}.pt"
    plan = {
        "format": "evokv_random_release_cost_plan_v2",
        "configuration": asdict(configuration),
        "sequence_length_per_user": sequence_length,
        "batch_size_users": args.batch_size,
        "repetitions": args.repetitions,
        "target_users": args.target_users,
        "checkpoint": str(checkpoint),
        "timing_boundary": "CUDA event time for model execution only",
        "recompute": "Current model prefill over tokens 1 through L",
        "reuse": "Append-only Current token L, conditioned on a precomputed tokens 1 through L-1 KV cache",
    }
    if not args.run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if not torch.cuda.is_available():
        raise SystemExit("--run requires an available CUDA GPU")

    device = torch.device(args.device)
    model = _load_model(checkpoint, configuration, device)
    items, behaviors, deltas = _synthetic_batch(
        batch_size=args.batch_size, sequence_length=sequence_length, device=device
    )
    # Prefix construction is intentionally outside the Reuse timing boundary:
    # it is the retained cache that already exists at release time.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prefix_cache = model.compute_kv(items[:, :-1], behaviors[:, :-1], deltas[:, :-1])
        # Kernel warm-up is outside both measurements.
        model.compute_kv(items, behaviors, deltas)
        model.forward_with_cache_new_kv(
            prefix_cache, items[:, -1:], behaviors[:, -1:], deltas[:, -1:]
        )
    torch.cuda.synchronize(device)

    recompute_seconds = _gpu_seconds(
        lambda: model.compute_kv(items, behaviors, deltas), device, args.repetitions
    )
    reuse_seconds = _gpu_seconds(
        lambda: model.forward_with_cache_new_kv(
            prefix_cache, items[:, -1:], behaviors[:, -1:], deltas[:, -1:]
        ),
        device,
        args.repetitions,
    )
    # The mean batch time is extrapolated from the number of users in one batch.
    recompute = estimate_release_card_hours(
        elapsed_seconds=recompute_seconds, sampled_users=args.batch_size, target_users=args.target_users
    )
    reuse = estimate_release_card_hours(
        elapsed_seconds=reuse_seconds, sampled_users=args.batch_size, target_users=args.target_users
    )
    plan["results"] = {
        "recompute": recompute.to_dict(),
        "reuse": reuse.to_dict(),
        "recompute_over_reuse_ratio": recompute.card_hours / reuse.card_hours,
    }
    result_path = args.output_dir / f"{configuration.name}_L{sequence_length}_gpu_only.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
