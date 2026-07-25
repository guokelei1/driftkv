from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from hstu_kvcache.migration import (
    CohortStreamingExecutor,
    CompiledCacheAdapter,
    MigrationCapsuleBatch,
    MigrationProgram,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="streamkv_vertical_slice_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--cohort-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--inflight", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--pre-pin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output")
    return parser.parse_args()


def make_program(args: argparse.Namespace) -> MigrationProgram:
    generator = torch.Generator().manual_seed(args.seed)
    weights = torch.randn(
        args.num_layers,
        args.hidden_size,
        2 * args.hidden_size,
        generator=generator,
        dtype=torch.float32,
    ) / args.hidden_size**0.5
    biases = torch.randn(
        args.num_layers,
        2 * args.hidden_size,
        generator=generator,
        dtype=torch.float32,
    )
    return MigrationProgram(
        source_version="theta-old",
        target_version="theta-current",
        adapter=CompiledCacheAdapter(
            weights=weights,
            biases=biases,
            source_rank=0,
            ridge=0.0,
        ),
    )


def make_batches(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[MigrationCapsuleBatch, ...]:
    generator = torch.Generator().manual_seed(args.seed + 1)
    dtype = getattr(torch, args.dtype)
    normed = torch.randn(
        args.num_layers,
        args.cohort_size,
        args.seq_len,
        args.hidden_size,
        generator=generator,
        dtype=dtype,
    )
    lengths = torch.randint(
        max(1, args.seq_len // 2),
        args.seq_len + 1,
        (args.cohort_size,),
        generator=generator,
    )
    capsule = MigrationCapsuleBatch(
        record_ids=tuple(range(args.cohort_size)),
        migration_anchor_version="theta-old",
        normed=normed,
        lengths=lengths,
    )
    batches = capsule.split(args.batch_size)
    if args.pre_pin and device.type == "cuda":
        batches = tuple(batch.pin_memory() for batch in batches)
    return batches


def benchmark(
    program: MigrationProgram,
    batches: tuple[MigrationCapsuleBatch, ...],
    device: torch.device,
    inflight: int,
    warmup_repeats: int,
    timing_repeats: int,
) -> dict:
    executor = CohortStreamingExecutor(
        program,
        device=device,
        max_inflight_batches=inflight,
    )
    for _ in range(warmup_repeats):
        executor.run(batches)
    elapsed = []
    records_per_second = []
    tokens_per_second = []
    gib_per_second = []
    report = None
    for index in range(timing_repeats):
        current = executor.run(batches)
        elapsed.append(current.metrics.elapsed_seconds)
        records_per_second.append(current.metrics.records_per_second)
        tokens_per_second.append(current.metrics.tokens_per_second)
        gib_per_second.append(current.metrics.gib_per_second)
        if index == timing_repeats - 1:
            report = current
        else:
            del current
    if report is None:
        raise RuntimeError("benchmark produced no report")
    max_padding_abs = 0.0
    for batch in report.batches:
        positions = torch.arange(batch.cache.seq_len).unsqueeze(0)
        invalid = positions >= batch.lengths.unsqueeze(1)
        if bool(torch.any(invalid)):
            mask = invalid.unsqueeze(0).unsqueeze(-1)
            max_padding_abs = max(
                max_padding_abs,
                float(batch.cache.k.masked_select(mask).abs().max()),
                float(batch.cache.v.masked_select(mask).abs().max()),
            )
    return {
        "max_inflight_batches": inflight,
        "elapsed_seconds": elapsed,
        "median_elapsed_seconds": statistics.median(elapsed),
        "median_records_per_second": statistics.median(records_per_second),
        "median_tokens_per_second": statistics.median(tokens_per_second),
        "median_gib_per_second": statistics.median(gib_per_second),
        "input_bytes": report.metrics.input_bytes,
        "output_bytes": report.metrics.output_bytes,
        "batch_count": report.metrics.batch_count,
        "record_count": report.metrics.record_count,
        "token_count": report.metrics.token_count,
        "auto_pinned_batches": report.metrics.auto_pinned_batches,
        "max_padding_abs": max_padding_abs,
    }


def main() -> None:
    args = parse_args()
    if args.num_layers < 1 or args.hidden_size < 1 or args.seq_len < 1:
        raise ValueError("model dimensions must be positive")
    if args.cohort_size < 1 or args.batch_size < 1:
        raise ValueError("cohort and batch sizes must be positive")
    if args.warmup_repeats < 0 or args.timing_repeats < 1:
        raise ValueError("repeat counts are invalid")
    if any(value < 1 for value in args.inflight):
        raise ValueError("inflight values must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    program = make_program(args)
    batches = make_batches(args, device)
    points = [
        benchmark(
            program,
            batches,
            device,
            inflight,
            args.warmup_repeats,
            args.timing_repeats,
        )
        for inflight in args.inflight
    ]
    result = {
        "protocol": args.protocol,
        "evidence_status": "synthetic systems diagnostic; not a paper result",
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU"
        ),
        "seed": args.seed,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "seq_len": args.seq_len,
        "cohort_size": args.cohort_size,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "pre_pin": args.pre_pin,
        "warmup_repeats": args.warmup_repeats,
        "timing_repeats": args.timing_repeats,
        "points": points,
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
