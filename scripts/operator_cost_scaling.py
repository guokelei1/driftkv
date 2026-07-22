from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from layerwise_validity import load_model
from motivation_validity import seed_everything

from hstu_kvcache.migration import (
    capture_layerwise_state,
    extra_state_numel,
    migrate_suffix_cache,
)
from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-result")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--model-t", type=int, default=5)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    parser.add_argument("--fixed-batch-size", type=int, default=32)
    parser.add_argument("--fixed-seq-len", type=int, default=128)
    parser.add_argument("--suffix-depths", type=int, nargs="+")
    parser.add_argument("--timing-repeats", type=int, default=15)
    parser.add_argument("--output", default="results/scaling/operator_cost_seed0.json")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    if args.run_result is None:
        args.run_result = f"results/validity/core6l_seed{args.seed}.json"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints/validity/core6l_seed{args.seed}"


def checkpoint_dimensions(checkpoint_dir: str) -> tuple[int, int]:
    state = torch.load(
        Path(checkpoint_dir) / "theta_0.pt",
        map_location="cpu",
        weights_only=True,
    )
    return state["item_emb.weight"].shape[0] - 1, state["behavior_emb.embed.weight"].shape[0] - 1


def time_call(fn, device: torch.device, repeats: int):
    fn()
    torch.cuda.synchronize(device)
    values = []
    output = None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return output, values


@torch.inference_mode()
def benchmark_shape(
    current,
    old,
    batch_size: int,
    seq_len: int,
    suffix_depths: list[int],
    device: torch.device,
    repeats: int,
) -> dict:
    item_ids = torch.randint(
        1,
        current.cfg.num_items + 1,
        (batch_size, seq_len),
        device=device,
    )
    behaviors = torch.randint(
        1,
        current.cfg.num_behaviors + 1,
        (batch_size, seq_len),
        device=device,
    )
    time_deltas = torch.rand(batch_size, seq_len, device=device) * 86400.0
    lengths = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)
    old_state = capture_layerwise_state(
        old,
        item_ids,
        behaviors,
        time_deltas,
        lengths,
    )
    fresh = current.compute_kv(item_ids, behaviors, time_deltas, lengths=lengths)
    records = {}
    outputs = {}
    for depth in suffix_depths:
        fn = partial(
            migrate_suffix_cache,
            current,
            old_state,
            item_ids,
            behaviors,
            time_deltas,
            depth,
        )
        output, elapsed = time_call(fn, device, repeats)
        name = "cheap_all" if depth == 0 else "recompute" if depth == len(current.blocks) else f"suffix_{depth}"
        outputs[name] = output
        median = float(np.median(elapsed))
        records[name] = {
            "top_n_full": depth,
            "latency_ms": median,
            "latency_samples_ms": elapsed,
            "ms_per_user": median / batch_size,
            "users_per_second": batch_size * 1000.0 / median,
            "extra_state_numel_per_user": extra_state_numel(old_state, depth) // batch_size,
        }
    full = records["recompute"]["latency_ms"]
    for value in records.values():
        value["ratio_to_recompute"] = value["latency_ms"] / full
    recompute = outputs["recompute"]
    max_abs = max(
        float((recompute.k - fresh.k).abs().max().item()),
        float((recompute.v - fresh.v).abs().max().item()),
    )
    cache_numel = fresh.k.numel() + fresh.v.numel()
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "cache_numel_per_user": cache_numel // batch_size,
        "optimized_full_kv_max_abs": max_abs,
        "configs": records,
    }


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("operator cost scaling requires CUDA")
    torch.cuda.set_device(device)
    seed_everything(args.seed)
    source = json.loads(Path(args.run_result).read_text())
    metadata = source["args"]
    num_items, num_behaviors = checkpoint_dimensions(args.checkpoint_dir)
    old = load_model(
        metadata,
        num_items,
        num_behaviors,
        args.device,
        args.checkpoint_dir,
        0,
    )
    current = load_model(
        metadata,
        num_items,
        num_behaviors,
        args.device,
        args.checkpoint_dir,
        args.model_t,
    )
    num_layers = len(current.blocks)
    suffix_depths = args.suffix_depths or sorted(
        {0, max(1, round(num_layers / 3)), max(1, round(2 * num_layers / 3)), num_layers - 1, num_layers}
    )
    if suffix_depths[0] != 0 or suffix_depths[-1] != num_layers:
        raise ValueError("suffix depths must include 0 and the full model depth")
    points = []
    for seq_len in args.seq_lens:
        point = benchmark_shape(
            current,
            old,
            args.fixed_batch_size,
            seq_len,
            suffix_depths,
            device,
            args.timing_repeats,
        )
        point["axis"] = "sequence_length"
        points.append(point)
        print(f"sequence_length={seq_len}", flush=True)
    for batch_size in args.batch_sizes:
        point = benchmark_shape(
            current,
            old,
            batch_size,
            args.fixed_seq_len,
            suffix_depths,
            device,
            args.timing_repeats,
        )
        point["axis"] = "batch_size"
        points.append(point)
        print(f"batch_size={batch_size}", flush=True)
    result = {
        "protocol": "operator_cost_scaling_v1_resident_cuda_events",
        "seed": args.seed,
        "source_run": args.run_result,
        "checkpoint_dir": args.checkpoint_dir,
        "model_t": args.model_t,
        "operator": "optimized deepest suffix with projection-only terminal layer",
        "input": "synthetic full-length resident GPU tensors; no transfer or allocator timing",
        "timing_repeats": args.timing_repeats,
        "suffix_depths": suffix_depths,
        "points": points,
    }
    save_json(result, args.output)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
