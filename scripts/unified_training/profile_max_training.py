#!/usr/bin/env python3
"""Bounded Max step profiling on real, already-collated CPU fixtures.

This is a disposable performance diagnostic, not a release training entry point.
It never saves candidate weights or edits the formal training implementation.
Run under torchrun after the formal four-rank job has released the GPUs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from train_yambda500m_foundation_fsdp import FoundationForward, contract_model_config, sha256_file
from hstu_kvcache.models import HSTU
from hstu_kvcache.models.backend_info import attention_backend_info

FIELDS = ("item_ids", "behaviors", "time_deltas", "candidate_ids",
          "query_time_deltas", "lengths", "labels", "weights")
PHASES = ("data_H2D", "forward", "loss", "zero_grad", "backward", "optimizer")
SOURCE_PATHS = [Path(__file__), ROOT / "scripts/train_yambda500m_foundation_fsdp.py",
                ROOT / "src/hstu_kvcache/training/foundation.py"] + [
    ROOT / "src/hstu_kvcache/models" / name for name in
    ("hstu.py", "attention.py", "triton_attention.py", "backend_info.py", "block.py", "embeddings.py")]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def source_snapshot() -> dict:
    return {str(path.resolve().relative_to(ROOT)): sha256_file(path) for path in SOURCE_PATHS}


def fixture_batches(directory: Path, rank: int) -> list[dict[str, torch.Tensor]]:
    with np.load(directory / f"rank{rank}.npz", allow_pickle=False) as saved:
        indices = sorted({int(key.split("_")[0][1:]) for key in saved.files})
        return [{name: torch.from_numpy(saved[f"b{index:03d}_{name}"].copy())
                 for name in FIELDS} for index in indices]


def statistics_ms(values: list[float]) -> dict:
    return {"median_ms": statistics.median(values), "mean_ms": statistics.mean(values),
            "min_ms": min(values), "max_ms": max(values)}


class StepTiming:
    """Events are read only after the final whole-step synchronization."""

    def __init__(self, labels: bool):
        self.labels = labels
        self.events = {}
        self.cpu_ms = {}

    @contextmanager
    def phase(self, name):
        first, last = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        first.record()
        started = time.perf_counter()
        with torch.profiler.record_function(f"phase::{name}") if self.labels else nullcontext():
            yield
        self.cpu_ms[name] = (time.perf_counter() - started) * 1000
        last.record()
        self.events[name] = (first, last)

    def result(self):
        return {"cuda_interval_ms": {name: first.elapsed_time(last)
                                      for name, (first, last) in self.events.items()},
                "cpu_enqueue_ms": self.cpu_ms}


def run_step(model, optimizer, batch_cpu, device, global_batch, *, profile=False, check_gradients=False):
    timing = StepTiming(profile)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with timing.phase("data_H2D"):
        batch = {name: value.to(device) for name, value in batch_cpu.items()}
    with timing.phase("forward"):
        logits = model(*(batch[name] for name in FIELDS[:6]))[:, 0]
    with timing.phase("loss"):
        per_request = F.binary_cross_entropy_with_logits(logits, batch["labels"], reduction="none")
        loss = (per_request * batch["weights"]).sum() * (dist.get_world_size() / global_batch)
    with timing.phase("zero_grad"):
        optimizer.zero_grad(set_to_none=True)
    with timing.phase("backward"):
        loss.backward()
    if check_gradients:
        # Only the disposable canary performs this full local-shard scan.
        finite = torch.ones((), device=device, dtype=torch.int32)
        for parameter in model.parameters():
            if parameter.grad is not None:
                finite *= torch.isfinite(parameter.grad).all().to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise RuntimeError("Nonfinite canary gradient; stop without saving weights")
    with timing.phase("optimizer"):
        optimizer.step()
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - started) * 1000
    loss_value = float(loss.detach())
    if not math.isfinite(loss_value):
        raise RuntimeError("Nonfinite diagnostic loss; stop without saving weights")
    return {"wall_ms": wall_ms, "loss": loss_value, **timing.result()}, logits.detach().cpu()


@contextmanager
def detail_labels(raw):
    def labeled(function, name):
        def call(*args, **kwargs):
            with torch.profiler.record_function(name):
                return function(*args, **kwargs)
        return call

    with patch.object(raw, "forward", labeled(raw.forward, "model::prefix")), \
         patch.object(raw, "_score_cc_from_prefix_cache", labeled(raw._score_cc_from_prefix_cache, "model::cc_queries")), \
         patch.object(raw, "lookup_item_embeddings", labeled(raw.lookup_item_embeddings, "model::item_lookup")):
        yield


def interval_union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def kernel_category(name: str, cat: str) -> str:
    lowered = name.lower()
    if "nccl" in lowered:
        return "communication"
    if "memcpy" in cat.lower() or "memcpy" in lowered:
        return "memory_copy"
    if "memset" in cat.lower() or "memset" in lowered:
        return "memory_zero"
    if "adam" in lowered:
        return "optimizer"
    if "embedding" in lowered:
        return "embedding"
    if lowered.startswith(("_forward", "_backward_q", "_backward_kv")) or any(word in lowered for word in ("_pma", "_attention", "_attn")):
        return "attention"
    if any(word in lowered for word in ("gemm", "gemv", "matmul")):
        return "matrix_multiply"
    if any(word in lowered for word in ("elementwise", "vectorized", "reduce", "index", "scatter")):
        return "elementwise_reduction_index"
    return "other"


def summarize_profile(profiler, trace_path, profile_steps):
    # CPU operator self device times and kernel trace durations answer different
    # questions. Retain both, and never add inclusive module/phase totals.
    operators = []
    for event in profiler.key_averages():
        if event.device_type != torch.autograd.DeviceType.CPU:
            continue
        operators.append({"name": event.key, "calls": event.count,
                          "self_cpu_ms": event.self_cpu_time_total / 1000,
                          "total_cpu_ms": event.cpu_time_total / 1000,
                          "self_device_ms": event.self_device_time_total / 1000,
                          "total_device_ms": event.device_time_total / 1000})
    operators.sort(key=lambda row: row["self_device_ms"], reverse=True)
    trace = json.loads(trace_path.read_text())
    kernels = []
    for event in trace["traceEvents"]:
        cat = event.get("cat", "")
        if event.get("ph") != "X" or cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        start, duration = event["ts"], event["dur"]
        kernels.append({"name": event["name"], "category": kernel_category(event["name"], cat),
                        "start_us": start, "duration_us": duration,
                        "stream": event.get("args", {}).get("stream", event.get("tid"))})
    if not kernels:
        raise RuntimeError("CUDA profiler returned no GPU activities; trace attribution is unavailable")
    kernels.sort(key=lambda row: row["start_us"])
    categories = {}
    for kernel in kernels:
        entry = categories.setdefault(kernel["category"], {"kernel_count": 0, "sum_kernel_ms": 0.0})
        entry["kernel_count"] += 1
        entry["sum_kernel_ms"] += kernel["duration_us"] / 1000
    comm = [(k["start_us"], k["start_us"] + k["duration_us"]) for k in kernels if k["category"] == "communication"]
    compute = [(k["start_us"], k["start_us"] + k["duration_us"]) for k in kernels if k["category"] != "communication"]
    all_intervals = comm + compute
    span = max(end for _, end in all_intervals) - min(start for start, _ in all_intervals) if all_intervals else 0
    relevant = [row for row in operators if row["name"].startswith(("phase::", "model::", "FullyShardedDataParallel"))
                or any(word in row["name"].lower() for word in ("embedding", "adam", "all_gather", "reduce_scatter"))]
    # Chronological kernel records remain local; the compact summary is retained.
    write_json(trace_path.with_name(trace_path.stem + ".kernels.json"), kernels)
    return {"profile_steps": profile_steps, "trace_path": str(trace_path),
            "categories": categories, "top_operators_by_self_device_ms": operators[:35],
            "selected_inclusive_operators": relevant,
            "timeline": {"device_span_ms": span / 1000,
                         "busy_union_ms": interval_union(all_intervals) / 1000,
                         "communication_union_ms": interval_union(comm) / 1000,
                         "noncommunication_union_ms": interval_union(compute) / 1000,
                         "communication_noncommunication_overlap_ms": (interval_union(comm) + interval_union(compute) - interval_union(all_intervals)) / 1000},
            "caveats": ["Kernel categories use names and are indicative; phase scopes provide stronger attribution.",
                        "Kernel duration sums may overlap across streams; union and overlap values are also reported.",
                        "CPU operators and phase/module device totals are nested and must not be summed.",
                        "Profiler runs separately after clean timing; trace overhead is excluded from benchmark medians."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--launch-contract", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", default="torch_default,auto_default,auto_fused")
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument("--canary-steps", type=int, default=0)
    parser.add_argument("--torch-cpu-threads", type=int, default=4)
    parser.add_argument("--cpu-affinity-by-rank")
    args = parser.parse_args()
    variants = args.variants.split(",")
    if not set(variants) <= {"torch_default", "auto_default", "auto_fused"}:
        raise ValueError("Only the three prospective backend/optimizer cells are supported")
    if args.canary_steps:
        args.warmup_steps, args.timed_steps, args.profile_steps = 0, args.canary_steps, 0
    if args.timed_steps < 1 or min(args.warmup_steps, args.profile_steps) < 0:
        raise ValueError("Expected positive timed steps and nonnegative warmup/profile steps")
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.cpu_affinity_by_rank:
        os.sched_setaffinity(0, {int(cpu) for cpu in args.cpu_affinity_by_rank.split(";")[local_rank].split(",")})
    torch.set_num_threads(args.torch_cpu_threads)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        fixture_config = json.loads((args.fixture_dir / "configuration.json").read_text())
        if fixture_config["launch_contract_sha256"] != sha256_file(args.launch_contract):
            raise RuntimeError("Fixture and selected launch contract differ")
        fixture_rank = next(row for row in fixture_config["ranks"] if row["rank"] == rank)
        if sha256_file(args.fixture_dir / f"rank{rank}.npz") != fixture_rank["file_sha256"]:
            raise RuntimeError("Fixture NPZ bytes differ from the retained configuration")
        batches = fixture_batches(args.fixture_dir, rank)
        launch = yaml.safe_load(args.launch_contract.read_text())
        cfg, _, _ = contract_model_config(launch, oov_buckets=int(launch["model"]["oov_buckets"]))
        global_batch = int(fixture_config["global_batch_size"])
        if world != 4 or global_batch != 80 or cfg.num_layers != 16:
            raise RuntimeError("This diagnostic fixes Max16L, four ranks, global batch80")
        if any(batch["labels"].shape[0] * world != global_batch for batch in batches):
            raise RuntimeError("Fixture batch sizes differ")
        parent = torch.load(args.parent, map_location="cpu", mmap=True, weights_only=False) if rank == 0 else None
        if rank == 0 and parent["config"] != asdict(cfg):
            raise RuntimeError("Parent and contract model configs differ")
        if rank == 0 and (parent.get("version") != "v1" or parent.get("progress") != 1.0
                          or parent.get("training_epochs_completed") != 1):
            raise RuntimeError("Expected the completed, frozen V1 epoch1 parent")
        if args.parent.resolve() != (ROOT / launch["frozen_inputs"]["parent_v1_checkpoint"]).resolve():
            raise RuntimeError("Diagnostic must use the frozen V1 parent")
        if fixture_config["parent_checkpoint_sha256"] != launch["frozen_inputs"]["parent_v1_checkpoint_sha256"]:
            raise RuntimeError("Fixture parent binding differs from the launch contract")
        if rank == 0 and sha256_file(args.parent) != fixture_config["parent_checkpoint_sha256"]:
            raise RuntimeError("Parent checkpoint bytes differ from the frozen hash")
        if asdict(cfg) != fixture_config["model_config"]:
            raise RuntimeError("Fixture and contract model configs differ")
        recipe = launch["training"]
        learning_rate = float(recipe.get("update_learning_rate", recipe.get("learning_rate", 2e-4)))
        weight_decay = float(recipe.get("weight_decay", 1e-4))
        frozen_sources = source_snapshot()
        provenance = {"script_sha256": sha256_file(Path(__file__)),
                      "source_sha256": frozen_sources,
                      "trainer_sha256": sha256_file(ROOT / "scripts/train_yambda500m_foundation_fsdp.py"),
                      "contract_sha256": sha256_file(args.launch_contract),
                      "fixture_config_sha256": sha256_file(args.fixture_dir / "configuration.json"),
                      "parent_path": str(args.parent.resolve()),
                      "parent_checkpoint_sha256": fixture_config.get("parent_checkpoint_sha256"),
                      "parent_hash_verified": True,
                      "config": asdict(cfg), "global_batch": global_batch, "world_size": world,
                      "learning_rate": learning_rate, "weight_decay": weight_decay,
                      "cpu_affinity_by_rank": args.cpu_affinity_by_rank,
                      "torch_cpu_threads": args.torch_cpu_threads,
                      "gpu_name": torch.cuda.get_device_name(device),
                      "fixture": fixture_config,
                      "timing_scope": "CPU prepared fixture H2D + original FoundationForward/loss/backward/FSDP/AdamW; excludes original CPU row lookup/collate, startup, checkpoint save, trace overhead",
                      "cuda_phase_scope": "Default-stream elapsed intervals; side-stream waits contribute at their consumers. No intermediate device synchronization. Not a sum of independent kernel costs.",
                      "formal_training": False, "saved_candidate_weights": False,
                      "canary": bool(args.canary_steps), "variants": variants,
                      "warmup_steps": args.warmup_steps, "timed_steps": args.timed_steps,
                      "profile_steps": args.profile_steps}
        results = []
        for variant in variants:
            if source_snapshot() != frozen_sources:
                raise RuntimeError("Execution source changed before a diagnostic cell")
            backend, optimizer_name = variant.split("_")
            os.environ["EVOKV_ATTENTION_BACKEND"] = backend
            torch.manual_seed(17)
            torch.cuda.manual_seed_all(17)
            raw = HSTU(cfg)
            if rank == 0:
                raw.load_state_dict(parent["model"], strict=True)
            model = FSDP(FoundationForward(raw), device_id=device, use_orig_params=True,
                         sharding_strategy=ShardingStrategy.FULL_SHARD,
                         mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32),
                         limit_all_gathers=True, sync_module_states=True)
            optimizer_args = {"fused": True} if optimizer_name == "fused" else {}
            optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                          lr=learning_rate, weight_decay=weight_decay, **optimizer_args)
            torch.cuda.reset_peak_memory_stats(device)
            # Reset CUDA dropout RNG after FSDP initialization/broadcast in every cell.
            torch.cuda.manual_seed_all(17)
            rows, initial_logits = [], None
            for step in range(args.warmup_steps + args.timed_steps):
                row, logits = run_step(model, optimizer, batches[step % len(batches)], device, global_batch,
                                       check_gradients=bool(args.canary_steps))
                if step == 0:
                    initial_logits = logits.tolist()
                row.update(step=step, fixture_batch=step % len(batches), warmup=step < args.warmup_steps)
                rows.append(row)
                if rank == 0:
                    print(json.dumps({"variant": variant, "rank": rank, **row}), flush=True)
            parameter_samples = {}
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.numel():
                        count = min(16, parameter.numel())
                        index = torch.arange(count, device=device, dtype=torch.long) * (parameter.numel() - 1) // max(count - 1, 1)
                        parameter_samples[name] = parameter.detach().flatten()[index].float().cpu().tolist()
            local = {"rank": rank, "rows": rows, "initial_logits": initial_logits,
                     "final_local_parameter_samples": parameter_samples,
                     "optimizer_defaults": optimizer.defaults,
                     "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                     "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}
            if args.profile_steps:
                dist.barrier()
                profile_rows = []
                with detail_labels(raw), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False, profile_memory=False, with_stack=False,
                ) as profiler:
                    for step in range(args.profile_steps):
                        batch_index = (args.warmup_steps + step * args.timed_steps // args.profile_steps) % len(batches)
                        profile_row, _ = run_step(model, optimizer, batches[batch_index], device, global_batch, profile=True)
                        profile_row["fixture_batch"] = batch_index
                        profile_rows.append(profile_row)
                        profiler.step()
                trace_path = args.output / f"{variant}.rank{rank}.trace.json"
                profiler.export_chrome_trace(str(trace_path))
                local["profile"] = summarize_profile(profiler, trace_path, args.profile_steps)
                local["profile"]["rows"] = profile_rows
                del profiler
            if source_snapshot() != frozen_sources:
                raise RuntimeError("Execution source changed during a diagnostic cell")
            local["source_hashes_unchanged"] = True
            gathered = [None] * world if rank == 0 else None
            dist.gather_object(local, gathered, dst=0)
            if rank == 0:
                max_steps = [{key: max(item["rows"][step][key] for item in gathered) for key in ("wall_ms",)}
                             for step in range(args.warmup_steps, len(rows))]
                cell = {"variant": variant, "attention_execution": attention_backend_info(), "ranks": gathered,
                        "max_rank_wall_ms_by_step": [row["wall_ms"] for row in max_steps],
                        "max_rank_wall": statistics_ms([row["wall_ms"] for row in max_steps]),
                        "max_rank_cuda_phase": {name: statistics_ms([max(item["rows"][step]["cuda_interval_ms"][name] for item in gathered)
                                                                     for step in range(args.warmup_steps, len(rows))]) for name in PHASES}}
                results.append(cell)
                write_json(args.output / f"{variant}.summary.json", cell)
                write_json(args.output / "summary.json", {**provenance, "cells": results})
                print(json.dumps({"completed_variant": variant, "max_rank_wall": cell["max_rank_wall"],
                                  "max_rank_cuda_phase": cell["max_rank_cuda_phase"]}), flush=True)
            dist.barrier()
            del model, raw, optimizer, local, rows, logits, parameter, index
            gc.collect()
            torch.cuda.empty_cache()
        if rank == 0:
            print(json.dumps({"status": "completed", "output": str(args.output)}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
