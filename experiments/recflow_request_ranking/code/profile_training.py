"""Bounded actual Max/small-model throughput; diagnostic states never become releases."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .batches import GlobalBatchSampler, RequestBatches
from .communication import synchronize_and_clip
from .configuration import load_proposal
from .data_common import json_save
from .model import RequestRanker
from .objective import ranking_loss
from .runtime import distributed_init, save_checkpoint, source_binding, to_device


def main():
    begin = time.monotonic()
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/processed/recflow_request_ranking_v1"))
    p.add_argument("--scale", choices=["pilot", "max"], default="max")
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-probe", action="store_true")
    args = p.parse_args()
    rank, world, device = distributed_init(17)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    vocabulary = "vocabulary_max" if args.scale == "max" else "vocabulary"
    model = RequestRanker(args.data / vocabulary, args.scale).to(device).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    if world > 1:
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0)
    initialization = time.monotonic()-begin
    results = []
    for day in [12, 24]:
        dataset = RequestBatches(args.data, [day], role=0, limit=(args.steps+args.warmup)*64)
        sampler = GlobalBatchSampler(len(dataset), micro_batch=args.micro_batch, world_size=world, rank=rank)
        options = {"multiprocessing_context": "spawn", "persistent_workers": True, "prefetch_factor": 2} if args.workers else {}
        loader = DataLoader(dataset, batch_size=None, sampler=sampler, pin_memory=True, num_workers=args.workers,
                            generator=torch.Generator().manual_seed(17+rank), **options)
        optimizer.zero_grad(set_to_none=True)
        measurements, io_times, sync_times, touched, rows = [], [], [], [], []
        torch.cuda.synchronize()
        previous, step_start = time.monotonic(), time.monotonic()
        for i, cpu_batch in enumerate(loader):
            io_times.append(time.monotonic()-previous)
            batch = to_device(cpu_batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = model(batch)
                loss, _ = ranking_loss(scores, batch["labels"], batch["candidate_mask"],
                                       listwise_weight=.1 if day >= 24 else 0,
                                       global_active_requests=64, ddp_world_size=world, validate=False)
            loss.backward()
            touched.extend([model.features.map_ids(batch["history_features"][..., 0]).reshape(-1),
                            model.features.map_ids(batch["candidate_features"][..., 0]).reshape(-1)])
            if (i+1) % sampler.accumulation == 0:
                torch.cuda.synchronize()
                sync_start = time.monotonic()
                _, communicated = synchronize_and_clip(model, touched, world)
                torch.cuda.synchronize()
                sync_times.append(time.monotonic()-sync_start)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                touched = []
                torch.cuda.synchronize()
                elapsed = time.monotonic()-step_start
                measurements.append(elapsed)
                rows.append(communicated)
                step_start = time.monotonic()
                if rank == 0 and len(measurements) % 10 == 0:
                    print(json.dumps({"day": day, "steps": len(measurements), "last_seconds": elapsed,
                                      "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30}), flush=True)
            previous = time.monotonic()
        timings = np.asarray(measurements[args.warmup:])
        report = {"day": day, "timed_steps": len(timings), "mean_seconds": float(timings.mean()),
                  "p50_seconds": float(np.median(timings)), "p95_seconds": float(np.quantile(timings,.95)),
                  "sync_and_clip_mean_seconds": float(np.mean(sync_times[args.warmup:])),
                  "loader_wait_total_seconds": sum(io_times), "communicated_rows_mean": float(np.mean(rows)),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "all_step_seconds": measurements}
        gathered = [None]*world
        if world > 1:
            dist.all_gather_object(gathered, report)
        else:
            gathered[0] = report
        results.append({"day": day, "ranks": gathered, "conservative_seconds_per_step": max(r["p95_seconds"] for r in gathered)})
        del loader
    save_seconds = None
    if args.save_probe:
        save_start = time.monotonic()
        save_checkpoint(args.output / "diagnostic_state.pt", model, optimizer, scheduler,
                        {"quality_use_permitted": False, "scope": "actual optimizer/save/restore timing probe",
                         "scale": args.scale, "proposal": load_proposal()},
                        started=begin, prior_elapsed=0, rank=rank, world=world)
        save_seconds = time.monotonic()-save_start
    if rank == 0:
        json_save(args.output / "summary.json", {"status": "passed", "scale": args.scale, "world_size": world,
                  "parameters": sum(p.numel() for p in model.parameters()), "item_embedding_parameters": model.features.item.weight.numel(),
                  "global_batch": 64, "micro_batch": args.micro_batch, "initialization_seconds": initialization,
                  "save_probe_seconds": save_seconds, "phases": results,
                  "source_binding": source_binding(args.data, vocabulary),
                  "scope": "throughput with real data, full table, backward, exact row communication and dense AdamW; no model-quality claim",
                  "wall_seconds": time.monotonic()-begin})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
