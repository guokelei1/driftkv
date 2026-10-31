"""Actual full-table restore, update, and FP32 evaluation-memory diagnostic."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .batches import GlobalBatchSampler, RequestBatches
from .communication import synchronize_and_clip
from .data_common import json_save, sha256
from .model import RequestRanker
from .objective import ranking_loss
from .runtime import distributed_init, load_checkpoint, source_binding, to_device


def main():
    start = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/recflow_request_ranking_v1"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    rank, world, device = distributed_init(17)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    payload, sidecar = load_checkpoint(args.checkpoint)
    if payload["quality_use_permitted"] is not False or payload["scale"] != "max":
        raise ValueError("This diagnostic only accepts the recorded Max profiling checkpoint")
    model = RequestRanker(args.data/"vocabulary_max", "max", payload["proposal"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, fused=True)
    optimizer.load_state_dict(payload["optimizer"])
    load_seconds = time.monotonic()-start
    parameter_elements = 0
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor.cpu(), payload["model"][name]):
            raise ValueError(f"Restored parameter differs: {name}")
        parameter_elements += tensor.numel()
    state_elements = 0
    restored = optimizer.state_dict()
    for key, state in restored["state"].items():
        for name, tensor in state.items():
            if torch.is_tensor(tensor):
                if not torch.equal(tensor.cpu(), payload["optimizer"]["state"][key][name]):
                    raise ValueError(f"Restored AdamW state differs: {key}/{name}")
                state_elements += tensor.numel()
    old_steps = {key: float(value["step"]) for key, value in restored["state"].items()}
    del restored, payload
    restore_audit_seconds = time.monotonic()-start-load_seconds
    model.train()
    dataset = RequestBatches(args.data, [24], role=0, limit=(args.steps+5)*64)
    sampler = GlobalBatchSampler(len(dataset), micro_batch=16, world_size=world, rank=rank)
    loader = DataLoader(dataset, batch_size=None, sampler=sampler, pin_memory=True, num_workers=4,
                        multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    timings, touched = [], []
    torch.cuda.synchronize()
    before = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for index, cpu in enumerate(loader):
        batch = to_device(cpu, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model(batch)
            loss, _ = ranking_loss(scores, batch["labels"], batch["candidate_mask"], listwise_weight=.1,
                                   global_active_requests=64, ddp_world_size=world, validate=False)
        loss.backward()
        touched.extend([model.features.map_ids(batch["history_features"][..., 0]).reshape(-1),
                        model.features.map_ids(batch["candidate_features"][..., 0]).reshape(-1)])
        if (index+1) % sampler.accumulation == 0:
            synchronize_and_clip(model, touched, world)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            touched = []
            torch.cuda.synchronize()
            timings.append(time.monotonic()-before)
            before = time.monotonic()
    updated = optimizer.state_dict()
    if any(float(value["step"]) != old_steps[key]+args.steps+5 for key, value in updated["state"].items()):
        raise ValueError("Optimizer step counters did not continue from the loaded state")
    training_peak = torch.cuda.max_memory_allocated()
    del updated, optimizer, loader, scores, loss
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Two full parameter copies, matching actual current/parent FP32 evaluation.
    parent = RequestRanker(args.data/"vocabulary_max", "max", model.proposal).to(device).eval()
    parent.load_state_dict(model.state_dict())
    eval_data = RequestBatches(args.data, [24], role=0, limit=1024)
    timings_eval, maximum_error = [], 0.
    with torch.inference_mode():
        for i in range(12):
            cpu = eval_data[list(range((i*world+rank)*16, (i*world+rank+1)*16))]
            batch = to_device(cpu, device)
            torch.cuda.synchronize()
            begin = time.monotonic()
            current_cache = model.encode_history(batch)
            old_cache = parent.encode_history(batch)
            current = model.read(batch, current_cache)
            native_parent = parent.read(batch, old_cache)
            reused = model.read(batch, old_cache)
            torch.cuda.synchronize()
            timings_eval.append(time.monotonic()-begin)
            maximum_error = max(maximum_error, float((current-reused).abs().max()),
                                float((current-native_parent).abs().max()))
            if not torch.isfinite(current).all() or current.dtype != torch.float32 or current_cache.k.dtype != torch.float32:
                raise ValueError("Actual Max FP32 evaluation is nonfinite or uses the wrong precision")
    if maximum_error != 0:
        raise ValueError("Same-version Max native/reuse identity failed")
    result = {"rank": rank, "load_seconds": load_seconds, "restore_audit_seconds": restore_audit_seconds,
              "all_parameter_elements_checked": parameter_elements, "all_optimizer_elements_checked": state_elements,
              "continued_optimizer_steps": len(timings), "training_p95_seconds": float(np.quantile(timings[5:], .95)),
              "training_all_step_seconds": timings, "training_peak_allocated_bytes": training_peak,
              "fp32_three_method_p95_seconds_per_16requests": float(np.quantile(timings_eval[2:], .95)),
              "fp32_evaluation_peak_bytes": torch.cuda.max_memory_allocated(), "same_model_score_max_abs": maximum_error}
    gathered = [None]*world
    if world > 1:
        dist.all_gather_object(gathered, result)
    else:
        gathered[0] = result
    if rank == 0:
        json_save(args.output/"summary.json", {"status": "passed", "quality_use_permitted": False,
                  "checkpoint_sha256": sidecar["checkpoint_sha256"], "world_size": world, "ranks": gathered,
                  "source_binding": source_binding(args.data, "vocabulary_max"), "checker_sha256": sha256(__file__),
                  "scope": "Full Max restore/optimizer continuation and evaluation timing; no recommendation-quality metric",
                  "wall_seconds": time.monotonic()-start})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
