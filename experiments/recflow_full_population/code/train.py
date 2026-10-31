"""Replicated multi-GPU full-candidate training with exact global request means."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .batches import GlobalBatchSampler, RequestBatches
from .communication import synchronize_and_clip
from .configuration import DEFAULT_CONFIG, load_proposal
from .data_common import json_save, sha256
from .model import RequestRanker
from .objective import ranking_loss
from .runtime import (
    distributed_init,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
    source_binding,
    to_device,
)


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("experiments/recflow_full_population/data"))
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--days", nargs="+", type=int, required=True)
    p.add_argument("--phase", choices=["warmup", "alignment", "update", "diagnostic"], required=True)
    p.add_argument("--scale", choices=["pilot", "max"], default="pilot")
    p.add_argument("--parent", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--global-batch", type=int, default=64)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--listwise-weight", type=float, default=.1)
    p.add_argument("--wall-hours", type=float, default=3)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--communication", choices=["rows", "dense"], default="rows")
    p.add_argument("--request-limit", type=int)
    p.add_argument("--stop-after-steps", type=int, help="Diagnostic clean interruption; planned LR schedule is unchanged")
    p.add_argument("--diagnostic-state", action="store_true", help="Forbid quality use even when testing an actual update phase")
    p.add_argument("--budget-spent-seconds", type=float, default=0.,
                   help="Already spent cumulative budget, including external validation; can only increase saved elapsed time")
    args = p.parse_args()
    rank, world, device = distributed_init(args.seed)
    config = load_proposal(args.config)
    if args.epochs < 1 or args.log_every < 1 or args.checkpoint_every < 1 or args.budget_spent_seconds < 0:
        raise ValueError("Invalid training length, logging cadence, or elapsed budget")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if args.wall_hours > (2 if args.phase == "update" else 15):
        raise ValueError("Requested budget exceeds protocol hard limit")
    if args.phase == "warmup" and not all(1 <= d <= 23 for d in args.days):
        raise ValueError("Warmup uses D1..23 exposures")
    if args.phase == "alignment" and not all(24 <= d <= 27 for d in args.days):
        raise ValueError("Foundation alignment uses D24..27 only")
    if args.phase == "update" and (len(args.days) != 1 or not 28 <= args.days[0] <= 36 or not (args.parent or args.resume)):
        raise ValueError("An update needs one D28..36 day and its actual parent")
    if args.request_limit is not None or args.pilot:
        raise ValueError("Formal full-population training must not cap requests or select old pilot UIDs")
    if args.resume and args.parent:
        raise ValueError("Use resume OR parent")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "plan.json").exists() and not args.resume:
        raise ValueError("Output already initialized; resume explicitly or use a new run")
    dataset = RequestBatches(args.data, args.days, role=0, pilot=args.pilot, limit=args.request_limit)
    if not len(dataset):
        raise ValueError("No eligible training requests")
    steps_per_epoch = math.ceil(len(dataset)/args.global_batch)
    planned_steps = min(steps_per_epoch*args.epochs, args.max_steps or 2**60)
    vocabulary_name = "vocabulary_max" if args.scale == "max" else "vocabulary"
    binding = source_binding(args.data, vocabulary_name)
    vocab_hash = sha256(args.data / vocabulary_name / "manifest.json")
    model = RequestRanker(args.data / vocabulary_name, args.scale, config).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    lr = config["optimization_proposal"]["update_learning_rate" if args.phase == "update" else "foundation_learning_rate"]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=.01, fused=True)
    warmup = max(1, int(planned_steps*.05))

    def lr_factor(step):
        return min((step+1)/warmup, 1.) * max(.05, .5*(1+math.cos(math.pi*max(0, step-warmup)/max(1, planned_steps-warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    epoch_start, step_start, total_step, prior_elapsed = 0, 0, 0, args.budget_spent_seconds
    parent_digest = None
    if args.parent or args.resume:
        payload, sidecar = load_checkpoint(args.resume or args.parent, "cpu")
        if payload.get("quality_use_permitted") is False and not (args.phase == "diagnostic" or args.diagnostic_state):
            raise ValueError("Diagnostic weights cannot become a quality model or release parent")
        if payload["scale"] != args.scale:
            raise ValueError("Parent architecture differs")
        if payload["proposal"] != config:
            raise ValueError("Parent/resume proposal differs from the current frozen configuration")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        parent_digest = sidecar["checkpoint_sha256"]
        if args.resume:
            # A recovery point is inside the same version, not a new release
            # parent. Preserve the actual cross-version lineage for KV reuse.
            parent_digest = payload["parent_checkpoint_sha256"]
            original = json.loads((args.output / "plan.json").read_text())
            current = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
            ignored = {"resume", "parent", "log_every", "checkpoint_every", "stop_after_steps", "budget_spent_seconds"}
            if (original["world_size"] != world or
                any(current[key] != value for key, value in original["arguments"].items() if key not in ignored)):
                raise ValueError("Resume training arguments or world size differ")
            selection_path = args.output / "request_selection.npz"
            with np.load(selection_path) as selected:
                if any(not np.array_equal(selected[key], dataset.records[key]) for key in selected.files):
                    raise ValueError("Reconstructed request order differs from the saved selection")
            epoch_start, step_start, total_step = payload["next_epoch"], payload["next_step"], payload["total_step"]
            scheduler.load_state_dict(payload["scheduler"])
            prior_elapsed = max(prior_elapsed, sidecar["elapsed_budget_seconds"])
            restore_rng(payload, rank, world)
        else:
            if args.phase == "update" and max(payload["days"]) != args.days[0]-1:
                raise ValueError("Update must continue the immediately previous daily cutoff")
            if args.phase == "alignment":
                prior_elapsed = max(prior_elapsed, sidecar["elapsed_budget_seconds"])
                completed = args.parent.parent / "completion.json"
                if completed.exists():
                    prior_elapsed = max(prior_elapsed, json.loads(completed.read_text())["elapsed_budget_seconds"])
            # Restoring AdamW also restores its last LR. Explicitly establish a
            # new daily schedule, keeping moments and parameter step counters.
            for group in optimizer.param_groups:
                group["lr"] = lr
                group["initial_lr"] = lr
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        del payload
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.resume:
        arguments = json.loads((args.output / "plan.json").read_text())["arguments"]
    plan = {"arguments": arguments, "source_binding": binding, "trainer_sha256": sha256(__file__),
            "requests": len(dataset), "planned_steps": planned_steps, "world_size": world,
            "trainable_parameters": sum(p.numel() for p in parameters), "vocabulary_sha256": vocab_hash,
            "parent_checkpoint_sha256": parent_digest, "history_limit": 1024,
            "global_batch": args.global_batch, "loss_normalization": "global_active_request_mean",
            "selection": "endpoint only; native pre-release validation is run separately",
            "budget_seconds": args.wall_hours*3600, "budget_already_spent_seconds": prior_elapsed}
    if rank == 0 and not args.resume:
        np.savez(args.output / "request_selection.npz", uid=dataset.records["uid"], rid=dataset.records["rid"],
                 day=dataset.records["day"], ts=dataset.records["ts"], history_length=dataset.records["history_length"])
        plan["request_selection_sha256"] = sha256(args.output / "request_selection.npz")
        json_save(args.output / "plan.json", plan)
    if prior_elapsed+time.monotonic()-started >= args.wall_hours*3600-120:
        raise ValueError("No training budget remains after initialization; previous checkpoint retained")
    if world > 1:
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0)
    model.train()
    metrics = torch.zeros(3, dtype=torch.float64, device=device)
    measured_steps, measured_start = 0, None
    stop_reason = "completed"
    next_epoch, next_step = epoch_start, step_start
    for epoch in range(epoch_start, args.epochs):
        remain = planned_steps-total_step
        if remain <= 0:
            break
        start_step = step_start if epoch == epoch_start else 0
        sampler = GlobalBatchSampler(len(dataset), global_batch=args.global_batch, micro_batch=args.micro_batch,
                                     world_size=world, rank=rank, seed=args.seed, epoch=epoch,
                                     start_step=start_step, stop_step=min(steps_per_epoch, start_step+remain))
        options = {"multiprocessing_context": "spawn", "persistent_workers": True, "prefetch_factor": 2} if args.workers else {}
        loader = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=args.workers,
                            pin_memory=True, generator=torch.Generator().manual_seed(args.seed+epoch*world+rank), **options)
        optimizer.zero_grad(set_to_none=True)
        touched = []
        for micro_index, cpu_batch in enumerate(loader):
            local_micro = micro_index % sampler.accumulation
            step = start_step + micro_index//sampler.accumulation
            active_global = min(args.global_batch, len(dataset)-step*args.global_batch)
            batch = to_device(cpu_batch, device)
            final_micro = local_micro == sampler.accumulation-1
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.dtype == "bfloat16"):
                logits = model(batch)
                loss, stats = ranking_loss(logits, batch["labels"], batch["candidate_mask"],
                                           listwise_weight=0 if args.phase == "warmup" else args.listwise_weight,
                                           global_active_requests=active_global, ddp_world_size=world, validate=False)
            loss.backward()
            touched.extend([model.features.map_ids(batch["history_features"][..., 0]).reshape(-1),
                            model.features.map_ids(batch["candidate_features"][..., 0]).reshape(-1)])
            metrics += torch.stack([stats["bce_sum"].detach(), stats["listwise_sum"].detach(),
                                    stats["active_requests"].detach()]).double()
            if not final_micro:
                continue
            if total_step == 0:
                missing = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is None]
                if missing:
                    raise ValueError(f"Trainable parameters have no gradient: {missing}")
            norm, communicated_rows = synchronize_and_clip(model, touched, world, dense=args.communication == "dense")
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite globally averaged gradient; optimizer was not advanced")
            touched = []
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total_step += 1
            next_epoch, next_step = epoch, step+1
            if next_step == steps_per_epoch:
                next_epoch, next_step = epoch+1, 0
            if measured_start is None and total_step >= 10:
                torch.cuda.synchronize()
                measured_start = time.monotonic()
            elif measured_start is not None:
                measured_steps += 1
            # This tiny collective keeps every rank on the same stop branch,
            # including a rank stalled on input. It avoids uneven DDP exits.
            flag = torch.tensor(int(prior_elapsed+time.monotonic()-started >= args.wall_hours*3600-120), device=device)
            if world > 1:
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            budget_stop = bool(flag.item())
            requested_stop = args.stop_after_steps is not None and total_step >= args.stop_after_steps
            check = total_step % args.log_every == 0 or total_step == planned_steps or budget_stop or requested_stop
            if check:
                reduced = metrics.clone()
                if world > 1:
                    dist.all_reduce(reduced)
                torch.cuda.synchronize()
                elapsed = prior_elapsed+time.monotonic()-started
                if rank == 0:
                    vals = reduced.cpu().tolist()
                    progress = {"step": total_step, "epoch": epoch, "bce": vals[0]/max(1, vals[2]),
                                "listwise": vals[1]/max(1, vals[2]), "requests": int(vals[2]),
                                "gradient_norm": float(norm), "lr": scheduler.get_last_lr()[0],
                                "communicated_item_rows": communicated_rows,
                                "elapsed_budget_seconds": elapsed, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "measured_steps": measured_steps,
                                "seconds_per_step": (time.monotonic()-measured_start)/measured_steps if measured_steps else None}
                    print(json.dumps(progress), flush=True)
                    json_save(args.output / "progress.json", progress)
            if budget_stop or requested_stop or total_step % args.checkpoint_every == 0 or total_step == planned_steps:
                metadata = {"scale": args.scale, "vocabulary_sha256": vocab_hash, "source_binding": binding,
                            "arguments": arguments, "parent_checkpoint_sha256": parent_digest,
                            "total_step": total_step, "next_epoch": next_epoch, "next_step": next_step,
                            "phase": args.phase, "days": args.days, "proposal": config,
                            "quality_use_permitted": args.phase != "diagnostic" and not args.diagnostic_state}
                save_checkpoint(args.output / f"step_{total_step:06d}.pt", model, optimizer, scheduler, metadata,
                                started=started, prior_elapsed=prior_elapsed, rank=rank, world=world)
                if rank == 0:
                    json_save(args.output / "latest.json", {"checkpoint": f"step_{total_step:06d}.pt"})
            if budget_stop:
                stop_reason = "wall_budget_reached_checkpoint_saved"
                break
            if requested_stop:
                stop_reason = "clean_interruption_for_resume_check"
                break
        del loader
        if stop_reason != "completed":
            break
    if rank == 0:
        json_save(args.output / "completion.json", {"status": stop_reason, "steps": total_step,
                  "next_epoch": next_epoch, "next_step": next_step,
                  "elapsed_budget_seconds": prior_elapsed+time.monotonic()-started,
                  "world_size": world, "optimizer_state_preserved": bool(args.parent or args.resume)})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
