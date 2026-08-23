#!/usr/bin/env python3
"""Four-GPU FSDP theta0 trainer for frozen 8L scale reproduction."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import scale_8l_common as scale
import scale_8l_fsdp as distributed
import train_p7_theta0 as p7
import train_scale_8l_theta0 as theta0
from hstu_kvcache.models import HSTU, HSTUConfig


def score(model, base, requests, device):
    tensors = p7.collate(requests, device)
    task = requests[0].workload
    residual = model(
        tensors["items"], tensors["behaviors"], tensors["deltas"],
        tensors["candidates"], tensors["query_deltas"], tensors["lengths"],
        tensors["query_types"], p7.CHUNK_SIZE[task],
    )
    with torch.no_grad():
        base_scores = base(tensors["features"].float()).float()
    return base_scores + residual, base_scores, residual, tensors


def train_batch(model, base, optimizer, requests, device, normalizer):
    optimizer.zero_grad(set_to_none=True)
    losses = []; base_values = []; residual_values = []
    for index, request in enumerate(requests):
        context = model.no_sync() if index + 1 < len(requests) else nullcontext()
        with context:
            deployment, base_scores, residual, tensors = score(model, base, [request], device)
            request_loss = p7.per_request_loss(request.workload, deployment, tensors)
            (request_loss * tensors["weights"] / normalizer).sum().backward()
        losses.append(float(request_loss.detach()))
        base_values.append(base_scores.detach().float()[tensors["candidate_mask"]].cpu())
        residual_values.append(residual.detach().float()[tensors["candidate_mask"]].cpu())
    grad_norm = float(model.clip_grad_norm_(float("inf")))
    optimizer.step()
    return {
        "task": requests[0].workload, "queries": len(requests),
        "candidate_rows": sum(len(row.candidate_ids) for row in requests),
        "history_tokens": sum(len(row.history_items) for row in requests),
        "token_layer_work": scale.LAYERS * sum(len(row.history_items) + len(row.candidate_ids) for row in requests),
        "loss": float(np.average(losses, weights=[row.request_weight for row in requests])),
        "gradient_norm": grad_norm,
        "base_score_std": float(torch.cat(base_values).std()),
        "residual_score_std": float(torch.cat(residual_values).std()),
    }


@torch.no_grad()
def evaluate(model, base, requests, device):
    model.eval(); task = requests[0].workload
    micro = {"N": 4, "R": 1, "F": 8}[task]
    numerator = base_numerator = denominator = 0.0
    residuals = []
    for start in range(0, len(requests), micro):
        batch = requests[start : start + micro]
        deployment, base_scores, residual, tensors = score(model, base, batch, device)
        loss = p7.per_request_loss(task, deployment, tensors)
        base_loss = p7.per_request_loss(task, base_scores, tensors)
        numerator += float((loss * tensors["weights"]).sum())
        base_numerator += float((base_loss * tensors["weights"]).sum())
        denominator += float(tensors["weights"].sum())
        residuals.append(residual[tensors["candidate_mask"]].cpu())
    model.train()
    return {
        "queries": len(requests), "users": len({row.uid for row in requests}),
        "deployment_loss": numerator / denominator,
        "base_only_loss": base_numerator / denominator,
        "residual_score_std": float(torch.cat(residuals).std()),
    }


def save_checkpoint(model, model_name, seed, epoch, output, rank):
    state = distributed.full_hstu_state_dict(model)
    if rank == 0:
        temporary = output / "theta0_selected.pt.partial"
        torch.save({
            "contract": "scale_8l_v1", "contract_hash": scale.sha256_file(scale.CONTRACT),
            "model_name": model_name, "seed": seed, "selected_epoch": epoch,
            "config": asdict(model.module.hstu.cfg), "model_state_dict": state,
            "base_bundle_hash": scale.sha256_file(scale.BASE_ROOT / "bundle_manifest.json"),
            "train_manifest_hash": scale.sha256_file(scale.P7_MANIFEST / "residual_train/manifest.index.json"),
            "development_manifest_hash": scale.sha256_file(scale.P7_MANIFEST / "development/manifest.index.json"),
            "history_limit": scale.CONTEXT, "qualification_or_theta3_scored": False,
            "distributed": {"type": "FSDP_FULL_SHARD", "world_size": dist.get_world_size(), "compute_param_dtype": "bfloat16", "master_param_dtype": "float32"},
        }, temporary)
        os.replace(temporary, output / "theta0_selected.pt")
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(scale.MODELS), required=True)
    parser.add_argument("--seed", choices=scale.SEEDS, type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--canary-steps", type=int, default=0)
    parser.add_argument("--canary-dev-queries", type=int, default=32)
    args = parser.parse_args()
    scale.contract()
    rank, world, device = distributed.initialize()
    try:
        if world != 4:
            raise RuntimeError("frozen 8L training requires four FSDP ranks")
        output = args.output or (
            scale.OUTPUT / "trainer_canary" / f"{args.model}_seed{args.seed}"
            if args.canary_steps else scale.OUTPUT / "theta0" / f"{args.model}_seed{args.seed}"
        )
        if rank == 0:
            if output.exists():
                raise FileExistsError(f"refusing to overwrite {output}")
            output.mkdir(parents=True)
        dist.barrier()
        tasks = scale.MODELS[args.model]
        bases, base_manifest = p7.load_bases(tasks, device)
        train_rows, dev_rows = scale.load_theta0_data(tasks)
        if args.canary_steps:
            dev_rows = {
                task: scale.deterministic_subset(rows, args.canary_dev_queries, f"trainer-canary-dev:{task}")
                for task, rows in dev_rows.items()
            }
        raw = scale.make_model(args.seed, device).train()
        model_meta = scale.model_metadata(raw)
        model = distributed.wrap(raw, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        best = float("inf"); best_epoch = None; trace = []; budgets = {task: {k: 0 for k in ("query_presentations", "candidate_rows", "history_tokens", "token_layer_work", "optimizer_steps")} for task in tasks}
        epochs = 1 if args.canary_steps else scale.EPOCHS
        for epoch in range(1, epochs + 1):
            batches = {task: p7.logical_batches(p7.deterministic_order(train_rows[task], args.seed * 10_000 + epoch * 100 + p7.QUERY_TYPES[task])) for task in tasks}
            normalizers = {task: sum(row.request_weight for row in train_rows[task]) / len(batches[task]) for task in tasks}
            schedule = ([(task, index) for index in range(max(map(len, batches.values()))) for task in tasks if index < len(batches[task])] if args.model == "m1" else [(tasks[0], index) for index in range(len(batches[tasks[0]]))])
            if args.canary_steps:
                schedule = schedule[: args.canary_steps]
            model.train()
            epoch_losses = {task: [] for task in tasks}
            for task, index in schedule:
                record = train_batch(model, bases[task], optimizer, batches[task][index], device, normalizers[task])
                epoch_losses[task].append(record["loss"])
                budgets[task]["query_presentations"] += record["queries"]
                for key in ("candidate_rows", "history_tokens", "token_layer_work"):
                    budgets[task][key] += record[key]
                budgets[task]["optimizer_steps"] += 1
            dev = {task: evaluate(model, bases[task], dev_rows[task], device) for task in tasks}
            objective = float(np.mean([dev[task]["deployment_loss"] for task in tasks]))
            selected = objective < best
            if selected:
                best = objective; best_epoch = epoch
            trace.append({"epoch": epoch, "objective": objective, "tasks": dev, "train_loss_mean": {task: float(np.mean(epoch_losses[task])) for task in tasks}, "selected_so_far": selected, "selection_uses_H_S_or_theta3": False})
            if selected:
                save_checkpoint(model, args.model, args.seed, epoch, output, rank)
        assert best_epoch is not None
        if rank == 0:
            checkpoint = output / "theta0_selected.pt"
            checkpoint_hash = scale.sha256_file(checkpoint)
            checkpoint_bytes = checkpoint.stat().st_size
            checkpoint_roundtrip = None
            if args.canary_steps:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                with torch.device("meta"):
                    reference = HSTU(HSTUConfig(**payload["config"]))
                checkpoint_roundtrip = {
                    "config_matches": payload["config"] == asdict(reference.cfg),
                    "state_keyset_matches": set(payload["model_state_dict"]) == set(reference.state_dict()),
                    "tensor_shapes_match": all(
                        tuple(payload["model_state_dict"][name].shape) == tuple(value.shape)
                        for name, value in reference.state_dict().items()
                    ),
                }
                if not all(checkpoint_roundtrip.values()):
                    raise RuntimeError(f"trainer canary checkpoint roundtrip failed: {checkpoint_roundtrip}")
                checkpoint.unlink()
            result = {
                "status": "scale_theta0_FSDP_trainer_canary_passed" if args.canary_steps else "scale_theta0_trained_FSDP_selected_on_development_only",
                "contract_hash": scale.sha256_file(scale.CONTRACT), "model_name": args.model,
                "seed": args.seed, "selected_epoch": best_epoch, "selected_objective": best,
                "checkpoint": None if args.canary_steps else str(checkpoint.relative_to(scale.ROOT)),
                "checkpoint_hash": checkpoint_hash, "checkpoint_bytes": checkpoint_bytes,
                "checkpoint_roundtrip": checkpoint_roundtrip,
                "checkpoint_retained": not bool(args.canary_steps),
                "model": model_meta, "history_length": 1024, "epochs": epochs,
                "distributed": {"type": "FSDP_FULL_SHARD", "world_size": world, "GPUs": [0, 1, 2, 3]},
                "budget": {task: {"unique_queries": len(train_rows[task]), **budgets[task]} for task in tasks},
                "selection_trace": trace, "base_parameter_hashes": base_manifest["files"],
                "qualification_or_theta3_scored": False,
            }
            scale.json_dump(output / "train_result.json", result)
            print(json.dumps({key: result[key] for key in ("status", "model_name", "seed", "selected_epoch", "checkpoint_hash", "checkpoint_retained")}, indent=2))
        dist.barrier()
    finally:
        distributed.finish()


if __name__ == "__main__":
    main()
