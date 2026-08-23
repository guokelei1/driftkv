#!/usr/bin/env python3
"""Four-GPU FSDP R0/R1/R2 trainer for the frozen 8L scale chain."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

import scale_8l_common as scale
import scale_8l_fsdp as distributed
import train_p7_theta0 as p7
import train_scale_8l_fsdp_theta0 as theta
from hstu_kvcache.models import HSTU, HSTUConfig

SPLITS = {
    "r0": ("update1_train", "update1_admission_dev"),
    "r1_edge1": ("update1_train", "update1_admission_dev"),
    "r1_edge2": ("update2_train", "update2_admission_dev"),
    "r2": ("update1_train", "update1_admission_dev"),
}


def parent_path(model: str, seed: int, release: str) -> Path:
    if release == "r1_edge2":
        return scale.OUTPUT / "releases/r1_edge1" / f"{model}_seed{seed}" / "selected.pt"
    return scale.OUTPUT / "theta0" / f"{model}_seed{seed}" / "theta0_selected.pt"


def load_raw(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = HSTU(HSTUConfig(**payload["config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device), payload


def request_rows(root, split, tasks):
    return {task: scale.load_requests(root, split, task) for task in tasks}


def data(model_name, release):
    tasks = scale.MODELS[model_name]
    train_split, dev_split = SPLITS[release]
    update = request_rows(scale.P8_MANIFEST, train_split, tasks)
    if release == "r2":
        original = request_rows(scale.P7_MANIFEST, "residual_train", tasks)
        update = {task: original[task] + update[task] for task in tasks}
    return update, request_rows(scale.P8_MANIFEST, dev_split, tasks)


def no_information(task, rows):
    weights = np.asarray([row.request_weight for row in rows], dtype=np.float64)
    if task in {"N", "R"}:
        values = np.log(np.asarray([len(row.candidate_ids) for row in rows], dtype=np.float64))
    else:
        labels = np.asarray([row.label for row in rows], dtype=np.float64)
        prevalence = min(max(float(np.average(labels, weights=weights)), 1e-8), 1 - 1e-8)
        values = -(labels * math.log(prevalence) + (1 - labels) * math.log(1 - prevalence))
    return float(np.average(values, weights=weights))


def save(model, args, epoch, parent, output, rank, admitted=None):
    state = distributed.full_hstu_state_dict(model)
    if rank == 0:
        temporary = output / "selected.pt.partial"
        torch.save({
            "contract": "scale_8l_v1", "contract_hash": scale.sha256_file(scale.CONTRACT),
            "release": args.release, "model_name": args.model, "seed": args.seed,
            "selected_epoch": epoch, "config": asdict(model.module.hstu.cfg),
            "model_state_dict": state, "parent_checkpoint": str(parent.relative_to(scale.ROOT)),
            "parent_checkpoint_hash": scale.sha256_file(parent), "admitted": admitted,
            "history_limit": scale.CONTEXT,
            "base_bundle_hash": scale.sha256_file(scale.BASE_ROOT / "bundle_manifest.json"),
            "qualification_or_theta3_scored": False,
            "distributed": {"type": "FSDP_FULL_SHARD", "world_size": dist.get_world_size()},
        }, temporary)
        os.replace(temporary, output / "selected.pt")
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(scale.MODELS), required=True)
    parser.add_argument("--seed", choices=scale.SEEDS, type=int, required=True)
    parser.add_argument("--release", choices=sorted(SPLITS), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scale.contract(); rank, world, device = distributed.initialize()
    try:
        if world != 4:
            raise RuntimeError("frozen 8L release training requires four FSDP ranks")
        output = args.output or scale.OUTPUT / "releases" / args.release / f"{args.model}_seed{args.seed}"
        if rank == 0:
            if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
            output.mkdir(parents=True)
        dist.barrier()
        tasks = scale.MODELS[args.model]; parent = parent_path(args.model, args.seed, args.release)
        if not parent.exists(): raise FileNotFoundError(parent)
        parent_raw, parent_payload = load_raw(parent, device)
        bases, base_manifest = p7.load_bases(tasks, device)
        train_rows, dev_rows = data(args.model, args.release)
        parent_model = distributed.wrap(parent_raw, device)
        previous_dev = {task: theta.evaluate(parent_model, bases[task], dev_rows[task], device) for task in tasks}
        if args.release == "r2":
            del parent_model, parent_raw
            torch.cuda.empty_cache(); dist.barrier()
            raw = scale.make_model(args.seed, device)
        else:
            raw = parent_model.module.hstu
            # Unwrap is not supported safely; keep the wrapped parent as current.
        if args.release == "r2":
            model = distributed.wrap(raw, device)
        else:
            model = parent_model
        if args.release == "r0":
            allowed = ("query_encoder.type_embedding", "query_encoder.action_embedding", "cc_score_head")
            for name, parameter in model.module.hstu.named_parameters():
                parameter.requires_grad_(name.startswith(allowed))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4)
        best = float("inf"); best_epoch = None; trace = []; budgets: Counter = Counter()
        for epoch in range(1, scale.EPOCHS + 1):
            batches = {task: p7.logical_batches(p7.deterministic_order(train_rows[task], args.seed * 10_000 + epoch * 100 + p7.QUERY_TYPES[task])) for task in tasks}
            normalizers = {task: sum(row.request_weight for row in train_rows[task]) / len(batches[task]) for task in tasks}
            schedule = ([(task, index) for index in range(max(map(len, batches.values()))) for task in tasks if index < len(batches[task])] if args.model == "m1" else [(tasks[0], index) for index in range(len(batches[tasks[0]]))])
            epoch_losses = {task: [] for task in tasks}; model.train()
            for task, index in schedule:
                record = theta.train_batch(model, bases[task], optimizer, batches[task][index], device, normalizers[task])
                epoch_losses[task].append(record["loss"])
                for key in ("queries", "candidate_rows", "history_tokens", "token_layer_work"):
                    budgets[(task, key)] += record[key]
                budgets[(task, "optimizer_steps")] += 1
            dev = {task: theta.evaluate(model, bases[task], dev_rows[task], device) for task in tasks}
            objective = float(np.mean([dev[task]["deployment_loss"] for task in tasks]))
            selected = objective < best
            if selected: best = objective; best_epoch = epoch
            trace.append({"epoch": epoch, "objective": objective, "tasks": dev, "train_loss_mean": {task: float(np.mean(epoch_losses[task])) for task in tasks}, "selected_so_far": selected, "selection_uses_H_S_or_theta3": False})
            if selected: save(model, args, epoch, parent, output, rank)
        assert best_epoch is not None
        final_dev = trace[best_epoch - 1]["tasks"]
        controls = {task: no_information(task, dev_rows[task]) for task in tasks}
        admitted = (
            final_dev["F"]["deployment_loss"] < final_dev["F"]["base_only_loss"]
            if args.model == "m0_f" else
            all(final_dev[task]["deployment_loss"] < controls[task] for task in tasks)
            and any(final_dev[task]["deployment_loss"] < final_dev[task]["base_only_loss"] for task in ("R", "F"))
        )
        # Rewrite only small metadata in a completed checkpoint payload after admission.
        if rank == 0:
            checkpoint = output / "selected.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["admitted"] = admitted
            temporary = output / "selected.pt.partial"
            torch.save(payload, temporary); os.replace(temporary, checkpoint)
            frozen_delta = None
            if args.release == "r0":
                frozen_delta = max(
                    float((payload["model_state_dict"][name] - parent_payload["model_state_dict"][name]).abs().max())
                    for name in payload["model_state_dict"]
                    if not name.startswith(("query_encoder.type_embedding", "query_encoder.action_embedding", "cc_score_head"))
                )
                if frozen_delta != 0.0: raise RuntimeError("R0 changed cache-producing state")
            result = {
                "status": "scale_release_trained_FSDP_frozen_before_HS_evaluation",
                "contract_hash": scale.sha256_file(scale.CONTRACT), "release": args.release,
                "model_name": args.model, "seed": args.seed,
                "parent": str(parent.relative_to(scale.ROOT)), "parent_hash": scale.sha256_file(parent),
                "checkpoint": str(checkpoint.relative_to(scale.ROOT)), "checkpoint_hash": scale.sha256_file(checkpoint),
                "selected_epoch": best_epoch, "selection_trace": trace,
                "final_admission_dev": final_dev, "previous_full_admission_dev": previous_dev,
                "no_information_controls": controls, "admitted": admitted,
                "admission_uses_H_S_or_theta3": False, "r0_frozen_parameter_max_abs_delta": frozen_delta,
                "budget": {task: {key: budgets[(task, key)] for key in ("queries", "candidate_rows", "history_tokens", "token_layer_work", "optimizer_steps")} for task in tasks},
                "base_parameter_hashes": base_manifest["files"], "qualification_or_theta3_scored": False,
                "distributed": {"type": "FSDP_FULL_SHARD", "world_size": world, "GPUs": [0, 1, 2, 3]},
            }
            scale.json_dump(output / "train_result.json", result)
            print(json.dumps({key: result[key] for key in ("status", "release", "model_name", "seed", "checkpoint_hash", "admitted")}, indent=2))
        dist.barrier()
    finally:
        distributed.finish()


if __name__ == "__main__":
    main()
