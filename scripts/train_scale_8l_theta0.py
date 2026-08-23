#!/usr/bin/env python3
"""Shared single-process helpers for the 8L trainer.

Direct single-GPU training is prohibited by the measured AdamW memory gate;
use ``train_scale_8l_fsdp_theta0.py`` through the frozen queue.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import scale_8l_common as scale
import train_p7_theta0 as p7
from hstu_kvcache.data.p7_training import QUERY_TYPES

# Memory-only execution choices; they do not change request-level objectives.
p7.MICROBATCH = {"N": 1, "R": 1, "F": 1}
p7.CHUNK_SIZE = {"N": 8, "R": 4, "F": 1}


def adjusted_record(record: dict) -> dict:
    record = dict(record)
    # Frozen P7 logger counted the original four-layer model.
    record["token_layer_work"] = int(record["history_tokens"] + record["candidate_rows"]) * scale.LAYERS
    return record


def train(model_name: str, seed: int, device: torch.device, output: Path) -> None:
    scale.contract()
    tasks = scale.MODELS[model_name]
    bases, base_manifest = p7.load_bases(tasks, device)
    train_rows, dev_rows = scale.load_theta0_data(tasks)
    model = scale.make_model(seed, device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best_objective = float("inf")
    best_epoch = None
    best_state = None
    trace: list[dict] = []
    budgets = {task: {key: 0 for key in ("query_presentations", "candidate_rows", "history_tokens", "token_layer_work", "optimizer_steps")} for task in tasks}

    for epoch in range(1, scale.EPOCHS + 1):
        batches = {
            task: p7.logical_batches(
                p7.deterministic_order(train_rows[task], seed * 10_000 + epoch * 100 + QUERY_TYPES[task])
            )
            for task in tasks
        }
        normalizers = {
            task: sum(row.request_weight for row in train_rows[task]) / len(batches[task])
            for task in tasks
        }
        if model_name == "m1":
            schedule = [
                (task, index)
                for index in range(max(map(len, batches.values())))
                for task in tasks if index < len(batches[task])
            ]
        else:
            schedule = [(tasks[0], index) for index in range(len(batches[tasks[0]]))]
        model.train()
        for task, index in schedule:
            record = adjusted_record(p7.train_logical_batch(
                model, bases[task], optimizer, batches[task][index], device,
                loss_normalizer=normalizers[task],
            ))
            budgets[task]["query_presentations"] += record["queries"]
            for key in ("candidate_rows", "history_tokens", "token_layer_work"):
                budgets[task][key] += record[key]
            budgets[task]["optimizer_steps"] += 1
        development = {task: p7.evaluate_task(model, bases[task], dev_rows[task], device) for task in tasks}
        objective = float(np.mean([development[task]["deployment_loss"] for task in tasks]))
        selected = objective < best_objective
        if selected:
            best_objective = objective
            best_epoch = epoch
            best_state = p7.clone_state_dict(model)
        trace.append({
            "epoch": epoch, "aggregate_dev_objective": objective,
            "tasks": development, "selected_so_far": selected,
            "selection_uses_H_S_or_theta3": False,
        })

    assert best_state is not None and best_epoch is not None
    model.load_state_dict(best_state)
    final_dev = {task: p7.evaluate_task(model, bases[task], dev_rows[task], device) for task in tasks}
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "theta0_selected.pt"
    torch.save({
        "contract": "scale_8l_v1", "contract_hash": scale.sha256_file(scale.CONTRACT),
        "model_name": model_name, "seed": seed, "selected_epoch": best_epoch,
        "config": asdict(model.cfg), "model_state_dict": best_state,
        "base_bundle_hash": scale.sha256_file(scale.BASE_ROOT / "bundle_manifest.json"),
        "train_manifest_hash": scale.sha256_file(scale.P7_MANIFEST / "residual_train/manifest.index.json"),
        "development_manifest_hash": scale.sha256_file(scale.P7_MANIFEST / "development/manifest.index.json"),
        "history_limit": scale.CONTEXT, "qualification_or_theta3_scored": False,
    }, checkpoint)
    result = {
        "status": "scale_theta0_trained_and_selected_on_development_only",
        "contract_hash": scale.sha256_file(scale.CONTRACT),
        "model_name": model_name, "seed": seed,
        "selected_epoch": best_epoch, "selected_objective": best_objective,
        "checkpoint": str(checkpoint.relative_to(scale.ROOT)),
        "checkpoint_hash": scale.sha256_file(checkpoint),
        "model": scale.model_metadata(model),
        "history_length": 1024, "epochs": scale.EPOCHS,
        "optimizer": {"type": "AdamW", "learning_rate": 2e-4, "weight_decay": 1e-4},
        "memory_execution": {"microbatch": dict(p7.MICROBATCH), "candidate_chunk": dict(p7.CHUNK_SIZE)},
        "budget": {task: {"unique_queries": len(train_rows[task]), **budgets[task]} for task in tasks},
        "selection_trace": trace, "final_development": final_dev,
        "base_manifest_hash": scale.sha256_file(scale.BASE_ROOT / "bundle_manifest.json"),
        "base_parameter_hashes": base_manifest["files"],
        "qualification_or_theta3_scored": False,
    }
    scale.json_dump(output / "train_result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "model_name", "seed", "selected_epoch", "checkpoint_hash")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(scale.MODELS), required=True)
    parser.add_argument("--seed", choices=scale.SEEDS, type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raise RuntimeError(
        "single-GPU 8L AdamW is invalidated by the memory preflight; "
        "use torchrun --nproc_per_node=4 scripts/train_scale_8l_fsdp_theta0.py"
    )


if __name__ == "__main__":
    main()
