#!/usr/bin/env python3
"""Retained single-process release helpers; direct execution is prohibited.

Use ``train_scale_8l_fsdp_release.py`` because the measured 8L AdamW state does
not fit one A40.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import scale_8l_common as scale
import train_p7_theta0 as p7
import train_scale_8l_theta0 as theta0
from hstu_kvcache.models import HSTU, HSTUConfig

SPLITS = {
    "r0": ("update1_train", "update1_admission_dev"),
    "r1_edge1": ("update1_train", "update1_admission_dev"),
    "r1_edge2": ("update2_train", "update2_admission_dev"),
    "r2": ("update1_train", "update1_admission_dev"),
}


def load_checkpoint(path: Path, device: torch.device) -> tuple[HSTU, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = HSTU(HSTUConfig(**payload["config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device), payload


def parent_path(model_name: str, seed: int, release: str) -> Path:
    if release == "r1_edge2":
        return scale.OUTPUT / "releases/r1_edge1" / f"{model_name}_seed{seed}" / "selected.pt"
    return scale.OUTPUT / "theta0" / f"{model_name}_seed{seed}" / "theta0_selected.pt"


def rows(root: Path, split: str, tasks: tuple[str, ...]) -> dict:
    return {task: scale.load_requests(root, split, task) for task in tasks}


def training_data(model_name: str, release: str):
    tasks = scale.MODELS[model_name]
    train_split, dev_split = SPLITS[release]
    update = rows(scale.P8_MANIFEST, train_split, tasks)
    if release == "r2":
        original = rows(scale.P7_MANIFEST, "residual_train", tasks)
        update = {task: original[task] + update[task] for task in tasks}
    return update, rows(scale.P8_MANIFEST, dev_split, tasks)


def no_information_loss(task, requests) -> float:
    weights = np.asarray([row.request_weight for row in requests], dtype=np.float64)
    if task in {"N", "R"}:
        values = np.log(np.asarray([len(row.candidate_ids) for row in requests], dtype=np.float64))
    else:
        labels = np.asarray([row.label for row in requests], dtype=np.float64)
        prevalence = min(max(float(np.average(labels, weights=weights)), 1e-8), 1 - 1e-8)
        values = -(labels * math.log(prevalence) + (1 - labels) * math.log(1 - prevalence))
    return float(np.average(values, weights=weights))


def train(model_name: str, seed: int, release: str, device: torch.device, output: Path) -> None:
    scale.contract()
    tasks = scale.MODELS[model_name]
    parent = parent_path(model_name, seed, release)
    if not parent.exists():
        raise FileNotFoundError(f"parent checkpoint missing: {parent}")
    parent_model, parent_payload = load_checkpoint(parent, device)
    model = scale.make_model(seed, device) if release == "r2" else parent_model
    if release == "r0":
        allowed = ("query_encoder.type_embedding", "query_encoder.action_embedding", "cc_score_head")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(allowed))
    bases, base_manifest = p7.load_bases(tasks, device)
    train_rows, dev_rows = training_data(model_name, release)
    previous = {task: p7.evaluate_task(parent_model, bases[task], dev_rows[task], device) for task in tasks}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4)
    frozen_before = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if release == "r0" and not name.startswith(("query_encoder.type_embedding", "query_encoder.action_embedding", "cc_score_head"))
    }
    best_objective = float("inf"); best_epoch = None; best_state = None
    trace = []; budgets: Counter = Counter()
    for epoch in range(1, scale.EPOCHS + 1):
        batches = {
            task: p7.logical_batches(p7.deterministic_order(train_rows[task], seed * 10_000 + epoch * 100 + p7.QUERY_TYPES[task]))
            for task in tasks
        }
        normalizers = {task: sum(row.request_weight for row in train_rows[task]) / len(batches[task]) for task in tasks}
        schedule = (
            [(task, index) for index in range(max(map(len, batches.values()))) for task in tasks if index < len(batches[task])]
            if model_name == "m1" else [(tasks[0], index) for index in range(len(batches[tasks[0]]))]
        )
        model.train()
        for task, index in schedule:
            record = theta0.adjusted_record(p7.train_logical_batch(
                model, bases[task], optimizer, batches[task][index], device,
                loss_normalizer=normalizers[task],
            ))
            for key in ("queries", "candidate_rows", "history_tokens", "token_layer_work"):
                budgets[(task, key)] += record[key]
            budgets[(task, "optimizer_steps")] += 1
        dev = {task: p7.evaluate_task(model, bases[task], dev_rows[task], device) for task in tasks}
        objective = float(np.mean([dev[task]["deployment_loss"] for task in tasks]))
        selected = objective < best_objective
        if selected:
            best_objective = objective; best_epoch = epoch; best_state = p7.clone_state_dict(model)
        trace.append({"epoch": epoch, "objective": objective, "tasks": dev, "selected_so_far": selected, "selection_uses_H_S_or_theta3": False})
    assert best_state is not None and best_epoch is not None
    model.load_state_dict(best_state)
    final_dev = {task: p7.evaluate_task(model, bases[task], dev_rows[task], device) for task in tasks}
    controls = {task: no_information_loss(task, dev_rows[task]) for task in tasks}
    if model_name == "m0_f":
        admitted = final_dev["F"]["deployment_loss"] < final_dev["F"]["base_only_loss"]
    else:
        admitted = all(final_dev[task]["deployment_loss"] < controls[task] for task in tasks) and any(
            final_dev[task]["deployment_loss"] < final_dev[task]["base_only_loss"] for task in ("R", "F")
        )
    frozen_delta = None
    if release == "r0":
        frozen_delta = max(float((model.state_dict()[name].detach().cpu() - value).abs().max()) for name, value in frozen_before.items())
        if frozen_delta != 0.0:
            raise RuntimeError("R0 changed a cache-producing parameter")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "selected.pt"
    torch.save({
        "contract": "scale_8l_v1", "contract_hash": scale.sha256_file(scale.CONTRACT),
        "release": release, "model_name": model_name, "seed": seed,
        "selected_epoch": best_epoch, "config": asdict(model.cfg), "model_state_dict": best_state,
        "parent_checkpoint": str(parent.relative_to(scale.ROOT)), "parent_checkpoint_hash": scale.sha256_file(parent),
        "admitted": admitted, "history_limit": scale.CONTEXT,
        "base_bundle_hash": scale.sha256_file(scale.BASE_ROOT / "bundle_manifest.json"),
        "qualification_or_theta3_scored": False,
    }, checkpoint)
    result = {
        "status": "scale_release_trained_and_frozen_before_HS_evaluation",
        "contract_hash": scale.sha256_file(scale.CONTRACT), "release": release,
        "model_name": model_name, "seed": seed, "parent": str(parent.relative_to(scale.ROOT)),
        "parent_hash": scale.sha256_file(parent), "checkpoint": str(checkpoint.relative_to(scale.ROOT)),
        "checkpoint_hash": scale.sha256_file(checkpoint), "selected_epoch": best_epoch,
        "selection_trace": trace, "final_admission_dev": final_dev,
        "previous_full_admission_dev": previous, "no_information_controls": controls,
        "admitted": admitted, "admission_uses_H_S_or_theta3": False,
        "r0_frozen_parameter_max_abs_delta": frozen_delta,
        "budget": {task: {key: budgets[(task, key)] for key in ("queries", "candidate_rows", "history_tokens", "token_layer_work", "optimizer_steps")} for task in tasks},
        "base_parameter_hashes": base_manifest["files"], "qualification_or_theta3_scored": False,
    }
    scale.json_dump(output / "train_result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "release", "model_name", "seed", "checkpoint_hash", "admitted")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(scale.MODELS), required=True)
    parser.add_argument("--seed", choices=scale.SEEDS, type=int, required=True)
    parser.add_argument("--release", choices=sorted(SPLITS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raise RuntimeError(
        "single-GPU 8L AdamW is invalidated by the memory preflight; "
        "use torchrun --nproc_per_node=4 scripts/train_scale_8l_fsdp_release.py"
    )


if __name__ == "__main__":
    main()
