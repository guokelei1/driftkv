#!/usr/bin/env python3
"""Train one frozen-contract P8 R0/R1/R2 release checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import train_p7_theta0 as p7

from hstu_kvcache.data.p7_training import P7Request, load_p7_requests
from hstu_kvcache.models import HSTU, HSTUConfig

ROOT = Path(__file__).resolve().parents[1]
P7_MANIFEST = ROOT / "data/manifests/p7_full_v1"
P8_MANIFEST = ROOT / "data/manifests/p8_release_v1"
RAW = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
P7_THETA = ROOT / "results/p7/theta0_training/runs"
OUTPUT = ROOT / "results/p8/release_training"
CONTRACT = ROOT / "configs/contracts/f_release_chain_contract_v1.yaml"
MODELS = {"m0_f": ("F",), "m1": ("N", "R", "F")}
RELEASES = {
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
        return OUTPUT / "r1_edge1" / f"{model_name}_seed{seed}" / "selected.pt"
    p7_name = "m0_f" if model_name == "m0_f" else "m1"
    return P7_THETA / f"{p7_name}_seed{seed}" / "theta0_selected.pt"


def load_split(root: Path, split: str, tasks: tuple[str, ...]) -> dict[str, list[P7Request]]:
    return {task: load_p7_requests(root, RAW, split, task) for task in tasks}


def training_data(model_name: str, release: str) -> tuple[dict[str, list[P7Request]], dict[str, list[P7Request]]]:
    tasks = MODELS[model_name]
    train_split, dev_split = RELEASES[release]
    update = load_split(P8_MANIFEST, train_split, tasks)
    if release == "r2":
        original = load_split(P7_MANIFEST, "residual_train", tasks)
        update = {task: original[task] + update[task] for task in tasks}
    return update, load_split(P8_MANIFEST, dev_split, tasks)


def configure_model(model_name: str, seed: int, release: str, device: torch.device) -> tuple[HSTU, dict, Path]:
    parent = parent_path(model_name, seed, release)
    parent_model, parent_payload = load_checkpoint(parent, device)
    if release == "r2":
        model = p7.make_model(seed, device)
    else:
        model = parent_model
    if release == "r0":
        allowed = ("query_encoder.type_embedding", "query_encoder.action_embedding", "cc_score_head")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(allowed))
    return model, parent_payload, parent


def no_information_loss(task: str, requests: list[P7Request]) -> float:
    weights = np.asarray([row.request_weight for row in requests], dtype=np.float64)
    if task in {"N", "R"}:
        values = np.log(np.asarray([len(row.candidate_ids) for row in requests], dtype=np.float64))
    else:
        labels = np.asarray([int(row.label) for row in requests], dtype=np.float64)
        prevalence = float(np.average(labels, weights=weights))
        prevalence = min(max(prevalence, 1e-8), 1 - 1e-8)
        values = -(labels * math.log(prevalence) + (1 - labels) * math.log(1 - prevalence))
    return float(np.average(values, weights=weights))


def train(model_name: str, seed: int, release: str, device: torch.device, output: Path) -> None:
    tasks = MODELS[model_name]
    model, parent_payload, parent = configure_model(model_name, seed, release, device)
    bases, base_manifest = p7.load_bases(tasks, device)
    train_rows, dev_rows = training_data(model_name, release)
    parent_model, _ = load_checkpoint(parent, device)
    previous_dev = {
        task: p7.evaluate_task(parent_model, bases[task], dev_rows[task], device)
        for task in tasks
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=2e-4,
        weight_decay=1e-4,
    )
    frozen_before = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if release == "r0" and not name.startswith(
            ("query_encoder.type_embedding", "query_encoder.action_embedding", "cc_score_head")
        )
    }
    best_objective = float("inf")
    best_epoch = 0
    best_state = None
    trace = []
    budgets: Counter = Counter()
    for epoch in range(1, 4):
        batches = {
            task: p7.logical_batches(
                p7.deterministic_order(train_rows[task], seed * 10_000 + epoch * 100 + p7.QUERY_TYPES[task])
            )
            for task in tasks
        }
        normalizers = {
            task: sum(row.request_weight for row in train_rows[task]) / len(batches[task])
            for task in tasks
        }
        schedule = (
            [(task, index) for index in range(max(map(len, batches.values()))) for task in tasks if index < len(batches[task])]
            if model_name == "m1"
            else [(tasks[0], index) for index in range(len(batches[tasks[0]]))]
        )
        model.train()
        for task, index in schedule:
            record = p7.train_logical_batch(
                model, bases[task], optimizer, batches[task][index], device,
                loss_normalizer=normalizers[task],
            )
            for key in ("queries", "candidate_rows", "history_tokens", "token_layer_work"):
                budgets[(task, key)] += record[key]
            budgets[(task, "optimizer_steps")] += 1
        dev = {task: p7.evaluate_task(model, bases[task], dev_rows[task], device) for task in tasks}
        objective = float(np.mean([dev[task]["deployment_loss"] for task in tasks]))
        selected = objective < best_objective
        if selected:
            best_objective = objective
            best_epoch = epoch
            best_state = p7.clone_state_dict(model)
        trace.append({"epoch": epoch, "objective": objective, "tasks": dev, "selected_so_far": selected})
    assert best_state is not None
    model.load_state_dict(best_state)
    final_dev = {task: p7.evaluate_task(model, bases[task], dev_rows[task], device) for task in tasks}
    controls = {task: no_information_loss(task, dev_rows[task]) for task in tasks}
    if model_name == "m0_f":
        admitted = final_dev["F"]["deployment_loss"] < final_dev["F"]["base_only_loss"]
    else:
        every_task = all(final_dev[task]["deployment_loss"] < controls[task] for task in tasks)
        r_or_f = any(final_dev[task]["deployment_loss"] < final_dev[task]["base_only_loss"] for task in ("R", "F"))
        admitted = every_task and r_or_f
    frozen_max_delta = 0.0
    if release == "r0":
        frozen_max_delta = max(
            float((model.state_dict()[name].detach().cpu() - before).abs().max())
            for name, before in frozen_before.items()
        )
        if frozen_max_delta != 0.0:
            raise RuntimeError("R0 changed a cache-producing parameter")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "selected.pt"
    torch.save({
        "contract": "f_release_chain_contract_v1", "release": release,
        "model_name": model_name, "seed": seed, "selected_epoch": best_epoch,
        "config": asdict(model.cfg), "model_state_dict": best_state,
        "parent_checkpoint": str(parent.relative_to(ROOT)), "parent_checkpoint_hash": p7.sha256_file(parent),
        "admitted": admitted, "base_bundle_hash": p7.sha256_file(p7.BASE_ROOT / "bundle_manifest.json"),
    }, checkpoint)
    result = {
        "status": "trained_and_frozen_before_staleness_evaluation",
        "contract_hash": p7.sha256_file(CONTRACT), "release": release,
        "model_name": model_name, "seed": seed, "parent": str(parent.relative_to(ROOT)),
        "parent_hash": p7.sha256_file(parent), "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_hash": p7.sha256_file(checkpoint), "selected_epoch": best_epoch,
        "selection_trace": trace, "final_admission_dev": final_dev,
        "previous_full_admission_dev": previous_dev, "no_information_controls": controls,
        "admitted": admitted, "admission_uses_H_or_S": False,
        "r0_frozen_parameter_max_abs_delta": frozen_max_delta if release == "r0" else None,
        "budget": {
            task: {key: budgets[(task, key)] for key in ("queries", "candidate_rows", "history_tokens", "token_layer_work", "optimizer_steps")}
            for task in tasks
        },
        "train_query_counts": {task: len(rows) for task, rows in train_rows.items()},
        "dev_query_counts": {task: len(rows) for task, rows in dev_rows.items()},
        "base_manifest_hash": p7.sha256_file(p7.BASE_ROOT / "bundle_manifest.json"),
        "base_parameter_hashes": base_manifest["files"],
        "staleness_scored": False,
    }
    (output / "train_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "release", "model_name", "seed", "checkpoint_hash", "admitted")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 37, 71), required=True)
    parser.add_argument("--release", choices=sorted(RELEASES), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or OUTPUT / args.release / f"{args.model}_seed{args.seed}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    train(args.model, args.seed, args.release, torch.device(args.device), output)


if __name__ == "__main__":
    main()
