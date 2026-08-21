#!/usr/bin/env python3
"""Train one frozen-contract P7.7 M0/M1 theta0 run without qualification access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hstu_kvcache.data import load_compact_index
from hstu_kvcache.data.p7_training import QUERY_TYPES, P7Request, load_p7_requests
from hstu_kvcache.models import HSTU, FrozenLinearBaseRanker, HSTUConfig

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "data/manifests/p7_full_v1"
RAW = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
BASE_ROOT = ROOT / "results/p7/base_fit/frozen_base_bundle_v1"
OUTPUT = ROOT / "results/p7/theta0_training/runs"
CONTRACT = ROOT / "configs/contracts/p7_7_theta0_training_contract_v1.yaml"
MODELS = {
    "m0_n": ("N",),
    "m0_r": ("R",),
    "m0_f": ("F",),
    "m1": ("N", "R", "F"),
}
MICROBATCH = {"N": 4, "R": 1, "F": 8}
CHUNK_SIZE = {"N": 25, "R": 16, "F": 1}
LOGICAL_BATCH = 8
EPOCHS = 3


def sha256_file(path: Path) -> str:
    output = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            output.update(block)
    return output.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def max_item_id() -> int:
    import pyarrow.parquet as pq

    table = pq.read_table(
        ROOT / "data/raw/yambda/artist_item_mapping.parquet", columns=["item_id"]
    )
    return int(table["item_id"].to_numpy(zero_copy_only=False).max())


def make_model(seed: int, device: torch.device) -> HSTU:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = HSTU(
        HSTUConfig(
            num_items=max_item_id(),
            num_behaviors=2,
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            max_seq_len=512,
            temporal_num_freqs=16,
            temporal_max_period=86_400.0,
            gating="silu_gate",
            activation="elu_plus1",
            input_dropout=0.1,
            attn_dropout=0.0,
            block_variant="legacy",
            relative_position_bias=False,
            causal_diagonal="inclusive",
            num_query_types=3,
            num_query_actions=1,
            query_type_id=0,
            query_action_id=0,
        )
    )
    return model.to(device)


def load_bases(tasks: tuple[str, ...], device: torch.device) -> tuple[dict[str, FrozenLinearBaseRanker], dict]:
    manifest = json.loads((BASE_ROOT / "bundle_manifest.json").read_text())
    output = {}
    for task in tasks:
        filename = f"base_{task.lower()}_v1.json"
        path = BASE_ROOT / filename
        if sha256_file(path) != manifest["files"][filename]:
            raise RuntimeError(f"frozen Base artifact hash changed: {task}")
        artifact = json.loads(path.read_text())
        scorer = FrozenLinearBaseRanker.from_frozen_artifact(artifact).to(device).eval()
        if list(scorer.parameters()):
            raise RuntimeError("Frozen Base unexpectedly exposes trainable parameters")
        output[task] = scorer
    return output, manifest


def collate(requests: list[P7Request], device: torch.device) -> dict[str, torch.Tensor]:
    batch = len(requests)
    history_width = max(len(row.history_items) for row in requests)
    candidate_width = max(len(row.candidate_ids) for row in requests)
    items = np.zeros((batch, history_width), dtype=np.int64)
    behaviors = np.zeros_like(items)
    deltas = np.zeros((batch, history_width), dtype=np.float32)
    lengths = np.zeros(batch, dtype=np.int64)
    candidates = np.zeros((batch, candidate_width), dtype=np.int64)
    candidate_mask = np.zeros((batch, candidate_width), dtype=bool)
    features = np.zeros((batch, candidate_width, 7), dtype=np.float32)
    targets = np.zeros(batch, dtype=np.int64)
    labels = np.zeros(batch, dtype=np.float32)
    weights = np.zeros(batch, dtype=np.float32)
    query_deltas = np.zeros(batch, dtype=np.float32)
    for index, row in enumerate(requests):
        history_length = len(row.history_items)
        candidate_count = len(row.candidate_ids)
        items[index, :history_length] = row.history_items
        behaviors[index, :history_length] = row.history_behaviors
        deltas[index, :history_length] = row.history_time_deltas
        lengths[index] = history_length
        candidates[index, :candidate_count] = row.candidate_ids
        candidate_mask[index, :candidate_count] = True
        features[index, :candidate_count] = row.base_features
        targets[index] = 0 if row.target_index is None else row.target_index
        labels[index] = 0.0 if row.label is None else row.label
        weights[index] = row.request_weight
        query_deltas[index] = row.query_time_delta
    return {
        "items": torch.from_numpy(items).to(device),
        "behaviors": torch.from_numpy(behaviors).to(device),
        "deltas": torch.from_numpy(deltas).to(device),
        "lengths": torch.from_numpy(lengths).to(device),
        "candidates": torch.from_numpy(candidates).to(device),
        "candidate_mask": torch.from_numpy(candidate_mask).to(device),
        "features": torch.from_numpy(features).to(device),
        "targets": torch.from_numpy(targets).to(device),
        "labels": torch.from_numpy(labels).to(device),
        "weights": torch.from_numpy(weights).to(device),
        "query_deltas": torch.from_numpy(query_deltas).to(device),
        "query_types": torch.full(
            (batch,), QUERY_TYPES[requests[0].workload], dtype=torch.long, device=device
        ),
    }


def score_microbatch(
    model: HSTU,
    base: FrozenLinearBaseRanker,
    requests: list[P7Request],
    device: torch.device,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    tensors = collate(requests, device)
    with autocast_context(device):
        chunks = model.score_cc_full_chunked(
            tensors["items"],
            tensors["behaviors"],
            tensors["deltas"],
            tensors["candidates"],
            tensors["query_deltas"],
            chunk_size=chunk_size,
            lengths=tensors["lengths"],
            query_type_ids=tensors["query_types"],
        )
        residual = torch.cat(chunks, dim=1).float()
    with torch.no_grad():
        base_scores = base(tensors["features"].float()).float()
    deployment = base_scores + residual
    if not torch.isfinite(deployment).all():
        raise FloatingPointError("deployment score is non-finite")
    return deployment, base_scores, residual, tensors


def per_request_loss(
    workload: str,
    deployment: torch.Tensor,
    tensors: dict[str, torch.Tensor],
) -> torch.Tensor:
    if workload in {"N", "R"}:
        masked = deployment.masked_fill(~tensors["candidate_mask"], -torch.inf)
        return -masked.gather(1, tensors["targets"][:, None]).squeeze(1) + torch.logsumexp(
            masked, dim=1
        )
    return F.binary_cross_entropy_with_logits(
        deployment[:, 0], tensors["labels"], reduction="none"
    )


def gradient_norm(model: HSTU) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(total)


def named_gradient_audit(model: HSTU) -> dict:
    groups = {
        "encoder_blocks": "blocks.",
        "query_embedding": "query_encoder.",
        "candidate_and_history_item_embedding": "item_emb.",
        "cc_score_head": "cc_score_head.",
    }
    output = {}
    for group, prefix in groups.items():
        values = [
            parameter.grad.detach().float().norm().item()
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        ]
        output[group] = float(sum(values))
    return output


def train_logical_batch(
    model: HSTU,
    base: FrozenLinearBaseRanker,
    optimizer: torch.optim.Optimizer,
    requests: list[P7Request],
    device: torch.device,
    *,
    loss_normalizer: float | None = None,
) -> dict:
    workload = requests[0].workload
    if any(row.workload != workload for row in requests):
        raise ValueError("logical batches must contain one task")
    optimizer.zero_grad(set_to_none=True)
    total_weight = sum(row.request_weight for row in requests)
    normalizer = total_weight if loss_normalizer is None else float(loss_normalizer)
    if normalizer <= 0:
        raise ValueError("logical loss normalizer must be positive")
    losses, base_values, residual_values = [], [], []
    candidate_rows = 0
    history_tokens = 0
    for start in range(0, len(requests), MICROBATCH[workload]):
        micro = requests[start : start + MICROBATCH[workload]]
        deployment, base_scores, residual, tensors = score_microbatch(
            model, base, micro, device, chunk_size=CHUNK_SIZE[workload]
        )
        request_losses = per_request_loss(workload, deployment, tensors)
        numerator = (request_losses * tensors["weights"]).sum()
        (numerator / normalizer).backward()
        losses.extend(request_losses.detach().float().cpu().tolist())
        base_values.append(base_scores.detach().float()[tensors["candidate_mask"]].cpu())
        residual_values.append(residual.detach().float()[tensors["candidate_mask"]].cpu())
        candidate_rows += sum(len(row.candidate_ids) for row in micro)
        history_tokens += sum(len(row.history_items) for row in micro)
    gradients = named_gradient_audit(model)
    norm = gradient_norm(model)
    optimizer.step()
    base_flat = torch.cat(base_values)
    residual_flat = torch.cat(residual_values)
    weights = np.asarray([row.request_weight for row in requests], dtype=np.float64)
    weighted_loss = float(np.dot(weights, np.asarray(losses)) / weights.sum())
    return {
        "task": workload,
        "queries": len(requests),
        "candidate_rows": candidate_rows,
        "history_tokens": history_tokens,
        "token_layer_work": 4 * (candidate_rows + history_tokens),
        "loss": weighted_loss,
        "gradient_loss_normalizer": normalizer,
        "gradient_norm": norm,
        "named_gradient_norms": gradients,
        "base_score_mean": float(base_flat.mean()),
        "base_score_std": float(base_flat.std()),
        "residual_score_mean": float(residual_flat.mean()),
        "residual_score_std": float(residual_flat.std()),
        "base_residual_std_ratio": float(base_flat.std() / max(residual_flat.std(), 1e-12)),
    }


@torch.no_grad()
def evaluate_task(
    model: HSTU,
    base: FrozenLinearBaseRanker,
    requests: list[P7Request],
    device: torch.device,
) -> dict:
    model.eval()
    workload = requests[0].workload
    numerator = 0.0
    denominator = 0.0
    base_numerator = 0.0
    residual_values = []
    base_values = []
    candidate_rows = 0
    history_tokens = 0
    microbatch = MICROBATCH[workload]
    for start in range(0, len(requests), microbatch):
        micro = requests[start : start + microbatch]
        deployment, base_scores, residual, tensors = score_microbatch(
            model, base, micro, device, chunk_size=CHUNK_SIZE[workload]
        )
        losses = per_request_loss(workload, deployment, tensors)
        base_losses = per_request_loss(workload, base_scores, tensors)
        numerator += float((losses * tensors["weights"]).sum())
        base_numerator += float((base_losses * tensors["weights"]).sum())
        denominator += float(tensors["weights"].sum())
        residual_values.append(residual[tensors["candidate_mask"]].float().cpu())
        base_values.append(base_scores[tensors["candidate_mask"]].float().cpu())
        candidate_rows += sum(len(row.candidate_ids) for row in micro)
        history_tokens += sum(len(row.history_items) for row in micro)
    residual_flat = torch.cat(residual_values)
    base_flat = torch.cat(base_values)
    model.train()
    return {
        "task": workload,
        "queries": len(requests),
        "users": len({row.uid for row in requests}),
        "candidate_rows": candidate_rows,
        "history_tokens": history_tokens,
        "deployment_loss": numerator / denominator,
        "base_only_loss": base_numerator / denominator,
        "residual_score_mean": float(residual_flat.mean()),
        "residual_score_std": float(residual_flat.std()),
        "base_score_mean": float(base_flat.mean()),
        "base_score_std": float(base_flat.std()),
    }


def deterministic_order(requests: list[P7Request], seed: int) -> list[P7Request]:
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(requests))
    return [requests[int(index)] for index in order]


def logical_batches(requests: list[P7Request]) -> list[list[P7Request]]:
    return [requests[start : start + LOGICAL_BATCH] for start in range(0, len(requests), LOGICAL_BATCH)]


def clone_state_dict(model: HSTU) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def permutation_check(
    model: HSTU,
    base: FrozenLinearBaseRanker,
    request: P7Request,
    device: torch.device,
) -> float:
    model.eval()
    _, _, original, _ = score_microbatch(
        model, base, [request], device, chunk_size=CHUNK_SIZE[request.workload]
    )
    permutation = np.arange(len(request.candidate_ids))[::-1].copy()
    permuted = P7Request(
        **{
            **request.__dict__,
            "candidate_ids": request.candidate_ids[permutation],
            "base_features": request.base_features[permutation],
            "target_index": None
            if request.target_index is None
            else int(np.flatnonzero(permutation == request.target_index)[0]),
        }
    )
    _, _, changed, _ = score_microbatch(
        model, base, [permuted], device, chunk_size=CHUNK_SIZE[request.workload]
    )
    aligned = changed[:, torch.from_numpy(np.argsort(permutation)).to(device)]
    delta = float((original - aligned).abs().max())
    model.train()
    return delta


@torch.no_grad()
def r_chunk_check(model: HSTU, request: P7Request, device: torch.device) -> float:
    model.eval()
    tensors = collate([request], device)
    with autocast_context(device):
        full = model.score_cc_full(
            tensors["items"],
            tensors["behaviors"],
            tensors["deltas"],
            tensors["candidates"],
            tensors["query_deltas"],
            lengths=tensors["lengths"],
            query_type_ids=tensors["query_types"],
        ).float()
        chunked = torch.cat(
            model.score_cc_full_chunked(
                tensors["items"],
                tensors["behaviors"],
                tensors["deltas"],
                tensors["candidates"],
                tensors["query_deltas"],
                chunk_size=min(3, len(request.candidate_ids)),
                lengths=tensors["lengths"],
                query_type_ids=tensors["query_types"],
            ),
            dim=1,
        ).float()
    model.train()
    return float((full - chunked).abs().max())


def load_data(tasks: tuple[str, ...]) -> tuple[dict[str, list[P7Request]], dict[str, list[P7Request]]]:
    train, development = {}, {}
    for task in tasks:
        train[task] = load_p7_requests(MANIFEST_ROOT, RAW, "residual_train", task)
        development[task] = load_p7_requests(MANIFEST_ROOT, RAW, "development", task)
        if len(train[task]) != 3939:
            raise RuntimeError(f"{task} training query budget differs")
    return train, development


def common_metadata(model_name: str, seed: int, base_manifest: dict) -> dict:
    return {
        "contract": "p7_7_theta0_training_contract_v1",
        "contract_hash": sha256_file(CONTRACT),
        "model_name": model_name,
        "seed": seed,
        "base_bundle_hash": sha256_file(BASE_ROOT / "bundle_manifest.json"),
        "base_parameter_hashes": base_manifest["files"],
        "train_manifest_hash": sha256_file(MANIFEST_ROOT / "residual_train/manifest.index.json"),
        "development_manifest_hash": sha256_file(
            MANIFEST_ROOT / "development/manifest.index.json"
        ),
        "qualification_manifest_hash": sha256_file(
            MANIFEST_ROOT / "qualification/manifest.index.json"
        ),
        "qualification_scored": False,
        "code_commit": git_commit(),
        "training_code_hash": sha256_file(Path(__file__)),
    }


def qualification_is_locked() -> bool:
    try:
        load_compact_index(MANIFEST_ROOT / "qualification/manifest.index.json")
    except PermissionError:
        return True
    return False


def run_preflight(model_name: str, seed: int, device: torch.device, output: Path) -> None:
    tasks = MODELS[model_name]
    base_hash_before = sha256_file(BASE_ROOT / "bundle_manifest.json")
    bases, base_manifest = load_bases(tasks, device)
    train, _ = load_data(tasks)
    model = make_model(seed, device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    subsets = {
        task: sorted(train[task], key=lambda row: hashlib.sha256(row.request_id.encode()).digest())[:32]
        for task in tasks
    }
    fixed_objective_before = float(
        np.mean(
            [evaluate_task(model, bases[task], subsets[task], device)["deployment_loss"] for task in tasks]
        )
    )
    losses = []
    gradient_audits = {}
    steps = 20 if model_name != "m1" else 30
    task_order = [tasks[index % len(tasks)] for index in range(steps)]
    task_positions = {task: 0 for task in tasks}
    for step, task in enumerate(task_order):
        values = subsets[task]
        position = task_positions[task]
        indices = [(position + offset) % len(values) for offset in range(LOGICAL_BATCH)]
        batch = [values[index] for index in indices]
        task_positions[task] = (position + LOGICAL_BATCH) % len(values)
        record = train_logical_batch(model, bases[task], optimizer, batch, device)
        record["optimizer_step"] = step + 1
        losses.append(record)
        gradient_audits.setdefault(task, record["named_gradient_norms"])
    first = float(np.mean([row["loss"] for row in losses[:5]]))
    last = float(np.mean([row["loss"] for row in losses[-5:]]))
    fixed_objective_after = float(
        np.mean(
            [evaluate_task(model, bases[task], subsets[task], device)["deployment_loss"] for task in tasks]
        )
    )
    permutation = {
        task: permutation_check(model, bases[task], subsets[task][0], device) for task in tasks
    }
    residual_std = {}
    model.eval()
    for task in tasks:
        _, _, residual, tensors = score_microbatch(
            model, bases[task], subsets[task][:4], device, chunk_size=CHUNK_SIZE[task]
        )
        residual_std[task] = float(
            residual[tensors["candidate_mask"]].detach().float().std()
        )
    model.train()
    r_delta = None
    if "R" in tasks:
        smallest = min(subsets["R"], key=lambda row: len(row.candidate_ids))
        r_delta = r_chunk_check(model, smallest, device)
    feedback_labels = None
    if "F" in tasks:
        feedback_labels = dict(Counter(row.label for row in train["F"]))
    base_hash_after = sha256_file(BASE_ROOT / "bundle_manifest.json")
    checks = {
        "finite_loss": all(math.isfinite(row["loss"]) for row in losses),
        "fixed_subset_loss_decreased": fixed_objective_after < fixed_objective_before,
        "base_hash_unchanged": base_hash_before == base_hash_after,
        "base_has_no_parameters_or_grad": all(not list(base.parameters()) for base in bases.values()),
        "required_gradient_groups_nonzero": all(
            all(value > 0 for value in audit.values()) for audit in gradient_audits.values()
        ),
        "residual_variance_nonzero": all(value > 0 for value in residual_std.values()),
        "candidate_permutation_equivalent": all(value <= 2e-5 for value in permutation.values()),
        "R_chunked_equals_full": r_delta is None or r_delta <= 2e-5,
        "F_natural_labels_include_both_classes": feedback_labels is None
        or set(feedback_labels) == {0, 1},
        "qualification_locked": qualification_is_locked(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"preflight failed: {checks}")
    payload = {
        **common_metadata(model_name, seed, base_manifest),
        "status": "passed_real_data_preflight",
        "checks": checks,
        "first_five_loss_mean": first,
        "last_five_loss_mean": last,
        "fixed_subset_objective_before": fixed_objective_before,
        "fixed_subset_objective_after": fixed_objective_after,
        "gradient_audits": gradient_audits,
        "residual_score_std": residual_std,
        "candidate_permutation_max_abs_delta": permutation,
        "R_chunked_full_max_abs_delta": r_delta,
        "F_full_training_natural_label_counts": feedback_labels,
        "optimizer_steps": steps,
        "retained_checkpoint": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "model": model_name, "checks": checks}, indent=2))


def run_training(model_name: str, seed: int, device: torch.device, output: Path) -> None:
    tasks = MODELS[model_name]
    bases, base_manifest = load_bases(tasks, device)
    train, development = load_data(tasks)
    model = make_model(seed, device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    step_logs = []
    selection_trace = []
    presentations = {task: 0 for task in tasks}
    best_objective = float("inf")
    best_epoch = None
    best_state = None
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        task_batches = {
            task: logical_batches(
                deterministic_order(train[task], seed * 10_000 + epoch * 100 + QUERY_TYPES[task])
            )
            for task in tasks
        }
        task_loss_normalizers = {
            task: sum(row.request_weight for row in train[task]) / len(task_batches[task])
            for task in tasks
        }
        if model_name == "m1":
            schedule = [
                (task, index)
                for index in range(max(len(values) for values in task_batches.values()))
                for task, values in task_batches.items()
                if index < len(values)
            ]
        else:
            task = tasks[0]
            schedule = [(task, index) for index in range(len(task_batches[task]))]
        for task, batch_index in schedule:
            global_step += 1
            record = train_logical_batch(
                model,
                bases[task],
                optimizer,
                task_batches[task][batch_index],
                device,
                loss_normalizer=task_loss_normalizers[task],
            )
            presentations[task] += record["queries"]
            record.update(
                {
                    "optimizer_step": global_step,
                    "epoch": epoch,
                    "cumulative_query_presentations": dict(presentations),
                }
            )
            step_logs.append(record)
        development_metrics = {
            task: evaluate_task(model, bases[task], development[task], device) for task in tasks
        }
        objective = float(
            np.mean([development_metrics[task]["deployment_loss"] for task in tasks])
        )
        selected_so_far = objective < best_objective
        if selected_so_far:
            best_objective = objective
            best_epoch = epoch
            best_state = clone_state_dict(model)
        selection_trace.append(
            {
                "epoch": epoch,
                "development": development_metrics,
                "checkpoint_objective": objective,
                "selected_so_far": selected_so_far,
                "selection_uses_H_or_qualification": False,
            }
        )
    assert best_state is not None and best_epoch is not None
    model.load_state_dict(best_state)
    final_development = {
        task: evaluate_task(model, bases[task], development[task], device) for task in tasks
    }
    checkpoint_path = output / "theta0_selected.pt"
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_payload = {
        "contract": "p7_7_theta0_training_contract_v1",
        "model_name": model_name,
        "seed": seed,
        "selected_epoch": best_epoch,
        "config": asdict(model.cfg),
        "model_state_dict": best_state,
        "base_bundle_hash": sha256_file(BASE_ROOT / "bundle_manifest.json"),
        "train_manifest_hash": sha256_file(MANIFEST_ROOT / "residual_train/manifest.index.json"),
        "development_manifest_hash": sha256_file(
            MANIFEST_ROOT / "development/manifest.index.json"
        ),
        "qualification_scored": False,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    del checkpoint_payload, best_state
    checkpoint_hash = sha256_file(checkpoint_path)
    total_budget = {
        task: {
            "unique_queries": len(train[task]),
            "query_presentations": sum(
                row["queries"] for row in step_logs if row["task"] == task
            ),
            "candidate_rows": sum(
                row["candidate_rows"] for row in step_logs if row["task"] == task
            ),
            "history_tokens": sum(
                row["history_tokens"] for row in step_logs if row["task"] == task
            ),
            "token_layer_work": sum(
                row["token_layer_work"] for row in step_logs if row["task"] == task
            ),
            "optimizer_steps": sum(row["task"] == task for row in step_logs),
        }
        for task in tasks
    }
    common = common_metadata(model_name, seed, base_manifest)
    train_result = {
        **common,
        "status": "trained_selected_checkpoint_frozen_dev_only",
        "epochs": EPOCHS,
        "optimizer": {"type": "AdamW", "learning_rate": 2e-4, "weight_decay": 1e-4},
        "selected_epoch": best_epoch,
        "selected_development_objective": best_objective,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_hash": checkpoint_hash,
        "budget": total_budget,
        "step_logs": step_logs,
        "base_bundle_hash_after": sha256_file(BASE_ROOT / "bundle_manifest.json"),
        "qualification_locked_after": qualification_is_locked(),
    }
    selection = {
        **common,
        "status": "selected_on_development_only",
        "rule": "minimum task loss" if model_name != "m1" else "minimum equal one-third N/R/F loss",
        "tie_break": "earliest_epoch",
        "selected_epoch": best_epoch,
        "trace": selection_trace,
        "checkpoint_hash": checkpoint_hash,
    }
    sanity = {
        **common,
        "status": "evaluated_selected_checkpoint_development_only",
        "tasks": final_development,
        "checkpoint_hash": checkpoint_hash,
        "qualification_locked": qualification_is_locked(),
        "recent_or_H_evaluated": False,
    }
    (output / "train.json").write_text(json.dumps(train_result, indent=2) + "\n")
    (output / "checkpoint_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (output / "dev_sanity.json").write_text(json.dumps(sanity, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": train_result["status"],
                "model": model_name,
                "seed": seed,
                "selected_epoch": best_epoch,
                "checkpoint_hash": checkpoint_hash,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 37, 71), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not qualification_is_locked():
        raise RuntimeError("qualification loader is not locked")
    device = torch.device(args.device)
    default = (
        OUTPUT / "preflight" / f"{args.model}_seed{args.seed}.json"
        if args.mode == "preflight"
        else OUTPUT / f"{args.model}_seed{args.seed}"
    )
    output = args.output or default
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if args.mode == "preflight":
        run_preflight(args.model, args.seed, device, output)
    else:
        run_training(args.model, args.seed, device, output)


if __name__ == "__main__":
    main()
