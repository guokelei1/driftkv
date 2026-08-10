from __future__ import annotations

import gc
import json
import os
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .kuairand_next_item_chain import _effective_document, _move_optimizer
from .kuairand_root_cause import (
    PROTOCOL,
    _atomic_json,
    _atomic_torch,
    _save_checkpoint,
    _train_epoch,
    file_sha256,
    load_plan,
    make_model,
)

STRENGTH_PROTOCOL = "evokv_kuairand_next_item_update_strength_v0"
GROUPS = ("item_embedding", "input_encoders", "kv_projections", "other_core")


def load_next_item_strength_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    source = document.get("source", {})
    candidates = document.get("candidates", [])
    primary = [
        {
            "candidate_id": "balanced_2x_e3",
            "epochs": 3,
            "learning_rates": {
                "item_embedding": 0.0002,
                "input_encoders": 0.0002,
                "kv_projections": 0.0002,
                "other_core": 0.0002,
            },
        },
        {
            "candidate_id": "kv_focused_e3",
            "epochs": 3,
            "learning_rates": {
                "item_embedding": 0.0001,
                "input_encoders": 0.0002,
                "kv_projections": 0.0005,
                "other_core": 0.0002,
            },
        },
    ]
    followup = [
        {
            "candidate_id": "kv_focused_high_e3",
            "epochs": 3,
            "learning_rates": {
                "item_embedding": 0.0001,
                "input_encoders": 0.0002,
                "kv_projections": 0.001,
                "other_core": 0.0002,
            },
        }
    ]
    bindings = [source.get("config", {}), source.get("resume", {}), source.get("theta2", {})]
    if (
        document.get("protocol") != STRENGTH_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("source_version") != 2
        or document.get("target_version") != 7
        or document.get("total_num_days") != 22
        or document.get("update_date_indices") != [16, 17, 18, 19, 20]
        or candidates not in (primary, followup)
        or any(file_sha256(value.get("path", "")) != value.get("sha256") for value in bindings)
    ):
        raise ValueError("KuaiRand next-item strength config differs")
    return document


def _group_name(name: str) -> str:
    if name == "item_emb.weight":
        return "item_embedding"
    if name.startswith(("behavior_emb.", "temporal_enc.", "in_proj.")):
        return "input_encoders"
    if ".attn.k_proj." in name or ".attn.v_proj." in name:
        return "kv_projections"
    return "other_core"


def _grouped_optimizer(
    model,
    source_optimizer: torch.optim.Optimizer,
    learning_rates: dict[str, float],
    weight_decay: float,
) -> torch.optim.Optimizer:
    grouped: dict[str, list[torch.nn.Parameter]] = {name: [] for name in GROUPS}
    for name, parameter in model.named_parameters():
        grouped[_group_name(name)].append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": grouped[name],
                "lr": float(learning_rates[name]),
                "group_name": name,
            }
            for name in GROUPS
        ],
        weight_decay=weight_decay,
    )
    for parameter, state in source_optimizer.state.items():
        optimizer.state[parameter] = deepcopy(state)
    return optimizer


def _candidate_training(
    config: dict[str, Any],
    config_path: Path,
    candidate: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    root = Path(config["checkpoint_parent"]) / candidate["candidate_id"]
    output = Path(config["result_parent"]) / candidate["candidate_id"] / "training.json"
    if output.is_file():
        return json.loads(output.read_text())
    document = _effective_document(
        {
            **config,
            "source": config["source"],
            "evaluation_methods": ["fresh_full_a", "fresh_full_b", "stale_previous", "no_prefix"],
            "record_limit_per_rank": 1,
        }
    )
    plan, metadata = load_plan(document)
    plan.init_base()
    dates = plan.base_dates + plan.stream_dates
    for date_index in (14, 15):
        plan.ingest_day(dates[date_index])
    model = make_model(document, plan, device)
    source_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(document["training"]["base_lr"]),
        weight_decay=float(document["training"]["weight_decay"]),
    )
    source_resume = torch.load(
        config["source"]["resume"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    if (
        source_resume.get("protocol") != PROTOCOL
        or source_resume.get("version") != config["source_version"]
        or source_resume.get("config_sha256") != config["source"]["config"]["sha256"]
    ):
        raise ValueError("KuaiRand strength source optimizer binding differs")
    model.load_state_dict(source_resume["model_state"])
    source_optimizer.load_state_dict(source_resume["optimizer_state"])
    optimizer = _grouped_optimizer(
        model,
        source_optimizer,
        candidate["learning_rates"],
        float(document["training"]["weight_decay"]),
    )
    del source_optimizer, source_resume
    _move_optimizer(optimizer, device)
    resume_path = root / "resume.pt"
    completed_version = int(config["source_version"])
    records = []
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (
            resume.get("protocol") != STRENGTH_PROTOCOL
            or resume.get("candidate_id") != candidate["candidate_id"]
            or resume.get("config_sha256") != file_sha256(config_path)
        ):
            raise ValueError("KuaiRand strength resume differs")
        completed_version = int(resume["version"])
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        _move_optimizer(optimizer, device)
        for version in range(3, completed_version + 1):
            manifest = json.loads((root / f"theta_{version}" / "manifest.json").read_text())
            records.append(manifest["training"])
    for date_index in config["update_date_indices"][: max(0, completed_version - 2)]:
        plan.ingest_day(dates[int(date_index)])
    started = time.perf_counter()
    for version, date_index in enumerate(config["update_date_indices"], start=3):
        if version <= completed_version:
            continue
        date = dates[int(date_index)]
        plan.ingest_day(date)
        epochs = []
        for epoch in range(int(candidate["epochs"])):
            np.random.seed(int(document["training"]["seed"]) + version * 1009 + epoch)
            batches = plan.iter_train_batches(
                date,
                int(document["training"]["batch_size"]),
                all_chunks=True,
                bucket_by_length=True,
                pad_to_max_seq_len=False,
            )
            epochs.append(
                _train_epoch(
                    model,
                    optimizer,
                    batches,
                    device,
                    int(document["training"]["negative_count"]),
                    document["training"]["maximum_update_batches_per_epoch"],
                    f"kuairand_{candidate['candidate_id']}_theta{version}_epoch{epoch + 1}",
                )
            )
        record = {
            "version": version,
            "role": "stream_update_strength",
            "date": date,
            "candidate_id": candidate["candidate_id"],
            "learning_rates": candidate["learning_rates"],
            "epochs": epochs,
        }
        _save_checkpoint(model, root, version, config_path, metadata, record)
        _atomic_torch(
            resume_path,
            {
                "protocol": STRENGTH_PROTOCOL,
                "candidate_id": candidate["candidate_id"],
                "config_sha256": file_sha256(config_path),
                "version": version,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
        )
        records.append(record)
        completed_version = version
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "protocol": STRENGTH_PROTOCOL,
        "source_protocol": PROTOCOL,
        "status": "complete_development_training",
        "scientific_result": False,
        "formal_result": False,
        "candidate": candidate,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "data": metadata,
        "model": asdict(model.cfg),
        "versions": records,
        "completed_version": completed_version,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    _atomic_json(output, result)
    del optimizer, model, plan
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_next_item_strength_training(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_next_item_strength_config(config_path)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("KuaiRand next-item strength training is single-rank")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    results = [
        _candidate_training(config, config_path, candidate, device)
        for candidate in config["candidates"]
    ]
    result = {
        "protocol": STRENGTH_PROTOCOL,
        "status": "complete_development_strength_sweep",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "candidates": [
            {
                "candidate_id": value["candidate"]["candidate_id"],
                "completed_version": value["completed_version"],
                "result": str(
                    Path(config["result_parent"])
                    / value["candidate"]["candidate_id"]
                    / "training.json"
                ),
            }
            for value in results
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(Path(config["sweep_result"]), result)
    return result


def validate_next_item_strength_result(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != STRENGTH_PROTOCOL
        or result.get("status") != "complete_development_strength_sweep"
        or result.get("scientific_result") is not False
        or [value.get("candidate_id") for value in result.get("candidates", [])]
        not in (["balanced_2x_e3", "kv_focused_e3"], ["kv_focused_high_e3"])
        or not all(value.get("completed_version") == 7 for value in result["candidates"])
    ):
        raise ValueError("KuaiRand next-item strength result differs")
